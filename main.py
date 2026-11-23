from collections import defaultdict
import json
import os
import time

import torch

import wandb
import yaml
from dataset import get_dataset
from easydict import EasyDict as edict
from trainer import Trainer
import numpy as np
from torch_geometric.loader import DataLoader

from eval.metrics import SamplingMetrics
from utils.func import batch_to_dense, get_edge_mask

from utils.parser import parse_args

def main() -> None:
    # Parse command line arguments
    args = parse_args()
    work_type = args.work_type
    dataset = args.dataset

    if args.wandb == 'no':
        args.wandb = 'disabled'
    elif args.wandb == 'on':
        args.wandb = 'online'
    elif args.wandb == 'off':
        args.wandb = 'offline'

    config_path = f'./config/{dataset}.yaml'
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    config = edict(config)
    config = update_config(config, dataset, work_type, args)

    if work_type == 'train':
        wandb.init(project=f'ModeFlowGraph_{dataset}', config=config, mode=args.wandb,
                   name=run_name(config))
        dataloader = get_dataset(config)
        trainer = Trainer(dataloader, config)
        trainer.train()

    elif work_type == 'sample':
        torch.manual_seed(42)
        runs = []
        wandb.init(project=f'ModeFlowGraph_{dataset}_sample', config=config, mode=args.wandb,
                   name=run_name(config))
        times = []

        # --fast_eval: as during training, n_val_samples graphs (see Trainer.eval_samples)
        n_samples = config.log.n_val_samples if args.fast_eval else config.log.n_samples_generation
        # sweep over the numbers of sampling steps T (--nfe, default: model.T of the config)
        for r in args.nfe or [config.model.T]:
            config.model.T = r
            dataloader = get_dataset(config)
            trainer = Trainer(dataloader, config)
            with torch.no_grad():
                start_time = time.time()
                X, A, mask, mask_adj = trainer.sampler(n_samples, trainer.denoiser,
                                                       method=config.sampling.id)
                sampling_time = time.time() - start_time
                wandb.log({'sampling_time': sampling_time, 'nfe': r})
                times.append(sampling_time)
                run = trainer.eval_samples(X, A, mask, mask_adj, ema=True)
                runs.append(run)
            save_result(results_path(config), result_key(config, r),
                        {'nfe': r, 'eta': config.sampling.eta, 'lambd': config.sampling.lambd, 'n_samples': n_samples,
                         'sampling_time': sampling_time, **run})

        keys = runs[0].keys()
        latex_format = {}
        for key in keys:
            if isinstance(runs[0][key], str):  # e.g. pgd_descriptor
                print(f'{key}: {[run[key] for run in runs]}')
                continue
            val = []
            for run in runs:
                val.append(run[key])
            mean = np.asarray(val).mean()
            std = np.asarray(val).std()
            print(f'mean {key}: {mean}')
            print(f'std {key}: {std}')
            if key in ['valid', 'unique', 'novel']:
                mean *= 100
                std *= 100
                latex_format[key] = f'${mean:.2f} \\pm {std:.2f}$'
            elif key in ['nspdk', 'degree', 'cluster', 'orbit', 'spectral']:
                mean *= 1000
                std *= 1000
                latex_format[key] = f'${mean:.3f} \\pm {std:.3f}$'
            else:
                latex_format[key] = f'${mean:.3f} \\pm {std:.3f}$'

        latex_format = defaultdict(lambda: '-', latex_format)   # e.g. no NSPDK with --fast_eval
        if dataset in ['qm9', 'qm9H', 'zinc']:
            print(
                f'{latex_format["valid"]} &  {latex_format["fcd"]} & {latex_format["nspdk"]} & {latex_format["unique"]} & {latex_format["novel"]} \\')
        else:
            latex_format["valid"], latex_format['novel'] = 0, 0
            print(
                f'{latex_format["valid"]} &  {latex_format["degree"]} & {latex_format["cluster"]} & {latex_format["orbit"]}& {latex_format["spectral"]} & {latex_format["novel"]} \\')
        times = np.asarray(times)
        print(f'time: {times.mean()} \\pm {times.std()}')
        wandb.finish()

    elif work_type == 'train_vs_test':
        # reference values of the metrics: training graphs evaluated against the test set, both of size
        # n_samples_generation
        torch.manual_seed(42)
        wandb.init(project=f'sid_{dataset}_train_vs_test', config=config, mode=args.wandb,
                   name=run_name(config))
        dataloader = get_dataset(config)
        n = config.log.n_samples_generation
        X, A, mask, mask_adj = sample_training_graphs(dataloader['train'].dataset, n, config.data.max_num_nodes)
        print(f'Evaluating {X.size(0)} training graphs against the test set')
        eval_samples = SamplingMetrics(dataset, config.data.max_num_nodes, sampling=True,
                                       ref_loader=dataloader['test'], ref_size=n, pgd=config.log.pgd)
        eval_samples(X, A, mask, mask_adj)
        wandb.finish()

    else:
        raise NotImplementedError(f'Unknown work_type {work_type}. Available: train, sample, train_vs_test')


def sample_training_graphs(train_set, n, max_num_nodes):
    """n random training graphs, in the format of the sampler: node / edge types, node mask and edge mask."""
    batch = next(iter(DataLoader(train_set, batch_size=n, shuffle=True)))
    X, A, mask = batch_to_dense(batch, max_num_nodes=max_num_nodes)
    adj_mask = get_edge_mask(mask)
    return X.argmax(-1) * mask, A.argmax(-1) * adj_mask, mask, adj_mask


def results_path(config, root='sweeps'):
    """JSON file of the sampling results (read by misc/plots.py), same format as sample_tuned_sweep.py:
    <root>/<sampler>_<dataset>_<schedule>_scale<dirichlet_scale>.json (prior 'dirichlet'), ..._c<schedule_c>.json
    otherwise, e.g. sweeps/quantile_map_qm9_cosine_scale3.json."""
    schedule = config.model.get('schedule', 'cosine')
    if config.model.prior == 'dirichlet':
        param = f'scale{config.model.dirichlet_scale:g}'
    else:
        param = f'c{config.model.schedule_c:g}'
    return os.path.join(root, f'{config.sampling.id}_{config.dataset}_{schedule}_{param}.json')


def result_key(config, nfe):
    """Key of a configuration in the results file: T<nfe>, with _eta<eta>_lambda<lambd> for 'tuned' (as
    sample_tuned_sweep.py)."""
    if config.sampling.id == 'tuned':
        return f'T{nfe}_eta{config.sampling.eta:g}_lambda{config.sampling.lambd:g}'
    return f'T{nfe}'


def save_result(path, key, result):
    """Adds (or replaces) results[key] in the JSON file path."""
    def to_json(v):
        if isinstance(v, str):
            return v
        try:
            return float(v)
        except (TypeError, ValueError):
            return str(v)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    results = {}
    if os.path.isfile(path):
        with open(path) as f:
            results = json.load(f)
    results[key] = {k: to_json(v) for k, v in result.items()}
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(results, f, indent=2)
    os.replace(tmp, path)
    print(f'Results saved to {path} ({key})')


def run_name(config):
    """wandb run name: <prior>_<sampler>_<input>, then the Dirichlet scale (dirichlet prior) or the schedule exponent c
    (discrete priors), with _<schedule> for a schedule other than cosine, e.g. dirichlet_quantile_map_sufficient_scale6,
    dirichlet_tuned_sufficient_lambda-1_eta0.5_scale6 ('tuned': with lambda and eta), marginal_sid_state_c2."""
    name = f'{config.model.prior}_{config.sampling.id}_{config.model.input}'
    if config.sampling.id == 'tuned':
        name = f'{name}_lambda{config.sampling.lambd:g}_eta{config.sampling.eta:g}'
    if config.model.get('schedule', 'cosine') != 'cosine':
        name = f"{name}_{config.model.schedule}"
    if config.model.prior == 'dirichlet':
        return f'{name}_scale{config.model.dirichlet_scale:g}'
    if config.model.get('schedule', 'cosine') == 'linear':
        return name
    return f'{name}_c{config.model.schedule_c:g}'


def update_config(config, dataset, work_type, args):
    config.dataset = dataset
    config.work_type = work_type
    config.denoiser_dir = args.denoiser_dir
    config.model.num_modes = args.num_modes
    config.model.hyp_loss = args.hyp_loss
    config.model.hyp_temperature = args.hyp_temperature
    config.model.schedule = args.schedule
    config.model.schedule_c = args.schedule_c
    config.model.dirichlet_scale = args.dirichlet_scale
    config.model.input = args.input
    config.model.time_input = args.time_input
    config.model.training_method = args.training_method
    config.sampling.sampling_method = args.sampling_method
    config.sampling.hyp_selection = args.hyp_selection
    config.sampling.greedy_last_step = args.greedy_last_step
    config.sampling.temperature = args.temperature
    config.sampling.lambd = args.lambd
    config.sampling.eta = args.eta
    config.sampling.fast_eval = args.fast_eval
    config.log.pgd = args.pgd
    return config


if __name__ == "__main__":
    main()


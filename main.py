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
        wandb.init(project=f'ModeFlowGraph_{dataset}', config=config, mode=args.wandb)
        dataloader = get_dataset(config)
        trainer = Trainer(dataloader, config)
        trainer.train()

    elif work_type == 'sample':
        torch.manual_seed(42)
        runs = []
        wandb.init(project=f'sid_{dataset}_sample', config=config, mode=args.wandb)
        times = []

        for r in [4, 8, 16, 32, 64, 128]:  # sweep over the number of sampling steps T
            config.model.T = r
            dataloader = get_dataset(config)
            trainer = Trainer(dataloader, config)
            with torch.no_grad():
                start_time = time.time()
                X, A, mask, mask_adj = trainer.sampler(config.log.n_samples_generation, trainer.denoiser,
                                                       method=config.sampling.id)
                sampling_time = time.time() - start_time
                wandb.log({'sampling_time': sampling_time})
                times.append(sampling_time)
                run = trainer.eval_samples(X, A, mask, mask_adj, ema=True)
                runs.append(run)

        keys = runs[0].keys()
        latex_format = {}
        for key in keys:
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
        wandb.init(project=f'sid_{dataset}_train_vs_test', config=config, mode=args.wandb)
        dataloader = get_dataset(config)
        n = config.log.n_samples_generation
        X, A, mask, mask_adj = sample_training_graphs(dataloader['train'].dataset, n, config.data.max_num_nodes)
        print(f'Evaluating {X.size(0)} training graphs against the test set')
        eval_samples = SamplingMetrics(dataset, config.data.max_num_nodes, sampling=True,
                                       ref_loader=dataloader['test'], ref_size=n)
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


def update_config(config, dataset, work_type, args):
    config.dataset = dataset
    config.work_type = work_type
    config.denoiser_dir = args.denoiser_dir
    config.model.num_modes = args.num_modes
    config.model.hyp_loss = args.hyp_loss
    config.model.hyp_temperature = args.hyp_temperature
    config.model.mi_weight = args.mi_weight
    config.model.div_weight = args.div_weight
    config.sampling.hyp_selection = args.hyp_selection
    return config


if __name__ == "__main__":
    main()


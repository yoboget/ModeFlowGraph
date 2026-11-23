"""
Sweep of the 'tuned' sampler (prior 'dirichlet', see sample.py and utils.noising.tuned_step) in a single run: every
combination of the number of sampling steps T (NFE), eta and lambd is sampled from the same trained denoiser and
evaluated.

    python sample_tuned_sweep.py --dataset qm9 --denoiser_dir ./checkpoints/qm9/<run>/files/ \
        --nfe 32 64 128 --eta 0 0.05 0.25 0.5 0.75 1 --lambd 0 0.1 1 10 100 -1

The numbers of sampling steps T are given by --nfe (utils/parser.py), by default the model.T of the config. The
sweep arguments are --eta, --lambd, --n_samples (graphs per configuration, default: the config's
log.n_samples_generation), --seed and --output. Every other argument is passed to main.py's parser (utils/parser.py),
e.g. --dataset, --denoiser_dir, --dirichlet_scale, --input, --training_method, --temperature, --wandb, --fast_eval
(evaluation as during training: validation reference, no NSPDK / PGD). sampling.id
is set to 'tuned'.

The dataset is loaded once and the denoiser once per T. Each configuration starts from the same seed, so for a given
T they share the initial noise and the number of nodes of each graph. The results (metrics, sampling time) are
written to the JSON file --output after each configuration; configurations already in it are skipped, so rerunning
the same command resumes the sweep. A summary table is printed at the end (and logged to wandb).
"""
import argparse
import json
import os
import sys
import time

import torch
import wandb
import yaml
from easydict import EasyDict as edict

from dataset import get_dataset
from main import update_config, run_name, results_path
from trainer import Trainer
from utils.parser import parse_args

SUMMARY_KEYS = {'qm9': ['valid', 'unique', 'novel', 'fcd', 'nspdk'], 'qm9H': ['valid', 'unique', 'novel', 'fcd', 'nspdk'],
                'zinc': ['valid', 'unique', 'novel', 'fcd', 'nspdk']}
GENERIC_KEYS = ['valid', 'degree', 'cluster', 'orbit', 'spectral', 'avg']


def parse_sweep_args():
    """The sweep arguments; the remaining ones are left in sys.argv for main.py's parser."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--eta', type=float, nargs='+', default=[0, 0.05, 0.25, 0.5, 0.75, 1],
                        help="'tuned' eta values, in [0, 1]")
    parser.add_argument('--lambd', type=float, nargs='+', default=[0, 0.1, 1, 10, 100, -1],
                        help="'tuned' lambda values (< 0: lambda = infinity)")
    parser.add_argument('--n_samples', type=int, default=None,
                        help='Graphs per configuration (default: log.n_samples_generation of the config, '
                             'log.n_val_samples with --fast_eval)')
    parser.add_argument('--seed', type=int, default=42, help='Seed of every configuration')
    parser.add_argument('--output', type=str, default=None,
                        help='Results JSON (default: ./sweeps/tuned_<dataset>_<schedule>_scale<scale>.json)')
    sweep_args, rest = parser.parse_known_args()
    sys.argv = sys.argv[:1] + rest
    for eta in sweep_args.eta:
        assert 0. <= eta <= 1., f'eta must be in [0, 1], got {eta}'
    return sweep_args


def to_json(value):
    if isinstance(value, str):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def key_of(nfe, eta, lambd):
    return f'T{nfe}_eta{eta:g}_lambda{lambd:g}'


def main():
    sweep = parse_sweep_args()
    args = parse_args()
    if args.wandb == 'no':
        args.wandb = 'disabled'
    elif args.wandb == 'on':
        args.wandb = 'online'
    elif args.wandb == 'off':
        args.wandb = 'offline'

    config = edict(yaml.load(open(f'./config/{args.dataset}.yaml', 'r'), Loader=yaml.FullLoader))
    config = update_config(config, args.dataset, 'sample', args)
    if config.model.prior != 'dirichlet':
        raise ValueError(f"The 'tuned' sampler needs the 'dirichlet' prior, the config of {args.dataset} has "
                         f"'{config.model.prior}'")
    config.sampling.id = 'tuned'
    n_samples = sweep.n_samples or (config.log.n_val_samples if args.fast_eval else config.log.n_samples_generation)
    nfes = args.nfe or [config.model.T]

    name = f'{run_name(config)}_sweep'
    output = sweep.output or results_path(config)   # sweeps/tuned_<dataset>_<schedule>_scale<scale>.json
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    results = {}
    if os.path.isfile(output):
        with open(output) as f:
            results = json.load(f)
        print(f'Resuming: {len(results)} configurations already in {output}')

    wandb.init(project=f'ModeFlowGraph_{args.dataset}_sample', config=config, mode=args.wandb, name=name)
    dataloader = get_dataset(config)
    n_configs = len(nfes) * len(sweep.eta) * len(sweep.lambd)
    done = 0
    for nfe in nfes:
        todo = [(eta, lambd) for eta in sweep.eta for lambd in sweep.lambd
                if key_of(nfe, eta, lambd) not in results]
        done += len(sweep.eta) * len(sweep.lambd) - len(todo)
        if not todo:
            continue
        config.model.T = nfe
        trainer = Trainer(dataloader, config)
        for eta, lambd in todo:
            done += 1
            print(f'=== [{done}/{n_configs}] T {nfe}, eta {eta:g}, lambda {lambd:g}', flush=True)
            trainer.sampler.eta, trainer.sampler.lambd = eta, lambd
            torch.manual_seed(sweep.seed)
            with torch.no_grad():
                start = time.time()
                X, A, mask, mask_adj = trainer.sampler(n_samples, trainer.denoiser, method='tuned')
                sampling_time = time.time() - start
                metrics = trainer.eval_samples(X, A, mask, mask_adj, ema=True)
            results[key_of(nfe, eta, lambd)] = {
                'nfe': nfe, 'eta': eta, 'lambd': lambd, 'n_samples': n_samples, 'sampling_time': sampling_time,
                **{k: to_json(v) for k, v in metrics.items()}}
            wandb.log({'sweep/nfe': nfe, 'sweep/eta': eta, 'sweep/lambd': lambd,
                       'sweep/sampling_time': sampling_time})
            tmp = output + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(results, f, indent=2)
            os.replace(tmp, output)

    # Summary of the requested configurations
    rows = [results[key_of(nfe, eta, lambd)] for nfe in nfes for eta in sweep.eta for lambd in sweep.lambd]
    keys = [k for k in SUMMARY_KEYS.get(args.dataset, GENERIC_KEYS) if k in rows[0]] + ['sampling_time']
    header = f'{"T":>5} {"eta":>6} {"lambda":>7} ' + ' '.join(f'{k:>13}' for k in keys)
    print(header)
    print('-' * len(header))
    for r in rows:
        print(f'{r["nfe"]:>5} {r["eta"]:>6g} {r["lambd"]:>7g} '
              + ' '.join(f'{r[k]:>13.4f}' if isinstance(r[k], float) else f'{str(r[k]):>13}' for k in keys))
    wandb.log({'sweep/summary': wandb.Table(columns=['nfe', 'eta', 'lambd'] + keys,
                                            data=[[r['nfe'], r['eta'], r['lambd']] + [r[k] for k in keys]
                                                  for r in rows])})
    print(f'Results: {output}')
    wandb.finish()


if __name__ == '__main__':
    main()

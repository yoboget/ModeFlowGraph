"""
Sweep of the Dirichlet scale (dirichlet_scale) for the samplers 'vector_field' and 'quantile_map' (prior 'dirichlet',
see sample.py) in a single run: every combination of scale, number of sampling steps T (NFE) and sampler is sampled
from the trained denoiser(s) and evaluated.

    python sample_scale_sweep.py --dataset qm9 --denoiser_dir ./checkpoints/qm9/<run>/files/ \
        --scales 3 4 5 6 8 10 --nfe 64 128

The sweep arguments (all optional) are
    --scales        the dirichlet_scale values (default 3 to 10)
    --samplers      default: vector_field quantile_map
    --n_samples     graphs per configuration (default: log.n_samples_generation, log.n_val_samples with --fast_eval)
    --seed          seed of every configuration (default 42)
    --sweeps_dir    directory of the results files (default sweeps)
Every configuration uses the single checkpoint --denoiser_dir, which must have input 'sufficient' and no time input:
such a denoiser does not depend on the scale (with input 'state' or --time_input the scale is fixed by the training).
The numbers of sampling steps are given by --nfe (utils/parser.py). Every other argument goes to main.py's parser
(--dataset, --denoiser_dir, --schedule, --input, --fast_eval, --pgd, --wandb, ...).

Timing (in seconds, CUDA synchronised), for each configuration:
    table_time       building the sampler's tables for this scale and T (Beta quantile tables / vector field tables)
    generation_time  generating the n_samples graphs (all the batches)
    sampling_time    total = table_time + generation_time (what main.py's sampling_time measures)

The denoiser is loaded once per (scale, T); for a given (scale, T) the samplers start from the same seed, so they share
the initial noise and the number of nodes of each graph. The results are saved after each configuration, with the
metrics, to sweeps/<sampler>_<dataset>_<schedule>_scale<scale>.json under the key T<nfe> (main.save_results, the
format read by misc/plots.py: python misc/plots.py --plot scale scale_time). Configurations already in their file are
skipped, so rerunning the same command resumes the sweep.
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
from main import update_config, results_path, save_result
from trainer import Trainer
from utils.parser import parse_args

SAMPLERS = ('vector_field', 'quantile_map')


def parse_sweep_args():
    """The sweep arguments; the remaining ones are left in sys.argv for main.py's parser."""
    # allow_abbrev=False: the arguments of main.py's parser must not be taken for abbreviations of these
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--scales', type=float, nargs='+', default=[1, 2, 3, 4, 5, 6],
                        help='dirichlet_scale values')
    parser.add_argument('--samplers', nargs='+', choices=SAMPLERS, default=list(SAMPLERS))
    parser.add_argument('--n_samples', type=int, default=None,
                        help='Graphs per configuration (default: log.n_samples_generation of the config, '
                             'log.n_val_samples with --fast_eval)')
    parser.add_argument('--seed', type=int, default=42, help='Seed of every configuration')
    parser.add_argument('--sweeps_dir', default='sweeps', help='Directory of the results files')
    sweep_args, rest = parser.parse_known_args()
    sys.argv = sys.argv[:1] + rest
    return sweep_args


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    sweep = parse_sweep_args()
    args = parse_args()
    args.wandb = {'no': 'disabled', 'on': 'online', 'off': 'offline'}.get(args.wandb, args.wandb)

    config = edict(yaml.load(open(f'./config/{args.dataset}.yaml', 'r'), Loader=yaml.FullLoader))
    config = update_config(config, args.dataset, 'sample', args)
    if config.model.prior != 'dirichlet':
        raise ValueError(f"The scale sweep needs the 'dirichlet' prior, the config of {args.dataset} has "
                         f"'{config.model.prior}'")
    if 'vector_field' in sweep.samplers and config.model.get('schedule', 'cosine') == 'sparse':
        raise ValueError("'vector_field' does not support the 'sparse' schedule: pass --schedule cosine or linear")
    if config.model.input != 'sufficient' or config.model.time_input:
        raise ValueError("The scale sweep samples every scale with the same checkpoint: it needs a denoiser with "
                         "--input sufficient and no --time_input (otherwise the scale is fixed by the training)")
    n_samples = sweep.n_samples or (config.log.n_val_samples if args.fast_eval else config.log.n_samples_generation)
    nfes = args.nfe or [config.model.T]

    name = f'dirichlet_scale_sweep_{config.model.input}_{config.model.get("schedule", "cosine")}'
    wandb.init(project=f'ModeFlowGraph_{args.dataset}_sample', config=config, mode=args.wandb, name=name)
    dataloader = get_dataset(config)
    n_configs = len(sweep.scales) * len(nfes) * len(sweep.samplers)
    done = 0
    summary = []
    for scale in sweep.scales:
        config.model.dirichlet_scale = scale
        for nfe in nfes:
            todo = []
            for sampler in sweep.samplers:
                config.sampling.id = sampler
                path = results_path(config, sweep.sweeps_dir)
                existing = json.load(open(path)) if os.path.isfile(path) else {}
                if f'T{nfe}' in existing:
                    done += 1
                    print(f'=== [{done}/{n_configs}] scale {scale:g}, T {nfe}, {sampler}: already in {path}')
                    summary.append(existing[f'T{nfe}'] | {'sampler': sampler, 'scale': scale})
                else:
                    todo.append(sampler)
            if not todo:
                continue
            config.model.T = nfe
            trainer = Trainer(dataloader, config)
            for sampler in todo:
                done += 1
                print(f'=== [{done}/{n_configs}] scale {scale:g}, T {nfe}, {sampler}', flush=True)
                config.sampling.id = sampler
                torch.manual_seed(sweep.seed)
                with torch.no_grad():
                    synchronize()
                    start = time.time()
                    trainer.sampler.build_tables(sampler)
                    synchronize()
                    table_time = time.time() - start
                    start = time.time()
                    X, A, mask, mask_adj = trainer.sampler(n_samples, trainer.denoiser, method=sampler)
                    synchronize()
                    generation_time = time.time() - start
                    sampling_time = table_time + generation_time
                    metrics = trainer.eval_samples(X, A, mask, mask_adj, ema=True)
                result = {'nfe': nfe, 'scale': scale, 'sampler': sampler, 'denoiser_dir': args.denoiser_dir,
                          'n_samples': n_samples, 'sampling_time': sampling_time, 'table_time': table_time,
                          'generation_time': generation_time, **metrics}
                save_result(results_path(config, sweep.sweeps_dir), f'T{nfe}', result)
                summary.append(result)
                wandb.log({'sweep/scale': scale, 'sweep/nfe': nfe, 'sweep/sampler': sampler,
                           'sweep/sampling_time': sampling_time, 'sweep/table_time': table_time,
                           'sweep/generation_time': generation_time})
                print(f'scale {scale:g}, T {nfe}, {sampler}: total sampling time {sampling_time:.1f} s '
                      f'(tables {table_time:.1f} s, generation {generation_time:.1f} s)', flush=True)

    # Summary
    keys = [k for k in ('valid', 'fcd', 'nspdk', 'pgd') if any(k in r for r in summary)]
    header = (f'{"scale":>6} {"T":>5} {"sampler":>13} ' + ' '.join(f'{k:>9}' for k in keys)
              + f' {"tables(s)":>10} {"total(s)":>9}')
    print(header)
    print('-' * len(header))
    for r in summary:
        vals = ' '.join(f'{r[k]:>9.4f}' if isinstance(r.get(k), (int, float)) else f'{"-":>9}' for k in keys)
        tt = f'{r["table_time"]:>10.1f}' if isinstance(r.get('table_time'), (int, float)) else f'{"-":>10}'
        print(f'{r["scale"]:>6g} {r["nfe"]:>5g} {r["sampler"]:>13} {vals} {tt} {r["sampling_time"]:>9.1f}')
    wandb.finish()


if __name__ == '__main__':
    main()

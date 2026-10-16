import time
import os
import torch
import wandb
from torchmetrics import MeanMetric, MaxMetric


class RunningMetric:
    def __init__(self, metric_names: list):
        self.metric_name = metric_names
        self.metrics = {'train': {}, 'iter': {}, 'val': {}}
        for metric_step in self.metrics:
            for metric in metric_names:
                if metric == 'n_max':
                    self.metrics[metric_step][metric] = MaxMetric()
                else:
                    self.metrics[metric_step][metric] = MeanMetric()

    def log(self, step, key, times=None, epoch=None):
        metrics = self.metrics[key]
        log_metrics = {name: metric.compute() for name, metric in metrics.items()}
        if times is not None:
            clock_time = time.time() - times[0]
            process_time = time.process_time() - times[1]
            log_metrics['clock_time'] = clock_time
            log_metrics['process_time'] = process_time
            log_metrics['step'] = step
            if epoch is not None:
                print(f'Running metrics for {key} after {epoch} epochs and {clock_time} seconds')
            else:
                print(f'Running metrics for {key} after {step} steps and {clock_time} seconds')
        else:
            if epoch is not None:
                print(f'Running metrics for {key} after {step} epochs')
            else:
                print(f'Running metrics for {key} after {step} steps')
        print(log_metrics)

        wandb.log({f'{key}/': log_metrics}, step=step)
        for metric in metrics.values():
            metric.reset()
        return log_metrics

    def step(self, to_log, train):
        train_metrics, iter_metrics, val_metrics = self.metrics.values()
        if train:
            for metric, values in zip(train_metrics.values(), to_log):
                metric.update(values)
            for metric, values in zip(iter_metrics.values(), to_log):
                metric.update(values)
        else:
            for metric, values in zip(val_metrics.values(), to_log):
                metric.update(values)

def save_model(metric, best_run, to_save, step, save_name,
               minimize=True):
    denoiser, opt = to_save

    if save_name not in best_run:
        best_run[save_name] = metric

    if minimize:
        condition = metric <= best_run[save_name]
    else:
        condition = metric >= best_run[save_name]
    if condition:
        best_run[save_name] = metric
        print(f'New best {save_name}: {metric}')
        save_dir = wandb.run.dir

        torch.save({
            'iteration': step,
            'denoiser': denoiser.state_dict(),
            'optimizer': opt.state_dict(),
            'metric': best_run[save_name]},
            os.path.join(save_dir, f'best_run_{save_name}.pt'))
    return best_run

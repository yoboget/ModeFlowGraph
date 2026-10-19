import time
import copy
import torch
import torch.nn.functional as F

from torch_geometric.data import Batch

from utils.extra_feat import ExtraFeatures
from utils.noising import NoiserDense, get_distribution
from utils.func import batch_to_dense, get_edge_mask
from utils.modes import mode_loss
from utils.node_distribution import get_num_nodes_distribution
from eval.metrics import SamplingMetrics
from models.loaders import load_denoiser
from logger import RunningMetric, save_model
from sample import Sampler


class Trainer:
    def __init__(self, loaders, config):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f'Run on {self.device}')

        self.sampling = config.work_type == 'sample'
        if not self.sampling:
            config.denoiser_dir = None

        self.loaders = loaders
        self.extra_features = ExtraFeatures(config)
        self.prior = config.model.prior
        self.num_modes = config.model.get('num_modes', 1)
        self.hyp_loss = config.model.get('hyp_loss', 'wta')
        self.mi_weight = config.model.get('mi_weight', 0.)
        self.div_weight = config.model.get('div_weight', 0.)

        ### LOAD MODELS ###
        self.denoiser, self.opt = load_denoiser(config, loaders, self.extra_features, self.device,
                                                prior=config.model.prior, denoiser_dir=config.denoiser_dir)
        # The EMA model is only used for validation and sampling, never trained directly
        self.denoiser_ema = copy.deepcopy(self.denoiser).eval()

        # Extract configuration variables
        self.epochs = config.training.epochs
        self.max_num_nodes = config.data.max_num_nodes

        # Define Logger
        # num_modes > 1 (0 otherwise): 'mode': commitment loss (wta) or responsibility entropy (mixture),
        # 'mi': I(k; G_0) with tempered responsibilities, 'div': mean divergence between the modes per element
        self.metric_names = ['loss', 'node', 'edge', 'mode', 'mi', 'div']
        self.metrics = RunningMetric(self.metric_names)

        self.n_logging_steps = config.log.n_loggin_steps
        self.n_logging_epochs = config.log.n_loggin_epochs
        self.best_run = {}

        ### NOISING AND SAMPLING ###
        node_marginals, edge_marginals = get_distribution(self.loaders['train'], config.dataset)
        self.noiser = NoiserDense(config.model.T, self.prior, node_marginals, edge_marginals)

        self.sampling_method = config.sampling.id
        num_nodes_distribution = get_num_nodes_distribution(self.loaders, self.max_num_nodes, config.dataset)
        self.sampler = Sampler(self.noiser, self.extra_features, num_nodes_distribution, self.max_num_nodes,
                               n_node_types=len(node_marginals), batch_size=config.log.sampling_batch_size,
                               device=self.device, hyp_selection=config.sampling.get('hyp_selection', 'sample'))

        ref_loader = self.loaders['test'] if self.sampling else self.loaders['val']
        self.eval_samples = SamplingMetrics(config.dataset, self.max_num_nodes, self.sampling, ref_loader=ref_loader)
        self.val_size = config.log.n_val_samples

        self.dataset = config.dataset

    def train(self) -> None:
        print(f'The training set contains {len(self.loaders["train"])} batches '
              f'with size {self.loaders["train"].batch_size}')
        starting_time = time.time(), time.process_time()
        self.step = 0
        print('Training starts...')
        for self.epoch in range(1, self.epochs + 1):
            # TRAIN
            for batch in self.loaders['train']:
                self.step += 1
                self.fit(batch.to(self.device), train=True)
                if self.step % self.n_logging_steps == 0:
                    self.metrics.log(self.step, key='iter', times=starting_time)

            self.metrics.log(self.step, key='train', times=starting_time)

            # VAL (EMA model)
            with torch.no_grad():
                for batch in self.loaders['val']:
                    self.fit(batch.to(self.device), train=False)
                val_metrics = self.metrics.log(self.step, key='val', times=starting_time)

                # SAMPLE EMA MODEL
                if self.epoch % self.n_logging_epochs == 0:
                    X, A, mask, mask_adj = self.sampler(self.val_size, self.denoiser_ema, method=self.sampling_method)
                    sampling_metrics = self.eval_samples(X, A, mask, mask_adj, step=self.step, epoch=self.epoch,
                                                         ema=True)
                    self.save_model((self.denoiser_ema, self.opt), sampling_metrics, val_metrics['loss'], ema=True)

    def fit(self, batch: Batch, train: bool = True):
        if train:
            self.opt.zero_grad()
            self.denoiser.train()
        # PREP
        X, A, mask = batch_to_dense(batch, max_num_nodes=self.max_num_nodes)
        adj_mask = get_edge_mask(mask)

        # NOISING: t ~ U(0, 1)
        alpha_bar = self.noiser.get_alpha_bar(torch.rand(X.size(0), device=self.device))
        X_noisy, A_noisy, (noised_nodes, noised_edges) = self.noiser(X, A, mask, alpha_bar)
        X_in = self.noiser.add_noise_level(X_noisy, alpha_bar)
        X_in, A_in = self.extra_features(X_in, A_noisy, mask)

        model = self.denoiser if train else self.denoiser_ema
        out = model(X_in, A_in, mask)

        if self.num_modes > 1:
            # each edge enters the likelihood once; with masking, only the masked elements are predicted
            edge_weight = adj_mask.triu(1)
            node_weight = mask if X.size(-1) > 1 else None
            if self.prior == 'masking':
                edge_weight = edge_weight & noised_edges
                if node_weight is not None:
                    node_weight = node_weight & noised_nodes
            loss, metrics = mode_loss(out, X, A.float(), node_weight, edge_weight, self.hyp_loss,
                                      self.mi_weight, self.div_weight, noise_level=1 - alpha_bar)
            metrics['loss'] = loss.item()
            self.optimize(loss, [metrics[name] for name in self.metric_names], train)
            return

        X_pred, A_pred = out

        X_pred = X_pred[mask]
        X_targ = X[mask]

        A_pred = A_pred[adj_mask]
        A_targ = A[adj_mask]

        if self.prior == 'masking':
            # Only the masked elements are predicted
            A_pred = A_pred[noised_edges[adj_mask]]
            A_targ = A_targ[noised_edges[adj_mask]]
            if noised_nodes is not None:
                X_pred = X_pred[noised_nodes[mask]]
                X_targ = X_targ[noised_nodes[mask]]

        if X_targ.size(-1) > 1:
            loss_x = F.cross_entropy(X_pred, X_targ)
            n, m = X_targ.size(0), A_targ.size(0)

            loss_a = F.cross_entropy(A_pred, A_targ)
            loss = (n / (m + n)) * loss_x + (m / (m + n)) * loss_a

        else:
            A_targ = A_targ[..., :1]
            loss_a = F.binary_cross_entropy(A_pred.sigmoid(), A_targ)
            loss = loss_a
            loss_x = torch.zeros(1, dtype=X_targ.dtype, device=X_targ.device)

        self.optimize(loss, [loss.item(), loss_x.item(), loss_a.item(), 0., 0., 0.], train)

    def optimize(self, loss, to_log, train):
        if train:
            loss.backward()
            self.opt.step()
            update_ema(self.denoiser, self.denoiser_ema)
        self.metrics.step(to_log, train)

    def save_model(self, to_save, metrics, loss, ema=False):
        suffix = '_ema' if ema else ''
        if self.dataset in ['zinc', 'qm9', 'qm9H']:
            for name in ['nspdk', 'fcd']:
                self.best_run = save_model(metrics[name], best_run=self.best_run, to_save=to_save,
                                           step=self.step, save_name=name + suffix)
        else:
            self.best_run = save_model(metrics['avg'], best_run=self.best_run, to_save=to_save,
                                       step=self.step, save_name='avg' + suffix)
            if 'valid' in metrics:
                self.best_run = save_model(metrics['valid'], best_run=self.best_run, to_save=to_save,
                                           step=self.step, save_name='valid' + suffix, minimize=False)
        self.best_run = save_model(loss, best_run=self.best_run, to_save=to_save,
                                   step=self.step, save_name='loss')


@torch.no_grad()
def update_ema(model, ema_model, ema_decay=0.999):
    for ema_param, model_param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data = ema_decay * ema_param.data + (1.0 - ema_decay) * model_param.data
    # Buffer copy for Batch_Norm modules
    for ema_buffer, model_buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.data = model_buffer.data.clone()


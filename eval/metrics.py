import importlib.util

import wandb

from eval.mol_metrics import gen_mol, mol_metric
from eval.pgd import compute_pgd, describe
from eval.stats3 import eval_graph_list, eval_fraction_unique_non_isomorphic_valid, is_planar_graph, is_sbm_graph
from utils.utils import get_networkx_from_dense
from utils.func import batch_to_dense

# graph_tool is only needed by is_sbm_graph, which imports it (see eval/stats3.py): check that it is installed without
# importing it here
GT = importlib.util.find_spec('graph_tool') is not None


# MMD kernel of the generic graph statistics, following the protocol under which each dataset is usually reported
MMD_KERNELS = {'enzymes': 'emd', 'planar': 'tv', 'sbm': 'tv'}


class SamplingMetrics:
    def __init__(self, dataset, max_num_nodes, sampling, ref_loader=None, ref_size=None, nspdk=True, pgd=False):
        """ref_size: if given, the test reference is a random subset of this size (default: the whole test set).
        nspdk: compute the NSPDK MMD (molecules).
        pgd: compute the PolyGraph Discrepancy (eval/pgd.py)."""
        self.dataset = dataset
        self.nspdk = nspdk
        self._pgd_ref = None   # PGD descriptors of the generic reference graphs, computed on first use
        self.ref_size = ref_size
        self.max_num_nodes = max_num_nodes
        self.sampling = sampling
        self.pgd = pgd
        if dataset in ['planar', 'sbm', 'enzymes']:
            self.val_loader = ref_loader
            batch = next(iter(ref_loader))
            X, A, mask = batch_to_dense(batch, max_num_nodes=max_num_nodes)
            self.ref_graphs = get_networkx_from_dense(X.argmax(-1), A.argmax(-1), mask)
            if ref_size is not None:  # the test loader is shuffled
                self.ref_graphs = self.ref_graphs[:ref_size]

    def __call__(self, X, A, mask, mask_adj, step=None, epoch=None, ema=False, invalid=None):
        """invalid: (n,) bool, graphs to count as invalid whatever their content (Sampler.nan_graphs: graphs whose
        sampling state became NaN / infinite, whose output is class 0 everywhere). They are left out of every metric
        computed on the generated graphs and count as invalid; their fraction is reported as nan_graphs."""
        ema_string = '_ema' if ema else ''
        n = X.size(0)
        keep = None
        if invalid is not None and invalid.any():
            keep = ~invalid.to(X.device)
            if not keep.any():
                metrics = {'valid': 0., 'nan_graphs': 1.}
                print(f'sampling{ema_string}: every graph has a NaN / infinite state', metrics)
                wandb.log({f'sampling{ema_string}/': metrics}, step=step)
                return metrics
            X, A, mask, mask_adj = X[keep], A[keep], mask[keep], mask_adj[keep]

        distrib_metric = self.eval_general_stats(X, A, mask, mask_adj)

        if self.dataset in ['qm9', 'zinc', 'qm9H']:
            metrics = self.molecular_metrics((X, A, mask), n_samples=n)

        elif self.dataset in ['planar', 'sbm', 'enzymes']:
            gen_graphs = get_networkx_from_dense(X, A, mask)
            metrics = self.generic_graph_metrics(gen_graphs, self.dataset)
            if keep is not None and 'valid' in metrics:     # fraction over all the graphs, the NaN ones invalid
                metrics['valid'] = metrics['valid'] * X.size(0) / n
        else:
            raise NotImplementedError('Metrics not implemented for this dataset.')
        if invalid is not None:
            metrics['nan_graphs'] = float(invalid.float().mean())
        if not self.sampling:
            metrics['epoch'] = epoch
        wandb.log({f'sampling{ema_string}/': metrics}, step=step)
        wandb.log({f'distributions{ema_string}/': distrib_metric}, step=step)
        print(f'sampling{ema_string}: ', metrics)
        print(f'distributions{ema_string}: ', distrib_metric)
        return metrics

    def molecular_metrics(self, gen_graphs, n_samples=None):
        """n_samples: number of generated graphs, including those left out as invalid (default: all are given)."""
        annots, adjs, mask = gen_graphs
        gen_mols, num_no_correct = gen_mol(annots, adjs, mask, self.dataset)
        n_samples = n_samples or annots.size(0)
        return mol_metric(gen_mols, self.dataset, num_no_correct / n_samples, n_samples, test_metrics=True,
                          ref_size=self.ref_size, nspdk=self.nspdk, pgd=self.pgd)

    def generic_graph_metrics(self, gen_graphs, dataset):
        methods = ['degree', 'cluster', 'orbit', 'spectral']
        metrics = eval_graph_list(self.ref_graphs, gen_graphs, methods=methods, kernel=MMD_KERNELS[dataset])
        avg = sum(metrics[m] for m in methods) / len(methods)    # the MMDs only, not their terms
        metrics['avg'] = avg
        if self.pgd:
            if self._pgd_ref is None:
                self._pgd_ref = describe(self.ref_graphs, 'graph')
            metrics.update(compute_pgd(self._pgd_ref, gen_graphs, 'graph'))
        if dataset == 'planar':
            u, n, v = eval_fraction_unique_non_isomorphic_valid(gen_graphs, self.ref_graphs,
                                                                validity_func=is_planar_graph)
            metrics['unique'], metrics['novel'], metrics['valid'] = u, n, v
        elif dataset == 'sbm':
            if GT:
                u, n, v = eval_fraction_unique_non_isomorphic_valid(gen_graphs, self.ref_graphs,
                                                                validity_func=is_sbm_graph)
                metrics['unique'], metrics['novel'], metrics['valid'] = u, n, v
        return metrics


    def eval_general_stats(self, X, A, mask, mask_adj):
        if self.dataset == 'zinc':
            node_names = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
            edge_names = ['no_bond', 'single', 'double', 'triple']
        elif self.dataset == 'qm9':
            node_names = ['C', 'N', 'O', 'F']
            edge_names = ['no_bond', 'single', 'double', 'triple']
        elif self.dataset == 'qm9H':
            node_names = ['H', 'C', 'N', 'O', 'F']
            edge_names = ['no_bond', 'single', 'double', 'triple']
        elif self.dataset in ['planar', 'sbm', 'enzymes']:
            node_names = []
            edge_names = ['no_edge', 'edge']

        if len(node_names) > 0:
            node_distrib = X[mask].bincount(minlength=X.shape[-1])
            node_distrib = node_distrib / node_distrib.sum()
            node_dict = {node_name: node_ratio for node_name, node_ratio in zip(node_names, node_distrib)}
        else:
            node_dict = {}

        edge_distrib = A[mask_adj].bincount(minlength=A.shape[-1])
        edge_distrib = edge_distrib / edge_distrib.sum()
        edge_dict = {edge_name: edge_ratio for edge_name, edge_ratio in zip(edge_names, edge_distrib)}

        return node_dict | edge_dict

"""
Forward (noising) process on dense graphs.

Graphs are dense one-hot tensors:
    x: (bs, n, dx)     node types (graphs without node types have dx = 1 and x = 1)
    a: (bs, n, n, de)  edge types, class 0 is 'no edge'
    node_mask: (bs, n) and adj_mask: (bs, n, n) (no self-loops) mark the existing nodes / node pairs.

Each node and edge independently keeps its clean value with probability alpha_bar and is otherwise
replaced by a draw from the prior:
    - 'marginal': the node / edge type distribution of the training set,
    - 'masking':  a mask token, added as an extra last class.
Graphs without node types only have their edges noised.
"""
import os.path
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from utils.func import get_edge_mask


PRIORS = ('marginal', 'masking')


class NoiserDense:
    def __init__(self, timesteps, prior, node_marginals, edge_marginals):
        """
        Args:
            timesteps: number of sampling steps T.
            prior: 'marginal' or 'masking'.
            node_marginals, edge_marginals: node / edge type distributions of the training set.
        """
        if prior not in PRIORS:
            raise NotImplementedError(f'Unknown prior {prior}. Available: {PRIORS}')
        self.T = timesteps
        self.prior = prior
        # Discrete cosine schedule (DiGress), used by the 'dif' and 'dfm' samplers
        self.schedule = DiscreteCosineSchedule(timesteps)

        if prior == 'marginal':
            self.node_prior, self.edge_prior = node_marginals, edge_marginals
        else:
            self.node_prior = mask_token(len(node_marginals) + 1)
            self.edge_prior = mask_token(len(edge_marginals) + 1)

    @staticmethod
    def get_alpha_bar(t):
        """Probability that an element is still clean at time t (t = 0: data, t = 1: prior)."""
        return torch.cos(0.5 * np.pi * torch.as_tensor(t, dtype=torch.float)) ** 2

    def to_noisy_classes(self, p):
        """Maps one-hot vectors / probabilities over the data classes to the noisy classes (adds the mask class)."""
        return F.pad(p, (0, 1)) if self.prior == 'masking' else p

    def from_noisy_classes(self, p):
        """Inverse of `to_noisy_classes` for elements that are not masked."""
        return p[..., :-1] if self.prior == 'masking' else p

    def add_noise_level(self, x, alpha_bar):
        """
        Builds the node input of the denoiser. With the 'marginal' prior, the noise level is given as two extra
        node features (alpha_bar, 1 - alpha_bar). With 'masking', the mask tokens already tell how noisy the graph is.
        alpha_bar: scalar or (bs,).
        """
        if self.prior == 'masking':
            return x
        alpha_bar = torch.as_tensor(alpha_bar, dtype=x.dtype, device=x.device)
        alpha_bar = alpha_bar.reshape(-1, 1, 1).expand(x.size(0), x.size(1), 1)
        return torch.cat((x, alpha_bar, 1 - alpha_bar), dim=-1)

    def __call__(self, x, a, node_mask, alpha_bar):
        """
        Samples G_t ~ q(G_t | G_0).

        Args:
            x, a: clean graph (one-hot over the data classes).
            node_mask: (bs, n).
            alpha_bar: scalar or (bs,), probability of keeping each node / edge clean.
        Returns:
            x_t, a_t: noisy graph (one-hot over the noisy classes).
            noised_nodes (bs, n) or None, noised_edges (bs, n, n): elements replaced by prior samples.
        """
        bs, n = node_mask.shape
        alpha_bar = torch.as_tensor(alpha_bar, dtype=torch.float, device=x.device).expand(bs)
        adj_mask = get_edge_mask(node_mask)

        if x.size(-1) > 1:
            noised_nodes = torch.bernoulli((1 - alpha_bar).view(bs, 1).expand(bs, n)).bool()
            prior_nodes = sample_nodes(self.node_prior.to(x.device).expand(bs, n, -1), node_mask)
            x = torch.where(noised_nodes.unsqueeze(-1), prior_nodes, self.to_noisy_classes(x))
            x = x * node_mask.unsqueeze(-1)
        else:
            noised_nodes = None

        noised_edges = torch.bernoulli((1 - alpha_bar).view(bs, 1, 1).expand(bs, n, n)).bool()
        noised_edges = noised_edges.triu(1)
        noised_edges = noised_edges | noised_edges.transpose(1, 2)
        prior_edges = sample_edges(self.edge_prior.to(a.device).expand(bs, n, n, -1), adj_mask)
        a = torch.where(noised_edges.unsqueeze(-1), prior_edges, self.to_noisy_classes(a))
        a = a * adj_mask.unsqueeze(-1)

        return x, a, (noised_nodes, noised_edges)


class DiscreteCosineSchedule:
    """Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ, indexed by t = 0, ..., T."""

    def __init__(self, timesteps, s=0.008):
        steps = timesteps + 2
        x = np.linspace(0, steps, steps)
        alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]

        self.betas = torch.from_numpy(betas).float()
        alphas = 1 - torch.clamp(self.betas, min=0, max=0.9999)
        self.alphas_bar = torch.exp(torch.cumsum(torch.log(alphas), dim=0))


class MarginalUniformTransition:
    """
    Transition matrices of a discrete diffusion with prior distribution m (rows: from, columns: to):
        Q_t     = (1 - beta_t) I + beta_t 1 m^T              one step, t - 1 -> t
        Q_bar_t = alpha_bar_t I + (1 - alpha_bar_t) 1 m^T    t steps,  0 -> t
    """

    def __init__(self, prior):
        self.prior = prior

    def get_Qt(self, beta_t):
        return self._mix(1 - beta_t)

    def get_Qt_bar(self, alpha_bar_t):
        return self._mix(alpha_bar_t)

    def _mix(self, w):
        k = self.prior.size(-1)
        identity = torch.eye(k, device=self.prior.device)
        return w.to(self.prior.device) * identity + (1 - w.to(self.prior.device)) * self.prior.expand(k, k)


def mask_token(n_classes):
    """Distribution putting all its mass on the last class (the mask token)."""
    return torch.eye(n_classes)[-1]


def sample_nodes(probs, node_mask):
    """One-hot samples of `probs` (bs, n, k) for the existing nodes, zeros elsewhere."""
    idx = torch.zeros(probs.shape[:-1], dtype=torch.long, device=probs.device)
    idx[node_mask] = Categorical(probs=probs[node_mask]).sample()
    return F.one_hot(idx, probs.size(-1)).float() * node_mask.unsqueeze(-1)


def sample_edges(probs, adj_mask):
    """Symmetric one-hot samples of `probs` (bs, n, n, k): the upper triangle is sampled and mirrored."""
    idx = torch.zeros(probs.shape[:-1], dtype=torch.long, device=probs.device)
    idx[adj_mask] = Categorical(probs=probs[adj_mask]).sample()
    idx = idx.triu(1)
    idx = idx + idx.transpose(1, 2)
    return F.one_hot(idx, probs.size(-1)).float() * adj_mask.unsqueeze(-1)


def get_distribution(loader, dataset):
    if os.path.isfile(f'./data/{dataset}/distributions.pt'):
        node_distrib, edge_distrib = torch.load(f'./data/{dataset}/distributions.pt')
    else:
        print('Recompute distributions')
        node_sum, edge_sum, n_node_pairs = 0, 0, 0
        for batch in loader:
            x = batch.x
            edge_attr = batch.edge_attr
            node_sum += x.sum(0)
            edge_sum += edge_attr.sum(0)
            n = batch.batch.bincount()
            n_node_pairs += (n * (n-1)).sum()
        edge_sum = torch.cat((n_node_pairs.unsqueeze(-1)-edge_sum.sum(-1), edge_sum), -1)
        node_distrib, edge_distrib = node_sum/node_sum.sum(), edge_sum/n_node_pairs
        torch.save((node_distrib, edge_distrib), f'./data/{dataset}/distributions.pt')
    return node_distrib, edge_distrib

"""
Forward (noising) process on dense graphs.

Graphs are dense one-hot tensors:
    x: (bs, n, dx)     node types (graphs without node types have dx = 1 and x = 1)
    a: (bs, n, n, de)  edge types, class 0 is 'no edge'
    node_mask: (bs, n) and adj_mask: (bs, n, n) (no self-loops) mark the existing nodes / node pairs.

Time runs from t = 0 (pure noise, the prior) to t = 1 (data). At time t, each node and edge independently keeps
its clean value with probability alpha_bar(t) = cos(pi (1 - t) / 2)^c and is otherwise replaced by a draw from the prior:
    - 'marginal': the node / edge type distribution of the training set,
    - 'masking':  a mask token, added as an extra last class.
Graphs without node types only have their edges noised.

NoiserDirichlet is a continuous alternative (prior 'dirichlet', as in dirisid's 'dirch'): each node / edge is a point
on the simplex, z_t ~ Dir(1 + alpha(t) z_1) with alpha(t) = -scale log(1 - t), so that z_t is uniform on the simplex at
t = 0 and concentrates on the vertex z_1 as t -> 1.
"""
import os.path
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.dirichlet import Dirichlet

from utils.func import get_edge_mask


PRIORS = ('marginal', 'masking', 'dirichlet')
# What the denoiser predicts (training_method) / what the 'dif' sampler uses (sampling_method), see
# NoiserDense.convert_parametrization
PARAMETRIZATIONS = ('denoiser', 'cavity')


class NoiserDense:
    def __init__(self, timesteps, prior, node_marginals, edge_marginals, schedule_c=2.):
        """
        Args:
            timesteps: number of sampling steps T, on the uniform grid t = 0, 1 / T, ..., 1.
            prior: 'marginal' or 'masking'.
            node_marginals, edge_marginals: node / edge type distributions of the training set.
            schedule_c: exponent c of the schedule alpha_bar(t) = cos(pi t / 2)^c.
        """
        if prior not in PRIORS or prior == 'dirichlet':
            raise NotImplementedError(f'Unknown prior {prior} for NoiserDense. Available: {PRIORS[:2]} '
                                      f"(use NoiserDirichlet for 'dirichlet')")
        self.T = timesteps
        self.prior = prior
        self.schedule_c = schedule_c

        if prior == 'marginal':
            self.node_prior, self.edge_prior = node_marginals, edge_marginals
        else:
            self.node_prior = mask_token(len(node_marginals) + 1)
            self.edge_prior = mask_token(len(edge_marginals) + 1)

    def get_alpha_bar(self, t):
        """Probability that an element is clean at time t (t = 0: prior, t = 1: data)."""
        # clamp: cos(pi / 2) is slightly negative in float, which gives NaN for non-integer c
        return torch.cos(0.5 * np.pi * (1 - torch.as_tensor(t, dtype=torch.float))).clamp(min=0) ** self.schedule_c

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
        Samples G_t ~ q(G_t | G_1).

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

    def log_q_zt_given_z1(self, z_t, alpha_bar, kind):
        """
        log q(z_t | z_1 = j) for every data class j, 'marginal' prior: alpha_bar delta(z_t, j) + (1 - alpha_bar) m(z_t).
        z_t: (bs, ..., k) one-hot, alpha_bar: scalar or (bs,), kind: 'node' or 'edge'. Returns (bs, ..., k).
        Padding elements (z_t = 0) get the same value for every class, so converting them changes nothing.
        """
        prior = (self.node_prior if kind == 'node' else self.edge_prior).to(z_t.device)
        alpha_bar = torch.as_tensor(alpha_bar, dtype=z_t.dtype, device=z_t.device)
        if alpha_bar.ndim == 1:
            alpha_bar = alpha_bar.view(-1, *[1] * (z_t.ndim - 1))
        m_zt = (z_t * prior).sum(-1, keepdim=True)
        return torch.log((alpha_bar * z_t + (1 - alpha_bar) * m_zt).clamp_min(1e-20))

    def convert_parametrization(self, logits, z_t, alpha_bar, kind, source, target):
        """
        Converts the prediction of a node / edge z between the two parametrizations of the denoiser:
            'denoiser': p(z_1 | G_t), the prediction given the whole noisy graph,
            'cavity':   c(z_1) = p(z_1 | G_t without z_t), without the evidence of the element's own noisy value.
        The elements are noised independently, so z_t only depends on z_1 and
            p(z_1 | G_t) ∝ c(z_1) q(z_t | z_1).
        'masking' prior: no conversion. A masked z_t has q(z_t | z_1) = 1 - alpha_bar for every z_1, so both are
        equal; an unmasked z_t is known to be clean, and the samplers never use the prediction there.

        Args:
            logits: (bs, ..., k) logits or log-probabilities over the data classes, or (bs, ..., M, k) for the M modes
                of a multi-mode denoiser.
            z_t: (bs, ..., k) one-hot noisy elements. alpha_bar: scalar or (bs,), the noise level of G_t.
        Returns:
            log-probabilities in the target parametrization, same shape as logits.
        """
        assert source in PARAMETRIZATIONS and target in PARAMETRIZATIONS, (source, target)
        if source == target or self.prior == 'masking':
            return logits
        log_q = self.log_q_zt_given_z1(z_t, alpha_bar, kind)
        if logits.ndim == z_t.ndim + 1:  # mode dimension
            log_q = log_q.unsqueeze(-2)
        sign = 1. if source == 'cavity' else -1.
        return (logits.log_softmax(-1) + sign * log_q).log_softmax(-1)


class NoiserDirichlet:
    """
    Dirichlet noising (dirisid's 'dirch' prior): each node / edge z_1 (one-hot over k classes) becomes
        z_t ~ Dir(1 + alpha(t) z_1),    alpha(t) = -scale log(1 - t),
    i.e. concentration 1 + alpha(t) on the data class and 1 on the others. At t = 0, Dir(1, ..., 1) is uniform on the
    simplex; as t -> 1, alpha(t) -> infinity and z_t -> z_1. Edges are sampled on the upper triangle and mirrored.

    Same interface as NoiserDense. The noise level passed around (get_alpha_bar) is t itself, and the denoiser gets
    (t, 1 - t) as two extra node features. Only the 'denoiser' parametrization and 'sid' sampling are supported.
    """

    def __init__(self, timesteps, scale=3.):
        """
        Args:
            timesteps: number of sampling steps T, on the uniform grid t = 0, 1 / T, ..., 1.
            scale: alpha(t) = -scale log(1 - t).
        """
        self.T = timesteps
        self.prior = 'dirichlet'
        self.scale = scale

    def get_alpha_bar(self, t):
        """Noise level of the Dirichlet path: t itself (t = 0: uniform on the simplex, t = 1: data)."""
        return torch.as_tensor(t, dtype=torch.float)

    def get_concentration(self, t):
        """alpha(t) = -scale log(1 - t), the extra concentration on the data class (t < 1)."""
        return -self.scale * torch.log1p(-t)

    def to_noisy_classes(self, p):
        return p

    def from_noisy_classes(self, p):
        return p

    def add_noise_level(self, x, t):
        """Node input of the denoiser: the noise level (t, 1 - t) as two extra node features. t: scalar or (bs,)."""
        t = torch.as_tensor(t, dtype=x.dtype, device=x.device)
        t = t.reshape(-1, 1, 1).expand(x.size(0), x.size(1), 1)
        return torch.cat((x, t, 1 - t), dim=-1)

    def __call__(self, x, a, node_mask, t):
        """
        Samples G_t ~ q(G_t | G_1).

        Args:
            x, a: clean graph (one-hot over the data classes).
            node_mask: (bs, n).
            t: scalar or (bs,) in [0, 1).
        Returns:
            x_t, a_t: noisy graph, points on the simplex (zeros on padding, no self-loops).
            (None, None): no element is 'masked' (kept for the interface of NoiserDense).
        """
        bs, n = node_mask.shape
        t = torch.as_tensor(t, dtype=torch.float, device=a.device).expand(bs)
        alpha = self.get_concentration(t)
        adj_mask = get_edge_mask(node_mask)

        if x.size(-1) > 1:
            x = Dirichlet(1 + alpha.view(bs, 1, 1) * x).sample() * node_mask.unsqueeze(-1)

        a = Dirichlet(1 + alpha.view(bs, 1, 1, 1) * a).sample()
        a = symmetrize_upper(a) * adj_mask.unsqueeze(-1)
        return x, a, (None, None)

    def sample_prior(self, node_mask, n_node_types, n_edge_types):
        """G_0: every node and edge uniform on the simplex (nodes without types are left at 1)."""
        bs, n = node_mask.shape
        device = node_mask.device
        if n_node_types > 1:
            x = Dirichlet(torch.ones(bs, n, n_node_types, device=device)).sample() * node_mask.unsqueeze(-1)
        else:
            x = node_mask.unsqueeze(-1).float()
        a = Dirichlet(torch.ones(bs, n, n, n_edge_types, device=device)).sample()
        a = symmetrize_upper(a) * get_edge_mask(node_mask).unsqueeze(-1)
        return x, a

    def convert_parametrization(self, logits, z_t, alpha_bar, kind, source, target):
        if source != target:
            raise NotImplementedError("The 'dirichlet' prior only supports the 'denoiser' parametrization.")
        return logits


def symmetrize_upper(a):
    """(bs, n, n, k): keeps the strict upper triangle and mirrors it."""
    a = a.permute(0, 3, 1, 2).triu(1)
    return (a + a.transpose(-1, -2)).permute(0, 2, 3, 1)


class MarginalUniformTransition:
    """
    Noising transition matrices with prior distribution m (rows: from, columns: to). Noising from time t to an
    earlier time s <= t keeps an element clean with probability alpha_bar(s) / alpha_bar(t):
        Q(alpha) = alpha I + (1 - alpha) 1 m^T,
    so q(z_s | z_1) = Q(alpha_bar(s)) and q(z_s | z_t) = Q(alpha_bar(s) / alpha_bar(t)).
    """

    def __init__(self, prior):
        self.prior = prior

    def get_Q(self, alpha):
        return self._mix(torch.as_tensor(alpha, dtype=torch.float))

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

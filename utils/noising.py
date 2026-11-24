"""
Forward (noising) process on dense graphs.

Graphs are dense one-hot tensors:
    x: (bs, n, dx)     node types (graphs without node types have dx = 1 and x = 1)
    a: (bs, n, n, de)  edge types, class 0 is 'no edge'
    node_mask: (bs, n) and adj_mask: (bs, n, n) (no self-loops) mark the existing nodes / node pairs.

Time runs from t = 0 (pure noise, the prior) to t = 1 (data). At time t, each node and edge independently keeps
its clean value with probability alpha_bar(t) and is otherwise replaced by a draw from the prior, with the schedule
'cosine' alpha_bar(t) = cos(pi (1 - t) / 2)^c or 'linear' alpha_bar(t) = t:
    - 'marginal': the node / edge type distribution of the training set,
    - 'masking':  a mask token, added as an extra last class.
Graphs without node types only have their edges noised.

NoiserDirichlet is a continuous alternative (prior 'dirichlet', as in dirisid's 'dirch'): each node / edge is a point
on the simplex, z_t ~ Dir(1 + alpha(t) z_1) with alpha(t) = -scale log(1 - t) (schedule 'cosine', the default) or
alpha(t) = scale t ('linear', as in Dirichlet flow matching with alpha_max = 1 + scale), so that z_t is uniform on the
simplex at t = 0 and concentrates on the vertex z_1 as t -> 1. The schedule 'sparse' (dirichlet only) is another path,
z_t ~ Dir(alpha(t) z_1 + (1 - alpha(t)) 1) with alpha(t) = 1 - (1 - t)^scale: the concentration of the other classes
goes from 1 to 0, and z_t reaches the vertex z_1 exactly at t = 1.
"""
import os.path
import numpy as np
from scipy.special import betainc, betaincinv
from scipy.special import beta as beta_function
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.beta import Beta
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.gamma import Gamma

from utils.func import get_edge_mask


PRIORS = ('marginal', 'masking', 'dirichlet')
# What the denoiser predicts (training_method) / what the 'dif' sampler uses (sampling_method), see
# NoiserDense.convert_parametrization
PARAMETRIZATIONS = ('denoiser', 'cavity')
SCHEDULES = ('cosine', 'linear', 'sparse')     # 'sparse': prior 'dirichlet' only


class NoiserDense:
    def __init__(self, timesteps, prior, node_marginals, edge_marginals, schedule_c=2., schedule='cosine'):
        """
        Args:
            timesteps: number of sampling steps T, on the uniform grid t = 0, 1 / T, ..., 1.
            prior: 'marginal' or 'masking'.
            node_marginals, edge_marginals: node / edge type distributions of the training set.
            schedule_c: exponent c of the 'cosine' schedule alpha_bar(t) = cos(pi (1 - t) / 2)^c.
            schedule: 'cosine' or 'linear' (alpha_bar(t) = t).
        """
        if prior not in PRIORS or prior == 'dirichlet':
            raise NotImplementedError(f'Unknown prior {prior} for NoiserDense. Available: {PRIORS[:2]} '
                                      f"(use NoiserDirichlet for 'dirichlet')")
        self.T = timesteps
        self.prior = prior
        self.schedule_c = schedule_c
        if schedule not in ('cosine', 'linear'):
            raise NotImplementedError(f"Unknown schedule {schedule} for the prior {prior}. Available: cosine, linear "
                                      f"('sparse' is for the 'dirichlet' prior)")
        self.schedule = schedule

        if prior == 'marginal':
            self.node_prior, self.edge_prior = node_marginals, edge_marginals
        else:
            self.node_prior = mask_token(len(node_marginals) + 1)
            self.edge_prior = mask_token(len(edge_marginals) + 1)

    def get_alpha_bar(self, t):
        """Probability that an element is clean at time t (t = 0: prior, t = 1: data)."""
        t = torch.as_tensor(t, dtype=torch.float)
        if self.schedule == 'linear':
            return t.clamp(0, 1)
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

    def log_likelihood(self, z_t, alpha_bar, kind):
        """
        log q(z_t | z_1 = b) for every data class b, up to a constant in b (see `sufficient_stat`).
            'marginal': alpha_bar delta(z_t, b) + (1 - alpha_bar) m(z_t),
            'masking':  0 for every b if z_t is masked (uniform), log one-hot(z_t) otherwise.
        z_t: (bs, ..., k) one-hot over the noisy classes, alpha_bar: scalar or (bs,), kind: 'node' or 'edge'.
        Returns (bs, ..., k) over the data classes. Padding elements (z_t = 0) get the same value for every class.
        """
        if self.prior == 'masking':
            is_masked = z_t[..., -1:].bool()
            return torch.where(is_masked, torch.zeros_like(z_t[..., :-1]), z_t[..., :-1].clamp_min(1e-20).log())
        prior = (self.node_prior if kind == 'node' else self.edge_prior).to(z_t.device)
        alpha_bar = expand_noise_level(alpha_bar, z_t)
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
        if self.prior == 'masking':
            assert source in PARAMETRIZATIONS and target in PARAMETRIZATIONS, (source, target)
            return logits
        return convert_parametrization(self, logits, z_t, alpha_bar, kind, source, target)


class NoiserDirichlet:
    """
    Dirichlet noising (dirisid's 'dirch' prior): each node / edge z_1 (one-hot over k classes) becomes
        z_t ~ Dir(1 + alpha(t) z_1),    alpha(t) = -scale log(1 - t)  (schedule 'cosine')  or  scale t  ('linear'),
    i.e. concentration 1 + alpha(t) on the data class and 1 on the others. At t = 0, Dir(1, ..., 1) is uniform on the
    simplex; as t -> 1, alpha(t) -> infinity and z_t -> z_1 ('cosine'), or alpha(1) = scale ('linear': the path is cut
    at the concentration 1 + scale, alpha_max of Dirichlet flow matching, and the samplers' last step draws from the
    prediction). Edges are sampled on the upper triangle and mirrored.

    Schedule 'sparse': z_t ~ Dir(alpha(t) z_1 + (1 - alpha(t)) 1), alpha(t) = 1 - (1 - t)^scale, i.e. concentration 1 on
    the data class and b(t) = (1 - t)^scale on the others. Uniform at t = 0, exactly the vertex z_1 at t = 1. It has the
    mean of the 'cosine' path with A = alpha / (1 - alpha), with all the concentrations divided by 1 + A: the points
    are sparser (close to the faces of the simplex). scale 1 is linear in alpha; scale ~2 spreads the information
    about z_1 evenly over t, larger values bring it earlier.

    In general z_t | z_1 = i ~ Dir(b(t) + (a(t) - b(t)) e_i) (get_dirichlet_params): a = 1 + alpha, b = 1 for
    'cosine' / 'linear', a = 1, b = 1 - alpha for 'sparse'. log q(z_t | z_1 = i) = (a - b) log z_i + const, with
    a - b = alpha(t) (get_concentration) in both cases.

    Same interface as NoiserDense. The noise level passed around (get_alpha_bar) is t itself, and the denoiser gets
    (t, 1 - t) as two extra node features (model input 'state'). Both parametrizations ('denoiser', 'cavity') are
    supported; the samplers are 'sid' and 'quantile_map'.
    """

    def __init__(self, timesteps, scale=3., schedule='cosine'):
        """
        Args:
            timesteps: number of sampling steps T, on the uniform grid t = 0, 1 / T, ..., 1.
            scale: alpha(t) = -scale log(1 - t) ('cosine'), scale t ('linear') or 1 - (1 - t)^scale ('sparse').
            schedule: 'cosine', 'linear' or 'sparse'.
        """
        if schedule not in SCHEDULES:
            raise NotImplementedError(f'Unknown schedule {schedule}. Available: {SCHEDULES}')
        self.T = timesteps
        self.prior = 'dirichlet'
        self.scale = scale
        self.schedule = schedule

    def get_alpha_bar(self, t):
        """Noise level of the Dirichlet path: t itself (t = 0: uniform on the simplex, t = 1: data)."""
        return torch.as_tensor(t, dtype=torch.float)

    def get_concentration(self, t):
        """alpha(t) = a(t) - b(t), the coefficient of log q(z_t | z_1 = i) = alpha(t) log z_i + const:
        -scale log(1 - t) ('cosine', t < 1), scale t ('linear') or 1 - (1 - t)^scale ('sparse')."""
        if self.schedule == 'linear':
            return self.scale * t
        if self.schedule == 'sparse':
            return 1 - (1 - t).clamp_min(0) ** self.scale
        return -self.scale * torch.log1p(-t)

    def get_dirichlet_params(self, t):
        """(a(t), b(t)): z_t | z_1 = i ~ Dir(b(t) + (a(t) - b(t)) e_i), concentration a on the data class and b on the
        others. t: tensor."""
        if self.schedule == 'sparse':
            b = (1 - t).clamp_min(0) ** self.scale
            return torch.ones_like(b), b
        return 1 + self.get_concentration(t), torch.ones_like(t)

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
        conc_true, conc_other = self.get_dirichlet_params(t)
        adj_mask = get_edge_mask(node_mask)

        if x.size(-1) > 1:
            x = Dirichlet(self._concentrations(x, conc_true, conc_other)).sample() * node_mask.unsqueeze(-1)

        a = Dirichlet(self._concentrations(a, conc_true, conc_other)).sample()
        a = symmetrize_upper(a) * adj_mask.unsqueeze(-1)
        return x, a, (None, None)

    @staticmethod
    def _concentrations(z1, conc_true, conc_other):
        """b + (a - b) z_1 for one-hot z_1 (..., k), a, b: (bs,). Padding rows (z_1 = 0) get 1 (uniform), and b is
        kept > 0 ('sparse' at t -> 1), so that the Dirichlet is always defined."""
        shape = (-1,) + (1,) * (z1.ndim - 1)
        conc_true, conc_other = conc_true.view(shape), conc_other.clamp_min(1e-20).view(shape)
        conc = conc_other + (conc_true - conc_other) * z1
        return torch.where(z1.sum(-1, keepdim=True) > 0, conc, torch.ones_like(conc))

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

    def log_likelihood(self, z_t, t, kind=None):
        """
        log q(z_t | z_1 = b) for every class b, up to a constant in b. The normalizer of Dir(b(t) + alpha(t) e_b) does
        not depend on b, so this is alpha(t) log z_{t,b} (get_concentration). z_t: (bs, ..., k) on the simplex, t: scalar or (bs,).
        Padding elements (z_t = 0) get the same value for every class.
        """
        alpha = self.get_concentration(expand_noise_level(t, z_t))
        return alpha * z_t.clamp_min(1e-30).log()

    def convert_parametrization(self, logits, z_t, t, kind, source, target):
        """Denoiser <-> cavity, see NoiserDense.convert_parametrization (z_t on the simplex, t the noise level)."""
        return convert_parametrization(self, logits, z_t, t, kind, source, target)


LOGIT_MAX = 700.     # |logit| bound of the quantile tables (float64 exponent range)


def beta_logit_quantiles(p, q, v):
    """
    logit F^{-1}(u) = log x - log(1 - x) of Beta(p, q) at the levels u = sigmoid(v) (v increasing), accurate at both
    ends: log x from betaincinv(p, q, u) when x < 1/2, log(1 - x) from betaincinv(q, p, 1 - u) when 1 - x < 1/2
    (1 - X ~ Beta(q, p)), with u and 1 - u = sigmoid(-v) each computed to full precision. Where betaincinv fails
    (NaN for very small levels) or x / 1 - x underflow, the tails are extrapolated linearly in v from the valid part:
    logit(x) ~ (v + c) / p as u -> 0 and (v + c') / q as u -> 1. Clipped to [-LOGIT_MAX, LOGIT_MAX] and made
    non-decreasing (numerical noise of betaincinv for small p, q).
    """
    with np.errstate(divide='ignore', under='ignore', invalid='ignore', over='ignore'):
        u, u_c = 1 / (1 + np.exp(-v)), 1 / (1 + np.exp(v))
        x = betaincinv(p, q, u)
        w = betaincinv(q, p, u_c)
        log_x = np.where(x < 0.5, np.log(x), np.log1p(-w))
        log_w = np.where(w < 0.5, np.log(w), np.log1p(-x))
        logit = log_x - log_w
    valid = np.isfinite(logit)
    assert valid.sum() >= 4, (p, q)
    idx = np.flatnonzero(valid)
    lo, hi = idx[:2], idx[-2:]
    left, right = np.arange(len(v)) < idx[0], np.arange(len(v)) > idx[-1]
    slope_lo = (logit[lo[1]] - logit[lo[0]]) / (v[lo[1]] - v[lo[0]])
    slope_hi = (logit[hi[1]] - logit[hi[0]]) / (v[hi[1]] - v[hi[0]])
    logit = np.where(left, logit[lo[0]] + (v - v[lo[0]]) * slope_lo, logit)
    logit = np.where(right, logit[hi[1]] + (v - v[hi[1]]) * slope_hi, logit)
    # isolated failures inside the valid range: interpolate
    inside = ~np.isfinite(logit)
    if inside.any():
        logit[inside] = np.interp(v[inside], v[~inside], logit[~inside])
    return np.maximum.accumulate(np.clip(logit, -LOGIT_MAX, LOGIT_MAX))


class BetaQuantileTable:
    """
    Tables of the CDF / inverse CDF of the Beta marginals of the Dirichlet path on the sampling grid t_s = s / T,
    s = 0, ..., T - 1 (the path may be singular at t = 1). With z_t | z_1 = i ~ Dir(b_s + (a_s - b_s) e_i)
    (NoiserDirichlet.get_dirichlet_params):
        family 0: z_i ~ Beta(a_s, (k - 1) b_s), what the quantile map transports;
        family r = 1, ..., k - 2 ('sparse' only, b varies): the stick-breaking fractions of the other coordinates,
                  y = z_{-i} / (1 - z_i) ~ Dir(b_s, ..., b_s), whose fraction with r coordinates after it is
                  Beta(b_s, r b_s).
    For 'cosine' / 'linear', b = 1 and the other coordinates are uniform at every t: only family 0 is needed.

    Both variables are logits, x as logit(x) = log x - log(1 - x) and the level u as logit(u), so that values within
    1e-300 of 0 or 1 are resolved (z_i is close to 1 at low noise, and with 'sparse' the other coordinates are close
    to 0) and the tables are close to linear in the Beta tails (logit(x) ~ logit(u) / p or / q there). Both directions
    are piecewise-linear interpolations of the same table (logit(x) as a function of logit(u)) on a sinh-spaced grid of
    logit(u) in [-LOGIT_MAX, LOGIT_MAX] (u from 1e-304 to 1 - 1e-304), extrapolated linearly beyond. float64.
    """

    def __init__(self, noiser, k, n_grid=4096, v_max=LOGIT_MAX, sinh_rate=13., device='cpu'):
        t = torch.arange(noiser.T, dtype=torch.float64) / noiser.T
        a, b = (v.numpy() for v in noiser.get_dirichlet_params(t))
        b = np.maximum(b, 1e-300)
        # logit of the levels, sinh-spaced: ~0.02 apart around 0, ~4 apart in the (near-linear) tails
        u = v_max * np.sinh(sinh_rate * np.linspace(-1., 1., n_grid)) / np.sinh(sinh_rate)
        self.sticks = bool(np.any(b != 1.)) and k > 2
        families = [(a, (k - 1) * b)] + ([(b, r * b) for r in range(1, k - 1)] if self.sticks else [])
        logit = np.stack([np.stack([beta_logit_quantiles(p, q, u) for p, q in zip(ps, qs)])
                          for ps, qs in families])                     # (F, T, G), non-decreasing in u
        assert np.isfinite(logit).all()
        self.u = torch.as_tensor(u, device=device)
        self.logit = torch.as_tensor(logit, device=device)
        self.a = torch.as_tensor(a, device=device)                     # (T,) concentration of the data class
        self.b = torch.as_tensor(b, device=device)                     # (T,) concentration of the other classes

    @staticmethod
    def _interp(q, xs, ys):
        """Piecewise-linear interpolation, extrapolated linearly from the end segments: in logit space the Beta tails
        are asymptotically linear (logit(x) ~ (logit(u) + c) / p), so levels beyond 1e-20 / 1 - 1e-20 stay accurate."""
        right = torch.searchsorted(xs, q.contiguous(), right=True).clamp(1, xs.numel() - 1)
        x_l, x_r, y_l, y_r = xs[right - 1], xs[right], ys[right - 1], ys[right]
        dx = x_r - x_l
        slope = torch.where(dx > 0, (y_r - y_l) / dx.clamp_min(1e-300), torch.zeros_like(dx))
        return y_l + (q - x_l) * slope

    def cdf(self, logit, s, family=0):
        """logit(F(x)) from logit(x), for the Beta of the family at t_s (family 0: Beta(a_s, (k - 1) b_s))."""
        return self._interp(logit, self.logit[family, s], self.u)

    def ppf(self, u, s, family=0):
        """logit(F^{-1}(u)) from logit(u), for the Beta of the family at t_s."""
        return self._interp(u, self.u, self.logit[family, s])

    def transport(self, logit, s, family=0):
        """Quantile map from t_s to t_{s+1} in logit space: logit F_{s+1}^{-1}(F_s(x))."""
        return self.ppf(self.cdf(logit, s, family), s + 1, family).clamp(-LOGIT_MAX, LOGIT_MAX)


class DirichletConditionalFlow:
    """
    C factor of the conditional vector field of the Dirichlet path, as in Dirichlet flow matching
    (utils/flow_utils.DirichletConditionalFlow, 'original' method): for z ~ Dir(1 + (a - 1) e_i), the coordinate z_i
    is Beta(a, k - 1) and the path in a is generated by
        u_i(z, a) = C(z_i, a) (e_i - z),    C(x, a) = -dI_x(a, b)/da B(a, b) / ((1 - x)^b x^(a - 1)),  b = k - 1,
    with I the regularized incomplete Beta function. Same numerics as the original: the Beta CDFs are tabulated on
    the concentrations a = alpha_min, alpha_min + alpha_spacing, ... and on 1000 uniform points of [0, 1], dI/da is
    their finite difference in a, taken at the tabulated a nearest to the query and linearly interpolated in x, and
    B(a, b) / ((1 - x)^b x^(a - 1)) is computed directly in the dtype of x (it can underflow / overflow).
    """

    def __init__(self, K, alpha_min=1., alpha_max=100., alpha_spacing=0.001):
        self.alphas = np.arange(alpha_min, alpha_max + alpha_spacing, alpha_spacing)
        self.bs = np.linspace(0, 1, 1000)
        self.beta_cdfs = np.array([betainc(alph, K - 1, self.bs) for alph in self.alphas])
        self.beta_cdfs_derivative = np.diff(self.beta_cdfs, axis=0) / alpha_spacing
        self.K = K

    def c_factor(self, bs, alpha):
        # As the original run with NumPy 1.x: the scalars do not change the precision of bs. alpha is a Python
        # float (the original passes s.item()), B(alpha, K - 1) and the zeros of np.where have the dtype of bs: with
        # NumPy >= 2 (NEP 50) a float64 scalar would turn the computation into float64 and remove the float32
        # underflow of bs ** (alpha - 1). The precision is chosen by the caller through the dtype of bs.
        alpha = float(alpha)
        zero = bs.dtype.type(0)
        out1 = bs.dtype.type(beta_function(alpha, self.K - 1))
        out2 = np.where(bs < 1, out1 / ((1 - bs) ** (self.K - 1)), zero)
        out = np.where((bs ** (alpha - 1)) > 0, out2 / (bs ** (alpha - 1)), zero)
        I_func = self.beta_cdfs_derivative[np.argmin(np.abs(alpha - self.alphas))]
        interp = -np.interp(bs, self.bs, I_func)
        final = interp * out
        return final


class VectorFieldTable:
    """
    DirichletConditionalFlow for k classes, covering the concentrations a_s = 1 + alpha(t_s), s = 0, ..., T - 1, of
    the sampling grid (alpha_max: the largest a_s plus one spacing, so that the nearest tabulated a has a finite
    difference).
    """

    def __init__(self, noiser, k, alpha_spacing=0.001, precision='float32'):
        """precision: of the computation of C. 'float32' as in the original (the state is float32 at inference),
        which underflows for small coordinates and large concentrations (C = inf, then NaN states); 'float64'
        avoids that underflow. Independent of the NumPy version."""
        assert precision in ('float32', 'float64'), precision
        self.dtype = np.float32 if precision == 'float32' else np.float64
        if noiser.schedule == 'sparse':
            raise NotImplementedError("'vector_field' is derived for the paths Dir(1 + alpha z_1) (schedules 'cosine' "
                                      "and 'linear'), not for 'sparse'")
        t = torch.arange(noiser.T, dtype=torch.float64) / noiser.T
        a = (1 + noiser.get_concentration(t)).numpy()
        self.flow = DirichletConditionalFlow(K=k, alpha_min=1., alpha_max=float(a.max()) + alpha_spacing,
                                             alpha_spacing=alpha_spacing)
        self.a = a
        self.n_nan = 0      # steps where C had NaNs (set to 0, as with allow_nan_cfactor in the original)
        self.n_proj = 0     # points projected back onto the simplex (vector_field_step)

    def __call__(self, z, s):
        """C(z, a_s), z: (...) tensor; computed in numpy, in the precision of the table, returned in the dtype of z."""
        c = self.flow.c_factor(z.cpu().numpy().astype(self.dtype, copy=False), float(self.a[s]))
        c = torch.from_numpy(np.asarray(c)).to(z)
        if torch.isnan(c).any():
            self.n_nan += 1
            print(f'NAN cfactor: z.min(): {z.min()}')
            c = torch.nan_to_num(c)
        return c


def simplex_proj(seq):
    """Euclidean projection of the last dimension onto the simplex, as utils/flow_utils.simplex_proj of Dirichlet flow
    matching (algorithm of https://arxiv.org/abs/1309.1541, Wang and Carreira-Perpinan). NaN rows stay NaN."""
    Y = seq.reshape(-1, seq.shape[-1])
    N, K = Y.shape
    X, _ = torch.sort(Y, dim=-1, descending=True)
    X_cumsum = torch.cumsum(X, dim=-1) - 1
    div_seq = torch.arange(1, K + 1, dtype=Y.dtype, device=Y.device)
    Xtmp = X_cumsum / div_seq.unsqueeze(0)
    greater_than_Xtmp = (X > Xtmp).sum(dim=1, keepdim=True)
    row_indices = torch.arange(N, dtype=torch.long, device=Y.device).unsqueeze(1)
    selected_Xtmp = Xtmp[row_indices, (greater_than_Xtmp - 1).clamp_min(0)]
    X = torch.max(Y - selected_Xtmp, torch.zeros_like(Y))
    return X.view(seq.shape)


def vector_field_step(z_t, p_z1, table, s):
    """
    Euler step of the Dirichlet flow from t_s to t_{s+1}, as the 'original' method of Dirichlet flow matching, in the
    concentration a:
        z_u = z_t + (a_{s+1} - a_s) sum_i p(z_1 = i | G_t) C(z_{t,i}, a_s) (e_i - z_t),
    then, as in the original, the points that left the simplex (a negative coordinate, or a sum off 1 by more than
    1e-4) are projected back onto it (simplex_proj). The original projects the whole tensor as soon as one point is
    off; here only those points are (the others are on the simplex already), and the padding rows (z_t = 0, masked
    by the sampler) are left out. table.n_proj counts the projected points.

    Args:
        z_t: (..., k) points. p_z1: (..., k) prediction. table: VectorFieldTable for k classes.
        s: index of t in the sampling grid (s + 1 < T).
    Returns:
        (..., k) next points, same dtype as z_t.
    """
    k = z_t.size(-1)
    eye = torch.eye(k, dtype=z_t.dtype, device=z_t.device)
    c_factor = table(z_t, s)
    cond_flows = (eye - z_t.unsqueeze(-1)) * c_factor.unsqueeze(-2)     # (..., k, k), column i: C(z_i) (e_i - z)
    flow = (p_z1.to(z_t.dtype).unsqueeze(-2) * cond_flows).sum(-1)
    z_u = z_t + flow * float(table.a[s + 1] - table.a[s])
    pad = (z_t == 0).all(-1)
    off = (((z_u.sum(-1) - 1).abs() > 1e-4) | (z_u < 0).any(-1)) & ~pad
    if off.any():
        table.n_proj += int(off.sum())
        z_u = torch.where(off.unsqueeze(-1), simplex_proj(z_u), z_u)
    return z_u


def _log(x):
    return x.clamp_min(1e-300).log()


def quantile_maps(z_t, table, s):
    """
    The per-class maps of the Dirichlet path from t_s to t_{s+1} (the 'cdf_trick' of Dirichlet flow matching).
    Conditional on the clean class i, z_t ~ Dir(b_s + (a_s - b_s) e_i): z_i ~ Beta(a_s, (k - 1) b_s) and the other
    coordinates, renormalized, y = z_{-i} / (1 - z_i) ~ Dir(b_s, ..., b_s), independent of z_i. So
        T_i(z)_i = F_{s+1}^{-1}(F_s(z_i)),    T_i(z)_{-i} = (1 - T_i(z)_i) S(y),
    maps Dir(b_s + (a_s - b_s) e_i) exactly onto Dir(b_{s+1} + (a_{s+1} - b_{s+1}) e_i), where S maps
    Dir(b_s, ..., b_s) onto Dir(b_{s+1}, ..., b_{s+1}): the identity when b = 1 ('cosine' / 'linear'), otherwise
    ('sparse') the quantile maps of the stick-breaking fractions of y, in the order of the classes (_stick_map).
    Computed in logit space, with 1 - z_i as the sum of the other coordinates, so that coordinates close to 0 or 1
    keep their precision.

    Args:
        z_t: (..., k) points on the simplex. table: BetaQuantileTable for k classes. s: index of t in the sampling
            grid (s + 1 < T).
    Returns:
        (..., k_i, k_j) float64, row i is T_i(z_t).
    """
    z = z_t.double().clamp_min(0)
    k = z.size(-1)
    maps = z.new_zeros(z.shape + (k,))
    for i in range(k):
        others = [j for j in range(k) if j != i]
        rest = z[..., others].sum(-1, keepdim=True)                 # 1 - z_i, summed directly (no cancellation)
        logit_next = table.transport(_log(z[..., i:i + 1]) - _log(rest), s)   # logit of T_i(z)_i
        y = z[..., others] / rest.clamp_min(1e-300)
        if table.sticks:
            y = _stick_map(y, table, s)
        maps[..., i, others] = torch.sigmoid(-logit_next) * y
        maps[..., i, i] = torch.sigmoid(logit_next[..., 0])
    return maps


def _stick_map(y, table, s):
    """
    Maps y ~ Dir(b_s, ..., b_s) (..., m) onto Dir(b_{s+1}, ..., b_{s+1}) through its stick-breaking fractions
    v_l = y_l / (y_l + ... + y_m) ~ Beta(b, (m - l) b), independent: each is moved by its quantile map (family m - l of
    the BetaQuantileTable, in logit space: logit v_l = log y_l - log(y_{l+1} + ... + y_m)), the last coordinate takes
    the rest.
    """
    m = y.size(-1)
    suffix = y.flip(-1).cumsum(-1).flip(-1)                          # suffix[l] = y_l + ... + y_m
    out = torch.zeros_like(y)
    rest_next = torch.ones_like(y[..., 0])
    for l in range(m - 1):
        logit_next = table.transport(_log(y[..., l]) - _log(suffix[..., l + 1]), s, m - 1 - l)
        out[..., l] = rest_next * torch.sigmoid(logit_next)
        rest_next = rest_next * torch.sigmoid(-logit_next)
    out[..., m - 1] = rest_next
    return out


def quantile_map(z_t, p_z1, table, s):
    """
    Deterministic step of the Dirichlet path from t_s to t_{s+1}: the per-class maps T_i (quantile_maps) averaged
    over the prediction, z_u = sum_i p(z_1 = i | G_t) T_i(z_t).

    Args:
        z_t: (..., k) points on the simplex. p_z1: (..., k) prediction. table: BetaQuantileTable for k classes.
        s: index of t in the sampling grid (s + 1 < T).
    Returns:
        (..., k) next points on the simplex, same dtype as z_t.
    """
    maps = quantile_maps(z_t, table, s)
    z_u = (p_z1.double().unsqueeze(-1) * maps).sum(-2)
    z_u = z_u / z_u.sum(-1, keepdim=True).clamp_min(1e-300)
    return z_u.to(z_t.dtype)


def partial_resampling(z, conc, rho):
    """
    Markov kernel M_rho on the simplex that leaves Dir(conc) invariant (Dirichlet flow matching's partial resampling):
        G = S z with S ~ Gamma(sum conc), so that G_j ~ Gamma(conc_j) independently (if z ~ Dir(conc)),
        G'_j = B_j G_j + E_j with B_j ~ Beta(rho conc_j, (1 - rho) conc_j) and E_j ~ Gamma((1 - rho) conc_j),
        z' = G' / sum G'.
    rho = 1 keeps z, rho = 0 draws a fresh z' ~ Dir(conc), in between keeps part of z.
    z, conc: (..., k), conc broadcastable. rho: float in [0, 1].
    """
    if rho >= 1:
        return z
    conc = conc.expand_as(z)
    if rho <= 0:
        return Dirichlet(conc).sample()
    S = Gamma(conc.sum(-1, keepdim=True), torch.ones_like(z[..., :1])).sample()
    B = Beta(rho * conc, (1 - rho) * conc).sample()
    E = Gamma((1 - rho) * conc, torch.ones_like(conc)).sample()
    G = B * S * z + E
    return G / G.sum(-1, keepdim=True).clamp_min(1e-300)


def tuned_step(z_t, p_z1, table, s, lambd, eta):
    """
    Tuned step of the Dirichlet path from t_s to t_{s+1} ('tuned' of Dirichlet flow matching):
      1. the per-class quantile maps T_i(z_t) (quantile_maps), each exactly distributed as the path conditional on
         z_1 = i at t_{s+1}, Dir(b + (a - b) e_i), when z_t is distributed as it at t_s;
      2. partial resampling of each T_i(z_t) by M_rho targeting that Dirichlet, rho = 1 - eta
         (partial_resampling): eta = 0 keeps the deterministic map, eta = 1 draws a fresh conditional sample;
      3. combination of the k results with weights w:
            lambd < 0 (lambda = infinity): w = p(z_1 | G_t), the average of the maps (with eta = 0: quantile_map),
            lambd = 0: w = e_i with i ~ p(z_1 | G_t), one class,
            lambd > 0: w ~ Dir(lambd p(z_1 | G_t)), from one class (lambd -> 0) to the average (lambd -> infinity).
    Args:
        z_t: (..., k) points on the simplex. p_z1: (..., k) prediction. table: BetaQuantileTable for k classes.
        s: index of t in the sampling grid (s + 1 < T). lambd: float. eta: float in [0, 1].
    Returns:
        (..., k) next points on the simplex, same dtype as z_t.
    """
    maps = quantile_maps(z_t, table, s)                                      # (..., k_i, k_j)
    k = maps.size(-1)
    # row i: the target Dir(b + (a - b) e_i) at t_{s+1}
    conc = table.b[s + 1] + (table.a[s + 1] - table.b[s + 1]) * torch.eye(k, dtype=maps.dtype, device=maps.device)
    conc = conc.clamp_min(1e-20)
    maps = partial_resampling(maps, conc, 1. - eta)
    p = p_z1.double()
    if lambd < 0:
        w = p
    elif lambd == 0:
        w = F.one_hot(Categorical(probs=p.clamp_min(0)).sample(), k).to(p.dtype)
    else:
        g = Gamma((lambd * p).clamp_min(1e-24), torch.ones_like(p)).sample()
        total = g.sum(-1, keepdim=True)
        # all the Gamma draws can underflow to 0 for a very peaked p: fall back to the average
        w = torch.where(total > 0, g / total.clamp_min(1e-300), p)
    z_u = (w.unsqueeze(-1) * maps).sum(-2)
    z_u = z_u / z_u.sum(-1, keepdim=True).clamp_min(1e-300)
    return z_u.to(z_t.dtype)


def expand_noise_level(alpha_bar, z):
    """Noise level, scalar or (bs,), broadcastable against z (bs, ...)."""
    alpha_bar = torch.as_tensor(alpha_bar, dtype=z.dtype, device=z.device)
    if alpha_bar.ndim == 1:
        alpha_bar = alpha_bar.view(-1, *[1] * (z.ndim - 1))
    return alpha_bar


def convert_parametrization(noiser, logits, z_t, alpha_bar, kind, source, target):
    """p(z_1 | G_t) ∝ c(z_1) q(z_t | z_1), with log q(z_t | z_1) = noiser.log_likelihood (see
    NoiserDense.convert_parametrization)."""
    assert source in PARAMETRIZATIONS and target in PARAMETRIZATIONS, (source, target)
    if source == target:
        return logits
    log_q = noiser.log_likelihood(z_t, alpha_bar, kind)
    if logits.ndim == z_t.ndim + 1:  # mode dimension
        log_q = log_q.unsqueeze(-2)
    sign = 1. if source == 'cavity' else -1.
    return (logits.log_softmax(-1) + sign * log_q).log_softmax(-1)


def sufficient_stat(noiser, z_t, alpha_bar, kind, mask):
    """
    phi(z_t, t)_b ∝ q(z_t | z_1 = b), normalized over the data classes b (Chandra et al., 2025). The elements are
    noised independently, so p(G_1 | G_t) depends on G_t only through the phi of its elements, whatever the noising
    process and the time: a denoiser that takes phi as input can be used with any of them.
        'marginal':  phi = r e_c + (1 - r) u, with c the observed class, u uniform and
                     r = alpha_bar / (alpha_bar + k (1 - alpha_bar) m_c),
        'masking':   e_c if z_t = c is unmasked, uniform if masked,
        'dirichlet': softmax(alpha(t) log z_t).
    z_t: (bs, ..., k') noisy elements, mask: (bs, ...) existing elements. Returns (bs, ..., k), zeros on padding.
    """
    return noiser.log_likelihood(z_t, alpha_bar, kind).softmax(-1) * mask.unsqueeze(-1)


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

"""
Graph generation with a trained denoiser.

Three sampling methods are available (config `sampling.id`). They share the noise schedule of the training,
alpha_bar(t) = noiser.get_alpha_bar(t), and the time grid t_k = k / T, k = 0, ..., T, from the prior (t = 0)
to the data (t = 1). Each step k goes from t = t_k to the next time u = t_{k+1}:
    - 'sid': simple iterative denoising. The denoiser predicts a clean graph G_1 from G_t, and the prediction
             is noised again to the next (lower) noise level u.
    - 'dif': ancestral sampling of the discrete diffusion (DiGress): G_u ~ sum_{G_1} q(G_u | G_t, G_1) p(G_1 | G_t).
             sampling_method 'denoiser' computes this sum with the denoiser p(z_1 | G_t); 'cavity' plugs the cavity
             c(z_1) = p(z_1 | G_t without z_t) into q(z_u | z_t, z_1), q(z_t | z_u) sum_{z_1} c(z_1) q(z_u | z_1),
             the same distribution when c is the exact cavity.
    - 'dfm': discrete flow matching, Euler steps along the mixture path between the prior and the data.

Sampling temperature tau (all samplers): the prediction p(z_1 | G_t) is replaced by p(z_1 | G_t)^(1 / tau),
renormalized per node / edge, i.e. its logits are divided by tau (tau < 1 sharper, tau > 1 flatter, tau = 1 the
model). It is applied to the denoiser p(z_1 | G_t); the 'cavity' used by 'dif' is derived from the tempered denoiser.

In all three, the last step (u = 1, alpha_bar(u) = 1) draws each element from the prediction p(z_1 | G_t). With
greedy_last_step, it takes the most likely class instead (argmax of p(z_1 | G_t), edges on the upper triangle and
mirrored).

The denoiser outputs the denoiser or the cavity (training_method); it is converted to the parametrization each
sampler uses when they differ (NoiserDense.convert_parametrization): 'sid', 'dfm' and 'quantile_map' use the denoiser,
'dif' the sampling_method. Its input is the noisy state or the sufficient statistics of G_t (utils/model_input.py).

With the 'dirichlet' prior (NoiserDirichlet), G_t is a point on the simplex for each element. Two samplers:
    - 'sid': predict G_1 from G_t, sample it, and noise it again with the Dirichlet path at u.
    - 'quantile_map': deterministic transport along the Dirichlet path (the 'cdf_trick' of Dirichlet flow matching,
             see utils.noising.quantile_map): z_u = sum_i p(z_1 = i | G_t) T_i(z_t), where T_i maps
             Dir(1 + alpha(t) e_i) onto Dir(1 + alpha(u) e_i) through the quantiles of the Beta marginal of z_i.
             Only the initial noise and the last step (a draw from the prediction, or its argmax with
             greedy_last_step) are random.
    - 'tuned': the 'tuned' sampler of Dirichlet flow matching (see utils.noising.tuned_step). Each map T_i(z_t) is
             partially resampled towards Dir(1 + alpha(u) e_i) by an amount eta (0: the deterministic map, 1: a fresh
             conditional sample), and the k results are combined with weights set by lambd (< 0: the average weighted
             by p(z_1 | G_t), 0: one class drawn from p(z_1 | G_t), > 0: Dirichlet(lambd p(z_1 | G_t)) weights).
             eta = 0 and lambd < 0 is 'quantile_map'.
    - 'vector_field': the 'original' sampler of Dirichlet flow matching (see utils.noising.vector_field_step), Euler
             steps of the marginal vector field sum_i p(z_1 = i | G_t) C(z_i, a) (e_i - z) in the concentration
             a = 1 + alpha(t), with the exact C factor, projected onto the simplex when a step leaves it.

See utils/noising.py for the dense graph representation and the priors.
"""
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from utils.func import get_edge_mask
from utils.modes import select_mode, to_class_logits
from utils.noising import (MarginalUniformTransition, PARAMETRIZATIONS, BetaQuantileTable, quantile_map,
                           tuned_step, VectorFieldTable, vector_field_step, sample_nodes, sample_edges, symmetrize_upper)


class Sampler:
    def __init__(self, noiser, model_input, num_nodes_distribution, max_num_nodes, n_node_types,
                 batch_size, device, hyp_selection='sample', training_method='denoiser',
                 sampling_method='denoiser', greedy_last_step=False, n_edge_types=None, temperature=1.,
                 lambd=-1., eta=1.):
        """
        Args:
            noiser: NoiserDense, defines the prior, the noise schedule and the number of steps T.
            model_input: ModelInput, builds the denoiser input from G_t (as during training).
            num_nodes_distribution: (max_num_nodes + 1,) distribution of the number of nodes.
            n_node_types: number of node types (1 for graphs without node types).
            batch_size: number of graphs generated in parallel.
            hyp_selection: multi-mode denoiser only, 'sample' one mode per graph at each step or use the
                'average' of the mixture (see utils/modes.py).
            training_method: what the denoiser outputs, 'denoiser' or 'cavity'.
            sampling_method: the parametrization used by 'dif', 'denoiser' or 'cavity'.
            greedy_last_step: at the last step, take the argmax of the prediction instead of sampling it.
            n_edge_types: number of edge types including 'no edge' (needed by the 'dirichlet' prior).
            temperature: sampling temperature tau > 0, p(z_1 | G_t) -> p(z_1 | G_t)^(1 / tau) renormalized.
            lambd, eta: 'tuned' sampler ('dirichlet' prior), see utils.noising.tuned_step. lambd: combination of the
                per-class maps (< 0: average weighted by the prediction, 0: one class drawn from it, > 0: Dirichlet
                weights); eta in [0, 1]: amount of resampling of each map (0: deterministic, 1: full).
        """
        assert 0. <= eta <= 1., f'eta must be in [0, 1], got {eta}'
        self.lambd = lambd
        self.eta = eta
        assert training_method in PARAMETRIZATIONS and sampling_method in PARAMETRIZATIONS
        self.training_method = training_method
        self.sampling_method = sampling_method
        self.noiser = noiser
        self.T = noiser.T
        self.model_input = model_input
        self.num_nodes_distribution = num_nodes_distribution.to(device)
        self.max_num_nodes = max_num_nodes
        self.has_node_types = n_node_types > 1
        self.batch_size = batch_size
        self.device = device
        self.hyp_selection = hyp_selection
        self.greedy_last_step = greedy_last_step
        assert temperature > 0, f'The sampling temperature must be positive, got {temperature}'
        self.temperature = temperature

        self.n_node_types = n_node_types
        self.n_edge_types = n_edge_types
        self.dirichlet = noiser.prior == 'dirichlet'
        if self.dirichlet:
            assert n_edge_types is not None, "The 'dirichlet' prior needs n_edge_types"
            self.node_transitions = self.edge_transitions = None
            self.node_table = self.edge_table = None    # built on the first 'quantile_map' / 'tuned' call
            self.node_field = self.edge_field = None    # built on the first 'vector_field' call
        else:
            self.node_transitions = MarginalUniformTransition(noiser.node_prior.to(device))
            self.edge_transitions = MarginalUniformTransition(noiser.edge_prior.to(device))
        self.model = None

    @torch.no_grad()
    def __call__(self, n_samples, model, method='sid'):
        """
        Generates `n_samples` graphs with `model`.

        Returns:
            x: (n_samples, n) node types, a: (n_samples, n, n) edge types (0 = no edge),
            node_mask: (n_samples, n), adj_mask: (n_samples, n, n).
        """
        samplers = {'sid': self.sample_sid, 'dif': self.sample_dif, 'dfm': self.sample_dfm,
                    'quantile_map': self.sample_quantile_map, 'tuned': self.sample_tuned,
                    'vector_field': self.sample_vector_field}
        if method not in samplers:
            raise NotImplementedError(f'Sampling method {method} not implemented. Available: {list(samplers)}')
        available = ('sid', 'quantile_map', 'tuned', 'vector_field') if self.dirichlet else ('sid', 'dif', 'dfm')
        if method not in available:
            raise NotImplementedError(f"Sampling method {method} is not available with the '{self.noiser.prior}' "
                                      f"prior. Available: {available}")

        was_training = model.training
        model.eval()
        self.model = model

        xs, as_, node_masks = [], [], []
        for start in range(0, n_samples, self.batch_size):
            print(f'Sampling graphs {start} to {min(start + self.batch_size, n_samples)} with {method}...')
            node_mask = self.sample_node_mask(min(self.batch_size, n_samples - start))
            x, a = samplers[method](node_mask)
            xs.append(x)
            as_.append(a)
            node_masks.append(node_mask)
        model.train(was_training)

        x, a, node_mask = torch.cat(xs), torch.cat(as_), torch.cat(node_masks)
        adj_mask = get_edge_mask(node_mask)
        return x.argmax(-1) * node_mask, a.argmax(-1) * adj_mask, node_mask, adj_mask

    # ----------------------------------------------------------------------------------------------------------------
    # Sampling methods. Each one takes the node mask of a batch and returns the generated graph as one-hot tensors
    # over the data classes.
    # ----------------------------------------------------------------------------------------------------------------

    def sample_sid(self, node_mask):
        """Simple iterative denoising: predict G_1 from G_t, then noise the prediction to the next time u."""
        adj_mask = get_edge_mask(node_mask)
        x_t, a_t = self.sample_prior(node_mask)

        for k in range(self.T):
            alpha_bar_t = self.noiser.get_alpha_bar(k / self.T)
            p_x1, p_a1 = self.predict_clean_graph(x_t, a_t, node_mask, alpha_bar_t)
            last = k == self.T - 1
            x_1 = self.draw_nodes(p_x1, node_mask, last) if self.has_node_types else x_t
            a_1 = self.draw_edges(p_a1, adj_mask, last)

            if self.noiser.prior == 'masking':
                # Elements that are not masked in G_t are known to be clean: keep them
                if self.has_node_types:
                    x_1 = keep_unmasked(x_t, x_1)
                a_1 = keep_unmasked(a_t, a_1)

            if last:
                return x_1, a_1
            alpha_bar_u = self.noiser.get_alpha_bar((k + 1) / self.T)
            x_t, a_t, _ = self.noiser(x_1, a_1, node_mask, alpha_bar_u)

    def sample_dif(self, node_mask):
        """Ancestral sampling of the discrete diffusion: G_u ~ sum_{G_1} q(G_u | G_t, G_1) p(G_1 | G_t)."""
        adj_mask = get_edge_mask(node_mask)
        x_t, a_t = self.sample_prior(node_mask)

        for k in range(self.T):
            last = k == self.T - 1
            alpha_bar_t = self.noiser.get_alpha_bar(k / self.T)
            alpha_bar_u = self.noiser.get_alpha_bar((k + 1) / self.T)

            p_x1, p_a1 = self.predict_clean_graph(x_t, a_t, node_mask, alpha_bar_t, target=self.sampling_method)
            posterior = diffusion_posterior if self.sampling_method == 'denoiser' else cavity_posterior

            p_au = posterior(a_t, self.noiser.to_noisy_classes(p_a1), self.edge_transitions,
                             alpha_bar_t, alpha_bar_u)
            if last:
                p_au = self.drop_mask_class(p_au)
            a_t = self.draw_edges(p_au, adj_mask, last)

            if self.has_node_types:
                p_xu = posterior(x_t, self.noiser.to_noisy_classes(p_x1), self.node_transitions,
                                 alpha_bar_t, alpha_bar_u)
                if last:
                    p_xu = self.drop_mask_class(p_xu)
                x_t = self.draw_nodes(p_xu, node_mask, last)

        x_1 = self.noiser.from_noisy_classes(x_t) if self.has_node_types else x_t
        return x_1, self.noiser.from_noisy_classes(a_t)

    def sample_quantile_map(self, node_mask):
        """
        Deterministic sampling of the Dirichlet path ('cdf_trick' of Dirichlet flow matching, see the module
        docstring): from t = t_k to u = t_{k+1}, z_u = sum_i p(z_1 = i | G_t) T_i(z_t), elementwise for the nodes and
        the edges (upper triangle, mirrored). The last step returns a draw from the prediction p(z_1 | G_t), or its
        argmax with greedy_last_step.
        """
        return self.sample_dirichlet_path(node_mask, quantile_map)

    def sample_tuned(self, node_mask):
        """
        'tuned' sampling of the Dirichlet path (Dirichlet flow matching, see the module docstring and
        utils.noising.tuned_step), with the parameters lambd and eta. The last step returns a draw from the prediction
        p(z_1 | G_t), or its argmax with greedy_last_step.
        """
        return self.sample_dirichlet_path(
            node_mask, lambda z_t, p_z1, table, k: tuned_step(z_t, p_z1, table, k, self.lambd, self.eta))

    def sample_vector_field(self, node_mask):
        """
        Euler integration of the vector field of the Dirichlet path ('original' method of Dirichlet flow matching,
        see the module docstring and utils.noising.vector_field_step). The last step returns a draw from the
        prediction p(z_1 | G_t), or its argmax with greedy_last_step.
        """
        if self.edge_field is None:
            if self.has_node_types:
                self.node_field = VectorFieldTable(self.noiser, self.n_node_types, device=self.device)
            self.edge_field = VectorFieldTable(self.noiser, self.n_edge_types, device=self.device)
        return self.sample_dirichlet_path(node_mask, vector_field_step, self.node_field, self.edge_field)

    def sample_dirichlet_path(self, node_mask, step, node_table=None, edge_table=None):
        """Steps z_u = step(z_t, p(z_1 | G_t), table, k) along the Dirichlet path, elementwise for the nodes and the
        edges (upper triangle, mirrored), then a draw (or the argmax) from the last prediction. The tables default to
        the quantile tables (BetaQuantileTable)."""
        if edge_table is None:   # quantile maps
            if self.edge_table is None:
                if self.has_node_types:
                    self.node_table = BetaQuantileTable(self.noiser, self.n_node_types, device=self.device)
                self.edge_table = BetaQuantileTable(self.noiser, self.n_edge_types, device=self.device)
            node_table, edge_table = self.node_table, self.edge_table
        adj_mask = get_edge_mask(node_mask)
        x_t, a_t = self.sample_prior(node_mask)

        for k in range(self.T):
            p_x1, p_a1 = self.predict_clean_graph(x_t, a_t, node_mask, self.noiser.get_alpha_bar(k / self.T))
            if k == self.T - 1:
                x_1 = self.draw_nodes(p_x1, node_mask, last=True) if self.has_node_types else x_t
                return x_1, self.draw_edges(p_a1, adj_mask, last=True)

            a_t = symmetrize_upper(step(a_t, p_a1, edge_table, k)) * adj_mask.unsqueeze(-1)
            if self.has_node_types:
                x_t = step(x_t, p_x1, node_table, k) * node_mask.unsqueeze(-1)

    def sample_dfm(self, node_mask):
        """
        Discrete flow matching along the mixture path of the noising process:
            p_t(z) = kappa(t) delta(z, z_1) + (1 - kappa(t)) prior(z),    kappa(t) = alpha_bar(t).
        Euler step from t to u, with the finite difference kappa(u) - kappa(t) in place of (u - t) kappa'(t):
            p(z_u) = z_t + (kappa(u) - kappa(t)) / (1 - kappa(t)) * (p(z_1 | z_t) - z_t).
        The step size is the probability that an element still on the prior jumps to the data in (t, u], in [0, 1].
        """
        if self.noiser.prior != 'marginal':
            raise NotImplementedError("'dfm' sampling is only implemented for the 'marginal' prior.")
        adj_mask = get_edge_mask(node_mask)
        x_t, a_t = self.sample_prior(node_mask)

        for k in range(self.T):
            last = k == self.T - 1
            alpha_bar_t = self.noiser.get_alpha_bar(k / self.T)
            alpha_bar_u = self.noiser.get_alpha_bar((k + 1) / self.T)
            p_x1, p_a1 = self.predict_clean_graph(x_t, a_t, node_mask, alpha_bar_t)

            step = ((alpha_bar_u - alpha_bar_t) / (1 - alpha_bar_t)).to(self.device)
            p_a = a_t + step * (p_a1 - a_t)
            a_t = self.draw_edges(p_a / p_a.sum(-1, keepdim=True), adj_mask, last)
            if self.has_node_types:
                p_x = x_t + step * (p_x1 - x_t)
                x_t = self.draw_nodes(p_x / p_x.sum(-1, keepdim=True), node_mask, last)

        return x_t, a_t

    # ----------------------------------------------------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------------------------------------------------

    def sample_node_mask(self, n_graphs):
        """Draws the number of nodes of each graph; the nodes of a graph are the first ones."""
        n_nodes = Categorical(probs=self.num_nodes_distribution).sample((n_graphs,))
        return torch.arange(self.max_num_nodes, device=self.device).unsqueeze(0) < n_nodes.unsqueeze(1)

    def sample_prior(self, node_mask):
        """G_0: every node and edge is drawn from the prior (nodes without types are left at 1)."""
        if self.dirichlet:
            return self.noiser.sample_prior(node_mask, self.n_node_types, self.n_edge_types)
        bs, n = node_mask.shape
        adj_mask = get_edge_mask(node_mask)
        if self.has_node_types:
            x = sample_nodes(self.node_transitions.prior.expand(bs, n, -1), node_mask)
        else:
            x = node_mask.unsqueeze(-1).float()
        a = sample_edges(self.edge_transitions.prior.expand(bs, n, n, -1), adj_mask)
        return x, a

    def predict_clean_graph(self, x_t, a_t, node_mask, alpha_bar, target='denoiser'):
        """
        Prediction of G_1 from G_t in the target parametrization, 'denoiser' p(z_1 | G_t) or 'cavity'
        p(z_1 | G_t without z_t), as probabilities over the data classes. The output of the model
        (training_method) is converted when it differs. With a temperature, the denoiser is tempered first and the
        target derived from it.
        """
        x_in, a_in = self.model_input(x_t, a_t, node_mask, alpha_bar)
        out = self.model(x_in, a_in, node_mask)
        source = self.training_method
        if len(out) == 3:  # multi-mode denoiser
            mode_logits, x_logits, a_logits = out
            if self.hyp_selection == 'average':
                # The mixture is an average of the modes' denoisers: convert each mode before averaging
                x_logits, a_logits = self.convert(x_logits, to_class_logits(a_logits), x_t, a_t, alpha_bar,
                                                  source, 'denoiser')
                source = 'denoiser'
            p_x, p_a = select_mode((mode_logits, x_logits, a_logits), self.hyp_selection)
            x_logits, a_logits = p_x.clamp_min(1e-30).log(), p_a.clamp_min(1e-30).log()
        else:
            # Binary edges: the single output is the logit of 'no edge' (class 0)
            x_logits, a_logits = out
            a_logits = to_class_logits(a_logits)
        if self.temperature != 1.:
            x_logits, a_logits = self.convert(x_logits, a_logits, x_t, a_t, alpha_bar, source, 'denoiser')
            x_logits, a_logits = x_logits / self.temperature, a_logits / self.temperature
            source = 'denoiser'
        x_logits, a_logits = self.convert(x_logits, a_logits, x_t, a_t, alpha_bar, source, target)
        return x_logits.softmax(-1), a_logits.softmax(-1)

    def convert(self, x_logits, a_logits, x_t, a_t, alpha_bar, source, target):
        """Node and edge logits from the source to the target parametrization (nodes only if they have types)."""
        if self.has_node_types:
            x_logits = self.noiser.convert_parametrization(x_logits, x_t, alpha_bar, 'node', source, target)
        a_logits = self.noiser.convert_parametrization(a_logits, a_t, alpha_bar, 'edge', source, target)
        return x_logits, a_logits

    def draw_nodes(self, probs, node_mask, last):
        """Samples the nodes, or takes their most likely class at the last step with greedy_last_step."""
        if last and self.greedy_last_step:
            return argmax_nodes(probs, node_mask)
        return sample_nodes(probs, node_mask)

    def draw_edges(self, probs, adj_mask, last):
        """Samples the edges, or takes their most likely class at the last step with greedy_last_step."""
        if last and self.greedy_last_step:
            return argmax_edges(probs, adj_mask)
        return sample_edges(probs, adj_mask)

    def drop_mask_class(self, p):
        """At the last step nothing may stay masked: remove the mask class and renormalize."""
        if self.noiser.prior != 'masking':
            return p
        p = torch.cat((p[..., :-1], torch.zeros_like(p[..., -1:])), dim=-1)
        return p / p.sum(-1, keepdim=True)


def argmax_nodes(probs, node_mask):
    """One-hot most likely class of `probs` (bs, n, k) for the existing nodes, zeros elsewhere."""
    return F.one_hot(probs.argmax(-1), probs.size(-1)).float() * node_mask.unsqueeze(-1)


def argmax_edges(probs, adj_mask):
    """Symmetric one-hot most likely class of `probs` (bs, n, n, k): the upper triangle is decoded and mirrored."""
    idx = probs.argmax(-1).triu(1)
    idx = idx + idx.transpose(1, 2)
    return F.one_hot(idx, probs.size(-1)).float() * adj_mask.unsqueeze(-1)


def keep_unmasked(z_t, z_1):
    """Masking prior: returns z_t where it is not masked (last class) and z_1 elsewhere, over the noisy classes."""
    is_masked = z_t[..., -1:].bool()
    return torch.where(is_masked, z_1, z_t[..., :-1])


def cavity_posterior(z_t, c_z1, transitions, alpha_bar_t, alpha_bar_u):
    """
    p(z_u | z_t) ∝ q(z_t | z_u) sum_{z_1} c(z_1) q(z_u | z_1): the plug-in posterior q(z_u | z_t, z_1 = c) with the
    cavity c(z_1) = p(z_1 | G_t without z_t). Equal to diffusion_posterior with the denoiser
    p(z_1 | G_t) ∝ c(z_1) q(z_t | z_1), since q(z_u | z_t, z_1) = q(z_t | z_u) q(z_u | z_1) / q(z_t | z_1).

    Args:
        z_t: (..., k) one-hot noisy nodes or edges.
        c_z1: (..., k) cavity, over the noisy classes.
    Returns:
        (..., k) distribution of z_u.
    """
    Q_tu = transitions.get_Q(alpha_bar_t / alpha_bar_u)   # Q_tu[i, j] = q(z_t = j | z_u = i)
    Q_u = transitions.get_Q(alpha_bar_u)                  # Q_u[i, j] = q(z_u = j | z_1 = i)
    p_zu = (z_t @ Q_tu.T) * (c_z1 @ Q_u)                  # (..., k_u)
    p_zu[p_zu.sum(-1) == 0] = 1e-5
    return p_zu / p_zu.sum(-1, keepdim=True)


def diffusion_posterior(z_t, p_z1, transitions, alpha_bar_t, alpha_bar_u):
    """
    p(z_u | z_t) = sum_{z_1} q(z_u | z_t, z_1) p(z_1 | z_t), for the next time u > t, with
    q(z_u | z_t, z_1) = q(z_t | z_u) q(z_u | z_1) / q(z_t | z_1).

    Args:
        z_t: (..., k) one-hot noisy nodes or edges.
        p_z1: (..., k) predicted clean distribution, over the noisy classes.
    Returns:
        (..., k) distribution of z_u.
    """
    Q_tu = transitions.get_Q(alpha_bar_t / alpha_bar_u)   # Q_tu[i, j] = q(z_t = j | z_u = i)
    Q_u = transitions.get_Q(alpha_bar_u)                  # Q_u[i, j] = q(z_u = j | z_1 = i)
    Q_t = transitions.get_Q(alpha_bar_t)                  # Q_t[i, j] = q(z_t = j | z_1 = i)

    q_zt_given_zu = (z_t @ Q_tu.T).unsqueeze(-2)        # (..., 1, k_u)
    q_zt_given_z1 = (z_t @ Q_t.T).unsqueeze(-1)         # (..., k_1, 1)
    q_zt_given_z1 = q_zt_given_z1.masked_fill(q_zt_given_z1 == 0, 1e-6)
    q_zu_given_zt_z1 = q_zt_given_zu * Q_u / q_zt_given_z1    # (..., k_1, k_u)

    p_zu = (p_z1.unsqueeze(-1) * q_zu_given_zt_z1).sum(-2)     # (..., k_u)
    p_zu[p_zu.sum(-1) == 0] = 1e-5
    return p_zu / p_zu.sum(-1, keepdim=True)

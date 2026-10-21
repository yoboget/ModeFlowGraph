"""
Graph generation with a trained denoiser.

Three sampling methods are available (config `sampling.id`). They share the noise schedule of the training,
alpha_bar(t) = noiser.get_alpha_bar(t), and the time grid t_k = k / T, k = 0, ..., T, from the prior (t = 0)
to the data (t = 1). Each step k goes from t = t_k to the next time u = t_{k+1}:
    - 'sid': simple iterative denoising. The denoiser predicts a clean graph G_1 from G_t, and the prediction
             is noised again to the next (lower) noise level u.
    - 'dif': ancestral sampling of the discrete diffusion (DiGress): G_u ~ sum_{G_1} q(G_u | G_t, G_1) p(G_1 | G_t).
    - 'dfm': discrete flow matching, Euler steps along the mixture path between the prior and the data.

See utils/noising.py for the dense graph representation and the priors.
"""
import torch
from torch.distributions.categorical import Categorical

from utils.func import get_edge_mask
from utils.modes import select_mode
from utils.noising import MarginalUniformTransition, sample_nodes, sample_edges


class Sampler:
    def __init__(self, noiser, extra_features, num_nodes_distribution, max_num_nodes, n_node_types,
                 batch_size, device, hyp_selection='sample'):
        """
        Args:
            noiser: NoiserDense, defines the prior, the noise schedule and the number of steps T.
            extra_features: ExtraFeatures, appended to the denoiser input (as during training).
            num_nodes_distribution: (max_num_nodes + 1,) distribution of the number of nodes.
            n_node_types: number of node types (1 for graphs without node types).
            batch_size: number of graphs generated in parallel.
            hyp_selection: multi-mode denoiser only, 'sample' one mode per graph at each step or use the
                'average' of the mixture (see utils/modes.py).
        """
        self.noiser = noiser
        self.T = noiser.T
        self.extra_features = extra_features
        self.num_nodes_distribution = num_nodes_distribution.to(device)
        self.max_num_nodes = max_num_nodes
        self.has_node_types = n_node_types > 1
        self.batch_size = batch_size
        self.device = device
        self.hyp_selection = hyp_selection

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
        samplers = {'sid': self.sample_sid, 'dif': self.sample_dif, 'dfm': self.sample_dfm}
        if method not in samplers:
            raise NotImplementedError(f'Sampling method {method} not implemented. Available: {list(samplers)}')

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
            x_1 = sample_nodes(p_x1, node_mask) if self.has_node_types else x_t
            a_1 = sample_edges(p_a1, adj_mask)

            if self.noiser.prior == 'masking':
                # Elements that are not masked in G_t are known to be clean: keep them
                if self.has_node_types:
                    x_1 = keep_unmasked(x_t, x_1)
                a_1 = keep_unmasked(a_t, a_1)

            if k == self.T - 1:
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

            p_x1, p_a1 = self.predict_clean_graph(x_t, a_t, node_mask, alpha_bar_t)

            p_au = diffusion_posterior(a_t, self.noiser.to_noisy_classes(p_a1), self.edge_transitions,
                                       alpha_bar_t, alpha_bar_u)
            if last:
                p_au = self.drop_mask_class(p_au)
            a_t = sample_edges(p_au, adj_mask)

            if self.has_node_types:
                p_xu = diffusion_posterior(x_t, self.noiser.to_noisy_classes(p_x1), self.node_transitions,
                                           alpha_bar_t, alpha_bar_u)
                if last:
                    p_xu = self.drop_mask_class(p_xu)
                x_t = sample_nodes(p_xu, node_mask)

        x_1 = self.noiser.from_noisy_classes(x_t) if self.has_node_types else x_t
        return x_1, self.noiser.from_noisy_classes(a_t)

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
            alpha_bar_t = self.noiser.get_alpha_bar(k / self.T)
            alpha_bar_u = self.noiser.get_alpha_bar((k + 1) / self.T)
            p_x1, p_a1 = self.predict_clean_graph(x_t, a_t, node_mask, alpha_bar_t)

            step = ((alpha_bar_u - alpha_bar_t) / (1 - alpha_bar_t)).to(self.device)
            p_a = a_t + step * (p_a1 - a_t)
            a_t = sample_edges(p_a / p_a.sum(-1, keepdim=True), adj_mask)
            if self.has_node_types:
                p_x = x_t + step * (p_x1 - x_t)
                x_t = sample_nodes(p_x / p_x.sum(-1, keepdim=True), node_mask)

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
        bs, n = node_mask.shape
        adj_mask = get_edge_mask(node_mask)
        if self.has_node_types:
            x = sample_nodes(self.node_transitions.prior.expand(bs, n, -1), node_mask)
        else:
            x = node_mask.unsqueeze(-1).float()
        a = sample_edges(self.edge_transitions.prior.expand(bs, n, n, -1), adj_mask)
        return x, a

    def predict_clean_graph(self, x_t, a_t, node_mask, alpha_bar):
        """Denoiser prediction p(G_1 | G_t), as probabilities over the data classes."""
        x_in = self.noiser.add_noise_level(x_t, alpha_bar)
        x_in, a_in = self.extra_features(x_in, a_t, node_mask)
        out = self.model(x_in, a_in, node_mask)
        if len(out) == 3:  # multi-mode denoiser
            return select_mode(out, self.hyp_selection)
        x_logits, a_logits = out

        p_x = x_logits.softmax(-1)
        if a_logits.size(-1) == 1:
            # Binary edges: the single output is the logit of 'no edge' (class 0)
            p_no_edge = a_logits.sigmoid()
            p_a = torch.cat((p_no_edge, 1 - p_no_edge), dim=-1)
        else:
            p_a = a_logits.softmax(-1)
        return p_x, p_a

    def drop_mask_class(self, p):
        """At the last step nothing may stay masked: remove the mask class and renormalize."""
        if self.noiser.prior != 'masking':
            return p
        p = torch.cat((p[..., :-1], torch.zeros_like(p[..., -1:])), dim=-1)
        return p / p.sum(-1, keepdim=True)


def keep_unmasked(z_t, z_1):
    """Masking prior: returns z_t where it is not masked (last class) and z_1 elsewhere, over the noisy classes."""
    is_masked = z_t[..., -1:].bool()
    return torch.where(is_masked, z_1, z_t[..., :-1])


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

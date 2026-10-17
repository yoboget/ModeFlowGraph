"""
Graph generation with a trained denoiser.

Three sampling methods are available (config `sampling.id`), all starting from the prior and running T steps:
    - 'sid': simple iterative denoising. The denoiser predicts a clean graph G_0 from G_t, and the prediction
             is noised again to the next (lower) noise level.
    - 'dif': ancestral sampling of the discrete diffusion (DiGress): G_s ~ sum_{G_0} q(G_s | G_t, G_0) p(G_0 | G_t).
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
            noiser: NoiserDense, defines the prior, the noise schedules and the number of steps T.
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
        """Simple iterative denoising: predict G_0 from G_t, then noise the prediction to level t - 1."""
        adj_mask = get_edge_mask(node_mask)
        x_t, a_t = self.sample_prior(node_mask)

        for t in range(self.T, 0, -1):
            alpha_bar_t = self.noiser.get_alpha_bar(t / self.T)
            p_x0, p_a0 = self.predict_clean_graph(x_t, a_t, node_mask, alpha_bar_t)
            x_0 = sample_nodes(p_x0, node_mask) if self.has_node_types else x_t
            a_0 = sample_edges(p_a0, adj_mask)

            if self.noiser.prior == 'masking':
                # Elements that are not masked in G_t are known to be clean: keep them
                if self.has_node_types:
                    x_0 = keep_unmasked(x_t, x_0)
                a_0 = keep_unmasked(a_t, a_0)

            if t == 1:
                return x_0, a_0
            alpha_bar_s = self.noiser.get_alpha_bar((t - 1) / self.T)
            x_t, a_t, _ = self.noiser(x_0, a_0, node_mask, alpha_bar_s)

    def sample_dif(self, node_mask):
        """Ancestral sampling of the discrete diffusion: G_{t-1} ~ sum_{G_0} q(G_{t-1} | G_t, G_0) p(G_0 | G_t)."""
        adj_mask = get_edge_mask(node_mask)
        schedule = self.noiser.schedule
        x_t, a_t = self.sample_prior(node_mask)

        for t in range(self.T, 0, -1):
            s = t - 1
            beta_t = schedule.betas[t]
            alpha_bar_t, alpha_bar_s = schedule.alphas_bar[t], schedule.alphas_bar[s]

            p_x0, p_a0 = self.predict_clean_graph(x_t, a_t, node_mask, alpha_bar_t)

            p_as = diffusion_posterior(a_t, self.noiser.to_noisy_classes(p_a0), self.edge_transitions,
                                       beta_t, alpha_bar_t, alpha_bar_s)
            if s == 0:
                p_as = self.drop_mask_class(p_as)
            a_t = sample_edges(p_as, adj_mask)

            if self.has_node_types:
                p_xs = diffusion_posterior(x_t, self.noiser.to_noisy_classes(p_x0), self.node_transitions,
                                           beta_t, alpha_bar_t, alpha_bar_s)
                if s == 0:
                    p_xs = self.drop_mask_class(p_xs)
                x_t = sample_nodes(p_xs, node_mask)

        x_0 = self.noiser.from_noisy_classes(x_t) if self.has_node_types else x_t
        return x_0, self.noiser.from_noisy_classes(a_t)

    def sample_dfm(self, node_mask):
        """
        Discrete flow matching along the mixture path, from the prior (time 0) to the data (time 1):
            p_t(z) = kappa(t) delta(z, z_1) + (1 - kappa(t)) prior(z),    kappa(t) = 1 - alpha_bar(t).
        Euler step of size dt = 1 / T:
            p(z_{t+dt}) = z_t + dt * kappa'(t) / (1 - kappa(t)) * (p(z_1 | z_t) - z_t).
        """
        if self.noiser.prior != 'marginal':
            raise NotImplementedError("'dfm' sampling is only implemented for the 'marginal' prior.")
        adj_mask = get_edge_mask(node_mask)
        alphas_bar = self.noiser.schedule.alphas_bar
        x_t, a_t = self.sample_prior(node_mask)

        for k in range(self.T):  # flow time t = k / T
            # The denoiser is conditioned on the diffusion noise level, i.e. at diffusion time 1 - t
            p_x1, p_a1 = self.predict_clean_graph(x_t, a_t, node_mask, alphas_bar[self.T - k])

            # dt * kappa'(t) / (1 - kappa(t)), with a finite difference for kappa'
            step = ((alphas_bar[k] - alphas_bar[k + 1]) / alphas_bar[k]).to(self.device)
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
        """G_T: every node and edge is drawn from the prior (nodes without types are left at 1)."""
        bs, n = node_mask.shape
        adj_mask = get_edge_mask(node_mask)
        if self.has_node_types:
            x = sample_nodes(self.node_transitions.prior.expand(bs, n, -1), node_mask)
        else:
            x = node_mask.unsqueeze(-1).float()
        a = sample_edges(self.edge_transitions.prior.expand(bs, n, n, -1), adj_mask)
        return x, a

    def predict_clean_graph(self, x_t, a_t, node_mask, alpha_bar):
        """Denoiser prediction p(G_0 | G_t), as probabilities over the data classes."""
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


def keep_unmasked(z_t, z_0):
    """Masking prior: returns z_t where it is not masked (last class) and z_0 elsewhere, over the noisy classes."""
    is_masked = z_t[..., -1:].bool()
    return torch.where(is_masked, z_0, z_t[..., :-1])


def diffusion_posterior(z_t, p_z0, transitions, beta_t, alpha_bar_t, alpha_bar_s):
    """
    p(z_s | z_t) = sum_{z_0} q(z_s | z_t, z_0) p(z_0 | z_t), for s = t - 1, with
    q(z_s | z_t, z_0) = q(z_t | z_s) q(z_s | z_0) / q(z_t | z_0).

    Args:
        z_t: (..., k) one-hot noisy nodes or edges.
        p_z0: (..., k) predicted clean distribution, over the noisy classes.
    Returns:
        (..., k) distribution of z_s.
    """
    Qt = transitions.get_Qt(beta_t)                # Qt[i, j] = q(z_t = j | z_s = i)
    Qs_bar = transitions.get_Qt_bar(alpha_bar_s)   # Qs_bar[i, j] = q(z_s = j | z_0 = i)
    Qt_bar = transitions.get_Qt_bar(alpha_bar_t)   # Qt_bar[i, j] = q(z_t = j | z_0 = i)

    q_zt_given_zs = (z_t @ Qt.T).unsqueeze(-2)          # (..., 1, k_s)
    q_zt_given_z0 = (z_t @ Qt_bar.T).unsqueeze(-1)      # (..., k_0, 1)
    q_zt_given_z0 = q_zt_given_z0.masked_fill(q_zt_given_z0 == 0, 1e-6)
    q_zs_given_zt_z0 = q_zt_given_zs * Qs_bar / q_zt_given_z0    # (..., k_0, k_s)

    p_zs = (p_z0.unsqueeze(-1) * q_zs_given_zt_z0).sum(-2)     # (..., k_s)
    p_zs[p_zs.sum(-1) == 0] = 1e-5
    return p_zs / p_zs.sum(-1, keepdim=True)

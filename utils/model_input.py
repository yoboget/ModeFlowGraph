"""
Input of the denoiser, built from the noisy graph G_t (shared by the training and the samplers).

Two kinds of input (config `model.input`):
    - 'state':      the noisy state itself (one-hot classes, or points on the simplex for the 'dirichlet' prior), with
                    the noise level as two extra node features (NoiserDense / NoiserDirichlet.add_noise_level).
    - 'sufficient': the sufficient statistic phi(z_t, t)_b ∝ q(z_t | z_1 = b) of every node and edge
                    (utils.noising.sufficient_stat), over the data classes. The elements are noised independently,
                    so p(G_1 | G_t) is the same function of the phi's for every noising process and every time: the
                    denoiser needs no time input (time_input adds the noise level anyway, as an ablation).

The extra features (spectral, cycles, rrwp, ...) need a graph. With 'sufficient', they are computed from the
single-element posterior p(z_1 = b | z_t) ∝ pi_b phi_b, pi the class marginals of the training set: a fixed function
of phi, so the input stays a function of the phi's only, and at full noise its soft adjacency has the edge density
of the data (phi alone is uniform there, a complete graph).
"""
import torch

from utils.func import get_edge_mask
from utils.noising import sufficient_stat

INPUTS = ('state', 'sufficient')


class ModelInput:
    def __init__(self, noiser, extra_features, input_type='state', time_input=False, node_marginals=None,
                 edge_marginals=None):
        """
        Args:
            noiser: NoiserDense or NoiserDirichlet, the noising process of G_t.
            extra_features: ExtraFeatures.
            input_type: 'state' or 'sufficient'.
            time_input: 'sufficient' only, also give the noise level to the denoiser.
            node_marginals, edge_marginals: data class marginals pi ('sufficient' only).
        """
        if input_type not in INPUTS:
            raise NotImplementedError(f'Unknown model input {input_type}. Available: {INPUTS}')
        self.noiser = noiser
        self.extra_features = extra_features
        self.input_type = input_type
        self.time_input = time_input
        if input_type == 'sufficient':
            assert node_marginals is not None and edge_marginals is not None
            self.node_marginals, self.edge_marginals = node_marginals, edge_marginals
            # phi has no mask class: the cycle features must not drop the last class
            extra_features.masking = False

    def __call__(self, x_t, a_t, node_mask, alpha_bar):
        """
        x_t: (bs, n, dx') noisy nodes (dx' = 1 without node types), a_t: (bs, n, n, de') noisy edges,
        node_mask: (bs, n), alpha_bar: scalar or (bs,) noise level of G_t.
        Returns the node and edge inputs of the denoiser.
        """
        if self.input_type == 'state':
            x_in = self.noiser.add_noise_level(x_t, alpha_bar)
            return self.extra_features(x_in, a_t, node_mask)

        adj_mask = get_edge_mask(node_mask)
        phi_a = sufficient_stat(self.noiser, a_t, alpha_bar, 'edge', adj_mask)
        post_a = naive_posterior(phi_a, self.edge_marginals, adj_mask)
        if x_t.size(-1) > 1:
            phi_x = sufficient_stat(self.noiser, x_t, alpha_bar, 'node', node_mask)
            post_x = naive_posterior(phi_x, self.node_marginals, node_mask)
        else:
            phi_x = post_x = x_t   # no node types: x_t is the node mask

        x_feat, a_feat = self.extra_features(post_x, post_a, node_mask)
        x_in = torch.cat((phi_x, x_feat[..., post_x.size(-1):]), dim=-1)
        a_in = torch.cat((phi_a, a_feat[..., post_a.size(-1):]), dim=-1)
        if self.time_input:
            x_in = self.noiser.add_noise_level(x_in, alpha_bar)
        return x_in, a_in


def naive_posterior(phi, marginals, mask):
    """p(z_1 = b | z_t) ∝ pi_b phi_b, the posterior of an element given its own noisy value only. Zeros on padding."""
    post = phi * marginals.to(phi.device)
    return post / post.sum(-1, keepdim=True).clamp_min(1e-30) * mask.unsqueeze(-1)

"""
Multi-mode denoiser: with num_modes = M > 1, the denoiser predicts a mixture over clean graphs
    p(G_0 | G_t) = sum_m pi_m(G_t) prod_i p_m(x_i | G_t) prod_{i<j} p_m(a_ij | G_t),
i.e. a graph-level mode head pi (bs, M) and M sets of node / edge logits, x: (bs, n, M, dx), a: (bs, n, n, M, de).
Within a mode, nodes and edges are independent; the mode carries the dependencies between them.

Training (hyp_loss):
    'wta':     winner-take-all, only the mode with the highest likelihood of G_0 is trained on it, and the
               mode head learns which mode wins (commitment loss).
    'mixture': negative log-likelihood of the mixture.
Optional regularizers:
    MI (mi_weight = lam > 0), with tempered responsibilities r_b = softmax_k(log pi_k + log p_k(G_0^b) / N_b),
        N_b the number of elements of the graph in the loss (without the temperature, r_b is one-hot as soon as
        the modes differ slightly on many elements):
            loss -= lam * (H(mean_b r_b) - mean_b H(r_b)) = lam * I(k; G_0) on the batch,
        i.e. confident per-graph responsibilities with a balanced usage of the modes.
    Divergence (div_weight = lam > 0), the generalized Jensen-Shannon divergence between the modes on each element,
            D_e = H(sum_k pi_k p_k(z_e)) - sum_k pi_k H(p_k(z_e)) = I(k; z_e | G_t) under the model,
        which is 0 when all modes make the same prediction. Weighted by the noise level 1 - alpha_bar_t of the
        graph, since at low noise G_t determines G_0 and there is nothing to specialize on:
            loss -= lam * sum_{b, e} (1 - alpha_bar_t^b) D_be / N.
Sampling (hyp_selection):
    'sample':  draw one mode per graph from pi at every step and use its prediction.
    'average': use the per-element expectation of the mixture sum_m pi_m p_m(z), which is the factorized
               prediction of the standard (single-mode) formulation.
"""
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


def to_class_logits(logits):
    """Binary edges have a single output, the logit of 'no edge' (class 0): returns the logits of both classes."""
    if logits.size(-1) == 1:
        return torch.cat((logits, torch.zeros_like(logits)), dim=-1)
    return logits


def mode_loss(out, x, a, node_weight, edge_weight, hyp_loss, mi_weight=0., div_weight=0., noise_level=None):
    """
    Args:
        out: (mode_logits (bs, M), x_logits (bs, n, M, dx), a_logits (bs, n, n, M, de)).
        x, a: clean graph, one-hot (bs, n, dx) and (bs, n, n, de).
        node_weight (bs, n), edge_weight (bs, n, n): elements entering the likelihood (each edge counted once).
            node_weight is None for graphs without node types.
        noise_level (bs,): 1 - alpha_bar_t, weight of the divergence regularizer (1 if None).
    Returns:
        loss and a dict of detached metrics: node and edge nll, mode loss (commitment for 'wta', responsibility
        entropy for 'mixture'), the batch estimate of I(k; G_0) and the mean divergence D_e between the modes.
    """
    mode_logits, x_logits, a_logits = out
    edge_lp = (to_class_logits(a_logits).log_softmax(-1) * a.unsqueeze(-2)).sum(-1)       # (bs, n, n, M)
    edge_lp = (edge_lp * edge_weight.unsqueeze(-1)).sum((1, 2))                           # (bs, M)
    n_edges = edge_weight.sum()
    if node_weight is not None:
        node_lp = (x_logits.log_softmax(-1) * x.unsqueeze(-2)).sum(-1)                    # (bs, n, M)
        node_lp = (node_lp * node_weight.unsqueeze(-1)).sum(1)                            # (bs, M)
        n_nodes = node_weight.sum()
    else:
        node_lp, n_nodes = torch.zeros_like(edge_lp), 0
    n_elements = (n_nodes + n_edges).clamp_min(1)
    graph_size = edge_weight.flatten(1).sum(-1) + (node_weight.sum(-1) if node_weight is not None else 0)  # (bs,)
    graph_lp = node_lp + edge_lp                                                          # (bs, M)
    joint_lp = mode_logits.log_softmax(-1) + graph_lp

    # graphs without any element in the loss (masking prior at low noise) carry no information on the mode
    has_elements = graph_size > 0

    if hyp_loss == 'mixture':
        loss = -torch.logsumexp(joint_lp, -1).sum() / n_elements
        resp = joint_lp.softmax(-1).detach()
        aux = -torch.xlogy(resp, resp).sum(-1).mean()
    elif hyp_loss == 'wta':
        best = graph_lp.argmax(-1)
        resp = F.one_hot(best, graph_lp.size(-1)).float()
        recon = -(resp * graph_lp).sum() / n_elements
        aux = F.cross_entropy(mode_logits[has_elements], best[has_elements]) if has_elements.any() \
            else mode_logits.sum() * 0
        loss = recon + aux
    else:
        raise NotImplementedError(f'Unknown hyp_loss {hyp_loss}. Available: wta, mixture')

    if has_elements.any():
        tempered_lp = mode_logits.log_softmax(-1) + graph_lp / graph_size.clamp_min(1).unsqueeze(-1)
        mi = mutual_information(tempered_lp[has_elements].softmax(-1))
    else:
        mi = joint_lp.sum() * 0

    # divergence between the modes on each element in the loss, weighted by the noise level of its graph
    pi = mode_logits.softmax(-1)
    noise_level = torch.ones_like(pi[:, 0]) if noise_level is None else noise_level
    div_e = (mode_divergence(pi, to_class_logits(a_logits)) * edge_weight).sum((1, 2))  # (bs,)
    if node_weight is not None:
        div_e = div_e + (mode_divergence(pi, x_logits) * node_weight).sum(1)
    div = (noise_level * div_e).sum() / n_elements

    loss = loss - mi_weight * mi - div_weight * div

    metrics = {'node': -(resp * node_lp).sum().item() / max(float(n_nodes), 1.),
               'edge': -(resp * edge_lp).sum().item() / max(float(n_edges), 1.),
               'mode': aux.item(), 'mi': mi.item(), 'div': div_e.sum().item() / float(n_elements)}
    return loss, metrics


def mutual_information(r):
    """I(k; G_0) = H(k) - H(k | G_0) on a batch of responsibilities r (bs, M)."""
    p = r.mean(0)
    h_k = -(p * p.clamp_min(1e-12).log()).sum()
    h_k_x = -(r * r.clamp_min(1e-12).log()).sum(1).mean()
    return h_k - h_k_x


def mode_divergence(pi, logits):
    """
    Generalized Jensen-Shannon divergence between the modes, H(sum_k pi_k p_k) - sum_k pi_k H(p_k), per element.
    pi: (bs, M), logits: (bs, ..., M, c). Returns (bs, ...).
    """
    logp = logits.log_softmax(-1)
    p = logp.exp()
    w = pi.view(pi.size(0), *[1] * (logits.dim() - 3), pi.size(1))  # (bs, 1, ..., M)
    p_mix = (w.unsqueeze(-1) * p).sum(-2)
    h_mix = -(p_mix * p_mix.clamp_min(1e-12).log()).sum(-1)
    h_modes = -(w * (p * logp).sum(-1)).sum(-1)
    return h_mix - h_modes


def select_mode(out, hyp_selection):
    """
    Turns the multi-mode prediction into per-element probabilities over the data classes,
    x: (bs, n, dx) and a: (bs, n, n, de), by sampling one mode per graph or taking the expectation of the mixture.
    """
    mode_logits, x_logits, a_logits = out
    p_x, p_a = x_logits.softmax(-1), to_class_logits(a_logits).softmax(-1)
    if hyp_selection == 'average':
        pi = mode_logits.softmax(-1)
        return (pi[:, None, :, None] * p_x).sum(-2), (pi[:, None, None, :, None] * p_a).sum(-2)
    if hyp_selection == 'sample':
        m = Categorical(logits=mode_logits).sample()
        b = torch.arange(len(m), device=m.device)
        return p_x[b, :, m], p_a[b, :, :, m]
    raise NotImplementedError(f'Unknown hyp_selection {hyp_selection}. Available: sample, average')

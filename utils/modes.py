"""
Multi-mode denoiser: with num_modes = M > 1, the denoiser predicts a mixture over clean graphs
    p(G_0 | G_t) = sum_m pi_m(G_t) prod_i p_m(x_i | G_t) prod_{i<j} p_m(a_ij | G_t),
i.e. a graph-level mode head pi (bs, M) and M sets of node / edge logits, x: (bs, n, M, dx), a: (bs, n, n, M, de).
Within a mode, nodes and edges are independent; the mode carries the dependencies between them.

Training (hyp_loss):
    'wta':     winner-take-all, only the mode with the highest likelihood of G_0 is trained on it, and the
               mode head learns which mode wins (commitment loss).
    'mixture': negative log-likelihood of the mixture.
    'tempered': interpolation between the two (annealed winner-take-all), with the temperature tau = hyp_temperature.
               Responsibilities r_k = softmax_k(log pi_k + log p_k(G_0) / tau), held fixed, weight the likelihood of
               each mode, and the mode head is trained towards them:
                   loss = -sum_k r_k log p_k(G_0) / N + CE(pi, r).
               tau -> 0: r is the winner, same gradients as 'wta'. tau = 1: r is the posterior of the mixture,
               same gradients as 'mixture' for the modes' predictions (the mode head gets a per-graph instead of a
               per-element weight, as in 'wta'). tau > 1: softer than the posterior, all modes learn from all graphs.
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


def mode_loss(out, x, a, node_weight, edge_weight, hyp_loss, hyp_temperature=1.):
    """
    Args:
        out: (mode_logits (bs, M), x_logits (bs, n, M, dx), a_logits (bs, n, n, M, de)).
        x, a: clean graph, one-hot (bs, n, dx) and (bs, n, n, de).
        node_weight (bs, n), edge_weight (bs, n, n): elements entering the likelihood (each edge counted once).
            node_weight is None for graphs without node types.
        hyp_temperature: temperature tau of the 'tempered' loss.
    Returns:
        loss and a dict of detached metrics: node and edge nll, mode loss (commitment for 'wta', responsibility
        entropy for 'mixture', soft commitment for 'tempered').
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
    elif hyp_loss == 'tempered':
        if hyp_temperature > 0:
            resp = (mode_logits.log_softmax(-1) + graph_lp / hyp_temperature).softmax(-1).detach()
        else:
            resp = F.one_hot(graph_lp.argmax(-1), graph_lp.size(-1)).float()
        recon = -(resp * graph_lp).sum() / n_elements
        commit = -(resp * mode_logits.log_softmax(-1)).sum(-1)
        aux = commit[has_elements].mean() if has_elements.any() else mode_logits.sum() * 0
        loss = recon + aux
    else:
        raise NotImplementedError(f'Unknown hyp_loss {hyp_loss}. Available: wta, mixture, tempered')

    metrics = {'node': -(resp * node_lp).sum().item() / max(float(n_nodes), 1.),
               'edge': -(resp * edge_lp).sum().item() / max(float(n_edges), 1.),
               'mode': aux.item()}
    return loss, metrics


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

import torch
from torch_geometric.utils import to_dense_batch, to_dense_adj


def get_edge_mask(mask, mask_diag=True):
    edge_mask = mask.float()
    edge_mask = edge_mask.unsqueeze(-1) * edge_mask.unsqueeze(-1).transpose(-2, -1)
    if mask_diag:
        device = mask.device
        mask_diag = 1-torch.eye(mask.shape[-1], device=device).unsqueeze(0)
        edge_mask = edge_mask * mask_diag
    return edge_mask.bool()

def batch_to_dense(batch, max_num_nodes):
    X, mask = to_dense_batch(batch.x, batch=batch.batch, max_num_nodes=max_num_nodes)
    A = to_dense_adj(batch.edge_index, edge_attr=batch.edge_attr, batch=batch.batch,
                     max_num_nodes=max_num_nodes)
    A = torch.cat((A.sum(-1, keepdim=True) == 0, A), dim=-1)
    return X, A, mask
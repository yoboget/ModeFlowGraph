import os
import torch

def get_num_nodes_distribution(loader, max_num_nodes, dataset):
    filepath = f'./data/{dataset}/node_distribution.pt'
    if os.path.exists(filepath):
        num_nodes_distribution = torch.load(filepath)
    else:
        num_nodes = 0
        for batch in loader['train']:
            n = batch.batch.bincount()
            num_nodes += n.bincount(minlength=max_num_nodes+1)
        num_nodes_distribution = num_nodes / num_nodes.sum()
        torch.save(num_nodes_distribution, filepath)
    return num_nodes_distribution

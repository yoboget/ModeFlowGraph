import os
import json
import torch
import pickle
import math
from torch_geometric.loader import DataLoader
from utils.utils import batch_to_networkx

from transforms import Qm9Transform, Qm9HTransform
from data.datasets import ZINC, QM9_, KekulizedMolDataset, SpectreGraphDataset, FromNetworkx

def get_dataset(config):

    transforms = None
    if config.dataset == 'qm9H':
        transforms = Qm9HTransform()
        data = QM9_(f'./data/qm9H', pre_transform=transforms)
        idx = torch.randperm(len(data), generator=torch.Generator().manual_seed(42))
        train_idx, test_idx = idx[10000:], idx[:10000]
        train_idx, val_idx = train_idx[config.training.val_size:], train_idx[:config.training.val_size]
        train, val, test = data[train_idx], data[val_idx], data[test_idx]
        test_size = 10_000

    elif config.dataset == 'qm9':
        transforms = Qm9Transform()
        data = KekulizedMolDataset(f'./data/qm9/', dataset='qm9', pre_transform=transforms)
        original_train_idx, test_idx = get_indices('qm9', len(data))
        original_train_idx = original_train_idx[torch.randperm(len(original_train_idx))]
        val_idx = original_train_idx[:config.log.n_val_samples]
        train_idx = original_train_idx[config.log.n_val_samples:]
        train, test, val = data[train_idx], data[test_idx], data[val_idx]
        test_size = 10_000

    elif config.dataset == 'zinc':
        train = ZINC(f'./data/zinc', split= 'train', pre_transform=transforms)
        val = ZINC(f'./data/zinc', split= 'val', pre_transform=transforms)
        test = ZINC(f'./data/zinc', split= 'test', pre_transform=transforms)
        test_size = 10_000

    elif config.dataset in ['planar', 'sbm']:
        train = SpectreGraphDataset(config.dataset, f'./data/{config.dataset}', split= 'train',
                                    pre_transform=transforms)
        val = SpectreGraphDataset(config.dataset, f'./data/{config.dataset}', split= 'val',
                                    pre_transform=transforms)
        test = SpectreGraphDataset(config.dataset, f'./data/{config.dataset}', split= 'test',
                                    pre_transform=transforms)
        test_size = len(test)

    elif config.dataset == 'enzymes':
        data = FromNetworkx(f'./data/{config.dataset}', dataset=config.dataset, pre_transform=transforms)
        train_idx, val_idx, test_idx = get_random_split_indices(len(data))
        train, val, test = data[train_idx], data[val_idx], data[test_idx]
        test_size = len(test_idx)

    train_loader = DataLoader(train, batch_size=config.training.batch_size,
                                  shuffle=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=config.log.n_val_samples)
    test_loader = DataLoader(test, shuffle= True, batch_size=test_size, drop_last=False)

    loaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    save_test_set(test_loader, config.dataset)
    return loaders

def get_indices(dataset, n_instances):
    with open(os.path.join('./data/qm9/raw/', f'test_idx_qm9.json')) as f:
        test_idx = json.load(f)
        if dataset == 'qm9':
            test_idx = test_idx['valid_idxs']
        test_idx = [int(i) for i in test_idx]

    # Create a boolean mask for the training indices
    train_mask = torch.ones(n_instances).bool()
    train_mask[test_idx] = False
    train_idx = torch.arange(n_instances)[train_mask]

    return train_idx, test_idx

def save_test_set(test_loader, dataset):
    filepath = f'./data/{dataset}/test_graph.pkl'
    if not os.path.exists(filepath):
        test_batch = next(iter(test_loader))
        if dataset in ['zinc', 'qm9']:
            test_batch.x = test_batch.x.argmax(-1, keepdims=True)
        test_graphs = batch_to_networkx(test_batch)
        with open(f'./data/{dataset}/test_graph.pkl', 'wb') as f:
            pickle.dump(test_graphs, f)

def get_random_split_indices(n_instances: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Splits a dataset of a given size into random train, validation, and test sets.

    The split is performed as follows:
    1. Test set: 20% of the total dataset size, rounded up.
    2. The remaining 80% is designated for training and validation.
    3. Validation set: 20% of this remaining set, rounded up.
    4. Training set: The rest of the instances.

    Args:
        n_instances (int): The total number of instances in the dataset.

    Returns:
        A tuple containing the training indices, validation indices, and test indices.
    """
    # Create and shuffle all indices to ensure random splits
    g = torch.Generator()
    g.manual_seed(42)
    all_indices = torch.randperm(n_instances, generator=g)

    # 1. Split into test and a temporary training set (20% / 80%)
    test_size = math.ceil(n_instances * 0.20)
    test_idx = all_indices[:test_size]
    train_val_idx = all_indices[test_size:]

    # 2. Split the temporary training set into final validation and train sets (20% / 80%)
    val_size = math.ceil(len(train_val_idx) * 0.20)
    val_idx = train_val_idx[:val_size]
    train_idx = train_val_idx[val_size:]

    return train_idx, val_idx, test_idx

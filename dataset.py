import os
import json
import torch
import pickle
import math
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch, to_dense_adj

from eval.mol_metrics import gen_mol, mols_to_nx, mols_to_smiles

from transforms import Qm9Transform, Qm9HTransform
from data.datasets import ZINC, QM9_, KekulizedMolDataset, SpectreGraphDataset, FromNetworkx

def get_dataset(config):
    """Returns the train / val / test loaders of `config.dataset`. The test loader yields the whole test set."""
    if config.dataset == 'qm9H':
        data = QM9_('./data/qm9H', pre_transform=Qm9HTransform())
        idx = torch.randperm(len(data), generator=torch.Generator().manual_seed(42))
        train_idx, test_idx = idx[10000:], idx[:10000]
        save_qm9H_references(data[train_idx], data[test_idx])
        train_idx, val_idx = train_idx[config.training.val_size:], train_idx[:config.training.val_size]
        train, val, test = data[train_idx], data[val_idx], data[test_idx]
        test_size = 10_000

    elif config.dataset == 'qm9':
        data = KekulizedMolDataset('./data/qm9/', dataset='qm9', pre_transform=Qm9Transform())
        train_idx, test_idx = get_indices('qm9', len(data))
        train_idx = train_idx[torch.randperm(len(train_idx), generator=torch.Generator().manual_seed(42))]
        train_idx, val_idx = train_idx[config.log.n_val_samples:], train_idx[:config.log.n_val_samples]
        train, val, test = data[train_idx], data[val_idx], data[test_idx]
        test_size = 10_000

    elif config.dataset == 'zinc':
        train, val, test = (ZINC('./data/zinc', split=split) for split in ['train', 'val', 'test'])
        test_size = 10_000

    elif config.dataset in ['planar', 'sbm']:
        train, val, test = (SpectreGraphDataset(config.dataset, f'./data/{config.dataset}', split=split)
                            for split in ['train', 'val', 'test'])
        test_size = len(test)

    elif config.dataset == 'enzymes':
        data = FromNetworkx(f'./data/{config.dataset}', dataset=config.dataset)
        train_idx, val_idx, test_idx = get_random_split_indices(len(data))
        train, val, test = data[train_idx], data[val_idx], data[test_idx]
        test_size = len(test_idx)

    else:
        raise NotImplementedError(f'Unknown dataset {config.dataset}')

    train_loader = DataLoader(train, batch_size=config.training.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=config.log.n_val_samples)
    test_loader = DataLoader(test, shuffle=True, batch_size=test_size, drop_last=False)

    return {'train': train_loader, 'val': val_loader, 'test': test_loader}

def get_indices(dataset, n_instances):
    """qm9 train / test indices from `./data/qm9/raw/test_idx_qm9.json`."""
    with open(os.path.join('./data/qm9/raw/', 'test_idx_qm9.json')) as f:
        test_idx = json.load(f)
        if dataset == 'qm9':
            test_idx = test_idx['valid_idxs']
        test_idx = [int(i) for i in test_idx]

    train_mask = torch.ones(n_instances).bool()
    train_mask[test_idx] = False
    train_idx = torch.arange(n_instances)[train_mask]

    return train_idx, test_idx

def save_qm9H_references(train, test):
    """
    Saves the references used to evaluate qm9H samples (once): the SMILES of the training (incl. validation) and
    test molecules, and the test molecules as networkx graphs. The molecules are rebuilt from their graphs in the
    same way as generated graphs (eval.mol_metrics.gen_mol).
    """
    raw_dir = './data/qm9H/raw'
    paths = {name: os.path.join(raw_dir, f'{name}.pkl')
             for name in ['smiles_qm9H_train', 'smiles_qm9H_test', 'test_nx_qm9H']}
    if all(os.path.exists(path) for path in paths.values()):
        return

    print('Building the qm9H evaluation references...')
    train_mols, test_mols = graphs_to_mols(train), graphs_to_mols(test)
    to_save = {'smiles_qm9H_train': mols_to_smiles(train_mols),
               'smiles_qm9H_test': mols_to_smiles(test_mols),
               'test_nx_qm9H': mols_to_nx(test_mols)}
    for name, obj in to_save.items():
        with open(paths[name], 'wb') as f:
            pickle.dump(obj, f)


def graphs_to_mols(dataset, batch_size=1024):
    """Rebuilds the (qm9H) molecules of a dataset from their graphs."""
    mols = []
    for batch in DataLoader(dataset, batch_size=batch_size):
        x, mask = to_dense_batch(batch.x, batch.batch)
        adj = to_dense_adj(batch.edge_index, batch.batch, edge_attr=batch.edge_attr)
        x = x[..., :5].argmax(-1)
        bonds = adj.argmax(-1) + 1  # 0: no bond, 1: single, 2: double, 3: triple
        bonds[adj.sum(-1) == 0] = 0
        mols += gen_mol(x, bonds, mask, dataset='qm9H')[0]
    return mols


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

"""
PyG datasets. Each dataset is downloaded to `<root>/raw` and processed once to `<root>/processed`.
The files used for evaluation (test indices, SMILES, test graphs) also live in `<root>/raw`.

    ZINC                 zinc250k (kekulized), fixed train / val / test split
    KekulizedMolDataset  qm9 (kekulized, without hydrogens)
    QM9_                 qm9 with explicit hydrogens ('qm9H'), built from the original sdf file
    SpectreGraphDataset  planar and sbm graphs from SPECTRE
    FromNetworkx         enzymes, from a pickled list of networkx graphs
"""
import os
import sys
import json
import time
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from torch_geometric.utils import dense_to_sparse, from_networkx, one_hot, scatter
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip


class ZINC(InMemoryDataset):
    def __init__(self, root, split, transform=None, pre_transform=None, pre_filter=None):
        # Atomic numbers 6, 7, 8, 9, 15, 16, 17, 35, 53 -> 1, ..., 9
        self.atom_encoder = torch.tensor([-1, -1, -1, -1, -1, -1, 1, 2, 3, 4, -1, -1, -1, -1, -1, 5, 6, 7, -1, -1, -1,
                                          -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8, -1, -1, -1, -1, -1,
                                          -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 9])

        self.split = split
        if split == 'train':
            self.split_idx = 0
        elif split == 'val':
            self.split_idx = 1
        elif split == 'test':
            self.split_idx = 2
        else:
            raise NotImplementedError('Choose either train or val or test as split')
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return ['zinc_kekulized.npz', 'test_idx_zinc.json', 'test_nx_zinc.pkl', 'smiles_zinc.csv']

    @property
    def processed_file_names(self):
        names = ['data_train.pt', 'data_val.pt', 'data_test.pt']
        return [names[self.split_idx]]

    def process(self):
        filepath = os.path.join(self.raw_dir, self.raw_file_names[0])
        load_data = np.load(filepath)

        xs = load_data['arr_0']
        adjs = load_data['arr_1']
        del load_data

        idx = self.get_indices(len(xs))

        data_list = []
        pbar = tqdm(total=len(xs[idx]))
        pbar.set_description(f'Processing {self.split} dataset')
        for x, adj in zip(xs[idx], adjs[idx]):
            x = x[x > 0]
            x = self.atom_encoder[x] - 1
            x = F.one_hot(x, num_classes=9).float()

            edge_index, edge_attr = from_dense_numpy_to_sparse(adj.squeeze())
            edge_attr = edge_attr.float()

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

            pbar.update(1)
        torch.save(self.collate(data_list), self.processed_paths[0])

    def download(self):
        download_url('https://drive.switch.ch/index.php/s/D8ilMxpcXNHtVUb/download', self.raw_dir,
                     filename=self.raw_file_names[0])
        download_url('https://drive.switch.ch/index.php/s/bBQ5ZHJwlaIKMnL/download', self.raw_dir,
                     filename='test_idx_zinc.json')
        download_url('https://drive.switch.ch/index.php/s/N42GJqdLYtIq4gD/download', self.raw_dir,
                     filename='test_nx_zinc.pkl')
        download_url('https://drive.switch.ch/index.php/s/5pVdYfx4e2TwLah/download', self.raw_dir,
                     filename='smiles_zinc.csv')

    def get_indices(self, n):
        """Fixed test set from `test_idx_zinc.json`; 10,000 random training molecules (seed 0) form the val set."""
        with open(os.path.join(self.raw_dir, 'test_idx_zinc.json')) as f:
            test_idx = [int(i) for i in json.load(f)]

        is_train = torch.ones(n).bool()
        is_train[test_idx] = False
        train_idx = is_train.nonzero()
        test_idx = (~is_train).nonzero()

        torch.manual_seed(0)
        perm = torch.randperm(len(train_idx))
        train_idx, val_idx = train_idx[perm][10000:], train_idx[perm][:10000]
        return [train_idx, val_idx, test_idx][self.split_idx]


class KekulizedMolDataset(InMemoryDataset):
    """Kekulized qm9 (heavy atoms C, N, O, F only)."""

    def __init__(self, root, dataset=None, transform=None, pre_transform=None, pre_filter=None):
        if dataset != 'qm9':
            raise NotImplementedError(f'KekulizedMolDataset only supports qm9, got {dataset}')
        self.dataset = dataset
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return ['qm9_kekulized.npz', 'test_idx_qm9.json', 'test_nx_qm9.pkl', 'smiles_qm9.csv']

    @property
    def processed_file_names(self):
        return ['data_qm9.pt']

    def download(self):
        download_url('https://drive.switch.ch/index.php/s/SESlx1ylQAopXsi/download', self.raw_dir,
                     filename='qm9_kekulized.npz')
        download_url('https://drive.switch.ch/index.php/s/jFUHlWB9xsVY9R3/download', self.raw_dir,
                     filename='test_idx_qm9.json')
        download_url('https://drive.switch.ch/index.php/s/47GzmvO808HpovN/download', self.raw_dir,
                     filename='test_nx_qm9.pkl')
        download_url('https://drive.switch.ch/index.php/s/sDQbTunpQqjD1oI/download', self.raw_dir,
                     filename='smiles_qm9.csv')

    def process(self):
        start = time.time()
        load_data = np.load(os.path.join(self.raw_dir, 'qm9_kekulized.npz'))
        xs = load_data['arr_0']
        adjs = load_data['arr_1']

        data_list = []
        for i, (x, adj) in enumerate(zip(xs, adjs)):
            x = x[x > 0]
            x = torch.eye(4)[x - 6]  # atomic numbers 6, 7, 8, 9 -> one-hot C, N, O, F
            edge_index, edge_attr = from_dense_numpy_to_sparse(adj)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, max_num_nodes=9)
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)
            if (i+1) % 1000 == 0:
                print(f'{i+1} graphs processed... process continue')

        print(f'{len(data_list)} graphs processed')
        data, slices = self.collate(data_list)
        print('Data collated')
        torch.save((data, slices), self.processed_paths[0])
        print(f'Preprocessing took {time.time() - start} seconds')


class FromNetworkx(InMemoryDataset):
    """Enzymes, from a pickled list of networkx graphs (no node / edge types)."""

    def __init__(self, root, dataset=None, transform=None, pre_transform=None, pre_filter=None):
        if dataset != 'enzymes':
            raise NotImplementedError(f'FromNetworkx only supports enzymes, got {dataset}')
        self.dataset = dataset
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return ['ENZYMES.pkl']

    @property
    def processed_file_names(self):
        return ['ENZYMES.pkl']

    def download(self):
        download_url('https://drive.switch.ch/index.php/s/dGo2OUFmOIqqDNt/download', self.raw_dir,
                     filename='ENZYMES.pkl')

    def process(self):
        with open(os.path.join(self.raw_dir, self.raw_file_names[0]), 'rb') as pickle_file:
            graph_list = pickle.load(pickle_file)

        data_list = []
        for g in graph_list:
            data = from_networkx(g)
            data.max_num_nodes = 125
            data.x = torch.ones(data.num_nodes, 1, dtype=torch.float)
            data.edge_attr = torch.ones(data.edge_index.shape[-1], 1, dtype=torch.float)
            data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[0])


def from_dense_numpy_to_sparse(adj):
    """(n_bond_types, n, n) one-hot bond tensor -> edge_index and one-hot edge_attr (single, double, triple)."""
    adj = torch.from_numpy(adj)
    no_edge = 1 - adj.sum(0, keepdim=True)
    adj = torch.cat((no_edge, adj), dim=0)
    adj = adj.argmax(0)
    edge_index, edge_attr = dense_to_sparse(adj)
    edge_attr = torch.eye(3)[edge_attr - 1]
    return edge_index, edge_attr


class SpectreGraphDataset(InMemoryDataset):
    """Planar and SBM graphs from SPECTRE (https://github.com/KarolisMart/SPECTRE)."""

    urls = {
        'sbm': 'https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/sbm_200.pt',
        'planar': 'https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/planar_64_200.pt',
    }

    def __init__(self, dataset_name, root, split, transform=None, pre_transform=None, pre_filter=None):
        if dataset_name not in self.urls:
            raise ValueError(f'Unknown dataset {dataset_name}')
        self.dataset_name = dataset_name
        self.split = split
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    @property
    def processed_file_names(self):
        return [f'{self.split}.pt' if self.split in ['train', 'val'] else 'test.pt']

    def download(self):
        """Downloads the SPECTRE file and saves a fixed 64 / 16 / 20 % train / val / test split of the adjacency
        matrices (split seed 1234, nodes randomly permuted)."""
        file_path = download_url(self.urls[self.dataset_name], self.raw_dir)
        adjs, eigvals, eigvecs, n_nodes, max_eigval, min_eigval, same_sample, n_max = torch.load(file_path)

        g_cpu = torch.Generator()
        g_cpu.manual_seed(1234)
        num_graphs = len(adjs)
        test_len = int(round(num_graphs * 0.2))
        train_len = int(round((num_graphs - test_len) * 0.8))
        val_len = num_graphs - train_len - test_len
        indices = torch.randperm(num_graphs, generator=g_cpu)
        print(f'Dataset sizes: train {train_len}, val {val_len}, test {test_len}')
        train_indices = indices[:train_len]
        val_indices = indices[train_len: train_len + val_len]
        test_indices = indices[train_len + val_len:]

        train_data, val_data, test_data = [], [], []
        for i, adj in enumerate(adjs):
            # permute randomly nodes as for molecular datasets
            random_order = torch.randperm(adj.shape[-1])
            adj = adj[random_order, :]
            adj = adj[:, random_order]

            if i in train_indices:
                train_data.append(adj)
            if i in val_indices:
                val_data.append(adj)
            if i in test_indices:
                test_data.append(adj)

        torch.save(train_data, self.raw_paths[0])
        torch.save(val_data, self.raw_paths[1])
        torch.save(test_data, self.raw_paths[2])

    def process(self):
        raw_dataset = torch.load(os.path.join(self.raw_dir, f'{self.split}.pt'))
        data_list = []
        for adj in raw_dataset:
            n = adj.shape[-1]
            edge_index, _ = dense_to_sparse(adj)
            data = Data(x=torch.ones(n, 1, dtype=torch.float), edge_index=edge_index,
                        edge_attr=torch.ones(edge_index.shape[-1], 1, dtype=torch.float),
                        n_nodes=n * torch.ones(1, dtype=torch.long))

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])


#### QM9 ####
HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414

conversion = torch.tensor([
    1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
    1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
])


class QM9_(InMemoryDataset):
    """
    QM9 with explicit hydrogens, adapted from the PyG QM9 dataset (MoleculeNet, https://arxiv.org/abs/1703.00564).
    Node features x: one-hot type (H, C, N, O, F), atomic number, aromatic, number of hydrogens; y: the 19
    regression targets. The evaluation references are built from the train / test split in dataset.py.
    """

    raw_url = ('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/'
               'molnet_publish/qm9.zip')
    raw_url2 = 'https://ndownloader.figshare.com/files/3195404'
    processed_url = 'https://data.pyg.org/datasets/qm9_v3.zip'

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        try:
            import rdkit  # noqa
            return ['gdb9.sdf', 'gdb9.sdf.csv', 'uncharacterized.txt']
        except ImportError:
            return ['qm9_v3.pt']

    @property
    def processed_file_names(self):
        return 'data_v3.pt'

    def download(self):
        try:
            import rdkit  # noqa
            file_path = download_url(self.raw_url, self.raw_dir)
            extract_zip(file_path, self.raw_dir)
            os.unlink(file_path)

            download_url(self.raw_url2, self.raw_dir)
            os.rename(os.path.join(self.raw_dir, '3195404'),
                      os.path.join(self.raw_dir, 'uncharacterized.txt'))
        except ImportError:
            path = download_url(self.processed_url, self.raw_dir)
            extract_zip(path, self.raw_dir)
            os.unlink(path)

    def process(self):
        try:
            import rdkit
            from rdkit import Chem, RDLogger
            from rdkit.Chem.rdchem import BondType as BT
            RDLogger.DisableLog('rdApp.*')
        except ImportError:
            rdkit = None

        if rdkit is None:
            print(("Using a pre-processed version of the dataset. Please "
                   "install 'rdkit' to alternatively process the raw data."),
                  file=sys.stderr)

            data_list = torch.load(self.raw_paths[0])
            data_list = [Data(**data_dict) for data_dict in data_list]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            torch.save(self.collate(data_list), self.processed_paths[0])
            return

        types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2}

        with open(self.raw_paths[1], 'r') as f:
            target = f.read().split('\n')[1:-1]
            target = [[float(x) for x in line.split(',')[1:20]]
                      for line in target]
            target = torch.tensor(target, dtype=torch.float)
            target = torch.cat([target[:, 3:], target[:, :3]], dim=-1)
            target = target * conversion.view(1, -1)

        with open(self.raw_paths[2], 'r') as f:
            skip = set(int(x.split()[0]) - 1 for x in f.read().split('\n')[9:-2])

        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False,
                                   sanitize=False)

        data_list = []
        for i, mol in enumerate(tqdm(suppl)):
            if i in skip:
                continue

            N = mol.GetNumAtoms()

            conf = mol.GetConformer()
            pos = conf.GetPositions()
            pos = torch.tensor(pos, dtype=torch.float)

            type_idx = []
            atomic_number = []
            aromatic = []

            for atom in mol.GetAtoms():
                type_idx.append(types[atom.GetSymbol()])
                atomic_number.append(atom.GetAtomicNum())
                aromatic.append(1 if atom.GetIsAromatic() else 0)
            z = torch.tensor(atomic_number, dtype=torch.long)

            row, col, edge_type = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_type += 2 * [bonds[bond.GetBondType()]]

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            edge_attr = one_hot(edge_type, num_classes=len(bonds))

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_attr = edge_attr[perm]

            row, col = edge_index
            hs = (z == 1).to(torch.float)
            num_hs = scatter(hs[row], col, dim_size=N, reduce='sum').tolist()

            x1 = one_hot(torch.tensor(type_idx), num_classes=len(types))
            x2 = torch.tensor([atomic_number, aromatic, num_hs],
                              dtype=torch.float).t().contiguous()
            x = torch.cat([x1, x2], dim=-1)

            y = target[i].unsqueeze(0)
            name = mol.GetProp('_Name')
            smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)

            data = Data(x=x, z=z, pos=pos, edge_index=edge_index,
                        edge_attr=edge_attr, y=y, name=name, smiles=smiles, idx=i)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

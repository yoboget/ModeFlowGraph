import os
import os.path as osp
import sys
import json
import time
import pickle
import networkx as nx

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from torch_geometric.utils import dense_to_sparse, from_networkx, to_dense_batch, to_dense_adj
from torch_geometric.utils import one_hot, scatter
from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from typing import Callable, List, Optional

from eval.mol_metrics import gen_mol, mols_to_nx


class ZINC(InMemoryDataset):
    def __init__(self, root, split, transform=None, pre_transform=None, pre_filter=None):
        self.atom_encoder = torch.tensor([-1, -1, -1, -1, -1, -1, 1, 2, 3, 4, -1, -1, -1, -1, -1, 5, 6, 7, -1, -1, -1,
                                          -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8, -1, -1, -1, -1, -1,
                                          -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 9])  # 6, 7, 8,... -> 1, 2, 3

        self.split = split
        if split == 'train':
            self.split_idx = 0
        elif split == 'val':
            self.split_idx = 1
        elif split == 'test':
            self.split_idx = 2
        else:
            NotImplementedError('Choose either train or val or test as split')
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

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

        n_graphs = len(xs)
        idx = self.get_indices(n_graphs)

        data_list = []
        pbar = tqdm(total=len(xs[idx]))
        pbar.set_description(f'Processing {self.split} dataset')
        for i, (x, adj) in enumerate(zip(xs[idx], adjs[idx])):
            x = x[x>0]
            x = self.atom_encoder[x] - 1
            x = F.one_hot(x, num_classes=9).float()

            adj = torch.from_numpy(adj).squeeze()
            no_edge = 1 - adj.sum(0, keepdim=True)
            adj = torch.cat((no_edge, adj), dim=0)
            adj = adj.argmax(0)
            edge_index, edge_attr = dense_to_sparse(adj)
            edge_attr = F.one_hot(edge_attr-1, 3).float()

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

            pbar.update(1)
        torch.save(self.collate(data_list), self.processed_paths[0])

    def download(self):
        download_url('https://drive.switch.ch/index.php/s/D8ilMxpcXNHtVUb/download', self.raw_dir,
                         filename=self.raw_file_names[0])

        index_filename = 'test_idx_zinc.json'
        test_filename = 'test_nx_zinc.pkl'
        smiles_filename = 'smiles_zinc.csv'
        download_url('https://drive.switch.ch/index.php/s/bBQ5ZHJwlaIKMnL/download', self.root,
                         filename=index_filename)
        download_url('https://drive.switch.ch/index.php/s/N42GJqdLYtIq4gD/download', self.root,
                         filename=test_filename)
        download_url('https://drive.switch.ch/index.php/s/5pVdYfx4e2TwLah/download', self.root,
                         filename=smiles_filename)

    def get_indices(self, n):
        with open(os.path.join(self.root, f'test_idx_zinc.json')) as f:
            test_idx = json.load(f)
            test_idx = [int(i) for i in test_idx]

        # Create a boolean mask for the training indices
        train_idx_bool = torch.ones(n).bool()
        train_idx_bool[test_idx] = False
        train_idx = train_idx_bool.nonzero()
        test_idx = (1-train_idx_bool.int()).nonzero()

        torch.manual_seed(0)
        idx = torch.randperm(len(train_idx))
        train_idx, val_idx = train_idx[idx][10000:], train_idx[idx][:10000]
        idx = [train_idx, val_idx, test_idx]
        return idx[self.split_idx]


class KekulizedMolDataset(InMemoryDataset):
    def __init__(self, root, dataset=None, transform=None, pre_transform=None, pre_filter=None):
        self.dataset = dataset
        super().__init__(root, transform, pre_transform, pre_filter)
        torch.load(self.processed_paths[0])
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        if self.dataset == 'qm9' or self.dataset == 'qm9_cc' or self.dataset == 'qm9_dg':
            return ['qm9_kekulized.npz',  'test_idx_qm9.json', 'test_nx_qm9.pkl', 'smiles_qm9.pkl']
        else:
            raise NotImplementedError()

    @property
    def processed_file_names(self):
        if self.dataset == 'zinc':
            return ['zinc_data.pt']
        elif self.dataset == 'qm9':
            return ['data_qm9.pt']
        elif self.dataset == 'qm9_cc':
            return ['data_qm9_cc.pt']
        elif self.dataset == 'qm9_dg':
            return ['data_qm9_dg.pt']

    def download(self):
        # Download to `self.raw_dir`.
        if self.dataset == 'qm9' or self.dataset == 'qm9_cc' or self.dataset == 'qm9_dg':
            download_url('https://drive.switch.ch/index.php/s/SESlx1ylQAopXsi/download', self.raw_dir,
                         filename='qm9_kekulized.npz')
            download_url('https://drive.switch.ch/index.php/s/jFUHlWB9xsVY9R3/download', self.root,
                         filename='test_idx_qm9.json')
            download_url('https://drive.switch.ch/index.php/s/47GzmvO808HpovN/download', self.root,
                         filename='test_nx_qm9.pkl')
            download_url('https://drive.switch.ch/index.php/s/sDQbTunpQqjD1oI/download', self.root,
                         filename='smiles_qm9.csv')

    def process(self):
        if self.dataset == 'zinc':
            filepath = os.path.join(self.raw_dir, 'zinc250k_kekulized.npz')
            max_num_nodes = 38
        elif self.dataset == 'qm9':
            filepath = os.path.join(self.raw_dir, 'qm9_kekulized.npz')
            max_num_nodes = 9
        elif self.dataset == 'qm9_cc' or self.dataset == 'qm9_dg':
            filepath = os.path.join(self.raw_dir, 'qm9_kekulized.npz')
            max_num_nodes = 9
        start = time.time()
        load_data = np.load(filepath)
        xs = load_data['arr_0']
        adjs = load_data['arr_1']
        data_list = []

        for i, (x, adj) in enumerate(zip(xs, adjs)):
            x = atom_number_to_one_hot(x, self.dataset)
            edge_index, edge_attr = from_dense_numpy_to_sparse(adj)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, max_num_nodes=max_num_nodes)
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)
            if (i+1) % 1000 == 0:
                print(f'{i+1} graphs processed... process continue')

        print(f'{len(data_list)} graphs processed')
        data, slices = self.collate(data_list)
        print('Data collated')
        torch.save((data, slices), self.processed_paths[0])
        time_taken = time.time() - start
        print(f'Preprocessing took {time_taken} seconds')

class FromNetworkx(InMemoryDataset):
    def __init__(self, root, dataset=None, transform=None, pre_transform=None, pre_filter=None):
        self.dataset = dataset
        super().__init__(root, transform, pre_transform, pre_filter)
        torch.load(self.processed_paths[0])
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self):
        if self.dataset == 'ego':
            return ['ego_small.pkl']
        elif self.dataset == 'community':
            return ['community_small.pkl']
        elif self.dataset == 'enzymes':
            return ['ENZYMES.pkl']
        else:
            raise NotImplementedError()

    def download(self):
        # Download to `self.raw_dir`.
        if self.dataset == 'ego':
            download_url('https://drive.switch.ch/index.php/s/KezKAJHY4bWNl9E/download', self.raw_dir,
                         filename='ego_small.pkl')
        elif self.dataset == 'community':
            download_url('https://drive.switch.ch/index.php/s/SLDFLYSBgsfV0ZA/download', self.raw_dir,
                         filename='community_small.pkl')

        elif self.dataset == 'enzymes':
            download_url('https://drive.switch.ch/index.php/s/dGo2OUFmOIqqDNt/download', self.raw_dir,
                         filename='ENZYMES.pkl')

    @property
    def processed_file_names(self):
        if self.dataset == 'ego':
            return ['ego_data.pt']
        elif self.dataset == 'community':
            return ['community_data.pt']
        elif self.dataset == 'enzymes':
            return ['ENZYMES.pkl']

    def process(self):
        filepath = os.path.join(self.raw_dir, self.raw_file_names[0])
        with open(filepath, 'rb') as pickle_file:
            graph_list = pickle.load(pickle_file)

        data_list = []
        for g in graph_list:
            data = from_networkx(g)
            if self.dataset == 'ego':
                data.max_num_nodes = 18
            elif self.dataset == 'community':
                data.max_num_nodes = 20
            elif self.dataset == 'enzymes':
                data.max_num_nodes = 125
                data.x = torch.ones(data.num_nodes, 1, dtype=torch.float)
                data.edge_attr = torch.ones(data.edge_index.shape[-1], 1, dtype=torch.float)
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def atom_number_to_one_hot(x, dataset):
    x = x[x > 0]
    if dataset == 'zinc':
        zinc250k_atomic_index = torch.tensor([0, 0, 0, 0, 0, 0, 1, 2, 3, 4,
                                              0, 0, 0, 0, 0, 5, 6, 7, 0, 0,
                                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                              0, 0, 0, 0, 0, 8, 0, 0, 0, 0,
                                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                              0, 0, 0, 9])  # 0, 6, 7, 8, 9, 15, 16, 17, 35, 53
        x = zinc250k_atomic_index[x] - 1  # 6, 7, 8, 9, 15, 16, 17, 35, 53 -> 0, 1, 2, 3, 4, 5, 6, 7, 8
        x = torch.eye(9)[x]
    else:
        x = torch.eye(4)[x - 6]
    return x


def from_dense_numpy_to_sparse(adj):
    adj = torch.from_numpy(adj)
    no_edge = 1 - adj.sum(0, keepdim=True)
    adj = torch.cat((no_edge, adj), dim=0)
    adj = adj.argmax(0)
    edge_index, edge_attr = dense_to_sparse(adj)
    edge_attr = torch.eye(3)[edge_attr - 1]
    return edge_index, edge_attr


class SpectreGraphDataset(InMemoryDataset):
    def __init__(
            self,
            dataset_name,
            root,
            split,
            transform=None,
            pre_transform=None,
            pre_filter=None,
    ):
        self.sbm_file = "sbm_200.pt"
        self.planar_file = "planar_64_200.pt"
        self.comm20_file = "community_12_21_100.pt"
        self.dataset_name = dataset_name

        self.split = split
        if self.split == "train":
            self.file_idx = 0
        elif self.split == "val":
            self.file_idx = 1
        else:
            self.file_idx = 2

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def raw_file_names(self):
        return ["train.pt", "val.pt", "test.pt"] #'ego.pkl', 'ego_ns.pkl'

    @property
    def split_file_name(self):
        return ["train.pt", "val.pt", "test.pt"]

    @property
    def split_paths(self):
        r"""The absolute filepaths that must be present in order to skip
        splitting."""
        files = self.split_file_name
        return [os.path.join(self.raw_dir, f) for f in files]

    @property
    def processed_file_names(self):
        if self.split == "train":
            return [
                f"train.pt",
                # f"train_n.pickle",
                # f"train_node_types.npy",
                # f"train_bond_types.npy",
            ]
        elif self.split == "val":
            return [
                f"val.pt",
                # f"val_n.pickle",
                # f"val_node_types.npy",
                # f"val_bond_types.npy",
            ]
        else:
            return [
                f"test.pt",
                # f"test_n.pickle",
                # f"test_node_types.npy",
                # f"test_bond_types.npy",
            ]

    def download(self):
        """
        Download raw qm9 files. Taken from PyG QM9 class
        """
        if self.dataset_name == "sbm":
            raw_url = "https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/sbm_200.pt"
        elif self.dataset_name == "planar":
            raw_url = "https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/planar_64_200.pt"
        elif self.dataset_name == "comm20":
            raw_url = "https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/community_12_21_100.pt"
        elif self.dataset_name == "ego":
            raw_url = "https://raw.githubusercontent.com/tufts-ml/graph-generation-EDGE/main/graphs/Ego.pkl"
        else:
            raise ValueError(f"Unknown dataset {self.dataset_name}")
        file_path = download_url(raw_url, self.raw_dir)

        if self.dataset_name == 'ego':
            networks = pickle.load(open(file_path, 'rb'))
            adjs = [torch.Tensor(nx.to_numpy_array(network)).fill_diagonal_(0) for network in networks]
        else:
            (
                adjs,
                eigvals,
                eigvecs,
                n_nodes,
                max_eigval,
                min_eigval,
                same_sample,
                n_max,
            ) = torch.load(file_path)

        g_cpu = torch.Generator()
        g_cpu.manual_seed(1234)
        self.num_graphs = len(adjs)

        if self.dataset_name == 'ego':
            test_len = int(round(self.num_graphs * 0.2))
            train_len = int(round(self.num_graphs * 0.8))
            val_len = int(round(self.num_graphs * 0.2))
            indices = torch.randperm(self.num_graphs, generator=g_cpu)
            print(f"Dataset sizes: train {train_len}, val {val_len}, test {test_len}")
            train_indices = indices[:train_len]
            val_indices = indices[:val_len]
            test_indices = indices[train_len:]
        else:
            test_len = int(round(self.num_graphs * 0.2))
            train_len = int(round((self.num_graphs - test_len) * 0.8))
            val_len = self.num_graphs - train_len - test_len
            indices = torch.randperm(self.num_graphs, generator=g_cpu)
            print(f"Dataset sizes: train {train_len}, val {val_len}, test {test_len}")
            train_indices = indices[:train_len]
            val_indices = indices[train_len: train_len + val_len]
            test_indices = indices[train_len + val_len:]

        print(f"Train indices: {train_indices}")
        print(f"Val indices: {val_indices}")
        print(f"Test indices: {test_indices}")
        train_data = []
        val_data = []
        test_data = []
        train_data_nx = []
        val_data_nx = []
        test_data_nx = []

        for i, adj in enumerate(adjs):
            # permute randomly nodes as for molecular datasets
            random_order = torch.randperm(adj.shape[-1])
            adj = adj[random_order, :]
            adj = adj[:, random_order]
            net = nx.from_numpy_matrix(adj.numpy()).to_undirected()

            if i in train_indices:
                train_data.append(adj)
                train_data_nx.append(net)
            if i in val_indices:
                val_data.append(adj)
                val_data_nx.append(net)
            if i in test_indices:
                test_data.append(adj)
                test_data_nx.append(net)

        torch.save(train_data, self.raw_paths[0])
        torch.save(val_data, self.raw_paths[1])
        torch.save(test_data, self.raw_paths[2])

        # import pdb; pdb.set_trace()
        # all_data = {'train': train_data_nx, 'val': val_data_nx, 'test': test_data_nx}
        # with open(self.raw_paths[3], 'wb') as handle:
        #     pickle.dump(all_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # with open(self.raw_paths[4], 'wb') as handle:
        #     pickle.dump(train_data_nx + val_data_nx + test_data_nx, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def process(self):
        raw_dataset = torch.load(os.path.join(self.raw_dir, "{}.pt".format(self.split)))
        data_list = []
        for adj in raw_dataset:
            n = adj.shape[-1]
            X = torch.ones(n, 1, dtype=torch.long)
            edge_index, _ = dense_to_sparse(adj)
            edge_attr = torch.ones(edge_index.shape[-1], 1, dtype=torch.float)
            n_nodes = n * torch.ones(1, dtype=torch.long)
            data = Data(
                x=X.float(), edge_index=edge_index, edge_attr=edge_attr.float(), n_nodes=n_nodes
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        # num_nodes = node_counts(data_list)
        # node_types = atom_type_counts(data_list, num_classes=1)
        # bond_types = edge_counts(data_list, num_bond_types=2)
        torch.save(self.collate(data_list), self.processed_paths[0])
        # save_pickle(num_nodes, self.processed_paths[1])
        # np.save(self.processed_paths[2], node_types)
        # np.save(self.processed_paths[3], bond_types)


#### QM9 ####
HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414

conversion = torch.tensor([
    1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
    1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
])
atomrefs = {
    6: [0., 0., 0., 0., 0.],
    7: [
        -13.61312172, -1029.86312267, -1485.30251237, -2042.61123593,
        -2713.48485589
    ],
    8: [
        -13.5745904, -1029.82456413, -1485.26398105, -2042.5727046,
        -2713.44632457
    ],
    9: [
        -13.54887564, -1029.79887659, -1485.2382935, -2042.54701705,
        -2713.42063702
    ],
    10: [
        -13.90303183, -1030.25891228, -1485.71166277, -2043.01812778,
        -2713.88796536
    ],
    11: [0., 0., 0., 0., 0.],
}

class QM9_(InMemoryDataset):
    r"""The QM9 dataset from the `"MoleculeNet: A Benchmark for Molecular
    Machine Learning" <https://arxiv.org/abs/1703.00564>`_ paper, consisting of
    about 130,000 molecules with 19 regression targets.
    Each molecule includes complete spatial information for the single low
    energy conformation of the atoms in the molecule.
    In addition, we provide the atom features from the `"Neural Message
    Passing for Quantum Chemistry" <https://arxiv.org/abs/1704.01212>`_ paper.

    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | Target | Property                         | Description                                                                       | Unit                                        |
    +========+==================================+===================================================================================+=============================================+
    | 0      | :math:`\mu`                      | Dipole moment                                                                     | :math:`\textrm{D}`                          |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 1      | :math:`\alpha`                   | Isotropic polarizability                                                          | :math:`{a_0}^3`                             |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 2      | :math:`\epsilon_{\textrm{HOMO}}` | Highest occupied molecular orbital energy                                         | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 3      | :math:`\epsilon_{\textrm{LUMO}}` | Lowest unoccupied molecular orbital energy                                        | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 4      | :math:`\Delta \epsilon`          | Gap between :math:`\epsilon_{\textrm{HOMO}}` and :math:`\epsilon_{\textrm{LUMO}}` | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 5      | :math:`\langle R^2 \rangle`      | Electronic spatial extent                                                         | :math:`{a_0}^2`                             |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 6      | :math:`\textrm{ZPVE}`            | Zero point vibrational energy                                                     | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 7      | :math:`U_0`                      | Internal energy at 0K                                                             | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 8      | :math:`U`                        | Internal energy at 298.15K                                                        | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 9      | :math:`H`                        | Enthalpy at 298.15K                                                               | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 10     | :math:`G`                        | Free energy at 298.15K                                                            | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 11     | :math:`c_{\textrm{v}}`           | Heat capavity at 298.15K                                                          | :math:`\frac{\textrm{cal}}{\textrm{mol K}}` |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 12     | :math:`U_0^{\textrm{ATOM}}`      | Atomization energy at 0K                                                          | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 13     | :math:`U^{\textrm{ATOM}}`        | Atomization energy at 298.15K                                                     | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 14     | :math:`H^{\textrm{ATOM}}`        | Atomization enthalpy at 298.15K                                                   | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 15     | :math:`G^{\textrm{ATOM}}`        | Atomization free energy at 298.15K                                                | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 16     | :math:`A`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 17     | :math:`B`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 18     | :math:`C`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+

    .. note::

        We also provide a pre-processed version of the dataset in case
        :class:`rdkit` is not installed. The pre-processed version matches with
        the manually processed version as outlined in :meth:`process`.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - #graphs
          - #nodes
          - #edges
          - #features
          - #tasks
        * - 130,831
          - ~18.0
          - ~37.3
          - 11
          - 19
    """  # noqa: E501

    raw_url = ('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/'
               'molnet_publish/qm9.zip')
    raw_url2 = 'https://ndownloader.figshare.com/files/3195404'
    processed_url = 'https://data.pyg.org/datasets/qm9_v3.zip'

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])


    def mean(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].mean())

    def std(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].std())

    def atomref(self, target) -> Optional[torch.Tensor]:
        if target in atomrefs:
            out = torch.zeros(100)
            out[torch.tensor([1, 6, 7, 8, 9])] = torch.tensor(atomrefs[target])
            return out.view(-1, 1)
        return None

    @property
    def raw_file_names(self) -> List[str]:
        try:
            import rdkit  # noqa
            return ['gdb9.sdf', 'gdb9.sdf.csv', 'uncharacterized.txt']
        except ImportError:
            return ['qm9_v3.pt']

    @property
    def processed_file_names(self) -> str:
        return 'data_v3.pt'

    def download(self):
        try:
            import rdkit  # noqa
            file_path = download_url(self.raw_url, self.raw_dir)
            extract_zip(file_path, self.raw_dir)
            os.unlink(file_path)

            file_path = download_url(self.raw_url2, self.raw_dir)
            os.rename(osp.join(self.raw_dir, '3195404'),
                      osp.join(self.raw_dir, 'uncharacterized.txt'))
        except ImportError:
            path = download_url(self.processed_url, self.raw_dir)
            extract_zip(path, self.raw_dir)
            os.unlink(path)

    def process(self):
        try:
            import rdkit
            from rdkit import Chem, RDLogger
            from rdkit.Chem.rdchem import BondType as BT
            from rdkit.Chem.rdchem import HybridizationType
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
            skip = [int(x.split()[0]) - 1 for x in f.read().split('\n')[9:-2]]

        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False,
                                   sanitize=False)

        data_list = []
        smiles_train = []
        smiles_test = []
        mols_train = []
        mols_test = []
        valid_total = 0
        train, test = get_indices('qm9', len(suppl))

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

            x, mask = to_dense_batch(data.x)
            adj = to_dense_adj(data.edge_index, edge_attr=data.edge_attr)
            x = x[..., :5].argmax(-1)
            adj_ = adj.argmax(-1) + 1
            adj_[adj.sum(-1) == 0] = 0
            mol, valid = gen_mol(x, adj_, mask, dataset='qm9H')
            mol = mol[0]
            valid_total += valid
            smi = Chem.MolToSmiles(mol)
            if i in train:
                smiles_train.append(smi)
                mols_train.append(mol)
            elif i in test:
                smiles_test.append(smi)
                mols_test.append(mol)


            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data, mol)

            data_list.append(data)

        nx_train = mols_to_nx(mols_train)
        nx_test = mols_to_nx(mols_test)

        with open(os.path.join(self.root, 'smiles_qm9_train.pkl'), 'wb') as f:
            pickle.dump(smiles_train, f)
        with open(os.path.join(self.root, 'smiles_qm9_test.pkl'), 'wb') as f:
            pickle.dump(smiles_test, f)

        with open(os.path.join(self.root, 'test_nx_qm9H.pkl'), 'wb') as f:
            pickle.dump(nx_train, f)
        with open(os.path.join(self.root, 'train_nx_qm9H.pkl'), 'wb') as f:
            pickle.dump(nx_test, f)
        torch.save(self.collate(data_list), self.processed_paths[0])


def get_indices(dataset, n_instances):
    with open(os.path.join('./data/qm9/', f'test_idx_qm9.json')) as f:
        test_idx = json.load(f)
        if dataset in ['qm9','qm9_cc', 'qm9H']:
            test_idx = test_idx['valid_idxs']
        test_idx = [int(i) for i in test_idx]

    # Create a boolean mask for the training indices
    train_idx = torch.ones(n_instances).bool()
    train_idx[test_idx] = False
    train_idx = torch.arange(len(train_idx))[train_idx].int()
    return train_idx, test_idx

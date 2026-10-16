import torch

class ExtraFeatures:
    def __init__(self, config):
        self.spectral_embeddings = config.extra_features.spectral_embeddings
        self.graph_size = config.extra_features.graph_size
        self.k = config.extra_features.k
        self.molecular_feat = config.extra_features.molecular_feat
        self.cycles = config.extra_features.cycles
        self.rrwp = config.extra_features.rrwp
        self.rrwp_k = config.extra_features.rrwp_k
        self.n_features = 2 * self.k + self.graph_size + self.rrwp_k + 2 * self.molecular_feat
        if self.cycles:
            self.n_features += 3
            self.ncycles = NodeCycleFeatures()

        transition = config.model.prior
        self.masking = True if transition == 'masking' else False

        if config.dataset == 'zinc':
            self.valencies = [4, 3, 2, 1, 3, 2, 1, 1, 1]
            self.bonds = [0, 1, 2, 3, 0]
        elif config.dataset == 'qm9':
            self.valencies = [4, 3, 2, 1]
            self.bonds = [0, 1, 2, 3, 0]
        elif config.dataset == 'qm9H':
            self.valencies = [-1, 4, 3, 2, 1]
            self.bonds = [0, 1, 2, 3, 1.5, 0]

        assert self.k <= 20, 'k max for the spectral embeddings is 20! If you want more, change the code!'

    def __call__(self, X, A, mask, t=None):
        mask = mask.unsqueeze(-1)
        self.device = A.device
        if X is not None:
            if t is None:
                X_feat = torch.zeros(*X.shape[:-1], 0).to(self.device)
            else:
                X_feat = t
        A_feat = torch.zeros(A.shape[0], A.shape[1], A.shape[2], 0).to(self.device)
        if self.spectral_embeddings:
            eigen_feat, eig_vals = self.eigen_features_dense(A, mask)
            X_feat = torch.cat((X_feat, eigen_feat, eig_vals.repeat(1, mask.size(-2), 1)), dim=-1)
        if self.graph_size:
            n_nodes = mask.sum(-2, keepdim=True)/mask.size(1)
            X_feat = torch.cat((X_feat, n_nodes.repeat(1, mask.size(-2), 1)), dim=-1)
        if self.molecular_feat:
            charge, valencies = self.molecular_features(X, A)
            X_feat = torch.cat((X_feat, charge.unsqueeze(-1), valencies.unsqueeze(-1)), dim=-1)
        if self.cycles:
            if self.masking:
                edge = A[..., 1:-1].sum(-1)
            else:
                edge = A[..., 1:].sum(-1)
            x_cycles, _ = self.ncycles(edge)
            X_feat = torch.cat((X_feat, x_cycles), dim=-1)

        if self.rrwp:
            A_ = A[..., 1:].sum(-1)  # bs, n, n
            rrwp_edge_attr = self.get_rrwp(A_, k=self.rrwp_k)
            diag_index = torch.arange(rrwp_edge_attr.shape[1])
            rrwp_node_attr = rrwp_edge_attr[:, diag_index, diag_index, :]

            A_feat = torch.cat((A_feat, rrwp_edge_attr), dim=-1)
            X_feat = torch.cat((X_feat, rrwp_node_attr), dim=-1)

        X = torch.cat((X, X_feat), dim=-1) if X is not None else X_feat
        A = torch.cat((A, A_feat), dim=-1)
        return X, A

    def get_rrwp(self, A, k=10):
        """
        A : Adjacency matrix (bs, n, n)
        k : number of steps for the random walk
        returns:
            rrwp_edge_attr : (bs, n, n, k) -- edge features corresponding to the random walk probabilities up to step k
        """
        bs, n, _ = A.shape

        degree = torch.zeros(bs, n, n, device=A.device)
        to_fill = 1 / (A.sum(dim=-1).float())
        to_fill[A.sum(dim=-1).float() == 0] = 0
        degree = torch.diagonal_scatter(degree, to_fill, dim1=1, dim2=2)
        A = degree @ A

        id = torch.eye(n, device=A.device).unsqueeze(0).repeat(bs, 1, 1)
        rrwp_list = [id]

        for i in range(k - 1):
            cur_rrwp = rrwp_list[-1] @ A
            rrwp_list.append(cur_rrwp)

        return torch.stack(rrwp_list, -1)

    def eigen_features_dense(self, adjs, mask):

        if self.masking:
            adjs = adjs[..., 1:-1].sum(-1)
        else:
            adjs = adjs[..., 1:].sum(-1)

        degrees = adjs.sum(dim=-1)
        degrees = torch.diag_embed(degrees)
        # symmetric normalized Laplacian
        degrees[degrees != 0] = 1 / degrees[degrees != 0].sqrt()
        lap = (degrees != 0) * 1. - (degrees @ adjs @ degrees)

        try:
            eigvals, eigvectors = torch.linalg.eigh(lap.float())
        except:
            print('linalg.eigh failed... trying linalg.eig')
            try:
                eigvals, eigvectors = torch.linalg.eig(lap.float()).float()
            except:
                print('linalg.eig also failed... eig vect. replaced by zeros')
                n_0 = lap.shape[-1]
                eigvectors = torch.zeros((1, n_0, n_0), device=lap.device).float()
                eigvals = torch.zeros((1, n_0), device=lap.device).float()

        mask = mask.squeeze()
        eigvectors = eigvectors * mask.unsqueeze(2) * mask.unsqueeze(1)
        n_connected_comp, eigvals = self.get_eigenvalues_features(eigenvalues=eigvals, k=self.k)

        # Retrieve eigenvectors features
        _, eigfeat = self.get_eigenvectors_features(vectors=eigvectors, node_mask=mask,
                                                         n_connected=n_connected_comp, k=self.k)
        return eigfeat, eigvals.unsqueeze(-2)

    def molecular_features(self, X, A):
        charges = self.charge_feature(X, A)
        valencies = self.valency_feature(A)
        return charges, valencies

    def charge_feature(self, X, A):
        bond_orders = torch.tensor(self.bonds, device=A.device).reshape(1, 1, 1, -1)
        bond_orders = bond_orders[..., :A.size(-1)]
        weighted_E = A * bond_orders  # (bs, n, n, de)
        current_valencies = weighted_E.argmax(dim=-1).sum(dim=-1)  # (bs, n)

        valencies = torch.tensor(self.valencies, device=X.device).reshape(1, 1, -1)
        X = X[..., :valencies.size(-1)] * valencies  # (bs, n, dx)
        normal_valencies = X.sum(-1)

        return (normal_valencies - current_valencies)

    def valency_feature(self, A):
        orders = torch.tensor(self.bonds, device=A.device).reshape(1, 1, 1, -1)
        orders = orders[..., :A.size(-1)]
        A = A * orders      # (bs, n, n, de)
        valencies = A.argmax(dim=-1).sum(dim=-1)    # (bs, n)
        return valencies

    def get_eigenvalues_features(self, eigenvalues, k=5):
        """
        values : eigenvalues -- (bs, n)
        node_mask: (bs, n)
        k: num of non zero eigenvalues to keep
        """
        ev = eigenvalues
        bs, n = ev.shape
        n_connected_components = (ev < 1e-5).sum(dim=-1)
        assert (n_connected_components > 0).all(), (n_connected_components, ev)

        to_extend = max(n_connected_components) + k - n
        if to_extend > 0:
            eigenvalues = torch.hstack((eigenvalues, 2 * torch.ones(bs, to_extend).type_as(eigenvalues)))
        indices = torch.arange(k).type_as(eigenvalues).long().unsqueeze(0) + n_connected_components.unsqueeze(1)
        first_k_ev = torch.gather(eigenvalues, dim=1, index=indices)
        return n_connected_components.unsqueeze(-1), first_k_ev

    def get_eigenvectors_features(self, vectors, node_mask, n_connected, k=2):
        """
        vectors (bs, n, n) : eigenvectors of Laplacian IN COLUMNS
        returns:
            not_lcc_indicator : indicator vectors of largest connected component (lcc) for each graph  -- (bs, n, 1)
            k_lowest_eigvec : k first eigenvectors for the largest connected component   -- (bs, n, k)
        """
        bs, n = vectors.size(0), vectors.size(1)

        # Create an indicator for the nodes outside the largest connected components
        first_ev = torch.round(vectors[:, :, 0], decimals=3) * node_mask  # bs, n
        # Add random value to the mask to prevent 0 from becoming the mode
        random = torch.randn(bs, n, device=node_mask.device) * (~node_mask)  # bs, n
        first_ev = first_ev + random
        most_common = torch.mode(first_ev, dim=1).values  # values: bs -- indices: bs
        mask = ~ (first_ev == most_common.unsqueeze(1))
        not_lcc_indicator = (mask * node_mask).unsqueeze(-1).float()

        # Get the eigenvectors corresponding to the first nonzero eigenvalues
        to_extend = max(n_connected) + k - n
        if to_extend > 0:
            vectors = torch.cat((vectors, torch.zeros(bs, n, to_extend).type_as(vectors)),
                                dim=2)  # bs, n , n + to_extend
        indices = torch.arange(k).long().unsqueeze(0).unsqueeze(0).to(self.device) + n_connected.unsqueeze(
            2)  # bs, 1, k
        indices = indices.expand(-1, n, -1)  # bs, n, k
        first_k_ev = torch.gather(vectors, dim=2, index=indices)  # bs, n, k
        first_k_ev = first_k_ev * node_mask.unsqueeze(2)

        return not_lcc_indicator, first_k_ev

class NodeCycleFeatures:
    def __init__(self):
        self.kcycles = KNodeCycles()

    def __call__(self, adj):
        x_cycles, y_cycles = self.kcycles.k_cycles(adj_matrix=adj)   # (bs, n_cycles)
        # Avoid large values when the graph is dense
        x_cycles = x_cycles / 10
        y_cycles = y_cycles / 10
        x_cycles[x_cycles > 1] = 1
        y_cycles[y_cycles > 1] = 1
        return x_cycles, y_cycles

class KNodeCycles:
    """ Builds cycle counts for each node in a graph.
    """

    def __init__(self):
        super().__init__()

    def calculate_kpowers(self):
        self.k1_matrix = self.adj_matrix.float()
        self.d = self.adj_matrix.sum(dim=-1)
        self.k2_matrix = self.k1_matrix @ self.adj_matrix.float()
        self.k3_matrix = self.k2_matrix @ self.adj_matrix.float()
        self.k4_matrix = self.k3_matrix @ self.adj_matrix.float()
        self.k5_matrix = self.k4_matrix @ self.adj_matrix.float()
        self.k6_matrix = self.k5_matrix @ self.adj_matrix.float()

    def k3_cycle(self):
        """ tr(A ** 3). """
        c3 = batch_diagonal(self.k3_matrix)
        return (c3 / 2).unsqueeze(-1).float(), (torch.sum(c3, dim=-1) / 6).unsqueeze(-1).float()

    def k4_cycle(self):
        diag_a4 = batch_diagonal(self.k4_matrix)
        c4 = diag_a4 - self.d * (self.d - 1) - (self.adj_matrix @ self.d.unsqueeze(-1)).sum(dim=-1)
        return (c4 / 2).unsqueeze(-1).float(), (torch.sum(c4, dim=-1) / 8).unsqueeze(-1).float()

    def k5_cycle(self):
        diag_a5 = batch_diagonal(self.k5_matrix)
        triangles = batch_diagonal(self.k3_matrix)
        c5 = diag_a5 - 2 * triangles * self.d - (self.adj_matrix @ triangles.unsqueeze(-1)).sum(dim=-1) + triangles
        return (c5 / 2).unsqueeze(-1).float(), (c5.sum(dim=-1) / 10).unsqueeze(-1).float()

    def k6_cycle(self):
        term_1_t = batch_trace(self.k6_matrix)
        term_2_t = batch_trace(self.k3_matrix ** 2)
        term3_t = torch.sum(self.adj_matrix * self.k2_matrix.pow(2), dim=[-2, -1])
        d_t4 = batch_diagonal(self.k2_matrix)
        a_4_t = batch_diagonal(self.k4_matrix)
        term_4_t = (d_t4 * a_4_t).sum(dim=-1)
        term_5_t = batch_trace(self.k4_matrix)
        term_6_t = batch_trace(self.k3_matrix)
        term_7_t = batch_diagonal(self.k2_matrix).pow(3).sum(-1)
        term8_t = torch.sum(self.k3_matrix, dim=[-2, -1])
        term9_t = batch_diagonal(self.k2_matrix).pow(2).sum(-1)
        term10_t = batch_trace(self.k2_matrix)

        c6_t = (term_1_t - 3 * term_2_t + 9 * term3_t - 6 * term_4_t + 6 * term_5_t - 4 * term_6_t + 4 * term_7_t +
                3 * term8_t - 12 * term9_t + 4 * term10_t)
        return None, (c6_t / 12).unsqueeze(-1).float()

    def k_cycles(self, adj_matrix, verbose=False):
        self.adj_matrix = adj_matrix
        self.calculate_kpowers()

        k3x, k3y = self.k3_cycle()
        assert (k3x >= -0.1).all()

        k4x, k4y = self.k4_cycle()
        assert (k4x >= -0.1).all()

        k5x, k5y = self.k5_cycle()
        assert (k5x >= -0.1).all(), k5x

        _, k6y = self.k6_cycle()
        assert (k6y >= -0.1).all()

        kcyclesx = torch.cat([k3x, k4x, k5x], dim=-1)
        kcyclesy = torch.cat([k3y, k4y, k5y, k6y], dim=-1)
        return kcyclesx, kcyclesy

def batch_trace(X):
    """
    Expect a matrix of shape B N N, returns the trace in shape B
    :param X:
    :return:
    """
    diag = torch.diagonal(X, dim1=-2, dim2=-1)
    trace = diag.sum(dim=-1)
    return trace


def batch_diagonal(X):
    """
    Extracts the diagonal from the last two dims of a tensor
    :param X:
    :return:
    """
    return torch.diagonal(X, dim1=-2, dim2=-1)

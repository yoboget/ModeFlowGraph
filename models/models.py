import torch.nn
import torch.nn as nn

from models.nn import DenseLayer2, DenseGraphNorm


class DenseGNN(nn.Module):
    def __init__(self, config, nnf_in, nef_in, nnf_out, nef_out, nhf, norm_out=False):
        super().__init__()
        self.sizes = (nnf_in, nef_in, nnf_out, nef_out, nhf)
        # num_modes > 1: M node / edge predictions and a graph-level mode head (see utils/modes.py)
        self.num_modes = config.model.get('num_modes', 1)

        self.normalization = config.model.normalization
        mlp_n_layers = config.model.mlp_n_layers
        mlp_hidden_size = nhf
        fact = config.model.edge_node_ratio

        self.linear_in_x = nn.Sequential(nn.Linear(nnf_in, nhf), nn.ReLU(), nn.Linear(nhf, nhf))
        self.linear_out_x = nn.Sequential(nn.Linear(nhf, nhf), nn.ReLU(), nn.Linear(nhf, nnf_out * self.num_modes))
        if self.num_modes > 1:
            self.mode_head = nn.Sequential(nn.Linear(nhf, nhf), nn.ReLU(), nn.Linear(nhf, self.num_modes))

        n_layers = config.model.n_layers
        layers = []
        attention_layers = []

        for layer in range(n_layers):
            layers.append(DenseLayer2(nhf, nhf, nhf, nhf, mlp_n_layers, mlp_hidden_size, fact,
                                   normalization=self.normalization))
            attention_layers.append(torch.nn.MultiheadAttention(nhf, 16, batch_first=True))

        self.linear_in_e = nn.Linear(nef_in, int(nhf / fact))
        self.linear_out_e = nn.Linear(int(nhf / fact), nef_out * self.num_modes)
        self.layers = nn.Sequential(*layers)
        self.attention_layers = nn.Sequential(*attention_layers)

        self.norm_out = norm_out
        if norm_out:
            if self.normalization == 'batch_norm':
                self.edge_norm = nn.BatchNorm1d(nef_out)
                self.node_norm = nn.BatchNorm1d(nhf)

            elif self.normalization == 'layer_norm':
                self.edge_norm = nn.LayerNorm(nef_out)
                self.node_norm = nn.LayerNorm(nhf)

            elif self.normalization == 'graph_norm':
                self.edge_norm = DenseGraphNorm(nef_out)
                self.node_norm = DenseGraphNorm(nhf)

    def forward(self, X, A, mask):
        X = self.linear_in_x(X)
        A = self.linear_in_e(A)

        for layer, attention_layer in zip(self.layers, self.attention_layers):
            X_new, A_new = layer(X, A, mask, skip_connection=True)
            X = X + X_new
            A = A + A_new
            X_new, _ = attention_layer(X, X, X, key_padding_mask=~mask)
            X = X + X_new
            if self.normalization == 'batch_norm':
                X = self.node_norm(X.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                X = self.node_norm(X)

        H = X
        X = self.linear_out_x(X)
        A = self.linear_out_e(A)
        A = (A + A.transpose(1, 2)) * 0.5
        if self.num_modes == 1:
            return X, A
        bs, n = X.shape[:2]
        # mode head on the mean of the node embeddings over the existing nodes
        m = mask.unsqueeze(-1).float()
        mode_logits = self.mode_head((H * m).sum(1) / m.sum(1).clamp_min(1))
        return mode_logits, X.view(bs, n, self.num_modes, -1), A.view(bs, n, n, self.num_modes, -1)

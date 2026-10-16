import torch
import torch.nn as nn

from utils.func import get_edge_mask


class Mlp(nn.Module):
    def __init__(self,
                 in_,
                 out_,
                 hidden_,
                 activation=nn.ReLU(),
                 dropout=0.0
                 ):
        super().__init__()
        n_layers = len(hidden_) - 1

        layers = [nn.Linear(in_, hidden_[0]), activation]
        for i in range(n_layers):
            layers.append(nn.Linear(hidden_[i], hidden_[i + 1]))
            layers.append(activation)
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_[-1], out_))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class DenseLayer2(nn.Module):
    def __init__(self, nnf_in, nef_in, nnf_out, nef_out, n_layers, hidden_size, fact, normalization=None):
        super(DenseLayer2, self).__init__()
        hidden = [hidden_size] * (n_layers-1)
        hidden_e = [int(hidden_size / fact)] * (n_layers-1)

        self.w_source = nn.Linear(nnf_in, int(nef_in/fact))
        self.w_target = nn.Linear(nnf_in, int(nef_in/fact))
        self.w_edge = nn.Linear(int(nef_in/fact), int(nef_in/fact))
        self.activation = nn.ReLU()

        self.mlp_a = Mlp(int(nef_in / fact), int(nef_out / fact), hidden_e)
        self.mlp_x = Mlp(int(nef_in / fact), nnf_out, hidden)
        self.mlp_n_node2 = Mlp(nnf_out, nnf_out, hidden)
        self.normalization = normalization

        if self.normalization == 'batch_norm':
            self.edge_norm = nn.BatchNorm1d(int(nef_out / fact))
            self.node_norm = nn.BatchNorm1d(nnf_out)
            self.node_norm2 = nn.BatchNorm1d(nnf_out)
        elif self.normalization == 'layer_norm':
            self.edge_norm = nn.LayerNorm(int(nef_out / fact))
            self.node_norm = nn.LayerNorm(nnf_out)
            self.node_norm2 = nn.LayerNorm(nnf_out)
        elif self.normalization == 'graph_norm':
            self.edge_norm = DenseGraphNorm(int(nef_out / fact))
            self.node_norm = DenseGraphNorm(nnf_out)
            self.node_norm2 = DenseGraphNorm(nnf_out)

    def forward(self, X, A, mask, skip_connection=False):
        batch_size, n, _ = X.size()

        # Compute messages for fully connected graph
        x_source = self.w_source(X)
        x_target = self.w_target(X)
        a = self.w_edge(A)

        a = x_source.unsqueeze(1) + x_target.unsqueeze(2) + a
        a = self.activation(a)

        # Apply MLP to concatenated features
        A = self.mlp_a(a)
        x_ij = self.mlp_x(a)
        edge_mask = get_edge_mask(mask)
        x_ij = x_ij * edge_mask.unsqueeze(-1)

        # Aggregate messages
        X_ = x_ij.sum(dim=2)  # Shape: (batch_size, n, nnf_out)

        # Normalize and apply MLP to node features
        if self.normalization:
            X_ = X_ + X
            if self.normalization == 'batch_norm':
                X_ = self.node_norm(X_.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                X_ = self.node_norm(X_)
        else:
            X_ = X_ + X

        X_ = self.mlp_n_node2(X_)

        X = X_ + X if skip_connection else X_

        if self.normalization is not None:
            A = self.edge_norm(A.view(batch_size * n * n, -1)).view(batch_size, n, n, -1)
            if self.normalization == 'batch_norm':
                X = self.node_norm2(X.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                X = self.node_norm2(X)

        return X, A


class DenseGraphNorm(nn.Module):
    r"""Applies GraphNorm for dense representations of shape
    (batch_size, num_max_nodes, d_features).

    The normalization follows:

    .. math::
        \mathbf{x}^{\prime} = \frac{\mathbf{x} - \alpha \odot \textrm{E}[\mathbf{x}]}
        {\sqrt{\textrm{Var}[\mathbf{x} - \alpha \odot \textrm{E}[\mathbf{x}]] + \epsilon}}
        \odot \gamma + \beta

    where :math:`\alpha` is a learnable scaling parameter for the mean,
    and :math:`\gamma, \beta` are affine transformation parameters.

    Args:
        in_channels (int): The number of input features per node.
        eps (float, optional): A small value added for numerical stability (default: 1e-5).
    """

    def __init__(self, in_channels: int, eps: float = 1e-5):
        super().__init__()
        self.in_channels = in_channels
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(in_channels))  # Scaling factor
        self.bias = nn.Parameter(torch.zeros(in_channels))  # Bias term
        self.mean_scale = nn.Parameter(torch.ones(in_channels))  # Learnable mean adjustment

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Forward pass of GraphNorm.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_max_nodes, d_features).

        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        batch_mean = x.mean(dim=1, keepdim=True)  # Mean across nodes per graph
        out = x - batch_mean * self.mean_scale  # Mean-centered with learnable scale

        batch_var = out.var(dim=1, unbiased=False, keepdim=True)  # Variance across nodes
        std = torch.sqrt(batch_var + self.eps)  # Standard deviation

        return self.weight * (out / std) + self.bias  # Normalize and scale

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channels})'

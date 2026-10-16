import networkx as nx

def get_networkx_from_dense(X, A, mask, x_min=None):
    graphs = []
    if X is not None:
        X.squeeze_()
        assert X.dim() == 2, 'The annotation matrix must be of shape (batch_size, n)'
        assert mask.size() == X.size(), 'The annotation matrix and the mask must be of same shape'
    assert A.dim() == 3, 'The adjacency matrix must be of shape (batch_size, n, n)'

    batch_size = A.size(0)
    for b in range(batch_size):
        G = nx.Graph()

        # Add nodes with annotations
        for i, node in enumerate(X[b, mask[b]]):
            if x_min is not None:
                node += x_min
            G.add_node(i, label=node.int().item())

        # Add edges with attributes
        for i in range(mask[b].sum()):
            for j in range(i + 1, mask[b].sum()):  # Only consider the upper triangle for undirected graphs
                if A[b, i, j] > 0:  # Check if there is an edge
                    G.add_edge(i, j, label=A[b, i, j].int().item())
                    G.add_edge(j, i, label=A[b, i, j].int().item())  # Add reverse for undirected

        G.remove_edges_from(nx.selfloop_edges(G))
        G.remove_nodes_from(list(nx.isolates(G)))
        graphs.append(G)
    return graphs

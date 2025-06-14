import os
import numpy as np
import torch
import networkx as nx


def compute_dist_matrix(data, data_name, save_dir, force_recompute=False):
    """
    Computes shortest path distance between each node pair in the Graph data once for the first time
    And stores the computed adjacency matrix for accessing it in the remaining runs
    """
    os.makedirs(save_dir, exist_ok=True)
    dist_path = os.path.join(save_dir, f"{data_name}_dist_matrix.npy")

    if os.path.exists(dist_path) and not force_recompute:
        print(f"Loading distance matrix from {dist_path}")
        dist_matrix = np.load(dist_path)
        return dist_matrix

    print("Computing distance matrix with Dijkstra's algorithm...")
    # Load number of nodes and edge indices from the data (precomputed and stored within read_datasets.py file)
    num_nodes = data["num_nodes"]
    edge_index = data["edge_index"].cpu().numpy()
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edges = list(zip(edge_index[0], edge_index[1]))
    G.add_edges_from(edges)

    # Shortest path distances between each node pair is computed using the networkx method all_pairs_shortest_path_length(...)
    lengths = dict(nx.all_pairs_shortest_path_length(G))
    dist_matrix = np.full((num_nodes, num_nodes), np.inf, dtype=np.float32)
    for i in range(num_nodes):
        for j, d in lengths[i].items():
            dist_matrix[i, j] = d

    np.save(dist_path, dist_matrix)
    print(f"Saved distance matrix to {dist_path}")
    return dist_matrix


def convert_to_sparse_matrix(dist_matrix):
    """
    Convert a dense distance matrix to a sparse matrix format.
    """
    # Convert inf values to 0 for sparse representation
    dist_matrix = np.where(np.isinf(dist_matrix), 0, dist_matrix)

    # Store this non-zero values in a sparse matrix format
    row_indices, col_indices = np.nonzero(dist_matrix)
    values = dist_matrix[row_indices, col_indices]

    sparse_matrix = torch.sparse_coo_tensor(
        indices=torch.tensor([row_indices, col_indices]),
        values=torch.tensor(values, dtype=torch.float32),
        size=dist_matrix.shape,
    )

    return sparse_matrix


def dist_matrix_to_topk_edges(dist_matrix, k=16):
    """
    Convert dense pairwise distance matrix to sparse edge_index and edge_attr format.
    Retains k nearest neighbors for each node.

    Returns: edge_index and edge_attr

    Example:
    # Store all edges in the graph
    edge_index = torch.tensor([
        [0, 1, 2],  # source nodes
        [1, 2, 0]   # target nodes
    ])

    # Store shortest distances corresponding to edges
    edge_attr = torch.tensor([1.0, 2.0, 3.0]) 
    """
    num_nodes = dist_matrix.shape[0]

    # Replace infs with a large number for sorting
    safe_dist_matrix = np.copy(dist_matrix)
    safe_dist_matrix[np.isinf(safe_dist_matrix)] = np.max(safe_dist_matrix[np.isfinite(safe_dist_matrix)]) + 1

    edge_sources = []
    edge_targets = []
    edge_distances = []

    for i in range(num_nodes):
        # Get indices of k smallest distances for node i (excluding self)
        dists = safe_dist_matrix[i]
        neighbors = np.argsort(dists)[:k+1]  # include self
        neighbors = neighbors[neighbors != i][:k]  # exclude self, ensure k

        for j in neighbors:
            edge_sources.append(i)
            edge_targets.append(j)
            edge_distances.append(dist_matrix[i, j])  # original distance (could be inf)

    edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long) # shape: [2, num_edges]
    edge_attr = torch.tensor(edge_distances, dtype=torch.float32) # shape: [num_edges]

    return edge_index, edge_attr
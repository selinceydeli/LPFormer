import os
import numpy as np
import torch
import networkx as nx

def compute_dist_matrix(data, data_name, save_dir, force_recompute=False):
    '''
    Computes shortest path distance between each node pair in the Graph data once for the first time
    And stores the computed adjacency matrix for accessing it in the remaining runs
    '''
    os.makedirs(save_dir, exist_ok=True)
    dist_path = os.path.join(save_dir, f"{data_name}_dist_matrix.npy")

    if os.path.exists(dist_path) and not force_recompute:
        print(f"Loading distance matrix from {dist_path}")
        dist_matrix = np.load(dist_path)
        return dist_matrix

    print("Computing distance matrix with Dijkstra's algorithm...")
    # Load number of nodes and edge indices from the data (precomputed and stored within read_datasets.py file)
    num_nodes = data['num_nodes']
    edge_index = data['edge_index'].cpu().numpy()
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


from torch_geometric.datasets import Planetoid
from typing import Literal
import numpy as np
import torch
from torch_geometric.utils import to_undirected, is_undirected
from sklearn.decomposition import PCA
from torch_geometric.data import Data
import copy
import networkx as nx

def is_graph_undirected(edge_index):
    """Check if the graph is undirected."""
    return is_undirected(edge_index)

def convert_to_undirected(edge_index):
    """Convert the graph to undirected format."""
    if not is_graph_undirected(edge_index):
        return to_undirected(edge_index)
    return edge_index

# def apply_pca_to_features(features, n_components=3):
#     """Apply PCA to reduce feature dimensionality."""
#     if n_components is None:
#         raise ValueError("n_components must not be None for PCA")
        
#     features_np = features.detach().cpu().numpy()
    
#     pca = PCA(n_components=n_components)
#     reduced_features = pca.fit_transform(features_np)
#     if np.count_nonzero(reduced_features) == 0:
#         print("WARNING: PCA resulted in all zero values")
#     reduced_tensor = torch.tensor(reduced_features, dtype=features.dtype, device=features.device)
    
#     if reduced_tensor.shape[1] != n_components:
#         raise ValueError(f"Expected {n_components} features after PCA but got {reduced_tensor.shape[1]}")
    
#     return reduced_tensor

# def normalize_feature_vectors(features, epsilon=1e-8):
#     """
#     Normalize feature vectors by scaling all vectors relative to the maximum norm,
#     ensuring the maximum norm is 1 while preserving relative scaling between vectors.
#     """
#     norms = torch.norm(features, p=2, dim=1)
#     max_norm = torch.max(norms) + epsilon
#     normalized_features = features / max_norm
    
#     return normalized_features

def grab_planetoid_data(dataset_name: Literal['Cora', 'CiteSeer', 'PubMed'], 
                        ensure_undirected=True):  
    """
    Retrieves the specified Planetoid dataset with options to reduce dimensionality
    of features using PCA and normalize feature vectors.
    """

    dataset = Planetoid(root=f'data/{dataset_name}', name=dataset_name)
    
    original_num_features = dataset.num_features
    processed_dataset = copy.deepcopy(dataset)
    
    if not hasattr(processed_dataset, '_data_list'):
        processed_dataset._data_list = [processed_dataset[i] for i in range(len(processed_dataset))]
    
    if ensure_undirected:
        for i in range(len(processed_dataset)):
            processed_dataset[i].edge_index = convert_to_undirected(processed_dataset[i].edge_index)
    

    final_dim = processed_dataset[0].x.shape[1]
    processed_dataset.__class__.num_features = property(lambda self: final_dim)
    
    return processed_dataset

def to_networkx_graph(data: Data) -> nx.Graph:
    """
    Convert a PyTorch Geometric Data object from a Planetoid dataset to a networkx Graph.
    """
    G = nx.Graph()
    
    num_nodes = data.x.size(0)
    features = data.x.cpu().numpy()
    labels = data.y.cpu().numpy()
    
    for i in range(num_nodes):
        node_attrs = {
            'coords': features[i],
            'comm': int(labels[i])
        }
        
        if hasattr(data, 'train_mask'):
            node_attrs['train_mask'] = bool(data.train_mask[i])
        if hasattr(data, 'val_mask'):
            node_attrs['val_mask'] = bool(data.val_mask[i])
        if hasattr(data, 'test_mask'):
            node_attrs['test_mask'] = bool(data.test_mask[i])
            
        G.add_node(i, **node_attrs)
    
    edge_list = data.edge_index.t().cpu().numpy()
    edges = list(map(tuple, edge_list))
    G.add_edges_from(edges)
    
    # Calculate Hamming distance between node feature vectors for each edge
    # First pass to find max distance
    max_dist = 0
    for u, v in G.edges():
        u_coords = G.nodes[u]['coords']
        v_coords = G.nodes[v]['coords']
        hamming_dist = np.sum(u_coords != v_coords)
        max_dist = max(max_dist, hamming_dist)
    
    # Second pass to normalize distances
    for u, v in G.edges():
        u_coords = G.nodes[u]['coords']
        v_coords = G.nodes[v]['coords'] 
        hamming_dist = np.sum(u_coords != v_coords)
        G.edges[u, v]['dist'] = 2.0 * float(hamming_dist) / max_dist
    G.graph['num_classes'] = len(np.unique(labels))
    
    return G

if __name__ == "__main__":
    dataset_name = 'Cora'  # Example dataset name
    dataset = grab_planetoid_data(dataset_name, ensure_undirected=True)
    graph = to_networkx_graph(dataset[0])
    # Print 3 example edges with their attributes
    print("\nExample edges from the graph:")
    for i, (u, v) in enumerate(list(graph.edges())[:3]):
        print(f"Edge {i+1}: ({u}, {v})")
        print(f"  Distance: {graph.edges[u,v]['dist']}")



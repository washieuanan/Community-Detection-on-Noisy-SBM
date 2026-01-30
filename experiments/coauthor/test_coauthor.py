import torch
import networkx as nx
from torch_geometric.datasets import Coauthor
from torch_geometric.datasets import KarateClub
import numpy as np
from torch_geometric.transforms import ToUndirected
from algorithms.bp.vectorized_bp import (
    get_true_communities,
    belief_propagation,
    belief_propagation_weighted,
    detection_stats
)
from algorithms.duo_spec import duo_spec
from algorithms.spectral_ops.attention import motif_spectral_embedding

def load_coauthor(name = 'Physics'):
    """
    Load the twitch dataset and convert it to a NetworkX graph.
    """
    # Load the dataset
    dataset = Coauthor(root='data/Coauthor', name=name)
    num_graphs = len(dataset)
    print(f"Number of graphs: {num_graphs}")
    data = dataset[0]
    # Convert to undirected graph
    
    # Create NetworkX graph
    edge_index = data.edge_index.numpy()
    G = nx.Graph()
    
    # Add nodes with community labels and features
    for i in range(data.num_nodes):
        G.add_node(i, 
                   comm=int(data.y[i].item()),
                   coords=data.x[i].numpy())
        
    # Add edges
    # Calculate distances and add edges with distance attributes
    edges = list(zip(edge_index[0], edge_index[1]))
    dists = []
    for u, v in edges:
        # Get coordinates for nodes
        coord_u = G.nodes[u]['coords'] 
        coord_v = G.nodes[v]['coords']
        # Calculate cosine similarity
        similarity = np.dot(coord_u, coord_v) / (np.linalg.norm(coord_u) * np.linalg.norm(coord_v))
        # Convert similarity to distance (0 similarity -> dist 2, 1 similarity -> dist 0)
        dist = 2 * (1 - similarity)
        dists.append(dist)
    
    dists = np.array(dists)
    
    # Add edges with distances
    G.add_edges_from([(u,v,{'dist':d}) for (u,v),d in zip(edges,dists)])
    return G

# def load_karate():
#     dataset = KarateClub()
#     num_graphs = len(dataset)
#     print(f"Number of graphs: {num_graphs}")
#     data = dataset[0]
#     # Convert to undirected graph

#     # Create NetworkX graph
#     edge_index = data.edge_index.numpy()
#     G = nx.Graph()

#     # Add nodes with community labels and features
#     for i in range(data.num_nodes):
#         G.add_node(i, 
#                    comm=int(data.y[i].item()),
#                    coords=data.x[i].numpy())
    
#     # Add edges
#     # Calculate distances and add edges with distance attributes
#     edges = list(zip(edge_index[0], edge_index[1]))
#     dists = []
#     for u, v in edges:
#         # Get coordinates for nodes
#         coord_u = G.nodes[u]['coords'] 
#         coord_v = G.nodes[v]['coords']
#         # Calculate cosine similarity
#         similarity = np.dot(coord_u, coord_v) / (np.linalg.norm(coord_u) * np.linalg.norm(coord_v))
#         # Convert similarity to distance (0 similarity -> dist 2, 1 similarity -> dist 0)
#         dist = 2 * (1 - similarity)
#         dists.append(dist)

#     dists = np.array(dists)

#     # Add edges with distances
#     G.add_edges_from([(u,v,{'dist':d}) for (u,v),d in zip(edges,dists)])
#     return G
if __name__ == "__main__":
    # G = load_karate()
    G = load_coauthor(name='Physics')
    num_comms = len(np.unique(G.nodes(data=True)[0]['comm']))
    true_labels = get_true_communities(G, node2idx=None, attr="comm")
    _, preds, _, _ = belief_propagation_weighted(
        G,
        q=num_comms,
        seed=0,
        init="spectral",
    )
    stats = detection_stats(preds, true_labels)
    print("\n=== BP Accuracy ===")
    for k, v in stats.items():
        print(f"{k:>25s} : {v}")
    
    res_duo = duo_spec(
        G,
        K=num_comms,
        max_em_iters=50,
        community_proxy="leiden"
    )
    
    G_res = res_duo['G_final']
    _, preds, _, _ = belief_propagation_weighted(
                                            G_res, 
                                            q=num_comms, 
                                            seed=0, 
                                            init="spectral"
                                                )
    stats = detection_stats(preds, true_labels)
    print("\n=== Post-Duospec BP ===")
    for k, v in stats.items():
        print(f"{k:>25s} : {v}")

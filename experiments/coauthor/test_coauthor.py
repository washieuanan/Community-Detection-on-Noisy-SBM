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

# def load_twitch(name = 'Physics'):
#     """
#     Load the twitch dataset and convert it to a NetworkX graph.
#     """
#     # Load the dataset
#     dataset = Coauthor(root='data/Coauthor', name=name)
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

def load_karate():
    dataset = KarateClub()
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
if __name__ == "__main__":
    G = load_karate()
    # Print example node with its attributes
    # Print first 5 nodes with their attributes
    example_nodes = list(G.nodes(data=True))[:5]
    print("\nExample nodes:")
    for node in example_nodes:
        print(node)

    # Print example edge with its attributes 
    example_edge = list(G.edges(data=True))[0]
    print("\nExample edge:", example_edge)
    duo_params = dict(
        K               = 4,                    # number of communities (2 for polblogs)
        num_balls       = 6,                   # finer geometry embedding
        config          = "motif",              # geometry estimator

        # — EM schedule
        max_em_iters    = 100,                  # allow more EM steps
        warmup_rounds   = 2,                    # hold off on any re-weighting
        anneal_steps    = 20,                   # then ramp λ from 0→full over 30 iter
        boost_geo       = 0.0,
        # — convergence
        tol             = 1e-5,
        patience        = 10,
        random_state    = 42,
        spec_params     = dict(
            dim = 1,
            walk_len = 2,
            num_walks = 2,
        )
    )
    true_labels = get_true_communities(G, node2idx=None, attr="comm")
    _, preds, _, _ = motif_spectral_embedding(G, q=4, **duo_params['spec_params'])
    stats = detection_stats(preds, true_labels)
    print("\n=== Community‑detection accuracy MotifSpectralEmbedding ===")
    for k, v in stats.items():
        print(f"{k:>25s} : {v}")
        
    # # Run DuoSpec
    res = duo_spec(G, **duo_params)
    G_fin = res["G_final"]
    preds = res['communities']
    true_labels = get_true_communities(G, node2idx=None, attr="comm")
    
    # Print DuoSpec results
    stats = detection_stats(preds, true_labels)
    print("\n=== Community‑detection accuracy DuoSpec ===")
    for k, v in stats.items():
        print(f"{k:>25s} : {v}")
    
    # Run DuoSpec + BP
    _, preds, _, _ = belief_propagation_weighted(G_fin, q=4)
    stats = detection_stats(preds, true_labels)
    print("\n=== Community‑detection accuracy DuoSpec-BP ===")
    for k, v in stats.items():
        print(f"{k:>25s} : {v}")
    
    _, preds, _, _ = belief_propagation(G, q=4)
    stats = detection_stats(preds, true_labels)
    print("\n=== Community‑detection accuracy BP ===")
    for k, v in stats.items():
        print(f"{k:>25s} : {v}")

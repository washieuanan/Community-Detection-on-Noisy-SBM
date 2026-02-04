import torch
import networkx as nx
from torch_geometric.datasets import Coauthor
from torch_geometric.datasets import KarateClub
import numpy as np
from torch_geometric.transforms import ToUndirected
import csv
from collections import defaultdict
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
def compute_graph_stats(G, true_labels):
    """Compute graph statistics."""
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    
    # Nodes per community
    unique_comms, counts = np.unique(true_labels, return_counts=True)
    nodes_per_comm = dict(zip(unique_comms, counts))
    nodes_per_comm_str = ",".join([f"{k}:{v}" for k, v in sorted(nodes_per_comm.items())])
    
    # Average degree
    if num_nodes > 0:
        avg_degree = 2.0 * num_edges / num_nodes
    else:
        avg_degree = 0.0
    
    # Clustering coefficient
    try:
        clustering_coeff = nx.average_clustering(G)
    except:
        clustering_coeff = float('nan')
    
    return {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'nodes_per_community': nodes_per_comm_str,
        'average_degree': avg_degree,
        'clustering_coefficient': clustering_coeff,
    }

if __name__ == "__main__":
    # G = load_karate()
    G = load_coauthor(name='Physics')
    true_labels = get_true_communities(G, node2idx=None, attr="comm")
    
    # ========================================================================
    # FILTER TO TOP 2 COMMUNITIES BY SIZE (comment out to use all communities)
    # ========================================================================
    unique_comms, counts = np.unique(true_labels, return_counts=True)
    top2_indices = np.argsort(counts)[-2:]  # Top 2 largest
    top2_comms = unique_comms[top2_indices]
    print(f"Original communities: {len(unique_comms)}, sizes: {dict(zip(unique_comms, counts))}")
    print(f"Filtering to top 2 communities: {top2_comms} with sizes: {counts[top2_indices]}")
    
    # Filter nodes to only those in top 2 communities
    nodes_to_keep = [n for n in G.nodes() if G.nodes[n]["comm"] in top2_comms]
    G = G.subgraph(nodes_to_keep).copy()
    
    # Relabel communities to 0 and 1
    comm_mapping = {int(top2_comms[0]): 0, int(top2_comms[1]): 1}
    for n in G.nodes():
        G.nodes[n]["comm"] = comm_mapping[G.nodes[n]["comm"]]
    
    true_labels = get_true_communities(G, node2idx=None, attr="comm")
    num_comms = 2
    print(f"After filtering: {len(G.nodes())} nodes, {len(G.edges())} edges, {num_comms} communities")
    # ========================================================================
    
    # Compute graph statistics
    graph_stats = compute_graph_stats(G, true_labels)
    
    # Pre-denoise BP
    _, preds_pre, _, _ = belief_propagation_weighted(
        G,
        q=num_comms,
        seed=0,
        init="bethe_hessian",
    )
    stats_pre = detection_stats(preds_pre, true_labels)
    print("\n=== BP Accuracy (Pre-Denoise) ===")
    for k, v in stats_pre.items():
        print(f"{k:>25s} : {v}")
    
    # Denoise
    res_duo = duo_spec(
        G,
        K=num_comms,
        max_em_iters=20,
        community_proxy="leiden",
        local_score="ra"
    )
    
    # Post-denoise BP
    G_res = res_duo['G_final']
    _, preds_post, _, _ = belief_propagation_weighted(
                                            G_res, 
                                            q=num_comms, 
                                            seed=0, 
                                            init="bethe_hessian"
                                                )
    stats_post = detection_stats(preds_post, true_labels)
    print("\n=== Post-Duospec BP ===")
    for k, v in stats_post.items():
        print(f"{k:>25s} : {v}")
    
    # Save to CSV
    csv_file = "experiments/coauthor/coauthor_results.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'num_nodes', 'num_edges', 'nodes_per_community', 'average_degree', 'clustering_coefficient',
            'pre_accuracy', 'pre_num_vertices', 'pre_num_communities_predicted', 'pre_perm_p',
            'pre_accuracy_0', 'pre_accuracy_1',
            'post_accuracy', 'post_num_vertices', 'post_num_communities_predicted', 'post_perm_p',
            'post_accuracy_0', 'post_accuracy_1',
        ])
        writer.writeheader()
        
        row = graph_stats.copy()
        # Pre-denoise metrics
        row['pre_accuracy'] = stats_pre.get('accuracy', float('nan'))
        row['pre_num_vertices'] = stats_pre.get('num vertices', float('nan'))
        row['pre_num_communities_predicted'] = stats_pre.get('num communities predicted', float('nan'))
        row['pre_perm_p'] = stats_pre.get('perm_p', float('nan'))
        row['pre_accuracy_0'] = stats_pre.get('accuracy_0', float('nan'))
        row['pre_accuracy_1'] = stats_pre.get('accuracy_1', float('nan'))
        # Post-denoise metrics
        row['post_accuracy'] = stats_post.get('accuracy', float('nan'))
        row['post_num_vertices'] = stats_post.get('num vertices', float('nan'))
        row['post_num_communities_predicted'] = stats_post.get('num communities predicted', float('nan'))
        row['post_perm_p'] = stats_post.get('perm_p', float('nan'))
        row['post_accuracy_0'] = stats_post.get('accuracy_0', float('nan'))
        row['post_accuracy_1'] = stats_post.get('accuracy_1', float('nan'))
        
        writer.writerow(row)
    
    print(f"\nResults saved to {csv_file}")
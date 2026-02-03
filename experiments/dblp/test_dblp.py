import graph_tool.all as gt
import networkx as nx
import numpy as np
import csv
from algorithms.bp.vectorized_bp import (
    get_true_communities,
    belief_propagation,
    belief_propagation_weighted,
    detection_stats
)
from algorithms.duo_spec import duo_spec

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
    G = nx.read_gml("experiments/dblp/dblp.gml")
    for u, v, d in G.edges(data=True):
        if "weight" not in d:
            d["weight"] = 1.0

    # Rename the 'value' node attribute to 'comm'

    # Save the fixed graph back to file
    print(f"Fixed and saved graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, undirected={not G.is_directed()}")

    true_labels = get_true_communities(G, node2idx=None, attr="comm")
    num_comms = len(np.unique(true_labels))

    # Compute graph statistics
    graph_stats = compute_graph_stats(G, true_labels)

    # Pre-denoise BP
    _, preds_pre, _, _ = belief_propagation_weighted(
        G,
        q=num_comms,
        seed=0,
        init="spectral",
    )
    stats_pre = detection_stats(preds_pre, true_labels)
    print("\n=== BP Accuracy (Pre-Denoise) ===")
    for k, v in stats_pre.items():
        print(f"{k:>25s} : {v}")

    # Denoise
    res_duo = duo_spec(
        G,
        K=num_comms,
        max_em_iters=50,
        community_proxy="leiden"
    )

    # Post-denoise BP
    G_res = res_duo['G_final']
    _, preds_post, _, _ = belief_propagation_weighted(
                                        G_res, 
                                        q=num_comms, 
                                        seed=0, 
                                        init="spectral"
                                            )
    stats_post = detection_stats(preds_post, true_labels)
    print("\n=== Post-Duospec BP ===")
    for k, v in stats_post.items():
        print(f"{k:>25s} : {v}")

    # Save to CSV
    csv_file = "experiments/dblp/dblp_results.csv"
    with open(csv_file, 'w', newline='') as f:
        # Collect all possible accuracy keys (handle variable number of communities)
        all_keys = set(stats_pre.keys()) | set(stats_post.keys())
        fieldnames = ['num_nodes', 'num_edges', 'nodes_per_community', 'average_degree', 'clustering_coefficient']
        for key in sorted(all_keys):
            if key not in ['num vertices', 'num communities predicted']:  # These are handled separately
                fieldnames.append(f'pre_{key}')
                fieldnames.append(f'post_{key}')
        fieldnames.extend(['pre_num_vertices', 'pre_num_communities_predicted', 'pre_perm_p',
                          'post_num_vertices', 'post_num_communities_predicted', 'post_perm_p'])
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        row = graph_stats.copy()
        # Pre-denoise metrics
        for key in all_keys:
            if key not in ['num vertices', 'num communities predicted']:
                row[f'pre_{key}'] = stats_pre.get(key, float('nan'))
        row['pre_num_vertices'] = stats_pre.get('num vertices', float('nan'))
        row['pre_num_communities_predicted'] = stats_pre.get('num communities predicted', float('nan'))
        row['pre_perm_p'] = stats_pre.get('perm_p', float('nan'))
        # Post-denoise metrics
        for key in all_keys:
            if key not in ['num vertices', 'num communities predicted']:
                row[f'post_{key}'] = stats_post.get(key, float('nan'))
        row['post_num_vertices'] = stats_post.get('num vertices', float('nan'))
        row['post_num_communities_predicted'] = stats_post.get('num communities predicted', float('nan'))
        row['post_perm_p'] = stats_post.get('perm_p', float('nan'))
        
        writer.writerow(row)

    print(f"\nResults saved to {csv_file}")

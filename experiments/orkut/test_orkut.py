# import gzip
# import networkx as nx
# import numpy as np
# from collections import defaultdict

# def load_orkut_graph():
#     """Load Orkut graph from edge list and community files."""
#     G = nx.Graph()
    
#     # Load edge list
#     print("Loading edge list from com-orkut.ungraph.txt.gz...")
#     with gzip.open("experiments/orkut/com-orkut.ungraph.txt.gz", "rt") as f:
#         for line in f:
#             line = line.strip()
#             if not line or line.startswith("#"):
#                 continue
#             parts = line.split("\t")
#             if len(parts) >= 2:
#                 u = int(parts[0])
#                 v = int(parts[1])
#                 G.add_edge(u, v)
    
#     print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
#     # Load community assignments
#     print("Loading community assignments from com-orkut.top5000.cmty.txt.gz...")
#     node_to_comm = {}
#     comm_to_nodes = defaultdict(list)
    
#     with gzip.open("experiments/orkut/com-orkut.top5000.cmty.txt.gz", "rt") as f:
#         comm_id = 0
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             nodes = [int(x) for x in line.split()]
#             for node in nodes:
#                 if node in G:  # Only assign if node exists in graph
#                     node_to_comm[node] = comm_id
#                     comm_to_nodes[comm_id].append(node)
#             comm_id += 1
    
#     print(f"Loaded {comm_id} communities")
    
#     # Set community attribute
#     for node in G.nodes():
#         if node in node_to_comm:
#             G.nodes[node]["comm"] = node_to_comm[node]
#         else:
#             # Nodes not in any community get a special label
#             G.nodes[node]["comm"] = -1
    
#     return G, comm_to_nodes

# def filter_to_top_communities(G, comm_to_nodes, top_k=2):
#     """Filter graph to only nodes in the top K largest communities."""
#     # Find top K communities by size
#     comm_sizes = [(comm_id, len(nodes)) for comm_id, nodes in comm_to_nodes.items()]
#     comm_sizes.sort(key=lambda x: -x[1])  # Sort descending by size
    
#     print(f"\nCommunity sizes (top 10):")
#     for comm_id, size in comm_sizes[:10]:
#         print(f"  Community {comm_id}: {size} nodes")
    
#     top_comm_ids = {comm_id for comm_id, _ in comm_sizes[:top_k]}
#     print(f"\nKeeping top {top_k} communities: {sorted(top_comm_ids)}")
    
#     # Get all nodes in top communities and ensure they have correct community labels
#     nodes_to_keep = set()
#     node_to_top_comm = {}  # Map node to its top community (in case of overlaps)
#     for comm_id in top_comm_ids:
#         for node in comm_to_nodes[comm_id]:
#             nodes_to_keep.add(node)
#             # If node appears in multiple top communities, use the first one we encounter
#             if node not in node_to_top_comm:
#                 node_to_top_comm[node] = comm_id
    
#     print(f"Nodes in top {top_k} communities: {len(nodes_to_keep)}")
    
#     # Create subgraph
#     G_filtered = G.subgraph(nodes_to_keep).copy()
    
#     # Relabel communities to 0 and 1 using the correct community assignment
#     comm_mapping = {comm_id: idx for idx, comm_id in enumerate(sorted(top_comm_ids))}
#     for node in G_filtered.nodes():
#         # Use the top community assignment we determined, not the node's current comm attribute
#         # (which might be from a different community if the node appeared in multiple)
#         if node in node_to_top_comm:
#             top_comm = node_to_top_comm[node]
#             G_filtered.nodes[node]["comm"] = comm_mapping[top_comm]
#         else:
#             # This shouldn't happen, but handle it
#             print(f"Warning: Node {node} not found in node_to_top_comm. Removing from graph.")
#             G_filtered.remove_node(node)
    
#     print(f"Filtered graph: {G_filtered.number_of_nodes()} nodes, {G_filtered.number_of_edges()} edges")
    
#     # Verify community distribution
#     comm_counts = defaultdict(int)
#     for node in G_filtered.nodes():
#         comm_counts[G_filtered.nodes[node]["comm"]] += 1
#     print(f"Community distribution after relabeling:")
#     for comm_id in sorted(comm_counts.keys()):
#         print(f"  Community {comm_id}: {comm_counts[comm_id]} nodes")
    
#     return G_filtered

# if __name__ == "__main__":
#     # Load graph and communities
#     G, comm_to_nodes = load_orkut_graph()
    
#     # Filter to top 2 communities
#     G_filtered = filter_to_top_communities(G, comm_to_nodes, top_k=2)
    
#     # Ensure graph is undirected
#     if G_filtered.is_directed():
#         G_filtered = G_filtered.to_undirected()
    
#     # Ensure all edges have weights (default to 1.0)
#     for u, v, d in G_filtered.edges(data=True):
#         if "weight" not in d:
#             d["weight"] = 1.0
    
#     # Save to GML
#     output_file = "experiments/orkut/orkut.gml"
#     print(f"\nSaving graph to {output_file}...")
#     nx.write_gml(G_filtered, output_file)
#     print(f"Saved: {G_filtered.number_of_nodes()} nodes, {G_filtered.number_of_edges()} edges")
#     print(f"Graph is undirected: {not G_filtered.is_directed()}")
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
    G = nx.read_gml("experiments/orkut/orkut.gml")  # returns a NetworkX graph
    
    
    # Ensure all edges have weights (default to 1.0)
    for u, v, d in G.edges(data=True):
        if "weight" not in d:
            d["weight"] = 1.0
        
    
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
    csv_file = "experiments/orkut/orkut_results.csv"
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

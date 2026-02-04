# import gzip
# import networkx as nx
# import numpy as np
# from collections import defaultdict

# def load_amazon_graph():
#     """Load Amazon graph from edge list and community files."""
#     G = nx.Graph()
    
#     # Load edge list
#     print("Loading edge list from com-amazon.ungraph.txt.gz...")
#     with gzip.open("experiments/amazon_snap/com-amazon.ungraph.txt.gz", "rt") as f:
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
#     print("Loading community assignments from com-amazon.top5000.cmty.txt.gz...")
#     node_to_comms = defaultdict(set)  # Track ALL communities each node appears in
#     comm_to_nodes = defaultdict(list)
    
#     with gzip.open("experiments/amazon_snap/com-amazon.top5000.cmty.txt.gz", "rt") as f:
#         comm_id = 0
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             nodes = [int(x) for x in line.split()]
#             for node in nodes:
#                 if node in G:  # Only assign if node exists in graph
#                     node_to_comms[node].add(comm_id)
#                     comm_to_nodes[comm_id].append(node)
#             comm_id += 1
    
#     print(f"Loaded {comm_id} communities")
    
#     # Report on multi-community assignments
#     multi_comm_nodes = {node: comms for node, comms in node_to_comms.items() if len(comms) > 1}
#     print(f"Nodes appearing in multiple communities: {len(multi_comm_nodes)} / {len(node_to_comms)}")
#     if len(multi_comm_nodes) > 0:
#         max_comms = max(len(comms) for comms in multi_comm_nodes.values())
#         print(f"  Maximum communities per node: {max_comms}")
    
#     # For the graph, assign each node to its LAST community (original behavior)
#     # This is just for the graph attribute - we'll use comm_to_nodes for filtering
#     node_to_comm = {}
#     for node, comms in node_to_comms.items():
#         node_to_comm[node] = max(comms)  # Use highest community ID (last one encountered)
    
#     # Set community attribute
#     for node in G.nodes():
#         if node in node_to_comm:
#             G.nodes[node]["comm"] = node_to_comm[node]
#         else:
#             # Nodes not in any community get a special label
#             G.nodes[node]["comm"] = -1
    
#     return G, comm_to_nodes

# def merge_overlapping_communities(comm_to_nodes, overlap_threshold=0.8):
#     """
#     Iteratively merge communities that have strong overlap until no more merges can be made.
    
#     Args:
#         comm_to_nodes: dict mapping community ID to list of nodes
#         overlap_threshold: Jaccard similarity threshold for merging (0.0 to 1.0)
    
#     Returns:
#         merged_comm_to_nodes: dict mapping merged community ID to set of nodes
#         comm_to_merged: dict mapping original community ID to merged community ID
#     """
#     print(f"\nMerging communities with overlap threshold (Jaccard) >= {overlap_threshold}...")
    
#     # Convert to sets for faster operations
#     comm_to_set = {comm_id: set(nodes) for comm_id, nodes in comm_to_nodes.items()}
    
#     # Use union-find (DSU) to track which communities should be merged
#     # Each community starts as its own group
#     parent = {comm_id: comm_id for comm_id in comm_to_set.keys()}
    
#     def find(x):
#         """Find root of x with path compression."""
#         if parent[x] != x:
#             parent[x] = find(parent[x])
#         return parent[x]
    
#     def union(x, y):
#         """Union two communities, merging smaller into larger."""
#         root_x = find(x)
#         root_y = find(y)
#         if root_x == root_y:
#             return False
        
#         # Merge smaller into larger (by node count)
#         if len(comm_to_set[root_x]) < len(comm_to_set[root_y]):
#             root_x, root_y = root_y, root_x
        
#         parent[root_y] = root_x
#         # Merge the node sets
#         comm_to_set[root_x].update(comm_to_set[root_y])
#         return True
    
#     # Iteratively merge until no more merges can be made
#     comm_ids = list(comm_to_set.keys())
#     merges_made = 0
#     max_iterations = 10  # Safety limit
#     iteration = 0
    
#     while iteration < max_iterations:
#         iteration += 1
#         iteration_merges = 0
        
#         # Check all pairs of root communities (not original communities, since they may have merged)
#         root_communities = set(find(cid) for cid in comm_ids)
#         root_list = list(root_communities)
        
#         for i in range(len(root_list)):
#             for j in range(i + 1, len(root_list)):
#                 root_i = root_list[i]
#                 root_j = root_list[j]
                
#                 # Skip if already in same group (shouldn't happen, but check anyway)
#                 if find(root_i) == find(root_j):
#                     continue
                
#                 # Get current merged sets (includes all communities in each group)
#                 set_i = comm_to_set[root_i]
#                 set_j = comm_to_set[root_j]
                
#                 # Calculate Jaccard similarity
#                 intersection = len(set_i & set_j)
#                 union_size = len(set_i | set_j)
#                 if union_size > 0:
#                     jaccard = intersection / union_size
#                     if jaccard >= overlap_threshold:
#                         if union(root_i, root_j):
#                             iteration_merges += 1
#                             merges_made += 1
#                             # Update the root's set to include the merged set
#                             new_root = find(root_i)
#                             comm_to_set[new_root].update(set_j)
#                             if merges_made <= 20:  # Limit print output
#                                 print(f"  Iteration {iteration}: Merged groups containing {root_i} and {root_j} (Jaccard={jaccard:.3f})")
        
#         if iteration_merges == 0:
#             break  # No more merges possible
    
#     if merges_made > 20:
#         print(f"  ... and {merges_made - 20} more merges")
#     print(f"  Completed in {iteration} iteration(s)")
    
#     # Build merged communities from union-find groups
#     merged_groups = defaultdict(list)
#     for comm_id in comm_ids:
#         root = find(comm_id)
#         merged_groups[root].append(comm_id)
    
#     # Create final merged communities
#     merged_comm_to_nodes = {}
#     comm_to_merged = {}
#     merged_id_counter = 0
    
#     # Sort merged groups by total size (largest first)
#     merged_group_sizes = [(root, sum(len(comm_to_nodes[c]) for c in group)) 
#                           for root, group in merged_groups.items()]
#     merged_group_sizes.sort(key=lambda x: -x[1])
    
#     for root, _ in merged_group_sizes:
#         merged_id = merged_id_counter
#         merged_id_counter += 1
        
#         # Collect all nodes from all communities in this group
#         merged_set = set()
#         for orig_comm_id in merged_groups[root]:
#             merged_set.update(comm_to_nodes[orig_comm_id])
#             comm_to_merged[orig_comm_id] = merged_id
        
#         merged_comm_to_nodes[merged_id] = merged_set
    
#     print(f"  Original communities: {len(comm_to_nodes)}")
#     print(f"  Merged communities: {len(merged_comm_to_nodes)}")
#     print(f"  Total merges made: {merges_made} (over {iteration} iteration(s))")
    
#     return merged_comm_to_nodes, comm_to_merged

# def filter_to_top_communities(G, comm_to_nodes, top_k=2, merge_overlap_threshold=0.8):
#     """Filter graph to only nodes in the top K largest communities (after merging overlapping ones)."""
#     # First, merge overlapping communities
#     merged_comm_to_nodes, comm_to_merged = merge_overlapping_communities(
#         comm_to_nodes, overlap_threshold=merge_overlap_threshold
#     )
    
#     # Find top K merged communities by size
#     merged_sizes = [(merged_id, len(nodes)) for merged_id, nodes in merged_comm_to_nodes.items()]
#     merged_sizes.sort(key=lambda x: -x[1])  # Sort descending by size
    
#     print(f"\nMerged community sizes (top 10):")
#     for merged_id, size in merged_sizes[:10]:
#         print(f"  Merged community {merged_id}: {size} nodes")
    
#     # Select top K merged communities
#     top_merged_ids = [merged_id for merged_id, _ in merged_sizes[:top_k]]
#     print(f"\nKeeping top {len(top_merged_ids)} merged communities: {top_merged_ids}")
    
#     # Get all nodes in top merged communities
#     nodes_to_keep = set()
#     node_to_comm = {}  # Map node to its community label (0 or 1)
    
#     # Process each top merged community
#     for idx, merged_id in enumerate(top_merged_ids):
#         comm_label = idx  # Will be 0, 1, etc.
#         nodes_in_merged = merged_comm_to_nodes[merged_id]
#         print(f"  Merged community {merged_id} (will be labeled {comm_label}): {len(nodes_in_merged)} nodes")
        
#         for node in nodes_in_merged:
#             nodes_to_keep.add(node)
#             # If node appears in multiple top merged communities, assign to the first one encountered
#             if node not in node_to_comm:
#                 node_to_comm[node] = comm_label
    
#     print(f"\nTotal unique nodes in top {len(top_merged_ids)} merged communities: {len(nodes_to_keep)}")
    
#     # Show overlap if any
#     if len(top_merged_ids) == 2:
#         comm0_nodes = merged_comm_to_nodes[top_merged_ids[0]]
#         comm1_nodes = merged_comm_to_nodes[top_merged_ids[1]]
#         overlap = comm0_nodes & comm1_nodes
#         only_comm0 = comm0_nodes - comm1_nodes
#         only_comm1 = comm1_nodes - comm0_nodes
#         print(f"\nOverlap analysis:")
#         print(f"  Nodes only in merged community {top_merged_ids[0]} (label 0): {len(only_comm0)}")
#         print(f"  Nodes only in merged community {top_merged_ids[1]} (label 1): {len(only_comm1)}")
#         print(f"  Nodes in both merged communities (assigned to label 0): {len(overlap)}")
#         print(f"  Total unique nodes: {len(nodes_to_keep)}")
        
#         if len(overlap) == len(comm0_nodes) and len(overlap) == len(comm1_nodes):
#             print(f"  WARNING: The two merged communities are identical!")
#         elif len(overlap) > 0:
#             print(f"  Note: {len(overlap)} nodes appear in both merged communities and are assigned to community 0")
    
#     # Create subgraph (induced subgraph - includes all edges between selected nodes)
#     G_filtered = G.subgraph(nodes_to_keep).copy()
    
#     # Assign community labels
#     for node in G_filtered.nodes():
#         if node in node_to_comm:
#             G_filtered.nodes[node]["comm"] = node_to_comm[node]
#         else:
#             # This shouldn't happen, but handle it
#             print(f"Warning: Node {node} not found in node_to_comm. Removing from graph.")
#             G_filtered.remove_node(node)
    
#     print(f"\nFiltered graph: {G_filtered.number_of_nodes()} nodes, {G_filtered.number_of_edges()} edges")
    
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
#     G, comm_to_nodes = load_amazon_graph()
    
#     # Filter to top 2 communities (after merging overlapping ones)
#     G_filtered = filter_to_top_communities(G, comm_to_nodes, top_k=2, merge_overlap_threshold=0.5)
    
#     # Ensure graph is undirected
#     if G_filtered.is_directed():
#         G_filtered = G_filtered.to_undirected()
    
#     # Ensure all edges have weights (default to 1.0)
#     for u, v, d in G_filtered.edges(data=True):
#         if "weight" not in d:
#             d["weight"] = 1.0
    
#     # Save to GML
#     output_file = "experiments/amazon_snap/amazon.gml"
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
    G = nx.read_gml("experiments/amazon_snap/amazon.gml")
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
        init="bethe_hessian",
        max_iter=10000,
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
        community_proxy="leiden",
        local_score="adamic_adar"
    )

    # Post-denoise BP
    G_res = res_duo['G_final']
    _, preds_post, _, _ = belief_propagation_weighted(
                                        G_res, 
                                        q=num_comms, 
                                        seed=0, 
                                        init="bethe_hessian",
                                        max_iter=10000,
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

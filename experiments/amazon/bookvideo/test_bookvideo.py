import numpy as np
import networkx as nx
import csv
import random
from collections import defaultdict
from algorithms.bp.vectorized_bp import (
    get_true_communities,
    belief_propagation_weighted,
    detection_stats
)
from algorithms.duo_spec import duo_spec

def filter_by_communities(G: nx.Graph, comm_pair: list, attr: str = "comm"):
    """
    Filter graph to nodes in the specified community pair.
    
    Args:
        G: Input graph
        comm_pair: List of community strings to keep (e.g., ["Book", "DVD"])
        attr: Node attribute name for community labels
        
    Returns:
        Filtered graph (may not be connected)
    """
    comm_set = set(comm_pair)
    nodes_to_keep = []
    for node, data in G.nodes(data=True):
        comm = data.get(attr, None)
        if comm in comm_set:
            nodes_to_keep.append(node)
    
    print(f"  Filtered to {len(nodes_to_keep)} nodes in communities {comm_pair}")
    
    # Create subgraph
    G_filtered = G.subgraph(nodes_to_keep).copy()
    print(f"  Filtered graph: {G_filtered.number_of_nodes()} nodes, {G_filtered.number_of_edges()} edges")
    
    return G_filtered

def sample_components_and_connect(G: nx.Graph, target_per_comm: int = 5000, tolerance: int = 500, seed: int = 0, attr: str = "comm"):
    """
    Select connected components from the whole graph (spanning both communities) and piece them
    together to meet the target per community (±tolerance).
    
    Args:
        G: Input graph (may not be connected)
        target_per_comm: Target number of nodes per community
        tolerance: Allowed deviation from target (±tolerance)
        seed: Random seed
        attr: Node attribute name for community labels
        
    Returns:
        Connected subgraph with selected components (edges added if needed)
    """
    rng = random.Random(seed)
    
    # Find all connected components in the whole graph
    comps = list(nx.connected_components(G))
    print(f"  Found {len(comps)} connected components in the graph")
    
    # For each component, count nodes per community
    comps_with_comm_counts = []
    for comp in comps:
        comm_counts = defaultdict(int)
        for node in comp:
            comm = G.nodes[node].get(attr, None)
            if comm is not None:
                comm_counts[comm] += 1
        comps_with_comm_counts.append((comp, comm_counts, len(comp)))
    
    # Sort by total size (descending)
    comps_with_comm_counts.sort(key=lambda x: x[2], reverse=True)
    
    # Target ranges
    target_low = target_per_comm - tolerance
    target_high = target_per_comm + tolerance
    
    # Greedily select components to reach target per community
    selected_comps = []
    current_counts = defaultdict(int)
    
    for comp, comm_counts, total_size in comps_with_comm_counts:
        # Check if adding this component would keep us within tolerance for all communities
        would_exceed = False
        for comm, count in comm_counts.items():
            if current_counts[comm] + count > target_high:
                would_exceed = True
                break
        
        # If it doesn't exceed, add it
        if not would_exceed:
            selected_comps.append(comp)
            for comm, count in comm_counts.items():
                current_counts[comm] += count
    
    # Check if we reached targets for all communities
    all_communities = set()
    for comp, comm_counts, _ in comps_with_comm_counts:
        all_communities.update(comm_counts.keys())
    
    for comm in all_communities:
        if current_counts[comm] < target_low:
            # Try to add more components to reach target
            for comp, comm_counts, total_size in comps_with_comm_counts:
                if comp in selected_comps:
                    continue
                # Check if this component helps this community without exceeding others
                helps = comm_counts.get(comm, 0) > 0
                would_exceed_any = False
                for c, count in comm_counts.items():
                    if current_counts[c] + count > target_high:
                        would_exceed_any = True
                        break
                
                if helps and not would_exceed_any:
                    selected_comps.append(comp)
                    for c, count in comm_counts.items():
                        current_counts[c] += count
                    if current_counts[comm] >= target_low:
                        break
    
    print(f"  Selected {len(selected_comps)} components")
    print(f"  Current nodes per community:")
    for comm in sorted(current_counts.keys()):
        print(f"    Community {comm}: {current_counts[comm]} nodes (target: {target_per_comm}±{tolerance})")
    
    # Create subgraph from selected components
    selected_nodes = set()
    for comp in selected_comps:
        selected_nodes.update(comp)
    
    G_sub = G.subgraph(selected_nodes).copy()
    
    # Ensure connectivity by adding shortest paths between components
    comps_sub = list(nx.connected_components(G_sub))
    if len(comps_sub) > 1:
        print(f"  Selected subgraph has {len(comps_sub)} components. Adding edges to connect...")
        # Use original graph to find shortest paths
        master = set(comps_sub[0])
        edges_added = 0
        for comp in comps_sub[1:]:
            # Find shortest path between components in original graph
            u = rng.choice(list(comp))
            v = rng.choice(list(master))
            try:
                path = nx.shortest_path(G, u, v)
                # Add all nodes along the path to subgraph (with their attributes)
                for node in path:
                    if node not in G_sub:
                        node_data = G.nodes[node].copy()
                        G_sub.add_node(node, **node_data)
                # Add all edges along the path
                for i in range(len(path) - 1):
                    u_edge, v_edge = path[i], path[i+1]
                    if not G_sub.has_edge(u_edge, v_edge):
                        # Copy edge attributes from original graph if edge exists
                        if G.has_edge(u_edge, v_edge):
                            edge_data = G.edges[u_edge, v_edge].copy()
                            G_sub.add_edge(u_edge, v_edge, **edge_data)
                        else:
                            # If path edge doesn't exist in original, add with default weight
                            G_sub.add_edge(u_edge, v_edge, weight=1.0, dist=1.0)
                        edges_added += 1
                master.update(path)
                master.update(comp)
            except nx.NetworkXNoPath:
                # If no path exists, just add a direct edge with default weight
                G_sub.add_edge(u, v, weight=1.0, dist=1.0)
                edges_added += 1
                master.add(u)
                master.add(v)
        print(f"  Added {edges_added} edges to ensure connectivity")
    else:
        print(f"  Selected subgraph is already connected")
    
    # Report final community distribution
    final_nodes_by_comm = defaultdict(int)
    for node in G_sub.nodes():
        comm = G_sub.nodes[node].get(attr, None)
        if comm is not None:
            final_nodes_by_comm[comm] += 1
    
    print(f"  Final nodes per community:")
    for comm in sorted(final_nodes_by_comm.keys()):
        print(f"    Community {comm}: {final_nodes_by_comm[comm]} nodes")
    
    print(f"  Final graph: {G_sub.number_of_nodes()} nodes, {G_sub.number_of_edges()} edges")
    print(f"  Graph is connected: {nx.is_connected(G_sub)}")
    
    return G_sub

def convert_string_comm_to_numeric(G: nx.Graph, comm_pair: list, attr: str = "comm"):
    """
    Convert string community labels to numeric (0, 1) for compatibility.
    
    Args:
        G: Input graph
        comm_pair: List of community strings (e.g., ["Book", "DVD"])
        attr: Node attribute name for community labels
        
    Returns:
        Graph with numeric community labels
    """
    comm_mapping = {comm_pair[i]: i for i in range(len(comm_pair))}
    for node in G.nodes():
        comm_str = G.nodes[node].get(attr, None)
        if comm_str in comm_mapping:
            G.nodes[node][attr] = comm_mapping[comm_str]
    return G

def coords_str2arr(G: nx.Graph, dim=16):
    """Convert string formatted coords to numpy array."""
    new_G = nx.Graph()
    for n in G.nodes():
        coord_str = G.nodes[n]["coords"]
        coord_arr = np.fromstring(coord_str, sep=",")
        if len(coord_arr) != dim:
            coord_arr = np.zeros(dim)
        # Copy all node attributes
        node_attrs = G.nodes[n].copy()
        node_attrs["coords"] = coord_arr
        new_G.add_node(int(n), **node_attrs)
        
    for u, v in G.edges():
        # Copy all edge attributes
        edge_attrs = G.edges[u, v].copy()
        # Convert dist if it's a string
        if "dist" in edge_attrs and isinstance(edge_attrs["dist"], str):
            edge_attrs["dist"] = float(edge_attrs["dist"])
        new_G.add_edge(int(u), int(v), **edge_attrs)
    new_G.graph = G.graph.copy()
    new_G = nx.relabel.convert_node_labels_to_integers(new_G, first_label=0)
    return new_G

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
    graph_name = "bookvideo"
    comm_pair = ["Book", "Video"]
    print(f"\n{'='*60}")
    print(f"Processing: {graph_name}")
    print(f"{'='*60}")
    
    # Load graph and make undirected
    print(f"\nLoading amazon_graph.gml...")
    G = nx.read_gml("amazon_metadata_test/amazon_graph.gml")
    G = G.to_undirected()
    print(f"Original graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Filter by community pair
    print(f"\nFiltering to communities {comm_pair}...")
    G = filter_by_communities(G, comm_pair=comm_pair, attr="comm")
    
    # Convert string community labels to numeric (0, 1)
    G = convert_string_comm_to_numeric(G, comm_pair=comm_pair, attr="comm")
    
    # Select connected components of similar sizes (5000 ± 500 per community) and ensure connectivity
    print(f"\nSelecting connected components (target: 5000±500 per community) and ensuring connectivity...")
    G = sample_components_and_connect(G, target_per_comm=5000, tolerance=500, seed=0, attr="comm")
    
    # Ensure all edges have weights
    for u, v, d in G.edges(data=True):
        if "weight" not in d:
            d["weight"] = 1.0
    
    # Convert coordinate strings to numpy arrays (required for duo_spec)
    print(f"\nConverting coordinate strings to numpy arrays...")
    G = coords_str2arr(G, dim=16)
    
    print(f"Final graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Graph is connected: {nx.is_connected(G)}")
    
    true_labels = get_true_communities(G, node2idx=None, attr="comm")
    num_comms = len(np.unique(true_labels))
    print(f"Number of communities: {num_comms}")
    
    # Compute graph statistics
    graph_stats = compute_graph_stats(G, true_labels)
    graph_stats['graph_name'] = graph_name
    
    # Pre-denoise BP
    print("\nRunning BP (pre-denoise)...")
    _, preds_pre, _, _ = belief_propagation_weighted(
        G,
        q=num_comms,
        seed=0,
        init="spectral",
        max_iter=10000,
    )
    stats_pre = detection_stats(preds_pre, true_labels)
    print("=== BP Accuracy (Pre-Denoise) ===")
    for k, v in stats_pre.items():
        print(f"{k:>25s} : {v}")
    
    # Denoise
    print("\nRunning DuoSpec...")
    res_duo = duo_spec(
        G,
        K=num_comms,
        max_em_iters=20,
        community_proxy="leiden"
    )
    
    # Post-denoise BP
    print("\nRunning BP (post-denoise)...")
    G_res = res_duo['G_final']
    _, preds_post, _, _ = belief_propagation_weighted(
        G_res, 
        q=num_comms, 
        seed=0, 
        init="spectral",
        max_iter=10000,
    )
    stats_post = detection_stats(preds_post, true_labels)
    print("=== Post-Duospec BP ===")
    for k, v in stats_post.items():
        print(f"{k:>25s} : {v}")
    
    # Save to CSV (append mode)
    csv_file = "experiments/amazon/amazon_results.csv"
    file_exists = False
    try:
        with open(csv_file, 'r') as f:
            file_exists = True
    except FileNotFoundError:
        file_exists = False
    
    with open(csv_file, 'a', newline='') as f:
        # Collect all possible accuracy keys (handle variable number of communities)
        all_keys = set(stats_pre.keys()) | set(stats_post.keys())
        fieldnames = ['graph_name', 'num_nodes', 'num_edges', 'nodes_per_community', 'average_degree', 'clustering_coefficient']
        for key in sorted(all_keys):
            if key not in ['num vertices', 'num communities predicted']:
                fieldnames.append(f'pre_{key}')
                fieldnames.append(f'post_{key}')
        fieldnames.extend(['pre_num_vertices', 'pre_num_communities_predicted', 'pre_perm_p',
                          'post_num_vertices', 'post_num_communities_predicted', 'post_perm_p'])
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
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

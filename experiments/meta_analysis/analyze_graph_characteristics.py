"""
Meta-analysis of graph characteristics: Citation Networks vs Social Networks

This script analyzes differences between citation networks and social networks
in terms of:
1. Basic graph statistics (size, density, degree distribution, etc.)
2. Geometric characteristics (edge distances, coordinate-based metrics)
3. Correlations between geometry and community structure
4. Characteristics that may affect DuoSpec performance
"""

import numpy as np
import networkx as nx
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Citation network imports
import torch
from torch_geometric.datasets import Coauthor, Planetoid
from experiments.planetoid_tests.load_planetoid_data import grab_planetoid_data, to_networkx_graph

# Algorithm imports
from algorithms.bp.vectorized_bp import get_true_communities

# ============================================================================
# Graph Loading Functions
# ============================================================================

def load_coauthor(name='Physics'):
    """Load Coauthor dataset and convert to NetworkX graph."""
    dataset = Coauthor(root='data/Coauthor', name=name)
    data = dataset[0]
    edge_index = data.edge_index.numpy()
    G = nx.Graph()
    
    # Add nodes with community labels and features
    for i in range(data.num_nodes):
        G.add_node(i, 
                   comm=int(data.y[i].item()),
                   coords=data.x[i].numpy())
    
    # Add edges with cosine distance
    edges = list(zip(edge_index[0], edge_index[1]))
    dists = []
    for u, v in edges:
        coord_u = G.nodes[u]['coords'] 
        coord_v = G.nodes[v]['coords']
        similarity = np.dot(coord_u, coord_v) / (np.linalg.norm(coord_u) * np.linalg.norm(coord_v) + 1e-10)
        dist = 2 * (1 - similarity)
        dists.append(dist)
    
    G.add_edges_from([(u, v, {'dist': d}) for (u, v), d in zip(edges, dists)])
    return G

def load_planetoid(dataset_name):
    """Load Planetoid dataset (Cora, CiteSeer, PubMed)."""
    dataset = grab_planetoid_data(dataset_name, ensure_undirected=True)
    G = to_networkx_graph(dataset[0])
    return G

def load_social_network(name):
    """Load social network from GML file."""
    gml_path = f"experiments/{name}/{name}.gml"
    try:
        G = nx.read_gml(gml_path)
        # Ensure edges have weights
        for u, v, d in G.edges(data=True):
            if "weight" not in d:
                d["weight"] = 1.0
        return G
    except Exception as e:
        print(f"Warning: Could not load {name}: {e}")
        return None

# ============================================================================
# Basic Graph Statistics
# ============================================================================

def compute_basic_stats(G, true_labels=None):
    """Compute basic graph statistics."""
    stats = {}
    
    # Size statistics
    stats['num_nodes'] = G.number_of_nodes()
    stats['num_edges'] = G.number_of_edges()
    stats['density'] = nx.density(G)
    
    # Degree statistics
    degrees = [d for n, d in G.degree()]
    stats['avg_degree'] = np.mean(degrees) if degrees else 0.0
    stats['median_degree'] = np.median(degrees) if degrees else 0.0
    stats['max_degree'] = np.max(degrees) if degrees else 0.0
    stats['min_degree'] = np.min(degrees) if degrees else 0.0
    stats['degree_std'] = np.std(degrees) if degrees else 0.0
    
    # Clustering
    try:
        stats['avg_clustering'] = nx.average_clustering(G)
        stats['transitivity'] = nx.transitivity(G)
    except:
        stats['avg_clustering'] = float('nan')
        stats['transitivity'] = float('nan')
    
    # Path length (sample for large graphs)
    if G.number_of_nodes() < 10000:
        try:
            stats['avg_path_length'] = nx.average_shortest_path_length(G)
        except:
            stats['avg_path_length'] = float('nan')
    else:
        # Sample-based estimation for large graphs
        try:
            sample_nodes = list(G.nodes())[:1000]
            paths = []
            for i, u in enumerate(sample_nodes[:100]):
                for v in sample_nodes[i+1:min(i+11, len(sample_nodes))]:
                    try:
                        path_len = nx.shortest_path_length(G, u, v)
                        paths.append(path_len)
                    except:
                        pass
            stats['avg_path_length'] = np.mean(paths) if paths else float('nan')
        except:
            stats['avg_path_length'] = float('nan')
    
    # Diameter (approximate for large graphs)
    if G.number_of_nodes() < 1000:
        try:
            stats['diameter'] = nx.diameter(G)
        except:
            stats['diameter'] = float('nan')
    else:
        stats['diameter'] = float('nan')  # Too expensive for large graphs
    
    # Community statistics
    if true_labels is not None:
        unique_comms, counts = np.unique(true_labels, return_counts=True)
        stats['num_communities'] = len(unique_comms)
        stats['avg_community_size'] = np.mean(counts)
        stats['community_size_std'] = np.std(counts)
        stats['community_size_imbalance'] = np.max(counts) / np.min(counts) if len(counts) > 1 else 1.0
    else:
        stats['num_communities'] = float('nan')
        stats['avg_community_size'] = float('nan')
        stats['community_size_std'] = float('nan')
        stats['community_size_imbalance'] = float('nan')
    
    return stats

# ============================================================================
# Geometric Characteristics
# ============================================================================

def compute_geometric_stats(G, true_labels=None):
    """Compute geometric characteristics of the graph."""
    stats = {}
    
    # Check if nodes have coordinates
    has_coords = False
    sample_node = list(G.nodes())[0] if G.number_of_nodes() > 0 else None
    if sample_node and 'coords' in G.nodes[sample_node]:
        has_coords = True
        coords = np.array([G.nodes[n]['coords'] for n in G.nodes()])
        stats['has_node_coords'] = True
        stats['coord_dim'] = coords.shape[1] if len(coords) > 0 else 0
    else:
        stats['has_node_coords'] = False
        stats['coord_dim'] = 0
    
    # Check if edges have distances
    has_edge_dists = False
    sample_edge = list(G.edges(data=True))[0] if G.number_of_edges() > 0 else None
    if sample_edge and 'dist' in sample_edge[2]:
        has_edge_dists = True
        edge_dists = [d.get('dist', float('nan')) for u, v, d in G.edges(data=True)]
        edge_dists = [d for d in edge_dists if not np.isnan(d)]
        if edge_dists:
            stats['has_edge_dists'] = True
            stats['avg_edge_dist'] = np.mean(edge_dists)
            stats['median_edge_dist'] = np.median(edge_dists)
            stats['edge_dist_std'] = np.std(edge_dists)
            stats['min_edge_dist'] = np.min(edge_dists)
            stats['max_edge_dist'] = np.max(edge_dists)
        else:
            stats['has_edge_dists'] = False
    else:
        stats['has_edge_dists'] = False
    
    # If we have coordinates, compute coordinate-based metrics
    if has_coords and len(coords) > 0:
        # Pairwise distances between all nodes (sample for large graphs)
        if len(coords) <= 5000:
            try:
                pairwise_dists = pdist(coords, metric='euclidean')
                stats['avg_pairwise_coord_dist'] = np.mean(pairwise_dists)
                stats['median_pairwise_coord_dist'] = np.median(pairwise_dists)
                stats['pairwise_coord_dist_std'] = np.std(pairwise_dists)
            except:
                stats['avg_pairwise_coord_dist'] = float('nan')
                stats['median_pairwise_coord_dist'] = float('nan')
                stats['pairwise_coord_dist_std'] = float('nan')
        else:
            # Sample-based for large graphs
            sample_indices = np.random.choice(len(coords), min(1000, len(coords)), replace=False)
            sample_coords = coords[sample_indices]
            try:
                pairwise_dists = pdist(sample_coords, metric='euclidean')
                stats['avg_pairwise_coord_dist'] = np.mean(pairwise_dists)
                stats['median_pairwise_coord_dist'] = np.median(pairwise_dists)
                stats['pairwise_coord_dist_std'] = np.std(pairwise_dists)
            except:
                stats['avg_pairwise_coord_dist'] = float('nan')
                stats['median_pairwise_coord_dist'] = float('nan')
                stats['pairwise_coord_dist_std'] = float('nan')
        
        # Coordinate spread/variance
        stats['coord_variance'] = np.var(coords.flatten())
        stats['coord_mean_norm'] = np.mean([np.linalg.norm(c) for c in coords])
        stats['coord_std_norm'] = np.std([np.linalg.norm(c) for c in coords])
    else:
        stats['avg_pairwise_coord_dist'] = float('nan')
        stats['median_pairwise_coord_dist'] = float('nan')
        stats['pairwise_coord_dist_std'] = float('nan')
        stats['coord_variance'] = float('nan')
        stats['coord_mean_norm'] = float('nan')
        stats['coord_std_norm'] = float('nan')
    
    # Correlation between edge distances and graph distances
    if has_edge_dists and G.number_of_nodes() < 5000:
        try:
            edge_dist_list = []
            graph_dist_list = []
            edges_sample = list(G.edges(data=True))[:min(1000, G.number_of_edges())]
            for u, v, d in edges_sample:
                if 'dist' in d:
                    edge_dist_list.append(d['dist'])
                    try:
                        graph_dist = nx.shortest_path_length(G, u, v)
                        graph_dist_list.append(graph_dist)
                    except:
                        pass
            if len(edge_dist_list) > 10 and len(graph_dist_list) > 10:
                corr, pval = stats.pearsonr(edge_dist_list[:len(graph_dist_list)], 
                                           graph_dist_list[:len(edge_dist_list)])
                stats['edge_dist_vs_graph_dist_corr'] = corr
                stats['edge_dist_vs_graph_dist_pval'] = pval
            else:
                stats['edge_dist_vs_graph_dist_corr'] = float('nan')
                stats['edge_dist_vs_graph_dist_pval'] = float('nan')
        except:
            stats['edge_dist_vs_graph_dist_corr'] = float('nan')
            stats['edge_dist_vs_graph_dist_pval'] = float('nan')
    else:
        stats['edge_dist_vs_graph_dist_corr'] = float('nan')
        stats['edge_dist_vs_graph_dist_pval'] = float('nan')
    
    return stats

# ============================================================================
# Geometry-Community Correlations
# ============================================================================

def compute_geometry_community_correlations(G, true_labels):
    """Compute correlations between geometric properties and community structure."""
    stats = {}
    
    if true_labels is None:
        return {k: float('nan') for k in [
            'intra_comm_edge_dist_mean', 'inter_comm_edge_dist_mean',
            'intra_comm_edge_dist_ratio', 'coord_separation_by_comm',
            'edge_dist_community_correlation'
        ]}
    
    # Check if edges have distances
    has_edge_dists = False
    sample_edge = list(G.edges(data=True))[0] if G.number_of_edges() > 0 else None
    if sample_edge and 'dist' in sample_edge[2]:
        has_edge_dists = True
    
    # Intra-community vs inter-community edge distances
    if has_edge_dists:
        intra_comm_dists = []
        inter_comm_dists = []
        
        # Create node to index mapping
        node_list = list(G.nodes())
        node_to_idx = {node: i for i, node in enumerate(node_list)}
        
        for u, v, d in G.edges(data=True):
            if 'dist' in d:
                u_idx = node_to_idx.get(u, u)
                v_idx = node_to_idx.get(v, v)
                
                if u_idx < len(true_labels) and v_idx < len(true_labels):
                    u_comm = true_labels[u_idx]
                    v_comm = true_labels[v_idx]
                    
                    if u_comm == v_comm:
                        intra_comm_dists.append(d['dist'])
                    else:
                        inter_comm_dists.append(d['dist'])
        
        if intra_comm_dists and inter_comm_dists:
            stats['intra_comm_edge_dist_mean'] = np.mean(intra_comm_dists)
            stats['inter_comm_edge_dist_mean'] = np.mean(inter_comm_dists)
            stats['intra_comm_edge_dist_ratio'] = (np.mean(intra_comm_dists) / 
                                                   np.mean(inter_comm_dists) 
                                                   if np.mean(inter_comm_dists) > 0 else float('nan'))
        else:
            stats['intra_comm_edge_dist_mean'] = float('nan')
            stats['inter_comm_edge_dist_mean'] = float('nan')
            stats['intra_comm_edge_dist_ratio'] = float('nan')
    else:
        stats['intra_comm_edge_dist_mean'] = float('nan')
        stats['inter_comm_edge_dist_mean'] = float('nan')
        stats['intra_comm_edge_dist_ratio'] = float('nan')
    
    # Coordinate separation by community
    sample_node = list(G.nodes())[0] if G.number_of_nodes() > 0 else None
    if sample_node and 'coords' in G.nodes[sample_node]:
        try:
            node_list = list(G.nodes())
            coords_by_comm = defaultdict(list)
            for i, node in enumerate(node_list):
                if i < len(true_labels) and 'coords' in G.nodes[node]:
                    comm = true_labels[i]
                    coords_by_comm[comm].append(G.nodes[node]['coords'])
            
            # Compute centroid for each community
            comm_centroids = {}
            for comm, comm_coords in coords_by_comm.items():
                if comm_coords:
                    comm_centroids[comm] = np.mean(comm_coords, axis=0)
            
            # Compute inter-community distances
            if len(comm_centroids) > 1:
                comm_list = list(comm_centroids.keys())
                inter_comm_dists = []
                for i, comm1 in enumerate(comm_list):
                    for comm2 in comm_list[i+1:]:
                        dist = np.linalg.norm(comm_centroids[comm1] - comm_centroids[comm2])
                        inter_comm_dists.append(dist)
                
                # Compute intra-community spread
                intra_comm_spreads = []
                for comm, comm_coords in coords_by_comm.items():
                    if len(comm_coords) > 1:
                        centroid = comm_centroids[comm]
                        spreads = [np.linalg.norm(c - centroid) for c in comm_coords]
                        intra_comm_spreads.extend(spreads)
                
                if inter_comm_dists and intra_comm_spreads:
                    stats['coord_separation_by_comm'] = (np.mean(inter_comm_dists) / 
                                                        np.mean(intra_comm_spreads) 
                                                        if np.mean(intra_comm_spreads) > 0 else float('nan'))
                else:
                    stats['coord_separation_by_comm'] = float('nan')
            else:
                stats['coord_separation_by_comm'] = float('nan')
        except Exception as e:
            stats['coord_separation_by_comm'] = float('nan')
    else:
        stats['coord_separation_by_comm'] = float('nan')
    
    # Edge distance correlation with same-community indicator
    if has_edge_dists:
        try:
            edge_dists = []
            same_comm = []
            node_list = list(G.nodes())
            node_to_idx = {node: i for i, node in enumerate(node_list)}
            
            for u, v, d in list(G.edges(data=True))[:min(5000, G.number_of_edges())]:
                if 'dist' in d:
                    u_idx = node_to_idx.get(u, u)
                    v_idx = node_to_idx.get(v, v)
                    
                    if u_idx < len(true_labels) and v_idx < len(true_labels):
                        edge_dists.append(d['dist'])
                        same_comm.append(1 if true_labels[u_idx] == true_labels[v_idx] else 0)
            
            if len(edge_dists) > 10:
                corr, pval = stats.pearsonr(edge_dists, same_comm)
                stats['edge_dist_community_correlation'] = corr
                stats['edge_dist_community_pval'] = pval
            else:
                stats['edge_dist_community_correlation'] = float('nan')
                stats['edge_dist_community_pval'] = float('nan')
        except:
            stats['edge_dist_community_correlation'] = float('nan')
            stats['edge_dist_community_pval'] = float('nan')
    else:
        stats['edge_dist_community_correlation'] = float('nan')
        stats['edge_dist_community_pval'] = float('nan')
    
    return stats

# ============================================================================
# DuoSpec-Relevant Characteristics
# ============================================================================

def compute_duospec_relevant_stats(G, true_labels=None):
    """Compute characteristics that may affect DuoSpec performance."""
    stats = {}
    
    # Locality-related metrics (common neighbors, triangles)
    try:
        # Average common neighbors per edge
        common_neighbors = []
        edges_sample = list(G.edges())[:min(5000, G.number_of_edges())]
        for u, v in edges_sample:
            cn = len(list(nx.common_neighbors(G, u, v)))
            common_neighbors.append(cn)
        stats['avg_common_neighbors'] = np.mean(common_neighbors) if common_neighbors else float('nan')
        stats['median_common_neighbors'] = np.median(common_neighbors) if common_neighbors else float('nan')
    except:
        stats['avg_common_neighbors'] = float('nan')
        stats['median_common_neighbors'] = float('nan')
    
    # Triangle count
    try:
        if G.number_of_nodes() < 10000:
            triangles = nx.triangles(G)
            stats['total_triangles'] = sum(triangles.values()) // 3
            stats['avg_triangles_per_node'] = np.mean(list(triangles.values())) if triangles else 0.0
        else:
            # Sample-based for large graphs
            sample_nodes = list(G.nodes())[:1000]
            triangle_count = 0
            for u in sample_nodes:
                neighbors = list(G.neighbors(u))
                for i, v in enumerate(neighbors):
                    for w in neighbors[i+1:]:
                        if G.has_edge(v, w):
                            triangle_count += 1
            stats['total_triangles'] = triangle_count  # Approximate
            stats['avg_triangles_per_node'] = float('nan')
    except:
        stats['total_triangles'] = float('nan')
        stats['avg_triangles_per_node'] = float('nan')
    
    # Modularity (community quality metric)
    if true_labels is not None:
        try:
            # Create community partition dict
            node_list = list(G.nodes())
            communities = {}
            for i, node in enumerate(node_list):
                if i < len(true_labels):
                    comm = true_labels[i]
                    if comm not in communities:
                        communities[comm] = []
                    communities[comm].append(node)
            
            partition = [set(comm_nodes) for comm_nodes in communities.values()]
            stats['modularity'] = nx.community.modularity(G, partition)
        except:
            stats['modularity'] = float('nan')
    else:
        stats['modularity'] = float('nan')
    
    # Edge weight statistics (if available)
    weights = [d.get('weight', 1.0) for u, v, d in G.edges(data=True)]
    stats['avg_edge_weight'] = np.mean(weights)
    stats['edge_weight_std'] = np.std(weights)
    stats['has_variable_weights'] = stats['edge_weight_std'] > 0.01
    
    # Degree assortativity (correlation of degrees of connected nodes)
    try:
        stats['degree_assortativity'] = nx.degree_assortativity_coefficient(G)
    except:
        stats['degree_assortativity'] = float('nan')
    
    return stats

# ============================================================================
# Graph Filtering Functions
# ============================================================================

def filter_to_two_communities(G, true_labels, filter_type='top2'):
    """
    Filter graph to 2 communities based on filter_type.
    
    Args:
        G: NetworkX graph
        true_labels: Array of community labels (aligned with G.nodes() order)
        filter_type: 'top2' for top 2 largest, 'balanced' for 2nd and 3rd largest
    
    Returns:
        Filtered graph, updated true_labels
    """
    # Create node to index mapping for original graph
    node_list_orig = list(G.nodes())
    node_to_idx_orig = {node: i for i, node in enumerate(node_list_orig)}
    
    unique_comms, counts = np.unique(true_labels, return_counts=True)
    
    if filter_type == 'balanced':
        # For coauthor: use 2nd and 3rd largest communities (balanced)
        sorted_indices = np.argsort(counts)
        second_third_indices = sorted_indices[-3:-1]  # [-3, -2]
        target_comms = unique_comms[second_third_indices]
        print(f"  Filtering to 2nd and 3rd largest communities: {target_comms} with sizes: {counts[second_third_indices]}")
    else:
        # For others: use top 2 largest communities
        top2_indices = np.argsort(counts)[-2:]  # Top 2 largest
        target_comms = unique_comms[top2_indices]
        print(f"  Filtering to top 2 communities: {target_comms} with sizes: {counts[top2_indices]}")
    
    print(f"  Original communities: {len(unique_comms)}, sizes: {dict(zip(unique_comms, counts))}")
    
    # Filter nodes to only those in target communities
    # Use true_labels to determine which nodes to keep
    nodes_to_keep = []
    for i, node in enumerate(node_list_orig):
        if i < len(true_labels) and true_labels[i] in target_comms:
            nodes_to_keep.append(node)
    
    G = G.subgraph(nodes_to_keep).copy()
    
    # Relabel communities to 0 and 1
    comm_mapping = {int(target_comms[0]): 0, int(target_comms[1]): 1}
    for n in G.nodes():
        G.nodes[n]["comm"] = comm_mapping[G.nodes[n]["comm"]]
    
    # Update true labels to match new node order
    node_list_new = list(G.nodes())
    true_labels = np.array([G.nodes[node].get('comm', 0) for node in node_list_new])
    
    print(f"  After filtering: {len(G.nodes())} nodes, {len(G.edges())} edges, 2 communities")
    
    return G, true_labels

# ============================================================================
# Main Analysis Function
# ============================================================================

def analyze_graph(name, graph_type, G, true_labels=None):
    """Analyze a single graph and return all statistics."""
    print(f"Analyzing {name} ({graph_type})...")
    
    results = {
        'name': name,
        'type': graph_type
    }
    
    # Basic statistics
    basic_stats = compute_basic_stats(G, true_labels)
    results.update(basic_stats)
    
    # Geometric statistics
    geo_stats = compute_geometric_stats(G, true_labels)
    results.update(geo_stats)
    
    # Geometry-community correlations
    if true_labels is not None:
        geom_comm_stats = compute_geometry_community_correlations(G, true_labels)
        results.update(geom_comm_stats)
    else:
        geom_comm_stats = compute_geometry_community_correlations(G, None)
        results.update(geom_comm_stats)
    
    # DuoSpec-relevant statistics
    duospec_stats = compute_duospec_relevant_stats(G, true_labels)
    results.update(duospec_stats)
    
    return results

# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Run analysis on all networks."""
    all_results = []
    
    # Citation networks
    citation_networks = {
        'coauthor': ('citation', lambda: load_coauthor('Physics')),
        'dblp': ('citation', lambda: load_social_network('dblp')),  # Loaded as GML
        'citeseer': ('citation', lambda: load_planetoid('CiteSeer')),
        'pubmed': ('citation', lambda: load_planetoid('PubMed')),
        'cora': ('citation', lambda: load_planetoid('Cora')),
    }
    
    # Social networks
    social_networks = {
        'email_eu': ('social', lambda: load_social_network('email_eu')),
        'friendster': ('social', lambda: load_social_network('friendster')),
        'livejournal': ('social', lambda: load_social_network('livejournal')),
        'orkut': ('social', lambda: load_social_network('orkut')),
        'polblogs': ('social', lambda: load_social_network('polblogs')),
    }
    
    all_networks = {**citation_networks, **social_networks}
    
    for name, (graph_type, load_func) in all_networks.items():
        try:
            print(f"\n{'='*60}")
            print(f"Loading {name}...")
            G = load_func()
            
            if G is None:
                print(f"  Failed to load {name}")
                continue
            
            # Get true labels
            try:
                true_labels = get_true_communities(G, node2idx=None, attr="comm")
                if len(true_labels) != G.number_of_nodes():
                    # Try to align labels with nodes
                    node_list = list(G.nodes())
                    true_labels = np.array([G.nodes[node].get('comm', 0) for node in node_list])
            except:
                true_labels = None
                print(f"  Warning: Could not extract community labels for {name}")
            
            # Filter to 2 communities for specific graphs
            if true_labels is not None:
                if name == 'coauthor':
                    # Use balanced filtering (2nd and 3rd largest) like test_coauthor_balanced.py
                    print(f"  Applying balanced community filtering (2nd and 3rd largest)...")
                    G, true_labels = filter_to_two_communities(G, true_labels, filter_type='balanced')
                elif name in ['cora', 'citeseer', 'pubmed']:
                    # Use top 2 largest communities like their test files
                    print(f"  Applying top 2 community filtering...")
                    G, true_labels = filter_to_two_communities(G, true_labels, filter_type='top2')
                # For other graphs (dblp, email_eu, friendster, livejournal, orkut, polblogs), use all communities
            
            # Analyze graph
            results = analyze_graph(name, graph_type, G, true_labels)
            all_results.append(results)
            
            print(f"  Completed {name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            
        except Exception as e:
            print(f"  Error analyzing {name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create DataFrame and save
    df = pd.DataFrame(all_results)
    
    # Save to CSV
    output_file = "experiments/meta_analysis/graph_characteristics_analysis.csv"
    df.to_csv(output_file, index=False)
    print(f"\n{'='*60}")
    print(f"Analysis complete! Results saved to {output_file}")
    print(f"Total networks analyzed: {len(all_results)}")
    
    # Print summary statistics by network type
    if len(df) > 0:
        print("\n" + "="*60)
        print("SUMMARY STATISTICS BY NETWORK TYPE")
        print("="*60)
        
        for net_type in ['citation', 'social']:
            type_df = df[df['type'] == net_type]
            if len(type_df) > 0:
                print(f"\n{net_type.upper()} NETWORKS ({len(type_df)} networks):")
                print("-" * 60)
                
                # Key metrics to summarize
                key_metrics = [
                    'num_nodes', 'num_edges', 'density', 'avg_degree',
                    'avg_clustering', 'num_communities', 'modularity',
                    'avg_common_neighbors', 'degree_assortativity'
                ]
                
                for metric in key_metrics:
                    if metric in type_df.columns:
                        values = type_df[metric].dropna()
                        if len(values) > 0:
                            print(f"  {metric:30s}: mean={np.mean(values):.4f}, "
                                  f"median={np.median(values):.4f}, "
                                  f"std={np.std(values):.4f}")
        
        # Print comparison
        print("\n" + "="*60)
        print("CITATION vs SOCIAL NETWORK COMPARISONS")
        print("="*60)
        
        citation_df = df[df['type'] == 'citation']
        social_df = df[df['type'] == 'social']
        
        comparison_metrics = [
            'density', 'avg_degree', 'avg_clustering', 'modularity',
            'avg_common_neighbors', 'degree_assortativity'
        ]
        
        for metric in comparison_metrics:
            if metric in df.columns:
                cit_vals = citation_df[metric].dropna()
                soc_vals = social_df[metric].dropna()
                
                if len(cit_vals) > 0 and len(soc_vals) > 0:
                    print(f"\n{metric}:")
                    print(f"  Citation:  mean={np.mean(cit_vals):.4f}, median={np.median(cit_vals):.4f}")
                    print(f"  Social:    mean={np.mean(soc_vals):.4f}, median={np.median(soc_vals):.4f}")
                    
                    # Statistical test if possible
                    if len(cit_vals) >= 3 and len(soc_vals) >= 3:
                        stat, pval = stats.mannwhitneyu(cit_vals, soc_vals, alternative='two-sided')
                        print(f"  Mann-Whitney U test: p={pval:.4f} "
                              f"{'***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''}")

if __name__ == "__main__":
    main()

from algorithms.bp.old.vectorized_geometric_bp import (
    belief_propagation,
    detection_stats,
    get_true_communities,
)
# from experiments.observations.standard_observe import PairSamplingObservation, get_coordinate_distance

import numpy as np
import networkx as nx

from algorithms.bp.old.duo_bp import (
    duo_bp,
    create_dist_observed_subgraph,
)
# from experiments.community_detection.bp.bethe_duo_bp import (
#     duo_bp
# )
from algorithms.duo_spec import duo_spec, duo_bprop
import os
import json
import logging
import random
from algorithms.bp.vectorized_bp import belief_propagation, belief_propagation_weighted
from algorithms.spectral_ops.attention import motif_spectral_embedding
import math
from typing import List, Dict
from experiments.planetoid_tests.load_planetoid_data import grab_planetoid_data, to_networkx_graph

def coords_str2arr(G: nx.Graph, dim = 16):
    """
    for each coord, convert string formatted coord to numpy array
    """
    new_G = nx.Graph()
    for n in G.nodes():
        coord_str = G.nodes[n]["coords"]
        coord_arr = np.fromstring(coord_str, sep=",")
        if len(coord_arr) != dim:
            coord_arr = np.zeros(dim)
        new_G.add_node(int(n), coords=coord_arr, asin=G.nodes[n]["asin"], comm=G.nodes[n]["comm"])
        
    for u, v in G.edges():
        if "dist" in G.edges[u, v]:
            dist = G.edges[u, v]["dist"]
            if isinstance(dist, str):
                dist = float(dist)
            new_G.add_edge(int(u), int(v), dist=dist)
    new_G.graph = G.graph.copy()
    new_G = nx.relabel.convert_node_labels_to_integers(new_G, first_label=0)
    return new_G

def connect_components(G: nx.Graph, weight: float = 1e-3) -> None:
    """
    Add edges between components so that G becomes connected.
    Picks one representative node per component, then links them in a chain.
    
    Parameters
    ----------
    G : nx.Graph
        The graph to modify in-place.
    weight : float
        The weight to assign to each new bridging edge (default 0.001).
    """
    # find each connected component
    comps = list(nx.connected_components(G))
    if len(comps) <= 1:
        return  # already connected

    # pick one node from each component
    reps = [next(iter(c)) for c in comps]

    # link them in a simple chain: reps[0]—reps[1], reps[1]—reps[2], …
    for u, v in zip(reps[:-1], reps[1:]):
        if not G.has_edge(u, v):
            G.add_edge(u, v, weight=weight)

def split_graph_into_patches(G: nx.Graph, patch_size: int = 2000) -> tuple[List[nx.Graph], Dict]:
    """
    Split a large graph into smaller patches of approximately patch_size nodes each.
    Maintains information about cross-patch edges for later reconstruction.
    
    Parameters:
    -----------
    G : nx.Graph
        Input graph to be split
    patch_size : int
        Target size for each patch
        
    Returns:
    --------
    tuple[List[nx.Graph], Dict]
        List of subgraphs and dictionary mapping original node IDs to (patch_idx, local_node_id)
    """
    all_nodes = list(G.nodes())
    num_patches = math.ceil(len(all_nodes) / patch_size)
    patches = []
    node_mapping = {}  # maps original node ID to (patch_idx, local_node_id)
    
    # Create patches
    for i in range(num_patches):
        start_idx = i * patch_size
        end_idx = min((i + 1) * patch_size, len(all_nodes))
        patch_nodes = all_nodes[start_idx:end_idx]
        patch = G.subgraph(patch_nodes).copy()
        
        # Create mapping for nodes in this patch
        local_mapping = {old: new for new, old in enumerate(patch_nodes)}
        for old_id in patch_nodes:
            node_mapping[old_id] = (i, local_mapping[old_id])
        
        # Relabel nodes in patch to local IDs
        patch = nx.relabel_nodes(patch, local_mapping)
        patches.append(patch)
    
    # Verify all nodes are mapped
    assert len(node_mapping) == len(G), f"Node mapping missing nodes: {len(G) - len(node_mapping)} nodes lost"
    
    return patches, node_mapping

def combine_processed_graphs(processed_graphs: List[nx.Graph], original_graph: nx.Graph, node_mapping: Dict) -> nx.Graph:
    """
    Combine multiple processed graphs into one large graph while preserving original structure.
    
    Parameters:
    -----------
    processed_graphs : List[nx.Graph]
        List of processed subgraphs to combine
    original_graph : nx.Graph
        The original graph to reference for structure
    node_mapping : Dict
        Dictionary mapping original node IDs to (patch_idx, local_node_id)
        
    Returns:
    --------
    nx.Graph
        Combined graph with original structure preserved
    """
    combined_G = nx.Graph()
    
    # Calculate max nodes per patch for offset
    max_nodes_per_patch = max(len(g.nodes()) for g in processed_graphs)
    
    # First, add all nodes from original graph to ensure none are lost
    for orig_node in original_graph.nodes():
        patch_idx, local_id = node_mapping[orig_node]
        global_id = patch_idx * max_nodes_per_patch + local_id
        
        # Get attributes from processed graph if available
        patch = processed_graphs[patch_idx]
        if local_id in patch.nodes():
            attrs = patch.nodes[local_id]
        else:
            # If node somehow not in processed graph, use original attributes
            attrs = original_graph.nodes[orig_node]
        
        combined_G.add_node(global_id, **attrs)
        
    # Create reverse mapping from (patch_idx, local_id) to global_id
    reverse_mapping = {
        (patch_idx, local_id): patch_idx * max_nodes_per_patch + local_id
        for orig_id, (patch_idx, local_id) in node_mapping.items()
    }
    
    # Add all edges from original graph
    for u, v in original_graph.edges():
        u_patch_info = node_mapping[u]
        v_patch_info = node_mapping[v]
        
        new_u = reverse_mapping[u_patch_info]
        new_v = reverse_mapping[v_patch_info]
        
        # If edge exists in processed graphs, use those attributes
        if u_patch_info[0] == v_patch_info[0]:  # same patch
            patch = processed_graphs[u_patch_info[0]]
            if patch.has_edge(u_patch_info[1], v_patch_info[1]):
                edge_data = patch.edges[u_patch_info[1], v_patch_info[1]]
                # Ensure weight attribute exists
                if 'weight' not in edge_data:
                    edge_data['weight'] = original_graph.edges[u, v].get('weight', 1.0)
                combined_G.add_edge(new_u, new_v, **edge_data)
            else:
                # If edge somehow not in processed graph, use original attributes
                edge_data = original_graph.edges[u, v].copy()
                if 'weight' not in edge_data:
                    edge_data['weight'] = 1.0
                combined_G.add_edge(new_u, new_v, **edge_data)
        else:  # cross-patch edge
            # Use original edge attributes
            edge_data = original_graph.edges[u, v].copy()
            if 'weight' not in edge_data:
                edge_data['weight'] = 1.0
            combined_G.add_edge(new_u, new_v, **edge_data)
    
    # Verify node count matches
    assert len(combined_G) == len(original_graph), f"Combined graph has {len(combined_G)} nodes, original has {len(original_graph)}"
    
    # Print edge weight statistics
    weights = [data['weight'] for _, _, data in combined_G.edges(data=True)]
    if weights:
        print(f"Edge weight stats - Min: {min(weights):.6f}, Max: {max(weights):.6f}, Mean: {sum(weights)/len(weights):.6f}")
    
    return combined_G

if __name__ == "__main__":
    # G = nx.read_gml("amazon_metadata_test/amz_bookmusic.gml")
    print("Loading graph")
    G = grab_planetoid_data("PubMed")
    G = to_networkx_graph(G)
    print(f"Original graph has {len(G)} nodes and {len(G.edges())} edges")
    
    # Split graph into patches
    # patches, node_mapping = split_graph_into_patches(G, patch_size=2000)
    # print(f"Split graph into {len(patches)} patches")
    # print(f"Node mapping contains {len(node_mapping)} nodes")
    
    # duo_params = dict(
    #     K               = 3,                    # number of communities
    #     num_balls       = 32,                   # finer geometry embedding
    #     config          = "motif",              # geometry estimator
    #     max_em_iters    = 100,                  # allow more EM steps
    #     warmup_rounds   = 2,                    # hold off on any re-weighting
    #     anneal_steps    = 20,                   # then ramp λ from 0→full over 30 iter
    #     tol             = 1e-5,
    #     patience        = 10,
    #     random_state    = 42,
    #     spec_params = dict(
    #         dim = 64,
    #         walk_len = 40,
    #         num_walks = 10,
    #         window = 5,
    #     )
    # )
    
    # # Process each patch
    # processed_graphs = []
    # for i, patch in enumerate(patches):
    #     print(f"Processing patch {i+1}/{len(patches)} with {len(patch)} nodes")
    #     res = duo_spec(patch, **duo_params)
    #     processed_graphs.append(res["G_final"])
    #     print(f"Processed patch has {len(res['G_final'])} nodes")
    
    # # Combine processed graphs while preserving original structure
    # G_combined = combine_processed_graphs(processed_graphs, G, node_mapping)
    # print(f"Combined graph has {len(G_combined)} nodes and {len(G_combined.edges())} edges")
    # print(f"Original graph had {len(G)} nodes and {len(G.edges())} edges")
    
    # # Run belief propagation on combined graph
    # _, preds, _, _ = belief_propagation_weighted(
    #     G_combined, 
    #     q=3, 
    #     max_iter=10000,
    # )
    
    # Get detection stats
    _, preds, _, _ = belief_propagation(G, q=3, max_iter=10000)
    true_communities = get_true_communities(G, attr="comm")
    stats = detection_stats(preds, true_communities)
    print(stats)
    print(f"Finished detection stats")   
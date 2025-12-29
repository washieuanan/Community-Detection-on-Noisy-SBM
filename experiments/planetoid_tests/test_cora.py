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
from algorithms.duo_spec import duo_spec
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
    
    duo_params = dict(
    K               = 3,
    num_balls       = 8,    
    config          = 'motif',

    max_em_iters    = 60,
    warmup_rounds   = 0,
    anneal_steps    = 8,

    # community masks & strengths
    comm_cut        = 0.80,
    shrink_comm     = 0.05,
    boost_cut_comm  = 0.90,
    boost_comm      = 0.80,

    # geometry disabled
    geo_cut         = 0.80,
    shrink_geo      = 0.40,
    boost_cut_geo   = 0.97,
    boost_geo       = 0.10,

    # weight bounds
    w_min           = 0.01,
    w_cap           = 2.00,

    tol             = 1e-5,
    patience        = 5,
    random_state    = 42,
    base_seed       = 0,
    theta           = 20,
    spec_params     = dict(
        dim       = 64,
        walk_len  = 40,
        num_walks = 2,
        window    = 5,
        weight_pow=1.0,
    )
    )
    
    res = duo_spec(G, **duo_params)
    G_combined = res["G_final"]
    preds_duo = res["communities"]
    true_communities = get_true_communities(G, attr="comm")
    stats = detection_stats(preds_duo, true_communities)
    print("DuoSpec stats")
    print(stats)
    print(f"Finished detection stats")   

    # # Run belief propagation on combined graph
    _, preds, _, _ = belief_propagation_weighted(
        G_combined, 
        q=3, 
        max_iter=10000,
    )
    stats = detection_stats(preds, true_communities)
    print("BP + DUO stats")
    print(stats)
    print(f"Finished detection stats")   

    # Get detection stats
    _, preds, _, _ = belief_propagation(G, q=3, max_iter=10000)
    stats = detection_stats(preds, true_communities)
    print("BP stats")
    print(stats)
    print(f"Finished detection stats")   
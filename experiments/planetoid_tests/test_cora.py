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
    num_comms = 3
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
from algorithms.bp.old.vectorized_geometric_bp import (
    detection_stats,
    get_true_communities,
)

import numpy as np
import networkx as nx


from algorithms.duo_spec import duo_spec
import os
import json
import logging
import random
from algorithms.bp.vectorized_bp import belief_propagation_weighted

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

    G = nx.read_gml("amazon_metadata_test/amz_allviddvd.gml") # 40k+ nodes
    # G = nx.read_gml("amazon_metadata_test/amz_bookmusic.gml")  # 2k nodes 
    G = coords_str2arr(G)



    print(f"Testing on classes: {G.graph['subclasses']} and {len(G.nodes())} nodes")
    print(f"Created Graph with {len(G.nodes())} nodes and {len(G.edges())} edges")
    
    duo_params = dict(
        # spectral-EM settings
        K               = 2,                    # number of communities
        num_balls       = 32,                   # finer geometry embedding
        config          = 'motif',        # geometry estimator

        # — EM schedule
        max_em_iters    = 100,                  # allow more EM steps
        warmup_rounds   = 0,                   # hold off on any re-weighting
        anneal_steps    = 2,  #20                 # then ramp λ from 0→full over 30 iter
        comm_cut        = 0.87, #0.90
        geo_cut         = 0.87, #0.90
        shrink_comm     = 0.65, #1.00
        shrink_geo      = 0.35,
        boost_comm      = 0.40,
        boost_geo       = 0.35,
        boost_cut_comm  = 0.97, #0.97
        boost_cut_geo   = 0.97,
        # — convergence
        tol             = 1e-5,
        patience        = 10,
        random_state    = 42,
        base_seed       = 0,
        spec_params     = dict(
            dim       = 64,
            walk_len  = 40,
            num_walks = 10,
            window    = 5,
        )
    )
    
    res = duo_spec(G, **duo_params)

    preds = res['communities']
    G_fin = res['G_final']

    _, preds, _, _ = belief_propagation_weighted(
        G_fin, 
        q=2, 
        max_iter = 10000,
    )

    true_communities = get_true_communities(G, attr="comm")
    stats = detection_stats(preds, true_communities)
    print(stats)
    # logging.info(f"Finished detection stats")
    print(f"Finished detection stats") 
    
        
            
        
from algorithms.bp.old.vectorized_geometric_bp import (
    belief_propagation,
    detection_stats,
    get_true_communities,
)

import numpy as np
import networkx as nx

from algorithms.bp.old.duo_bp import (
    duo_bp,
    create_dist_observed_subgraph,
)

from algorithms.duo_spec import duo_spec
import os
import json
import logging
import random
from algorithms.bp.vectorized_bp import belief_propagation, belief_propagation_weighted
from algorithms.spectral_ops.attention import motif_spectral_embedding

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
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    
    G = nx.read_gml(os.path.join(data_dir, "amz_allviddvd.gml"))
    G = coords_str2arr(G)



    print(f"Testing on classes: {G.graph['subclasses']} and {len(G.nodes())} nodes")
    print(f"Created Graph with {len(G.nodes())} nodes and {len(G.edges())} edges")
    
    duo_params = dict(
        # spectral-EM settings
        K               = 2,                    # number of communities
        num_balls       = 32,                   # finer geometry embedding
        config          = ("bethe_hessian",     # community estimator
                        "bethe_hessian"),        # geometry estimator

        # — EM schedule
        max_em_iters    = 100,                  # allow more EM steps
        warmup_rounds   = 2,                   # hold off on any re-weighting
        anneal_steps    = 20,                   # then ramp λ from 0→full over 30 iter

        # — convergence
        tol             = 1e-5,
        patience        = 10,
        random_state    = 42,
    )
    
    res = duo_spec(G, **duo_params)

    preds = res['communities']
    # G_fin = res['G_final']

    # _, preds, _, _ = belief_propagation_weighted(
    #     G_fin, 
    #     q=2, 
    #     max_iter = 1000,
    # )

    
    preds = res["communities"]

    print(f"Finished bethe_duo_bp with {len(preds)} predictions")
    true_communities = get_true_communities(G, attr="comm")
    stats = detection_stats(preds, true_communities)
    print(stats)
    # logging.info(f"Finished detection stats")
    print(f"Finished detection stats") 
    
        
            
        
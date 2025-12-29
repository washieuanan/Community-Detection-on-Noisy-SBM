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

if __name__ == "__main__":

    # G = nx.read_gml("amazon_metadata_test/amz_allviddvd.gml") # 40k+ nodes
    # G = nx.read_gml("amazon_metadata_test/amz_bookmusic.gml")  # 2k nodes 
    G = nx.read_gml("amazon_metadata_test/amazon_hamming_videoDVD.gml")
    G = coords_str2arr(G)
    print(nx.is_connected(G))
    print(nx.number_connected_components(G))
    connect_components(G, weight=1)


    print(f"Created Graph with {len(G.nodes())} nodes and {len(G.edges())} edges")
    
    duo_params = dict(
    K               = 2,
    num_balls       = 16,    
    config          = 'motif',

    max_em_iters    = 60,
    warmup_rounds   = 0,
    anneal_steps    = 20,

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

    spec_params     = dict(
        dim       = 64,
        walk_len  = 64,
        num_walks = 20,
        window    = 5,
        weight_pow=1.0,
    )
    )
    
    res = duo_spec(G, **duo_params)

    preds_duo = res['communities']
    G_fin = res['G_final']

    _, preds_bp, _, _ = belief_propagation_weighted(
        G_fin, 
        q=2, 
        max_iter = 10000,
    )

    true_communities = get_true_communities(G, attr="comm")
    stats = detection_stats(preds_bp, true_communities)
    print("BP Results")
    print(stats)
    print("Duo Results")
    stats = detection_stats(preds_duo, true_communities)
    print(stats)
    # logging.info(f"Finished detection stats")
    print(f"Finished detection stats") 
    
        
            
        
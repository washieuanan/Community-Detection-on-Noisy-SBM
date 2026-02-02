import os
import sys

from algorithms.bp.vectorized_bp import (
    get_true_communities,
    belief_propagation,
    belief_propagation_weighted,
    detection_stats
)
from experiments.airports.load_airports import load_airports
import numpy as np
import networkx as nx
from algorithms.duo_spec import duo_spec
from algorithms.spectral_ops.attention import motif_spectral_embedding


if __name__ == "__main__":
    # Load polblogs dataset - it has 2 communities
    G = load_airports()
    true_labels = get_true_communities(G, node2idx=None, attr="comm")
    
    # ========================================================================
    # FILTER TO TOP 2 COMMUNITIES BY SIZE (comment out to use all communities)
    # ========================================================================
    unique_comms, counts = np.unique(true_labels, return_counts=True)
    top2_indices = np.argsort(counts)[-2:]  # Top 2 largest
    top2_comms = unique_comms[top2_indices]
    print(f"Original communities: {len(unique_comms)}, sizes: {dict(zip(unique_comms, counts))}")
    print(f"Filtering to top 2 communities: {top2_comms} with sizes: {counts[top2_indices]}")
    
    # Filter nodes to only those in top 2 communities
    nodes_to_keep = [n for n in G.nodes() if G.nodes[n]["comm"] in top2_comms]
    G = G.subgraph(nodes_to_keep).copy()
    
    # Relabel communities to 0 and 1
    comm_mapping = {int(top2_comms[0]): 0, int(top2_comms[1]): 1}
    for n in G.nodes():
        G.nodes[n]["comm"] = comm_mapping[G.nodes[n]["comm"]]
    
    true_labels = get_true_communities(G, node2idx=None, attr="comm")
    num_comms = 2
    print(f"After filtering: {len(G.nodes())} nodes, {len(G.edges())} edges, {num_comms} communities")
    # ========================================================================
    
    _, preds, _, _ = belief_propagation_weighted(
        G,
        q=num_comms,
        seed=0,
        init="spectral",
        max_iter=10000,
    )
    stats = detection_stats(preds, true_labels)
    print("\n=== BP Accuracy ===")
    for k, v in stats.items():
        print(f"{k:>25s} : {v}")
    
    res_duo = duo_spec(
        G,
        K=num_comms,
        max_em_iters=20,
        community_proxy="leiden"
    )
    
    G_res = res_duo['G_final']
    _, preds, _, _ = belief_propagation_weighted(
                                            G_res, 
                                            q=num_comms, 
                                            seed=0, 
                                            init="spectral",
                                            max_iter=10000,
                                                )
    stats = detection_stats(preds, true_labels)
    print("\n=== Post-Duospec BP ===")
    for k, v in stats.items():
        print(f"{k:>25s} : {v}")
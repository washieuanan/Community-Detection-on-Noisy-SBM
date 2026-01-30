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
    _, preds, _, _ = belief_propagation_weighted(
        G,
        q=4,
        seed=0,
        init="spectral",
    )
    stats = detection_stats(preds, true_labels)
    print("\n=== BP Accuracy ===")
    for k, v in stats.items():
        print(f"{k:>25s} : {v}")
    
    res_duo = duo_spec(
        G,
        K=4,
        max_em_iters=50,
        community_proxy="leiden"
    )
    
    G_res = res_duo['G_final']
    _, preds, _, _ = belief_propagation_weighted(
                                            G_res, 
                                            q=4, 
                                            seed=0, 
                                            init="spectral"
                                                )
    stats = detection_stats(preds, true_labels)
    print("\n=== Post-Duospec BP ===")
    for k, v in stats.items():
        print(f"{k:>25s} : {v}")
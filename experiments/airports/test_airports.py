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
    _, preds, _, _ = motif_spectral_embedding(G, q=4)
    # # Print DuoSpec results
    stats = detection_stats(preds, true_labels)
    print("\n=== Community‑detection accuracy MASO ===")
    for k, v in stats.items():
        print(f"{k:>25s} : {v}")
    
    
    _, preds, _, _ = belief_propagation(G, q=4)
    stats = detection_stats(preds, true_labels)
    print("\n=== Community‑detection accuracy BP ===")
    for k, v in stats.items():
        print(f"{k:>25s} : {v}")
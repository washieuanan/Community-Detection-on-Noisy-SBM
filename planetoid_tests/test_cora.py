from experiments.community_detection.bp.vectorized_geometric_bp import (
    get_true_communities,
    belief_propagation,
    detection_stats
)
from planetoid_tests.load_planetoid_data import grab_planetoid_data, to_networkx_graph
import numpy as np
import networkx as nx
from experiments.community_detection.duo_spec import duo_spec


if __name__ == "__main__":
    cora = grab_planetoid_data('Cora', ensure_undirected=True, target_dim=None, normalize_features=True)
    data = cora[0]
    G = to_networkx_graph(data)
    
    for u, v in G.edges():
        G.edges[u, v]['dist'] = np.linalg.norm(G.nodes[u]['coords'] - G.nodes[v]['coords'])
    
    res = duo_spec(G,
                   K=3,
                   num_balls=32)
    preds = res['communities']
    true_labels = get_true_communities(G, node2idx=None, attr="comm")
    stats = detection_stats(preds, true_labels)
    print("\n=== Community‑detection accuracy ===")
    for k, v in stats.items():
        print(f"{k:>25s} : {v}")
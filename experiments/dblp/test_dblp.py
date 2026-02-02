import graph_tool.all as gt
import networkx as nx
import numpy as np
from algorithms.bp.vectorized_bp import (
    get_true_communities,
    belief_propagation,
    belief_propagation_weighted,
    detection_stats
)
from algorithms.duo_spec import duo_spec

G = nx.read_gml("experiments/dblp/dblp.gml")
for u, v, d in G.edges(data=True):
    if "weight" not in d:
        d["weight"] = 1.0

# Rename the 'value' node attribute to 'comm'

# Save the fixed graph back to file
nx.write_gml(G, "experiments/polblogs/polblogs.gml")
print(f"Fixed and saved graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, undirected={not G.is_directed()}")

true_labels = get_true_communities(G, node2idx=None, attr="comm")
num_comms = len(np.unique(true_labels))

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
    max_em_iters=20,
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
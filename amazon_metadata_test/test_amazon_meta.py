from experiments.community_detection.bp.vectorized_geometric_bp import (
    get_true_communities,
    belief_propagation,
    detection_stats
)
from experiments.observations.standard_observe import PairSamplingObservation, get_coordinate_distance

from experiments.community_detection.bp.vectorized_bayes import BayesianGraphInference
from experiments.observations.sparsity import calc_num_pairs
from experiments.community_detection.bp.gbm_bp import create_observed_subgraph
import numpy as np
import networkx as nx

def coords_str2arr(G: nx.Graph, dim = 16):
    """
    for each coord, convert string formatted coord to numpy array
    """
    for n in G.nodes():
        coord_str = G.nodes[n]["coords"]
        coord_arr = np.fromstring(coord_str[1:-1], sep=",")
        if len(coord_arr) != dim:
            coord_arr = np.zeros(dim)
        G.nodes[n]["coords"] = coord_arr
    return G
if __name__ == "__main__":
    import json
    G = nx.read_gml("amazon_metadata_test/amazon_graph.gml")
    G = coords_str2arr(G)
    all_nodes_ok = True
    for n in G.nodes():
        if "comm" not in G.nodes[n]:
            print(f"Node {n} is missing the 'comm' attribute.")
            all_nodes_ok = False
        if "coords" not in G.nodes[n]:
            print(f"Node {n} is missing the 'coord' attribute.")
            all_nodes_ok = False
    if all_nodes_ok:
        print("All nodes have both 'comm' and 'coord' attributes.")
    print(f"Created Graph with {len(G.nodes())} nodes and {len(G.edges())} edges")
    num_pairs = calc_num_pairs(G, scale_factor=0.1)
    def weight_func(c1, c2):
        return np.exp(-0.5 * get_coordinate_distance(c1, c2))

    sampler = PairSamplingObservation(G, num_samples=num_pairs, weight_func=weight_func, seed=42)
    observations = sampler.observe()
    json.dump(observations, open("amazon_metadata_test/observations_01.json", "w"))
    print(f"Created {len(observations)} observations")
    obs_nodes: set[int] = set()
    for u, v in observations:
            obs_nodes.add(u)
            obs_nodes.add(v)
    print("Beginning Bayesian Inference")
    bayes = BayesianGraphInference(
        observations=observations,
        observed_nodes=obs_nodes,
        total_nodes=G.number_of_nodes(),
        obs_format="base",
        n_candidates=2 ** 20,
        seed=42,
    )
    G_pred = bayes.infer()
    print("Finished Bayesian Inference")
    subG = create_observed_subgraph(G.number_of_nodes(), observations)
    for n in subG.nodes():
        subG.nodes[n]["coord"] = G_pred.nodes[n]["coord"]

    gamma = 1.0
    K = 4
    for g in (G_pred, subG):
        for u, v in g.edges():
            d = np.linalg.norm(G_pred.nodes[u]["coord"] - G_pred.nodes[v]["coord"])
            psi = np.ones((K, K))
            np.fill_diagonal(psi, np.exp(-gamma * d))
            g[u][v]["psi"] = psi
    print(G.nodes[0]['comm'])
    print("Running Loopy BP …")
    _, preds, node2idx, idx2node = belief_propagation(
        subG,
        q=K,
        seed=42,
        init_beliefs="spectral",
        message_init="random",
        max_iter=5000,
        damping=0.10,
        balance_regularization=0.01,
    )
    print("Finished Loopy BP")
    true_labels = get_true_communities(G, node2idx=node2idx, attr="comm")
    stats = detection_stats(preds, true_labels)
    print("\n=== Community‑detection accuracy ===")
    for k, v in stats.items():
        print(f"{k:>25s} : {v}")
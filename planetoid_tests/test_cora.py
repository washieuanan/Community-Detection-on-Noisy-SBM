from experiments.community_detection.bp.vectorized_geometric_bp import (
    get_true_communities,
    belief_propagation,
    detection_stats
)
from experiments.observations.standard_observe import PairSamplingObservation, get_coordinate_distance

from experiments.community_detection.bp.vectorized_bayes import BayesianGraphInference
from planetoid_tests.load_planetoid_data import grab_planetoid_data, to_networkx_graph
from experiments.observations.sparsity import calc_num_pairs
from experiments.community_detection.bp.gbm_bp import create_observed_subgraph
import numpy as np
import networkx as nx


if __name__ == "__main__":
    cora = grab_planetoid_data('PubMed', ensure_undirected=True, target_dim=None, normalize_features=True)
    data = cora[0]
    G = to_networkx_graph(data)
    # get observations
    num_pairs = calc_num_pairs(G, scale_factor=1.0)
    def weight_func(c1, c2):
        return np.exp(-0.5 * get_coordinate_distance(c1, c2))

    sampler = PairSamplingObservation(G, num_samples=num_pairs, weight_func=weight_func, seed=42)
    observations = sampler.observe()
    obs_nodes: set[int] = set()
    for u, v in observations:
            obs_nodes.add(u)
            obs_nodes.add(v)
            
    bayes = BayesianGraphInference(
        observations=observations,
        observed_nodes=obs_nodes,
        total_nodes=G.number_of_nodes(),
        obs_format="base",
        n_candidates=2 ** 20,
        seed=42,
    )
    G_pred = bayes.infer()
    
    subG = create_observed_subgraph(G.number_of_nodes(), observations)
    for n in subG.nodes():
        subG.nodes[n]["coord"] = G_pred.nodes[n]["coord"]

    gamma = 1.0
    K = 7
    for g in (G_pred, subG):
        for u, v in g.edges():
            d = np.linalg.norm(G_pred.nodes[u]["coord"] - G_pred.nodes[v]["coord"])
            psi = np.ones((K, K))
            np.fill_diagonal(psi, np.exp(-gamma * d))
            g[u][v]["psi"] = psi
    print(G.nodes[0]['comm'])

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

    true_labels = get_true_communities(G, node2idx=node2idx, attr="comm")
    stats = detection_stats(preds, true_labels)
    print("\n=== Community‑detection accuracy ===")
    for k, v in stats.items():
        print(f"{k:>25s} : {v}")
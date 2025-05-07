import numpy as np
import networkx as nx
import scipy
from scipy.stats import qmc
from scipy.spatial import distance
from sklearn.utils.extmath import row_norms
from typing import Literal, Tuple, List, Dict, Set


class BayesianGraphInference:
    """Optimised Bayesian inference of latent positions + pseudo‑GBM reconstruction.

    Major changes from reference version
    ------------------------------------
    * O(n log n) k‑means++ seeding replaces quadratic farthest‑point
    * Pair‑likelihood matrix pre‑computed once (vectorised)
    * Posterior updates performed with NumPy broadcasting (no Python loops)
    * Vectorised distance‑weighted assignment for un‑inferred nodes
    * Vectorised pseudo‑GBM generation
    """

    def __init__(
        self,
        observations: List[Tuple[int, int]] | List[list] | List[Dict],
        observed_nodes: Set[int],
        total_nodes: int,
        obs_format: Literal["base", "GMS", "GRW"] = "base",
        dim: int = 3,
        n_candidates: int = 2 ** 20,
        seed: int | None = None,
        num_categories: int = 10
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.obs = observations
        self.d = dim
        self.obs_nodes = list(observed_nodes)
        self.num_obs = len(self.obs_nodes)
        self.n = total_nodes
        self.num_grids = 2*self.n

        self.centers, self.radius = self._split_sphere(n_candidates)

        if obs_format == "base":
            self.obs_dict = self._process_observations_base()
        elif obs_format == "GMS":
            self.obs_dict = self._process_observations_GMS()
        elif obs_format == "GRW":
            self.obs_dict = self._process_observations_GRW()
        else:
            raise ValueError("obs_format must be 'base', 'GMS' or 'GRW'")
        self.obs_format = obs_format

        # Build likelihood matrix once
        self._build_pair_likelihood_matrix()
        self.num_categories = num_categories
        self.radii_splits = np.linspace(1/num_categories, 1.0, num_categories, endpoint=True)

    def _split_sphere(self, n_candidates: int) -> Tuple[np.ndarray, float]:
        """Return (centers, radius) using k‑means++ seeding inside unit ball."""
        sobol = qmc.Sobol(self.d, scramble=False, seed=0)
        X = sobol.random(n_candidates) * 2.0 - 1.0  # map to [-1,1]^d
        inside = np.linalg.norm(X, axis=1) <= 1.0
        X = X[inside]
        if X.shape[0] < self.num_grids:
            raise ValueError(
                f"Need at least {self.num_grids} candidate points inside unit ball; got {X.shape[0]}."
            )

        centres = np.empty((self.num_grids, self.d), dtype=np.float64)
        centres[0] = X[0]
        closest2 = row_norms(X - centres[0], squared=True)

        for k in range(1, self.num_grids):
            probs = closest2 / closest2.sum()
            idx = self.rng.choice(X.shape[0], p=probs)
            centres[k] = X[idx]
            new_d2 = row_norms(X - centres[k], squared=True)
            closest2 = np.minimum(closest2, new_d2)

        radius = float(np.sqrt(closest2.max()))
        return centres, radius

    def _split_cluster_radii(self):
        """Split the cluster centers into num_categories intervals."""
        radii_assignments = np.zeros(self.num_grids, dtype=int)
        for i in range(self.num_categories - 1):
            in_category_radii = np.where((np.linalg.norm(self.centers, axis=1) < self.radii_splits[i+1])
                                         & (np.linalg.norm(self.centers, axis=1) >= self.radii_splits[i]))
            radii_assignments[in_category_radii] = i
        return radii_assignments
   
    def _split_nodes_by_degree(self):
        """split nodes into num_categories intervals based on degree"""
        degrees = np.array([len(self.obs_dict.get(node, [])) for node in range(self.n)])
        sorted_idx = np.argsort(degrees)
        category_volumes = np.zeros(self.num_categories)
        for i in range(self.num_categories - 1):
            vol = 4/3 * np.pi * self.radii_splits[i]**3
            if i > 0:
                vol -= category_volumes[i-1]
            category_volumes[i] = vol
        # Determine the number of nodes in each category based on volumes.
        counts = (category_volumes * self.n).astype(int)
        # Adjust counts so their sum equals self.n (due to rounding)
        diff = self.n - counts.sum()
        counts[-1] += diff
        node_categories = np.empty(self.n, dtype=int)
        start = 0
        for cat in range(self.num_categories):
            end = start + counts[cat]
            node_categories[sorted_idx[start:end]] = cat
            start = end
        return node_categories
        
    def _process_observations_base(self) -> Dict[int, Set[int]]:
        obs_dict = {o_n: set() for o_n in self.obs_nodes}
        for u, v in self.obs:
            obs_dict[u].add(v)
            obs_dict[v].add(u)
        return obs_dict

    def _process_observations_GRW(self) -> Dict[int, Set[int]]:
        obs_dict = {o_n: set() for o_n in self.obs_nodes}
        for g in self.obs:  # List[List[edge]]
            for u, v in g:
                obs_dict[u].add(v)
                obs_dict[v].add(u)
        return obs_dict

    def _process_observations_GMS(self) -> Dict[int, Set[int]]:
        obs_dict = {o_n: set() for o_n in self.obs_nodes}
        for grp in self.obs:  # List[Dict[radius, List[edge]]]
            for _, edge_list in grp.items():
                for u, v in edge_list:
                    obs_dict[u].add(v)
                    obs_dict[v].add(u)
        return obs_dict

    def _build_pair_likelihood_matrix(self) -> None:
        """Populate self.Lmat[cu, cv] = P(edge observed | cu, cv)."""
        exp_edges = (3.0 / 8.0) * self.n * np.log(self.n)
        W = exp_edges * np.exp(-0.5 * 0.25)  # constant part

        D = distance.cdist(self.centers, self.centers, metric="euclidean")
        W_p = np.exp(-0.5 * D)
        self.Lmat = 1.0 - np.power(1.0 - W_p / W, self.num_obs, dtype=np.float64)

   
    def _initialize_priors(self) -> None:
        radii_assignments = self._split_cluster_radii()
        node_categories = self._split_nodes_by_degree()
        self.priors = self.rng.normal(loc=1.0, scale=0.1, size=(self.n, self.num_grids))
        bias_value = 1.0  # Adjust the bias strength as needed
        for i in range(self.n):
            assigned_cat = node_categories[i]
            self.priors[i, radii_assignments == assigned_cat] += bias_value
        self.priors /= self.priors.sum(axis=1, keepdims=True)

    def _update_posteriors(self, u: int, v: int) -> None:
        cu = int(self.priors[u].argmax())
        cv = int(self.priors[v].argmax())

        # Likelihood vectors (shape n,)
        L_u = self.Lmat[:, cv]
        L_v = self.Lmat[:, cu]

        self.priors[u] *= L_u
        self.priors[v] *= L_v
        self.priors[[u, v]] /= self.priors[[u, v]].sum(axis=1, keepdims=True)

    
    def _build_obs_sequence(self, epochs: int = 1000) -> np.ndarray:
        all_pairs = np.array(
            [(u, v) for u in self.obs_nodes for v in self.obs_dict[u]], dtype=int
        )
        obs_seq = np.repeat(all_pairs, epochs, axis=0)
        self.rng.shuffle(obs_seq)
        return obs_seq

    def _infer_center_assignments(self) -> Dict[int, int]:
        self._initialize_priors()
        obs_seq = self._build_obs_sequence()
        for u, v in obs_seq:
            self._update_posteriors(int(u), int(v))
        # MAP estimate for all nodes
        self.preds = {node: int(self.priors[node].argmax()) for node in self.obs_nodes}
        return self.preds


    def _assign_unseen_nodes(self) -> Dict[int, int]:
        seen_centres = np.array(list(self.preds.values()), dtype=int)
        open_mask = np.ones(self.num_grids, dtype=bool)
        open_mask[seen_centres] = False
        open_grids = np.where(open_mask)[0]
        # if open_grids.size == 0:
        #     return {}
        # # distance from each open centre to set of assigned centres
        dist = distance.cdist(self.centers, self.centers[seen_centres], "euclidean").sum(axis=1)
        probs = dist / dist.sum()
        choices = self.rng.choice(self.centers, size=self.n - len(self.preds), replace=False, p=probs)
        return choices

    def _pseudo_gbm_gen(self, G: nx.Graph) -> nx.Graph:
        coords = np.vstack([G.nodes[n]["coords"] for n in G.nodes()])
        D = distance.squareform(distance.pdist(coords, metric="euclidean"))
        P = 0.1 * np.exp(-1.0 * (5 * D) ** 2)
        tri_mask = np.triu(np.ones_like(P, dtype=bool), k=1)
        random_mat = self.rng.random(P.shape)
        add_mask = (random_mat < P) & tri_mask
        idx_i, idx_j = np.where(add_mask)
        G.add_edges_from(zip(idx_i.tolist(), idx_j.tolist()))
        return G

    def infer(self) -> nx.Graph:
        self._infer_center_assignments()

        # Build graph with coords
        G = nx.Graph()
        for node, centre_idx in self.preds.items():
            G.add_node(node, coords=self.centers[centre_idx])

        # unseen_map = self._assign_unseen_nodes()
        # for node, centre_idx in unseen_map.items():
        #     G.add_node(node, coords=self.centers[centre_idx])
        unseen_assignments = self._assign_unseen_nodes()
        unseen_nodes = set(range(self.n)) - set(self.preds.keys())
        for ix, node in enumerate(unseen_nodes):
            G.add_node(node, coords=unseen_assignments[ix])

        # # Ensure coordinates present for all nodes
        # for node in range(self.n):
        #     if node not in G.nodes():
        #         G.add_node(node, coords=self.centers[self.rng.integers(self.num_grids)])

        # Add observed edges first (guaranteed)
        G.add_edges_from(self.obs)
        # Stochastically add further edges
        return self._pseudo_gbm_gen(G)



if __name__ == "__main__":
    from graph_generation.gbm import generate_gbm
    from observations.standard_observe import PairSamplingObservation, get_coordinate_distance
    from community_detection.bp.vectorized_geometric_bp import (
        belief_propagation,
        detection_stats,
        get_true_communities,
    )

    # Generate latent GBM graph
    G_true = generate_gbm(n=500, K=3, a=100, b=50, seed=123)
    avg_deg = np.mean([G_true.degree[n] for n in G_true.nodes()])
    original_density = avg_deg / len(G_true.nodes)
    C = 0.01 * original_density

    def weight_func(c1, c2):
        return np.exp(-0.5 * get_coordinate_distance(c1, c2))

    num_pairs = int(C * len(G_true.nodes) ** 2 / 2)
    sampler = PairSamplingObservation(G_true, num_samples=num_pairs, weight_func=weight_func, seed=42)
    observations = sampler.observe()

    obs_nodes: Set[int] = set()
    for u, v in observations:
        obs_nodes.add(u)
        obs_nodes.add(v)

    # Run Bayesian inference
    bayes = BayesianGraphInference(
        observations=observations,
        observed_nodes=obs_nodes,
        total_nodes=G_true.number_of_nodes(),
        obs_format="base",
        n_candidates=2 ** 20,
        seed=42,
    )
    G_pred = bayes.infer()

    # Build subgraph of observed nodes with inferred coords (for BP)
    from community_detection.bp.gbm_bp import create_observed_subgraph

    subG = create_observed_subgraph(G_true.number_of_nodes(), observations)
    for n in subG.nodes():
        subG.nodes[n]["coords"] = G_pred.nodes[n]["coords"]

    # Attach edge potentials & run BP (unchanged vs. original)
    gamma = 1.0
    K = 3
    for G in (G_pred, subG):
        for u, v in G.edges():
            d = np.linalg.norm(G_pred.nodes[u]["coords"] - G_pred.nodes[v]["coords"])
            psi = np.ones((K, K))
            np.fill_diagonal(psi, np.exp(-gamma * d))
            G[u][v]["psi"] = psi

    print("Running Loopy BP …")
    _, preds, node2idx, idx2node = belief_propagation(
        subG,
        q=K,
        seed=42,
        init_beliefs="spectral",
        message_init="random",
        max_iter=5000,
        damping=0.15,
        balance_regularization=0.05,
    )

    true_labels = get_true_communities(G_true, node2idx=node2idx, attr="comm")
    stats = detection_stats(preds, true_labels)
    print("\n=== Community‑detection accuracy ===")
    for k, v in stats.items():
        print(f"{k:>25s} : {v}")

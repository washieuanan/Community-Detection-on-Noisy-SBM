# assume uniform prior

# find prob of data given assumed model

# update model

# return a model and assumed params and generate a subgraph

import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import qmc
from typing import Literal
import matplotlib.pyplot as plt
import networkx as nx
class BayesianGraphInference():
    """Inference on the latent space of a graph using Bayesian methods."""
    
    def __init__(self, observations, observed_nodes, total_nodes, obs_format: tuple[Literal['base', 'GMS', 'GRW']], dim = 3, n_candidates=2**20):
        self.obs = observations
        self.d = dim
        self.obs_nodes = observed_nodes
        self.num_obs = len(observed_nodes)
        self.n = total_nodes
        self.centers, self.radius = self._split_sphere(n_candidates=n_candidates)
        if obs_format == 'base':
            self.obs_dict = self._process_observations_base()
        elif obs_format == 'GMS':
            self.obs_dict = self._process_observations_GMS()
        elif obs_format == 'GRW':
            self.obs_dict = self._process_observations_GRW()
        self.obs_format = obs_format
        
        
    def _split_sphere(self, n_candidates):
        """split the unit sphere space into num_obs balls"""
        # 1) generate deterministic Sobol candidates in [0,1]^d
        sobol = qmc.Sobol(self.d, scramble=False)
        U = sobol.random(n_candidates)
        # map to [-1,1]^d
        X = U * 2 - 1
        # keep only points inside the unit ball
        norms = np.linalg.norm(X, axis=1)
        inside = norms <= 1
        X = X[inside]
        if X.shape[0] < self.n:
            raise ValueError(f"Need more candidates inside ball, got {X.shape[0]}.")
        # 2) greedy k-center on X
        centers = np.zeros((self.n, self.d))
        centers[0] = X[0]
        dist = np.linalg.norm(X - centers[0], axis=1)
        for k in range(1, self.n):
            idx = np.argmax(dist)
            centers[k] = X[idx]
            newd = np.linalg.norm(X - centers[k], axis=1)
            dist = np.minimum(dist, newd)
        radius = dist.max()
        return centers, radius
    
    def _process_observations_base(self):
        """return dict of observations with nodes as keys"""
        obs_dict = {o_n: set() for o_n in self.obs_nodes}
        for u, v in self.obs:
            obs_dict[u].add(v)
            obs_dict[v].add(u)
        return obs_dict
    
    def _process_observations_GRW(self):
        obs_dict = {o_n: set() for o_n in self.obs_nodes}
        for g in self.obs:
            for u, v in g:
                obs_dict[u].add(v)
                obs_dict[v].add(u)
        return obs_dict
    
    def _process_observations_GMS(self):
        obs_dict = {o_n: set() for o_n in self.obs_nodes}
        for g in self.obs:
            for r in g:
                for u, v in g[r]:
                    obs_dict[u].add(v)
                    obs_dict[v].add(u)
        return obs_dict
    
    def _dist(self, c1, c2):
        """distance between two centers"""
        return np.linalg.norm(c1 - c2)
    
    def _initialize_priors(self):
        """initialize uniform priors for each vertex"""
        self.priors = np.zeros((self.n, len(self.centers)))
        for i, o_n in enumerate(self.obs_nodes):
            self.priors[i] = np.ones(len(self.centers)) + np.random.normal(0, 0.1, size=len(self.centers))
            self.priors[i] /= np.sum(self.priors[i])
    
    def _initialize_likelihood_pair(self):
        """initialize likelihood func"""
        # weight func : np.exp(-0.5 * dist(c1, c2))
        # likelihood of observing (u, v) given C(u), C(v)
        def L(cu, cv):
            exp_edges = (3/8) * self.n * np.log(self.n)
            dist = self._dist(self.centers[cu], self.centers[cv])
            w_p = np.exp(-0.5 * dist)
            W = exp_edges * np.exp(-0.5 * 4 / 3)
            m = self.num_obs
            return 1 - (1 - w_p/W)**m
        self.likelihood = L
    
    def _initialize_likelihood_GRW(self):
        """initialize likelihood func"""
        # random walk has exponential prob dist for each neighbor
        self.likelihood = None
    
    def _initialize_likelihood_GMS(self):
        """initialize likelihood func"""
        # take advantage of radius to estimate likelihood --> Bernoulli likelihood that
        # observation must be within some radius of center
        self.likelihood = None
    
    def _update_posterior(self, obs):
        """update posterior using Bayes rule"""
        posterior_u = np.zeros(len(self.centers))
        posterior_v = np.zeros(len(self.centers))
        u, v = obs
        for i, c in enumerate(self.centers):
            # update posterior for center c
            # likelihood of observing (u, v) given C(u), C(v)
            c_v = np.argmax(self.priors[v]) # correct prediction for each node
            c_u = np.argmax(self.priors[u])
            prior_v = self.priors[v][i]
            prior_u = self.priors[u][i]
            L_u = self.likelihood(i, c_v) # likelihood for u given c_v
            L_v = self.likelihood(i, c_u)
            
            posterior_u[i] = L_u * prior_u
            posterior_v[i] = L_v * prior_v
        
        # normalize
        posterior_u /= np.sum(posterior_u)
        posterior_v /= np.sum(posterior_v)
        self.priors[u] = posterior_u
        self.priors[v] = posterior_v
    
    def _build_obs_seq(self, epochs=20):
        """build sequence of observations"""
        all_pairs = [(u, v) for u in self.obs_nodes for v in self.obs_dict[u]] * epochs
        np.random.shuffle(all_pairs)
        return all_pairs
    
    def _infer_center_assignments(self):
        """bayesian inference algorithm"""
        self._initialize_priors()
        if self.obs_format == 'base':
            self._initialize_likelihood_pair()
        elif self.obs_format == 'GMS':
            self._initialize_likelihood_GMS()
        elif self.obs_format == 'GRW':
            self._initialize_likelihood_GRW()
        
        obs_seq = self._build_obs_seq()
        for u, v in obs_seq:
            self._update_posterior((u, v))

        self.preds = {}
        for node in self.obs_nodes:
            self.preds[node] = np.argmax(self.priors[node])
        return self.preds
    
    def infer(self):
        """infer center assignments"""
        self._infer_center_assignments()
        G = nx.Graph()
        open_nodes = [i for i in range(self.n) if i not in list(self.preds.values())]
        open_nodes_dist_weights = np.zeros(len(open_nodes))
        for ix, oni in enumerate(open_nodes):
            open_nodes_dist_weights[ix] = np.sum([self._dist(self.centers[oni], self.centers[i]) for i in self.preds.values()])
        open_nodes_dist_weights /= np.sum(open_nodes_dist_weights)
        for node in range(self.n):
            if node in self.preds:
                G.add_node(node, coord=self.centers[self.preds[node]])
            else:
                rand_node = np.random.choice(open_nodes, p=open_nodes_dist_weights)
                G.add_node(node, coord=self.centers[rand_node])
        for u, v in self.obs:
            G.add_edge(u, v)
        
        pred_graph = self._pseudo_gbm_gen(G)
        return pred_graph


    def _pseudo_gbm_gen(self, G):
        """generate graph from inferred graph"""
        for i in range(len(G.nodes)):
            for j in range(i + 1, len(G.nodes)):
                if G.has_edge(i, j):
                    continue
                d = np.linalg.norm(G.nodes[i]['coord'] - G.nodes[j]['coord'])
                p = 0.1 * np.exp(-1*(5*d)**2)
                if np.random.random_sample() < np.clip(p, 0, 1):
                    G.add_edge(i, j)
        return G
    
if __name__ == "__main__":
    from graph_generation.gbm import generate_gbm
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    from observations.standard_observe import (
        PairSamplingObservation,
        get_coordinate_distance,
    )
    from observations.sensor_observe import GroupedMultiSensorObservation

    from community_detection.bp.geometric_belief_prop import (
        detection_stats,
        initialize_beliefs,
        belief_propagation,
        get_marginals_and_preds,)
    from community_detection.bp.gbm_bp import (create_observed_subgraph)
    G2 = generate_gbm(n=500, K=3, a = 100, b = 50, seed=123)
    avg_deg = np.mean([G2.degree[n] for n in G2.nodes()])
    orig_sparsity = avg_deg/len(G2.nodes)
    print("Original Sparsity:", orig_sparsity)
    C = 0.15 * orig_sparsity                      
    def weight_func(c1, c2):
        return np.exp(-0.5 * get_coordinate_distance(c1, c2))

    num_pairs = int(C * len(G2.nodes) ** 2 / 2)
    print("NUM EDGES:", len(G2.edges))
    print("SAMPLINHG EDGES:", num_pairs)

    avg_degree = np.mean(list(dict(G2.degree()).values()))
    original_density = avg_degree / len(G2.nodes)
    C = 0.1 * original_density 
    num_sensors = max(1, int((C * len(G2.nodes)) / ((0.25)**3 * avg_degree)))


    # Pair-based sampling
    # pair_sampler = PairSamplingObservation(G2, num_samples=num_pairs, weight_func=weight_func, seed=42)
    grouped_multi_sensor_sampler = GroupedMultiSensorObservation(G2, 
                                                                 seed = 42,
                                                                 num_sensors=num_sensors, 
                                                                 min_sep=0.2, 
                                                                 radii=np.linspace(0.1, 1.0, 10),
                                                                 deduplicate_edges=True)
    observations = grouped_multi_sensor_sampler.observe()
    observed_nodes = set()
    

    # for group ms 
    for sensor_obs in observations:   
        for k,v in sensor_obs.items(): 
            for obs in v: 
                observed_nodes.add(obs[0])
                observed_nodes.add(obs[1])
    
    
    # for u, v in observations:
    #     observed_nodes.add(u)
    #     observed_nodes.add(v)

    bayes = BayesianGraphInference(
        observations=observations,
        observed_nodes=observed_nodes,
        total_nodes=len(G2.nodes),
        obs_format='GMS',
        n_candidates=2**20
    )
    
    pred_graph = bayes.infer()
    # import matplotlib.pyplot as plt

    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot(111, projection='3d')

    # # Plot a unit sphere
    # phi = np.linspace(0, np.pi, 50)
    # theta = np.linspace(0, 2 * np.pi, 50)
    # phi, theta = np.meshgrid(phi, theta)
    # x_sphere = np.sin(phi) * np.cos(theta)
    # y_sphere = np.sin(phi) * np.sin(theta)
    # z_sphere = np.cos(phi)
    # ax.plot_surface(x_sphere, y_sphere, z_sphere, color='lightblue', alpha=0.3, linewidth=0)

    # # Plot graph nodes
    # node_coords = np.array([pred_graph.nodes[n]['coord'] for n in pred_graph.nodes])
    # xs, ys, zs = node_coords[:, 0], node_coords[:, 1], node_coords[:, 2]
    # ax.scatter(xs, ys, zs, color='red', s=50)

    # # Plot graph edges
    # for u, v in pred_graph.edges():
    #     coord_u = np.array(pred_graph.nodes[u]['coord'])
    #     coord_v = np.array(pred_graph.nodes[v]['coord'])
    #     ax.plot([coord_u[0], coord_v[0]],
    #             [coord_u[1], coord_v[1]],
    #             [coord_u[2], coord_v[2]],
    #             color='grey', alpha=0.5)

    # ax.set_xlim([-1.2, 1.2])
    # ax.set_ylim([-1.2, 1.2])
    # ax.set_zlim([-1.2, 1.2])
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_title('Inferred Graph on Unit Sphere')
    # plt.show()
    subG = create_observed_subgraph(len(G2.nodes), observations)

    
    for n in subG.nodes():
        subG.nodes[n]['coord'] = pred_graph.nodes[n]['coord']


    

    gamma = 1.0       
    K = 2             

    for G in (pred_graph, subG):
        for u, v in G.edges():
            # Euclidean distance in latent space
            d = np.linalg.norm(pred_graph.nodes[u]['coord'] - pred_graph.nodes[v]['coord'])
            # start with no effect off-diagonal, repulsive on the diagonal
            psi = np.ones((K, K))
            np.fill_diagonal(psi, np.exp(-gamma * d))
            # store it on the edge
            G[u][v]['psi'] = psi

    # # subG =iG2
    initialize_beliefs(pred_graph, 2)
    initialize_beliefs(subG, 3)
    # belief_propagation(
    #     pred_graph,
    #     q=2,
    #     max_iter=5000,
    #     damping=0.15,
    #     balance_regularization=0.05,
    #     min_steps=50,
    #     message_init="random",
    #     group_obs=None,
    #     min_sep=0.15,
    # )
    belief_propagation(
        subG,
        q=3,
        max_iter=5000,
        damping=0.15,
        balance_regularization=0.05,
        min_steps=50,
        message_init="pre-group",
        group_obs=observations,
        min_sep=0.15,
    )
    # marginals, preds = get_marginals_and_preds(pred_graph)
    # cluster_map = np.array(cluster_map2)
    cluster_map = nx.get_node_attributes(G2, "comm")
    cluster_map = np.array(list(cluster_map.values()))
    # sub_preds = np.array([preds[i] for i in range(len(preds)) if i in observed_nodes])
    # sub_cluster_map = np.array(
    #     [cluster_map[i] for i in range(len(cluster_map)) if i in observed_nodes]
    # )
    # print(detection_stats(preds, cluster_map))
    # print(detection_stats(sub_preds, sub_cluster_map))
    
    submarginals_fuck, sub_preds_fuck = get_marginals_and_preds(subG)
    # cluster_map = np.array(cluster_map2)
    sub_preds_subG = np.array([sub_preds_fuck[i] for i in range(len(sub_preds_fuck)) if i in observed_nodes])
    sub_cluster_map_subG = np.array(
        [cluster_map[i] for i in range(len(cluster_map)) if i in observed_nodes]
    )
    print(detection_stats(sub_preds_fuck, cluster_map))
    print(detection_stats(sub_preds_subG, sub_cluster_map_subG))
    
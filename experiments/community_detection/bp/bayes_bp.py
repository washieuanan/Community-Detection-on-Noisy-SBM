# assume uniform prior

# find prob of data given assumed model

# update model

# return a model and assumed params and generate a subgraph

import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import qmc
from typing import Literal
class BayesianGraphInference():
    """Inference on the latent space of a graph using Bayesian methods."""
    
    def __init__(self, observations, observed_nodes, total_nodes, obs_format: tuple[Literal['base', 'GMS', 'GRW']], n_candidates=2**20):
        self.obs = observations
        self.d = dim
        self.obs_nodes = observed_nodes
        self.num_obs = len(observed_nodes)
        self.centers, self.radius = self._split_sphere(n_candidates=n_candidates)
        if obs_format == 'base':
            self.obs_dict = self._process_observations_base()
        elif obs_format == 'GMS':
            self.obs_dict = self._process_observations_GMS()
        elif obs_format == 'GRW':
            self.obs_dict = self._process_observations_GRW()
        self.obs_format = obs_format
        self.n = total_nodes
        
        
    def _split_sphere(self):
        """split the unit sphere space into num_obs balls"""
        # TODO: verify this function
        sobol = qmc.Sobol(self.d, scramble=False)
        U = sobol.random(n_candidates)
        X = U * 2 - 1
        norms = np.linalg.norm(X, axis=1)
        mask = norms > eps              
        X = X[mask] / norms[mask, None]
        if X.shape[0] < self.num_obs:
            raise ValueError("Not enough sphere candidates; increase n_candidates or lower num_obs.")
        centers = np.empty((self.num_obs, self.d))
        centers[0] = X[0]
        dist = np.linalg.norm(X - centers[0], axis=1)
        for k in range(1, self.num_obs):
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
        self.priors = self.ndarray(len(self.obs_nodes), len(self.centers))
        for i, o_n in enumerate(self.obs_nodes):
            self.priors[i] = np.ones(len(self.centers)) + np.random.normal(0, 0.1, size=len(self.centers))
            self.priors[i] /= np.sum(self.priors[i])
    
    def _initialize_likelihood_pair(self):
        """initialize likelihood func"""
        # weight func : np.exp(-0.5 * dist(c1, c2))
        # likelihood of observing (u, v) given C(u), C(v)
        def L(cu, cv):
            exp_edges = np.log(self.n)
            dist = self._dist(self.centers[cu], self.centers[cv])
            w_p = np.exp(-0.5 * dist)
            m = self.num_obs
            return (w_p * m)/(exp_edges - m + 1)
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
            prior_v = self.priors[v][c]
            prior_u = self.priors[u][c]
            L_u = self.likelihood(c, c_v) # likelihood for u given c_v
            L_v = self.likelihood(c, c_u)
            
            posterior_u[i] = L_u * prior_u
            posterior_v[i] = L_v * prior_v
        
        # normalize
        posterior_u /= np.sum(posterior_u)
        posterior_v /= np.sum(posterior_v)
        self.priors[u] = posterior_u
        self.priors[v] = posterior_v
    
    def _build_obs_seq(self):
        """build sequence of observations"""
        all_pairs = np.array([(u, v) for u in self.obs_nodes for v in self.obs_dict[u]])
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
            self.preds[node] = self.centers[np.argmax(self.priors[node])]
        return self.preds
    
    def infer(self):
        """infer center assignments"""
        self._infer_center_assignments()
        G = nx.Graph()
        for node in range(self.n):
            if n in self.preds:
                G.add_node(node, coord=self.preds[node])
            else:
                G.add_node(node, coord=np.random.choice(self.centers))
        for u, v in self.obs:
            G.add_edge(u, v)
        
        pred_graph = self._pseudo_gbm_gen(G)
        return pred_graph


    def _pseudo_gbm_gen(self, G):
        """generate graph from inferred graph"""
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if G.has_edge(i, j):
                    continue
                d = np.linalg.norm(G.nodes[i]['coord'] - G.nodes[j]['coord'])
                p = 0.5 * np.exp(-0.5 * (d/0.2)**2)
                if np.random.random_sample() < np.clip(p, 0, 1):
                    G.add_edge(i, j)
        return G
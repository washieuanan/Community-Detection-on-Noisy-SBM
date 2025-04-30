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
    
    def __init__(self, observations, observed_nodes, obs_format: tuple[Literal['base', 'GMS', 'GRW']], n_candidates=2**20):
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
        
        
    def _split_sphere(self):
        """split the unit sphere space into num_obs balls"""
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
    
    def _initialize_priors(self):
        """initialize uniform priors for each vertex"""
        self.priors = self.ndarray(len(self.obs_nodes), len(self.centers))
        for i, o_n in enumerate(self.obs_nodes):
            self.priors[i] = np.ones(len(self.centers)) / len(self.centers)
    
    def _initialize_likelihood_base(self):
        """initialize likelihood func"""
        # use exponential distribution from weight function to estimate likelihood
        self.likelihood = None
    
    def _initialize_likelihood_GRW(self):
        """initialize likelihood func"""
        # random walk is a uniform distribution over edges in the node
        self.likelihood = None
    
    def _initialize_likelihood_GMS(self):
        """initialize likelihood func"""
        # take advantage of radius to estimate likelihood --> Bernoulli likelihood that
        # observation must be within some radius of center
        self.likelihood = None
    
    def _update_posterior(self, data):
        """update posterior using Bayes rule"""
        pass
    
    def infer_graph(self):
        """bayesian inference algorithm"""
        self._initialize_priors()
        if self.obs_format == 'base':
            self._initialize_likelihood_base()
        elif self.obs_format == 'GMS':
            self._initialize_likelihood_GMS()
        elif self.obs_format == 'GRW':
            self._initialize_likelihood_GRW()
                
        pass
    
from typing import List, Tuple, Any, Optional

import numpy as np
import networkx as nx
from scipy.special import expit  
from numpy.linalg import norm
from scipy.sparse.csgraph import laplacian
from sklearn.cluster import KMeans

#pylint: disable=unused-import, redefined-outer-name
from experiments.graph_generation.gbm import generate_gbm
from deprecated.observations.random_walk_obs import random_walk_observations
from deprecated.observations.standard_observe import get_coordinate_distance
from algorithms.detect import Detection


def initialize_latent(G, dim):
    """Spectral initialization of latent positions on the sphere."""
    # build adjacency matrix
    A = nx.to_scipy_sparse_array(G, format='csr')
    # normalized Laplacian
    L = laplacian(A, normed=True)
    # eigen-decomposition
    vals, vecs = np.linalg.eigh(L.toarray())
    X = vecs[:, 1:dim+1]
    # normalize to unit sphere
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-10
    return X

def initialize_params(G, observations, K, dim):
    """Initialize parameters for EM."""
    N = G.number_of_nodes()
    nodes = list(G.nodes())
    # latent positions X
    X = initialize_latent(G, dim)
 
    
    km = KMeans(n_clusters=K, random_state=0).fit(X)
    q = np.zeros((N, K))
    q[np.arange(N), km.labels_] = 1
    # mixing weights
    pi = q.mean(axis=0)
    # cluster centers on sphere
    mu = np.zeros((K, dim))
    for k in range(K):
        mean_vec = (q[:, k:k+1] * X).sum(axis=0)
        mu[k] = mean_vec / (norm(mean_vec) + 1e-10)
    # concentration parameter
    kappa = np.ones(K) * 5.0
    # logistic link parameters
    beta = 1.0
    beta0 = 0.0
    return nodes, X, q, pi, mu, kappa, beta, beta0

def e_step(X, mu, kappa, pi):
    """E-step: update responsibilities q."""
    N, K = len(X), mu.shape[0]
    log_prior = np.log(pi + 1e-10)  # shape (K,)
    logits = np.zeros((N, K))
    for k in range(K):
        logits[:, k] = log_prior[k] + kappa[k] * (X @ mu[k])
    # stabilize
    max_vals = logits.max(axis=1)
    logits -= max_vals[:, np.newaxis]
    q = np.exp(logits)
    sum_vals = q.sum(axis=1)
    q /= sum_vals[:, np.newaxis]
    return q

def m_step_pi(q):
    """M-step: update mixing weights pi."""
    return q.mean(axis=0)

def m_step_mu(X, q):
    """M-step: update cluster centers mu."""
    K = q.shape[1]
    mu = np.zeros((K, X.shape[1]))
    for k in range(K):
        weighted = (q[:, k:k+1] * X).sum(axis=0)
        mu[k] = weighted / (norm(weighted) + 1e-10)
    return mu

def m_step_beta(X, observations, beta, beta0, lr=0.1, iters=100):
    """Gradient ascent to update beta and beta0."""
    for _ in range(iters):
        grad_b = 0.0
        grad_b0 = 0.0
        for i, j in observations:
            xi, xj = X[i], X[j]
            dot = beta * xi.dot(xj) + beta0
            p = expit(dot)
            # gradient of log-likelihood for edge =1
            grad = 1 - p
            grad_b += grad * xi.dot(xj)
            grad_b0 += grad
        # negative samples (non-edges)
        # sample a subset for efficiency
        M = len(observations)
        nodes = X.shape[0]
        for _ in range(M):
            i, j = np.random.choice(nodes, 2, replace=False)
            if (i, j) not in observations and (j, i) not in observations:
                xi, xj = X[i], X[j]
                dot = beta * xi.dot(xj) + beta0
                p = expit(dot)
                grad = -p
                grad_b += grad * xi.dot(xj)
                grad_b0 += grad
        beta  += lr * grad_b
        beta0 += lr * grad_b0
    return beta, beta0

def m_step_X(X, observations, beta, beta0, mu, kappa, q, lr=0.01, iters=10):
    """Gradient ascent on each latent x_i on the sphere."""
    N, dim = X.shape
    for _ in range(iters):
        grads = np.zeros_like(X)
        # edge likelihood term
        for i, j in observations:
            xi, xj = X[i], X[j]
            dot = beta * xi.dot(xj) + beta0
            p = expit(dot)
            grad_i = beta * (1 - p) * xj
            grads[i] += grad_i
            grads[j] += beta * (1 - p) * xi
        # prior term from responsibilities
        for i in range(N):
            for k in range(mu.shape[0]):
                grads[i] += q[i, k] * kappa[k] * mu[k]
        # update and re-normalize on sphere
        X += lr * grads
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-10
    return X

def latent_space_em(G, observations, K, dim=2, max_iter=20):
    """
    Runs the EM algorithm for the latent-space model.

    Returns
    -------
    dict with final 'X', 'q', 'pi', 'mu', 'kappa', 'beta', 'beta0'
    """
    # create observed subgraph
    G_obs = nx.Graph()
    for n in G.nodes():
        G_obs.add_node(n)
    for u, v in observations:
        G_obs.add_edge(u, v)
    nodes, X, q, pi, mu, kappa, beta, beta0 = initialize_params(G_obs, observations, K, dim)
    for it in range(max_iter):
        # E-step
        q = e_step(X, mu, kappa, pi)
        # M-step
        pi = m_step_pi(q)
        mu = m_step_mu(X, q)
        # we could update kappa via MoM
        beta, beta0 = m_step_beta(X, observations, beta, beta0)
        X = m_step_X(X, observations, beta, beta0, mu, kappa, q)
        print(f"Iteration {it+1}/{max_iter} complete.")
    return {
        'nodes': nodes, 'X': X, 'q': q, 'pi': pi,
        'mu': mu, 'kappa': kappa, 'beta': beta, 'beta0': beta0
    }

class LatentSpaceEMDetection(Detection):
    """
    EM detection using the latent-space model.
    """
    def __init__(
        self,
        graph: nx.Graph,
        observations: List[Tuple[Any, Any]],
        K: int,
        dim: int = 2,
        max_iter: int = 20
    ):
        super().__init__(graph, observations, K)
        self.dim = dim
        self.max_iter = max_iter

    def output(self) -> np.ndarray:
        """
        Perform EM and return hard community labels.
        """
        results = latent_space_em(
            self.graph,
            self.observations,
            self.k,
            dim=self.dim,
            max_iter=self.max_iter
        )
        q = results['q']
        labels = np.argmax(q, axis=1)
        return labels

def detection_stats(preds, true):
    """
    Calculates basic stats for community detection.

    Parameters
    ----------
    preds : array-like, shape (n_nodes,)   
        Predicted community labels (ints from 0..K-1).
    true : array-like, shape (n_nodes,)
        True community labels (ints from 0..K-1).

    Returns
    -------
    stats : dict
        {
         "accuracy": float,          # overall fraction correct
         "accuracy_0": float,        # per-community accuracies
         "accuracy_1": float,
         ...
        }
    """
    preds = np.asarray(preds)
    true = np.asarray(true)
    num_communities = np.unique(true).size

    true_grouping = {c: np.where(true == c)[0] for c in range(num_communities)}
    pred_grouping = {c: np.where(preds == c)[0] for c in range(num_communities)}

    perm = np.zeros(num_communities, dtype=int)
    for c in range(num_communities):
        best_match, best_size = 0, 0
        for c2 in range(num_communities):
            size = np.intersect1d(pred_grouping[c], true_grouping[c2]).size
            if size > best_size:
                best_size, best_match = size, c2
        perm[c] = best_match

    correct = (true == perm[preds]).sum()
    stats = {"accuracy": correct / len(preds)}

    for c in range(num_communities):
        idxs = true_grouping[c]
        stats[f"accuracy_{c}"] = (perm[preds[idxs]] == c).sum() / len(idxs)

    return stats


def eval_em_accuracy(nodes, q, G):
    """
    Evaluates EM-based community assignments.

    Parameters
    ----------
    nodes : list
        Ordering of nodes corresponding to rows in q.
    q : array-like, shape (n_nodes, K)
        Soft assignments (responsibilities) from EM.
    G : networkx.Graph
        Each node in nodes must have attribute "cluster" (true label).

    Returns
    -------
    stats : dict
        Detection stats (accuracy, per-community accuracies).
    """
    # Hard assignments via max responsibility
    preds = np.argmax(q, axis=1)

    true = np.array([G.nodes[n]['comm'] for n in nodes], dtype=int)
    return detection_stats(preds, true)


if __name__ == "__main__":
    # distributions = ['normal', 'normal', 'normal']
    # dist_params = [
    #     {'loc':[-0.3, 0.1, 0.3], 'scale':0.2, 'constrain_to_unit_sphere':True},
    #     {'loc':[0.1, 0.4, -0.2], 'scale':0.1, 'constrain_to_unit_sphere':True},
    #     {'loc':[0.3, -0.1, 0.1], 'scale':0.2, 'constrain_to_unit_sphere':True},
    # ]
    
    # def weight_func(coord1, coord2):
    #     distance = get_coordinate_distance(coord1, coord2)
    #     return np.exp(-0.4 * distance)
    
    # def prob_weight_func(dist):
    #     return max(np.exp(-0.6 * dist), 0.1)
    
    # G2, coords2, cluster_map2 = generate_latent_geometry_graph(
    #     [40, 40, 40], 
    #     distributions=distributions,
    #     dist_params=dist_params,
    #     edge_prob_fn= prob_weight_func # Higher threshold to create more edges
    # )  
    G2 = generate_gbm(
        n=300,
        K=3,
        a = 100, 
        b = 50,
        seed=123
    )

    observations = random_walk_observations(G2, num_walkers=8, stopping_param=0.2, leaky=0.1)

    em_results = latent_space_em(G2, observations, 3, dim=2, max_iter=1000)
    print(em_results)
    print(eval_em_accuracy(em_results['nodes'], em_results['q'], G2))





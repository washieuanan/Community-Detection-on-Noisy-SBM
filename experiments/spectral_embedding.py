import networkx as nx
import numpy as np
from scipy.sparse import csgraph
from sklearn.cluster import KMeans

from generate_graph import generate_latent_geometry_graph
from random_walk_obs import random_walk_observations
from observe import get_coordinate_distance

def spectral_embedding_clustering(G, observations, k):
    """
    Spectral embedding + k-means clustering on observed edges.

    Parameters
    ----------
    G : networkx.Graph
        The full set of vertices (possibly without all edges observed).
    observations : list of (u, v)
        Observed edges between nodes u and v.
    k : int
        Number of communities to find.

    Returns
    -------
    dict
        Mapping node -> community label in {0,1,...,k-1}.
    """
    # Build a graph H containing only your observed edges, but with the same node set as G so isolated nodes are included.
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    H.add_edges_from(observations)

    # Get the (normalized) Laplacian matrix as a sparse SciPy array, with rows/cols in the order of G.nodes().
    nodes = list(G.nodes())
    A = nx.to_scipy_sparse_array(H, nodelist=nodes, format="csr")
    L = csgraph.laplacian(A, normed=True)

    # Compute the first k+1 eigenpairs of L (we skip the 0-th eigenvector which is constant)
    _, vecs = np.linalg.eigh(L.toarray())

    # Take the eigenvectors 1..k (columns 1 through k)
    X = vecs[:, 1 : k + 1]  # shape = (n_nodes, k)

    # Cluster rows of X with k-means
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X)
    labels = km.labels_

    # Build the mapping back to the original node IDs
    return {node: int(labels[i]) for i, node in enumerate(nodes)}


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


def eval_spectral_accuracy(G, preds):
    """
    Aligns your predicted labels with the true labels stored in G.nodes[n]['cluster'],
    then returns the detection_stats.

    Parameters
    ----------
    G : networkx.Graph
        Each node must have a node‐attribute "cluster" = its true community (int).
    preds : dict
        Mapping node -> predicted community label (int).

    Returns
    -------
    stats : dict
        As from detection_stats.
    """

    nodes = list(G.nodes())

    true = np.array([G.nodes[n]["cluster"] for n in nodes], dtype=int)
    pred = np.array([preds[n] for n in nodes], dtype=int)

    return detection_stats(pred, true)


if __name__ == "__main__":
    distributions = ['normal', 'normal', 'normal']
    dist_params = [
        {'loc':[-0.3, 0.1, 0.3], 'scale':0.2, 'constrain_to_unit_sphere':True},
        {'loc':[0.1, 0.4, -0.2], 'scale':0.1, 'constrain_to_unit_sphere':True},
        {'loc':[0.3, -0.1, 0.1], 'scale':0.2, 'constrain_to_unit_sphere':True},
    ]
    
    def weight_func(coord1, coord2):
        distance = get_coordinate_distance(coord1, coord2)
        return np.exp(-0.4 * distance)
    
    def prob_weight_func(dist):
        return max(np.exp(-0.6 * dist), 0.1)
    
    G2, coords2, cluster_map2 = generate_latent_geometry_graph(
        [40, 40, 40], 
        distributions=distributions,
        dist_params=dist_params,
        edge_prob_fn= prob_weight_func 
    )  

    observations = random_walk_observations(G2, num_walkers=8, stopping_param=0.2, leaky=0.1)

    detection = spectral_embedding_clustering(G2, observations, 3)
    print(detection)
    print(eval_spectral_accuracy(G2, detection))




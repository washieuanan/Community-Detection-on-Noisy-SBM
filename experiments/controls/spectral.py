import networkx as nx
import numpy as np
from scipy.sparse import csgraph
from sklearn.cluster import KMeans

from graph_generation.generate_graph import generate_latent_geometry_graph
from observations.random_walk_obs import random_walk_observations
from observations.observe import get_coordinate_distance

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
    # Build a graph H containing only edges from observation
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    H.add_edges_from(observations)

    # Get the Laplacian matrix as a sparse array, with rows/cols in the order of G.nodes
    nodes = list(G.nodes())
    A = nx.to_scipy_sparse_array(H, nodelist=nodes, format="csr")
    L = csgraph.laplacian(A, normed=True)

    _, vecs = np.linalg.eigh(L.toarray())

    X = vecs[:, 1 : k + 1]  # shape = (n_nodes, k)

    # Cluster rows of X with k-means
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X)
    labels = km.labels_

    return np.array(int(labels[i]) for i in range(len(labels)))

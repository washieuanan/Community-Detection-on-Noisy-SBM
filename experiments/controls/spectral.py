import networkx as nx
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from scipy.sparse import csgraph

def bethe_hessian_clustering(G, observations, q, use_nonbacktracking=False):
    """
    Community detection via the Bethe–Hessian operator on a weighted graph.

    Parameters
    ----------
    G : networkx.Graph
        The full set of vertices (possibly without all edges observed).
    observations : list of (u, v, d_uv)
        Observed pairs with their Euclidean distance.
    q : int
        Number of communities.
    use_nonbacktracking : bool
        If True, estimate r from the non‐backtracking operator; else use sqrt(mean degree).

    Returns
    -------
    np.ndarray
        Array of length n, where entry i is the community label of node i.
    """
    # 1) build weighted adjacency from (u, v, d)
    dists = np.array([d for (u,v), d in observations])
    sigma = np.median(dists) or 1.0
    weight = lambda d: np.exp(-0.5 * (d / sigma) ** 2)

    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    H.add_weighted_edges_from([(u, v, weight(d)) for (u, v), d in observations], weight="w")

    # 2) form adjacency A and degree diagonal D
    nodes = list(G.nodes())
    A = nx.to_scipy_sparse_array(H, nodelist=nodes, format="csr", weight="w")
    deg = np.array(A.sum(axis=1)).ravel()
    D = diags(deg)

    # 3) choose parameter r
    if use_nonbacktracking:
        # Placeholder for non-backtracking estimate:
        # compute leading eigenvalue rho of the non-backtracking matrix B, then:
        # r = np.sqrt(rho)
        raise NotImplementedError("Non-backtracking estimate not yet implemented")
    else:
        r = np.sqrt(deg.mean())

    # 4) form the Bethe–Hessian H(r)
    n = len(nodes)
    I = diags([1.0] * n)
    Hr = (r * r - 1) * I - r * A + D

    # 5) find the q eigenvectors of H(r) with the most negative eigenvalues
    vals, vecs = eigsh(-Hr, k=q, which="LA")
    labels_sub = vecs  # Not used directly, but km.labels_ refers to this embedding

    # 6) cluster in that embedding space
    km = KMeans(n_clusters=q, random_state=42).fit(vecs)
    labels_sub = km.labels_

    # 7) format labels exactly as requested:
    nodes_sub = nodes
    n_sub = len(nodes_sub)
    labels = {nodes_sub[i]: int(labels_sub[i]) for i in range(n_sub)}

    rng = np.random.default_rng()
    other_nodes = set(G.nodes()) - set(nodes_sub)
    for node in other_nodes:
        labels[node] = int(rng.integers(0, q))

    return np.array([labels[i] for i in range(len(nodes))])

def spectral_clustering_community_detection(G: nx.Graph, K: int) -> np.ndarray:
    """
    Perform spectral clustering on G to recover K communities.

    Parameters
    ----------
    G : networkx.Graph
        Undirected graph with n nodes.
    K : int
        Number of communities.

    Returns
    -------
    np.ndarray of shape (n,)
        Community labels in {0, 1, …, K-1}, ordered to match sorted(G.nodes()).
    """
    # 1) Remap nodes to indices 0..n-1
    nodes = list(G.nodes())
    idx = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)

    # 2) Build the adjacency matrix A (CSR sparse format)
    #    using nx.adjacency_matrix, which returns a SciPy CSR matrix
    A = nx.adjacency_matrix(G, nodelist=nodes)

    # 3) Compute the normalized Laplacian L_norm = D^{-1/2} (D - A) D^{-1/2}
    L_norm = csgraph.laplacian(A, normed=True)

    # 4) Compute the first K eigenvectors of the normalized Laplacian
    #    (corresponding to the K smallest eigenvalues)
    vals, vecs = eigsh(L_norm, k=K, which='SM')

    # 5) Cluster the rows of the eigenvector matrix with KMeans
    km = KMeans(n_clusters=K, n_init=10, random_state=0)
    labels = km.fit_predict(vecs)

    return labels
# def spectral_clustering(G, observations, q):
#     """
#     Community detection via classic spectral clustering on an unweighted graph
#     built from observed vertex‐pairs.

#     Parameters
#     ----------
#     G : networkx.Graph
#         The full set of vertices.
#     observations : list of (u, v, d_uv)
#         Observed pairs (with distances d_uv, which will be ignored here).
#     q : int
#         Number of communities.

#     Returns
#     -------
#     np.ndarray
#         Array of length n, where entry i is the community label of node i.
#     """
#     # 1) build unweighted observation‐graph H
#     H = nx.Graph()
#     H.add_nodes_from(G.nodes())
#     H.add_edges_from([(u, v) for (u, v),  _ in observations])

#     # 2) form the normalized Laplacian L
#     nodes = list(G.nodes())
#     A = nx.to_scipy_sparse_array(H, nodelist=nodes, format="csr")
#     L = csgraph.laplacian(A, normed=True)

#     # 3) compute the bottom q eigenvectors of L (skip the trivial zero‐mode)
#     evals, evecs = np.linalg.eigh(L.toarray())
#     X = evecs[:, 1 : q + 1]  # shape = (n_nodes, q)

#     # 4) cluster rows of X with k‐means
#     km = KMeans(n_clusters=q, random_state=42).fit(X)
#     labels_sub = km.labels_

#     # 5) format labels exactly as requested
#     nodes_sub = nodes
#     n_sub = len(nodes_sub)
#     labels = {nodes_sub[i]: int(labels_sub[i]) for i in range(n_sub)}

#     # 6) assign random labels for any missing nodes (shouldn't happen here)
#     rng = np.random.default_rng()
#     other_nodes = set(G.nodes()) - set(nodes_sub)
#     for node in other_nodes:
#         labels[node] = int(rng.integers(0, q))

#     # 7) return array of labels in node‐index order
#     return np.array([labels[i] for i in range(len(nodes))])

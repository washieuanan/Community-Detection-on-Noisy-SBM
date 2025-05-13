import numpy as np
import networkx as nx
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from cbsm.sbm import generate_noisy_sbm
from algorithms.duo_spec import detection_stats, get_true_communities

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


# === Example usage ===
if __name__ == "__main__":
    # build or load your graph G and true labels true_labels here
    G = generate_noisy_sbm(500, K=2, p_in=0.6, p_out=0.1, noise=0.8, dim=2, seed=42, r=0.15)
    pred = spectral_clustering_community_detection(G, 2)
    
    # assuming you have detection_stats(pred, true_labels) → dict or metrics
    true_labels = get_true_communities(G, node2idx=None, attr="comm")
    stats = detection_stats(pred, true_labels)
    print(stats)

import numpy as np
import itertools
import networkx as nx
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from cbsm.sbm import generate_noisy_sbm
from community_detection.duo_spec import detection_stats, get_true_communities

def motif_counting_community_detection(G: nx.Graph, K: int) -> np.ndarray:
    """
    Count the number of triangles (3‐node motifs) shared by each pair of nodes,
    build a motif‐adjacency matrix M, then do spectral clustering on M into K groups.

    Parameters
    ----------
    G : networkx.Graph
        Undirected graph on n nodes (nodes can be any hashable, but must be 0..n-1 or remapped).
    K : int
        Number of communities to find.

    Returns
    -------
    assignments : np.ndarray, shape (n,)
        Community labels in {0,1,...,K-1}, ordered to match sorted(G.nodes()).
    """
    # 1) Remap nodes to indices 0..n-1
    nodes = list(G.nodes())
    idx = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)

    # 2) Build the motif‐adjacency matrix M
    M = np.zeros((n, n), dtype=float)
    for center in nodes:
        neigh = list(G.neighbors(center))
        for u, v in itertools.combinations(neigh, 2):
            iu, iv = idx[u], idx[v]
            M[iu, iv] += 1
            M[iv, iu] += 1

    # 3) Compute the top‐K eigenvectors of M
    #    Using 'LA' (largest algebraic) since M is nonnegative
    vals, vecs = eigsh(M, k=K, which='LA')

    # 4) Cluster rows of the eigenvector matrix
    km = KMeans(n_clusters=K, n_init=10)
    labels = km.fit_predict(vecs)

    # 5) Return as a numpy array
    return np.array(labels, dtype=int)

# === Example usage ===
if __name__ == "__main__":
    # build or load your graph G and true labels true_labels here
    G = generate_noisy_sbm(500, K = 2, p_in = 0.6, p_out = 0.1, noise=0.8, dim=2, seed=42, r=0.15)
    pred = motif_counting_community_detection(G, 2)
    
    # assuming you have detection_stats(pred, true_labels) → dict or metrics
    true_labels = get_true_communities(G, node2idx=None, attr="comm")
    stats = detection_stats(pred, true_labels)
    print(stats)

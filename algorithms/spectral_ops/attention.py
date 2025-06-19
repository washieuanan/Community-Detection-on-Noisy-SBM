from __future__ import annotations

from typing import Dict

import numpy  as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans

from .pmi_funcs import grab_pmi_func

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  DeepWalk random-walk corpus
# ═══════════════════════════════════════════════════════════════════════════════

def _conf_from_center(X, mu):
    """
    Confidence matrix Q[i,c] = 1 / (1 + ||x_i − μ_c||_2)
    Rows are re-normalised to sum to 1.
    """
    dists = np.linalg.norm(X[:, None, :] - mu[None, :, :], axis=-1)  # (n,q)
    Q     = 1.0 / (1.0 + dists)
    Q    /= Q.sum(axis=1, keepdims=True)
    return Q

pmi_svd_embeddings = grab_pmi_func(fast=True, weighted=True)  # ADJUST SETTINGS HERE

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Attention Laplacian  H = D^{-1/2}  softmax(Z Zᵀ/√d | edges)  D^{-1/2}
# ═══════════════════════════════════════════════════════════════════════════════
def attention_laplacian(
    G        : nx.Graph,
    Z        : Dict[str, np.ndarray],   # node-id → embedding vec
    clip_max : float = 1e2,
) -> sp.csr_matrix:
    """Return sparse symmetric PSD operator H_{BYOE}."""
    n      = G.number_of_nodes()
    node2i = {u: i for i, u in enumerate(G.nodes())}

    dim = next(iter(Z.values())).shape[0]
    # fast access matrix  Z_mat  (n × dim)
    Z_mat = np.vstack([Z[str(u)] for u in G.nodes()]).astype(np.float32)
    Z_mat /= np.linalg.norm(Z_mat, axis=1, keepdims=True) + 1e-9  # unit length

    iu, iv = [], []
    data   = []
    dot    = Z_mat @ Z_mat.T      # dense (n×n) but accessed sparsely below
    scale  = 1.0 / np.sqrt(dim)

    for u, v in G.edges():
        i, j   = node2i[u], node2i[v]
        weight = np.exp( np.clip(scale * dot[i, j], a_min=None, a_max=np.log(clip_max)) )
        iu.append(i); iv.append(j); data.append(weight)

    W = sp.coo_matrix((data, (iu, iv)), shape=(n, n))
    W = W + W.T
    deg = np.asarray(W.sum(axis=1)).ravel() + 1e-9
    Dinv_half = sp.diags(1.0 / np.sqrt(deg))
    H = Dinv_half @ W @ Dinv_half        # CSR
    return H.tocsr()

def motif_attention_laplacian(
    H_obs        : nx.Graph,
    Z            : dict,          # node-id -> embedding vector
    *,
    beta         : float  = 0.3,  # mix factor for motif weights
    clip_max     : float  = 1e2,  # cap for exp(.) to avoid overflow
    weight_pow   : float  = 1.0,  # NEW: exponent applied to edge weights
    weight_eps   : float  = 1e-12,# NEW: avoids zeroing very small weights
    random_state : int    = 42,
) -> sp.csr_matrix:
    """
    motif attention laplacian with weights -- should still work without weights but haven't tested
    """
    rng      = np.random.RandomState(random_state)
    nodes    = list(H_obs.nodes())
    node2i   = {u: i for i, u in enumerate(nodes)}
    n        = len(nodes)

    # 1) base attention W with edge-weight factor
    d_emb  = next(iter(Z.values())).shape[0]
    scale  = 1.0 / np.sqrt(d_emb)

    iu, iv, data = [], [], []
    for u, v, d in H_obs.edges(data=True):
        i, j = node2i[u], node2i[v]
        zu, zv = Z[str(u)], Z[str(v)]
        score  = np.dot(zu, zv) * scale
        att    = np.exp(np.clip(score, a_min=None,
                                a_max=np.log(clip_max)))

        # ----------  ★ incorporate the stored edge weight  ---------------
        w_edge = float(d.get("weight", 1.0))
        w_edge = max(w_edge, weight_eps)          # avoid exact 0
        att   *= w_edge ** weight_pow
        # -----------------------------------------------------------------

        iu.append(i); iv.append(j); data.append(att)

    W = sp.coo_matrix((data, (iu, iv)), shape=(n, n)).tocsr()
    W = W + W.T                                    # undirected

    # 2) two-hop (triangle) counts
    W2 = W @ W                                     # sparse matmul

    # 3) motif weights on original edges
    M  = W.multiply(W2)

    # 4) mix
    W_mix = (1.0 - beta) * W + beta * M

    # 5) symmetric normalisation
    deg = np.array(W_mix.sum(axis=1)).ravel() + 1e-9
    D_inv_sqrt = sp.diags(1.0 / np.sqrt(deg))
    H = D_inv_sqrt @ W_mix @ D_inv_sqrt

    return H.tocsr()

def motif_spectral_embedding(
    H_obs        : nx.Graph,
    q            : int,
    *,
    beta         : float = 0.3,
    clip_max     : float = 1e2,
    dim          : int   = 128,
    walk_len     : int   = 60,
    num_walks    : int   = 20,
    window       : int   = 10,
    random_state : int   = 42,
) -> tuple[np.ndarray,np.ndarray,dict,dict]:
    """
    1) Build node embeddings Z via PPMI+SVD (pure NumPy).
    2) Build motif-enhanced Attention Laplacian H.
    3) Spectral clustering on H: top-q eigenvecs + k-means.
    Returns (Q, hard, node2idx, idx2node).
    """
    # --- node indexing -------------------------------------------------------
    nodes     = list(H_obs.nodes())
    node2idx  = {u:i for i,u in enumerate(nodes)}
    idx2node  = {i:u for u,i in node2idx.items()}

    # --- 1) get Z embeddings -----------------------------------------------
    Z = pmi_svd_embeddings(
        H_obs,
        dim=dim,
        walk_len=walk_len,
        num_walks=num_walks,
        window=window,
        seed=random_state
    )

    print("Finished PPMI-SVD embeddings")
    # --- 2) build motif‐attention Laplacian -------------------------------
    H = motif_attention_laplacian(
        H_obs,
        Z,
        beta=beta,
        clip_max=clip_max,
        random_state=random_state
    )
    print("Finished motif-attention Laplacian")
    # --- 3) spectral clustering -------------------------------------------
    ncv = 2 * min(H.shape[0]-1, max(2*q+1, q+20))
    vals, vecs = eigsh(H, k=q, which="LA", ncv=ncv, tol=1e-4)

    km   = KMeans(n_clusters=q, n_init=20, random_state=random_state).fit(vecs)
    hard = km.labels_
    mu   = km.cluster_centers_
    Q    = _conf_from_center(vecs, mu)

    return Q, hard, node2idx, idx2node


def motif_laplacian_spectral_embedding(
    G      : nx.Graph,
    q      : int,
    *,
    normalized : bool = True,
    random_state: int = 42
):
    """
    Spectral clustering using the triangle‐motif Laplacian.

    1) Build sparse adjacency A (unweighted).
    2) Compute A2 = A @ A  (counts of common neighbors).
    3) Motif adjacency M = A.multiply(A2)  (only on original edges).
    4) If normalized, form H = D^{-1/2} M D^{-1/2}; else H = M.
    5) Take the q leading eigenvectors of H, then k-means.

    Returns
    -------
    Q        : (n,q) soft cluster confidences (rows sum to 1)
    hard     : (n,) hard labels (0..q-1)
    node2idx : mapping from node → row index
    idx2node : mapping from row index → node
    """
    # 1) node ordering & adjacency
    nodes    = list(G.nodes())
    node2idx = {u:i for i,u in enumerate(nodes)}
    idx2node = {i:u for u,i in node2idx.items()}
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, format="csr", weight=None)

    # 2) two-hop counts
    A2 = A.dot(A)  # (i,j) = # common neighbors of i,j

    # 3) motif adjacency
    M = A.multiply(A2)  # only keeps entries where A_ij = 1

    # 4) normalize (optional)
    if normalized:
        deg = np.array(M.sum(axis=1)).ravel() + 1e-9
        D_inv_sqrt = sp.diags(1.0/np.sqrt(deg))
        H = D_inv_sqrt @ M @ D_inv_sqrt
    else:
        H = M

    # 5) spectral embedding
    ncv = 2 * min(H.shape[0]-1, max(2*q+1, q+20))
    vals, vecs = eigsh(H, k=q, which="LA", ncv=ncv, tol=1e-4)

    # k-means + soft confidence
    km    = KMeans(n_clusters=q, n_init=20, random_state=random_state).fit(vecs)
    hard  = km.labels_
    # simple Gaussian‐based confidence
    sqdist = np.square(vecs[:,None,:] - km.cluster_centers_[None,:,:]).sum(-1)
    Q      = np.exp(-0.5*sqdist)
    Q     /= Q.sum(axis=1, keepdims=True)

    return Q, hard, node2idx, idx2node


def byoe_embedding(
    H_obs        : nx.Graph,
    q            : int,
    *,
    dim          : int   = 64,
    walk_len     : int   = 40,
    num_walks    : int   = 10,
    window       : int   = 5,
    random_state : int   = 42,
) -> tuple[np.ndarray, np.ndarray, dict[str,int], dict[int,str]]:
    """
    BYOE spectral clustering based on (DeepWalk ➜ PPMI ➜ TruncatedSVD).
    Returns (Q, hard, node2idx, idx2node).
    """
    # node ordering
    nodes    = list(H_obs.nodes())
    node2idx = {u:i for i,u in enumerate(nodes)}
    idx2node = {i:u for u,i in node2idx.items()}

    # (A)+(B) embeddings
    Z_dict = pmi_svd_embeddings(
        H_obs,
        dim=dim,
        walk_len=walk_len,
        num_walks=num_walks,
        window=window,
        seed=random_state
    )

    # (C) Attention Laplacian
    H = attention_laplacian(H_obs, Z_dict)

    # (D) spectral embedding
    ncv = 2 * min(H.shape[0] - 1, max(2*q + 1, q + 20))
    vals, vecs = eigsh(H, k=q, which="LA", ncv=ncv, tol=1e-4)

    km    = KMeans(n_clusters=q, n_init=20, random_state=random_state).fit(vecs)
    hard  = km.labels_
    mu    = km.cluster_centers_
    Q     = _conf_from_center(vecs, mu)

    return Q, hard, node2idx, idx2node


if __name__ == "__main__": 
    from block_models.sbm.sbm import generate_noisy_sbm
    from algorithms.duo_spec import detection_stats, bethe_hessian, geometric_censor

    # Generate test graph
    n = 1000
    K = 3
    G_true = generate_noisy_sbm(
        n=n,
        K=K, 
        p_in=0.56,
        p_out=0.196,
        sigma=0.2,
        seed=42
    )


    # G_censored = geometric_censor(G_true, 0.5) 

    print("Generated graph with", len(G_true.nodes()), "nodes and", len(G_true.edges()), "edges")

    # Run BYOE embedding
    Q_byoe, preds_byoe, node2idx_byoe, idx2node_byoe = motif_spectral_embedding(
        G_true,
        q=K
    )

    # Run Bethe-Hessian
    Q_bh, preds_bh, node2idx_bh, idx2node_bh = motif_laplacian_spectral_embedding(
        G_true,
        q=K,
    )

    # Get true labels and calculate stats for both methods
    true_labels = np.array([G_true.nodes[u]["comm"] for u in G_true.nodes()])
    
    stats_byoe = detection_stats(preds_byoe, true_labels)
    stats_bh = detection_stats(preds_bh, true_labels)

    print("\n=== BYOE Community‑detection accuracy ===")
    for k, v in stats_byoe.items():
        print(f"{k:>25s} : {v}")

    print("\n=== Bethe-Hessian Community‑detection accuracy ===") 
    for k, v in stats_bh.items():
        print(f"{k:>25s} : {v}")

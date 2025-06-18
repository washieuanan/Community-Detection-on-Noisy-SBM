# ───────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import random
import itertools
from typing   import List, Tuple, Dict

import numpy  as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm
import torch
from collections import Counter
from itertools import islice
from random import choices, randint
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import svds
from scipy.sparse import lil_matrix, csr_matrix, identity, diags, coo_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.utils.extmath import randomized_svd

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


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  SGNS / word2vec embeddings  ≈  shifted PPMI factorisation
# ═══════════════════════════════════════════════════════════════════════════════

# def pmi_svd_embeddings(
#     G              : nx.Graph,
#     dim            : int   = 64,
#     *,
#     walk_len       : int   = 60,
#     num_walks      : int   = 15,
#     window         : int   = 10,
#     seed           : int   = 42,
# ) -> dict[str, np.ndarray]:
#     """
#     Return dict {node-id (str) -> ℝ^dim embedding} using
#     PPMI -> rank-dim randomized SVD via TruncatedSVD.
#     """
#     rng     = random.Random(seed)
#     nodes   = list(G.nodes())
#     n       = len(nodes)
#     node2i  = {u: i for i, u in enumerate(nodes)}

#     # --- Build a sparse co-occurrence matrix ------------------------------
#     C = lil_matrix((n, n), dtype=np.float32)
#     for _ in range(num_walks):
#         rng.shuffle(nodes)
#         for start in nodes:
#             walk = [start]
#             for _ in range(walk_len - 1):
#                 nbrs = list(G.neighbors(walk[-1]))
#                 if not nbrs:
#                     break
#                 walk.append(rng.choice(nbrs))
#             for i, u in enumerate(walk):
#                 ui = node2i[u]
#                 for j in range(max(0, i - window), min(len(walk), i + window + 1)):
#                     if i != j:
#                         vi = node2i[walk[j]]
#                         C[ui, vi] += 1.0
#     C = C.tocsr()

#     # --- Compute PPMI ------------------------------------------------------
#     row_sum = np.asarray(C.sum(axis=1)).ravel()
#     col_sum = np.asarray(C.sum(axis=0)).ravel()
#     total   = row_sum.sum() + 1e-9
#     # formula: PPMI = max(log((count * total)/(row_sum*col_sum)) - log(neg_samples), 0)
#     # shift = 1.0
#     # C.data = np.log((C.data * total) /
#     #             (row_sum[C.indices] * col_sum[C.indices]) + 1e-9) - np.log(shift)
#     C.data = np.log(C.data * total / (row_sum[C.indices] * col_sum[C.indices] + 1e-9) + 1e-9)
#     C.data = np.clip(C.data, 0, None)

#     # mask = C.data > 0.5
#     # C.data, C.indices, C.indptr = C.data[mask], C.indices[mask], C.indptr

#     # --- Randomized SVD via TruncatedSVD -----------------------------------
#     svd = TruncatedSVD(
#         n_components=dim,
#         n_iter=7,
#         random_state=seed
#     )
#     Z = svd.fit_transform(C)
#     # Optionally scale by sqrt of singular values: embed = U * S^0.5
#     # but TruncatedSVD returns U * Sigma, so we take Z directly.

#     return {str(nodes[i]): Z[i] for i in range(n)}

# fast version
def pmi_svd_embeddings(
    G              : nx.Graph,
    dim            : int   = 64,
    *,
    walk_len       : int   = 60,
    num_walks      : int   = 15,
    window         : int   = 10,
    seed           : int   = 42,
) -> Dict[str, np.ndarray]:
    """
    Same DeepWalk-style PPMI-SVD as the original, but vectorised.
    Returns dict {node-id(str) → ℝ^dim}.
    """
    rng_np = np.random.RandomState(seed)

    # ---------- node order & neighbour tables ----------------------------
    nodes   = list(G.nodes())
    n       = len(nodes)
    node2i  = {u: i for i, u in enumerate(nodes)}
    idx2node = {i: u for u, i in node2i.items()}

    neigh = [np.fromiter((node2i[v] for v in G.neighbors(u)),
                         dtype=np.int32, count=G.degree(u))
             for u in nodes]

    # ---------- generate walks & collect (u,v) pairs ---------------------
    rows, cols = [], []                         # use normal lists

    for _ in range(num_walks):
        for start_idx in rng_np.permutation(n):
            walk = [start_idx]
            cur  = start_idx
            for _ in range(walk_len - 1):
                nbrs = neigh[cur]
                if nbrs.size == 0:
                    break
                cur = int(rng_np.choice(nbrs))
                walk.append(cur)

            w = np.asarray(walk, dtype=np.int32)
            L = len(w)
            max_off = min(window, L - 1)
            for off in range(1, max_off + 1):
                rows.extend(w[:-off]); cols.extend(w[off:])   # (u,v)
                rows.extend(w[off:]); cols.extend(w[:-off])   # (v,u)

    C = coo_matrix((np.ones(len(rows), dtype=np.float32),
                    (rows, cols)), shape=(n, n))
    C.sum_duplicates()
    C = C.tocsr()

    # ---------- Positive PMI --------------------------------------------
    row_sum = np.asarray(C.sum(axis=1)).ravel()
    col_sum = np.asarray(C.sum(axis=0)).ravel()
    total   = float(row_sum.sum()) + 1e-9

    C = C.tocoo()
    pmi = np.log((C.data * total) /
                 (row_sum[C.row] * col_sum[C.col] + 1e-9))
    pmi[pmi < 0] = 0.0
    X = csr_matrix((pmi, (C.row, C.col)), shape=C.shape)

    # ---------- randomized SVD ------------------------------------------
    U, _, _ = randomized_svd(
        X,
        n_components=dim,
        n_iter=7,
        random_state=seed,  # Pass the integer seed instead of RandomState object
    )

    return {str(idx2node[i]): U[i] for i in range(n)}



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
    beta         : float = 0.3,   # how much to mix in the motif weights
    clip_max     : float = 1e2,   # cap for the base attention exp()
    random_state : int   = 42,
) -> sp.csr_matrix:
    """
    Build a triangle-motif–enhanced Attention-Laplacian.

    1. Compute base attention weights W_uv = exp(<z_u,z_v>/sqrt(d)) for (u,v) in edges.
    2. Let W2 = W @ W  (weighted 2-hop counts).
    3. Extract motif weights: M = W.multiply(W2)  (only on original edges).
    4. Mix: W_mix = (1-beta)*W + beta*M.
    5. Degree-normalize: H = D^{-1/2} W_mix D^{-1/2}.
    """
    rng = np.random.RandomState(random_state)
    nodes   = list(H_obs.nodes())
    node2i  = {u:i for i,u in enumerate(nodes)}
    n       = len(nodes)

    # 1) build base attention W
    #    assume Z is dict node->np.array(d,)
    d_emb    = next(iter(Z.values())).shape[0]
    scale    = 1.0/np.sqrt(d_emb)
    iu, iv, data = [], [], []
    for u,v in H_obs.edges():
        i,j = node2i[u], node2i[v]
        zu, zv = Z[str(u)], Z[str(v)]
        score = np.dot(zu, zv) * scale
        w = np.exp(np.clip(score, a_min=None, a_max=np.log(clip_max)))
        iu.append(i); iv.append(j); data.append(w)
    W = sp.coo_matrix((data,(iu,iv)),shape=(n,n)).tocsr()
    W = W + W.T  # symmetric

    # 2) compute 2-hop weighted counts
    W2 = W.dot(W)    # (n×n) sparse, entry (i,j)=sum_k W[i,k]*W[k,j]

    # 3) motif weights on original edges: M_ij = W_ij * W2_ij
    M = W.multiply(W2)  # keeps only those (i,j) where W_ij>0

    # 4) mix them
    W_mix = (1.0 - beta) * W + beta * M

    # 5) build normalized Laplacian
    deg = np.array(W_mix.sum(axis=1)).ravel() + 1e-9
    D_inv_sqrt = sp.diags(1.0/np.sqrt(deg))
    H = D_inv_sqrt @ W_mix @ D_inv_sqrt

    return H.tocsr()
# ---------- fast-prune helper ------------------------------------------------

# ---------- motif-spectral embedding w/ k-NN prune + Nyström ---------------
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
    # --- speed-up knobs -------------------------------------------------
    size_threshold: int  = 5_000,     # switch to Nyström above this |V|
    n_anchors     : int | None = None,
    k_prune       : int  = 15,        # neighbour budget for fast-prune
):
    """
    Fast variant:
      • (optional) k-NN sparsification           — O(|E|) → O(k|V|)
      • Nyström sub-sampling for eigendecomp     — O(|E| m) + O(m³)
    Returns (Q, hard, node2idx, idx2node) exactly like the original.
    """
    
    
    # -------- node indexing --------------------------------------------------
    nodes     = list(H_obs.nodes())
    node2idx  = {u: i for i, u in enumerate(nodes)}
    idx2node  = {i: u for u, i in node2idx.items()}

    # -------- 1) PPMI-SVD embeddings (unchanged) -----------------------------
    Z = pmi_svd_embeddings(
        H_obs,
        dim=dim,
        walk_len=walk_len,
        num_walks=num_walks,
        window=window,
        seed=random_state,
    )

    # -------- 2) motif-attention Laplacian (unchanged) -----------------------
    H = motif_attention_laplacian(
        H_obs,
        Z,
        beta=beta,
        clip_max=clip_max,
        random_state=random_state,
    )
    print("Finished motif-attention Laplacian")
    n = H.shape[0]

    # --- 3) spectral step ----------------------------------------------
    if n < size_threshold:
        # exact eigsh for small graphs
        ncv  = 2 * min(n - 1, max(2 * q + 1, q + 20))
        _, vecs = eigsh(H, k=q, which="LA", ncv=ncv, tol=1e-4)
    else:
        # 3a choose anchors
        if n_anchors is None:
            n_anchors = min(2_000, int(2 * np.sqrt(n * q)))
        rng = np.random.RandomState(random_state)
        anchors = np.sort(rng.choice(n, size=n_anchors, replace=False))

        mask_anchor = np.zeros(n, dtype=bool)
        mask_anchor[anchors] = True
        others = np.where(~mask_anchor)[0]

        # 3b build Laplacian blocks
        H_oo = H[anchors][:, anchors].toarray()      # (m,m)
        H_ro = H[others][:, anchors]                 # (n-m,m)

        # 3c eigendecompose anchor block
        evals, evecs = np.linalg.eigh(H_oo)
        idx  = np.argsort(evals)[-q:]
        Λ    = evals[idx]
        U_o  = evecs[:, idx]

        # 3d Nyström extension
        U_r  = H_ro @ (U_o / Λ)
        vecs = np.empty((n, q), dtype=H.dtype)
        vecs[anchors] = U_o
        vecs[others]  = U_r

        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        vecs /= norms

    # --- 4) k-means on spectral coords ---------------------------------
    km   = KMeans(n_clusters=q, n_init=20, random_state=random_state).fit(vecs)
    hard = km.labels_
    mu   = km.cluster_centers_
    Q    = _conf_from_center(vecs, mu)

    return Q, hard, node2idx, idx2node

# def motif_spectral_embedding(
#     H_obs        : nx.Graph,
#     q            : int,
#     *,
#     beta         : float = 0.3,
#     clip_max     : float = 1e2,
#     dim          : int   = 128,
#     walk_len     : int   = 60,
#     num_walks    : int   = 20,
#     window       : int   = 10,
#     random_state : int   = 42,
#     size_threshold : int = 5_000,   # -- when to switch to randomized solver
#     n_iter_rand   : int = 4,        # -- power iterations for Lanczos
# ) -> Tuple[np.ndarray, np.ndarray, Dict[int,int], Dict[int,int]]:
#     """
#     1) Node embeddings Z via PPMI+SVD.
#     2) Motif-attention Laplacian H.
#     3) Spectral clustering (ARPACK for small graphs, randomized Lanczos for large ones).
#     Returns (Q, hard, node2idx, idx2node).
#     """

#     # -------- node indexing --------------------------------------------------
#     nodes     = list(H_obs.nodes())
#     node2idx  = {u: i for i, u in enumerate(nodes)}
#     idx2node  = {i: u for u, i in node2idx.items()}

#     # -------- 1) PPMI-SVD embeddings (unchanged) -----------------------------
#     Z = pmi_svd_embeddings(
#         H_obs,
#         dim=dim,
#         walk_len=walk_len,
#         num_walks=num_walks,
#         window=window,
#         seed=random_state,
#     )

#     # -------- 2) motif-attention Laplacian (unchanged) -----------------------
#     H = motif_attention_laplacian(
#         H_obs,
#         Z,
#         beta=beta,
#         clip_max=clip_max,
#         random_state=random_state,
#     )
#     print("Finished PPMI-SVD embeddings")

#     # -------- 3) eigenvectors  ----------------------------------------------
#     n = H.shape[0]
#     if n >= size_threshold:
#         # ---- FAST PATH: randomized block-Lanczos / power iteration ----------
#         # returns U (n×q), Σ, V^T;  U spans the dominant q-dim subspace
#         U, _, _ = randomized_svd(
#             H,
#             n_components=q,
#             n_iter=n_iter_rand,
#             random_state=random_state,
#         )
#         vecs = U                       # (n,q) eigen-like vectors
#     else:
#         # ---- fallback: ARPACK (as before) -----------------------------------
#         ncv = 2 * min(n - 1, max(2 * q + 1, q + 20))
#         _, vecs = eigsh(
#             H,
#             k=q,
#             which="LA",                # same sign convention as before
#             ncv=ncv,
#             tol=1e-4,
#         )

#     # -------- k-means on the eigen-map --------------------------------------
#     km   = KMeans(n_clusters=q, n_init=20, random_state=random_state).fit(vecs)
#     hard = km.labels_
#     mu   = km.cluster_centers_
#     Q    = _conf_from_center(vecs, mu)

#     return Q, hard, node2idx, idx2node
# def motif_spectral_embedding(
#     H_obs        : nx.Graph,
#     q            : int,
#     *,
#     beta         : float = 0.3,
#     clip_max     : float = 1e2,
#     dim          : int   = 128,
#     walk_len     : int   = 60,
#     num_walks    : int   = 20,
#     window       : int   = 10,
#     random_state : int   = 42,
# ) -> tuple[np.ndarray,np.ndarray,dict,dict]:
#     """
#     1) Build node embeddings Z via PPMI+SVD (pure NumPy).
#     2) Build motif-enhanced Attention Laplacian H.
#     3) Spectral clustering on H: top-q eigenvecs + k-means.
#     Returns (Q, hard, node2idx, idx2node).
#     """
#     # --- node indexing -------------------------------------------------------
#     nodes     = list(H_obs.nodes())
#     node2idx  = {u:i for i,u in enumerate(nodes)}
#     idx2node  = {i:u for u,i in node2idx.items()}

#     # --- 1) get Z embeddings -----------------------------------------------
#     Z = pmi_svd_embeddings(
#         H_obs,
#         dim=dim,
#         walk_len=walk_len,
#         num_walks=num_walks,
#         window=window,
#         seed=random_state
#     )

#     # --- 2) build motif‐attention Laplacian -------------------------------
#     H = motif_attention_laplacian(
#         H_obs,
#         Z,
#         beta=beta,
#         clip_max=clip_max,
#         random_state=random_state
#     )

#     # --- 3) spectral clustering -------------------------------------------
#     ncv = 2 * min(H.shape[0]-1, max(2*q+1, q+20))
#     vals, vecs = eigsh(H, k=q, which="LA", ncv=ncv, tol=1e-4)

#     km   = KMeans(n_clusters=q, n_init=20, random_state=random_state).fit(vecs)
#     hard = km.labels_
#     mu   = km.cluster_centers_
#     Q    = _conf_from_center(vecs, mu)

#     return Q, hard, node2idx, idx2node


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

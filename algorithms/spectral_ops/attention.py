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
from scipy.sparse import lil_matrix, csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

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




def pmi_svd_embeddings(
    G              : nx.Graph,
    dim            : int   = 64,
    *,
    walk_len       : int   = 60,
    num_walks      : int   = 15,
    window         : int   = 10,
    seed           : int   = 42,
) -> dict[str, np.ndarray]:
    """
    Return dict {node-id (str) -> ℝ^dim embedding} using
    PPMI -> rank-dim randomized SVD via TruncatedSVD.
    """
    rng     = random.Random(seed)
    nodes   = list(G.nodes())
    n       = len(nodes)
    node2i  = {u: i for i, u in enumerate(nodes)}

    # --- Build a sparse co-occurrence matrix ------------------------------
    C = lil_matrix((n, n), dtype=np.float32)
    for _ in range(num_walks):
        rng.shuffle(nodes)
        for start in nodes:
            walk = [start]
            for _ in range(walk_len - 1):
                nbrs = list(G.neighbors(walk[-1]))
                if not nbrs:
                    break
                walk.append(rng.choice(nbrs))
            for i, u in enumerate(walk):
                ui = node2i[u]
                for j in range(max(0, i - window), min(len(walk), i + window + 1)):
                    if i != j:
                        vi = node2i[walk[j]]
                        C[ui, vi] += 1.0
    C = C.tocsr()

    # --- Compute PPMI ------------------------------------------------------
    row_sum = np.asarray(C.sum(axis=1)).ravel()
    col_sum = np.asarray(C.sum(axis=0)).ravel()
    total   = row_sum.sum() + 1e-9
    # formula: PPMI = max(log((count * total)/(row_sum*col_sum)) - log(neg_samples), 0)
    # shift = 1.0
    # C.data = np.log((C.data * total) /
    #             (row_sum[C.indices] * col_sum[C.indices]) + 1e-9) - np.log(shift)
    C.data = np.log(C.data * total / (row_sum[C.indices] * col_sum[C.indices] + 1e-9) + 1e-9)
    C.data = np.clip(C.data, 0, None)

    # mask = C.data > 0.5
    # C.data, C.indices, C.indptr = C.data[mask], C.indices[mask], C.indptr

    # --- Randomized SVD via TruncatedSVD -----------------------------------
    svd = TruncatedSVD(
        n_components=dim,
        n_iter=7,
        random_state=seed
    )
    Z = svd.fit_transform(C)
    # Optionally scale by sqrt of singular values: embed = U * S^0.5
    # but TruncatedSVD returns U * Sigma, so we take Z directly.

    return {str(nodes[i]): Z[i] for i in range(n)}





# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Attention Laplacian  H = D^{-1/2}  softmax(Z Zᵀ/√d | edges)  D^{-1/2}
# ═══════════════════════════════════════════════════════════════════════════════
def attention_laplacian(
    G        : nx.Graph,
    Z        : Dict[str, np.ndarray],   # node-id → embedding vec
    clip_max : float = 1e2,
    *,
    # Evaluation-time option: scale attention by existing edge weights.
    use_edge_weights: bool = False,
    weight_key: str = "weight",
    weight_pow: float = 1.0,
    weight_floor: float = 0.0,
    weight_cap: float | None = None,
) -> sp.csr_matrix:
    """
    Return sparse symmetric PSD operator H_{BYOE}.

    If `use_edge_weights=True`, multiply the attention weight on each edge (u,v)
    by a monotone function of the graph's edge weight G[u][v][weight_key],
    after optional clipping and exponentiation. This lets DuoSpec/GeoDe's
    reweighted graph influence BYOE without changing embeddings.
    """
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
        i, j = node2i[u], node2i[v]
        score = scale * dot[i, j]

        if use_edge_weights:
            d = G[u][v]
            w_e = float(d.get(weight_key, 1.0))
            w_floor = max(weight_floor, 1e-12)
            w_e = max(w_floor, w_e)
            if weight_cap is not None:
                w_e = min(weight_cap, w_e)
            score = score + weight_pow * np.log(w_e)

        weight = np.exp(
            np.clip(score, a_min=None, a_max=np.log(clip_max))
        )

        iu.append(i)
        iv.append(j)
        data.append(weight)

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
    # Evaluation-time option: scale attention by existing edge weights.
    use_edge_weights: bool = False,
    weight_key: str = "weight",
    weight_pow: float = 1.0,
    weight_floor: float = 0.0,
    weight_cap: float | None = None,
) -> sp.csr_matrix:
    """
    Build a triangle-motif–enhanced Attention-Laplacian.

    1. Compute base attention weights W_uv = exp(<z_u,z_v>/sqrt(d)) for (u,v) in edges.
    2. Let W2 = W @ W  (weighted 2-hop counts).
    3. Extract motif weights: M = W.multiply(W2)  (only on original edges).
    4. Mix: W_mix = (1-beta)*W + beta*M.
    5. Degree-normalize: H = D^{-1/2} W_mix D^{-1/2}.

    If `use_edge_weights=True`, each base attention W_uv is further scaled
    by a monotone function of the existing edge weight H_obs[u][v][weight_key],
    after optional clipping and exponentiation. This allows DuoSpec/GeoDe
    reweighted graphs to influence motif embeddings.
    """
    rng = np.random.RandomState(random_state)
    nodes   = list(H_obs.nodes())
    node2i  = {u:i for i,u in enumerate(nodes)}
    n       = len(nodes)

    # 1) build base attention W_base that ignores edge weights
    d_emb    = next(iter(Z.values())).shape[0]
    scale    = 1.0/np.sqrt(d_emb)
    iu, iv, data = [], [], []
    for u, v in H_obs.edges():
        i, j = node2i[u], node2i[v]
        zu, zv = Z[str(u)], Z[str(v)]
        score = np.dot(zu, zv) * scale
        base = np.exp(np.clip(score, a_min=None, a_max=np.log(clip_max)))
        iu.append(i)
        iv.append(j)
        data.append(base)
    W_base = sp.coo_matrix((data,(iu,iv)),shape=(n,n)).tocsr()
    W_base = W_base + W_base.T  # symmetric

    # 2) compute 2-hop weighted counts on base attention
    W2_base = W_base.dot(W_base)    # (n×n) sparse

    # Optional edge-weight multiplier for motif term
    edge_mult = None
    if use_edge_weights and H_obs.number_of_edges() > 0:
        w_arr = np.array(
            [float(H_obs[u][v].get(weight_key, 1.0)) for u, v in H_obs.edges()],
            dtype=float,
        )
        w_lo = float(np.percentile(w_arr, 5.0))
        w_hi = float(np.percentile(w_arr, 95.0))
        if np.isfinite(w_lo) and np.isfinite(w_hi) and w_hi > w_lo:
            mults = []
            for w in w_arr:
                w_s = min(max(w, w_lo), w_hi)
                mults.append(float(w_s ** weight_pow))
            edge_mult = np.asarray(mults, dtype=float)
        else:
            edge_mult = np.ones(len(w_arr), dtype=float)

    # 3) build mixed weights per original edge: W_mix_ij = (1-beta)*W_base_ij + beta*M_ij
    iu_mix, iv_mix, data_mix = [], [], []
    for idx, (u, v) in enumerate(H_obs.edges()):
        i, j = node2i[u], node2i[v]
        base_ij = float(W_base[i, j])
        motif_ij = base_ij * float(W2_base[i, j])
        if edge_mult is not None:
            motif_ij *= float(edge_mult[idx])
        w_ij = (1.0 - beta) * base_ij + beta * motif_ij
        iu_mix.append(i)
        iv_mix.append(j)
        data_mix.append(w_ij)
    W_mix = sp.coo_matrix((data_mix, (iu_mix, iv_mix)), shape=(n, n)).tocsr()
    W_mix = W_mix + W_mix.T  # symmetric

    # 4) build normalized Laplacian
    deg = np.array(W_mix.sum(axis=1)).ravel() + 1e-9
    D_inv_sqrt = sp.diags(1.0/np.sqrt(deg))
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
    # Allow DuoSpec/GeoDe weights to influence motif attention.
    use_edge_weights: bool = False,
    weight_key: str = "weight",
    weight_pow: float = 1.0,
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

    # --- 2) build motif‐attention Laplacian (optionally scaled by edge weights)
    H = motif_attention_laplacian(
        H_obs,
        Z,
        beta=beta,
        clip_max=clip_max,
        random_state=random_state,
        use_edge_weights=use_edge_weights,
        weight_key=weight_key,
        weight_pow=weight_pow,
    )

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
    # Allow DuoSpec/GeoDe weights to influence BYOE via attention.
    use_edge_weights: bool = False,
    weight_key: str = "weight",
    weight_pow: float = 1.0,
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

    # (C) Attention Laplacian (optionally scaled by edge weights)
    H = attention_laplacian(
        H_obs,
        Z_dict,
        use_edge_weights=use_edge_weights,
        weight_key=weight_key,
        weight_pow=weight_pow,
    )

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
import random
import numpy as np
import networkx as nx
from typing import Dict
from scipy.sparse import lil_matrix, coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import svds
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd


# different versions of pmi_svd_embeddings
def slow_unweighted(
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

# slow weighted
def slow_weighted(
    G              : nx.Graph,
    dim            : int   = 64,
    *,
    walk_len       : int   = 60,
    num_walks      : int   = 15,
    window         : int   = 10,
    seed           : int   = 42,
    weight_key     : str   = "weight",  # NEW
    weight_pow     : float = 1.0,       # NEW (edge weight exponent)
) -> Dict[str, np.ndarray]:
    """
    Return dict {node-id (str) -> ℝ^dim embedding} using
    weighted random-walk PPMI  ➜  randomized SVD (TruncatedSVD).

    * If an edge has no `weight_key`, weight defaults to 1.0.
    * If all outgoing weights of a vertex are zero, a uniform choice is used.
    """
    rng  = random.Random(seed)
    nodes   = list(G.nodes())
    n       = len(nodes)
    node2i  = {u: i for i, u in enumerate(nodes)}

    # --- sparse co-occurrence matrix -----------------------------------
    C = lil_matrix((n, n), dtype=np.float32)

    for _ in range(num_walks):
        rng.shuffle(nodes)
        for start in nodes:
            walk = [start]
            for _ in range(walk_len - 1):
                cur = walk[-1]
                nbrs = list(G.neighbors(cur))
                if not nbrs:
                    break

                # -----  weighted neighbour choice  ---------------------
                wts = [max(float(G[cur][v].get(weight_key, 1.0)), 0.0)
                       for v in nbrs]
                wts = [w ** weight_pow for w in wts]
                if sum(wts) == 0.0:                     # fallback uniform
                    nxt = rng.choice(nbrs)
                else:
                    nxt = random.choices(nbrs, weights=wts, k=1)[0]
                # -------------------------------------------------------

                walk.append(nxt)

            # ----- context window updates ------------------------------
            L = len(walk)
            for i, u in enumerate(walk):
                ui = node2i[u]
                for j in range(max(0, i - window), min(L, i + window + 1)):
                    if i == j:
                        continue
                    vi = node2i[walk[j]]
                    C[ui, vi] += 1.0
    C = C.tocsr()

    # --- Positive PMI --------------------------------------------------
    row_sum = np.asarray(C.sum(axis=1)).ravel()
    col_sum = np.asarray(C.sum(axis=0)).ravel()
    total   = row_sum.sum() + 1e-9

    C = C.tocoo()
    pmi = np.log(
        (C.data * total) /
        (row_sum[C.row] * col_sum[C.col] + 1e-9)
    )
    pmi[pmi < 0.0] = 0.0
    X = csr_matrix((pmi, (C.row, C.col)), shape=C.shape)

    # --- Randomised SVD -----------------------------------------------
    svd = TruncatedSVD(
        n_components=dim,
        n_iter=7,
        random_state=seed,
    )
    Z = svd.fit_transform(X)

    return {str(nodes[i]): Z[i] for i in range(n)}

# fast version
def fast_unweighted(
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
        random_state=seed, 
    )

    return {str(idx2node[i]): U[i] for i in range(n)}

# weighted fast version
def fast_weighted(
    G              : nx.Graph,
    dim            : int   = 64,
    *,
    walk_len       : int   = 60,
    num_walks      : int   = 15,
    window         : int   = 10,
    seed           : int   = 42,
    weight_key     : str   = "weight",
) -> Dict[str, np.ndarray]:
    """
    DeepWalk-style PPMI-SVD node embeddings that respect `G[u][v][weight_key]`.
    Returns dict {node-id(str) -> embedding (ℝ^dim)}.
    """
    rng_np  = np.random.RandomState(seed)
    nodes   = list(G.nodes())
    n       = len(nodes)
    node2i  = {u: i for i, u in enumerate(nodes)}
    idx2node = {i: u for u, i in node2i.items()}

    # --- neighbour tables with normalised weight distributions ----------
    neigh_idx  = []
    neigh_prob = []
    for u in nodes:
        neighbors = list(G.neighbors(u))
        if not neighbors:
            neigh_idx.append(np.array([], dtype=np.int32))
            neigh_prob.append(np.array([], dtype=np.float64))
            continue
        nbrs, wts = zip(*[(node2i[v], float(G[u][v].get(weight_key, 1.0)))
                          for v in neighbors])
        wts = np.asarray(wts, dtype=np.float64)
        tot = wts.sum()
        if tot == 0.0:
            # all weights shrunk to zero – fall back to uniform
            prob = np.full_like(wts, 1.0 / len(wts))
        else:
            prob = wts / tot
        neigh_idx.append(np.asarray(nbrs, dtype=np.int32))
        neigh_prob.append(prob)

    # --- generate (u,v) context pairs via weighted walks ----------------
    rows, cols = [], []
    for _ in range(num_walks):
        for start_idx in rng_np.permutation(n):
            walk = [start_idx]
            cur  = start_idx
            for _ in range(walk_len - 1):
                nbrs = neigh_idx[cur]
                if nbrs.size == 0:
                    break
                cur = int(rng_np.choice(nbrs, p=neigh_prob[cur]))
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

    # --- Positive PMI ---------------------------------------------------
    row_sum = np.asarray(C.sum(axis=1)).ravel()
    col_sum = np.asarray(C.sum(axis=0)).ravel()
    total   = float(row_sum.sum()) + 1e-9

    C = C.tocoo()
    pmi = np.log((C.data * total) /
                 (row_sum[C.row] * col_sum[C.col] + 1e-9))
    pmi[pmi < 0] = 0.0
    X = csr_matrix((pmi, (C.row, C.col)), shape=C.shape)

    # --- Randomised SVD -------------------------------------------------
    U, _, _ = randomized_svd(
        X,
        n_components=dim,
        n_iter=7,
        random_state=seed,
    )
    return {str(idx2node[i]): U[i] for i in range(n)}

def grab_pmi_func(fast: bool = False, weighted: bool = True):
    if fast and weighted:
        return fast_weighted
    if fast and not weighted:
        return fast_unweighted
    if not fast and weighted:
        return slow_weighted
    if not fast and not weighted:
        return slow_unweighted
    return
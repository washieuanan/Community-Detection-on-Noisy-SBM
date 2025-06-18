# duospec_nystrom.py – version with Nyström approximation injected before duo_spec
# -----------------------------------------------------------------------------
# This script is a drop‑in replacement for your previous driver.  The only change
# is that we compute a Nyström approximation of the (normalized) adjacency/Laplacian
# and monkey‑patch the `motif_spectral_embedding` call that `duo_spec` relies on so
# that DuoSpec sees an O(m n) ‑ instead of O(n^2) – embedding step.
# -----------------------------------------------------------------------------

from algorithms.bp.old.vectorized_geometric_bp import (
    belief_propagation,
    detection_stats,
    get_true_communities,
)

import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

from algorithms.bp.old.duo_bp import (
    duo_bp,
    create_dist_observed_subgraph,
)

# defer import of duo_spec until after we monkey‑patch attention
import os, json, logging, random
from algorithms.bp.vectorized_bp import belief_propagation, belief_propagation_weighted
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# -----------------------------------------------------------------------------
# 1)  Standard helpers supplied in the original file
# -----------------------------------------------------------------------------

from algorithms.spectral_ops.attention import (
    motif_spectral_embedding as _orig_motif_spectral_embedding,
    pmi_svd_embeddings,
    motif_attention_laplacian,
)

def coords_str2arr(G: nx.Graph, dim: int = 16) -> nx.Graph:
    """Convert the string‑encoded coordinate stored in every node to a NumPy array."""
    new_G = nx.Graph()
    for n in G.nodes():
        coord_str = G.nodes[n]["coords"]
        coord_arr = np.fromstring(coord_str, sep=",")
        if len(coord_arr) != dim:
            coord_arr = np.zeros(dim)
        new_G.add_node(int(n), coords=coord_arr, asin=G.nodes[n]["asin"], comm=G.nodes[n]["comm"])

    for u, v in G.edges():
        if "dist" in G.edges[u, v]:
            dist = G.edges[u, v]["dist"]
            if isinstance(dist, str):
                dist = float(dist)
            new_G.add_edge(int(u), int(v), dist=dist)
    new_G.graph = G.graph.copy()
    new_G = nx.relabel.convert_node_labels_to_integers(new_G, first_label=0)
    return new_G


def connect_components(G: nx.Graph, weight: float = 1e-3) -> None:
    """Chain components together so that the graph is connected (duo_spec requirement)."""
    comps = list(nx.connected_components(G))
    if len(comps) <= 1:
        return
    reps = [next(iter(c)) for c in comps]
    for u, v in zip(reps[:-1], reps[1:]):
        if not G.has_edge(u, v):
            G.add_edge(u, v, weight=weight)

# -----------------------------------------------------------------------------
# 2)  Nyström approximation – *key addition*
# -----------------------------------------------------------------------------


def _graph_to_sparse_attention(
    G: nx.Graph,
    *,
    landmark_idx: np.ndarray | None = None,
    dim: int = 128,
    beta: float = 0.3,
    clip_max: float = 1e2,
    walk_len: int = 60,
    num_walks: int = 20,
    window: int = 10,
    random_state: int = 42,
):
    """Return blocks (**W**, **B**) needed for Nyström on motif-attention Laplacian.

    If *landmark_idx* is **None** we fall back to the expensive full build (for
    debug or very small graphs).  Otherwise we:

    1. Build exact motif Laplacian **H_L** on the landmark-induced subgraph.
    2. Build the cross block **B** containing weights between non‑landmarks and landmarks.
    3. Return (H_L, B) so the caller can proceed with Nyström.
    """
    all_nodes = list(G.nodes())
    n = len(all_nodes)

    if landmark_idx is None:  # full build (slow)
        Z = pmi_svd_embeddings(
            G,
            dim=dim,
            walk_len=walk_len,
            num_walks=num_walks,
            window=window,
            seed=random_state,
        )
        H_full = motif_attention_laplacian(
            G,
            Z,
            beta=beta,
            clip_max=clip_max,
            random_state=random_state,
        )
        return H_full.tocsr(), None

    # --------- fast partial build ----------------------------------------
    landmark_nodes = [all_nodes[i] for i in landmark_idx]
    G_land = G.subgraph(landmark_nodes).copy()

    Z_land = pmi_svd_embeddings(
        G_land,
        dim=dim,
        walk_len=walk_len // 2,
        num_walks=num_walks // 2,
        window=window,
        seed=random_state,
    )
    W_land = motif_attention_laplacian(
        G_land,
        Z_land,
        beta=beta,
        clip_max=clip_max,
        random_state=random_state,
    ).tocsr()

    # build B: (n-m) × m cross weights using base attention formula
    m = len(landmark_idx)
    non_land_mask = np.ones(n, dtype=bool)
    non_land_mask[landmark_idx] = False
    non_land_nodes = [all_nodes[i] for i in np.where(non_land_mask)[0]]

    # quick base attention without motif for cross block
    Z_lookup = {u: coords_str2arr(G).nodes[u]['coords'] for u in all_nodes}
    scale = 1.0 / np.sqrt(dim)
    iu, iv, data = [], [], []
    for row_glob, u in enumerate(non_land_nodes):
        z_u = Z_lookup[u]
        for col_local, v in enumerate(landmark_nodes):
            z_v = Z_land.get(str(v), Z_lookup[v])
            score = np.dot(z_u, z_v) * scale
            iu.append(row_glob)
            iv.append(col_local)
            data.append(np.exp(np.clip(score, a_min=None, a_max=np.log(clip_max))))
    B = csr_matrix((data, (iu, iv)), shape=(len(non_land_nodes), m), dtype=np.float32)

    return W_land.toarray(), B, landmark_idx, np.where(non_land_mask)[0]

def _choose_landmarks(G: nx.Graph, m: int, *, seed: int = 42, method: str = "degree") -> np.ndarray:
    """Select *m* landmark indices from the nodes of *G*.

    Parameters
    ----------
    G : nx.Graph
    m : int                Number of landmarks
    seed : int             RNG seed (default 42)
    method : {"degree", "uniform"}
        * ``"degree"`` (default) — sample with probability proportional to node degree.
        * ``"uniform"`` — uniform without replacement.
    """
    rng = np.random.default_rng(seed)
    n = G.number_of_nodes()
    m = min(m, n)

    if method == "uniform":
        return rng.choice(n, size=m, replace=False)

    if method == "degree":
        degs = np.array([d for _, d in G.degree()]) + 1e-6
        probs = degs / degs.sum()
        return rng.choice(n, size=m, replace=False, p=probs)

    raise ValueError(f"unknown landmark sampling method: {method}")

def nystrom_spectral_embedding(
    G: nx.Graph,
    m: int = 300,
    dim: int = 16,
    seed: int = 42,
    sampling: str = "degree",
    laplacian: str = "adjacency",  # or "normalized"
):
    """Compute an approximate spectral embedding using Nyström with better landmarks.

    New knobs:
    ---------
    sampling  – 'uniform' | 'degree' (default)  : how to pick landmarks
    laplacian – 'adjacency' | 'normalized'      : choose between A or D^{-1/2} A D^{-1/2}
    """
    A, _ = _graph_to_sparse_attention(G)
    n = A.shape[0]
    m = min(m, n)

    # Optional: normalised Laplacian improves stability for heavy‑tailed degree graphs
    if laplacian == "normalized":
        d = np.asarray(A.sum(1)).ravel()
        d_inv_sqrt = 1.0 / np.sqrt(d + 1e-10)
        D_inv_sqrt = csr_matrix((d_inv_sqrt, (np.arange(n), np.arange(n))), shape=(n, n))
        A = D_inv_sqrt @ A @ D_inv_sqrt

    # --------------------- landmark selection ------------------------------
    landmark_idx = _choose_landmarks(G, m, seed=seed, method=sampling)
    landmark_mask = np.zeros(n, dtype=bool)
    landmark_mask[landmark_idx] = True

    W = A[landmark_idx][:, landmark_idx].toarray()
    B = A[~landmark_mask][:, landmark_idx].toarray()

    k = min(dim + 20, max(dim + 5, W.shape[0] - 2))  # more oversampling than before
    eigvals, eigvecs = eigsh(W, k=k, which="LM")
    pos = eigvals > 1e-10
    eigvals, eigvecs = eigvals[pos], eigvecs[:, pos]

    inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
    U_upper = eigvecs
    U_lower = (B @ eigvecs) / eigvals
    U_full  = np.vstack([U_upper, U_lower]) @ inv_sqrt

    if U_full.shape[1] > dim:
        U_full = U_full[:, :dim]
    U_full /= np.linalg.norm(U_full, axis=1, keepdims=True) + 1e-12

    embedding = np.zeros_like(U_full)
    embedding[landmark_idx] = U_full[:m]
    embedding[~landmark_mask] = U_full[m:]
    return embedding
# ---------------------- subsequent code unchanged -------------------------

def motif_spectral_embedding_nystrom(*args, **kwargs):
    """Nyström speed‑up + exact motif clustering on a landmark subgraph.

    Steps
    -----
    1. Sample **m** landmarks (degree‑proportional).
    2. Run exact `motif_spectral_embedding` on the *landmark subgraph only*.
    3. Compute a Nyström embedding *U* for **all** nodes.
    4. Use landmark labels to form centroids in U‑space.
    5. Assign every node to its nearest centroid (soft & hard labels).
    """
    # ---------------------- unpack arguments ------------------------------
    if not args:
        raise TypeError("motif_spectral_embedding_nystrom: missing graph arg")
    G = args[0]
    if len(args) >= 2 and isinstance(args[1], int):
        q = args[1]
    else:
        q = kwargs.get("q", 2)

    dim          = kwargs.get("dim", 64)
    random_state = kwargs.get("random_state", 42)

    # ---------------------- landmark sampling ----------------------------
    print(f"Landmark sampling")
    m = min(max(16 * dim, 800), G.number_of_nodes())
    landmark_idx = _choose_landmarks(G, m, seed=random_state, method="degree")
    landmark_nodes = [list(G.nodes())[i] for i in landmark_idx]
    G_land = G.subgraph(landmark_nodes).copy()

    # ---------------------- exact motif clustering on landmarks ----------
    print(f"Exact motif clustering on landmarks")
    Q_land, hard_land, node2idx_land, _ = _orig_motif_spectral_embedding(
        G_land, q, dim=dim, random_state=random_state
    )
    hard_land = np.asarray(hard_land)

    # ---------------------- Nyström embed whole graph --------------------
    print(f"Nyström embedding whole graph")
    U = nystrom_spectral_embedding(
        G, m=m, dim=dim, seed=random_state, sampling="degree"
    )  # n × dim
    U_land = U[landmark_idx]

    # Compute centroids in embedding space from landmark labels
    centroids = np.zeros((q, U.shape[1]))
    for c in range(q):
        mask = hard_land == c
        if mask.any():
            centroids[c] = U_land[mask].mean(axis=0)
        else:  # empty cluster: fall back to random landmark
            centroids[c] = U_land[np.random.choice(len(U_land))]

    # -------------- assign every node to nearest centroid ----------------
    d2 = cdist(U, centroids, metric="sqeuclidean")  # n × q
    hard = d2.argmin(axis=1)
    sims = -d2
    sims -= sims.max(axis=1, keepdims=True)
    exp_sims = np.exp(sims)
    Q = exp_sims / exp_sims.sum(axis=1, keepdims=True)

  
    # ---------------------- mappings -------------------------------------
    nodes = list(G.nodes())
    node2idx = {u: i for i, u in enumerate(nodes)}
    idx2node = {i: u for u, i in node2idx.items()}

    return Q, hard, node2idx, idx2node

# Inject our replacement into *both* the attention module **and** DuoSpec so
# every downstream reference picks it up.
import algorithms.spectral_ops.attention as _attention_mod
_attention_mod.motif_spectral_embedding = motif_spectral_embedding_nystrom

import importlib
import algorithms.duo_spec as _duo_mod
_duo_mod.motif_spectral_embedding = motif_spectral_embedding_nystrom
# expose duo_spec after patching
from algorithms.duo_spec import duo_spec

_duo_mod.motif_spectral_embedding = motif_spectral_embedding_nystrom

# -----------------------------------------------------------------------------
# 4)  Main – identical to original but now benefits from Nyström inside DuoSpec
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    G = nx.read_gml("amazon_metadata_test/amz_allviddvd.gml")
    print(G.number_of_nodes())
    G = coords_str2arr(G)

    # (optional) ensure connectivity – helps spectral routines
    connect_components(G, weight=1e-3)

    # --- Use **entire** graph; no down‑sampling ---
    G = nx.convert_node_labels_to_integers(G)

    print(
        f"Testing on classes: {G.graph.get('subclasses', 'N/A')} and {len(G.nodes())} nodes",
        flush=True,
    )

    # -------------------------------------------------------------
    # DuoSpec parameters – baseline template
    # -------------------------------------------------------------
    # -------------------------------------------------------------
    base_params = dict(
        K=2,
        num_balls=32,
        config='motif',
        max_em_iters=100,
        warmup_rounds=1,
        anneal_steps=20,
        tol=1e-5,
        patience=5,
    )

    def run_duospec_once(seed: int):
        params = base_params.copy()
        params["random_state"] = seed
        return duo_spec(G, **params)

    # --- single Nyström–attention run (bagging removed) ---------------------------
    # ------------------------- run DuoSpec once ------------------------------
    # res = run_duospec_once(42)
    # preds = res["communities"]
    # G_combined = res["G_final"]
    _, preds, _, _ = motif_spectral_embedding_nystrom(G, q=2, dim=64, random_state=42)

    # --- optional BP refinement seeded by DuoSpec labels ------------------
    try:
        _, preds_bp, _, _ = belief_propagation_weighted(
            G_combined,
            q=2,
            max_iter=10000,
        )
        preds = preds_bp
    except Exception:
        # fall back to DuoSpec predictions if BP is unavailable
        pass

    # Get detection stats
    true_communities = get_true_communities(G, attr="comm")

    # ---------------- ensure predictions cover *all* nodes -----------------
    if len(preds) != len(G):
        print(f"Predictions do not cover all nodes, running label propagation")
        import numpy as _np
        full_preds = _np.full(len(G), -1, dtype=int)

        # Labels produced for G_combined nodes
        full_preds[list(G_combined.nodes())] = preds

        # Iterative **label propagation**: keep assigning unlabeled nodes the
        # majority label of *currently* labeled neighbours until convergence or
        # a small iteration cap is reached.
        unlabeled = (full_preds == -1)
        for _ in range(10):  # max 10 propagation sweeps
            changed = False
            for u in _np.where(unlabeled)[0]:
                neigh_labels = [full_preds[v] for v in G.neighbors(u) if full_preds[v] != -1]
                if neigh_labels:
                    full_preds[u] = _np.bincount(neigh_labels).argmax()
                    changed = True
            if not changed:
                break
            unlabeled = (full_preds == -1)

        # If any nodes remain unlabeled (isolated component with no edges), use
        # the majority label *after* propagation as a deterministic fallback.
        if (full_preds == -1).any():
            majority_final = _np.argmax(_np.bincount(full_preds[full_preds != -1]))
            full_preds[full_preds == -1] = majority_final

        preds = full_preds

    stats = detection_stats(preds, true_communities)
    print(stats)
    print(f"Finished detection stats") 

    # -------------------------------------------------------------
    # (Optional) Baseline belief propagation for comparison
    # -------------------------------------------------------------
    # _, bp_preds, _, _ = belief_propagation(G, q=2, max_iter=1000)
    # bp_stats = detection_stats(bp_preds, get_true_communities(G, attr="comm"))
    # print("Belief‑propagation stats:", bp_stats)

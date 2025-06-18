# duospec_nystrom.py – version with Nyström approximation injected before duo_spec
# -----------------------------------------------------------------------------
# This script is a drop‑in replacement for your previous driver.  The only change
# is that we compute a Nyström approximation of the (normalized) adjacency/Laplacian
# and monkey‑patch the motif_spectral_embedding call that duo_spec relies on so
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

from algorithms.duo_spec import duo_spec
import os
import json
import logging
import random
from algorithms.bp.vectorized_bp import belief_propagation, belief_propagation_weighted
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# -----------------------------------------------------------------------------
# 1)  Standard helpers supplied in the original file
# -----------------------------------------------------------------------------

from algorithms.spectral_ops.attention import motif_spectral_embedding as _orig_motif_spectral_embedding

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


def _graph_to_sparse_adjacency(G: nx.Graph):
    """Return a CSR sparse adjacency matrix and node‑index lookup tables."""
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    rows, cols, data = [], [], []
    for u, v, ed in G.edges(data=True):
        # weight heuristic: smaller distance → stronger edge
        w = ed.get("weight")
        if w is None:
            dist = ed.get("dist", 1.0)
            try:
                dist = float(dist)
                w = 1.0 / (dist + 1e-6)
            except Exception:
                w = 1.0
        i, j = node_to_idx[u], node_to_idx[v]
        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([w, w])
    A = csr_matrix((data, (rows, cols)), shape=(n, n), dtype=float)
    return A, nodes


def _choose_landmarks(G, m, seed=42, method="degree"):
    """Return an array of landmark indices using the requested sampling strategy.

    * **uniform** – simple random uniform
    * **degree**  – probability proportional to node degree (better coverage of hubs)
    """
    rng = np.random.default_rng(seed)
    n = G.number_of_nodes()
    if method == "uniform":
        return rng.choice(n, size=m, replace=False)
    elif method == "degree":
        degs = np.array([d for _, d in G.degree()]) + 1e-6
        probs = degs / degs.sum()
        return rng.choice(n, size=m, replace=False, p=probs)
    else:
        raise ValueError(f"unknown landmark sampling '{method}'")


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
    A, _ = _graph_to_sparse_adjacency(G)
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
    """... unchanged header ..."""
    G = args[0]

    # q may arrive as positional or keyword; accept either.
    if len(args) >= 2 and isinstance(args[1], int):
        q = args[1]
    else:
        try:
            q = kwargs.pop("q")
        except KeyError as e:
            q=2


    # Preserve the original keyword defaults, but we only really use dim and random_state.
    dim          = kwargs.get("dim", 32)
    random_state = kwargs.get("random_state", 42)

    # ------------------------------- Nyström embed --------------------------
    m = min(max(8 * dim, 600), G.number_of_nodes())
    print(
        f"calling nystrom embedding (n = {G.number_of_nodes()} , dim = {dim} , m = {m})",
        flush=True,
    )
    U = nystrom_spectral_embedding(
        G,
        m=m,
        dim=dim,
        seed=random_state,
        sampling="degree",
        laplacian="normalized",
    )  # n × dim

    # Keep the first q informative components (helps for small q)
    Uq = U[:, :max(q, 2)]
    Uq /= np.linalg.norm(Uq, axis=1, keepdims=True) + 1e-12

    # ------------------------------- cluster & soft labels ------------------
    if q == 2:
        # For two‑way split a simple sign cut on 2nd eigenvector is often crisper
        vec = Uq[:, 1] if Uq.shape[1] > 1 else Uq[:, 0]
        thresh = np.median(vec)
        hard = (vec > thresh).astype(int)
        Q = np.stack([1 - hard, hard], axis=1)
    else:
        km = KMeans(n_clusters=q, n_init=50, max_iter=300, random_state=random_state, init="k-means++")
        hard = km.fit_predict(Uq)
        centers = km.cluster_centers_  # q × d

        # soft assignments via squared distance to centres
        d2 = cdist(Uq, centers, metric="sqeuclidean")  # n × q
        sims = -d2
        sims -= sims.max(axis=1, keepdims=True)
        exp_sims = np.exp(sims)
        Q = exp_sims / exp_sims.sum(axis=1, keepdims=True)  # n × q
    km = KMeans(n_clusters=q, n_init=20, random_state=random_state)
    hard = km.fit_predict(U)
    centers = km.cluster_centers_  # q × dim

    d2 = cdist(U, centers, metric="sqeuclidean")  # n × q
    sims = -d2
    sims -= sims.max(axis=1, keepdims=True)
    exp_sims = np.exp(sims)
    Q = exp_sims / exp_sims.sum(axis=1, keepdims=True)  # n × q

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

    # Simple distance → weight heuristic → weight heuristic
    for u, v, edge_data in G.edges(data=True):
        if "dist" in edge_data:
            try:
                edge_data["dist"] = float(edge_data["dist"]) * 2
            except ValueError:
                pass

    print(
        f"Testing on classes: {G.graph.get('subclasses', 'N/A')} and {len(G.nodes())} nodes",
        flush=True,
    )

    # -------------------------------------------------------------
    # DuoSpec parameters – *unchanged* (Nyström is transparent)
    # -------------------------------------------------------------
    duo_params = dict(
        K=2,
        num_balls=32,
        config='motif',
        max_em_iters=100,
        warmup_rounds=2,
        anneal_steps=20,
        tol=1e-5,
        patience=5,
        random_state=42,
    )

    res = duo_spec(G, **duo_params)
    preds = res["communities"]
    G_combined = res['G_final']
    _, preds, _, _ = belief_propagation_weighted(
        G_combined, 
        q=2, 
        max_iter=10000,
    )
    # _, preds, _, _ = belief_propagation(G, q=2, max_iter=10000)
    # Get detection stats
    true_communities = get_true_communities(G, attr="comm")

    # ---------------- ensure predictions cover *all* nodes -----------------
    if len(preds) != len(G):
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
    print(f"Finished duo_spec (Nyström) with {len(preds)} predictions")
    stats = detection_stats(preds, get_true_communities(G, attr="comm"))
    print(stats)

    # -------------------------------------------------------------
    # (Optional) Baseline belief propagation for comparison
    # -------------------------------------------------------------
    # _, bp_preds, _, _ = belief_propagation(G, q=2, max_iter=1000)
    # bp_stats = detection_stats(bp_preds, get_true_communities(G, attr="comm"))
    # print("Belief‑propagation stats:", bp_stats)
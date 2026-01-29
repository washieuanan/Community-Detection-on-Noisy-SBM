from __future__ import annotations
from collections import defaultdict
from copy import deepcopy
import math
import random
import time
from typing import Any, Dict, Iterable, Tuple, Union, List

# NumPy and SciPy
import numpy as np
from numpy.linalg import norm
from numpy.random import default_rng
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from scipy.stats import mode, permutation_test, spearmanr, pearsonr, rankdata, beta as beta_dist

# NetworkX
import networkx as nx

# Scikit-learn
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KernelDensity


def create_dist_observed_subgraph(num_coords, observations):
    """
    create subgraph containing all nodes and observed paths as edges
    """
    subG = nx.Graph()
    
    for c in range(num_coords):
        subG.add_node(c)
     
    for p, d in observations:
        subG.add_edge(p[0], p[1], dist=d)
    return subG

def _conf_from_center(X, mu):
    """
    Confidence matrix Q[i,c] = 1 / (1 + ||x_i − μ_c||_2)
    Rows are re-normalised to sum to 1.
    """
    dists = np.linalg.norm(X[:, None, :] - mu[None, :, :], axis=-1)  # (n,q)
    Q     = 1.0 / (1.0 + dists)
    Q    /= Q.sum(axis=1, keepdims=True)
    return Q


def edge_locality_scores(
    G: nx.Graph,
    edges: np.ndarray,
    *,
    metric: str = "cn_over_sqrtdeg",
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Fast per-edge locality score L(u,v) using only adjacency structure.

    Parameters
    ----------
    G : nx.Graph
    edges : np.ndarray of shape (m,2)
        Edge list whose order matches any masks/confidences that will
        be applied later (e.g. in `_scale_edges`).
    metric : {"cn_over_sqrtdeg","jaccard","common_neighbors"}
    """
    m = len(edges)
    if m == 0:
        return np.zeros(0, dtype=float)

    adj = G.adj
    # Precompute degrees once
    deg = {u: len(adj[u]) for u in G.nodes()}

    scores = np.empty(m, dtype=float)

    for i, (u, v) in enumerate(edges):
        # Handle possible self-loops defensively
        if u == v:
            scores[i] = 0.0
            continue

        adj_u = adj[u]
        adj_v = adj[v]

        # Iterate over smaller neighborhood for common-neighbor count
        if len(adj_u) > len(adj_v):
            adj_u, adj_v = adj_v, adj_u
            u_deg, v_deg = deg[v], deg[u]
        else:
            u_deg, v_deg = deg[u], deg[v]

        cn = 0
        for w in adj_u:
            if w in adj_v:
                cn += 1

        if metric == "common_neighbors":
            s = float(cn)
        elif metric == "jaccard":
            # union size = d(u) + d(v) - CN
            denom = (u_deg + v_deg - cn) + eps
            s = float(cn) / denom
        else:  # "cn_over_sqrtdeg" default
            denom = math.sqrt(u_deg * v_deg + eps)
            s = float(cn) / denom if denom > 0 else 0.0

        scores[i] = s

    return scores


def mixture_diag_gauss_posteriors(
    X: np.ndarray,
    r_init: np.ndarray,
    *,
    a0: float = 1.0,
    b0: float = 1.0,
    var_floor: float = 1e-4,
    inner_iters: int = 2,
) -> np.ndarray:
    """
    Simple 2-component diagonal-Gaussian mixture with EM over responsibilities.

    Components:
      k=0 : GEO
      k=1 : COMM
    Returns updated responsibilities r_geo = P(z=GEO | x).
    """
    n, d = X.shape
    if n == 0:
        return np.zeros(0, dtype=float)

    r = r_init.astype(float).clip(1e-3, 1.0 - 1e-3)

    for _ in range(max(1, inner_iters)):
        # M-step
        N_geo = float(r.sum()) + 1e-8
        N_comm = float(n - r.sum()) + 1e-8

        pi_geo = (a0 + r.sum()) / (a0 + b0 + n)
        pi_geo = float(np.clip(pi_geo, 1e-3, 1.0 - 1e-3))
        pi_comm = 1.0 - pi_geo

        # Means
        mu_geo = (r[:, None] * X).sum(axis=0) / N_geo
        mu_comm = ((1.0 - r)[:, None] * X).sum(axis=0) / N_comm

        # Variances (diagonal)
        diff_geo = X - mu_geo
        diff_comm = X - mu_comm
        var_geo = (r[:, None] * (diff_geo ** 2)).sum(axis=0) / N_geo
        var_comm = ((1.0 - r)[:, None] * (diff_comm ** 2)).sum(axis=0) / N_comm

        var_geo = np.maximum(var_geo, var_floor)
        var_comm = np.maximum(var_comm, var_floor)

        # Enforce identifiability: component 0 ("geo") must have higher
        # mean locality (x1) than component 1 ("comm"). If not, swap them.
        if mu_geo[0] < mu_comm[0]:
            mu_geo, mu_comm = mu_comm, mu_geo
            var_geo, var_comm = var_comm, var_geo
            pi_geo, pi_comm = pi_comm, pi_geo

        # E-step: compute log posteriors
        # log N(x | mu, var) for diagonal Gaussian
        log2pi = d * math.log(2.0 * math.pi)

        def log_norm(x, mu, var):
            diff = x - mu
            term = (diff * diff / var).sum(axis=1)
            return -0.5 * (log2pi + np.log(var).sum() + term)

        log_p_geo = log_norm(X, mu_geo, var_geo) + math.log(pi_geo)
        log_p_comm = log_norm(X, mu_comm, var_comm) + math.log(pi_comm)

        # log-sum-exp for normalisation
        m = np.maximum(log_p_geo, log_p_comm)
        log_den = m + np.log(np.exp(log_p_geo - m) + np.exp(log_p_comm - m))
        r = np.exp(log_p_geo - log_den)

    return r.clip(1e-3, 1.0 - 1e-3)


def geometry_scores_fineblob_persistence(
    G: nx.Graph,
    edges: np.ndarray,
    scores_local: np.ndarray,
    node2idx: Dict[Any, int],
    *,
    frac_sweep: Tuple[float, ...] = (0.995, 0.99, 0.98, 0.97, 0.95),
    fine_frac: float = 0.99,
    S0: int = 20,
    stable_k: int = 3,
    debug: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Fine-blob persistence geometry discriminator (structure-only).

    1) Build a fine partition H_fine using high-locality edges (global percentile fine_frac).
       This yields many small blobs across the graph.
    2) For each sweep level in frac_sweep, run a DSU over edges with score ≥ tau_f and
       count edges that lie in small components (size ≤ S0).
    3) Combine persistence with fine-blob membership:
         r_geo(e) = pers_sharp(e) * 1[same_blob(e)] * 1[blob_size(u) <= S0]
       where pers_sharp enforces stable_k occurrences across scales.
    4) Bridge indicator:
         bridge_score(e) = 1 if endpoints lie in different fine blobs, else 0.
    """
    m = len(edges)
    if m == 0:
        return {
            "r_geo": np.zeros(0, dtype=float),
            "same_blob": np.zeros(0, dtype=bool),
            "blob_id_per_node": np.zeros(0, dtype=int),
            "blob_size_per_node": np.zeros(0, dtype=int),
            "bridge_score": np.zeros(0, dtype=float),
            "count_small": np.zeros(0, dtype=int),
        }

    if not frac_sweep:
        raise ValueError("frac_sweep must be non-empty for fineblob persistence discriminator.")

    nodes = list(G.nodes())
    n = len(nodes)
    idx_of = node2idx

    # --- fine partition via high-locality subgraph --------------------------
    thr_fine = np.percentile(scores_local, fine_frac * 100.0)
    high_mask_fine = scores_local >= thr_fine

    parent_fine = np.arange(n, dtype=int)
    size_fine = np.ones(n, dtype=int)

    def find_f(i: int) -> int:
        while parent_fine[i] != i:
            parent_fine[i] = parent_fine[parent_fine[i]]
            i = parent_fine[i]
        return i

    def union_f(i: int, j: int):
        ri, rj = find_f(i), find_f(j)
        if ri == rj:
            return
        if size_fine[ri] < size_fine[rj]:
            ri, rj = rj, ri
        parent_fine[rj] = ri
        size_fine[ri] += size_fine[rj]

    for idx, (u, v) in enumerate(edges):
        if not high_mask_fine[idx]:
            continue            
        ui = idx_of[u]
        vi = idx_of[v]
        union_f(ui, vi)

    blob_id_per_node = np.zeros(n, dtype=int)
    blob_size_per_node = np.zeros(n, dtype=int)
    for i in range(n):
        r = find_f(i)
        blob_id_per_node[i] = r
        blob_size_per_node[i] = size_fine[r]

    same_blob = np.zeros(m, dtype=bool)
    bridge_score = np.zeros(m, dtype=float)
    for idx, (u, v) in enumerate(edges):
        ui = idx_of[u]
        vi = idx_of[v]
        bu = blob_id_per_node[ui]
        bv = blob_id_per_node[vi]
        if bu == bv:
            same_blob[idx] = True
        else:
            bridge_score[idx] = 1.0

    # --- multi-threshold persistence over locality --------------------------
    levels = sorted(frac_sweep, reverse=True)
    L = len(levels)
    thresholds = np.percentile(scores_local, [f * 100.0 for f in levels])

    count_small = np.zeros(m, dtype=int)

    def dsu_small_components(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        parent = np.arange(n, dtype=int)
        size = np.ones(n, dtype=int)

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int):
            ri, rj = find(i), find(j)
            if ri == rj:
                return
            if size[ri] < size[rj]:
                ri, rj = rj, ri
            parent[rj] = ri
            size[ri] += size[rj]

        for idx, (u, v) in enumerate(edges):
            if not mask[idx]:
                continue
            ui = idx_of[u]
            vi = idx_of[v]
            union(ui, vi)

        comp_size = np.zeros(n, dtype=int)
        for i in range(n):
            r = find(i)
            comp_size[r] += 1

        return parent, comp_size

    for _, thr in enumerate(thresholds):
        high_mask = scores_local >= thr
        if not high_mask.any():
            continue

        parent, comp_size = dsu_small_components(high_mask)

        for idx, (u, v) in enumerate(edges):
            if not high_mask[idx]:
                continue
            ui = idx_of[u]
            vi = idx_of[v]
            while parent[ui] != ui:
                parent[ui] = parent[parent[ui]]
                ui = parent[ui]
            while parent[vi] != vi:
                parent[vi] = parent[parent[vi]]
                vi = parent[vi]
            if ui != vi:
                continue
            size = comp_size[ui]
            if size <= S0:
                count_small[idx] += 1

    # persistence score
    if stable_k > 1:
        offset = stable_k - 1
        denom = max(1, L - offset)
        pers_sharp = (count_small.astype(float) - float(offset)) / float(denom)
        pers_sharp = np.clip(pers_sharp, 0.0, 1.0)
    else:
        pers_sharp = count_small.astype(float) / float(L)

    # Enforce fine-blob theory: small, persistent blobs only
    r_geo = np.zeros(m, dtype=float)
    for idx, (u, v) in enumerate(edges):
        if not same_blob[idx]:
            continue
        ui = idx_of[u]
        if blob_size_per_node[ui] > S0:
            continue
        r_geo[idx] = pers_sharp[idx]

    if debug:
        unique, counts = np.unique(count_small, return_counts=True)
        hist_str = ", ".join(f"{u}:{c}" for u, c in zip(unique, counts))
        print(
            f"[GeoFineDSU] small-count histogram (count_small -> num_edges): {hist_str}"
        )

    return {
        "r_geo": r_geo,
        "same_blob": same_blob,
        "blob_id_per_node": blob_id_per_node,
        "blob_size_per_node": blob_size_per_node,
        "bridge_score": bridge_score,
        "count_small": count_small,
    }


def build_blob_supergraph(
    G: nx.Graph,
    edges: np.ndarray,
    node2idx: Dict[Any, int],
    blob_id_per_node: np.ndarray,
    r_geo: np.ndarray,
    *,
    weight_key: str = "weight",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a coarse blob-level graph where:
      - supernodes are fine blobs,
      - edges connect blobs that share at least one original edge,
      - edge weights aggregate non-geo mass: sum_e (1 - r_geo[e]) * w_e.

    Returns
    -------
    blob_edges : (m_B, 2) np.ndarray of int
        Undirected edges between blob indices in [0, n_blobs).
    blob_weights : (m_B,) np.ndarray of float
        Aggregated weights per blob edge.
    blob_comp_per_node : (n,) np.ndarray of int
        Compact blob index for each node in `G` (0..n_blobs-1).
    """
    n = len(blob_id_per_node)
    if n == 0 or len(edges) == 0:
        return np.zeros((0, 2), dtype=int), np.zeros(0, dtype=float), np.zeros(0, dtype=int)

    # Compact blob ids to [0, n_blobs)
    unique_blobs, blob_comp_per_node = np.unique(blob_id_per_node, return_inverse=True)
    n_blobs = int(len(unique_blobs))

    agg: Dict[Tuple[int, int], float] = {}

    for e_idx, (u, v) in enumerate(edges):
        ui = node2idx[u]
        vi = node2idx[v]
        bu = int(blob_comp_per_node[ui])
        bv = int(blob_comp_per_node[vi])
        if bu == bv:
            continue

        key = (bu, bv) if bu < bv else (bv, bu)
        w_attr = float(G[u][v].get(weight_key, 1.0))
        contrib = (1.0 - float(r_geo[e_idx])) * w_attr
        if contrib <= 0.0:
            continue
        agg[key] = agg.get(key, 0.0) + contrib

    if not agg:
        return np.zeros((0, 2), dtype=int), np.zeros(0, dtype=float), blob_comp_per_node

    blob_edges = np.array(list(agg.keys()), dtype=int)
    blob_weights = np.array(list(agg.values()), dtype=float)

    return blob_edges, blob_weights, blob_comp_per_node


def dsu_exact_k_partition(
    num_nodes: int,
    edges_u: np.ndarray,
    edges_v: np.ndarray,
    weights: np.ndarray,
    K: int,
    *,
    seed: int = 0,
    tie_break: str = "lex",
) -> np.ndarray:
    """
    DSU-based exact-K partition on a weighted graph.

    - Runs a Kruskal-style union over edges sorted by descending weight
      (with deterministic lexicographic tie-breaking).
    - Stops when exactly K components remain (if possible).
    - If the graph is too disconnected to reach K via edges alone, performs
      deterministic forced merges of the smallest components until K remain.

    Returns
    -------
    labels : (num_nodes,) np.ndarray[int]
        Community labels in {0, ..., K-1}.
    """
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}.")
    if num_nodes <= 0:
        return np.zeros(0, dtype=int)
    if K > num_nodes:
        raise ValueError(f"K={K} cannot exceed num_nodes={num_nodes}.")

    parent = np.arange(num_nodes, dtype=int)
    size = np.ones(num_nodes, dtype=int)
    num_sets = num_nodes

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> bool:
        nonlocal num_sets
        ri, rj = find(i), find(j)
        if ri == rj:
            return False
        if size[ri] < size[rj]:
            ri, rj = rj, ri
        parent[rj] = ri
        size[ri] += size[rj]
        num_sets -= 1
        return True

    m = len(edges_u)
    if m != len(edges_v) or m != len(weights):
        raise ValueError("edges_u, edges_v, and weights must have the same length.")

    if m > 0:
        # Deterministic sort: primary -weight (descending), then u, then v.
        u = edges_u.astype(int)
        v = edges_v.astype(int)
        w = weights.astype(float)
        # lexsort keys: last key is primary
        order = np.lexsort((v, u, -w))
        u_sorted = u[order]
        v_sorted = v[order]

        for uu, vv in zip(u_sorted, v_sorted):
            if num_sets <= K:
                break
            union(int(uu), int(vv))

    # If we still have more than K components, merge smallest ones deterministically.
    def current_roots_and_sizes() -> Tuple[np.ndarray, np.ndarray]:
        roots = np.arange(num_nodes, dtype=int)
        for i in range(num_nodes):
            roots[i] = find(i)
        unique_roots, counts = np.unique(roots, return_counts=True)
        return unique_roots, counts

    if num_sets > K:
        while num_sets > K:
            roots, counts = current_roots_and_sizes()
            # sort ascending by (size, root id) so we merge the two smallest
            idx = np.argsort(np.stack([counts, roots], axis=1).tolist(), axis=0)[:, 0]
            r1 = int(roots[idx[0]])
            r2 = int(roots[idx[1]])
            union(r1, r2)

    # After unions and forced merges, we expect exactly K components.
    roots, counts = current_roots_and_sizes()
    if len(roots) != K:
        raise RuntimeError(
            f"DSU exact-K partition invariant violated: expected {K} components, "
            f"found {len(roots)}."
        )

    # Relabel components to contiguous 0..K-1 by descending size then root id.
    order = np.lexsort((roots, -counts))  # primary: -size, tie-breaker: root id
    selected_roots = roots[order]  # length K

    label_of_root: Dict[int, int] = {}
    for new_id, r in enumerate(selected_roots):
        label_of_root[int(r)] = int(new_id)

    labels = np.zeros(num_nodes, dtype=int)
    for i in range(num_nodes):
        r = find(i)
        labels[i] = label_of_root[int(r)]

    return labels

def community_proxy_persistence_on_blob_graph(
    blob_edges: np.ndarray,
    blob_weights: np.ndarray,
    num_blobs: int,
    K: int,
    *,
    frac_sweep: Tuple[float, ...],
    stable_k: int = 3,
) -> np.ndarray:
    """
    Exact‑K DSU partition over a contracted blob graph to obtain a coarse
    community proxy with exactly K (non-empty) communities when possible.

    Returns
    -------
    coarse_id_per_blob : (num_blobs,) np.ndarray[int]
        Coarse community id for each blob (0..C-1).
    """
    if num_blobs == 0:
        return np.zeros(0, dtype=int)
    if K > num_blobs:
        raise ValueError(f"K={K} cannot exceed num_blobs={num_blobs}.")

    if blob_edges.size == 0:
        # No inter-blob edges: rely solely on DSU exact-K forced merges.
        edges_u = np.zeros(0, dtype=int)
        edges_v = np.zeros(0, dtype=int)
        weights = np.zeros(0, dtype=float)
    else:
        edges_u = blob_edges[:, 0].astype(int)
        edges_v = blob_edges[:, 1].astype(int)
        weights = blob_weights.astype(float)

    coarse_ids = dsu_exact_k_partition(
        num_nodes=num_blobs,
        edges_u=edges_u,
        edges_v=edges_v,
        weights=weights,
        K=K,
        seed=0,
        tie_break="lex",
    )
    return coarse_ids


def reweight_edges_from_posteriors(
    G: nx.Graph,
    edges: np.ndarray,
    r_geo: np.ndarray,
    p_same: np.ndarray,
    *,
    lam_geo: float,
    lam_comm_boost: float,
    w_min: float,
    w_cap: float,
    delta_cap: float,
    weight_key: str = "weight",
    bridge_score: np.ndarray | None = None,
    geo_gate_enabled: bool = True,
    gate_power: float = 2.0,
    gate_floor: float = 0.05,
    use_comm_boost: bool = True,
) -> Dict[str, float]:
    """
    Smooth, monotone reweighting based on geometry and community signals.

    Per-edge update:
        r = clip(r_geo, 0, 1)
        ps = clip(p_same, 0, 1)
        gate = (1 - ps) ** gate_power    (if geo_gate_enabled else 1)
        r_eff = r * gate
        comm_eff = (1 - r) * ps
        b_eff = bridge_score (if provided) else 0

        delta = (-lam_geo * r_eff) + (lam_comm * comm_eff)
        delta = clip(delta, -delta_cap, +delta_cap)
        w_new = w * (1 + delta), then clamped into [w_min, w_cap].
    """
    m = len(edges)
    if m == 0:
        return dict(
            mean_abs_delta=0.0,
            frac_w_min=0.0,
            frac_w_cap=0.0,
            num_changed=0,
            mean_delta_geo=0.0,
            mean_delta_comm=0.0,
            frac_boosted=0.0,
            frac_shrunk=0.0,
        )

    # Basic consistency checks to avoid silent no-ops.
    assert len(r_geo) == m and len(p_same) == m, "Length mismatch between edges and signals."
    assert np.all(np.isfinite(r_geo)), "Non-finite r_geo encountered."
    assert np.all(np.isfinite(p_same)), "Non-finite p_same encountered."
    assert w_cap > w_min, "w_cap must be strictly greater than w_min."

    if bridge_score is None:
        bridge_score = np.zeros(m, dtype=float)
    else:
        assert len(bridge_score) == m, "bridge_score length mismatch."

    lam_geo_eff = float(lam_geo) if lam_geo is not None else 0.0
    lam_comm_boost_eff = float(lam_comm_boost) if lam_comm_boost is not None else 0.0
    delta_cap_eff = float(delta_cap)
    assert delta_cap_eff >= 0.0, "delta_cap must be non-negative."

    deltas = []
    deltas_geo = []
    deltas_comm = []
    at_min = 0
    at_cap = 0
    num_changed = 0
    n_boosted = 0
    n_shrunk = 0

    for (u, v), r, ps, br in zip(edges, r_geo, p_same, bridge_score):
        w = float(G[u][v].get(weight_key, 1.0))

        # Clip signals
        r_clipped = float(np.clip(r, 0.0, 1.0))
        ps_clipped = float(np.clip(ps, 0.0, 1.0))

        # Geometry gate: suppress shrinkage on strong community edges.
        if geo_gate_enabled:
            gate = (1.0 - ps_clipped) ** gate_power
            if gate_floor is not None and gate_floor > 0.0:
                gate = max(gate, float(gate_floor))
        else:
            gate = 1.0

        r_eff = r_clipped * gate
        comm_eff = (1.0 - r_clipped) * ps_clipped
        b_eff = float(br) if br is not None else 0.0

        # Geometry shrink component (always multiplicative in this implementation).
        delta_geo = -lam_geo_eff * r_eff

        # Community boost component (only if enabled and in same coarse community).
        if use_comm_boost and ps_clipped > 0.0 and lam_comm_boost_eff > 0.0:
            delta_comm = lam_comm_boost_eff * comm_eff
        else:
            delta_comm = 0.0

        # Combined signed delta before clipping (geometry + community channels).
        delta = delta_geo + delta_comm
        delta = float(np.clip(delta, -delta_cap_eff, delta_cap_eff))

        # Apply update multiplicatively; boost/shrink modes are kept for future
        # extensions but currently only "mul" is supported.
        w_new = w * (1.0 + delta)
        w_new = float(min(w_cap, max(w_min, w_new)))

        if abs(w_new - w) > 0.0:
            num_changed += 1

        deltas.append(abs(w_new - w))
        deltas_geo.append(delta_geo)
        deltas_comm.append(delta_comm)

        if delta_geo < 0.0:
            n_shrunk += 1
        if delta_comm > 0.0:
            n_boosted += 1

        if w_new <= w_min + 1e-12:
            at_min += 1
        if w_new >= w_cap - 1e-12:
            at_cap += 1

        G[u][v][weight_key] = w_new

    mean_abs_delta = float(np.mean(deltas)) if deltas else 0.0
    frac_w_min = at_min / float(m)
    frac_w_cap = at_cap / float(m)
    mean_delta_geo = float(np.mean(deltas_geo)) if deltas_geo else 0.0
    mean_delta_comm = float(np.mean(deltas_comm)) if deltas_comm else 0.0
    frac_boosted = n_boosted / float(m)
    frac_shrunk = n_shrunk / float(m)

    return dict(
        mean_abs_delta=mean_abs_delta,
        frac_w_min=frac_w_min,
        frac_w_cap=frac_w_cap,
        num_changed=num_changed,
        mean_delta_geo=mean_delta_geo,
        mean_delta_comm=mean_delta_comm,
        frac_boosted=frac_boosted,
        frac_shrunk=frac_shrunk,
    )


def prune_by_weight_keep_connected(
    G: nx.Graph,
    *,
    weight_key: str = "weight",
    prune_frac: float = 0.20,
    seed: int = 0,
) -> nx.Graph:
    """
    Prune a fraction of the lowest-weight edges while preserving connectivity
    of the returned component. Operates on existing edges only and returns a
    new graph with binary edge weights (all remaining edges have weight=1.0).
    """
    if G.number_of_nodes() <= 1 or G.number_of_edges() == 0:
        H = G.copy()
    else:
        if nx.is_connected(G):
            H = G.copy()
        else:
            comps = list(nx.connected_components(G))
            comps.sort(key=lambda c: (-len(c), min(c)))
            largest = comps[0]
            H = G.subgraph(largest).copy()

    m = H.number_of_edges()
    if m == 0:
        G_pruned = nx.Graph()
        for u, data in H.nodes(data=True):
            G_pruned.add_node(u, **data)
        return G_pruned

    target_remove = int(prune_frac * m)
    if target_remove <= 0:
        G_pruned = nx.Graph()
        for u, data in H.nodes(data=True):
            G_pruned.add_node(u, **data)
        for u, v, data in H.edges(data=True):
            attrs = dict(data)
            attrs["weight"] = 1.0
            G_pruned.add_edge(u, v, **attrs)
        return G_pruned

    edges_sorted = sorted(
        H.edges(data=True),
        key=lambda e: (
            float(e[2].get(weight_key, 1.0)),
            min(e[0], e[1]),
            max(e[0], e[1]),
        ),
    )

    removed = 0
    for u, v, _ in edges_sorted:
        if removed >= target_remove:
            break
        bridges = set()
        for a, b in nx.bridges(H):
            bridges.add((a, b))
            bridges.add((b, a))
        if (u, v) in bridges:
            continue
        H.remove_edge(u, v)
        removed += 1

    if H.number_of_edges() == 0:
        G_pruned = nx.Graph()
        for u, data in H.nodes(data=True):
            G_pruned.add_node(u, **data)
    else:
        if not nx.is_connected(H):
            comps = list(nx.connected_components(H))
            comps.sort(key=lambda c: (-len(c), min(c)))
            largest = comps[0]
            H = H.subgraph(largest).copy()
        G_pruned = nx.Graph()
        for u, data in H.nodes(data=True):
            G_pruned.add_node(u, **data)
        for u, v, data in H.edges(data=True):
            attrs = dict(data)
            attrs["weight"] = 1.0
            G_pruned.add_edge(u, v, **attrs)

    return G_pruned


def denoise_then_prune_binary(
    G_denoised: nx.Graph,
    *,
    prune_frac: float,
    weight_key: str = "weight",
    seed: int = 0,
) -> nx.Graph:
    """
    Convenience wrapper: prune a denoised graph by weight while preserving
    connectivity, then return a binary-weight version for downstream methods.
    """
    return prune_by_weight_keep_connected(
        G_denoised, weight_key=weight_key, prune_frac=prune_frac, seed=seed
    )


def preserve_node_strengths(
    G: nx.Graph,
    *,
    base_strength: np.ndarray,
    node2idx: Dict[Any, int],
    weight_key: str = "weight",
    eta: float = 0.25,
    eps: float = 1e-12,
    w_min: float,
    w_cap: float,
) -> None:
    """
    Softly renormalise edge weights so each node’s weighted degree (strength)
    stays close to its initial strength.

    For node i:
        a_i = (base_strength[i] / (curr_strength[i] + eps)) ** eta

    For edge (u, v):
        w <- w * sqrt(a_u * a_v), then clamped into [w_min, w_cap].
    """
    n = len(node2idx)
    if n == 0 or G.number_of_edges() == 0:
        return

    curr_strength = np.zeros(n, dtype=float)
    for u, v, d in G.edges(data=True):
        w = float(d.get(weight_key, 1.0))
        iu = node2idx[u]
        iv = node2idx[v]
        curr_strength[iu] += w
        curr_strength[iv] += w

    a = np.ones(n, dtype=float)
    mask = curr_strength > eps
    a[mask] = (base_strength[mask] / (curr_strength[mask] + eps)) ** float(eta)

    for u, v, d in G.edges(data=True):
        iu = node2idx[u]
        iv = node2idx[v]
        w = float(d.get(weight_key, 1.0))
        scale = math.sqrt(a[iu] * a[iv])
        w_new = w * scale
        w_new = float(min(w_cap, max(w_min, w_new)))
        d[weight_key] = w_new


def proxy_weight_locality_correlation(
    G: nx.Graph,
    *,
    weight_key: str = "weight",
    local_score: str = "cn_over_sqrtdeg",
    corr: str = "spearman",
    jitter: float = 1e-9,
    eps: float = 1e-12,
    seed: int = 0,
) -> dict:
    """
    Proxy geometry–weight alignment: correlate edge weights w(e) with
    locality scores L_e (triangle closure) on edges only.
    """
    edges = np.asarray(list(G.edges()), dtype=object)
    m = len(edges)
    if m == 0:
        return {
            "spearman": float("nan"),
            "p_value": float("nan"),
            "n_edges": 0,
            "reason": "empty_graph",
        }

    scores = edge_locality_scores(G, edges, metric=local_score)
    weights = np.asarray(
        [float(G[u][v].get(weight_key, 1.0)) for u, v in edges], dtype=float
    )

    n_valid = len(weights)
    std_w = float(np.std(weights))
    std_s = float(np.std(scores))

    rng = np.random.default_rng(seed)

    def _maybe_jitter(x):
        return x + jitter * rng.standard_normal(len(x)) if jitter > 0.0 else x

    used_jitter = False
    reason = None

    if std_w < eps or std_s < eps:
        if jitter > 0.0:
            weights_corr = _maybe_jitter(weights)
            scores_corr = _maybe_jitter(scores)
            used_jitter = True
            reason = (
                f"computed_with_jitter_due_to_near_constant_inputs(std_w={std_w:.3e},"
                f" std_L={std_s:.3e})"
            )
        else:
            reason = (
                f"near_constant_inputs(std_w={std_w:.3e}, std_L={std_s:.3e})"
            )
            return {
                "spearman": float("nan"),
                "p_value": float("nan"),
                "n_edges": n_valid,
                "reason": reason,
            }
    else:
        weights_corr = weights
        scores_corr = scores

    if corr == "spearman":
        val, pval = spearmanr(weights_corr, scores_corr)
    else:
        val, pval = pearsonr(weights_corr, scores_corr)

    if (val is None or np.isnan(val)) and jitter > 0.0 and not used_jitter:
        weights_corr = _maybe_jitter(weights)
        scores_corr = _maybe_jitter(scores)
        used_jitter = True
        reason = (reason + "; " if reason else "") + "nan_fallback_to_jitter"
        val, pval = spearmanr(weights_corr, scores_corr)

    return {
        "spearman": float(val) if val is not None else float("nan"),
        "p_value": float(pval) if pval is not None else float("nan"),
        "n_edges": n_valid,
        "reason": reason,
    }


# DEPRECATED/UNUSED (kept for now): distance-based geometry–graph correlation.
def weight_coord_distance_correlation(
    G: nx.Graph,
    *,
    coord_key: str = "coords",
    weight_key: str = "weight",
    corr: str = "spearman",
    jitter: float = 1e-9,
    eps: float = 1e-12,
    seed: int = 0,
    min_edges: int = 5,
) -> dict:
    """
    Coordinate-based geometry alignment:
    correlate edge weights w(e) with Euclidean distance d(e) computed from node coords.

    EDGES ONLY. Read-only: must not modify graph.
    """
    coords = {}
    missing_endpoints = 0

    for u in G.nodes():
        dat = G.nodes[u]
        if coord_key in dat:
            coords[u] = np.asarray(dat[coord_key], dtype=float)

    dists = []
    weights = []

    for u, v, ed in G.edges(data=True):
        if u not in coords or v not in coords:
            missing_endpoints += 1
            continue
        cu = coords[u]
        cv = coords[v]
        d = float(np.linalg.norm(cu - cv))
        w = float(ed.get(weight_key, 1.0))
        dists.append(d)
        weights.append(w)

    n_valid = len(dists)
    if n_valid < min_edges:
        reason = f"not_enough_edges_with_coords(n_valid={n_valid}, min_edges={min_edges})"
        print(
            "[Metric] weight-coord-distance correlation: "
            f"{reason}"
        )
        return {
            "corr": float("nan"),
            "p_value": float("nan"),
            "n_edges": int(n_valid),
            "missing_endpoints": int(missing_endpoints),
            "metric": corr,
            "used_jitter": False,
            "reason": reason,
        }

    dists = np.asarray(dists, dtype=float)
    weights = np.asarray(weights, dtype=float)

    # Basic stats
    std_w = float(np.std(weights))
    std_d = float(np.std(dists))

    rng = np.random.default_rng(seed)

    def _maybe_jitter(x: np.ndarray) -> np.ndarray:
        return x + jitter * rng.standard_normal(len(x)) if jitter > 0.0 else x

    used_jitter = False
    reason = None

    # Guard against near-constant inputs
    if std_w < eps or std_d < eps:
        if jitter > 0.0:
            weights_corr = _maybe_jitter(weights)
            dists_corr = _maybe_jitter(dists)
            used_jitter = True
            reason = (
                f"computed_with_jitter_due_to_near_constant_inputs(std_w={std_w:.3e},"
                f" std_d={std_d:.3e})"
            )
        else:
            reason = (
                f"near_constant_inputs(std_w={std_w:.3e}, std_d={std_d:.3e})"
            )
            print(
                "[Metric] weight-coord-distance correlation: "
                f"{reason}"
            )
            return {
                "corr": float("nan"),
                "p_value": float("nan"),
                "n_edges": int(n_valid),
                "missing_endpoints": int(missing_endpoints),
                "metric": corr,
                "used_jitter": False,
                "reason": reason,
            }
    else:
        weights_corr = weights
        dists_corr = dists

    # Compute correlation
    if corr == "spearman":
        val, pval = spearmanr(weights_corr, dists_corr)
    elif corr == "pearson":
        val, pval = pearsonr(weights_corr, dists_corr)
    else:
        raise ValueError(f"Unknown corr '{corr}' (expected 'spearman' or 'pearson').")

    # Fallback if correlation is NaN and jitter is allowed
    if (val is None or np.isnan(val)) and jitter > 0.0 and not used_jitter:
        weights_corr = _maybe_jitter(weights)
        dists_corr = _maybe_jitter(dists)
        used_jitter = True
        reason = (reason + "; " if reason else "") + "nan_fallback_to_jitter"
        if corr == "spearman":
            val, pval = spearmanr(weights_corr, dists_corr)
        else:
            val, pval = pearsonr(weights_corr, dists_corr)

    return {
        "corr": float(val) if val is not None else float("nan"),
        "p_value": float(pval) if pval is not None else float("nan"),
        "n_edges": int(n_valid),
        "missing_endpoints": int(missing_endpoints),
        "metric": corr,
        "used_jitter": used_jitter,
        "reason": reason,
    }


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid used for monotone BP rescaling."""
    return 1.0 / (1.0 + np.exp(-x))


def rescale_graph_weights_for_downstream(
    G: nx.Graph,
    *,
    method: str,
    weight_key: str = "weight",
    w_min: float | None = None,
    w_cap: float | None = None,
    mode: str = "monotone",
    copy_graph: bool = True,
    # BP-specific:
    bp_beta: float = 5.0,
    bp_mode: str = "sigmoid",   # {"sigmoid","quantile"}
    bp_a: float = 2.0,
    bp_b: float = 2.0,
    # Motif-specific:
    motif_alpha: float = 0.5,
    # BH-specific:
    bh_alpha: float = 0.7,
    bh_mode: str = "rank_sigmoid",  # {"power","log","rank_sigmoid"}
    bh_beta: float = 3.0,
    eps: float = 1e-12,
) -> nx.Graph:
    """
    Return a graph with rescaled edge weights for a specific downstream method.

    This is *evaluation-only*:
    - Preserves the edge set and all non-weight attributes.
    - Applies a monotone transformation of the original weights.
    - Does NOT touch coordinates or any DuoSpec internals.
    """
    if mode != "monotone":
        raise ValueError(f"Unsupported mode '{mode}' (only 'monotone' is supported).")

    # Optionally work on a copy to avoid mutating the original graph.
    H = deepcopy(G) if copy_graph else G

    # Collect current weights
    weights = []
    edges = []
    for u, v, d in H.edges(data=True):
        w = float(d.get(weight_key, 1.0))
        weights.append(w)
        edges.append((u, v))

    if not weights:
        return H

    w_arr = np.asarray(weights, dtype=float)

    # Infer bounds if not provided
    w_min_eff = float(np.min(w_arr)) if w_min is None else float(w_min)
    w_cap_eff = float(np.max(w_arr)) if w_cap is None else float(w_cap)

    if w_cap_eff <= w_min_eff + eps:
        # Near-constant weights: nothing meaningful to rescale.
        return H

    denom = max(w_cap_eff - w_min_eff, eps)
    # Normalised position in [0,1] (linear in w)
    r_linear = np.clip((w_arr - w_min_eff) / denom, 0.0, 1.0)
    # Rank-based quantiles in [0,1] (monotone in w)
    if len(w_arr) > 1:
        ranks = rankdata(w_arr, method="average")  # 1..m
        r_rank = (ranks - 1.0) / float(len(w_arr) - 1.0)
    else:
        r_rank = np.zeros_like(w_arr)

    method_norm = method.lower()

    if method_norm in {"bp", "belief_propagation"}:
        # BP: stronger separation, monotone. Two modes:
        #  - "sigmoid": sigmoid on linearly normalised weights.
        #  - "quantile": rank-based mapping through a Beta CDF inverse.
        if bp_mode == "sigmoid":
            r2 = _sigmoid(bp_beta * (r_linear - 0.5))
        elif bp_mode == "quantile":
            # Clamp into (0,1) to avoid Beta PPF infinities.
            r_clamped = np.clip(r_rank, eps, 1.0 - eps)
            r2 = beta_dist.ppf(r_clamped, bp_a, bp_b)
        else:
            raise ValueError(f"Unknown bp_mode '{bp_mode}' (expected 'sigmoid' or 'quantile').")
        w2 = w_min_eff + (w_cap_eff - w_min_eff) * r2
    elif method_norm == "motif":
        # Power compression for heavy tails, then renormalise.
        w_raw = np.power(w_arr, motif_alpha)
        w_raw_min = float(np.min(w_raw))
        w_raw_max = float(np.max(w_raw))
        if w_raw_max <= w_raw_min + eps:
            w2 = np.clip(w_arr, w_min_eff, w_cap_eff)
        else:
            w2 = w_min_eff + (w_cap_eff - w_min_eff) * (
                (w_raw - w_raw_min) / (w_raw_max - w_raw_min + eps)
            )
    elif method_norm in {"bethe_hessian", "bh"}:
        # BH: more expressive monotone mappings.
        if bh_mode == "power":
            w_raw = np.power(w_arr, bh_alpha)
            w_raw_min = float(np.min(w_raw))
            w_raw_max = float(np.max(w_raw))
            if w_raw_max <= w_raw_min + eps:
                w2 = np.clip(w_arr, w_min_eff, w_cap_eff)
            else:
                w2 = w_min_eff + (w_cap_eff - w_min_eff) * (
                    (w_raw - w_raw_min) / (w_raw_max - w_raw_min + eps)
                )
        elif bh_mode == "log":
            w_raw = np.log1p(np.maximum(0.0, w_arr - w_min_eff))
            w_raw_min = float(np.min(w_raw))
            w_raw_max = float(np.max(w_raw))
            if w_raw_max <= w_raw_min + eps:
                w2 = np.clip(w_arr, w_min_eff, w_cap_eff)
            else:
                w2 = w_min_eff + (w_cap_eff - w_min_eff) * (
                    (w_raw - w_raw_min) / (w_raw_max - w_raw_min + eps)
                )
        elif bh_mode == "rank_sigmoid":
            r2 = _sigmoid(bh_beta * (r_rank - 0.5))
            w2 = w_min_eff + (w_cap_eff - w_min_eff) * r2
        else:
            raise ValueError(
                f"Unknown bh_mode '{bh_mode}' (expected 'power', 'log', or 'rank_sigmoid')."
            )
    else:
        raise ValueError(
            f"Unknown method '{method}'. Expected one of "
            "['bp', 'belief_propagation', 'motif', 'bethe_hessian', 'bh']."
        )

    # Monotone mapping applied; write back weights (edge set and other attrs preserved).
    for (u, v), new_w in zip(edges, w2):
        H[u][v][weight_key] = float(new_w)

    return H

def proxy_geometry_graph_correlation(
    G: nx.Graph,
    *,
    weight_key: str = "weight",
    local_score: str = "cn_over_sqrtdeg",
    corr: str = "spearman",
    sample_non_edges: bool = True,
) -> dict:
    """
    Structure-only proxy for "geometry–graph" correlation.

    Uses a locality score L(u,v) (e.g. triangle closure) as a stand-in
    for geometry; measures correlation between edge weights and L.
    """
    edges = np.asarray(list(G.edges()), dtype=object)
    m = len(edges)
    if m == 0:
        return {
            "corr_value": float("nan"),
            "p_value": float("nan"),
            "n_edges_used": 0,
            "metric": corr,
            "local_score": local_score,
        }

    # --- locality on existing edges -----------------------------------------
    scores_edges = edge_locality_scores(G, edges, metric=local_score)
    weights_edges = np.asarray(
        [float(G[u][v].get(weight_key, 1.0)) for u, v in edges],
        dtype=float,
    )

    # Edges-only correlation: do NOT mix in non-edges
    scores_all = scores_edges
    weights_all = weights_edges

    # If weights or scores are (near) constant, correlation is undefined
    if np.allclose(weights_all, weights_all[0]) or np.allclose(scores_all, scores_all[0]):
        print(
            "[GeoDe] proxy_geometry_graph_correlation: "
            "weights or locality scores are (near) constant – correlation undefined."
        )
        return {
            "corr_value": float("nan"),
            "p_value": float("nan"),
            "n_edges_used": int(len(weights_all)),
            "metric": corr,
            "local_score": local_score,
        }

    if corr == "spearman":
        val, pval = spearmanr(weights_all, scores_all)
    elif corr == "pearson":
        val, pval = pearsonr(weights_all, scores_all)
    else:
        raise ValueError(f"Unknown corr '{corr}' (expected 'spearman' or 'pearson').")

    return {
        "corr_value": float(val),
        "p_value": float(pval) if pval is not None else float("nan"),
        "n_edges_used": int(len(weights_all)),
        "metric": corr,
        "local_score": local_score,
    }


def weight_distance_correlation(
    G: nx.Graph,
    *,
    coord_key: str = "coords",
    weight_key: str = "weight",
    corr: str = "spearman",
    jitter: float = 1e-9,
    eps: float = 1e-12,
    seed: int = 0,
    debug: bool = False,
) -> dict:
    """
    Correlate edge weights with geometric distances based on node coordinates.

    For each edge (u,v):
        - w_e = G[u][v][weight_key] (defaults to 1.0)
        - d_e = ||coords[u] - coords[v]||_2

    Returns both Spearman and Pearson correlations, or NaNs if
    insufficient data or missing coordinates.
    """
    coords: Dict[Any, np.ndarray] = {}
    coords_missing = False
    for u in G.nodes():
        dat = G.nodes[u]
        if coord_key in dat:
            coords[u] = np.asarray(dat[coord_key], dtype=float)
        else:
            coords_missing = True

    dists = []
    weights = []
    for u, v, ed in G.edges(data=True):
        if u not in coords or v not in coords:
            continue
        cu = coords[u]
        cv = coords[v]
        d = float(np.linalg.norm(cu - cv))
        w = float(ed.get(weight_key, 1.0))
        dists.append(d)
        weights.append(w)

    n_valid = len(dists)
    if n_valid < 5:
        reason = f"not_enough_edges_with_coords(n={n_valid})"
        print(
            "[Metric] weight-distance correlation: "
            f"{reason}"
        )
        return {
            "spearman": float("nan"),
            "spearman_p": float("nan"),
            "pearson": float("nan"),
            "pearson_p": float("nan"),
            "n_edges": int(n_valid),
            "reason": reason,
            "used_jitter": False,
            "coords_missing": coords_missing or n_valid == 0,
        }

    dists = np.asarray(dists, dtype=float)
    weights = np.asarray(weights, dtype=float)

    std_w = float(np.std(weights))
    std_d = float(np.std(dists))

    used_jitter = False
    reason = None

    rng = np.random.default_rng(seed)

    def _maybe_jitter(x: np.ndarray) -> np.ndarray:
        return x + jitter * rng.standard_normal(len(x)) if jitter > 0.0 else x

    # Handle near-constant inputs
    if std_w < eps or std_d < eps:
        if jitter > 0.0:
            weights_corr = _maybe_jitter(weights)
            dists_corr = _maybe_jitter(dists)
            used_jitter = True
            reason = (
                f"computed_with_jitter_due_to_near_constant_inputs(std_w={std_w:.3e},"
                f" std_d={std_d:.3e})"
            )
        else:
            reason = (
                f"near_constant_inputs(std_w={std_w:.3e}, std_d={std_d:.3e})"
            )
            print(
                "[Metric] weight-distance correlation: "
                f"{reason}"
            )
            return {
                "spearman": float("nan"),
                "spearman_p": float("nan"),
                "pearson": float("nan"),
                "pearson_p": float("nan"),
                "n_edges": int(n_valid),
                "reason": reason,
                "used_jitter": False,
                "coords_missing": coords_missing,
            }
    else:
        weights_corr = weights
        dists_corr = dists

    # Always attempt both Spearman and Pearson
    sp_val, sp_p = spearmanr(weights_corr, dists_corr)

    # If Spearman is nan and we haven't tried jitter yet, fall back to jitter
    if (sp_val is None or np.isnan(sp_val)) and not used_jitter and jitter > 0.0:
        weights_corr = _maybe_jitter(weights)
        dists_corr = _maybe_jitter(dists)
        used_jitter = True
        reason = (
            (reason + "; " if reason else "")
            + "spearman_nan_fallback_to_jitter"
        )
        sp_val, sp_p = spearmanr(weights_corr, dists_corr)

    pe_val, pe_p = pearsonr(weights_corr, dists_corr)

    # Debug / sanity checks ---------------------------------------------------
    if debug:
        n_total = G.number_of_edges()
        frac_used = n_valid / n_total if n_total > 0 else float("nan")
        num_unique_w = int(len(np.unique(weights)))
        num_unique_d = int(len(np.unique(dists)))

        ranks_w = rankdata(weights_corr, method="average")
        ranks_d = rankdata(dists_corr, method="average")
        num_ties_w = int(n_valid - len(np.unique(ranks_w)))
        num_ties_d = int(n_valid - len(np.unique(ranks_d)))

        is_sorted_w_inc = bool(np.all(weights_corr[:-1] <= weights_corr[1:]))
        is_sorted_w_dec = bool(np.all(weights_corr[:-1] >= weights_corr[1:]))
        is_sorted_d_inc = bool(np.all(dists_corr[:-1] <= dists_corr[1:]))
        is_sorted_d_dec = bool(np.all(dists_corr[:-1] >= dists_corr[1:]))

        # Shuffled sanity checks
        rng_debug = np.random.default_rng(seed + 123)
        idx_shuffle = rng_debug.permutation(n_valid)
        weights_shuf = weights_corr[idx_shuffle]
        dists_shuf = dists_corr[idx_shuffle]

        sp_neg, _ = spearmanr(weights_corr, -dists_corr)
        sp_w_shuf, _ = spearmanr(weights_shuf, dists_corr)
        sp_d_shuf, _ = spearmanr(weights_corr, dists_shuf)

        # Check accidental self-correlation
        sp_self, _ = spearmanr(weights_corr, weights_corr)
        same_arrays = bool(np.allclose(weights_corr, dists_corr))

        print(
            "[Metric][Debug] weight-distance basic stats: "
            f"n_total={n_total}, n_used={n_valid}, frac_used={frac_used:.3f}, "
            f"std_w={std_w:.3e}, min_w={weights.min():.4f}, max_w={weights.max():.4f}, "
            f"unique_w={num_unique_w}, "
            f"std_d={std_d:.3e}, min_d={dists.min():.4f}, max_d={dists.max():.4f}, "
            f"unique_d={num_unique_d}"
        )
        print(
            "[Metric][Debug] ties/sorting: "
            f"num_ties_w={num_ties_w}, num_ties_d={num_ties_d}, "
            f"is_sorted_w_inc={is_sorted_w_inc}, is_sorted_w_dec={is_sorted_w_dec}, "
            f"is_sorted_d_inc={is_sorted_d_inc}, is_sorted_d_dec={is_sorted_d_dec}"
        )
        print(
            "[Metric][Debug] spearman checks: "
            f"sp(W,D)={sp_val:.4f}, sp(W,-D)={sp_neg:.4f}, "
            f"sp(W_shuf,D)={sp_w_shuf:.4f}, sp(W,D_shuf)={sp_d_shuf:.4f}, "
            f"sp(W,W)={sp_self:.4f}, D==W?={same_arrays}"
        )

        # Spot-check edges at extremes
        try:
            edges_arr = np.asarray(list(G.edges()), dtype=object)
            idx_small = np.argsort(dists)[:10]
            idx_large = np.argsort(dists)[-10:]
            idx_w_top = np.argsort(weights)[-10:]

            print("[Metric][Debug] 10 smallest-distance edges (u,v,d,w):")
            for i in idx_small:
                u, v = edges_arr[i]
                print(f"  ({u},{v}), d={dists[i]:.4f}, w={weights[i]:.4f}")

            print("[Metric][Debug] 10 largest-distance edges (u,v,d,w):")
            for i in idx_large:
                u, v = edges_arr[i]
                print(f"  ({u},{v}), d={dists[i]:.4f}, w={weights[i]:.4f}")

            print("[Metric][Debug] 10 largest-weight edges (u,v,d,w):")
            for i in idx_w_top:
                u, v = edges_arr[i]
                print(f"  ({u},{v}), d={dists[i]:.4f}, w={weights[i]:.4f}")
        except Exception as e:
            print(f"[Metric][Debug] edge spot-check skipped due to error: {e}")

        # Coords validation heuristic
        if coords:
            sample_u = next(iter(coords))
            sample_c = coords[sample_u]
            dim = sample_c.shape[0]
            print(
                f"[Metric][Debug] sample coords for node {sample_u}: "
                f"shape={sample_c.shape}, first3={sample_c[:3]}"
            )
            if dim >= 32:
                print(
                    "[Metric][Warn] coords dimension is large (>=32); they may be "
                    "learned embeddings rather than true geometric positions."
                )

        # Final structured log if things look suspicious
        if np.isnan(sp_val) or np.isnan(pe_val) or abs(sp_val) == 1.0:
            if reason is None:
                reason = "correlation_nan_or_extreme"
            print(
                "[Metric] weight-distance correlation: "
                f"n_edges={n_valid}, std_w={std_w:.3e}, std_d={std_d:.3e}, "
                f"used_jitter={used_jitter}, reason={reason}"
            )

    return {
        "spearman": float(sp_val) if sp_val is not None else float("nan"),
        "spearman_p": float(sp_p) if sp_p is not None else float("nan"),
        "pearson": float(pe_val),
        "pearson_p": float(pe_p),
        "n_edges": int(n_valid),
        "reason": reason,
        "used_jitter": used_jitter,
        "coords_missing": coords_missing,
    }


def bethe_hessian(
    H_obs           : nx.Graph,
    q               : int,
    *,
    use_nonbacktracking : bool = False,
    # NOTE: for DuoSpec denoising we do NOT want to overwrite existing weights
    # from 'dist'. This flag is kept for backwards compatibility, but when
    # True we only initialise weights on edges that lack a 'weight' attr.
    weight_from_dist    : bool = True,
    sigma_scale         : float = 1.0,
    random_state        : int   = 42,
):
    """
    Bethe–Hessian spectral embedding with soft confidence output.
    Returns
    -------
    Q      : (n,q) confidence; rows sum to 1
    hard   : argmax(Q,1)
    node2idx / idx2node
    """
    nodes      = list(H_obs.nodes())
    node2idx   = {u: i for i, u in enumerate(nodes)}
    idx2node   = {i: u for u, i in node2idx.items()}
    n          = len(nodes)

    if weight_from_dist:
        d_vals = np.array([d.get("dist", 1.0) for _, _, d in H_obs.edges(data=True)])
        sigma  = (np.median(d_vals) or 1.0) * sigma_scale
        for u, v, d in H_obs.edges(data=True):
            if "weight" in d:
                # Respect existing weights (e.g. from DuoSpec); do not overwrite.
                continue
            if "dist" in d:
                d["weight"] = np.exp(-0.5 * (d["dist"] / sigma) ** 2)
        else:
                d["weight"] = 1.0  # Default weight when dist is not available
    else:
        # Ensure a weight exists but never overwrite existing values.
        for _, _, d in H_obs.edges(data=True):
            d.setdefault("weight", 1.0)

    A   = nx.to_scipy_sparse_array(
        H_obs, nodelist=nodes, format="csr", weight="weight"
    )
    deg = np.array(A.sum(axis=1)).ravel()
    D   = diags(deg)

    if use_nonbacktracking:
        raise NotImplementedError("non-backtracking r not yet hooked in")
    r = np.sqrt(deg.mean())

    I  = diags(np.ones(n))
    Hr = (r * r - 1.0) * I - r * A + D
    k = q + 1
    ncv = 2 * min(n - 1, max(2*k + 1, k + 20))
    vals, vecs = eigsh(-Hr, k=q, which="LA", ncv = ncv)   # (n,q)

    km    = KMeans(n_clusters=q, n_init=20, random_state=random_state).fit(vecs)
    hard  = km.labels_
    mu    = km.cluster_centers_
    Q     = _conf_from_center(vecs, mu)

    return Q, hard, node2idx, idx2node

__all__ = [
    "duo_spec",
    "bethe_hessian",
    "detection_stats",
    "get_true_communities",
    "rescale_graph_weights_for_downstream",
    "compute_initial_avg_degree",
    "prune_degree_preserving_connected",
]


def compute_initial_avg_degree(G: nx.Graph) -> float:
    """Return the average degree 2m/n of the given graph."""
    n = G.number_of_nodes()
    if n == 0:
        return 0.0
    m = G.number_of_edges()
    return 2.0 * float(m) / float(n)


def prune_degree_preserving_connected(
    G_weighted: nx.Graph,
    target_avg_deg: float,
    *,
    weight_key: str = "weight",
    k_min: int = 1,
    k_max: int = 30,
    blend: float = 1.0,
    ensure_connected: bool = True,
) -> nx.Graph:
    """
    Degree-preserving pruning with connectivity-preserving binarisation.

    - Compute a global k_target from target_avg_deg, clamped to [k_min, k_max].
    - For each node, keep its top-k_target incident edges by weight.
    - Union edges chosen by either endpoint.
    - Binarise remaining edges (set weight=1.0).
    - Optionally, ensure each original connected component remains connected
      by adding back high-weight edges from G_weighted within that component.
    """
    # Copy nodes (and their attributes) first.
    H = nx.Graph()
    for u, data in G_weighted.nodes(data=True):
        H.add_node(u, **data)

    n = G_weighted.number_of_nodes()
    if n == 0:
        return H

    # Determine global k_target.
    k_target = int(round(float(blend) * float(target_avg_deg)))
    if k_min is not None:
        k_target = max(k_target, int(k_min))
    if k_max is not None:
        k_target = min(k_target, int(k_max))
    if k_target <= 0:
        # Degenerate case: keep no edges.
        return H

    # Collect candidate edges to keep: union over per-node top-k.
    kept_edges = set()
    for u in G_weighted.nodes():
        inc = []
        for v, d in G_weighted[u].items():
            w = float(d.get(weight_key, 1.0))
            a = u if u <= v else v
            b = v if u <= v else u
            inc.append((w, a, b))
        if not inc:
            continue
        # Sort descending by weight, tie-break by (a,b).
        inc.sort(key=lambda t: (-t[0], t[1], t[2]))
        for _, a, b in inc[:k_target]:
            kept_edges.add((a, b))

    # Add kept edges with binary weight.
    for a, b in kept_edges:
        if G_weighted.has_edge(a, b):
            data = dict(G_weighted[a][b])
            data[weight_key] = 1.0
            H.add_edge(a, b, **data)

    if not ensure_connected or G_weighted.number_of_edges() == 0:
        return H

    # For each original connected component, ensure H is at least as connected
    # as G_weighted, by adding back high-weight edges within that component.
    for comp_nodes in nx.connected_components(G_weighted):
        comp_nodes = list(comp_nodes)
        if len(comp_nodes) <= 1:
            continue

        # Restrict to this component.
        H_sub = H.subgraph(comp_nodes).copy()
        G_sub = G_weighted.subgraph(comp_nodes)

        # Union-find over nodes in this component based on H_sub edges.
        nodes_list = list(comp_nodes)
        idx_of = {u: i for i, u in enumerate(nodes_list)}
        parent = np.arange(len(nodes_list), dtype=int)
        size = np.ones(len(nodes_list), dtype=int)

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int) -> bool:
            ri, rj = find(i), find(j)
            if ri == rj:
                return False
            if size[ri] < size[rj]:
                ri, rj = rj, ri
            parent[rj] = ri
            size[ri] += size[rj]
            return True

        # Initialise DSU with existing edges in H_sub.
        for u, v in H_sub.edges():
            iu = idx_of[u]
            iv = idx_of[v]
            union(iu, iv)

        # Helper to count current number of components in this DSU.
        def num_sets() -> int:
            roots = {find(i) for i in range(len(nodes_list))}
            return len(roots)

        if num_sets() <= 1:
            # Already connected within this component.
            continue

        # Candidate edges from G_sub, sorted by descending weight, deterministic ties.
        cand_edges = []
        for u, v, d in G_sub.edges(data=True):
            w = float(d.get(weight_key, 1.0))
            a = u if u <= v else v
            b = v if u <= v else u
            cand_edges.append((w, a, b))
        if not cand_edges:
            continue

        cand_edges.sort(key=lambda t: (-t[0], t[1], t[2]))

        for _, a, b in cand_edges:
            iu = idx_of[a]
            iv = idx_of[b]
            if find(iu) == find(iv):
                continue
            # Add edge back with binary weight.
            data = dict(G_weighted[a][b])
            data[weight_key] = 1.0
            H.add_edge(a, b, **data)
            union(iu, iv)
            if num_sets() <= 1:
                break

    return H

def duo_spec(
    H_obs: nx.Graph,
    K: int,
    *,
    # EM controls
    max_em_iters: int = 20,
    min_em_iters: int = 2,
    # Weight bounds (slightly wider by default for downstream BH/Motif/BP)
    w_min: float = 0.05,
    w_cap: float = 3.0,
    # New strict convergence controls
    conv_tol: float = 1e-8,
    conv_window: int = 3,
    # Global scale on update strengths
    update_scale: float = 0.8,
    metric_debug: bool = False,
    # Edge-denoising strengths (geometry shrink + optional community boost)
    lam_geo: float = 0.22,
    lam_comm_boost: float = 0.05,
    # Geometry / community DSU controls
    S0: int = 20,
    frac_sweep: Tuple[float, ...] = (0.995, 0.99, 0.98, 0.97, 0.95),
    local_score: str = "cn_over_sqrtdeg",
    # Community gate & stability controls
    geo_gate_enabled: bool = True,
    gate_power: float = 1.5,
    gate_floor: float = 0.02,
    stable_k: int = 3,
    delta_cap: float = 0.25,
    # Community-boost controls (second channel)
    use_comm_boost: bool = True,
):
    """Purely structural EM denoiser (no spectral methods; DSU-based geometry & community proxies)."""
    print(
        "[EM] running EM iterations "
        f"(max={max_em_iters}, min={min_em_iters}, tol={conv_tol:.1e}, window={conv_window})"
    )

    subG = deepcopy(H_obs)
    for _, _, d in subG.edges(data=True):
        d.setdefault("weight", 1.0)

    node2idx = {u: i for i, u in enumerate(subG.nodes())}

    base_strength = np.zeros(len(node2idx), dtype=float)
    for u, v, d in subG.edges(data=True):
        w0 = float(d.get("weight", 1.0))
        iu = node2idx[u]
        iv = node2idx[v]
        base_strength[iu] += w0
        base_strength[iv] += w0

    best, hist, no_imp = {"obj": -np.inf}, [], 0
    m0 = subG.number_of_edges()

    geom_corr_before = proxy_weight_locality_correlation(
        subG,
        weight_key="weight",
        local_score=local_score,
        corr="spearman",
    )

    coord_corr_before = weight_coord_distance_correlation(
        subG,
        coord_key="coords",
        weight_key="weight",
        corr="spearman",
    )

    balls = np.zeros(len(node2idx), dtype=int)

    lam_geo_curr = lam_geo

    for em in range(1, max_em_iters + 1):
        print(f"[EM] iter {em} / {max_em_iters}")

        edges = np.asarray(list(subG.edges()), dtype=object)

        t_geo_start = time.perf_counter()
        if len(edges) == 0:
            scores_local = np.zeros(0, dtype=float)
            r_geo = np.zeros(0, dtype=float)
            bridge_score = None
            blob_comp_per_node = np.zeros(len(node2idx), dtype=int)
            conf_min = conf_med = conf_max = float("nan")
        else:
            scores_local = edge_locality_scores(subG, edges, metric=local_score)

            geo_info = geometry_scores_fineblob_persistence(
                subG,
                edges,
                scores_local,
                node2idx,
                frac_sweep=frac_sweep,
                fine_frac=frac_sweep[0] if frac_sweep else 0.99,
                S0=S0,
                stable_k=stable_k,
                debug=metric_debug,
            )
            r_geo = geo_info["r_geo"]
            bridge_score = geo_info["bridge_score"]
            blob_id_per_node = geo_info["blob_id_per_node"]
            # Compact blob ids to 0..n_blobs-1 for downstream community proxy.
            _, blob_comp_per_node = np.unique(blob_id_per_node, return_inverse=True)
            balls = blob_comp_per_node.copy()

            if r_geo.size == 0:
                conf_min = conf_med = conf_max = float("nan")
            else:
                conf_min = float(r_geo.min())
                conf_med = float(np.mean(r_geo))
                conf_max = float(r_geo.max())
        t_geo_end = time.perf_counter()

        t_comm_start = time.perf_counter()
        if len(edges) == 0:
            p_same = np.zeros(0, dtype=float)
            n_blobs = 0
            n_coarse = 0
        else:
            blob_edges, blob_weights, blob_comp_per_node = build_blob_supergraph(
                subG,
                edges,
                node2idx,
                blob_comp_per_node,
                r_geo,
                weight_key="weight",
            )
            n_blobs = int(len(np.unique(blob_comp_per_node))) if blob_comp_per_node.size else 0
            if blob_edges.size == 0 or n_blobs <= 1:
                coarse_ids = np.zeros(n_blobs, dtype=int)
                n_coarse = int(n_blobs)
            else:
                coarse_ids = community_proxy_persistence_on_blob_graph(
                    blob_edges,
                    blob_weights,
                    num_blobs=n_blobs,
                    K=K,
                    frac_sweep=frac_sweep,
                    stable_k=stable_k,
                )
                n_coarse = int(len(np.unique(coarse_ids)))

            if blob_comp_per_node.size:
                coarse_id_per_node = coarse_ids[blob_comp_per_node]
            else:
                coarse_id_per_node = np.zeros(len(node2idx), dtype=int)

            p_same = np.zeros(len(edges), dtype=float)
            for idx, (u, v) in enumerate(edges):
                ui = node2idx[u]
                vi = node2idx[v]
                p_same[idx] = 1.0 if coarse_id_per_node[ui] == coarse_id_per_node[vi] else 0.0
        t_comm_end = time.perf_counter()

        t_rw_start = time.perf_counter()

        geo_stats = reweight_edges_from_posteriors(
            subG,
            edges,
            r_geo,
            p_same,
            lam_geo=lam_geo_curr * update_scale,
            lam_comm_boost=lam_comm_boost * update_scale,
            w_min=w_min,
            w_cap=w_cap,
            delta_cap=delta_cap,
            bridge_score=bridge_score,
            geo_gate_enabled=geo_gate_enabled,
            gate_power=gate_power,
            gate_floor=gate_floor,
            use_comm_boost=use_comm_boost,
        )
        t_rw_end = time.perf_counter()


        print(
            "[EM] iter "
            f"{em}/{max_em_iters}: "
            f"geo_mean|Δw|={geo_stats['mean_abs_delta']:.4e}, "
            f"frac_min={geo_stats['frac_w_min']:.3f}, "
            f"frac_cap={geo_stats['frac_w_cap']:.3f}, "
            f"n_blobs={n_blobs}, "
            f"n_coarse={n_coarse}"
        )
        if metric_debug:
            print(
                "[EM][Timing] "
                f"community={t_comm_end - t_comm_start:.3f}s, "
                f"geometry={t_geo_end - t_geo_start:.3f}s, "
                f"reweight={t_rw_end - t_rw_start:.3f}s"
            )


        obj = -float(geo_stats["mean_abs_delta"])
        hist.append(
            dict(
                it=em,
                obj=obj,
                edges=subG.number_of_edges(),
                mean_abs_delta=geo_stats["mean_abs_delta"],
                frac_w_min=geo_stats["frac_w_min"],
                frac_w_cap=geo_stats["frac_w_cap"],
                mean_delta_geo=geo_stats.get("mean_delta_geo", 0.0),
                mean_delta_comm=geo_stats.get("mean_delta_comm", 0.0),
                frac_boosted=geo_stats.get("frac_boosted", 0.0),
                frac_shrunk=geo_stats.get("frac_shrunk", 0.0),
                n_blobs=n_blobs,
                n_coarse=n_coarse,
            )
        )

        if obj > best["obj"]:
            best.update(obj=obj)
            no_imp = 0
        else:
            no_imp += 1

        if em >= min_em_iters and len(hist) >= conv_window + 1:
            deltas = [
                abs(hist[-i]["obj"] - hist[-i - 1]["obj"])
                for i in range(1, conv_window + 1)
            ]
            max_delta = max(deltas)
            if max_delta <= conv_tol:
                print(
                    "[EM] strict convergence reached at iter "
                    f"{em} (max |Δobj| over last {conv_window} iters = "
                    f"{max_delta:.3e})"
                )
                break

    scores_local = edge_locality_scores(subG, edges, metric=local_score) if len(edges) > 0 else np.zeros(0, dtype=float)
    if len(edges) > 0:
        geo_info_final = geometry_scores_fineblob_persistence(
            subG,
            edges,
            scores_local,
            node2idx,
            frac_sweep=frac_sweep,
            fine_frac=frac_sweep[0] if frac_sweep else 0.99,
            S0=S0,
            stable_k=stable_k,
            debug=metric_debug,
        )
        r_geo_final = geo_info_final["r_geo"]
        blob_id_final = geo_info_final["blob_id_per_node"]
        _, blob_comp_final = np.unique(blob_id_final, return_inverse=True)
        blob_edges_final, blob_weights_final, blob_comp_final = build_blob_supergraph(
            subG,
            edges,
            node2idx,
            blob_comp_final,
            r_geo_final,
            weight_key="weight",
        )
        n_blobs_final = int(len(np.unique(blob_comp_final))) if blob_comp_final.size else 0
        if blob_edges_final.size == 0 or n_blobs_final <= 1:
            coarse_ids_final = np.zeros(n_blobs_final, dtype=int)
        else:
            coarse_ids_final = community_proxy_persistence_on_blob_graph(
                blob_edges_final,
                blob_weights_final,
                num_blobs=n_blobs_final,
                K=K,
                frac_sweep=frac_sweep,
                stable_k=stable_k,
            )
        if blob_comp_final.size:
            coarse_id_per_node_final = coarse_ids_final[blob_comp_final]
        else:
            coarse_id_per_node_final = np.zeros(len(node2idx), dtype=int)
    else:
        r_geo_final = np.zeros(0, dtype=float)
        coarse_id_per_node_final = np.zeros(len(node2idx), dtype=int)
        blob_comp_final = np.zeros(len(node2idx), dtype=int)

    geom_corr_after = proxy_weight_locality_correlation(
        subG,
        weight_key="weight",
        local_score=local_score,
        corr="spearman",
    )
    before_val = geom_corr_before.get("spearman", float("nan"))
    after_val = geom_corr_after.get("spearman", float("nan"))

    if not np.isnan(before_val) and not np.isnan(after_val):
        geom_corr_delta = after_val - before_val
        print(
            "[Metric] weight-locality spearman "
            f"before={before_val:.4f}, "
            f"after={after_val:.4f}, "
            f"delta={geom_corr_delta:.4f}, "
            f"n={geom_corr_after.get('n_edges', 0)}, "
            f"reason_after={geom_corr_after.get('reason')}"
        )
    else:
        geom_corr_delta = float("nan")
        if np.isnan(before_val) and not np.isnan(after_val):
            print(
                "[Metric] weight-locality spearman: "
                f"before is NaN, after={after_val:.4f}; "
                "delta undefined (before NaN)."
            )
        elif np.isnan(before_val) and np.isnan(after_val):
            print(
                "[Metric] weight-locality spearman: "
                "metric undefined both before and after."
            )
        else:
            print(
                "[Metric] weight-locality spearman: "
                f"before={before_val:.4f}, after is NaN; delta undefined."
            )

    coord_corr_after = weight_coord_distance_correlation(
        subG,
        coord_key="coords",
        weight_key="weight",
        corr="spearman",
    )
    c_before = coord_corr_before.get("corr", float("nan"))
    c_after = coord_corr_after.get("corr", float("nan"))
    if not np.isnan(c_before) and not np.isnan(c_after):
        coord_corr_delta = c_after - c_before
        print(
            "[Metric] weight-distance spearman "
            f"before={c_before:.4f}, "
            f"after={c_after:.4f}, "
            f"delta={coord_corr_delta:.4f}, "
            f"n={coord_corr_after.get('n_edges', 0)}, "
            f"reason_after={coord_corr_after.get('reason')}, "
            f"used_jitter_after={coord_corr_after.get('used_jitter')}"
        )
    else:
        coord_corr_delta = float("nan")
        print(
            "[Metric] weight-distance spearman: "
            f"before={c_before:.4f}, after={c_after:.4f}; delta undefined."
        )
    result = dict(
        beliefs=None,
        communities=coarse_id_per_node_final,
        balls=blob_comp_final,
        node2idx=node2idx,
        idx2node={i: u for u, i in node2idx.items()},
        history=hist,
        G_final=subG,
        geom_corr_before=geom_corr_before,
        geom_corr_after=geom_corr_after,
        geom_corr_delta=geom_corr_delta,
        coord_corr_before=coord_corr_before,
        coord_corr_after=coord_corr_after,
        coord_corr_delta=coord_corr_delta,
    )
    proxy_before = dict(
        corr_value=float(geom_corr_before.get("spearman", float("nan"))),
        p_value=float(geom_corr_before.get("p_value", float("nan"))),
        n_edges_used=int(geom_corr_before.get("n_edges", 0)),
        metric="spearman",
        local_score=local_score,
    )
    proxy_after = dict(
        corr_value=float(geom_corr_after.get("spearman", float("nan"))),
        p_value=float(geom_corr_after.get("p_value", float("nan"))),
        n_edges_used=int(geom_corr_after.get("n_edges", 0)),
        metric="spearman",
        local_score=local_score,
    )
    result["proxy_corr_before"] = proxy_before   # backward-compat alias; will remove later
    result["proxy_corr_after"] = proxy_after     # backward-compat alias; will remove later
    result["proxy_corr_delta"] = geom_corr_delta # backward-compat alias; will remove later

    return result



def detection_stats(preds: np.ndarray, true: np.ndarray, *, n_perm: int = 10_000):
    """Compute accuracy, per‑community stats, and permutation‑test p‑value."""
    k = int(max(preds.max(), true.max()) + 1)
    C = confusion_matrix(true, preds, labels=np.arange(k))
    r, c = linear_sum_assignment(-C)
    perm = np.arange(k); perm[c] = r
    aligned = perm[preds]

    stats = {
        "accuracy": accuracy_score(true, aligned),
        "num vertices": len(true),
        "num communities predicted": len(np.unique(aligned)),
    }
    for t in range(k):
        mask = true == t
        stats[f"accuracy_{t}"] = accuracy_score(true[mask], aligned[mask])

    res = permutation_test(
        (true, aligned),
        statistic=lambda x, y: accuracy_score(x, y),
        vectorized=False,
        n_resamples=n_perm,
        alternative="greater",
        random_state=0,
    )
    stats["perm_p"] = float(res.pvalue)
    return stats

def get_true_communities(G: nx.Graph, *, node2idx: Dict[int,int] | None = None, attr: str = "block") -> np.ndarray:
    if node2idx is None:
        return np.array([G.nodes[u][attr] for u in G])
    arr = np.empty(len(node2idx), int)
    for u,i in node2idx.items():
        arr[i] = G.nodes[u][attr]
    return arr


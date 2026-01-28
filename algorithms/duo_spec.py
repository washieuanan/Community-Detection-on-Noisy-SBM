from __future__ import annotations
from collections import defaultdict
from copy import deepcopy
import math
import random
from typing import Any, Dict, Iterable, Tuple, Union

# NumPy and SciPy
import numpy as np
from numpy.linalg import norm
from numpy.random import default_rng
import scipy.sparse as sp
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix, csr_matrix, diags, identity
from scipy.sparse import linalg as splinalg
import scipy.sparse.linalg as sla
from scipy.sparse.linalg import eigs, eigsh, LinearOperator, lobpcg
from scipy.sparse.csgraph import laplacian as cs_lap, shortest_path
from scipy.stats import mode, permutation_test, spearmanr, pearsonr, rankdata

# NetworkX
import networkx as nx

# Scikit-learn
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KernelDensity, NearestNeighbors

from algorithms.spectral_ops.attention import byoe_embedding, motif_spectral_embedding


# censoring schemes

def erdos_renyi_mask(
    G: nx.Graph,
    rho: float,
    *,
    seed: int | None = None,
    copy_node_attrs: bool = True,
) -> nx.Graph:
    """
    Return a *censored* graph in which each edge of ``G`` is kept
    independently with probability ``rho`` and deleted otherwise.

    Parameters
    ----------
    G : networkx.Graph
        The original (latent) graph.
    rho : float in (0,1]
        Retention probability P(edge is observed).
    seed : int or None
        Random-state seed for reproducibility.
    copy_node_attrs : bool
        If True, copy node attributes to the censored graph.

    Returns
    -------
    H : networkx.Graph
        Graph with the same node set as ``G`` but with
        each edge kept w.p. ``rho``.
    """
    if not (0.0 <= rho <= 1.0):
        raise ValueError("rho must be in [0,1]")

    rng = default_rng(seed)
    H = nx.Graph()
    if copy_node_attrs:
        for u, attrs in G.nodes(data=True):
            H.add_node(u, **deepcopy(attrs))
    else:
        H.add_nodes_from(G.nodes())

    for u, v, attrs in G.edges(data=True):
        if rng.random() < rho:
            H.add_edge(u, v, **deepcopy(attrs))

    return H

def geometric_censor(
    G: nx.Graph,
    r: float,
    p: float = 0.75,
    *,
    coord_key: str = "coords",
    metric: str = "euclidean",
    copy_node_attrs: bool = True,
    seed: int = 42
) -> nx.Graph:
    """
    Keep only those edges whose *geometric* distance between the
    incident vertices is ≤ r.

    Each node is expected to have a coordinate attribute (default
    name ``"coords"``) that is an iterable of floats, e.g. a 2- or
    3-dimensional position.

    Parameters
    ----------
    G : networkx.Graph
        Original graph (must have node attribute ``coord_key``).
    r : float
        Retention distance threshold (Euclidean by default).
    coord_key : str
        Node-attribute name that contains coordinates.
    metric : {"euclidean"}  (placeholder for future metrics)
    copy_node_attrs : bool
        If True, node attributes are copied into the censored graph.

    Returns
    -------
    H : networkx.Graph
        Graph with exactly those edges (u,v) whose coordinate distance
        ≤ r.  All vertices of ``G`` are preserved.
    """
    if r < 0:
        raise ValueError("distance threshold r must be non-negative")
    random.seed(seed)
    if metric != "euclidean":
        raise NotImplementedError("Only Euclidean metric supported")

    # --- helper to compute Euclidean distance quickly ---------------
    def _dist(a: Iterable[float], b: Iterable[float]) -> float:
        diff = np.fromiter(a, float) - np.fromiter(b, float)
        return float(np.sqrt(np.dot(diff, diff)))

    # --- create new graph with same nodes ---------------------------
    H = nx.Graph()
    if copy_node_attrs:
        for u, attrs in G.nodes(data=True):
            H.add_node(u, **deepcopy(attrs))
    else:
        H.add_nodes_from(G.nodes())

    # --- iterate over edges & keep those within r --------------------
    for u, v, attrs in G.edges(data=True):
        try:
            cu = G.nodes[u][coord_key]
            cv = G.nodes[v][coord_key]
        except KeyError as exc:
            raise KeyError(
                f"Node missing '{coord_key}' attribute needed for "
                "geometric censoring"
            ) from exc

        if _dist(cu, cv) <= r:
            H.add_edge(u, v, dist = _dist(cu, cv))
        else:
            if random.random() < p:
                H.add_edge(u, v, dist = _dist(cu, cv))

    return H


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

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _edge_same_prob(bel: np.ndarray, iu, iv) -> np.ndarray:
    """P[u,v same label] for many edges at once (einsum)."""
    return np.einsum("ij,ij->i", bel[iu], bel[iv])


def weighted_percentile(x, q, w=None):
    """
    Percentile that respects optional weights.

    Parameters
    ----------
    x : 1-D data
    q : percentile 0–100
    w : same length weights (defaults to 1)

    Returns
    -------
    float – value 'v' s.t.  q percent of weighted mass lies below v.
    """
    x = np.asarray(x, float)
    if w is None:
        w = np.ones_like(x)
    else:
        w = np.asarray(w, float)

    idx = np.argsort(x)
    x, w = x[idx], w[idx]
    cdf = np.cumsum(w)
    cdf /= cdf[-1]
    return np.interp(q / 100.0, cdf, x)
  

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


# DEPRECATED/UNUSED (kept for now): distance-based geometry–graph correlation.
# New code uses `proxy_geometry_graph_correlation`, which only depends on
# structure-induced locality scores (no coordinates or explicit distances).
def geometry_graph_correlation(
    G: nx.Graph,
    *,
    dist_key: str = "dist",
    weight_key: str = "weight",
    metric: str = "spearman",
    sample_non_edges: bool = False,
    seed: int = 0,
) -> dict:
    """
    [DEPRECATED] Correlation between geometric similarity and edge weights.

    Retained for backwards compatibility for experiments that still have
    explicit distances, but not used in the new GeoDe denoiser.
    """
    dists = []
    weights = []
    for _, _, data in G.edges(data=True):
        if dist_key in data:
            dists.append(float(data[dist_key]))
            weights.append(float(data.get(weight_key, 1.0)))
    if not dists:
        return {
            "corr_value": float("nan"),
            "p_value": float("nan"),
            "n_edges_used": 0,
            "metric": metric,
        }
    dists = np.asarray(dists, float)
    weights = np.asarray(weights, float)
    sim = 1.0 / (1.0 + dists)
    if metric == "spearman":
        corr, pval = spearmanr(weights, sim)
    else:
        corr, pval = pearsonr(weights, sim)
    return {
        "corr_value": float(corr),
        "p_value": float(pval) if pval is not None else float("nan"),
        "n_edges_used": int(len(weights)),
        "metric": metric,
    }


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

    scores_all = scores_edges
    weights_all = weights_edges

    # --- optionally augment with sampled non-edges --------------------------
    if sample_non_edges:
        nodes = list(G.nodes())
        n = len(nodes)
        if n >= 2:
            # Total possible undirected edges (no self-loops)
            total_pairs = n * (n - 1) // 2
            num_non_edges_total = max(0, total_pairs - m)

            if num_non_edges_total > 0:
                # Sample up to m non-edges
                target = min(m, num_non_edges_total)
                rng = default_rng()
                adj = G.adj
                # Degrees reused for locality; pre-compute once
                deg = {u: len(adj[u]) for u in nodes}

                non_edges = []
                attempts = 0
                max_attempts = 10 * target

                while len(non_edges) < target and attempts < max_attempts:
                    i = rng.integers(0, n)
                    j = rng.integers(0, n - 1)
                    if j >= i:
                        j += 1
                    u, v = nodes[i], nodes[j]
                    # undirected graph: skip if edge already exists
                    if v in adj[u]:
                        attempts += 1
                        continue
                    non_edges.append((u, v))
                    attempts += 1

                if non_edges:
                    non_edges_arr = np.asarray(non_edges, dtype=object)
                    # Compute locality for non-edges using same metric
                    scores_ne = np.empty(len(non_edges), dtype=float)
                    eps = 1e-12
                    for idx, (u, v) in enumerate(non_edges):
                        if u == v:
                            scores_ne[idx] = 0.0
                            continue
                        adj_u = adj[u]
                        adj_v = adj[v]
                        if len(adj_u) > len(adj_v):
                            adj_u, adj_v = adj_v, adj_u
                            du, dv = deg[v], deg[u]
                        else:
                            du, dv = deg[u], deg[v]
                        cn = 0
                        for w in adj_u:
                            if w in adj_v:
                                cn += 1
                        if local_score == "common_neighbors":
                            s = float(cn)
                        elif local_score == "jaccard":
                            denom = (du + dv - cn) + eps
                            s = float(cn) / denom if denom > 0 else 0.0
                        else:  # "cn_over_sqrtdeg"
                            denom = math.sqrt(du * dv + eps)
                            s = float(cn) / denom if denom > 0 else 0.0
                        scores_ne[idx] = s

                    scores_all = np.concatenate([scores_edges, scores_ne])
                    weights_all = np.concatenate(
                        [weights_edges, np.zeros(len(non_edges), dtype=float)]
                    )

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
            if "dist" in d:
                d["weight"] = np.exp(-0.5 * (d["dist"] / sigma) ** 2)
            else:
                d["weight"] = 1.0  # Default weight when dist is not available

    A   = nx.to_scipy_sparse_array(H_obs, nodelist=nodes,
                                   format="csr", weight="weight")
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


def dwpe(
    H_obs                   : nx.Graph,
    q                       : int,
    *,
    L                       : int   = 3,      # max walk length
    alpha                   : float = 0.6,    # geometric decay for longer walks
    weight_from_dist        : bool  = True,   # optional Gaussian edge re-weight
    sigma_scale             : float = 1.0,    # bandwidth multiplier
    random_state            : int   = 42,
):
    """
    Distance-Weighted Path-Expansion (DWPE) spectral embedding.

    Parameters
    ----------
    H_obs : nx.Graph
        Observed (possibly weighted) sub-graph that DuoSpec provides each iteration.
        If edges carry attribute 'dist', a Gaussian kernel is applied.
    q : int
        Expected number of communities.
    L : int, optional
        Maximum walk length used in the expansion.  L=2 or 3 is usually plenty.
    alpha : float, optional
        Geometric decay factor (0<alpha<1) penalising longer walks.
    weight_from_dist : bool, optional
        If True and edge attribute 'dist' exists, converts distances to weights.
    sigma_scale : float, optional
        Multiplier on the median distance to set the Gaussian bandwidth σ.
    random_state : int, optional
        KMeans reproducibility.

    Returns
    -------
    Q    : (n,q) soft assignments (rows sum to 1)
    hard : np.ndarray, shape (n,)
        Hard labels = argmax(Q,1)
    node2idx / idx2node : mapping <-> index
    """

    nodes      = list(H_obs.nodes())
    node2idx   = {u: i for i, u in enumerate(nodes)}
    idx2node   = {i: u for u, i in node2idx.items()}
    n          = len(nodes)

    if weight_from_dist:
        d_vals = np.array([d.get("dist", 1.0) for _, _, d in H_obs.edges(data=True)])
        sigma  = (np.median(d_vals) or 1.0) * sigma_scale
        for u, v, d in H_obs.edges(data=True):
            if "dist" in d:
                d["weight"] = np.exp(-0.5 * (d["dist"] / sigma) ** 2)
            else:
                d["weight"] = 1.0
    else:
        for _, _, d in H_obs.edges(data=True):
            d["weight"] = d.get("weight", 1.0)

    A = nx.to_scipy_sparse_array(
        H_obs, nodelist=nodes, format="csr", weight="weight", dtype=float
    )

    B = A.copy()               
    A_power = A.copy()

    for ℓ in range(2, L + 1):
        A_power = A_power @ A  
        B += (alpha ** (ℓ - 1)) * A_power 

    deg = np.array(B.sum(axis=1)).ravel()
    D_inv_sqrt = diags(np.power(deg, -0.5, where=deg > 0))
    S = D_inv_sqrt @ B @ D_inv_sqrt      

    k = q                        
    ncv = 2 * min(n - 1, max(2*k + 1, k + 20))
    vals, vecs = eigsh(S, k=k, which="LA", ncv=ncv)

    km    = KMeans(n_clusters=q, n_init=20, random_state=random_state).fit(vecs)
    hard  = km.labels_
    mu    = km.cluster_centers_

    Q = _conf_from_center(vecs, mu)

    return Q, hard, node2idx, idx2node

def bethe_hessian_fast(
    H_obs                : nx.Graph,
    q                    : int,
    *,
    weight_from_dist     : bool    = True,
    sigma_scale          : float   = 1.0,
    use_lobpcg           : bool    = True,
    tol                  : float   = 1e-3,
    maxiter              : int     = 200,
    random_state         : int     = 42,
    prev_evecs           : np.ndarray = None,
):
    """
    Fast Bethe–Hessian embedding via LOBPCG or warm-started ARPACK.
    """
    nodes    = list(H_obs.nodes())
    idx      = {u:i for i,u in enumerate(nodes)}
    n        = len(nodes)

    if weight_from_dist:
        d_vals = np.array([d.get("dist",1.0) for *_,d in H_obs.edges(data=True)])
        sigma  = max(np.median(d_vals), 1.0) * sigma_scale
        for u,v,d in H_obs.edges(data=True):
            d["weight"] = np.exp(-0.5*(d.get("dist",1.0)/sigma)**2)

    A   = nx.to_scipy_sparse_array(H_obs, nodelist=nodes,
                                   weight="weight", format="csr")
    deg = np.ravel(A.sum(axis=1))
    D   = diags(deg)

    r = np.sqrt(deg.mean())

    I  = diags(np.ones(n))
    Hr = (r*r - 1.0)*I - r*A + D

    if use_lobpcg:
        X0 = (prev_evecs 
              if (prev_evecs is not None and prev_evecs.shape==(n,q))
              else np.random.RandomState(random_state).randn(n,q))
        vals, vecs = lobpcg(Hr, X0, tol=tol, maxiter=maxiter)
    else:

        eig_kwargs = dict(which="SM", tol=tol, maxiter=maxiter)
        if prev_evecs is not None:
            eig_kwargs["v0"] = prev_evecs[:,0]
        vals, vecs = eigsh(Hr, k=q, **eig_kwargs)

    km   = KMeans(n_clusters=q, n_init=10, random_state=random_state).fit(vecs)
    hard = km.labels_
    mu   = km.cluster_centers_
    diff = vecs[:,None,:] - mu[None,:,:]       # shape (n,q,q)
    Q    = np.exp(-np.sum(diff**2, axis=2))
    Q   /= Q.sum(axis=1, keepdims=True)

    return Q, hard, idx, {i:u for u,i in idx.items()}, vecs


def laplacian(
    H_obs        : nx.Graph,
    q            : int,
    *,
    random_state : int = 42,
):
    """
    Spectral clustering (norm-Laplacian) with 1/(1+dist) confidence,
    using ARPACK for the bottom q+1 eigenpairs of L_norm.
    """
    nodes    = list(H_obs.nodes())
    node2idx = {u: i for i, u in enumerate(nodes)}
    idx2node = {i: u for u, i in node2idx.items()}
    n        = len(nodes)

    A = nx.to_scipy_sparse_array(H_obs, nodelist=nodes, format="csr")
    L = cs_lap(A, normed=True)

    m   = q + 1
    ncv = min(n - 1, max(2 * (m + 20), 10 * m))

    I = sp.eye(n, format="csr")
    M = I - L
    evals_M, evecs = eigsh(
        M,
        k=m,
        which="LA",     # largest algebraic
        ncv=ncv,
        tol=1e-4,
        maxiter=10000
    )
    evals = 1 - evals_M
    X = evecs[:, 1 : m]

    km   = KMeans(n_clusters=q, n_init=20, random_state=random_state).fit(X)
    hard = km.labels_
    mu   = km.cluster_centers_
    Q    = _conf_from_center(X, mu)

    return Q, hard, node2idx, idx2node


def regularized_laplacian(
    H_obs        : nx.Graph,
    q            : int,
    *,
    random_state : int = 42,
):
    """
    Spectral clustering on the regularized Laplacian
      L_reg = (D + τI)^(-1/2) A (D + τI)^(-1/2),
    using ARPACK to get the top-q eigenvectors.
    """
    nodes    = list(H_obs.nodes())
    node2idx = {u: i for i, u in enumerate(nodes)}
    idx2node = {i: u for u, i in node2idx.items()}
    n        = len(nodes)

    A   = nx.to_scipy_sparse_array(H_obs, nodelist=nodes, format="csr")
    deg = np.array(A.sum(axis=1)).ravel()

    tau     = 1.0
    D_inv_s = sp.diags(1.0 / np.sqrt(deg + tau))

    L_reg = D_inv_s @ A @ D_inv_s

    m   = q
    ncv = min(n - 1, max(m + 20, 5 * m))

    evals, evecs = eigsh(
        L_reg,
        k=m,
        which="LA",
        ncv=ncv,
        tol=1e-4,
        maxiter=10000
    )

    X = evecs

    km   = KMeans(n_clusters=q, n_init=20, random_state=random_state).fit(X)
    hard = km.labels_
    mu   = km.cluster_centers_
    Q    = _conf_from_center(X, mu)

    return Q, hard, node2idx, idx2node


def get_callable(calls: Union[Tuple, str]):
    func_dict = {
        "bethe_hessian":     bethe_hessian,
        "laplacian":         laplacian,
        "regularized_laplacian": regularized_laplacian,
        "bethe_hessian_fast": bethe_hessian_fast,
        'byoe_embedding': byoe_embedding,
        'motif': motif_spectral_embedding, 
        "dwpe":              dwpe,
    }
    if isinstance(calls, str):
        return (func_dict[calls], func_dict[calls])
    elif isinstance(calls, tuple):
        if len(calls) == 2:
            return (func_dict[calls[0]], func_dict[calls[1]])
        elif len(calls) == 1:
            return (func_dict[calls[0]], func_dict[calls[0]])
        else:
            raise ValueError("Invalid callable input")

def _to_idx(edge_arr, n2i):
    iu = np.fromiter((n2i[u] for u, _ in edge_arr), int, len(edge_arr))
    iv = np.fromiter((n2i[v] for _, v in edge_arr), int, len(edge_arr))
    return iu, iv

def _scale_edges(G, mask, conf, lam, w_min, w_cap, mode="shrink"):
    """Scale selected edges up/down.

    Parameters
    ----------
    mask : bool array over `edges`
    conf : confidence values (same shape)
    lam  : scalar 0–1 shrink / boost magnitude
    mode : "shrink" | "boost"
    """
    cnt = 0
    for (flag, (u, v), c) in zip(mask, G.edges(), conf):
        if not flag:
            continue
        w = G[u][v]["weight"]
        if mode == "shrink":
            lam_i = lam * (1 - c)             # if c≈1 (very confident), lam_i≈0 
            new_w = max(w_min, w * (1 - lam_i))
        else:                                   # boost
            fac   = 1.0 + lam * c              # confidence-adaptive
            new_w = min(w_cap, w * fac)
        if abs(new_w - w) > 1e-12:
            G[u][v]["weight"] = new_w
            cnt += 1
    return cnt


def _geometry_scores_triangle_closure(
    G: nx.Graph,
    edges: np.ndarray,
    *,
    S0: int,
    geo_cut: float,
    boost_cut_geo: float,  # kept for possible future use
    closure_metric: str = "cn_over_sqrtdeg",
    scores: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    OPTION B: geometry scores from local triangle closure + small components.

    Returns
    -------
    conf_g : np.ndarray, shape (m,)
        Confidence scores in [0,1) for each edge.
    mask_geo_shrink : np.ndarray[bool], shape (m,)
        Geometry‑induced edges to shrink.
    mask_geo_boost : np.ndarray[bool], shape (m,)
        Currently empty – geometry edges are only shrunk.
    """
    m = len(edges)
    if m == 0:
        return (
            np.zeros(0, dtype=float),
            np.zeros(0, dtype=bool),
            np.zeros(0, dtype=bool),
        )

    if scores is None:
        scores = edge_locality_scores(G, edges, metric=closure_metric)

    # Confidence in [0,1)
    conf_g = scores / (scores + 1.0)

    # High-closure edges define H_high
    cutoff = np.percentile(scores, geo_cut * 100.0) if m > 0 else 0.0
    high_mask = scores >= cutoff

    H_high = nx.Graph()
    H_high.add_nodes_from(G.nodes())
    for (u, v), flag in zip(edges, high_mask):
        if flag:
            H_high.add_edge(u, v)

    comp_id: Dict[int, int] = {}
    comp_size: Dict[int, int] = {}
    for cid, comp in enumerate(nx.connected_components(H_high)):
        size = len(comp)
        for u in comp:
            comp_id[u] = cid
        comp_size[cid] = size

    # Geometry-induced: edge in small high-closure component
    geom_mask = np.zeros(m, dtype=bool)
    for idx, (u, v) in enumerate(edges):
        if not high_mask[idx]:
            continue
        cu = comp_id.get(u, None)
        cv = comp_id.get(v, None)
        if cu is None or cu != cv:
            continue
        size = comp_size.get(cu, 0)
        if size <= S0:
            geom_mask[idx] = True

    mask_geo_shrink = geom_mask
    # Geometry edges are not boosted by default; keep mask for future tuning.
    mask_geo_boost = np.zeros(m, dtype=bool)
    return conf_g, mask_geo_shrink, mask_geo_boost


def _geometry_scores_persistence_no_coords(
    G: nx.Graph,
    edges: np.ndarray,
    scores: np.ndarray,
    node2idx: Dict,
    *,
    frac_sweep: Tuple[float, ...],
    t0: float,
    S0: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    OPTION A (no coords): multi-scale locality persistence via union-find.

    - Build a filtration over edges based on descending locality scores.
    - For each sweep level τ in `frac_sweep` (interpreted as percentiles),
      we keep edges with score ≥ percentile(scores, τ*100) and maintain DSU.
    - s(e) = earliest sweep index where endpoints of e are connected.
    - Geometry-induced if:
        (a) e is local enough to be in E_{t0},
        (b) s(e) is at or before t0's sweep index,
        (c) component size at t0 for u is ≤ S0.
    """
    m = len(edges)
    if m == 0:
        return (
            np.zeros(0, dtype=float),
            np.zeros(0, dtype=bool),
            np.zeros(0, dtype=bool),
        )

    if not frac_sweep:
        raise ValueError("frac_sweep must be non-empty for geo_method='A_persistence_no_coords'.")
    if t0 not in frac_sweep:
        raise ValueError("t0 must be one of the values in frac_sweep.")

    # Sort frac_sweep from most stringent (largest percentile) downwards
    levels = sorted(frac_sweep, reverse=True)
    level_index = {tau: idx for idx, tau in enumerate(levels)}
    idx_t0 = level_index[t0]

    thresholds = np.percentile(scores, [tau * 100.0 for tau in levels])

    nodes = list(G.nodes())
    n = len(nodes)
    idx_of = node2idx

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

    # Edges sorted by descending locality
    order = np.argsort(-scores)
    ptr = 0

    s_idx = np.full(m, np.inf, dtype=float)

    parent_t0 = None
    size_t0 = None

    for li, (tau, thr) in enumerate(zip(levels, thresholds)):
        # Add all edges with score >= thr that haven't been added yet
        while ptr < m and scores[order[ptr]] >= thr:
            e_idx = int(order[ptr])
            u, v = edges[e_idx]
            ui = idx_of[u]
            vi = idx_of[v]
            union(ui, vi)
            ptr += 1

        if li == idx_t0:
            parent_t0 = parent.copy()
            size_t0 = size.copy()

        # For edges whose connectivity time is not yet set, check if
        # their endpoints are connected at this level.
        for e_idx in range(m):
            if not np.isinf(s_idx[e_idx]):
                continue
            u, v = edges[e_idx]
            ui = idx_of[u]
            vi = idx_of[v]
            if find(ui) == find(vi):
                s_idx[e_idx] = float(li)

    if parent_t0 is None or size_t0 is None:
        raise RuntimeError("t0 level was not visited in persistence computation.")

    # Component sizes at t0
    parent0 = parent_t0
    size0 = size_t0

    def find0(i: int) -> int:
        while parent0[i] != i:
            parent0[i] = parent0[parent0[i]]
            i = parent0[i]
        return i

    comp_size_k0 = np.zeros(n, dtype=int)
    for i in range(n):
        r = find0(i)
        comp_size_k0[i] = size0[r]

    # Confidence from locality scores (normalised to [0,1))
    conf_g = scores / (scores + 1.0)

    # Geometry-induced mask
    geom_mask = np.zeros(m, dtype=bool)
    thr_t0 = thresholds[idx_t0]
    for e_idx, (u, v) in enumerate(edges):
        # (a) e is local enough to be in E_{t0}
        if scores[e_idx] < thr_t0:
            continue
        # (b) s(e) at or before t0
        if not np.isfinite(s_idx[e_idx]) or s_idx[e_idx] > idx_t0:
            continue
        # (c) small component at t0
        ui = idx_of[u]
        if comp_size_k0[ui] <= S0:
            geom_mask[e_idx] = True

    mask_geo_shrink = geom_mask
    mask_geo_boost = np.zeros(m, dtype=bool)
    return conf_g, mask_geo_shrink, mask_geo_boost



def duo_spec(
    H_obs: nx.Graph,
    K: int,
    num_balls: int = 16,  # DEPRECATED/UNUSED: legacy geometry spectral dim
    config: tuple = ("bethe_hessian", "bethe_hessian"),
    *,
    # EM
    max_em_iters=7,
    min_em_iters=2,
    anneal_steps=0,
    warmup_rounds=0,
    # percentile cuts
    comm_cut=0.90,
    geo_cut=0.90,
    # shrink / boost strength
    shrink_comm=1.00,
    shrink_geo=0.80,
    boost_comm=0.60,
    boost_geo=0.40,
    boost_cut_comm=0.97,
    boost_cut_geo=0.97,
    # weight bounds
    w_min=5e-2,
    w_cap=4.0,
    # misc
    # DEPRECATED/UNUSED convergence controls (kept for backwards compat):
    tol=1e-4,
    patience=7,
    # New strict convergence controls
    conv_tol: float = 1e-8,
    conv_window: int = 3,
    # Global scale on update strengths: <1.0 makes reweighting less aggressive
    # Default 0.2 is intentionally conservative to avoid over-shrinking edges.
    update_scale: float = 0.2,
    metric_debug: bool = False,
    random_state=0,
    base_seed=None,
    # DEPRECATED/UNUSED: legacy pruning percentile (no longer used)
    # theta=30,
    spec_params: dict = {},
    # New geometry-discriminator controls
    geo_method: str = "B_triangle_closure",
    S0: int = 20,
    frac_sweep: Tuple[float, ...] = (0.99, 0.97, 0.95, 0.90, 0.85),
    t0: float = 0.95,
    closure_metric: str = "cn_over_sqrtdeg",
    local_score: str = "cn_over_sqrtdeg",
    # DEPRECATED/UNUSED legacy geometry params (kept for backwards-compat):
    # k_sweep: Tuple[int, ...] = (2, 3, 5, 8, 10, 15, 20),
    # k0: int = 8,
):
    """Pure-spectral EM with both up- and down-weighting of edges."""
    # Clamp EM iterations to a hard cap of 10 (can be lowered by caller)
    max_em_cap = 10
    if max_em_iters > max_em_cap:
        print(
            f"[EM] max_em_iters={max_em_iters} requested; "
            f"clamping to {max_em_cap}."
        )
        max_em_iters = max_em_cap
    # Ensure min_em_iters does not exceed the cap
    if min_em_iters > max_em_iters:
        min_em_iters = max_em_iters

    print(
        "[EM] running up to 10 iterations "
        f"(min={min_em_iters}, tol={conv_tol:.1e}, window={conv_window})"
    )

    rng = np.random.default_rng(random_state)
    if base_seed is not None:
        rng_base = np.random.RandomState(base_seed)
    subG = deepcopy(H_obs)
    for _, _, d in subG.edges(data=True):
        d.setdefault("weight", 1.0)

    edges = np.asarray(subG.edges(), dtype=object)
    node2idx = {u: i for i, u in enumerate(subG.nodes())}
    iu_glob, iv_glob = _to_idx(edges, node2idx)

    best, hist, no_imp = {"obj": -np.inf}, [], 0
    config = get_callable(config)

    # def _lam(step, base):            # linear ramp-up after warm-up
    #     d = step - warmup_rounds
    #     if d <= 0:  return 0.0
    #     return base if d >= anneal_steps else base * d / anneal_steps
    def _lam(step, base, warmup_rounds=warmup_rounds, anneal_steps=anneal_steps):
        if step < warmup_rounds:           # ❶ pure warm-up
            return 0.0
        d = step - warmup_rounds
        if d < anneal_steps:               # ❷ linear ramp
            return base * d / anneal_steps
        # ❸ harmonic decay after plateau
        t = d - anneal_steps
        return base / (1 + t)

    # Interpret geo_method aliases
    geo_method_effective = geo_method
    if geo_method_effective == "A_persistence_no_coords":
        geo_method_effective = "A_persistence_no_coords"
    elif geo_method_effective == "A_multiscale_unionfind":
        # Backwards-compat alias: old name now maps to structure-only option A
        geo_method_effective = "A_persistence_no_coords"
    elif geo_method_effective == "B_triangle_closure":
        geo_method_effective = "B_triangle_closure"
    else:
        raise ValueError(
            f"Unknown geo_method '{geo_method}' "
            "(expected 'A_persistence_no_coords' or 'B_triangle_closure')."
        )

    # --- weight–distance correlation BEFORE denoising ------------------------
    geom_corr_before = weight_distance_correlation(
        subG,
        coord_key="coords",
        weight_key="weight",
        corr="spearman",
        debug=metric_debug,
    )

    # "balls" placeholder for backwards compatibility: one group per node
    balls = np.zeros(len(node2idx), dtype=int)

    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------
    for em in range(1, max_em_iters + 1):
        print(f"[EM] iter {em} / {max_em_iters}")
        if base_seed is not None:
            random_state = rng_base.randint(0, 2**32 - 1)
        # ---------------- community embedding -----------------------------
        Q_comm, hard_comm, *_ = config[0](
            subG, q=K, random_state=random_state, **spec_params
        )
        p_same = _edge_same_prob(Q_comm, iu_glob, iv_glob)
        if p_same.size == 0:
            mask_comm_shrink = np.zeros(0, dtype=bool)
            mask_comm_boost = np.zeros(0, dtype=bool)
            p_same_min = p_same_med = p_same_max = float("nan")
            thr_comm_shrink = thr_comm_boost = float("nan")
        else:
            thr_comm_shrink = np.percentile(p_same, comm_cut * 100)
            thr_comm_boost = np.percentile(p_same, boost_cut_comm * 100)
            mask_comm_shrink = p_same > thr_comm_shrink
            mask_comm_boost = p_same > thr_comm_boost
            p_same_min = float(p_same.min())
            p_same_med = float(np.median(p_same))
            p_same_max = float(p_same.max())

        # ---------------- fast geometry discriminator (recomputed each iter) --
        if len(edges) == 0:
            conf_g = np.zeros(0, dtype=float)
            mask_geo_shrink = np.zeros(0, dtype=bool)
            mask_geo_boost = np.zeros(0, dtype=bool)
            conf_min = conf_med = conf_max = float("nan")
            thr_geo_shrink = thr_geo_boost = float("nan")
        else:
            # Locality scores for current graph structure
            scores_local = edge_locality_scores(subG, edges, metric=local_score)

            if geo_method_effective == "A_persistence_no_coords":
                conf_g, mask_geo_shrink, mask_geo_boost = (
                    _geometry_scores_persistence_no_coords(
                        subG,
                        edges,
                        scores_local,
                        node2idx=node2idx,
                        frac_sweep=frac_sweep,
                        t0=t0,
                        S0=S0,
                    )
                )
            elif geo_method_effective == "B_triangle_closure":
                conf_g, mask_geo_shrink, mask_geo_boost = (
                    _geometry_scores_triangle_closure(
                        subG,
                        edges,
                        S0=S0,
                        geo_cut=geo_cut,
                        boost_cut_geo=boost_cut_geo,
                        closure_metric=closure_metric,
                        scores=scores_local,
                    )
                )

            if conf_g.size == 0:
                conf_min = conf_med = conf_max = float("nan")
                thr_geo_shrink = thr_geo_boost = float("nan")
            else:
                conf_min = float(conf_g.min())
                conf_med = float(np.median(conf_g))
                conf_max = float(conf_g.max())
                thr_geo_shrink = float(np.percentile(conf_g, geo_cut * 100))
                thr_geo_boost = float(np.percentile(conf_g, boost_cut_geo * 100))

        # Debug statistics for masks and scores
        print(
            "[EM][Debug] p_same stats: "
            f"min={p_same_min:.4f}, med={p_same_med:.4f}, max={p_same_max:.4f}, "
            f"thr_shrink={thr_comm_shrink:.4f}, thr_boost={thr_comm_boost:.4f}"
        )
        print(
            "[EM][Debug] conf_g stats: "
            f"min={conf_min:.4f}, med={conf_med:.4f}, max={conf_max:.4f}, "
            f"thr_shrink={thr_geo_shrink:.4f}, thr_boost={thr_geo_boost:.4f}"
        )

        # ---------------- edge re-weighting  ------------------------------
        λc, λg = _lam(em, shrink_comm), _lam(em, shrink_geo)
        λcB, λgB = _lam(em, boost_comm), _lam(em, boost_geo)

        # Make per-iteration updates less aggressive via a global scale.
        # update_scale in (0,1] softens shrink/boost magnitudes.
        λc *= update_scale
        λg *= update_scale
        λcB *= update_scale
        λgB *= update_scale

        drop_c = _scale_edges(subG, mask_comm_shrink, p_same, λc,
                              w_min, w_cap, "shrink")
        drop_g = _scale_edges(subG, mask_geo_shrink,  conf_g, λg,
                              w_min, w_cap, "shrink")
        boost_c = _scale_edges(subG, mask_comm_boost, p_same, λcB,
                               w_min, w_cap, "boost")
        boost_g = _scale_edges(subG, mask_geo_boost,  conf_g, λgB,
                               w_min, w_cap, "boost")

        if em == 1 and (
            mask_comm_shrink.sum() == 0
            and mask_comm_boost.sum() == 0
            and mask_geo_shrink.sum() == 0
            and mask_geo_boost.sum() == 0
        ):
            print(
                "[EM][Debug] no edges selected on iter1; "
                "check cuts/percentiles and score distributions."
            )

        print(
            f"[EM]   reweighted edges - "
            f"comm_shrink: {drop_c}, geo_shrink: {drop_g}, "
            f"comm_boost: {boost_c}, geo_boost: {boost_g}"
        )

        # ---------------- objective & bookkeeping -------------------------
        obj = float(np.max(Q_comm, axis=1).sum())
        hist.append(dict(it=em, obj=obj, edges=subG.number_of_edges()))

        if obj > best["obj"]:
            best.update(obj=obj, beliefs=Q_comm, node2idx=node2idx)
            no_imp = 0
        else:
            no_imp += 1

        # early-stop conditions
        # DEPRECATED/UNUSED: patience-based early stopping (kept for reference)
        # if no_imp >= patience:
        #     print(f"[EM] patience reached ({patience}) at iter {em}")
        #     break
        # DEPRECATED/UNUSED: old single-step tol-based convergence
        # if em > 1 and abs(obj - hist[-2]['obj']) < tol:
        #     print(f"[EM] converged at iter {em}")
        #     break
        if subG.number_of_edges() == 0:
            print("[EM] graph emptied – stop")
            break

        # Strict convergence: only after min_em_iters, based on stability of
        # the objective over the last `conv_window` iterations.
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

    # Final community embedding on the reweighted graph (no pruning)
    Q, hard, _, _ = config[0](
        subG,
        q=K,
        random_state=random_state,
        **spec_params,
    )

    # --- weight–distance correlation AFTER denoising -------------------------
    geom_corr_after = weight_distance_correlation(
        subG,
        coord_key="coords",
        weight_key="weight",
        corr="spearman",
        debug=metric_debug,
    )
    before_val = geom_corr_before.get("spearman", float("nan"))
    after_val = geom_corr_after.get("spearman", float("nan"))

    if not np.isnan(before_val) and not np.isnan(after_val):
        geom_corr_delta = after_val - before_val
        print(
            "[Metric] weight-distance spearman "
            f"before={before_val:.4f}, "
            f"after={after_val:.4f}, "
            f"delta={geom_corr_delta:.4f}, "
            f"n={geom_corr_after.get('n_edges', 0)}, "
            f"reason_after={geom_corr_after.get('reason')}, "
            f"used_jitter_after={geom_corr_after.get('used_jitter')}"
        )
    else:
        geom_corr_delta = float("nan")
        if np.isnan(before_val) and not np.isnan(after_val):
            print(
                "[Metric] weight-distance spearman: "
                f"before is NaN, after={after_val:.4f}; "
                "delta undefined (before NaN)."
            )
        elif np.isnan(before_val) and np.isnan(after_val):
            print(
                "[Metric] weight-distance spearman: "
                "metric undefined both before and after."
            )
        else:
            print(
                "[Metric] weight-distance spearman: "
                f"before={before_val:.4f}, after is NaN; delta undefined."
            )
    result = dict(
        beliefs=Q,
        communities=Q.argmax(1),
        balls=balls,
        node2idx=node2idx,
        idx2node={i: u for u, i in node2idx.items()},
        history=hist,
        G_final=subG,
        geom_corr_before=geom_corr_before,
        geom_corr_after=geom_corr_after,
        geom_corr_delta=geom_corr_delta,
    )

    # Backward-compat aliases for older callers expecting proxy_* keys
    # (will remove later once all call sites are updated).
    # Old proxy_geometry_graph_correlation returned:
    #   {"corr_value", "p_value", "n_edges_used", "metric", "local_score"}
    # We map our new geometry metric onto that structure minimally.
    proxy_before = dict(
        corr_value=float(geom_corr_before.get("spearman", float("nan"))),
        p_value=float(geom_corr_before.get("spearman_p", float("nan"))),
        n_edges_used=int(geom_corr_before.get("n_edges", 0)),
        metric="spearman",
        local_score="distance",
    )
    proxy_after = dict(
        corr_value=float(geom_corr_after.get("spearman", float("nan"))),
        p_value=float(geom_corr_after.get("spearman_p", float("nan"))),
        n_edges_used=int(geom_corr_after.get("n_edges", 0)),
        metric="spearman",
        local_score="distance",
    )
    result["proxy_corr_before"] = proxy_before   # backward-compat alias; will remove later
    result["proxy_corr_after"] = proxy_after     # backward-compat alias; will remove later
    result["proxy_corr_delta"] = geom_corr_delta # backward-compat alias; will remove later

    return result
    
    # return dict(
    #     beliefs=best["beliefs"],
    #     communities=hard_final,
    #     balls=best["balls"],
    #     node2idx=best["node2idx"],
    #     idx2node={i: u for u, i in best["node2idx"].items()},
    #     history=hist,
    #     G_final=subG,
    # )



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

if __name__ == "__main__":
    from block_models.sbm.sbm import generate_noisy_sbm
    from deprecated.observations.standard_observe import PairSamplingObservation, get_coordinate_distance
    from algorithms.bp.vectorized_bp import belief_propagation, beta_param
    # G_true = generate_gbm_poisson(lam=50, K=K, a=a, b=b, seed=42)
# {"parameters": {"n": 650, "K": 2, "a": 40, "b": 6, "p_in": 0.39858291463936507, "p_out": 0.05978743719590476, "sigma": 0.75},
    # G_true = generate_noisy_sbm(
    #     n=650,
    #     K=2,
    #     p_in=0.39858291463936507,
    #     p_out=0.05978743719590476,
    #     sigma=0.90,
    #     seed=42
    # )
    # G_true = generate_noisy_sbm(
    #     n=400,
    #     K=2,
    #     p_in=1,
    #     p_out=0.1,
    #     sigma=0.75,
    #     seed=42
    # )
# {"parameters": {"n": 300, "K": 2, "a": 55, "b": 6, "p_in": 1.0456934536869702, "p_out": 0.11407564949312402, "sigma": 0.5}    
    G_true = generate_noisy_sbm(
        n=300,
        K=2,
        p_in=1.0456934536869702,
        p_out=0.11407564949312402,
        sigma=0.75,
        seed=42
    )
    print("Generated graph with", len(G_true.nodes()), "nodes and", len(G_true.edges()), "edges")

    # subG = geometric_censor(G_true, r=0.5, seed=42)
    duo_params = dict(
    K               = 2,
    num_balls       = 8,    
    config          = 'motif',

    max_em_iters    = 60,
    warmup_rounds   = 0,
    anneal_steps    = 8,

    # community masks & strengths
    comm_cut        = 0.80,
    shrink_comm     = 0.05,
    boost_cut_comm  = 0.90,
    boost_comm      = 0.80,

    # geometry disabled
    geo_cut         = 0.80,
    shrink_geo      = 0.40,
    boost_cut_geo   = 0.97,
    boost_geo       = 0.10,

    # weight bounds
    w_min           = 0.01,
    w_cap           = 2.00,

    tol             = 1e-5,
    patience        = 5,
    random_state    = 42,
    base_seed       = 0,

    spec_params     = dict(
        dim       = 64,
        walk_len  = 40,
        num_walks = 2,
        window    = 5,
        weight_pow=1.0,
    )
    )
   
    res = duo_spec(
        G_true, **duo_params
    )

    preds = res["communities"]


    true_labels = get_true_communities(G_true, node2idx=None, attr="comm")
    stats = detection_stats(preds, true_labels)

    print("\n=== Community‑detection accuracy ===")
    for k, v in stats.items():
        print(f"{k:>25s} : {v}")
        
    _, preds, _, _ = motif_spectral_embedding(
        G_true,
        q=2,
        random_state=42,
        **duo_params["spec_params"]
    )
    stats = detection_stats(preds, true_labels)

    print("\n=== Community‑detection accuracy Motif===")
    for k, v in stats.items():
        print(f"{k:>25s} : {v}")
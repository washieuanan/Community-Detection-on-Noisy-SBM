from __future__ import annotations
from typing import Dict, List, Tuple, Literal
import networkx as nx
import numpy as np
import scipy.sparse.linalg as sla
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.optimize import linear_sum_assignment
from scipy.stats import permutation_test, mode
from scipy.sparse import coo_matrix, csr_matrix, linalg as splinalg
from algorithms.bp.vectorized_bp import belief_propagation, belief_propagation_weighted
from algorithms.bp.vectorized_bp import spectral_clustering
from collections import defaultdict
from algorithms.bp.old.duo_bp import duo_bp
from copy import deepcopy
from scipy.sparse import diags
from sklearn.cluster import KMeans
from scipy.sparse import csgraph
from scipy.sparse.csgraph import laplacian as cs_lap
from numpy.linalg import norm
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
from typing import Union, Tuple
from scipy.sparse.linalg import LinearOperator, eigsh, lobpcg
from scipy.sparse.csgraph import shortest_path
import math
from copy import deepcopy
from typing import Hashable, Iterable
from numpy.random import default_rng
from block_models.gbm import generate_graph 
from deprecated.observations.standard_observe import PairSamplingObservation, get_coordinate_distance
from algorithms.bp.vectorized_bp import belief_propagation, beta_param
from sklearn.neighbors import KernelDensity

from algorithms.spectral_ops.attention import byoe_embedding, multihead_spectral_embedding, motif_spectral_embedding_vec
# censoring schemes
# ---------------------------------------------------------------------
# 1)  Erdős–Rényi edge–mask  (keep each edge independently with ρ)
# ---------------------------------------------------------------------
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
    # --- create empty graph with same node set -----------------------
    H = nx.Graph()
    if copy_node_attrs:
        # deep-copy node attributes
        for u, attrs in G.nodes(data=True):
            H.add_node(u, **deepcopy(attrs))
    else:
        H.add_nodes_from(G.nodes())

    # --- Bernoulli retention for each edge ---------------------------
    for u, v, attrs in G.edges(data=True):
        if rng.random() < rho:
            H.add_edge(u, v, **deepcopy(attrs))

    return H


# ---------------------------------------------------------------------
# 2)  Geometric censoring  (keep edges whose ‖coords_u – coords_v‖ ≤ r)
# ---------------------------------------------------------------------
def geometric_censor(
    G: nx.Graph,
    r: float,
    *,
    coord_key: str = "coords",
    metric: str = "euclidean",
    copy_node_attrs: bool = True,
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


def ising_cut_cleaner(G, labels, weight="weight"):
    """
    Perform one deterministic Ising-cut sweep on G.
    
    Parameters
    ----------
    G : networkx.Graph
        Graph whose edges carry a numeric `weight` attribute.
    labels : dict
        Mapping node -> community label (any hashable).
    weight : str
        Name of the edge attribute to use as the coupling strength.
    
    Returns
    -------
    new_labels : dict
        Copy of labels with some nodes flipped if it reduces the Ising energy.
    """
    new_labels = labels.copy()
    for u in G.nodes():
        # aggregate neighbor‐weights by their labels
        wsum = defaultdict(float)
        for v, dat in G[u].items():
            w = dat.get(weight, 1.0)
            wsum[labels[v]] += w
        # pick the neighbor‐label with maximum total weight
        best_label, best_w = max(wsum.items(), key=lambda x: x[1], default=(labels[u], 0.0))
        # current-label weight
        curr_w = wsum.get(labels[u], 0.0)
        # flip if another label has strictly more support
        if best_label != labels[u] and best_w > curr_w:
            new_labels[u] = best_label
    return new_labels


# ---------- New helper  -----------------------------------------------
def add_gaussian_weights_from_dist(G, sigma=None, scale=1.0):
    dists = [dat["dist"] for _,_,dat in G.edges(data=True)]
    if sigma is None:
        sigma = np.median(dists) or 1.0
    for _,_,dat in G.edges(data=True):
        if "weight" not in dat:          # preserve EM-updated weights
            dat["weight"] = np.exp(-0.5*(dat["dist"]/(sigma*scale))**2)

def add_localscale_weights(G, k=5):
    dmat = defaultdict(list)
    for u, v, d in G.edges(data=True):
        dmat[u].append(d["dist"])
        dmat[v].append(d["dist"])

    sigma = {u: np.median(sorted(ds)[:k]) for u, ds in dmat.items()}

    for u, v, d in G.edges(data=True):
        d["weight"] = np.exp(-0.5*(d["dist"]/(sigma[u]*sigma[v]))**2)


def renormalise_for_sampling(G, C):
    n      = G.number_of_nodes()
    p_obs  = C / (n*(n-1)/2)
    for _,_,d in G.edges(data=True):
        d["weight"] /= max(p_obs, 1e-9)        # debias, clip for safety


def add_jaccard_edges(
    G: nx.Graph,
    frac_keep: float = 0.02,          # add at most this fraction of |E| edges
    min_common: int  = 2,             # need ≥ 2 shared neighbours
    w_scale: float   = 0.05,          # ≤ 5 % of the weakest real edge
    tag: str         = "ghost",       # so _scale_or_prune() can skip them
) -> int:
    """
    Two-hop Jaccard edge completion.
    Adds light auxiliary edges between pairs that are *not* observed but
    share many neighbours.  The weight is
        w_ij = min{ Jaccard(i,j) , w_scale } · w_min,
    where w_min is the smallest existing real edge weight.
    Returns
    -------
    n_added : how many edges were inserted.
    """
    # ---------- basic book-keeping ------------------------------------------
    adj  = {u: set(G.neighbors(u)) for u in G}
    w_min = min(d.get("weight", 1.0) for _, _, d in G.edges(data=True))
    cand  = {}

    # ---------- gather candidates (two hops) ---------------------------------
    for u in G:
        for w in adj[u]:
            for v in adj[w]:
                if v <= u or G.has_edge(u, v):
                    continue                      # ignore existing / duplicates
                inter = adj[u].intersection(adj[v])
                if len(inter) < min_common:
                    continue
                jacc  = len(inter) / len(adj[u].union(adj[v]))
                cand[(u, v)] = jacc

    if not cand:
        return 0

    # ---------- keep strongest frac_keep -------------------------------------
    k_top   = max(1, int(frac_keep * G.number_of_edges()))
    top     = sorted(cand.items(), key=lambda kv: kv[1], reverse=True)[:k_top]

    # ---------- insert -------------------------------------------------------
    for (u, v), j in top:
        w     = min(j, w_scale) * w_min
        d_est = np.sqrt(-2.0 * np.log(j + 1e-12))   # crude inverse-Gaussian
        G.add_edge(u, v,
                weight = w,
                dist   = d_est,   #  ←  NEW, any positive placeholder works
                **{tag: True})

    return len(top)

def kl_refine(G, labels):
    improved = True
    while improved:
        improved = False
        for u in G:
            cur = labels[u]
            # Δscore if we flip to the other block
            delta = sum(d["weight"]*(labels[v]==cur) for v,d in G[u].items()) \
                  - sum(d["weight"]*(labels[v]!=cur) for v,d in G[u].items())
            if delta < 0:  # flipping lowers "cut" → better clustering
                labels[u] = 1-cur
                improved = True
    return labels

def label_propagation(G, labels, rounds=10):
    for _ in range(rounds):
        for u in G.nodes():
            votes = defaultdict(float)
            for v,d in G[u].items():
                votes[labels[v]] += d["weight"]
            labels[u] = max(votes, key=votes.get)
    return labels


###############################################################################
# Utility: minimal k‑means++ (pure NumPy)                                     #
###############################################################################

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _edge_same_prob(bel: np.ndarray, iu, iv) -> np.ndarray:
    """P[u,v same label] for many edges at once (einsum)."""
    return np.einsum("ij,ij->i", bel[iu], bel[iv])


def _to_idx(edges, node2idx):
    iu = np.fromiter((node2idx[u] for u, _ in edges), int, len(edges))
    iv = np.fromiter((node2idx[v] for _, v in edges), int, len(edges))
    return iu, iv


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


def _scale_or_prune(
    G: nx.Graph,
    mask: np.ndarray,
    probs: np.ndarray,
    lam: float,
    w_min: float,
) -> int:
    """
    Edge re-weighting / pruning helper – identical logic, just tidied.
    """
    edges = np.asarray(G.edges(), dtype=object)
    n_removed = 0
    for (u, v), m, p in zip(edges, mask, probs):
        if not m:
            continue
        dat = G[u][v]
        if dat.get("ghost"): 
            continue
        if dat.get("ghost_knn"):
            continue
        w = G[u][v].get("weight", 1.0)
        w *= 1.0 - lam * p        # shrink
        if w < w_min:
            G.remove_edge(u, v)
            n_removed += 1
        else:
            G[u][v]["weight"] = w
    return n_removed


# ----------------------------------------------------------------------
# EM driver – spectral dual (“duo_spec”)
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# 1)  Bethe–Hessian  (with NB-based r  + better softmax confidence)
# ----------------------------------------------------------------------
def _largest_nb_eig(G, nodes, max_iter=200, tol=1e-5, seed=42):
    """
    Power-iteration for the leading eigenvalue of the non-backtracking matrix
    without materialising it explicitly (O(|E|) memory).
    """
    rng   = np.random.default_rng(seed)
    m     = G.number_of_edges()
    # directed edge list -------------------------------------------------
    edges = []
    for u, v in G.edges():
        edges.append((u, v))
        edges.append((v, u))
    e2idx = {e: i for i, e in enumerate(edges)}
    idx2e = edges
    n_dir = len(edges)

    # adjacency helper  --------------------------------------------------
    nbr = [[] for _ in range(n_dir)]
    for idx, (u, v) in enumerate(idx2e):
        for w in G.neighbors(v):
            if w == u:
                continue            # non-backtracking
            nbr[idx].append(e2idx[(v, w)])

    # power iteration ----------------------------------------------------
    x     = rng.standard_normal(n_dir)
    x    /= norm(x)
    lam   = 0.0
    # for _ in range(max_iter):
    #     x_new = np.zeros_like(x)
    #     for i, js in enumerate(nbr):
    #         x_new[i] = x[js].sum()
    #     lam_new = norm(x_new)
    #     x_new  /= lam_new
    #     if abs(lam_new - lam) < tol * lam_new:
    #         lam = lam_new
    #         break
    #     x, lam = x_new, lam_new
    for _ in range(max_iter):
        x_new.fill(0.0)
        for i, js in enumerate(nbr):
            if js:                       # skip dangling edges
                x_new[i] = x[js].sum()

        lam_new = norm(x_new)
        if lam_new < 1e-12:              # <- NEW: vector collapsed
            x = rng.standard_normal(n_dir)
            x /= norm(x)
            continue                     # fresh restart

        x_new /= lam_new
        if abs(lam_new - lam) < tol * lam_new:
            return lam_new               # converged
        x, lam = x_new, lam_new

    return lam        # leading eigen-value ρ


# ---------------------------------------------------------------------------
# Confidence helper: 1 / (1 + ‖x − μ‖)
# ---------------------------------------------------------------------------
def _conf_from_center(X, mu):
    """
    Confidence matrix Q[i,c] = 1 / (1 + ||x_i − μ_c||_2)
    Rows are re-normalised to sum to 1.
    """
    dists = np.linalg.norm(X[:, None, :] - mu[None, :, :], axis=-1)  # (n,q)
    Q     = 1.0 / (1.0 + dists)
    Q    /= Q.sum(axis=1, keepdims=True)
    return Q


# ---------------------------------------------------------------------------
# 1) Bethe–Hessian geometry step  (soft confidence)
# ---------------------------------------------------------------------------
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
    # ---------- node order ----------------------------------------------------
    nodes      = list(H_obs.nodes())
    node2idx   = {u: i for i, u in enumerate(nodes)}
    idx2node   = {i: u for u, i in node2idx.items()}
    n          = len(nodes)

    # ---------- (optional) weight from 'dist' ---------------------------------
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

    # ---------- choose r ------------------------------------------------------
    # ---------- choose r ----------------  ----------------------------------
    if use_nonbacktracking:
        raise NotImplementedError("non-backtracking r not yet hooked in")
    r = np.sqrt(deg.mean())

    # ---------- Bethe–Hessian -------------------------------------------------
    I  = diags(np.ones(n))
    Hr = (r * r - 1.0) * I - r * A + D
    k = q + 1
    ncv = 2 * min(n - 1, max(2*k + 1, k + 20))
    vals, vecs = eigsh(-Hr, k=q, which="LA", ncv = ncv)   # (n,q)

    # ---------- k-means & confidence -----------------------------------------
    km    = KMeans(n_clusters=q, n_init=20, random_state=random_state).fit(vecs)
    hard  = km.labels_
    mu    = km.cluster_centers_
    Q     = _conf_from_center(vecs, mu)

    return Q, hard, node2idx, idx2node

def hybrid_bethe_dirichlet(
    H_obs               : nx.Graph,
    q                   : int,
    *,
    p_same_key          : str   = "p_same",     # edge-attr key ⇢ community score ∈ [0,1]
    g_geo_key           : str   = "g_geo",      # edge-attr key ⇢ geometry  score ∈ [0,1]
    tau                 : float = 0.15,         # geometry-suppression coefficient
    weight_from_dist    : bool  = True,
    sigma_scale         : float = 1.0,
    random_state        : int   = 42,
):
    """
    Hybrid Bethe–Dirichlet spectral embedding (one-shot DUOSPEC variant).

    Parameters
    ----------
    H_obs : nx.Graph
        Observed (possibly masked) graph.  Edges should carry:
            • p_same_key : P(edge is within-community)  ∈ [0,1]
            • g_geo_key  : P(edge is geometric noise)   ∈ [0,1]
            • dist       : (optional) latent distance   (for weight_from_dist)
    q : int
        Number of clusters / communities.
    tau : float
        Amount of geometry suppression  (0 ⇒ ignore g_geo, 1 ⇒ full subtraction).
    weight_from_dist, sigma_scale
        Same semantics as in `bethe_hessian`.

    Returns
    -------
    Q      : (n,q) soft cluster confidence (rows sum to 1)
    hard   : (n,)  hard labels 0..q-1  (argmax rows of Q)
    node2idx / idx2node : mapping helpers
    """
    # ---------- node order ----------------------------------------------------
    nodes      = list(H_obs.nodes())
    node2idx   = {u: i for i, u in enumerate(nodes)}
    idx2node   = {i: u for u, i in node2idx.items()}
    n          = len(nodes)

    # ---------- optional weight from 'dist' -----------------------------------
    if weight_from_dist:
        d_vals = np.array([d.get("dist", 1.0) for *_, d in H_obs.edges(data=True)])
        sigma  = (np.median(d_vals) or 1.0) * sigma_scale
        for u, v, d in H_obs.edges(data=True):
            d["weight"] = np.exp(-0.5 * (d.get("dist", 0.0) / sigma) ** 2)
    else:
        for *_, d in H_obs.edges(data=True):
            d.setdefault("weight", 1.0)

    # ---------- edge weights  w_uv  -------------------------------------------
    # w_uv =  (p_same - tau * g_geo) *  base_weight
    for u, v, d in H_obs.edges(data=True):
        p_same = d.get(p_same_key, 0.0)
        g_geo  = d.get(g_geo_key,  0.0)
        base   = d["weight"]
        signal        = max(0.0, p_same - tau * g_geo)         # ⬅ clip
        d["w_hybrid"] = signal * base
        # d["w_hybrid"] = (p_same - tau * g_geo) * base

    # ---------- sparse matrices ------------------------------------------------
    A   = nx.to_scipy_sparse_array(H_obs, nodelist=nodes,
                                   format="csr", weight="w_hybrid")
    deg = np.asarray(A.sum(axis=1)).ravel()
    D   = diags(deg)

    # ---------- Bethe radius  r  (Saade et al. 2015 heuristic) -----------------
    m      = A.nnz // 2
    avg_d  = deg.mean()
    r      = np.sqrt(avg_d / max(avg_d - 2 * m / n, 1e-8))

    # ---------- Hybrid operator  H(r,tau) -------------------------------------
    I  = diags(np.ones(n))
    Hr = (r * r - 1.0) * I - r * D + A

    # ---------- eigendecomposition --------------------------------------------
    k   = q + 1                      # grab one extra for safety
    ncv = 2 * min(n - 1, max(2 * k + 1, k + 20))
    vals, vecs = eigsh(-Hr, k=q, which="LA", ncv=ncv, tol=1e-5)

    # ---------- k-means + soft confidence -------------------------------------
    km    = KMeans(n_clusters=q, n_init=20, random_state=random_state).fit(vecs)
    hard  = km.labels_
    mu    = km.cluster_centers_
    Q     = _conf_from_center(vecs, mu)          # ← your existing helper

    return Q, hard, node2idx, idx2node

from scipy.sparse import eye, diags, csr_matrix, linalg as spla
from sklearn.cluster import KMeans

def ppr_proj(
    H_obs        : nx.Graph,
    q            : int,
    *,
    alpha        : float = 0.3,
    beta         : float = 0.40,
    random_state : int   = 42,
):
    """
    Spectral embedding based on Projected Personalized-PageRank (PPR-Proj).

    Returns
    -------
    Q      : (n,q) soft cluster confidence (rows sum to 1)
    hard   : (n,)  hard labels
    node2idx / idx2node : mapping helpers
    """
    # ---- node order ---------------------------------------------------------
    nodes      = list(H_obs.nodes())
    node2idx   = {u:i for i,u in enumerate(nodes)}
    idx2node   = {i:u for u,i in node2idx.items()}
    n          = len(nodes)

    # ---- sparse adjacency (unweighted) --------------------------------------
    A = nx.to_scipy_sparse_array(H_obs, nodelist=nodes,
                                 format="csr", weight=None)
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg_safe = deg.copy()
    deg_safe[deg_safe == 0] = 1          # prevent division by zero
    Dinv = sp.diags(1.0 / deg_safe)

    # ---- truncated PPR kernel  ---------------------------------------------
    # Y = (1-α) D⁻¹ A   (row-stochastic)
    Y = (1.0 - alpha) * Dinv @ A
    X = alpha * sp.eye(n, format="csr")          # 0-hop
    X = X + Y                                    # 1-hop
    X = X + Y @ Y                                # 2-hop
    X = X + Y @ Y @ Y                            # 3-hop   (⟹ good for log-degree)

    # degree-normalise:   K̃ = D^{-1/2} X D^{-1/2}
    dinv_sqrt = sp.diags(1.0 / np.sqrt(deg_safe))
    Ktil = dinv_sqrt @ X @ dinv_sqrt

    # ---- Projected operator  H = (1-β)K̃ + β K̃² -----------------------------
    H = (1.0 - beta) * Ktil + beta * (Ktil @ Ktil)

    # ---- top-q eigenvectors  (largest algebraic) ---------------------------
    k   = q                     # we only need q vectors
    ncv = 2 * min(n - 1, max(2 * k + 1, k + 20))
    vals, vecs = spla.eigsh(H, k=q, which="LA", ncv=ncv, tol=1e-4)

    # ---- k-means & confidence ----------------------------------------------
    km    = KMeans(n_clusters=q, n_init=20, random_state=random_state).fit(vecs)
    hard  = km.labels_
    mu    = km.cluster_centers_
    Q     = _conf_from_center(vecs, mu)       # your existing helper

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
    # --- build sparse adjacency with weights ---
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

    # --- choose r ---
    r = np.sqrt(deg.mean())

    # --- build Hr ---
    I  = diags(np.ones(n))
    Hr = (r*r - 1.0)*I - r*A + D

    # --- spectral solve ---
    if use_lobpcg:
        # Lobpcg tends to converge in O(n·q) per iteration
        X0 = (prev_evecs 
              if (prev_evecs is not None and prev_evecs.shape==(n,q))
              else np.random.RandomState(random_state).randn(n,q))
        vals, vecs = lobpcg(Hr, X0, tol=tol, maxiter=maxiter)
    else:
        # ARPACK on the smallest eigenvalues of Hr
        # warm‐start with prev_evecs flattened to v0
        eig_kwargs = dict(which="SM", tol=tol, maxiter=maxiter)
        if prev_evecs is not None:
            eig_kwargs["v0"] = prev_evecs[:,0]
        vals, vecs = eigsh(Hr, k=q, **eig_kwargs)

    # --- k-means & confidence ---
    km   = KMeans(n_clusters=q, n_init=10, random_state=random_state).fit(vecs)
    hard = km.labels_
    mu   = km.cluster_centers_
    # soft confidences
    diff = vecs[:,None,:] - mu[None,:,:]       # shape (n,q,q)
    Q    = np.exp(-np.sum(diff**2, axis=2))
    Q   /= Q.sum(axis=1, keepdims=True)

    return Q, hard, idx, {i:u for u,i in idx.items()}, vecs
# # ---------------------------------------------------------------------------
# # 2) Normalised-Laplacian community step  (soft confidence)
# # ---------------------------------------------------------------------------
# def laplacian(
#     H_obs        : nx.Graph,
#     q            : int,
#     *,
#     random_state : int = 42,
# ):
#     """
#     Spectral clustering (norm-Laplacian) with 1/(1+dist) confidence.
#     """
#     nodes    = list(H_obs.nodes())
#     node2idx = {u: i for i, u in enumerate(nodes)}
#     idx2node = {i: u for u, i in node2idx.items()}

#     A = nx.to_scipy_sparse_array(H_obs, nodelist=nodes, format="csr")
#     L = csgraph.laplacian(A, normed=True).astype(float)

#     evals, evecs = np.linalg.eigh(L.toarray())
#     X = evecs[:, 1 : q + 1]  # bottom q (skip trivial)

#     km    = KMeans(n_clusters=q, n_init=20, random_state=random_state).fit(X)
#     hard  = km.labels_
#     mu    = km.cluster_centers_
#     Q     = _conf_from_center(X, mu)

#     return Q, hard, node2idx, idx2node

# def regularized_laplacian(
#     H_obs        : nx.Graph,
#     q            : int,
#     *,
#     random_state : int = 42,
# ):
#     """
#     Spectral clustering (regularized Laplacian) with 1/(1+dist) confidence.
#     Uses L_reg = (D + τ I)^(-1/2) A (D + τ I)^(-1/2) with τ = 1.
#     """
#     # --- build index mappings
#     nodes    = list(H_obs.nodes())
#     node2idx = {u: i for i, u in enumerate(nodes)}
#     idx2node = {i: u for u, i in node2idx.items()}
#     n        = len(nodes)

#     # --- adjacency and degrees
#     A        = nx.to_scipy_sparse_array(H_obs, nodelist=nodes, format="csr")
#     deg      = np.array(A.sum(axis=1)).flatten()

#     # --- regularization
#     tau      = 1.0
#     D_reg    = sp.diags(deg + tau)
#     D_inv_s  = sp.diags(1.0 / np.sqrt(deg + tau))

#     # --- regularized Laplacian operator
#     L_reg    = D_inv_s @ A @ D_inv_s

#     # --- eigen-decomposition (dense for simplicity)
#     evals, evecs = np.linalg.eigh(L_reg.toarray())
#     # take top-q eigenvectors (largest eigenvalues)
#     X = evecs[:, -q:]

#     # --- k-means clustering
#     km   = KMeans(n_clusters=q, n_init=20, random_state=random_state).fit(X)
#     hard = km.labels_
#     mu   = km.cluster_centers_

#     # --- soft confidence from distance to centers
#     Q    = _conf_from_center(X, mu)

#     return Q, hard, node2idx, idx2node


# def laplacian(
#     H_obs        : nx.Graph,
#     q            : int,
#     *,
#     random_state : int = 42,
# ):
#     """
#     Spectral clustering (norm-Laplacian) with 1/(1+dist) confidence,
#     using ARPACK for the bottom q+1 eigenpairs of L_norm.
#     """
#     # --- build index mappings
#     nodes    = list(H_obs.nodes())
#     node2idx = {u: i for i, u in enumerate(nodes)}
#     idx2node = {i: u for u, i in node2idx.items()}
#     n        = len(nodes)

#     # --- normalized Laplacian (sparse)
#     A = nx.to_scipy_sparse_array(H_obs, nodelist=nodes, format="csr")
#     L = csgraph.laplacian(A, normed=True)

#     # --- choose ARPACK params
#     m   = q + 1
#     ncv = min(n - 1, max(2 * (m + 20), 10 * m))


#     # # --- compute smallest m eigenpairs of L
#     # evals, evecs = eigsh(
#     #     L,
#     #     k=m,
#     #     sigma=0.0,
#     #     which="LM",
#     #     ncv=ncv,
#     #     tol=1e-4,
#     #     maxiter=10000
#     # )
#     I = sp.eye(n, format="csr")
#     M = I - L
#     evals_M, evecs = eigsh(
#         M,
#         k=m,
#         which="LA",     # largest algebraic
#         ncv=ncv,
#         tol=1e-4,
#         maxiter=10000
#     )
# # recover L's top-of-the-bottom spectrum:
#     evals = 1 - evals_M


#     # --- drop the trivial eigenvector and keep next q
#     X = evecs[:, 1 : m]

#     # --- k-means + soft confidence
#     km   = KMeans(n_clusters=q, n_init=20, random_state=random_state).fit(X)
#     hard = km.labels_
#     mu   = km.cluster_centers_
#     Q    = _conf_from_center(X, mu)

#     return Q, hard, node2idx, idx2node

def laplacian(
    H_obs        : nx.Graph,
    q            : int,
    *,
    random_state : int = 42,
):
    nodes    = list(H_obs.nodes())
    node2idx = {u: i for i, u in enumerate(nodes)}
    idx2node = {i: u for u, i in node2idx.items()}
    n        = len(nodes)

    A = nx.to_scipy_sparse_array(H_obs, nodelist=nodes, format="csr")
    L = csgraph.laplacian(A, normed=True)

    m   = q + 1
    base_ncv = min(n - 1, max(m + 20, 5 * m))

    # STEP 1: try eigsh on M = I - L
    try:
        M = sp.eye(n, format="csr") - L
        evals_M, evecs = eigsh(
            M,
            k=m,
            which="LA",
            ncv=base_ncv,
            tol=1e-4,
            maxiter=10000
        )
        evals = 1 - evals_M

    except (ArpackNoConvergence, ArpackError):
        # bump ncv and retry once
        try:
            bigger_ncv = min(n - 1, base_ncv * 2)
            evals_M, evecs = eigsh(
                M,
                k=m,
                which="LA",
                ncv=bigger_ncv,
                tol=1e-4,
                maxiter=10000
            )
            evals = 1 - evals_M

        except (ArpackNoConvergence, ArpackError):
            # STEP 2: fallback to shift-invert on L (if you still want it)
            try:
                evals, evecs = eigsh(
                    L,
                    k=m,
                    sigma=0.0,
                    which="LM",
                    ncv=base_ncv,
                    tol=1e-4,
                    maxiter=10000
                )
            except (ArpackNoConvergence, RuntimeError, ArpackError):
                # STEP 3: try LOBPCG
                np.random.seed(random_state)
                X0 = np.random.randn(n, m)

                try:
                    evals, evecs = lobpcg(L, X0, largest=False, tol=1e-4, maxiter=200)
                except Exception:
                    # FINAL FALLBACK: dense eigh
                    L_dense = L.toarray()
                    all_vals, all_vecs = np.linalg.eigh(L_dense)
                    evals, evecs = all_vals[:m], all_vecs[:, :m]

    # drop trivial eigenvector (first one) and keep next q
    X = evecs[:, 1:m]

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
    # --- build index mappings
    nodes    = list(H_obs.nodes())
    node2idx = {u: i for i, u in enumerate(nodes)}
    idx2node = {i: u for u, i in node2idx.items()}
    n        = len(nodes)

    # --- adjacency & degree
    A   = nx.to_scipy_sparse_array(H_obs, nodelist=nodes, format="csr")
    deg = np.array(A.sum(axis=1)).ravel()

    # --- regularization
    tau     = 1.0
    D_inv_s = sp.diags(1.0 / np.sqrt(deg + tau))

    # --- form L_reg
    L_reg = D_inv_s @ A @ D_inv_s

    # --- choose ARPACK params
    m   = q
    ncv = min(n - 1, max(m + 20, 5 * m))

    # --- compute the q largest eigenpairs of L_reg
    evals, evecs = eigsh(
        L_reg,
        k=m,
        which="LA",
        ncv=ncv,
        tol=1e-4,
        maxiter=10000
    )

    # --- use those q eigenvectors directly
    X = evecs

    # --- k-means + soft confidence
    km   = KMeans(n_clusters=q, n_init=20, random_state=random_state).fit(X)
    hard = km.labels_
    mu   = km.cluster_centers_
    Q    = _conf_from_center(X, mu)

    return Q, hard, node2idx, idx2node



def morans(
    H_obs        : nx.Graph,
    q            : int,
    *,
    random_state : int = 42,
):
    """
    Spectral clustering (Moran's I operator) with 1/(1+dist) confidence.
    Uses M = D^(-1/2) A D^(-1/2) - (1 1^T) / vol, where vol = sum_i d_i.
    """
    # --- build index mappings
    nodes    = list(H_obs.nodes())
    node2idx = {u: i for i, u in enumerate(nodes)}
    idx2node = {i: u for u, i in node2idx.items()}
    n        = len(nodes)

    # --- adjacency and degrees
    A        = nx.to_scipy_sparse_array(H_obs, nodelist=nodes, format="csr")
    deg      = np.array(A.sum(axis=1)).flatten()
    vol      = deg.sum()

    # --- add regularization to avoid division by zero
    tau      = 1e-8  # small regularization constant
    deg_reg  = deg + tau
    vol_reg  = deg_reg.sum()

    # --- normalized adjacency with regularization
    D_inv_s = sp.diags(1.0/np.sqrt(deg_reg))
    P       = D_inv_s @ A @ D_inv_s  # still sparse

    n = len(deg)
    
    # --- direct sparse matrix approach for better stability
    # Create the constant term (1/vol) * (1 1^T) as a sparse matrix
    ones_vec = np.ones(n) / np.sqrt(vol_reg)
    J = sp.csr_matrix(np.outer(ones_vec, ones_vec))
    M = P - J

    # --- eigen-decomposition (using sparse eigensolvers with better parameters)
    # Using shift-invert mode for better numerical stability
    sigma = 0.5  # shift value near the eigenvalues we want
    try:
        # First attempt with shift-invert mode
        evals, evecs = eigsh(M, k=q, which='LM', tol=1e-5, maxiter=10000,
                             sigma=sigma, mode='normal')
    except Exception:
        # Fallback approach with simpler parameters if first attempt fails
        try:
            evals, evecs = eigsh(M, k=q, which='LM', tol=1e-3, maxiter=5000)
        except Exception:
            # Last resort: use dense eigensolver if sparse methods fail
            M_dense = M.toarray()
            evals, evecs = np.linalg.eigh(M_dense)
            # Take largest eigenvalues
            idx = np.argsort(evals)[-q:]
            evals = evals[idx]
            evecs = evecs[:, idx]
    
    # Sort by eigenvalue (if using sparse method)
    if len(evals) == q:  # Only sort if we got exactly q eigenvalues
        idx = np.argsort(evals)
        evals = evals[idx]
        evecs = evecs[:, idx]
    
    # take top-q eigenvectors (largest eigenvalues)
    X = evecs[:, -q:]

    # --- k-means clustering
    km   = KMeans(n_clusters=q, n_init=20, random_state=random_state).fit(X)
    hard = km.labels_
    mu   = km.cluster_centers_

    # --- soft confidence from distance to centers
    Q    = _conf_from_center(X, mu)

    return Q, hard, node2idx, idx2node


# ---------------------------------------------------------------------------
# 4) SCORE community step  (soft confidence)
# ---------------------------------------------------------------------------
def score(
    H_obs        : nx.Graph,
    q            : int,
    *,
    random_state : int = 42,
):
    """
    SCORE: Spectral Clustering On Ratios‐of‐Eigenvectors.
    1) Regularize adjacency A -> (D+τI)^(-1/2) A (D+τI)^(-1/2)
    2) Compute top‐q eigenvectors v1…vq
    3) Build ratio matrix R[i,ℓ] = v_{ℓ+1}[i]/(v1[i] + eps)
    4) k-means on R, then soft confidence via 1/(1+dist)
    """
    # — build index mappings
    nodes    = list(H_obs.nodes())
    node2idx = {u: i for i, u in enumerate(nodes)}
    idx2node = {i: u for u, i in node2idx.items()}
    n        = len(nodes)

    # — adjacency + regularization
    A    = nx.to_scipy_sparse_array(H_obs, nodelist=nodes, format='csr')
    deg  = np.array(A.sum(axis=1)).ravel()
    τ    = 1.0
    D_s  = diags(1.0 / np.sqrt(deg + τ))
    A_reg = D_s @ A @ D_s

    # — leading q eigenvectors of A_reg
    vals, vecs = eigsh(A_reg, k=q, which='LA', tol=1e-5, maxiter=10000)
    v1 = vecs[:, 0]
    eps = 1e-8

    # — build ratio matrix R ∈ ℝ^{n×(q-1)}
    R = np.zeros((n, q-1))
    for ℓ in range(1, q):
        R[:, ℓ-1] = vecs[:, ℓ] / (v1 + eps)

    # — k-means + soft confidence
    km   = KMeans(n_clusters=q, n_init=20, random_state=random_state).fit(R)
    hard = km.labels_
    Q    = _conf_from_center(R, km.cluster_centers_)
    return Q, hard, node2idx, idx2node


# ---------------------------------------------------------------------------
# Update get_callable to include the new methods
# ---------------------------------------------------------------------------
def get_callable(calls: Union[Tuple, str]):
    func_dict = {
        "bethe_hessian":     bethe_hessian,
        "laplacian":         laplacian,
        "regularized_laplacian": regularized_laplacian,
        "morans":            morans,
        "score":             score,
        "bethe_hessian_fast": bethe_hessian_fast,
        'hybrid_bethe_dirichlet': hybrid_bethe_dirichlet, 
        'ppr_proj': ppr_proj, 
        'byoe_embedding': byoe_embedding,
        'multihead': multihead_spectral_embedding, 
        'motif': motif_spectral_embedding_vec 
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


# ---------------------------------------------------------------------------
# helpers --------------------------------------------------------------------
def _to_idx(edge_arr, n2i):
    iu = np.fromiter((n2i[u] for u, _ in edge_arr), int, len(edge_arr))
    iv = np.fromiter((n2i[v] for _, v in edge_arr), int, len(edge_arr))
    return iu, iv


def _edge_same_prob(Q, iu, iv):
    """probability that two endpoints share the same label under Q."""
    return (Q[iu] * Q[iv]).sum(1)


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
            new_w = max(w_min, w * (1.0 - lam))
        else:                                   # boost
            fac   = 1.0 + lam * c              # confidence-adaptive
            new_w = min(w_cap, w * fac)
        if abs(new_w - w) > 1e-12:
            G[u][v]["weight"] = new_w
            cnt += 1
    return cnt


# ---------------------------------------------------------------------------
# main -----------------------------------------------------------------------
def duo_spec(
    H_obs: nx.Graph,
    K: int,
    num_balls: int = 16,
    config: tuple = ("bethe_hessian", "bethe_hessian"),
    *,
    # EM
    max_em_iters=50,
    anneal_steps=6,
    warmup_rounds=2,
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
    tol=1e-4,
    patience=7,
    random_state=0,
):
    """Pure-spectral EM with both up- and down-weighting of edges."""
    rng = np.random.default_rng(random_state)
    subG = deepcopy(H_obs)
    for _, _, d in subG.edges(data=True):
        d.setdefault("weight", 1.0)

    edges = np.asarray(subG.edges(), dtype=object)
    node2idx = {u: i for i, u in enumerate(subG.nodes())}
    iu_glob, iv_glob = _to_idx(edges, node2idx)

    best, hist, no_imp = {"obj": -np.inf}, [], 0
    config = get_callable(config)

    def _lam(step, base):            # linear ramp-up after warm-up
        d = step - warmup_rounds
        if d <= 0:  return 0.0
        return base if d >= anneal_steps else base * d / anneal_steps

    # -----------------------------------------------------------------------
    for em in range(1, max_em_iters + 1):
        print(f"[EM] iter {em} / {max_em_iters}")
        # ---------------- community embedding -----------------------------
        Q_comm, hard_comm, *_ = config[0](
            subG, q=K, random_state=random_state
        )
        p_same = _edge_same_prob(Q_comm, iu_glob, iv_glob)
        mask_comm_shrink = p_same > np.percentile(p_same, comm_cut * 100)
        mask_comm_boost  = p_same > np.percentile(p_same, boost_cut_comm * 100)

        # ---------------- geometry embedding ------------------------------
        Q_geo, hard_geo, *_ = config[1](
            subG, q=num_balls, random_state=random_state
        )
        same_ball = hard_geo[iu_glob] == hard_geo[iv_glob]
        conf_g = 0.5 * (Q_geo[iu_glob, hard_geo[iu_glob]] +
                        Q_geo[iv_glob, hard_geo[iv_glob]])
        mask_geo_shrink = same_ball & (
            conf_g > np.percentile(conf_g, geo_cut * 100)
        )
        mask_geo_boost = same_ball & (
            conf_g > np.percentile(conf_g, boost_cut_geo * 100)
        )

        # ---------------- edge re-weighting  ------------------------------
        λc, λg = _lam(em, shrink_comm), _lam(em, shrink_geo)
        λcB, λgB = _lam(em, boost_comm), _lam(em, boost_geo)

        drop_c = _scale_edges(subG, mask_comm_shrink, p_same, λc,
                              w_min, w_cap, "shrink")
        drop_g = _scale_edges(subG, mask_geo_shrink,  conf_g, λg,
                              w_min, w_cap, "shrink")
        boost_c = _scale_edges(subG, mask_comm_boost, p_same, λcB,
                               w_min, w_cap, "boost")
        boost_g = _scale_edges(subG, mask_geo_boost,  conf_g, λgB,
                               w_min, w_cap, "boost")

        # ---------------- objective & bookkeeping -------------------------
        obj = float(np.max(Q_comm, axis=1).sum())
        hist.append(dict(it=em, obj=obj, edges=subG.number_of_edges(),
                         shrink_comm=drop_c, shrink_geo=drop_g,
                         boost_comm=boost_c,  boost_geo=boost_g))

        if obj > best["obj"]:
            best.update(obj=obj, beliefs=Q_comm, balls=hard_geo,
                        node2idx=node2idx)
            no_imp = 0
        else:
            no_imp += 1

        # early-stop conditions
        if no_imp >= patience:
            print(f"[EM] patience reached ({patience}) at iter {em}")
            break
        if em > 1 and abs(obj - hist[-2]["obj"]) < tol:
            print(f"[EM] converged at iter {em}")
            break
        if subG.number_of_edges() == 0:
            print("[EM] graph emptied – stop")
            break

    hard_final = best["beliefs"].argmax(1)
    return dict(
        beliefs=best["beliefs"],
        communities=hard_final,
        balls=best["balls"],
        node2idx=best["node2idx"],
        idx2node={i: u for u, i in best["node2idx"].items()},
        history=hist,
        G_final=subG,
    )
# ----------------------------------------------------------- safe EM driver
def duo_bprop(
    G_obs              : nx.Graph,
    K                  : int,
    *,
    # ---------------- algorithmic knobs ------------------------------
    max_em_iters     = 50,
    anneal_steps     = 8,
    warmup_rounds    = 2,
    # cut-offs
    boost_cut        = 0.97,
    add_cut          = 0.995,
    geo_cut          = 0.90,
    # magnitudes
    boost_lambda0    = 0.30,
    shrink_geo       = 0.80,
    boost_cap        = 3.0,
    w_add            = 0.5,
    max_virtual      = 3,
    w_min            = 5e-2,
    # BP, misc ...
    **kwargs,
):
    """
    Safe EM-style refinement that *only* re-scales up/down existing edges
    and (optionally) adds lightweight virtual edges between highly probable
    same-community pairs.
    """
    rng   = np.random.default_rng(kwargs.get("seed", 0))
    G     = deepcopy(G_obs)

    # --- keep original weights for later restoration -----------------
    for u, v, d in G.edges(data=True):
        d.setdefault("weight", 1.0)
        d["w_orig"]  = d["weight"]
        d["w_extra"] = 0.0           # up-weights + virtual edges live here

    hist, best, no_imp = [], {"obj":-np.inf}, 0

    for em in range(1, max_em_iters+1):
        # ---------------- BP for communities -------------------------
        bel_c, _, n2i, _ = belief_propagation_weighted(G, q=K, **kwargs)
        edges   = np.asarray(G.edges(), dtype=object)
        iu, iv  = _to_idx(edges, n2i)
        p_same  = _edge_same_prob(bel_c, iu, iv)

        # percentile thresholds
        th_boost = np.percentile(p_same, boost_cut*100)
        mask_boost = p_same > th_boost                 # existing edges
        # ------------------------------------------------------------- Boost
        λ = min(boost_lambda0 * em / max(1, anneal_steps), boost_lambda0)
        for flag, (u, v) in zip(mask_boost, edges):
            if not flag:
                continue
            data = G.edges[u, v]
            factor = 1 + λ
            factor = min(factor, boost_cap)
            data["w_extra"] = (factor-1)*data["w_orig"]

        # ---------------- Geometry step  (soft down-weight) ----------
        bel_g, _, n2i_g, _ = belief_propagation_weighted(G, q=min(16, K*2), **kwargs)
        lbls_g = bel_g.argmax(1)
        iu_g, iv_g = _to_idx(edges, n2i_g)
        same_ball  = lbls_g[iu_g] == lbls_g[iv_g]
        th_geo     = np.percentile(
            0.5*(bel_g[iu_g,lbls_g[iu_g]]+bel_g[iv_g,lbls_g[iv_g]]),
            geo_cut*100,
        )
        mask_geo = same_ball & (
            0.5*(bel_g[iu_g,lbls_g[iu_g]]+bel_g[iv_g,lbls_g[iv_g]]) > th_geo
        )

        λ_geo = shrink_geo
        for flag, (u, v) in zip(mask_geo, edges):
            if not flag:
                continue
            data = G.edges[u, v]
            data["w_extra"]-= λ_geo*data["w_orig"]      # soft shrink
            # clip
            if data["w_orig"]+data["w_extra"] < w_min:
                data["w_extra"] = w_min-data["w_orig"]

        # ------------------- Add virtual edges -----------------------
        # sample up to max_virtual per node among absent pairs
        th_add = np.percentile(p_same, add_cut*100)
        # rank candidate non-edges
        cand_idx = np.where(p_same > th_add)[0]
        rng.shuffle(cand_idx)
        added = 0
        per_node = {u:0 for u in G}
        for idx in cand_idx:
            u, v = edges[idx]
            if G.has_edge(u, v):          # only absent edges
                continue
            if per_node[u] >= max_virtual or per_node[v] >= max_virtual:
                continue
            G.add_edge(u, v,
                       weight = w_add,
                       w_orig = 0.0,
                       w_extra= w_add)
            per_node[u]+=1; per_node[v]+=1
            added += 1

        # ------------------ objective & bookkeeping ------------------
        obj = float(np.max(bel_c, axis=1).sum())
        hist.append(dict(it=em, obj=obj, added=added))

        if obj > best["obj"]:
            best.update(obj=obj, beliefs=bel_c, node2idx=n2i)
            no_imp = 0
        else:
            no_imp += 1
        if no_imp >= 7:
            break

    hard = best["beliefs"].argmax(1)
    return dict(
        beliefs=best["beliefs"],
        communities=hard,
        node2idx=best["node2idx"],
        idx2node={i:u for u,i in best["node2idx"].items()},
        history=hist,
        G_final=G,                 # in case one wants to inspect weights
    )

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
    a = 30
    b = 5
    n = 1000
    K = 3
    r_in = np.sqrt(a * np.log(n) / n)
    r_out = np.sqrt(b * np.log(n) / n)
    print(f"r_in = {r_in:.4f}, r_out = {r_out:.4f}")
    # G_true = generate_gbm_poisson(lam=50, K=K, a=a, b=b, seed=42)
    G_true = generate_noisy_sbm(
        n=900,
        K=K,
        p_in=0.566,
        p_out=0.196,
        sigma=0.4, 
        seed=42
    )
    print("Generated graph with", len(G_true.nodes()), "nodes and", len(G_true.edges()), "edges")
    for u, v in G_true.edges():
        G_true[u][v]["dist"] = np.linalg.norm(np.array(G_true.nodes[u]["coords"]) - np.array(G_true.nodes[v]["coords"]))
    avg_deg = np.mean([G_true.degree[n] for n in G_true.nodes()])
    print("avg_deg:", avg_deg)
    original_density = avg_deg / len(G_true.nodes)
    C = 0.8 * original_density
    # print("C:", C)
    def weight_func(c1, c2):
        # return np.exp(-0.5 * get_coordinate_distance(c1, c2))
        return 1.0

    num_pairs = int(C * len(G_true.nodes) ** 2 / 2)
    sampler = PairSamplingObservation(G_true, num_samples=num_pairs, weight_func=weight_func, seed=42)
    observations = sampler.observe()

    obs_nodes: Set[int] = set()
    for p, d in observations:
        obs_nodes.add(p[0])
        obs_nodes.add(p[1])

    # subG = create_dist_observed_subgraph(G_true.number_of_nodes(), observations)
    subG = erdos_renyi_mask(G_true, 0.005, seed=42)
    # subG = geometric_censor(G_true, 0.3, coord_key="coords")
    # # print("Running Loopy BP …")
    # beliefs, preds, node2idx, idx2node = belief_propagation(
    #     subG,
    #     q=K,
    #     seed=42,
    #     init="random",
    #     msg_init="random",
    #     max_iter=100000,3
    #     damping=0.15,   
    #     balance_regularization=0.05,
    # )
    
    res = duo_spec(
        G_true,
        K=K,
        config='motif'
    )
    preds = res["communities"]

    # labels = spectral_clustering(subG, q=K, seed=42)
    # preds = np.array([labels[i] for i in range(len(labels))])


    true_labels = get_true_communities(G_true, node2idx=None, attr="comm")
    stats = detection_stats(preds, true_labels)
    # sub_preds = np.array([preds[i] for i in obs_nodes])
    # sub_true_labels = np.array([true_labels[i] for i in obs_nodes])
    # sub_stats = detection_stats(sub_preds, sub_true_labels)
    print("\n=== Community‑detection accuracy ===")
    for k, v in stats.items():
        print(f"{k:>25s} : {v}")
    # print("\n=== Subgraph community‑detection accuracy ===")
    # for k, v in sub_stats.items():
    #     print(f"{k:>25s} : {v}")
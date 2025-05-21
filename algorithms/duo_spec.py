from __future__ import annotations
from collections import defaultdict
from copy import deepcopy
import math
import random

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
from scipy.stats import mode, permutation_test

# NetworkX
import networkx as nx

# Scikit-learn
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KernelDensity

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

    nbr = [[] for _ in range(n_dir)]
    for idx, (u, v) in enumerate(idx2e):
        for w in G.neighbors(v):
            if w == u:
                continue            # non-backtracking
            nbr[idx].append(e2idx[(v, w)])

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
            if js:                     
                x_new[i] = x[js].sum()

        lam_new = norm(x_new)
        if lam_new < 1e-12:       
            x = rng.standard_normal(n_dir)
            x /= norm(x)
            continue            

        x_new /= lam_new
        if abs(lam_new - lam) < tol * lam_new:
            return lam_new     
        x, lam = x_new, lam_new

    return lam     

def _conf_from_center(X, mu):
    """
    Confidence matrix Q[i,c] = 1 / (1 + ||x_i − μ_c||_2)
    Rows are re-normalised to sum to 1.
    """
    dists = np.linalg.norm(X[:, None, :] - mu[None, :, :], axis=-1)  # (n,q)
    Q     = 1.0 / (1.0 + dists)
    Q    /= Q.sum(axis=1, keepdims=True)
    return Q


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
    L = csgraph.laplacian(A, normed=True)

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
    spec_params: dict = {},
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
    # def _lam(step, base, warmup_rounds=0, anneal_steps=50):
    #     if step < warmup_rounds:           # ❶ pure warm-up
    #         return 0.0
    #     d = step - warmup_rounds
    #     if d < anneal_steps:               # ❷ linear ramp
    #         return base * d / anneal_steps
    #     # ❸ harmonic decay after plateau
    #     t = d - anneal_steps
    #     return base / (1 + t)
    # -----------------------------------------------------------------------
    for em in range(1, max_em_iters + 1):
        print(f"[EM] iter {em} / {max_em_iters}")
        # ---------------- community embedding -----------------------------
        Q_comm, hard_comm, *_ = config[0](
            subG, q=K, random_state=random_state, **spec_params
        )
        p_same = _edge_same_prob(Q_comm, iu_glob, iv_glob)
        mask_comm_shrink = p_same > np.percentile(p_same, comm_cut * 100)
        mask_comm_boost  = p_same > np.percentile(p_same, boost_cut_comm * 100)

        # ---------------- geometry embedding ------------------------------
        Q_geo, hard_geo, *_ = config[1](
            subG, q=num_balls, random_state=random_state, **spec_params
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
    K = 4
    r_in = np.sqrt(a * np.log(n) / n)
    r_out = np.sqrt(b * np.log(n) / n)
    print(f"r_in = {r_in:.4f}, r_out = {r_out:.4f}")
    # G_true = generate_gbm_poisson(lam=50, K=K, a=a, b=b, seed=42)
    G_true = generate_noisy_sbm(
        n=900,
        K=2,
        p_in=0.7,
        p_out=0.196,
        sigma=0.5,
        seed=42
    )
    print("Generated graph with", len(G_true.nodes()), "nodes and", len(G_true.edges()), "edges")

    # subG = geometric_censor(G_true, r=0.5, seed=42)
    
    res = duo_spec(
        subG,
        K=2,
        config='bethe_hessian',
    )
    # Q, preds, node2idx, idx2node = dwpe(
    #     subG,
    #     q=2,
    #     random_state=42,
    # )
    preds = res["communities"]


    true_labels = get_true_communities(G_true, node2idx=None, attr="comm")
    stats = detection_stats(preds, true_labels)

    print("\n=== Community‑detection accuracy ===")
    for k, v in stats.items():
        print(f"{k:>25s} : {v}")

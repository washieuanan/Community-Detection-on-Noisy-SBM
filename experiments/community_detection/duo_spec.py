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
from community_detection.bp.vectorized_bp import belief_propagation
from collections import defaultdict
from community_detection.bp.vectorized_bp import spectral_clustering
from copy import deepcopy
from scipy.sparse import diags
from sklearn.cluster import KMeans
from scipy.sparse import csgraph
from scipy.sparse.csgraph import laplacian as cs_lap
from numpy.linalg import norm
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
from typing import Union, Tuple
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.sparse.csgraph import shortest_path
from scipy.sparse.linalg import lobpcg
from scipy.sparse.linalg import eigsh, lobpcg, ArpackNoConvergence, ArpackError

import numpy as np
import networkx as nx
from collections import defaultdict, deque

def build_pairwise_potentials(G, a, b):
    """
    Attach to each edge (u,v) a 2×2 potential matrix Phi_uv 
    so that Phi_uv[k,k] = exp(-dist^2/(2*r_in^2)),
             Phi_uv[k,ℓ] = exp(-dist^2/(2*r_out^2)) for k!=ℓ.
    """
    n = G.number_of_nodes()
    r_in  = np.sqrt(a * np.log(n) / n)
    r_out = np.sqrt(b * np.log(n) / n)
    for u, v, dat in G.edges(data=True):
        d = dat["dist"]
        w_in  = np.exp(-0.5 * (d / r_in)**2)
        w_out = np.exp(-0.5 * (d / r_out)**2)
        # potential as a 2×2 array: index 0,1 for two communities
        dat["Phi"] = np.array([[w_in, w_out],
                               [w_out, w_in]])
def loopy_bp(G, max_iter=20, tol=1e-3):
    """
    Run 2‐state sum‐product BP on G, using per‐edge dat["Phi"] (2×2 potentials).
    Returns marginals: bel[u] = [P(c_u=0), P(c_u=1)].
    """
    nodes = list(G.nodes())
    # 1) Init all directed messages m[u->v] = uniform
    m = {}
    for u, v in G.edges():
        m[(u, v)] = np.ones(2)
        m[(v, u)] = np.ones(2)

    prior = np.array([0.5, 0.5])
    for _ in range(max_iter):
        delta = 0.0
        # 2) For each directed edge u->v, update m[u->v]
        for u, v in list(m.keys()):
            Phi = G[u][v]["Phi"]           # shape (2,2)
            # product of all incoming messages to u except from v
            prod = prior.copy()
            for w in G.neighbors(u):
                if w == v:
                    continue
                prod *= m[(w, u)]
            # compute new message:  m_uv[k] = sum_ℓ Phi[ℓ,k] * prod[ℓ]
            new = Phi.T.dot(prod)
            new_sum = new.sum()
            if new_sum == 0:
                new = np.ones(2)  # avoid division by zero
                new_sum = 2.0
            new /= new_sum
            delta = max(delta, np.max(np.abs(new - m[(u, v)])))
            m[(u, v)] = new
        if delta < tol:
            break

    # 3) Compute node marginals
    bel = {}
    for u in nodes:
        prod = prior.copy()
        for w in G.neighbors(u):
            prod *= m[(w, u)]
        s = prod.sum()
        bel[u] = prod / (s if s else 1.0)
    return bel


def two_core_periphery_vote(G, core_labels, k=2, weight="weight"):
    """
    G : nx.Graph
    core_labels : dict node->label for all nodes in the k-core
    k : core number
    Returns labels dict for ALL nodes reached from the core.
    """
    # 1) find the k-core
    core_nodes = set(nx.k_core(G, k=k).nodes())
    # 2) seed the vote with the core labels
    labels = { u: core_labels[u] for u in core_nodes }
    # 3) BFS‐spread outwards
    q = deque(core_nodes)
    visited = set(core_nodes)
    while q:
        u = q.popleft()
        for v, dat in G[u].items():
            if v in visited:
                continue
            visited.add(v)
            # weight‐majority vote from neighbors already labeled
            votes = defaultdict(float)
            for nbr, dat2 in G[v].items():
                lbl = labels.get(nbr)
                if lbl is not None:
                    votes[lbl] += dat2.get(weight, 1.0)
            # assign the best (or default to 0 if no votes)
            labels[v] = max(votes, key=votes.get) if votes else 0
            q.append(v)
    return labels

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
            if delta < 0:  # flipping lowers “cut” → better clustering
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
    use_nonbacktracking : bool = True,
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
        d_vals = np.array([d["dist"] for _, _, d in H_obs.edges(data=True)])
        sigma  = (np.median(d_vals) or 1.0) * sigma_scale
        for u, v, d in H_obs.edges(data=True):
            d["weight"] = np.exp(-0.5 * (d["dist"] / sigma) ** 2)

    A   = nx.to_scipy_sparse_array(H_obs, nodelist=nodes,
                                   format="csr", weight="weight")
    deg = np.array(A.sum(axis=1)).ravel()
    D   = diags(deg)

    # ---------- choose r ------------------------------------------------------
    # ---------- choose r ----------------  ----------------------------------
    if use_nonbacktracking:
        try:
            rho_nb = _largest_nb_eig(H_obs, nodes, seed=random_state)
            if not np.isfinite(rho_nb) or rho_nb <= 0:
                raise ValueError
            r = np.sqrt(rho_nb)
        except Exception:
            # Fallback: average degree (works even on trees / 1-cores)
            r = np.sqrt(max(deg.mean(), 1e-12))
    else:
        r = np.sqrt(max(deg.mean(), 1e-12))


    # ---------- Bethe–Hessian -------------------------------------------------
    I  = diags(np.ones(n))
    Hr = (r * r - 1.0) * I - r * A + D

    k   = q
    ncv = min(n - 2, max(4 * k + 20, 80))


    vals, vecs = eigsh(-Hr, k=q, which="LA", ncv=ncv, tol=1e-4, maxiter=10000)   # (n,q)

    # ---------- k-means & confidence -----------------------------------------
    km    = KMeans(n_clusters=q, n_init=20, random_state=random_state).fit(vecs)
    hard  = km.labels_
    mu    = km.cluster_centers_
    Q     = _conf_from_center(vecs, mu)

    return Q, hard, node2idx, idx2node


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
# EM driver  – spectral dual
# ---------------------------------------------------------------------------
def duo_spec(
    H_obs            : nx.Graph,
    K                : int,
    num_balls        : int = 16,
    config           : tuple = ('bethe_hessian', 'bethe_hessian'),
    max_em_iters     : int   = 50,
    init_beliefs     : np.ndarray | None = None,
    anneal_steps     : int   = 6,
    warmup_rounds    : int   = 2,
    comm_cut         : float = 0.90,
    geo_cut          : float = 0.90,
    shrink_comm      : float = 1.00,
    shrink_geo       : float = 0.80,
    w_min            : float = 5e-2,
    tol              : float = 1e-4,
    patience         : int   = 7,
    random_state     : int   = 0,
):
    """
    Alternating EM-like refinement driven by the new confidence definition.
    """
    rng             = np.random.default_rng(random_state)
    subG            = deepcopy(H_obs)
    edges           = np.asarray(subG.edges(), dtype=object)
    node2idx_global = {u: i for i, u in enumerate(subG.nodes())}
    iu_global, iv_global = _to_idx(edges, node2idx_global)

    best, hist = {"obj": -np.inf}, []
    no_imp     = 0
    config = get_callable(config)
    def _current_lambda(step, base, ramp):
        if step <= 0:
            return 0.0
        if step >= ramp:
            return base
        return base * step / ramp

    # --------------------------- EM loop -------------------------------------
    for em in range(1, max_em_iters + 1):

        # ------------------------ community step ------------------------------
        # Q_comm, hard_comm, *_ = config[0](
        #     subG, q=K, random_state=random_state
        # )

                # ---------------------- init override --------------------------
        if em == 1 and init_beliefs is not None:
            Q_comm   = init_beliefs.copy()
            hard_comm = Q_comm.argmax(axis=1)
        else:
            # ---------------- community step ---------------------------
            Q_comm, hard_comm, *_ = config[0](
                subG, q=K, random_state=random_state
            )

        p_same = _edge_same_prob(Q_comm, iu_global, iv_global)
        mask_comm   = p_same > np.percentile(p_same, comm_cut * 100)

        # ------------------------ geometry step -------------------------------
        Q_geo, hard_geo, *_ = config[1](
            subG, q=num_balls, random_state=random_state
        )
        same_ball  = hard_geo[iu_global] == hard_geo[iv_global]
        conf_g     = 0.5 * (Q_geo[iu_global, hard_geo[iu_global]] +
                            Q_geo[iv_global, hard_geo[iv_global]])
        mask_geo = same_ball & (
            conf_g > np.percentile(conf_g, geo_cut * 100)
        )

        # ------------------------ shrink / prune ------------------------------
        λ_comm = _current_lambda(em - warmup_rounds, shrink_comm, anneal_steps)
        λ_geo  = _current_lambda(em - warmup_rounds, shrink_geo , anneal_steps)

        n_drop_comm = n_drop_geo = 0
        if em > warmup_rounds:
            n_drop_comm = _scale_or_prune(subG, mask_comm, p_same, λ_comm, w_min)
            n_drop_geo  = _scale_or_prune(subG, mask_geo , np.ones_like(mask_geo),
                                          λ_geo , w_min)

        # ------------------------ objective & logging -------------------------
        obj = float(np.max(Q_comm, axis=1).sum())   # higher is better
        hist.append(dict(
            iter=em, obj=obj, edges=subG.number_of_edges(),
            drop_comm=n_drop_comm, drop_geo=n_drop_geo,
            λ_comm=λ_comm, λ_geo=λ_geo,
        ))

        if obj > best["obj"]:
            best.update(obj=obj, beliefs=Q_comm, balls=hard_geo,
                        node2idx=node2idx_global)
            no_imp = 0
        else:
            no_imp += 1

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
        beliefs     = best["beliefs"],
        communities = hard_final,
        balls       = best["balls"],
        node2idx    = best["node2idx"],
        idx2node    = {i: u for u, i in best["node2idx"].items()},
        history     = hist,
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
    from graph_generation.gbm import generate_gbm_poisson, generate_gbm
    from observations.standard_observe import PairSamplingObservation, get_coordinate_distance
    from community_detection.bp.vectorized_bp import belief_propagation, beta_param
    from sklearn.manifold import MDS
    from sklearn.cluster import KMeans
    a = 35
    b = 8
    n = 150
    K = 2
    r_in = np.sqrt(a * np.log(n) / n)
    r_out = np.sqrt(b * np.log(n) / n)
    print(f"r_in = {r_in:.4f}, r_out = {r_out:.4f}")
    # G_true = generate_gbm_poisson(lam=50, K=K, a=a, b=b, seed=42)
    G_true = generate_gbm(
        n=750,
        K=K,
        a=a,
        b=b,
        seed=42
    )
    print("Generated graph with", len(G_true.nodes()), "nodes and", len(G_true.edges()), "edges")
    for u, v in G_true.edges():
        G_true[u][v]["dist"] = np.linalg.norm(np.array(G_true.nodes[u]["coords"]) - np.array(G_true.nodes[v]["coords"]))
    avg_deg = np.mean([G_true.degree[n] for n in G_true.nodes()])
    print("avg_deg:", avg_deg)
    original_density = avg_deg / len(G_true.nodes)
    C = 0.1 * original_density
    # print("C:", C)
    def weight_func(c1, c2):
        # return np.exp(-0.5 * get_coordinate_distance(c1, c2))
        return 1.0

    num_pairs = int(C * len(G_true.nodes) ** 2 / 2)
    sampler = PairSamplingObservation(G_true, num_samples=num_pairs, weight_func=weight_func, seed=42)
    observations = sampler.observe()

    # obs_nodes: Set[int] = set()
    # for p, d in observations:
    #     obs_nodes.add(p[0])
    #     obs_nodes.add(p[1])

    subG = create_dist_observed_subgraph(G_true.number_of_nodes(), observations)
    # add_gaussian_weights_from_dist(subG)
    # add_localscale_weights(subG, 2)
    # renormalise_for_sampling(subG, C)
    add_jaccard_edges(subG, frac_keep=0.02)     #  ←  NEW
    # augment_knn_via_sp(subG, k=2)
    # labels = spectral_clustering(G_true, q=K, seed=42)
    # preds = np.array([labels[i] for i in range(len(labels))])
    # print("Running Loopy BP …")
    # beliefs, preds, node2idx, idx2node = belief_propagation(
    #     subG,
    #     q=K,
    #     seed=42,
    #     init="random",
    #     msg_init="random",
    #     max_iter=100000,
    #     damping=0.15,   
    #     balance_regularization=0.05,
    # )
    # _, labels, _, _ = gbm_em(
    #     subG,
    #     k=K,
    #     dim=3,
    #     epochs=100,
    #     lr_x=0.05,
    #     inner_grad_steps=500,
    #     eps=1e-9
    # )
    # labels = spectral_clustering(subG, q=K, seed=42)
    # preds = np.array([labels[i] for i in range(len(labels))])
    # res = em_geometry_community(
    #     subG,
    #     K=K,
    #     bp_kwargs=dict(
    #         init = "random",
    #         max_iter=1000,             # Increase max iterations
    #         damping=0.4,                # Start with moderate damping 
    #         balance_regularization=0.2, # Increase balance regularization
    #         min_steps=70,               # Minimum steps before early convergence
    #     ),
    #     num_balls=32,                   # Reduced from 20 for better stability
    #     max_em_iters=100,
    #     # conf_threshold=0.75,             # Lower threshold for less aggressive pruning
    #     tol=1e-6,
    #     seed=42,
    #     # ball_weight=0.55,                # Reduced ball pruning influence
    #     # adaptive_min_edges=True,        # Adaptively decrease min_edges
    #     # early_stopping_window=5,        # Stop if no improvement for 5 iterations
    # )
    # res = duo_bp(
    #     subG,
    #     K=K,
    #     num_balls=32,
    # )

    build_pairwise_potentials(subG, a, b)

    # --- 1) Build the same node2idx that duo_spec will use internally ----
    nodes    = list(subG.nodes())               # networkx preserves insertion order
    node2idx = { u:i for i,u in enumerate(nodes) }
    n        = len(nodes)

    # --- 2) Run BP to get marginals bel[u] = [P(c_u=0), P(c_u=1)] --------
    bel = loopy_bp(subG, max_iter=30)            # from the BP helper above

    # --- 3) Fill init_beliefs using your precomputed node2idx ----------
    K = 2
    init_beliefs = np.zeros((n, K), float)
    for u, bvec in bel.items():
        init_beliefs[node2idx[u]] = bvec
    result = duo_spec(
        subG,
        K=K,
        init_beliefs=init_beliefs,
        num_balls=32,
        config= ("score", "bethe_hessian"),
        random_state=42
    )
    preds = result["communities"]
    hard     = result["communities"]     # list of length n
    idx2node = result["idx2node"]        # dict: index -> node_id

    # build original node->label map
    orig_labels = { node: hard[i] for i, node in idx2node.items() }

    # isolate core labels
    core_nodes  = set(nx.k_core(subG, k=2).nodes())
    core_labels = { u: orig_labels[u] for u in core_nodes }

    # run the periphery voting
    full_labels = two_core_periphery_vote(subG, core_labels, k=2)

    # fallback for any node never reached by BFS (e.g. isolated)
    for u in subG.nodes():
        if u not in full_labels:
            full_labels[u] = orig_labels[u]

    # rebuild the final hard assignment array in index order
    hard_refined = np.array([ full_labels[idx2node[i]] for i in range(len(hard)) ])
    true_labels = get_true_communities(G_true, node2idx=None, attr="comm")
    stats = detection_stats(hard_refined, true_labels)
    # sub_preds = np.array([preds[i] for i in obs_nodes])
    # sub_true_labels = np.array([true_labels[i] for i in obs_nodes])
    # sub_stats = detection_stats(sub_preds, sub_true_labels)
    print("\n=== Community‑detection accuracy ===")
    for k, v in stats.items():
        print(f"{k:>25s} : {v}")
    # print("\n=== Subgraph community‑detection accuracy ===")
    # for k, v in sub_stats.items():
    #     print(f"{k:>25s} : {v}")
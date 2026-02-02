from __future__ import annotations
from typing import Dict, List, Tuple, Literal
import networkx as nx
import numpy as np
import scipy.sparse as sp

from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.optimize import linear_sum_assignment
from scipy.stats import permutation_test, mode

from scipy.sparse.linalg import eigsh


# -----------------------------------------------------------------------------
#  Graph → contiguous numpy arrays
# -----------------------------------------------------------------------------

def build_arrays(G: nx.Graph):
    node2idx = {u: i for i, u in enumerate(G)}
    idx2node = {i: u for u, i in node2idx.items()}
    m = G.number_of_edges()
    src = np.empty(2 * m, np.int32)
    dst = np.empty(2 * m, np.int32)
    k = 0
    for u, v in G.edges():
        ui, vi = node2idx[u], node2idx[v]
        src[k], dst[k] = ui, vi
        src[k + 1], dst[k + 1] = vi, ui
        k += 2
    rev = np.empty_like(src)
    rev[0::2] = 1 + np.arange(0, 2 * m, 2)
    rev[1::2] = 0 + np.arange(0, 2 * m, 2)
    return node2idx, idx2node, src, dst, rev

# -----------------------------------------------------------------------------
#  Spectral initialisation helpers
# -----------------------------------------------------------------------------
def spectral_clustering(G: nx.Graph, q: int, *, seed: int = 0):
    # 1) build normalized Laplacian
    A = nx.adjacency_matrix(G).astype(np.float64)
    n = A.shape[0]
    degrees = np.ravel(A.sum(axis=1))
    inv_sqrt = np.where(degrees>0, 1.0/np.sqrt(degrees), 0.0)
    D_inv_sqrt = sp.diags(inv_sqrt)
    L = sp.eye(n, format="csr") - D_inv_sqrt @ A @ D_inv_sqrt
    # symmetrize to avoid tiny nonsymmetric noise
    L = (L + L.T) * 0.5

    # 2) try ARPACK for the smallest q+1 eigenpairs
    k = q + 1
    ncv = min(n - 1, max(2*k + 1, k + 20))
    try:
        vals, vecs = eigsh(
            L,
            k=k,
            which="SM",
            tol=1e-4,
            ncv=ncv,
            maxiter=1000
        )
        # drop the first (trivial) eigenvector
        idx = np.argsort(vals)
        vecs = vecs[:, idx[1 : q+1]]
    except ArpackNoConvergence:
        # fallback to LOBPCG for the q smallest eigenvectors
        X0 = np.random.RandomState(seed).randn(n, q)
        _, vecs = lobpcg(
            L,
            X0,
            tol=1e-4,
            maxiter=200,
            largest=False
        )

    # 3) cluster & return
    km = KMeans(n_clusters=q, random_state=seed).fit(np.real(vecs))
    return {node: int(label) for node, label in zip(G.nodes(), km.labels_)}


def init_beliefs(n: int, q: int, rng, labels=None, node2idx=None, bias: float = 0.2):
    B = rng.dirichlet(np.ones(q), size=n)
    if labels and node2idx:
        for u, lbl in labels.items():
            B[node2idx[u], lbl] += bias
        B /= B.sum(1)[:, None]
    return B.astype(np.float64)

# -----------------------------------------------------------------------------
#  Message initialisation (random / copy / pre‑group)
# -----------------------------------------------------------------------------

def init_messages(
    q: int,
    src: np.ndarray,
    dst: np.ndarray,
    *,
    method: Literal["random", "copy", "pre-group"],
    rng: np.random.Generator,
    beliefs: np.ndarray,
    spec_arr: np.ndarray,
    node2idx: Dict[int, int],
    group_obs: List | None = None,
    min_sep: float | None = None,
    eps: float = 0.1,
):
    """Return (2m,q) array of initial messages."""
    m = src.size // 2
    M = rng.dirichlet(np.ones(q), size=2 * m) + 1e-3

    if method == "random":
        M[np.arange(2 * m), spec_arr[src]] += eps
        M /= M.sum(1)[:, None]

    elif method == "copy":
        M[:] = beliefs[src]
        M[np.arange(2 * m), spec_arr[src]] += eps
        M /= M.sum(1)[:, None]

    elif method == "pre-group":
        if group_obs is None:
            raise ValueError("group_obs must be provided for pre‑group init")

        # edge id map uses *indices* (not raw node labels)
        edge_id = {(src[i], dst[i]): i for i in range(2 * m)}
        base_bias = np.sqrt(min_sep if min_sep is not None else 0.15)

        # assign a dominant spectral label to each group
        bias_assign = np.empty(len(group_obs), dtype=int)
        for g, obs in enumerate(group_obs):
            idx_vertices: List[int] = []
            if isinstance(obs, dict):
                edge_lists = obs.values()
            else:
                edge_lists = [obs]
            for edges in edge_lists:
                idx_vertices += [node2idx[u] for u, _ in edges]
            bias_assign[g] = mode(spec_arr[idx_vertices])[0][0] if idx_vertices else -1

        # inject bias into messages for edges in each group
        for g, obs in enumerate(group_obs):
            t = bias_assign[g]
            if t == -1:
                continue
            if isinstance(obs, dict):
                items = obs.items()
            else:
                items = [(None, obs)]  # type: ignore
            for rad, edges in items:
                extra = 0.0
                if rad is not None:
                    extra = max(-0.2 * np.exp(float(rad)), -base_bias)
                for u_raw, v_raw in edges:
                    ui = node2idx[u_raw]
                    vi = node2idx[v_raw]
                    for a, b in ((ui, vi), (vi, ui)):
                        e = edge_id.get((a, b))
                        if e is None:
                            continue
                        M[e, t] += base_bias + extra
                        M[e] /= M[e].sum()
    else:
        raise ValueError("unknown message init method")

    return M.astype(np.float64)

# -----------------------------------------------------------------------------
#  Initial prior helper (for imbalance robustness)
# -----------------------------------------------------------------------------

def _compute_pi0_from_labels(labels: np.ndarray, q: int, *, alpha: float = 2.0, floor: float = 1e-4) -> np.ndarray:
    """Compute smoothed class prior from spectral labels."""
    # labels shape (n,), integer in [0,q)
    # returns pi0 shape (q,)
    counts = np.bincount(labels, minlength=q).astype(np.float64)
    pi0 = (counts + alpha) / (counts.sum() + q * alpha)
    pi0 = np.maximum(pi0, floor)
    pi0 = pi0 / pi0.sum()
    return pi0

# -----------------------------------------------------------------------------
#  β parameter (Zhang et al. 2014‑style)
# -----------------------------------------------------------------------------

def beta_param(G, q):
    d = np.fromiter((deg for _, deg in G.degree()), float)
    a = d.mean(); eps = 1e-3
    return np.log((q * (1 + (q - 1) * eps)) / (max(a * (1 - eps) - (1 + (q - 1) * eps), 1e-10)) + 1) * 1.2

# -----------------------------------------------------------------------------
#  Main BP routine
# -----------------------------------------------------------------------------

def belief_propagation(
    G: nx.Graph,
    q: int,
    *,
    beta: float | None = None,
    max_iter: int = 1000,
    tol: float = 1e-4,
    damping: float = 0.20,
    balance_regularization: float = 0.10,
    seed: int = 0,
    min_steps: int = 0,
    init: Literal["random", "spectral"] = "random",
    msg_init: Literal["random", "copy", "pre-group"] = "random",
    group_obs: List | None = None,
    min_sep: float | None = None,
    eps: float = 0.1,
):
    """Vectorised BP that reproduces the exact math/logic of the reference loop."""

    rng = np.random.default_rng(seed)
    
    # Check if the graph has enough edges to run BP
    if G.number_of_edges() < 1:
        print("[BP] Warning: Graph has no edges, returning random beliefs")
        node2idx = {u: i for i, u in enumerate(G)}
        idx2node = {i: u for u, i in node2idx.items()}
        n = len(node2idx)
        beliefs = init_beliefs(n, q, rng)
        preds = beliefs.argmax(1)
        return beliefs, preds, node2idx, idx2node

    # ---------------------------------------------------------------------
    #  Pre‑compute arrays & constants
    # ---------------------------------------------------------------------
    node2idx, idx2node, src, dst, rev = build_arrays(G)
    n, m = len(node2idx), src.size // 2
    deg = np.fromiter((G.degree[u] for u in G), int)

    if beta is None:
        beta = beta_param(G, q) * 1.1  # match reference scaling
    exp_beta = np.exp(beta)

    # ---------------------------------------------------------------------
    #  Initial beliefs & messages
    # ---------------------------------------------------------------------
    spectral_labels = spectral_clustering(G, q, seed=seed) if init == "spectral" else {}
    beliefs = init_beliefs(n, q, rng, labels=spectral_labels, node2idx=node2idx)
    spec_arr = np.array([spectral_labels.get(idx2node[i], 0) for i in range(n)], int)

    # Init prior + soften beliefs/messages (only for spectral init, q>1)
    pi0 = None
    if init == "spectral" and q > 1 and len(spectral_labels) > 0:
        tiny = 1e-12
        alpha_pi0 = 2.0       # stronger smoothing for imbalance
        pi0_floor = 1e-4
        eta_belief = 0.08     # how much to mix pi0 into beliefs (works for q>2)
        
        # Clamp spec_arr to valid range
        spec_arr = np.asarray(spec_arr, dtype=int)
        spec_arr = np.mod(spec_arr, q)
        
        # Compute pi0
        pi0 = _compute_pi0_from_labels(spec_arr, q, alpha=alpha_pi0, floor=pi0_floor)
        
        # Soften initial beliefs toward pi0
        beliefs = beliefs.astype(np.float64, copy=False)
        beliefs = np.maximum(beliefs, tiny)
        beliefs /= beliefs.sum(1, keepdims=True).clip(tiny)
        beliefs[:] = (1.0 - eta_belief) * beliefs + eta_belief * pi0[None, :]
        beliefs = np.maximum(beliefs, tiny)
        beliefs /= beliefs.sum(1, keepdims=True).clip(tiny)

    messages_old = init_messages(
        q, src, dst,
        method=msg_init,
        rng=rng,
        beliefs=beliefs,
        spec_arr=spec_arr,
        node2idx=node2idx,
        group_obs=group_obs,
        min_sep=min_sep,
        eps=eps,
    )
    
    # Soften initial messages toward pi0 (if spectral init was used)
    if init == "spectral" and q > 1 and len(spectral_labels) > 0 and pi0 is not None:
        tiny = 1e-12
        eta_msg = 0.06        # message softening amount
        messages_old = messages_old.astype(np.float64, copy=False)
        messages_old = np.maximum(messages_old, tiny)
        messages_old /= messages_old.sum(1, keepdims=True).clip(tiny)
        messages_old[:] = (1.0 - eta_msg) * messages_old + eta_msg * pi0[None, :]
        messages_old = np.maximum(messages_old, tiny)
        messages_old /= messages_old.sum(1, keepdims=True).clip(tiny)
    
    messages = np.empty_like(messages_old)

    # Scratch arrays
    S = np.empty((n, q))
    convergence_history: List[float] = []

    for it in range(max_iter):
        # --------------------------------------------------------------
        #  Belief update  (matches reference inner loops)
        # --------------------------------------------------------------
        edge_fac = 1.0 + (exp_beta - 1.0) * messages_old.clip(1e-10)
        log_fac = np.log(edge_fac)
        S.fill(0.0)
        np.add.at(S, dst, log_fac)
        # Compute beliefs in log space then exp
        log_beliefs = S - S.max(1)[:, None]
        beliefs[:] = np.exp(log_beliefs - np.log(np.exp(log_beliefs).sum(1)[:, None] + 1e-10))

        # --------------------------------------------------------------
        #  Community sizes & theta (same formulas)
        # --------------------------------------------------------------
        comm_sz = beliefs.mean(0).clip(1e-10)              # community_sizes with clipping
        theta = (deg[:, None] * beliefs).sum(0).clip(1e-10)  # theta with clipping

        # --------------------------------------------------------------
        #  Message update  (vectorised reference equation)
        # --------------------------------------------------------------
        # Safe handling of zero edge case
        if m > 0:
            # Compute messages in log space for numerical stability
            log_messages = (
                -beta * deg[src, None] * theta / (2.0 * m) +   # term1
                S[src] -                                       # Σ over neighbours except dst
                log_fac[rev] -                                 # subtract k→i contribution
                balance_regularization * np.log(comm_sz)       # size_penalty
            )
            # Subtract max for numerical stability
            log_max = log_messages.max(1)[:, None]
            messages_new = np.exp(log_messages - log_max)
            messages_new /= messages_new.sum(1)[:, None].clip(1e-10)
        else:
            # If m=0, provide a fallback to prevent division by zero
            messages_new = np.ones_like(messages_old)
            messages_new /= messages_new.sum(1)[:, None].clip(1e-10)

        # Damp
        messages[:] = (1.0 - damping) * messages_new + damping * messages_old

        # --------------------------------------------------------------
        #  Convergence check & optional entropy‑based noise reinjection
        # --------------------------------------------------------------
        if messages.size == 0:  # Safeguard against empty message arrays
            print("[BP] Warning: Empty message arrays detected, aborting loop")
            delta = 0.0
            break
        
        # Protected maximum calculation
        try:
            delta = np.max(np.abs(messages - messages_old))
            if np.isnan(delta):
                print("[BP] Warning: NaN values detected, reducing learning rate")
                damping = min(damping * 1.5, 0.9)  # Increase damping
                messages[:] = messages_old  # Revert to previous state
                continue
        except ValueError as e:
            if "zero-size array" in str(e):
                print("[BP] Warning: Zero-size array in delta calculation, aborting loop")
                delta = 0.0
                break
            else:
                raise
                
        convergence_history.append(float(delta))

        if delta < tol and it >= min_steps:
            entropy = -np.sum(comm_sz * np.log(comm_sz + 1e-10))
            entropy_ratio = entropy / (-np.log(1.0 / q))
            if entropy_ratio > 0.7:
                # Converged with sufficiently mixed communities
                print(f"[BP] converged in {it+1} iterations; entropy ratio={entropy_ratio:.3f}")
                break
            # Otherwise inject noise as in reference
            noise = rng.random(messages.shape) * 0.15 / (comm_sz + 1e-10)
            messages[:] = messages * 0.85 + noise
            messages /= messages.sum(1)[:, None]

        messages_old, messages = messages, messages_old  # swap buffers
    else:
        print(f"[BP] did not converge within {max_iter} iterations (Δ={delta:.2e})")

    preds = beliefs.argmax(1)
    return beliefs, preds, node2idx, idx2node

def estimate_pi_from_duospec_weights(
    G: nx.Graph,
    q: int,
    *,
    seed: int = 0,
    weight_attr: str = "weight",
    alpha: float = 2.0,    # Dirichlet smoothing strength
    floor: float = 1e-4
) -> np.ndarray:
    """
    Estimate class proportions pi from weighted spectral clustering labels.

    pi_k ∝ count_k + alpha  (then normalized), with floor for safety.
    """
    labels_dict = weighted_spectral_clustering_nx(
        G, q, seed=seed, weight_attr=weight_attr, ensure_nonneg=True, embed="laplacian"
    )
    if len(labels_dict) == 0:
        return np.full(q, 1.0 / q, dtype=np.float64)

    nodes = list(G.nodes())
    z = np.array([labels_dict.get(u, 0) for u in nodes], dtype=int) % q
    counts = np.bincount(z, minlength=q).astype(np.float64)

    pi = counts + alpha
    pi = np.maximum(pi, floor)
    pi = pi / pi.sum()
    return pi


def _kmeans_lloyd(X: np.ndarray, k: int, seed: int = 0, n_iter: int = 50) -> np.ndarray:
    """
    Lightweight k-means (Lloyd). Used if sklearn isn't available.
    Returns labels shape (n,).
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape
    # init centers by random points
    centers = X[rng.choice(n, size=k, replace=False)].copy()

    labels = np.zeros(n, dtype=int)
    for _ in range(n_iter):
        # assign
        dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = dists.argmin(axis=1)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        # update
        for j in range(k):
            mask = labels == j
            if mask.any():
                centers[j] = X[mask].mean(axis=0)
            else:
                centers[j] = X[rng.integers(0, n)]
    return labels

def weighted_spectral_clustering_nx(
    G: nx.Graph,
    q: int,
    *,
    seed: int = 0,
    weight_attr: str = "weight",
    ensure_nonneg: bool = True,
    embed: str = "laplacian",   # "laplacian" recommended
):
    """
    Weighted spectral clustering on a NetworkX graph using edge attribute `weight_attr`.

    - Builds weighted adjacency A (symmetric).
    - Computes normalized Laplacian embedding (smallest eigenvectors).
    - Row-normalizes embedding.
    - KMeans -> labels.

    Returns: dict {node: label}
    """
    if G.number_of_nodes() == 0:
        return {}

    nodes = list(G.nodes())
    node2idx = {u: i for i, u in enumerate(nodes)}
    n = len(nodes)

    # Collect weights
    w_list = []
    for u, v, data in G.edges(data=True):
        w = float(data.get(weight_attr, 1.0))
        w_list.append(w)

    if len(w_list) == 0:
        # no edges -> arbitrary labels
        rng = np.random.default_rng(seed)
        labels = rng.integers(0, q, size=n)
        return {nodes[i]: int(labels[i]) for i in range(n)}

    w_arr = np.asarray(w_list, dtype=np.float64)
    if ensure_nonneg:
        # shift weights so min is 0, then clip tiny
        w_min = float(w_arr.min())
        shift = -w_min if w_min < 0 else 0.0
    else:
        shift = 0.0

    # Build adjacency (dense by default; if you need sparse, I can convert this)
    A = np.zeros((n, n), dtype=np.float64)
    for u, v, data in G.edges(data=True):
        i, j = node2idx[u], node2idx[v]
        w = float(data.get(weight_attr, 1.0)) + shift
        if ensure_nonneg:
            w = max(w, 0.0)
        A[i, j] += w
        A[j, i] += w

    # If graph has isolated nodes, add tiny self-loop to avoid D^{-1/2} blowups
    deg = A.sum(axis=1)
    tiny = 1e-12
    deg_safe = np.maximum(deg, tiny)

    if embed == "laplacian":
        # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
        D_inv_sqrt = 1.0 / np.sqrt(deg_safe)
        M = (D_inv_sqrt[:, None] * A) * D_inv_sqrt[None, :]
        L = np.eye(n) - M

        # Smallest eigenvectors of L
        # For dense fallback, use eigh
        evals, evecs = np.linalg.eigh(L)
        idx = np.argsort(evals)[:q]
        X = evecs[:, idx]
    else:
        # Adjacency embedding: top eigenvectors of normalized adjacency
        D_inv_sqrt = 1.0 / np.sqrt(deg_safe)
        M = (D_inv_sqrt[:, None] * A) * D_inv_sqrt[None, :]
        evals, evecs = np.linalg.eigh(M)
        idx = np.argsort(evals)[::-1][:q]
        X = evecs[:, idx]

    # Row-normalize embedding (standard spectral clustering step)
    row_norm = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.maximum(row_norm, tiny)

    # KMeans
    labels = None
    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=q, n_init=10, random_state=seed)
        labels = km.fit_predict(X)
    except Exception:
        labels = _kmeans_lloyd(X, q, seed=seed, n_iter=60)

    return {nodes[i]: int(labels[i]) for i in range(n)}

# def belief_propagation_weighted(
#     G: nx.Graph,
#     q: int,
#     *,
#     beta: float | None = None,
#     max_iter: int = 1000,
#     tol: float = 1e-4,
#     damping: float = 0.20,
#     balance_regularization: float = 0.10,
#     seed: int = 0,
#     min_steps: int = 0,
#     init: Literal["random", "spectral"] = "random",
#     msg_init: Literal["random", "copy", "pre-group"] = "random",
#     group_obs: List | None = None,
#     min_sep: float | None = None,
#     eps: float = 0.1,
#     # ---------------- NEW (optional) ----------------
#     pi: np.ndarray | None = None,          # expected class proportions, shape (q,)
#     comm_blend: float = 0.05,              # blend comm_sz with pi for stability, esp. imbalance
#     exp_clip: float = 50.0,                # prevents overflow in exp(beta*w)
# ):
#     """
#     Vectorised BP with per-edge 'weight' scaling the pairwise compatibility,
#     generalized to q>2 and class-imbalance via a prior-aware balance term.

#     If pi is None: defaults to uniform (recovers old 'push-to-uniform' behavior).
#     If pi is provided: balance term encourages comm_sz to match pi, not uniform.
#     """
#     rng = np.random.default_rng(seed)
#     if pi is None and q > 1:
#         pi = estimate_pi_from_duospec_weights(G, q, seed=seed, weight_attr="weight")
#     else:
#         pi = np.asarray(pi, dtype=np.float64) if pi is not None else np.full(q, 1.0 / q)
#         pi = np.maximum(pi, 1e-12)
#         pi = pi / pi.sum()

#     if G.number_of_edges() < 1:
#         print("[BP] Warning: Graph has no edges, returning random beliefs")
#         node2idx = {u: i for i, u in enumerate(G)}
#         idx2node = {i: u for u, i in node2idx.items()}
#         n = len(node2idx)
#         beliefs = init_beliefs(n, q, rng)
#         preds = beliefs.argmax(1)
#         return beliefs, preds, node2idx, idx2node

#     # ---------------------------------------------------------------------
#     #  Pre-compute arrays & constants
#     # ---------------------------------------------------------------------
#     node2idx, idx2node, src, dst, rev = build_arrays(G)
#     n, m = len(node2idx), src.size // 2

#     # IMPORTANT: make deg aligned to node indices (robust to any graph iteration order)
#     deg = np.array([G.degree[idx2node[i]] for i in range(n)], dtype=np.float64)

#     # Directed-edge weights array (one per directed edge)
#     w = np.array(
#         [G.edges[idx2node[src[i]], idx2node[dst[i]]].get("weight", 1.0) for i in range(src.size)],
#         dtype=np.float64,
#     )

#     # Robust weight normalization: map to roughly [0, 2] with mean ~1, preserve ranking.
#     # (Keeps your "effective weight behavior", avoids extreme exponentials.)
#     w_mean = float(w.mean())
#     w_std = float(w.std())
#     if w_std < 1e-12:
#         w = np.ones_like(w)
#     else:
#         w = (w - w_mean) / w_std
#         w = (w / np.max(np.abs(w))) + 1.0  # now in [0, 2] approximately

#     if beta is None:
#         beta = beta_param(G, q) * 1.2

#     # ---------------------------------------------------------------------
#     #  Initial beliefs & messages
#     # ---------------------------------------------------------------------
#     spectral_labels = spectral_clustering(G, q, seed=seed) if init == "spectral" else {}
#     beliefs = init_beliefs(n, q, rng, labels=spectral_labels, node2idx=node2idx)
#     spec_arr = np.array([spectral_labels.get(idx2node[i], 0) for i in range(n)], dtype=int)
#     spec_arr = np.mod(spec_arr, q)

#     # Prior (pi0) for init-softening:
#     # - If spectral init is used: estimate pi0 from spec labels (smoothed)
#     # - Else: use provided pi (or uniform if pi None)
#     pi0 = None
#     if init == "spectral" and q > 1 and len(spectral_labels) > 0:
#         alpha_pi0 = 2.0
#         pi0_floor = 1e-4
#         pi0 = _compute_pi0_from_labels(spec_arr, q, alpha=alpha_pi0, floor=pi0_floor)
#     else:
#         pi0 = pi.copy()

#     # Soften initial beliefs toward pi0 (helps stability for q>2 and imbalance)
#     if pi0 is not None and q > 1:
#         tiny = 1e-12
#         eta_belief = 0.08
#         beliefs = beliefs.astype(np.float64, copy=False)
#         beliefs = np.maximum(beliefs, tiny)
#         beliefs /= beliefs.sum(1, keepdims=True).clip(tiny)
#         beliefs[:] = (1.0 - eta_belief) * beliefs + eta_belief * pi0[None, :]
#         beliefs = np.maximum(beliefs, tiny)
#         beliefs /= beliefs.sum(1, keepdims=True).clip(tiny)

#     messages_old = init_messages(
#         q, src, dst,
#         method=msg_init,
#         rng=rng,
#         beliefs=beliefs,
#         spec_arr=spec_arr,
#         node2idx=node2idx,
#         group_obs=group_obs,
#         min_sep=min_sep,
#         eps=eps,
#     )

#     # Soften initial messages toward pi0 as well
#     if pi0 is not None and q > 1:
#         tiny = 1e-12
#         eta_msg = 0.06
#         messages_old = messages_old.astype(np.float64, copy=False)
#         messages_old = np.maximum(messages_old, tiny)
#         messages_old /= messages_old.sum(1, keepdims=True).clip(tiny)
#         messages_old[:] = (1.0 - eta_msg) * messages_old + eta_msg * pi0[None, :]
#         messages_old = np.maximum(messages_old, tiny)
#         messages_old /= messages_old.sum(1, keepdims=True).clip(tiny)

#     messages = np.empty_like(messages_old)
#     S = np.empty((n, q), dtype=np.float64)
#     convergence_history: List[float] = []

#     for it in range(max_iter):
#         # --------------------------------------------------------------
#         #  Belief update
#         # --------------------------------------------------------------
#         # Weight-scaled compatibility: exp(beta * w_e)
#         # Clip to avoid overflow/underflow on difficult graphs.
#         bw = np.clip(beta * w, -exp_clip, exp_clip)
#         exp_beta_w = np.exp(bw)

#         edge_fac = 1.0 + (exp_beta_w[:, None] - 1.0) * messages_old.clip(1e-10)

#         log_fac = np.log(edge_fac)
#         S.fill(0.0)
#         np.add.at(S, dst, log_fac)

#         log_beliefs = S - S.max(1)[:, None]
#         beliefs[:] = np.exp(
#             log_beliefs - np.log(np.exp(log_beliefs).sum(1)[:, None] + 1e-10)
#         )

#         # --------------------------------------------------------------
#         #  Community size estimate (stabilized + prior-aware)
#         # --------------------------------------------------------------
#         comm_emp = beliefs.mean(0).clip(1e-12)        # empirical mean
#         comm_sz = ((1.0 - comm_blend) * comm_emp + comm_blend * pi).clip(1e-12)

#         theta = (deg[:, None] * beliefs).sum(0).clip(1e-12)

#         # --------------------------------------------------------------
#         #  Message update
#         # --------------------------------------------------------------
#         if m > 0:
#             # Prior-aware balance term:
#             # - Old:  -bal * log(comm_sz)           (pushes toward uniform)
#             # - New:  -bal * log(comm_sz / pi)     (pushes toward *pi*)
#             # If pi is uniform -> same up to additive constant.
#             balance_term = - balance_regularization * np.log(comm_sz / pi)

#             log_messages = (
#                 -beta * deg[src, None] * theta / (2.0 * m)
#                 + S[src]
#                 - log_fac[rev]
#                 + balance_term[None, :]
#             )

#             log_max = log_messages.max(1)[:, None]
#             messages_new = np.exp(log_messages - log_max)
#             messages_new /= messages_new.sum(1)[:, None].clip(1e-12)
#         else:
#             messages_new = np.ones_like(messages_old)
#             messages_new /= messages_new.sum(1)[:, None].clip(1e-12)

#         messages[:] = (1.0 - damping) * messages_new + damping * messages_old

#         # --------------------------------------------------------------
#         #  Convergence check & optional entropy-based noise reinjection
#         # --------------------------------------------------------------
#         if messages.size == 0:
#             print("[BP] Warning: Empty message arrays detected, aborting loop")
#             delta = 0.0
#             break

#         delta = float(np.max(np.abs(messages - messages_old)))
#         if not np.isfinite(delta):
#             damping = min(damping * 1.5, 0.9)
#             messages[:] = messages_old
#             continue

#         convergence_history.append(delta)

#         if delta < tol and it >= min_steps:
#             # Use entropy relative to uniform as a "collapsed" detector; works for any q.
#             entropy = -np.sum(comm_sz * np.log(comm_sz + 1e-12))
#             entropy_ratio = entropy / (np.log(q) + 1e-12)

#             # If it's not collapsed, accept convergence.
#             if entropy_ratio > 0.70:
#                 print(f"[BP] converged in {it+1} iterations; entropy ratio={entropy_ratio:.3f}")
#                 break

#             # If collapsed, inject mild, prior-shaped noise (NOT uniform noise).
#             noise = rng.random(messages.shape) * 0.15
#             noise = noise * (pi[None, :] / (pi.max() + 1e-12))
#             messages[:] = messages * 0.85 + noise
#             messages /= messages.sum(1)[:, None].clip(1e-12)

#         messages_old, messages = messages, messages_old
#     else:
#         print(f"[BP] did not converge within {max_iter} iterations (Δ={delta:.2e})")

#     preds = beliefs.argmax(1)
#     return beliefs, preds, node2idx, idx2node


# def belief_propagation_weighted(
#     G: nx.Graph,
#     q: int,
#     *,
#     beta: float | None = None,
#     max_iter: int = 1000,
#     tol: float = 1e-4,
#     damping: float = 0.20,
#     balance_regularization: float = 0.10,
#     seed: int = 0,
#     min_steps: int = 0,
#     init: Literal["random", "spectral"] = "random",
#     msg_init: Literal["random", "copy", "pre-group"] = "random",
#     group_obs: List | None = None,
#     min_sep: float | None = None,
#     eps: float = 0.1,
# ):
#     """Vectorised BP that reproduces the exact math/logic of the reference loop,
#        but uses each edge’s 'weight' as a scaling on the pairwise compatibility."""
#     rng = np.random.default_rng(seed)
    
#     if G.number_of_edges() < 1:
#         print("[BP] Warning: Graph has no edges, returning random beliefs")
#         node2idx = {u: i for i, u in enumerate(G)}
#         idx2node = {i: u for u, i in node2idx.items()}
#         n = len(node2idx)
#         beliefs = init_beliefs(n, q, rng)
#         preds = beliefs.argmax(1)
#         return beliefs, preds, node2idx, idx2node

#     # ---------------------------------------------------------------------
#     #  Pre-compute arrays & constants
#     # ---------------------------------------------------------------------
#     node2idx, idx2node, src, dst, rev = build_arrays(G)
#     n, m = len(node2idx), src.size // 2
#     deg = np.fromiter((G.degree[u] for u in G), int)

#     # --- NEW: directed-edge weights array ---
#     w = np.array([
#         G.edges[idx2node[src[i]], idx2node[dst[i]]].get("weight", 1.0)
#         for i in range(src.size)
#     ])
#     w = (w - w.mean())/w.std()
#     w = (w/np.max(abs(w))) + 1.0
#     if beta is None:
#         beta = beta_param(G, q) * 1.2
#     exp_beta = np.exp(beta)

#     # ---------------------------------------------------------------------
#     #  Initial beliefs & messages
#     # ---------------------------------------------------------------------
#     spectral_labels = (
#         spectral_clustering(G, q, seed=seed) if init == "spectral" else {}
#     )
#     beliefs = init_beliefs(n, q, rng, labels=spectral_labels, node2idx=node2idx)
#     spec_arr = np.array([spectral_labels.get(idx2node[i], 0) for i in range(n)], int)

#     # Init prior + soften beliefs/messages (only for spectral init, q>1)
#     pi0 = None
#     if init == "spectral" and q > 1 and len(spectral_labels) > 0:
#         tiny = 1e-12
#         alpha_pi0 = 2.0       # stronger smoothing for imbalance
#         pi0_floor = 1e-4
#         eta_belief = 0.08     # how much to mix pi0 into beliefs (works for q>2)
        
#         # Clamp spec_arr to valid range
#         spec_arr = np.asarray(spec_arr, dtype=int)
#         spec_arr = np.mod(spec_arr, q)
        
#         # Compute pi0
#         pi0 = _compute_pi0_from_labels(spec_arr, q, alpha=alpha_pi0, floor=pi0_floor)
        
#         # Soften initial beliefs toward pi0
#         beliefs = beliefs.astype(np.float64, copy=False)
#         beliefs = np.maximum(beliefs, tiny)
#         beliefs /= beliefs.sum(1, keepdims=True).clip(tiny)
#         beliefs[:] = (1.0 - eta_belief) * beliefs + eta_belief * pi0[None, :]
#         beliefs = np.maximum(beliefs, tiny)
#         beliefs /= beliefs.sum(1, keepdims=True).clip(tiny)

#     messages_old = init_messages(
#         q, src, dst,
#         method=msg_init,
#         rng=rng,
#         beliefs=beliefs,
#         spec_arr=spec_arr,
#         node2idx=node2idx,
#         group_obs=group_obs,
#         min_sep=min_sep,
#         eps=eps,
#     )
    
#     # Soften initial messages toward pi0 (if spectral init was used)
#     if init == "spectral" and q > 1 and len(spectral_labels) > 0 and pi0 is not None:
#         tiny = 1e-12
#         eta_msg = 0.06        # message softening amount
#         messages_old = messages_old.astype(np.float64, copy=False)
#         messages_old = np.maximum(messages_old, tiny)
#         messages_old /= messages_old.sum(1, keepdims=True).clip(tiny)
#         messages_old[:] = (1.0 - eta_msg) * messages_old + eta_msg * pi0[None, :]
#         messages_old = np.maximum(messages_old, tiny)
#         messages_old /= messages_old.sum(1, keepdims=True).clip(tiny)
    
#     messages = np.empty_like(messages_old)

#     S = np.empty((n, q))
#     convergence_history: List[float] = []

#     for it in range(max_iter):
#         # --------------------------------------------------------------
#         #  Belief update
#         # --------------------------------------------------------------
#         # ------------------ MODIFIED LINE ------------------
#         # edge_fac = 1.0 + (exp_beta - 1.0) * messages_old.clip(1e-10) * w[:, None]
#         exp_beta_w = np.exp(beta * w)
#         edge_fac   = 1 + (exp_beta_w[:, None] - 1) * messages_old.clip(1e-10)
        
#         # ----------------------------------------------------
#         log_fac = np.log(edge_fac)
#         S.fill(0.0)
#         np.add.at(S, dst, log_fac)

#         log_beliefs = S - S.max(1)[:, None]
#         beliefs[:] = np.exp(
#             log_beliefs - np.log(np.exp(log_beliefs).sum(1)[:, None] + 1e-10)
#         )

#         comm_sz = beliefs.mean(0).clip(1e-10)
#         theta = (deg[:, None] * beliefs).sum(0).clip(1e-10)

#         # --------------------------------------------------------------
#         #  Message update
#         # --------------------------------------------------------------
#         if m > 0:
#             log_messages = (
#                 -beta * deg[src, None] * theta / (2.0 * m)
#                 + S[src]
#                 - log_fac[rev]
#                 - balance_regularization * np.log(comm_sz)
#             )
#             log_max = log_messages.max(1)[:, None]
#             messages_new = np.exp(log_messages - log_max)
#             messages_new /= messages_new.sum(1)[:, None].clip(1e-10)
#         else:
#             messages_new = np.ones_like(messages_old)
#             messages_new /= messages_new.sum(1)[:, None].clip(1e-10)

#         messages[:] = (1.0 - damping) * messages_new + damping * messages_old

#         # --------------------------------------------------------------
#         #  Convergence check & optional entropy-based noise reinjection
#         # --------------------------------------------------------------
#         if messages.size == 0:
#             print("[BP] Warning: Empty message arrays detected, aborting loop")
#             delta = 0.0
#             break

#         try:
#             delta = np.max(np.abs(messages - messages_old))
#             if np.isnan(delta):
#                 damping = min(damping * 1.5, 0.9)
#                 messages[:] = messages_old
#                 continue
#         except ValueError as e:
#             if "zero-size array" in str(e):
#                 delta = 0.0
#                 break
#             else:
#                 raise

#         convergence_history.append(float(delta))

#         if delta < tol and it >= min_steps:
#             entropy = -np.sum(comm_sz * np.log(comm_sz + 1e-10))
#             entropy_ratio = entropy / (-np.log(1.0 / q))
#             if entropy_ratio > 0.7:
#                 print(f"[BP] converged in {it+1} iterations; entropy ratio={entropy_ratio:.3f}")
#                 break
#             noise = rng.random(messages.shape) * 0.15 / (comm_sz + 1e-10)
#             messages[:] = messages * 0.85 + noise
#             messages /= messages.sum(1)[:, None]

#         messages_old, messages = messages, messages_old
#     else:
#         print(f"[BP] did not converge within {max_iter} iterations (Δ={delta:.2e})")

#     preds = beliefs.argmax(1)
#     return beliefs, preds, node2idx, idx2node


def belief_propagation_weighted(
    G: nx.Graph,
    q: int,
    *,
    beta: float | None = None,
    max_iter: int = 1000,
    tol: float = 1e-4,
    damping: float = 0.20,
    balance_regularization: float = 0.10,
    seed: int = 0,
    min_steps: int = 0,
    init: Literal["random", "spectral"] = "random",
    msg_init: Literal["random", "copy", "pre-group"] = "random",
    group_obs: List | None = None,
    min_sep: float | None = None,
    eps: float = 0.1,
    pi: np.ndarray | None = None,
    
):
    """Vectorised BP that reproduces the exact math/logic of the reference loop,
       but uses each edge’s 'weight' as a scaling on the pairwise compatibility."""
    rng = np.random.default_rng(seed)
    if pi is None and q > 1:
        pi = estimate_pi_from_duospec_weights(G, q, seed=seed, weight_attr="weight")
    else:
        pi = np.asarray(pi, dtype=np.float64) if pi is not None else np.full(q, 1.0 / q)
        pi = np.maximum(pi, 1e-12)
        pi = pi / pi.sum()

    if G.number_of_edges() < 1:
        print("[BP] Warning: Graph has no edges, returning random beliefs")
        node2idx = {u: i for i, u in enumerate(G)}
        idx2node = {i: u for u, i in node2idx.items()}
        n = len(node2idx)
        beliefs = init_beliefs(n, q, rng)
        preds = beliefs.argmax(1)
        return beliefs, preds, node2idx, idx2node

    # ---------------------------------------------------------------------
    #  Pre-compute arrays & constants
    # ---------------------------------------------------------------------
    node2idx, idx2node, src, dst, rev = build_arrays(G)
    n, m = len(node2idx), src.size // 2
    deg = np.fromiter((G.degree[u] for u in G), int)

    # --- NEW: directed-edge weights array ---
    w = np.array([
        G.edges[idx2node[src[i]], idx2node[dst[i]]].get("weight", 1.0)
        for i in range(src.size)
    ])
    w = (w - w.mean())/w.std()
    w = (w/np.max(abs(w))) + 1.0
    if beta is None:
        beta = beta_param(G, q) * 1.2
    exp_beta = np.exp(beta)

    # ---------------------------------------------------------------------
    #  Initial beliefs & messages
    # ---------------------------------------------------------------------
    spectral_labels = (
        spectral_clustering(G, q, seed=seed) if init == "spectral" else {}
    )
    beliefs = init_beliefs(n, q, rng, labels=spectral_labels, node2idx=node2idx)
    spec_arr = np.array([spectral_labels.get(idx2node[i], 0) for i in range(n)], int)

    # Init prior + soften beliefs/messages (only for spectral init, q>1)
    pi0 = None
    if init == "spectral" and q > 1 and len(spectral_labels) > 0:
        tiny = 1e-12
        alpha_pi0 = 2.0       # stronger smoothing for imbalance
        pi0_floor = 1e-4
        eta_belief = 0.08     # how much to mix pi0 into beliefs (works for q>2)
        
        # Clamp spec_arr to valid range
        spec_arr = np.asarray(spec_arr, dtype=int)
        spec_arr = np.mod(spec_arr, q)
        
        # Compute pi0
        pi0 = _compute_pi0_from_labels(spec_arr, q, alpha=alpha_pi0, floor=pi0_floor)
        
        # Soften initial beliefs toward pi0
        beliefs = beliefs.astype(np.float64, copy=False)
        beliefs = np.maximum(beliefs, tiny)
        beliefs /= beliefs.sum(1, keepdims=True).clip(tiny)
        beliefs[:] = (1.0 - eta_belief) * beliefs + eta_belief * pi0[None, :]
        beliefs = np.maximum(beliefs, tiny)
        beliefs /= beliefs.sum(1, keepdims=True).clip(tiny)
    else:
        pi0 = pi.copy()

    messages_old = init_messages(
        q, src, dst,
        method=msg_init,
        rng=rng,
        beliefs=beliefs,
        spec_arr=spec_arr,
        node2idx=node2idx,
        group_obs=group_obs,
        min_sep=min_sep,
        eps=eps,
    )
    
    # Soften initial messages toward pi0 (if spectral init was used)
    if init == "spectral" and q > 1 and len(spectral_labels) > 0 and pi0 is not None:
        tiny = 1e-12
        eta_msg = 0.06        # message softening amount
        messages_old = messages_old.astype(np.float64, copy=False)
        messages_old = np.maximum(messages_old, tiny)
        messages_old /= messages_old.sum(1, keepdims=True).clip(tiny)
        messages_old[:] = (1.0 - eta_msg) * messages_old + eta_msg * pi0[None, :]
        messages_old = np.maximum(messages_old, tiny)
        messages_old /= messages_old.sum(1, keepdims=True).clip(tiny)
    
    messages = np.empty_like(messages_old)

    S = np.empty((n, q))
    convergence_history: List[float] = []

    for it in range(max_iter):
        # --------------------------------------------------------------
        #  Belief update
        # --------------------------------------------------------------
        # ------------------ MODIFIED LINE ------------------
        # edge_fac = 1.0 + (exp_beta - 1.0) * messages_old.clip(1e-10) * w[:, None]
        exp_beta_w = np.exp(beta * w)
        edge_fac   = 1 + (exp_beta_w[:, None] - 1) * messages_old.clip(1e-10)
        
        # ----------------------------------------------------
        log_fac = np.log(edge_fac)
        S.fill(0.0)
        np.add.at(S, dst, log_fac)

        log_beliefs = S - S.max(1)[:, None]
        beliefs[:] = np.exp(
            log_beliefs - np.log(np.exp(log_beliefs).sum(1)[:, None] + 1e-10)
        )

        comm_sz = beliefs.mean(0).clip(1e-10)
        theta = (deg[:, None] * beliefs).sum(0).clip(1e-10)

        # --------------------------------------------------------------
        #  Message update
        # --------------------------------------------------------------
        if m > 0:
            log_messages = (
                -beta * deg[src, None] * theta / (2.0 * m)
                + S[src]
                - log_fac[rev]
                - balance_regularization * np.log(comm_sz / pi)
            )
            log_max = log_messages.max(1)[:, None]
            messages_new = np.exp(log_messages - log_max)
            messages_new /= messages_new.sum(1)[:, None].clip(1e-10)
        else:
            messages_new = np.ones_like(messages_old)
            messages_new /= messages_new.sum(1)[:, None].clip(1e-10)

        messages[:] = (1.0 - damping) * messages_new + damping * messages_old

        # --------------------------------------------------------------
        #  Convergence check & optional entropy-based noise reinjection
        # --------------------------------------------------------------
        if messages.size == 0:
            print("[BP] Warning: Empty message arrays detected, aborting loop")
            delta = 0.0
            break

        try:
            delta = np.max(np.abs(messages - messages_old))
            if np.isnan(delta):
                damping = min(damping * 1.5, 0.9)
                messages[:] = messages_old
                continue
        except ValueError as e:
            if "zero-size array" in str(e):
                delta = 0.0
                break
            else:
                raise

        convergence_history.append(float(delta))

        if delta < tol and it >= min_steps:
            entropy = -np.sum(comm_sz * np.log(comm_sz + 1e-10))
            entropy_ratio = entropy / (-np.log(1.0 / q))
            if entropy_ratio > 0.7:
                print(f"[BP] converged in {it+1} iterations; entropy ratio={entropy_ratio:.3f}")
                break
            noise = rng.random(messages.shape) * 0.15 / (comm_sz + 1e-10)
            messages[:] = messages * 0.85 + noise
            messages /= messages.sum(1)[:, None]

        messages_old, messages = messages, messages_old
    else:
        print(f"[BP] did not converge within {max_iter} iterations (Δ={delta:.2e})")

    preds = beliefs.argmax(1)
    return beliefs, preds, node2idx, idx2node

# -----------------------------------------------------------------------------
#  Evaluation helpers
# -----------------------------------------------------------------------------

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
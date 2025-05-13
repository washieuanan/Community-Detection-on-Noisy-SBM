from __future__ import annotations
from typing import Dict, List, Tuple, Literal
import networkx as nx
import numpy as np
import scipy.sparse.linalg as sla

import scipy.sparse as sp



from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.optimize import linear_sum_assignment
from scipy.stats import permutation_test, mode

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

# def spectral_clustering(G: nx.Graph, q: int, *, seed: int = 0):
#     vals, vecs = sla.eigs(nx.adjacency_matrix(G), k=q, which="LM", tol=1e-2)
#     km = KMeans(n_clusters=q, random_state=seed).fit(np.real(vecs))
#     return {n: int(l) for n, l in zip(G.nodes(), km.labels_)}

def bethe_hessian_embedding(G: nx.Graph, q: int):
    A = nx.to_scipy_sparse_array(G, dtype=float)
    k = np.array([d for _, d in G.degree()], float)
    r = np.sqrt(max(k.mean(), 1e-8))
    I = sp.eye(A.shape[0], format="csr", dtype=float)   
    H = (r ** 2 - 1) * I - r * A + sp.diags(k, format="csr")

    ncv = min(H.shape[0], max(2*q + 1, 10*q))
    vals, vecs = sla.eigsh(H, k=q, which="SM", tol=1e-2, ncv= ncv)
    return np.real(vecs)

def spectral_clustering(G: nx.Graph, q: int, *, seed: int = 0):
    emb = bethe_hessian_embedding(G, q)
    km = KMeans(n_clusters=q, random_state=seed).fit(emb)
    return {n: int(l) for n, l in zip(G.nodes(), km.labels_)}


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
    init: Literal["random", "spectral"] = "spectral",
    msg_init: Literal["random", "copy", "pre-group"] = "random",
    group_obs: List | None = None,
    min_sep: float | None = None,
    eps: float = 0.1,
    mode: Literal["sbm", "dc"] = "dc",
):
    """Vectorised BP that reproduces the exact math/logic of the reference loop."""

    rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------------
    #  Pre‑compute arrays & constants
    # ---------------------------------------------------------------------
    node2idx, idx2node, src, dst, rev = build_arrays(G)
    n, m = len(node2idx), src.size // 2
    deg = np.fromiter((G.degree[u] for u in G), int)

    if beta is None:
        beta = beta_param(G, q) * 1.1  # match reference scaling
    exp_beta = np.exp(beta)

    adapt_every = 10

    # ---------------------------------------------------------------------
    #  Initial beliefs & messages
    # ---------------------------------------------------------------------
    spectral_labels = spectral_clustering(G, q, seed=seed) if init == "spectral" else {}
    beliefs = init_beliefs(n, q, rng, labels=spectral_labels, node2idx=node2idx)
    spec_arr = np.array([spectral_labels.get(idx2node[i], 0) for i in range(n)], int)

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
    messages = np.empty_like(messages_old)

    # Scratch arrays
    S = np.empty((n, q))
    convergence_history: List[float] = []

    for it in range(max_iter):
        # --------------------------------------------------------------
        #  Belief update  (matches reference inner loops)
        # --------------------------------------------------------------
        edge_fac = 1.0 + (exp_beta - 1.0) * messages_old
        log_fac = np.log(edge_fac.clip(1e-10))
        S.fill(0.0)
        np.add.at(S, dst, log_fac)
        beliefs[:] = np.exp(S - S.max(1)[:, None])
        beliefs /= beliefs.sum(1)[:, None]

        if it < 100:                              # only during the first few rounds
            iso_mask = deg == 0                  # vertices with *observed* degree 0
            if iso_mask.any():
                # mix 1 % of the uniform distribution into their current belief
                beliefs[iso_mask] = (
                    0.99 * beliefs[iso_mask] + 0.02 / q
                )   # theta
        # --------------------------------------------------------------
        #  Community sizes & theta (same formulas)
        # --------------------------------------------------------------
        comm_sz = beliefs.mean(0)                          # community_sizes
        theta = (deg[:, None] * beliefs).sum(0)         
        
        

        # --------------------------------------------------------------
        #  Message update  (vectorised reference equation)
        # --------------------------------------------------------------
        # messages_new = np.exp(
        #     -beta * deg[src, None] * theta / (2.0 * m) +   # term1
        #     S[src] -                                       # Σ over neighbours except dst
        #     log_fac[rev] -                                 # subtract k→i contribution
        #     balance_regularization * np.log(comm_sz + 1e-10)  # size_penalty
        # )

        if mode == "dc": 
            dc_fac =    1.0 / np.sqrt(deg[src, None] * deg[dst, None])
        else: 
            dc_fac = 1.0 

        messages_new = np.exp(
            -beta * deg[src, None] * theta / (2.0 * m) + S[src] - log_fac[rev] - balance_regularization * np.log(comm_sz + 1e-10)  # size_penalty
        ) * dc_fac
        messages_new /= messages_new.sum(1)[:, None]

        # Damp
        messages[:] = (1.0 - damping) * messages_new + damping * messages_old

        # --------------------------------------------------------------
        #  Convergence check & optional entropy‑based noise reinjection
        # --------------------------------------------------------------
        delta = np.max(np.abs(messages - messages_old))
        if it < 20: 
            cur_damp = damping
        else:
            cur_damp = min(0.05 + 0.95 * (delta / 0.5), damping)
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
            messages[:] = (1.0 - damping) * messages_new + damping * messages_old

            messages /= messages.sum(1)[:, None]

        preds = beliefs.argmax(1)
        messages_old, messages = messages, messages_old  # swap buffers
        if (it + 1) % adapt_every == 0:
            same = (preds[src] == preds[dst]).mean() 
            uniform = 1 / q 
            rho = max(same - uniform, 1e-3) 
            beta = np.log((1 + (q-1) * rho) / (1 - rho))
            exp_beta = np.exp(beta)
                
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

if __name__ == "__main__": 
    from graph_generation.gbm import generate_gbm
    from observations.standard_observe import PairSamplingObservation, get_coordinate_distance
    from community_detection.bp.vectorized_geometric_bp import (
        detection_stats,
        get_true_communities,
    )

    # Generate latent GBM graph
    G_true = generate_gbm(n=700, K=3, a=75, b=20, seed=42)
    avg_deg = np.mean([G_true.degree[n] for n in G_true.nodes()])
    original_density = avg_deg / len(G_true.nodes)
    C = 0.025 * original_density

    def weight_func(c1, c2):
        return np.exp(-0.5 * get_coordinate_distance(c1, c2))

    num_pairs = int(C * len(G_true.nodes) ** 2 / 2)
    sampler = PairSamplingObservation(G_true, num_samples=num_pairs, weight_func=weight_func, seed=42)
    observations = sampler.observe()


    observations_ = [(u, v, d_uv) for (u, v), d_uv in observations]

   
    obs_nodes: set[int] = set()
    for u, v, _ in observations_:
        obs_nodes.update((u, v))

    # Build subgraph of observed nodes with inferred coords (for BP)
    from community_detection.bp.gbm_bp import create_observed_subgraph

    subG = create_observed_subgraph(G_true.number_of_nodes(), observations_)
  

    print("Running Loopy BP …")
    _, preds, node2idx, idx2node = belief_propagation(
        subG,
        q=3,
        seed=42,
        init="spectral",
        msg_init="random",
        max_iter=5000,
        damping=0.27,
        balance_regularization=0.05,
        mode="dc",
    )

    true_labels = get_true_communities(G_true, node2idx=node2idx, attr="comm")
    stats = detection_stats(preds, true_labels)
    print("\n=== Community‑detection accuracy ===")
    for k, v in stats.items():
        print(f"{k:>25s} : {v}")

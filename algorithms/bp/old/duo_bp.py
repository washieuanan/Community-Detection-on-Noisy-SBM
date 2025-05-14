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
from block_models.gbm import generate_graph
from deprecated.observations.standard_observe import PairSamplingObservation, get_coordinate_distance
from algorithms.bp.vectorized_bp import belief_propagation, beta_param
from collections import defaultdict
from copy import deepcopy

# def belief_propagation(
#         G: nx.Graph,
#         q: int,
#         *,
#         beta: float | None = None,
#         max_iter: int = 1_000,
#         tol: float = 1e-4,
#         damping: float = 0.20,
#         balance_regularization: float = 0.10,
#         seed: int = 0,
#         gamma: float = 1.0,
#         init_beliefs: Literal["random", "spectral"] = "random",
#         message_init: Literal["random", "copy", "pre-group"] = "random",
#         group_obs=None,
#         min_sep=None,
#         eps: float = 0.1,
# ):
#     rng = np.random.default_rng(seed)


#     node2idx, idx2node, src, dst, rev = build_arrays(G)
#     n, m = len(node2idx), src.size // 2
#     deg  = np.fromiter((G.degree[u] for u in G), dtype=np.int32)

#     # coords = np.vstack([G.nodes[u]["coords"] for u in G])
#     geo_w  = np.exp(-gamma * G[src][dst]["dist"])

#     if beta is None:
#         beta = calc_beta_param(G, q)

#     beliefs = rng.dirichlet(np.ones(q), size=n)
#     if init_beliefs == "spectral":
#         spec = spectral_clustering(G, q, seed=seed)
#         for u, idx in node2idx.items():
#             beliefs[idx, spec[u]] += 0.2
#         beliefs /= beliefs.sum(1, keepdims=True)

#     messages_old = initialize_messages(
#         G, q, method=message_init, beliefs=beliefs,
#         node_2_idx=node2idx, idx_2_node=idx2node, src=src, dst=dst,
#         seed=seed, group_obs=group_obs, min_sep=min_sep, eps=eps,
#     )
#     messages = np.empty_like(messages_old)
#     S        = np.empty((n, q), dtype=np.float64)
#     exp_beta = np.exp(beta)

#     for it in range(max_iter):
#         #belief update 
#         edge_fac = 1.0 + (exp_beta - 1.0) * geo_w[:, None] * messages_old
#         log_fac  = np.log(edge_fac.clip(1e-10))
#         S.fill(0.0)
#         np.add.at(S, dst, log_fac)
#         beliefs[:] = np.exp(S)
#         beliefs   /= beliefs.sum(1, keepdims=True)

#         # message update
#         comm_sz = beliefs.mean(0)
#         theta   = (deg[:, None] * beliefs).sum(0)
#         msg_new = np.exp(
#             -beta * deg[src, None] * theta / (2.0 * m)
#             + S[src] - log_fac[rev]
#             - balance_regularization * np.log(comm_sz + 1e-10)
#         )
#         msg_new /= msg_new.sum(1, keepdims=True)
#         messages[:] = (1.0 - damping) * msg_new + damping * messages_old

#         delta = np.max(np.abs(messages - messages_old))
#         if delta < tol:
#             print(f"[BP] converged in {it+1} iterations (delta={delta:.2e})")
#             break
#         messages_old, messages = messages, messages_old   # swap

#     preds = beliefs.argmax(1)
#     return beliefs, preds, node2idx, idx2node

# def weighted_percentile(data, percentile, weights=None):
#     """
#     Compute the weighted percentile of a dataset.
    
#     Parameters
#     ----------
#     data : array-like
#         The data to compute percentile on.
#     percentile : float
#         The percentile to compute (between 0 and 100).
#     weights : array-like, optional
#         The weights for each data point. If None, uniform weights are used.
        
#     Returns
#     -------
#     float
#         The weighted percentile value.
#     """
#     if weights is None:
#         return np.percentile(data, percentile)
        
#     # Convert to numpy arrays if they aren't already
#     data = np.asarray(data)
#     weights = np.asarray(weights)
    
#     # Sort data and weights together
#     sorted_idx = np.argsort(data)
#     sorted_data = data[sorted_idx]
#     sorted_weights = weights[sorted_idx]
    
#     # Calculate cumulative weights (normalized)
#     cumulative_weights = np.cumsum(sorted_weights)
#     if cumulative_weights[-1] <= 0:
#         return np.nan
#     cumulative_weights = cumulative_weights / cumulative_weights[-1]
    
#     # Interpolate to find the percentile
#     return np.interp(percentile/100, cumulative_weights, sorted_data)


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
        w = G[u][v].get("weight", 1.0)
        w *= 1.0 - lam * p        # shrink
        if w < w_min:
            G.remove_edge(u, v)
            n_removed += 1
        else:
            G[u][v]["weight"] = w
    return n_removed


# ----------------------------------------------------------------------
# EM driver
# ----------------------------------------------------------------------
def duo_bp(
    G_obs          : nx.Graph,
    K              : int,
    num_balls      : int = 16,
    *,
    # ------- EM knobs -------
    max_em_iters         : int   = 50,
    anneal_steps         : int   = 6,
    warmup_rounds        : int   = 2,
    shrink_comm          : float = 0.90,   # 1.00 → keep, <1 → shrink
    shrink_geo           : float = 0.85,
    w_floor              : float = 1e-2,   # never go below this weight
    forget               : bool  = False,  # if True re-start from pristine
    # ------- BP control ----
    max_iter_bp          : int   = 1000,
    damp_high            : float = 0.50,
    damp_low             : float = 0.25,
    balance_regularization : float = 0.20,
    # ------- misc ----------
    tol                  : float = 1e-4,
    patience             : int   = 7,
    bp_kwargs            : Dict | None = None,
    seed                 : int   = 0,
):
    """
    Conservative variant of duo_bp:
      • never deletes an edge (only rescales)
      • keeps track of pristine weights (optional 'forget' reset)
      • linear annealing of scale factors up to shrink_comm/geo
    """
    rng        = np.random.default_rng(seed)
    subG       = deepcopy(G_obs)
    pristine_w = {(u, v): d.get("weight", 1.0) for u, v, d in subG.edges(data=True)}

    # ------- BP kwargs defaults -----------
    bp_kwargs = dict(bp_kwargs or {})
    bp_kwargs.setdefault("seed", seed)
    bp_kwargs.setdefault("init", "spectral")
    bp_kwargs.setdefault("max_iter", max_iter_bp)
    bp_kwargs.setdefault("balance_regularization", balance_regularization)
    bp_kwargs.pop("damping", None)        # we control it

    # ------- bookkeeping ------------------
    best   = {"obj": -np.inf}
    hist   : list[dict] = []
    no_imp = 0

    # ------- pre-compute helpers -----------
    edges_all = np.asarray(subG.edges(), dtype=object)
    node2idx_global = {u: i for i, u in enumerate(subG.nodes())}
    iu_glob, iv_glob = _to_idx(edges_all, node2idx_global)

    # main EM loop ----------------------------------------------------------
    for em in range(1, max_em_iters + 1):

        # -- optionally re-start weights (“forget”) --------------------------
        if forget and em > 1:
            for (u, v) in subG.edges():
                subG[u][v]["weight"] = pristine_w[(u, v)]

        # -- adaptive damping for BP ----------------------------------------
        frac = em / max_em_iters
        bp_kwargs["damping"] = damp_high - frac * (damp_high - damp_low)

        # ======== COMMUNITY step (BP, q=K) ================================
        bel_c, _, n2i_c, _ = belief_propagation_weighted(
            subG, q=K, **bp_kwargs
        )
        p_same = _edge_same_prob(bel_c, iu_glob, iv_glob)

        # soft scale factor for community information
        λ_c   = min(shrink_comm,
                    (em - warmup_rounds) / max(1, anneal_steps))
        λ_c   = max(0.0, λ_c)        # during warm-up λ→0

        # ======== GEOMETRIC step (BP, q=num_balls) ========================
        bel_g, _, n2i_g, _ = belief_propagation_weighted(
            subG, q=num_balls, **bp_kwargs
        )
        lbls_g   = bel_g.argmax(1)
        conf_g   = 0.5 * (bel_g[iu_glob, lbls_g[iu_glob]]
                        +  bel_g[iv_glob, lbls_g[iv_glob]])

        same_ball = lbls_g[iu_glob] == lbls_g[iv_glob]

        λ_g   = min(shrink_geo,
                    (em - warmup_rounds) / max(1, anneal_steps))
        λ_g   = max(0.0, λ_g)

        # ======== RE-WEIGHT edges (never drop) ============================
        for (u, v), psame, sb, cg in zip(edges_all, p_same, same_ball, conf_g):
            w0 = subG[u][v].get("weight", 1.0)
            w_new = w0
            # community shrink
            if psame > np.percentile(p_same, 90):
                w_new *= 1.0 - λ_c * (1.0 - psame)
            # geometry shrink (opposite effect: penalise edges inside big balls)
            if sb and cg > np.percentile(conf_g, 90):
                w_new *= 1.0 - λ_g * cg
            # floor safeguard
            subG[u][v]["weight"] = max(w_new, w_floor)

        # ======== OBJECTIVE + early stopping ==============================
        obj = float(np.max(bel_c, axis=1).sum())
        hist.append(dict(iter=em, obj=obj,
                         λ_c=λ_c, λ_g=λ_g,
                         damping=bp_kwargs["damping"]))

        if obj > best["obj"]:
            best.update(obj=obj, beliefs=bel_c,
                        balls=lbls_g, node2idx=node2idx_global)
            no_imp = 0
        else:
            no_imp += 1

        if no_imp >= patience:
            print(f"[safe-duoBP] patience {patience} reached at iter {em}")
            break

        if em > 1 and abs(obj - hist[-2]["obj"]) < tol:
            print(f"[safe-duoBP] converged at iter {em}")
            break

    hard = best["beliefs"].argmax(1)
    return dict(
        beliefs     = best["beliefs"],
        communities = hard,
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
    a = 125
    b = 10
    n = 800
    K = 2
    r_in = a * np.log(n) / n
    r_out = b * np.log(n) / n
    print(f"r_in = {r_in:.4f}, r_out = {r_out:.4f}")
    G_true = generate_gbm(n=n, K=K, a=a, b=b, seed=42)
    print("Generated graph with", len(G_true.nodes()), "nodes and", len(G_true.edges()), "edges")
    # for u, v in G_true.edges():
    #     G_true[u][v]["dist"] = np.linalg.norm(np.array(G_true.nodes[u]["coords"]) - np.array(G_true.nodes[v]["coords"]))
    avg_deg = np.mean([G_true.degree[n] for n in G_true.nodes()])
    print("avg_deg:", avg_deg)
    original_density = avg_deg / len(G_true.nodes)
    C = 0.05 * original_density
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

    subG = create_dist_observed_subgraph(G_true.number_of_nodes(), observations)
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
    bel_c, preds, n2i, idx2n = belief_propagation(
        subG,
        q=K,
        seed=42,
        init="spectral",
        msg_init="random",
        damping=0.15,
        min_steps = 50,
        balance_regularization=0.05,
    )
    res = {
        "beliefs": bel_c,
        "communities": preds,
        "node2idx": n2i,
        "idx2node": idx2n,
    }
    # res = duo_bp(
    #     G_true,
    #     K=K,
    #     num_balls=32
    # )
    preds = res["communities"]
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
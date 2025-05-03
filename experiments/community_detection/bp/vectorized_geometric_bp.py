from __future__ import annotations
import numpy as np
import networkx as nx
import scipy.sparse.linalg as sla
from sklearn.cluster import KMeans
from typing import Literal, Dict, Tuple, List
from scipy.stats import mode

from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.optimize import linear_sum_assignment
from scipy.stats import permutation_test



def get_true_communities(
        G: nx.Graph,
        *,
        node2idx: Dict[int, int] | None = None,
        attr: str = "comm",
) -> np.ndarray:
    """
    Parameters
    ----------
    G         : the original NetworkX graph
    node2idx  : the mapping returned by `build_arrays`.  
                If None, nodes are taken in `list(G.nodes())` order.
    attr      : node attribute that stores the ground‑truth label
    Returns
    -------
    true : (n,) int ndarray   – community id for each node index
    """
    if node2idx is None:
        node_order = list(G.nodes())
        n = len(node_order)
        true = np.empty(n, dtype=int)
        for i, u in enumerate(node_order):
            true[i] = G.nodes[u][attr]
    else:
        n = len(node2idx)
        true = np.empty(n, dtype=int)
        for u, i in node2idx.items():
            true[i] = G.nodes[u][attr]
    return true


def spectral_clustering(
        G: nx.Graph,
        q: int, 
        seed: int = 0) -> Dict[int, int]: 
    A = nx.adjacency_matrix(G) 
    _, vecs = sla.eigs(A, k=q, which='LM', tol=1e-2) 

    coords = np.real(vecs) 
    km = KMeans(n_clusters=q, random_state=seed).fit(coords) 

    return {node: int(km.labels_[i]) for i, node in enumerate(G.nodes())}

def calc_beta_param(G: nx.Graph, q: int) -> float: 
    degs = np.fromiter((G.degree[n] for n in G), dtype = float) 
    avg_d = degs.mean() 

    eps = 1e-3 
    num = q * (1 + (q-1) * eps) 
    den = max(avg_d * (1 - eps) - (1 + (q-1) * eps), 1e-10)

    return np.log(num / den + 1) * 1.2 

def build_arrays(G: nx.Graph) -> Tuple[Dict[int, int], Dict[int, int], np.ndarray, np.ndarray]: 

    node_2_idx = {u: i for i, u in enumerate(G.nodes())}
    idx_2_node = {i: u for i, u in node_2_idx.items()}
    m = G.number_of_edges()

    src = np.empty(2*m, dtype=np.int32) 
    dst = np.empty(2*m, dtype=np.int32)

    k = 0 

    for u, v in G.edges():
        u_i, v_i = node_2_idx[u], node_2_idx[v]
        src[k], dst[k] = u_i, v_i
        src[k+1], dst[k+1] = v_i, u_i
        k+= 2 

    rev = np.empty_like(src)
    rev[0::2] = 1 + np.arange(0, 2*m, 2) 
    rev[1::2] = 0 + np.arange(0, 2*m, 2) 

    return node_2_idx, idx_2_node, src, dst, rev

def initialize_messages(
        G: nx.Graph, 
        q: int, 
        *, 
        method: Literal['random', 'copy', 'pre-group'] = 'random', 
        beliefs: np.ndarray, 
        node_2_idx: Dict[int, int],
        idx_2_node: Dict[int, int],
        src: np.ndarray, 
        dst: np.ndarray,
        seed: int = 0,
        group_obs: List | None = None, 
        min_sep: float | None = None, 
        eps: float = 0.1
) -> np.ndarray: 
    
    rng = np.random.default_rng(seed)
    m = src.size // 2 

    msgs = rng.dirichlet(np.ones(q), size=2*m) + 1e-3 

    spec = spectral_clustering(G, q, seed =seed) 

    if isinstance(spec, dict):
        spec_arr = np.empty(len(node_2_idx), dtype=int)
        for node, lab in spec.items(): 
            spec_arr[node_2_idx[node]] = lab 
    else: 
        spec_arr = spec 


    if method == "random":
        for e in range(2 * m):
            u_idx = src[e]
            label = spec_arr[u_idx]
            msgs[e, label] += eps
            msgs[e] /= msgs[e].sum()

    elif method == "copy":
        for e in range(2 * m):
            u_idx = src[e]
            msgs[e] = beliefs[u_idx].copy()
            label  = spec_arr[u_idx]
            msgs[e, label] += eps
            msgs[e] /= msgs[e].sum()
    
    elif method == "pre-group": 
        if group_obs is None: 
            raise ValueError("group_obs must be provided when method is 'pre-group'")
        
        e_id = {(src[e], dst[e]): e for e in range(2*m)}
        bias_assignment = np.zeros(len(group_obs), dtype=int)

        for g, obs in enumerate(group_obs): 
            verts: List[int] = [] 

            if isinstance(obs, dict): 
                for r in obs: 
                    verts += [u for u, _ in obs[r]]
            else: 
                verts += [u for u, _ in obs] 

            if verts: 
                bias_assignment[g] = mode([spec[v] for v in verts])[0][0]
            else: 
                bias_assignment[g] = -1


        bias_vec = np.zeros(q) 
        for g, obs in enumerate(group_obs):
            t = bias_assignment[g]

            if t == -1: 
                continue

            bias_vec[t] = np.sqrt(min_sep) if min_sep else np.sqrt(0.15) 

            def _apply(u,v, extra=None): 
                i = e_id.get((node_2_idx[u], node_2_idx[v]))
                j = e_id.get((node_2_idx[v], node_2_idx[u]))

                if i is None or j is None:
                    return 
                
                msgs[i] += bias_vec + (extra if extra is not None else 0) 
                msgs[i] /= msgs[i].sum()

                msgs[j] += bias_vec + (extra if extra is not None else 0)
                msgs[j] /= msgs[j].sum()

            if isinstance(obs, dict):
                for rad, edges in obs.items(): 
                    extra = np.zeros(q) 
                    extra[t] = max(-0.2 * np.exp(float(rad)), -bias_vec[t])

                    for u, v in edges:
                        _apply(u, v, extra=extra)
            else: 
                for u, v in obs: 
                    _apply(u, v)


    return msgs.astype(np.float64)
        

def belief_propagation(
        G: nx.Graph,
        q: int,
        *,
        beta: float | None = None,
        max_iter: int = 1_000,
        tol: float = 1e-4,
        damping: float = 0.20,
        balance_regularization: float = 0.10,
        seed: int = 0,
        gamma: float = 1.0,
        init_beliefs: Literal["random", "spectral"] = "random",
        message_init: Literal["random", "copy", "pre-group"] = "random",
        group_obs=None,
        min_sep=None,
        eps: float = 0.1,
):
    rng = np.random.default_rng(seed)


    node2idx, idx2node, src, dst, rev = build_arrays(G)
    n, m = len(node2idx), src.size // 2
    deg  = np.fromiter((G.degree[u] for u in G), dtype=np.int32)

    coords = np.vstack([G.nodes[u]["coord"] for u in G])
    geo_w  = np.exp(-gamma * np.linalg.norm(coords[src] - coords[dst], axis=1))

    if beta is None:
        beta = calc_beta_param(G, q)

    beliefs = rng.dirichlet(np.ones(q), size=n)
    if init_beliefs == "spectral":
        spec = spectral_clustering(G, q, seed=seed)
        for u, idx in node2idx.items():
            beliefs[idx, spec[u]] += 0.2
        beliefs /= beliefs.sum(1, keepdims=True)

    messages_old = initialize_messages(
        G, q, method=message_init, beliefs=beliefs,
        node_2_idx=node2idx, idx_2_node=idx2node, src=src, dst=dst,
        seed=seed, group_obs=group_obs, min_sep=min_sep, eps=eps,
    )
    messages = np.empty_like(messages_old)
    S        = np.empty((n, q), dtype=np.float64)
    exp_beta = np.exp(beta)

    for it in range(max_iter):
        #belief update 
        edge_fac = 1.0 + (exp_beta - 1.0) * geo_w[:, None] * messages_old
        log_fac  = np.log(edge_fac.clip(1e-10))
        S.fill(0.0)
        np.add.at(S, dst, log_fac)
        beliefs[:] = np.exp(S)
        beliefs   /= beliefs.sum(1, keepdims=True)

        # message update
        comm_sz = beliefs.mean(0)
        theta   = (deg[:, None] * beliefs).sum(0)
        msg_new = np.exp(
            -beta * deg[src, None] * theta / (2.0 * m)
            + S[src] - log_fac[rev]
            - balance_regularization * np.log(comm_sz + 1e-10)
        )
        msg_new /= msg_new.sum(1, keepdims=True)
        messages[:] = (1.0 - damping) * msg_new + damping * messages_old

        delta = np.max(np.abs(messages - messages_old))
        if delta < tol:
            print(f"[BP] converged in {it+1} iterations (delta={delta:.2e})")
            break
        messages_old, messages = messages, messages_old   # swap

    preds = beliefs.argmax(1)
    return beliefs, preds, node2idx, idx2node


def detection_stats(preds: np.ndarray, true: np.ndarray, *, n_perm: int = 10_000):
    """
    Parameters
    ----------
    preds, true : 1‑D integer arrays with the same length.
    n_perm      : #resamples for the permutation test
    Returns
    -------
    stats : dict   – identical keys to your original function
    """
   
    k = int(max(preds.max(), true.max()) + 1)
    C = confusion_matrix(true, preds, labels=np.arange(k))  
    row_ind, col_ind = linear_sum_assignment(-C)            
    perm_map = np.arange(k)
    perm_map[col_ind] = row_ind
    perm_preds = perm_map[preds]

    stats = {
        "accuracy": accuracy_score(true, perm_preds),
        "num vertices": len(true),
        "num communities predicted": len(np.unique(perm_preds)),
    }
    # per‑community accuracy
    for t in range(k):
        mask = true == t
        stats[f"accuracy_{t}"] = accuracy_score(true[mask], perm_preds[mask])

    # permutation test on accuracy
    res = permutation_test(
        (true, perm_preds),
        statistic=lambda x, y: accuracy_score(x, y),
        vectorized=False,
        n_resamples=n_perm,
        alternative="greater",
        random_state=0,
    )
    stats["perm_p"] = float(res.pvalue)
    return stats


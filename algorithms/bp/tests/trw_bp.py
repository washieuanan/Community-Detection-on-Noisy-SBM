import numpy as np
import networkx as nx
from typing import Dict, Tuple, Optional, Callable

def tree_reweighted_belief_propagation(
    G: nx.Graph,
    q: int,
    weight_func: Optional[Callable[[float], float]] = None,
    rho: Optional[Dict[Tuple[int,int], float]] = None,
    max_iter: int = 100,
    tol: float = 1e-4,
    damping: float = 0.5,
    seed: Optional[int] = None,
):
    """
    Perform tree-reweighted belief propagation (TRW-BP) for community detection.

    Returns beliefs, MAP labels, and final messages.
    """
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    psi_i = {i: np.ones(q) for i in nodes}
    psi_ij = {}
    for u, v, data in G.edges(data=True):
        w = weight_func(data.get('dist', 1.0)) if weight_func else 1.0
        mat = np.ones((q, q), dtype=float)
        np.fill_diagonal(mat, np.exp(w))
        psi_ij[(u, v)] = psi_ij[(v, u)] = mat
    if rho is None:
        rho = {(u, v): 1.0 for u, v in G.edges()}
        rho.update({(v, u): val for (u, v), val in rho.items()})
    messages = {}
    for u, v in G.edges():
        m_uv = rng.random(q) + 1e-6
        m_vu = rng.random(q) + 1e-6
        messages[(u, v)] = m_uv / m_uv.sum()
        messages[(v, u)] = m_vu / m_vu.sum()
    for _ in range(max_iter):
        max_diff = 0.0
        new_msgs = {}
        for (u, v), old_msg in messages.items():
            prod = psi_i[u].copy()
            for w in G.neighbors(u):
                if w == v:
                    continue
                prod *= messages[(w, u)] ** rho.get((w, u), 1.0)
            mat = psi_ij[(u, v)] ** rho.get((u, v), 1.0)
            m = mat.T.dot(prod)
            m += 1e-12
            m /= m.sum()
            m = damping * old_msg + (1 - damping) * m
            new_msgs[(u, v)] = m
            max_diff = max(max_diff, np.abs(m - old_msg).max())
        messages = new_msgs
        if max_diff < tol:
            break
    beliefs = {}
    for u in nodes:
        b = psi_i[u].copy()
        for w in G.neighbors(u):
            b *= messages[(w, u)] ** rho.get((w, u), 1.0)
        b += 1e-12
        b /= b.sum()
        beliefs[u] = b
    labels = {u: int(np.argmax(beliefs[u])) for u in nodes}
    return beliefs, labels, messages

def detect_communities_trw(
    G: nx.Graph,
    q: int,
    observations: list,
    weight_func: Callable[[float], float],
    C_frac: float = 0.05,
    **bp_params
):
    """
    Build subgraph from observations and run TRW-BP.
    observations: list of (u, v, dist)
    C_frac: max fraction of avg degree to use
    """
    avg_deg = np.mean([d for _, d in G.degree()])
    max_obs = max(1, int(0.5 * avg_deg * len(G) * C_frac))
    obs = observations[:max_obs]
    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))
    for (u, v), dist in obs:
        H.add_edge(u, v, dist=dist)
    _, labels, _ = tree_reweighted_belief_propagation(
        H, q, weight_func=weight_func, **bp_params
    )
    return labels

if __name__ == "__main__":
    # Use Geometric Block Model (GBM) with PairSamplingObservation
    from graph_generation.gbm import generate_gbm
    from observations.standard_observe import PairSamplingObservation

    # Parameters
    n = 600
    K = 2
    a = 70
    b = 18
    seed = 42

    # Generate GBM graph with true community labels in 'comm'
    G_true = generate_gbm(n=n, K=K, a=a, b=b, seed=seed)

    # Sample pairs (includes distances)
    avg_deg = np.mean([deg for _, deg in G_true.degree()])
    original_density = avg_deg / n
    C = 0.02 * original_density
    num_pairs = int(C * n**2 / 2)
    sampler = PairSamplingObservation(
        G_true,
        num_samples=num_pairs,
        seed=seed
    )
    observations = sampler.observe()  # yields (u, v, dist)

    # Define weight on distance
    weight_func = lambda d: np.exp(-0.5 * d)

    # Run TRW‑BP
    labels_pred = detect_communities_trw(
        G_true,
        q=K,
        observations=observations,
        weight_func=weight_func,
        C_frac=0.05,
        max_iter=2000,
        tol=1e-3,
        damping=0.2,
        seed=seed
    )   

    # Evaluate accuracy (allow label flip)
    true_comm = {u: G_true.nodes[u]['comm'] for u in G_true.nodes()}
    pred = labels_pred
    acc_dir = sum(pred[u] == true_comm[u] for u in G_true) / n
    acc_flip = sum(pred[u] != true_comm[u] for u in G_true) / n
    acc = max(acc_dir, acc_flip)
    print(f"Accuracy: {acc:.3f} (direct={acc_dir:.3f}, flipped={acc_flip:.3f})")

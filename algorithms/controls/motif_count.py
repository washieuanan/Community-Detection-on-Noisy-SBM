import numpy as np
import networkx as nx
from collections import deque

def motif_counting(G: nx.Graph, q: int):
    """
    Vectorized GBM community recovery via triangle‐counting on the largest connected
    component of G; nodes outside that component are assigned at random.

    Parameters
    ----------
    G : nx.Graph
        Undirected graph (may be disconnected).
    q : int
        Number of communities.

    Returns
    -------
    labels : dict
        Mapping node -> community label in {0, 1, ..., q-1}.
    """
    comps = list(nx.connected_components(G))
    lcc = max(comps, key=len)
    other_nodes = set(G.nodes()) - lcc

    G_sub = G.subgraph(lcc)
    nodes_sub = list(G_sub.nodes())
    n_sub = len(nodes_sub)
    idx_sub = {node: i for i, node in enumerate(nodes_sub)}

    A_sub = nx.to_numpy_array(G_sub, nodelist=nodes_sub, dtype=int)
    T_sub = A_sub @ A_sub

    tri_counts = T_sub[A_sub == 1]
    if tri_counts.size == 0:
        raise ValueError("Largest component has no edges.")
    threshold = np.median(tri_counts)

    assigned = np.zeros(n_sub, dtype=bool)
    labels_sub = -np.ones(n_sub, dtype=int)
    clusters = []

    # 6) Grow up to q clusters via motif‐BFS
    for comm in range(q):
        unassigned_idxs = np.where(~assigned)[0]
        if unassigned_idxs.size == 0:
            break
        seed = unassigned_idxs[0]
        assigned[seed] = True
        labels_sub[seed] = comm
        clusters.append({seed})
        queue = deque([seed])

        while queue:
            u = queue.popleft()
            cands = np.where((~assigned) & (T_sub[u, :] >= threshold))[0]
            for v in cands:
                assigned[v] = True
                labels_sub[v] = comm
                clusters[comm].add(v)
                queue.append(v)

        if assigned.all():
            break

    if not assigned.all():
        seeds = np.array([next(iter(c)) for c in clusters]) if clusters else np.array([], dtype=int)
        for v in np.where(~assigned)[0]:
            if seeds.size > 0:
                counts = T_sub[v, seeds]
                best_comm = int(np.argmax(counts))
            else:
                best_comm = 0  # fallback if no clusters found
            assigned[v] = True
            labels_sub[v] = best_comm
            clusters[best_comm].add(v)

    labels = {nodes_sub[i]: int(labels_sub[i]) for i in range(n_sub)}

    rng = np.random.default_rng()
    for node in other_nodes:
        labels[node] = int(rng.integers(0, q))

    return np.array([labels[i] for i in range(len(G.nodes))])

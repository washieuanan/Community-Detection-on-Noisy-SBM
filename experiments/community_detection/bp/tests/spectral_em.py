#!/usr/bin/env python3
"""
EM‑style geometry‑spectral community detection for a latent Geometric Block Model
constrained to the unit sphere.

Author: ChatGPT (May‑2025)
"""
from __future__ import annotations

import random
from typing import Sequence

import numpy as np
import networkx as nx
from scipy.linalg import eigh
from sklearn.cluster import KMeans

#!/usr/bin/env python3
"""
Geometry‑Spectral EM (v2) for extremely sparse Geometric Block Models.

Public entry point
------------------
    labels = em_geo_spectral(G_obs, K, grid_centers,
                             gamma=1.5, beta=3.0,
                             geom_mix=0.3, max_iters=1000, seed=42)

The signature is unchanged so your existing driver runs as‑is.
"""



# ----------------------------------------------------------------------
#  Spectral embedding :  Bethe–Hessian  (works at degree ≈ O(1))
# ----------------------------------------------------------------------
def _bethe_hessian_embedding(A: np.ndarray, q: int, rng: np.random.Generator) -> np.ndarray:
    k = A.sum(axis=1)
    r = np.sqrt(max(k.mean(), 1e-8))      # non‑back‑tracking radius
    H = (r**2 - 1) * np.eye(A.shape[0]) - r * A + np.diag(k)
    # q smallest eigenvectors (H is symmetric)
    _, vecs = eigh(H, subset_by_index=[0, q - 1])
    return vecs


def _spectral_clustering(A: np.ndarray, K: int, rng: np.random.Generator) -> np.ndarray:
    emb = _bethe_hessian_embedding(A, K, rng)
    return KMeans(n_clusters=K, n_init=10, random_state=rng.integers(1e9)).fit_predict(emb)


# ----------------------------------------------------------------------
#  Geometry E‑step  (soft posterior, with optional annealing)
# ----------------------------------------------------------------------
def _geometry_expectation(
    G: nx.Graph,
    grid: np.ndarray,
    coords_prev: np.ndarray,
    gamma: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return expected coords (n,dim) and posterior matrix (n,G).
    """
    n, Gsz = len(G), grid.shape[0]
    delta = np.linalg.norm(grid[:, None, :] - coords_prev[None, :, :], axis=-1)  # (G,n)
    post = np.full((n, Gsz), 1.0 / Gsz, dtype=float)

    for v in G.nodes():
        neigh = list(G.neighbors(v))
        if not neigh:
            continue
        d_obs = np.array([G[v][u]["dist"] for u in neigh])
        cost = np.sum((delta[:, neigh].T - d_obs[:, None]) ** 2, axis=0)
        p = np.exp(-gamma * cost)
        p /= p.sum()
        post[v] = p

    coords_exp = post @ grid
    return coords_exp, post


# ----------------------------------------------------------------------
#  Additive geometry mask  (close ⇒ small boost, far ⇒ big boost)
# ----------------------------------------------------------------------
def _sigmoid_confidence(dist_mat: np.ndarray, beta: float) -> np.ndarray:
    d0 = np.median(dist_mat[dist_mat > 0])
    return 1.0 / (1.0 + np.exp(-beta * (dist_mat - d0)))


def _build_weighted_adjacency(
    A_obs: np.ndarray,
    coords: np.ndarray,
    beta: float,
    geom_mix: float,
) -> np.ndarray:
    dist_mat = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    W = _sigmoid_confidence(dist_mat, beta=beta)
    A_mod = A_obs + geom_mix * W            # ADDITIVE, never weakens an edge
    A_mod = 0.5 * (A_mod + A_mod.T)
    np.fill_diagonal(A_mod, 0.0)
    return A_mod


# ----------------------------------------------------------------------
#  Main EM routine
# ----------------------------------------------------------------------
def em_geo_spectral(
    G_obs: nx.Graph,
    K: int,
    grid_centers: np.ndarray,
    *,
    gamma: float = 1.5,
    beta: float = 3.0,
    geom_mix: float = 0.3,
    max_iters: int = 100,
    gamma_anneal: float = 1.2,          # 1 → no annealing
    gamma_cap: float = 4.0,
    seed: int | None = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Parameters keep the same meaning as v1; new `gamma_anneal` allows gradual
    sharpening of posteriors.  Set it to 1 to disable.
    """
    rng = np.random.default_rng(seed)
    nodes = list(G_obs.nodes())
    n = len(nodes)

    # Binary observed adjacency
    A_obs = nx.to_numpy_array(G_obs, nodelist=nodes, weight=None)

    # Initial community labels (Bethe‑Hessian on raw observed graph)
    labels = _spectral_clustering(A_obs, K, rng)

    # Start with random grid assignment for coords
    coords = grid_centers[rng.integers(grid_centers.shape[0], size=n)]

    for it in range(1, max_iters + 1):
        if verbose:
            print(f"[EM] iteration {it}   (γ={gamma:.3g})")

        # Geometry E‑step
        coords, _ = _geometry_expectation(G_obs, grid_centers, coords, gamma, rng)

        # Geometry‑aware adjacency
        A_mod = _build_weighted_adjacency(A_obs, coords, beta, geom_mix)

        # Spectral M‑step
        new_labels = _spectral_clustering(A_mod, K, rng)

        changed = np.count_nonzero(new_labels != labels)
        if verbose:
            print(f"    label changes: {changed}/{n}")
        if changed == 0:
            if verbose:
                print("    converged.")
            break
        labels = new_labels

        # Anneal γ
        gamma = min(gamma * gamma_anneal, gamma_cap)

    return labels


# -------------------------------------------------------------------------
#                            Usage example
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
#  Demo / sanity‑check for the geometry‑spectral EM algorithm
# -------------------------------------------------------------------------
if __name__ == "__main__":
    from graph_generation.gbm import generate_gbm
    from observations.standard_observe import PairSamplingObservation, get_coordinate_distance
    from community_detection.bp.vectorized_geometric_bp import (
        detection_stats,
        get_true_communities,
    )
    
    # 1) Generate a latent GBM for testing
    n, K = 600, 2
    G_true = generate_gbm(n=n, K=K, a=70, b=18, seed=123)

    # 2) Take sparse pair‑sampling observations exactly like before
    avg_deg = np.mean([G_true.degree[v] for v in G_true.nodes()])
    original_density = avg_deg / n
    C = 0.05 * original_density        # ↓ sparsity level

    def weight_func(c1, c2):
        return 1.0

    num_pairs = int(C * n**2 / 2)
    sampler = PairSamplingObservation(
        G_true,
        num_samples=num_pairs,
        weight_func=weight_func,
        seed=42,
    )
    observations = sampler.observe()

    # 3) Build the *observed* graph (all vertices present, only observed edges)
    import numpy as np, networkx as nx

    G_obs = nx.Graph()
    G_obs.add_nodes_from(range(n))
    for (u, v), dist in observations:
        G_obs.add_edge(u, v, dist=float(dist))

    # 4) Prepare geometric grid (here: 2 n Fibonacci‑sphere points)
    def fibonacci_sphere(m: int, d: int = 3) -> np.ndarray:
        phi = (1 + 5 ** 0.5) / 2
        theta = 2 * np.pi * np.arange(m) / phi
        z = 1 - 2 * np.arange(m) / (m - 1)
        r = np.sqrt(1 - z ** 2)
        x, y = r * np.cos(theta), r * np.sin(theta)
        return np.column_stack((x, y, z))[:, :d]

    grid_centres = fibonacci_sphere(2 * n)

    # 5) Run our EM loop
    print("Running geometry‑spectral EM …")
    labels_pred = em_geo_spectral(
        G_obs,
        K=K,
        grid_centers=grid_centres,
        max_iters=1000,
        geom_mix=0.3,
        seed=42,
        gamma=1.5, 
        beta=3.0,
        verbose=True,
    )

    # 6) Evaluate community‑detection performance
    node_order = list(G_obs.nodes())
    true_labels = get_true_communities(G_true, node2idx={v: i for i, v in enumerate(node_order)}, attr="comm")
    stats = detection_stats(labels_pred, true_labels)

    print("\n=== Community‑detection accuracy (geometry‑spectral EM) ===")
    for k, v in stats.items():
        print(f"{k:>25s} : {v:.4f}")

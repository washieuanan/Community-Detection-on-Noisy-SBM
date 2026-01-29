
import numpy as np
import networkx as nx

from algorithms.duo_spec import (
    duo_spec,
    detection_stats,
    get_true_communities,
)
import os
import json
import logging
import random
from algorithms.bp.vectorized_bp import belief_propagation_weighted

def coords_str2arr(G: nx.Graph, dim = 16):
    """
    for each coord, convert string formatted coord to numpy array
    """
    new_G = nx.Graph()
    for n in G.nodes():
        coord_str = G.nodes[n]["coords"]
        coord_arr = np.fromstring(coord_str, sep=",")
        if len(coord_arr) != dim:
            coord_arr = np.zeros(dim)
        new_G.add_node(int(n), coords=coord_arr, asin=G.nodes[n]["asin"], comm=G.nodes[n]["comm"])
        
    for u, v in G.edges():
        if "dist" in G.edges[u, v]:
            dist = G.edges[u, v]["dist"]
            if isinstance(dist, str):
                dist = float(dist)
            new_G.add_edge(int(u), int(v), dist=dist)
    new_G.graph = G.graph.copy()
    new_G = nx.relabel.convert_node_labels_to_integers(new_G, first_label=0)
    return new_G

if __name__ == "__main__":

    G = nx.read_gml("amazon_metadata_test/amz_allviddvd.gml")
    G = coords_str2arr(G)

    print(f"Testing on classes: {G.graph['subclasses']} and {len(G.nodes())} nodes")
    print(f"Created Graph with {len(G.nodes())} nodes and {len(G.edges())} edges")

    K = 2

    # BP on original graph (pre-denoising)
    beliefs_pre, preds_pre, node2idx_pre, _ = belief_propagation_weighted(
        G,
        q=K,
        max_iter=200,
        tol=1e-4,
        damping=0.5,
        seed=0,
        init="random",
    )
    true_communities = get_true_communities(G, node2idx=node2idx_pre, attr="comm")
    stats_pre = detection_stats(preds_pre, true_communities)

    # DuoSpec denoising (structural)
    res = duo_spec(
        G,
        K=K,
        max_em_iters=75,
        min_em_iters=10,
        w_min=0.05,
        w_cap=3.0,
        conv_tol=1e-10,
        conv_window=10,
        update_scale=0.8,
        lam_geo=0.30,
        lam_comm_boost=0.02,
        S0=20,
        frac_sweep=(0.995, 0.99, 0.98, 0.97, 0.95),
        local_score="cn_over_sqrtdeg",
        geo_gate_enabled=True,
        gate_power=1.0,
        gate_floor=0.10,
        stable_k=2,
        delta_cap=0.10,
        use_comm_boost=True,
        random_state=0,
    )

    G_den = res["G_final"]

    # BP on denoised graph (post-denoising)
    beliefs_post, preds_post, node2idx_post, _ = belief_propagation_weighted(
        G_den,
        q=K,
        max_iter=200,
        tol=1e-4,
        damping=0.5,
        seed=0,
        init="random",
    )
    true_communities_post = get_true_communities(G_den, node2idx=node2idx_post, attr="comm")
    stats_post = detection_stats(preds_post, true_communities_post)

    # Extract correlation metrics from DuoSpec (fallback if missing)
    proxy_before = res.get("proxy_corr_before")
    proxy_after = res.get("proxy_corr_after")
    proxy_delta = res.get("proxy_corr_delta")
    if proxy_before is None or proxy_after is None or proxy_delta is None:
        from algorithms.duo_spec import proxy_weight_locality_correlation

        proxy_before = proxy_weight_locality_correlation(
            G, weight_key="weight", local_score="cn_over_sqrtdeg", corr="spearman"
        )
        proxy_after = proxy_weight_locality_correlation(
            G_den, weight_key="weight", local_score="cn_over_sqrtdeg", corr="spearman"
        )
        before_val = proxy_before.get("spearman", float("nan"))
        after_val = proxy_after.get("spearman", float("nan"))
        proxy_delta = after_val - before_val

    coord_before = res.get("coord_corr_before")
    coord_after = res.get("coord_corr_after")
    coord_delta = res.get("coord_corr_delta")
    if coord_before is None or coord_after is None or coord_delta is None:
        from algorithms.duo_spec import weight_coord_distance_correlation

        coord_before = weight_coord_distance_correlation(
            G, coord_key="coords", weight_key="weight", corr="spearman"
        )
        coord_after = weight_coord_distance_correlation(
            G_den, coord_key="coords", weight_key="weight", corr="spearman"
        )
        cb = coord_before.get("spearman", float("nan"))
        ca = coord_after.get("spearman", float("nan"))
        coord_delta = ca - cb

    # Build result row
    acc_pre = float(stats_pre.get("accuracy", float("nan")))
    acc_post = float(stats_post.get("accuracy", float("nan")))
    acc_delta = acc_post - acc_pre

    row = {
        "dataset": G.graph.get("subclasses", ""),
        "n": G.number_of_nodes(),
        "m": G.number_of_edges(),
        "K": K,
        "acc_pre": acc_pre,
        "acc_post": acc_post,
        "acc_delta": acc_delta,
        "proxy_spearman_before": float(proxy_before.get("spearman", float("nan"))),
        "proxy_spearman_after": float(proxy_after.get("spearman", float("nan"))),
        "proxy_spearman_delta": float(proxy_delta),
        "coord_spearman_before": float(coord_before.get("spearman", float("nan"))),
        "coord_spearman_after": float(coord_after.get("spearman", float("nan"))),
        "coord_spearman_delta": float(coord_delta),
        "max_em_iters": 75,
        "min_em_iters": 10,
        "w_min": 0.05,
        "w_cap": 3.0,
        "update_scale": 0.8,
        "lam_geo": 0.30,
        "lam_comm_boost": 0.02,
        "delta_cap": 0.10,
        "gate_power": 1.0,
        "gate_floor": 0.10,
    }

    os.makedirs("results/amazon_orig", exist_ok=True)
    out_path = os.path.join("results/amazon_orig", "amazon_orig_duospec_bp.csv")
    try:
        import pandas as pd

        pd.DataFrame([row]).to_csv(out_path, index=False)
    except ImportError:
        import csv

        write_header = not os.path.exists(out_path)
        with open(out_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    
        
            
        
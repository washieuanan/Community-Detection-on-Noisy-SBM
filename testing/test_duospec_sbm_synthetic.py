import os
from typing import List, Dict, Any

import networkx as nx
import numpy as np
import pandas as pd

from block_models.sbm.sbm import generate_noisy_sbm
from algorithms.duo_spec import (
    duo_spec,
    bethe_hessian,
    detection_stats,
    get_true_communities,
)
from algorithms.spectral_ops.attention import motif_spectral_embedding


def _ensure_edge_weights(G: nx.Graph) -> None:
    """Ensure every edge has a numeric 'weight' attribute."""
    for _, _, d in G.edges(data=True):
        d.setdefault("weight", 1.0)


def _accuracy_bethe(
    G: nx.Graph,
    K: int,
    random_state: int = 0,
) -> float:
    """Bethe–Hessian spectral clustering accuracy."""
    _ensure_edge_weights(G)
    Q, hard, node2idx, _ = bethe_hessian(
        G,
        q=K,
        random_state=random_state,
        weight_from_dist=False,   # use existing edge weights (or 1.0)
    )
    true_labels = get_true_communities(G, node2idx=node2idx, attr="comm")
    stats = detection_stats(hard, true_labels)
    return float(stats["accuracy"])


def _accuracy_motif(
    G: nx.Graph,
    K: int,
    random_state: int = 0,
) -> float:
    """Motif spectral embedding accuracy."""
    _ensure_edge_weights(G)
    Q, hard, node2idx, _ = motif_spectral_embedding(
        G,
        q=K,
        random_state=random_state,
    )
    true_labels = get_true_communities(G, node2idx=node2idx, attr="comm")
    stats = detection_stats(hard, true_labels)
    return float(stats["accuracy"])


def run_duospec_sbm_experiment(
    num_graphs: int = 10,
    *,
    n: int = 300,
    K: int = 2,
    p_in: float = 0.12,
    p_out: float = 0.04,
    sigma: float = 0.5,
    output_csv: str = "results/duospec_sbm_eval.csv",
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic graphs via `generate_noisy_sbm` and evaluate:

    1) Bethe–Hessian spectral clustering (control, pre‑denoising).
    2) Motif spectral embedding (control, pre‑denoising).
    3) Both (1) and (2) on the DuoSpec‑denoised graph.
    4) Pre/post proxy geometry‑noise correlation from DuoSpec.
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    rng = np.random.default_rng(random_seed)
    records: List[Dict[str, Any]] = []

    for g_idx in range(num_graphs):
        seed = int(rng.integers(0, 2**32 - 1))
        print(f"\n=== Graph {g_idx+1}/{num_graphs} (seed={seed}) ===")

        # 1) Generate graph
        G_true = generate_noisy_sbm(
            n=n,
            K=K,
            p_in=p_in,
            p_out=p_out,
            sigma=sigma,
            seed=seed,
        )
        print(
            f"Generated SBM graph with n={len(G_true.nodes())}, "
            f"m={len(G_true.edges())}"
        )

        # 2) Pre‑denoising accuracies
        try:
            acc_bh_pre = _accuracy_bethe(G_true, K=K, random_state=seed)
        except Exception as e:
            print(f"[WARN] Bethe-Hessian pre-denoising failed: {e}")
            acc_bh_pre = np.nan

        try:
            acc_motif_pre = _accuracy_motif(G_true, K=K, random_state=seed)
        except Exception as e:
            print(f"[WARN] motif_spectral_embedding pre-denoising failed: {e}")
            acc_motif_pre = np.nan

        # 3) DuoSpec denoising (structure-only GeoDe)
        try:
            res_duo = duo_spec(
                G_true,
                K=K,
                geo_method="A_persistence_no_coords",
                local_score="cn_over_sqrtdeg",
            )
            G_denoised = res_duo["G_final"]
            proxy_before = res_duo["proxy_corr_before"]["corr_value"]
            proxy_after = res_duo["proxy_corr_after"]["corr_value"]
            proxy_delta = res_duo["proxy_corr_delta"]
        except Exception as e:
            print(f"[ERROR] duo_spec denoising failed: {e}")
            G_denoised = G_true
            proxy_before = np.nan
            proxy_after = np.nan
            proxy_delta = np.nan

        # 4) Post‑denoising accuracies
        try:
            acc_bh_post = _accuracy_bethe(G_denoised, K=K, random_state=seed)
        except Exception as e:
            print(f"[WARN] Bethe-Hessian post-denoising failed: {e}")
            acc_bh_post = np.nan

        try:
            acc_motif_post = _accuracy_motif(G_denoised, K=K, random_state=seed)
        except Exception as e:
            print(f"[WARN] motif_spectral_embedding post-denoising failed: {e}")
            acc_motif_post = np.nan

        records.append(
            dict(
                graph_id=g_idx,
                seed=seed,
                n=n,
                K=K,
                p_in=p_in,
                p_out=p_out,
                sigma=sigma,
                acc_bh_pre=acc_bh_pre,
                acc_motif_pre=acc_motif_pre,
                acc_bh_post=acc_bh_post,
                acc_motif_post=acc_motif_post,
                proxy_corr_before=proxy_before,
                proxy_corr_after=proxy_after,
                proxy_corr_delta=proxy_delta,
            )
        )

    df = pd.DataFrame.from_records(records)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved DuoSpec SBM evaluation results to '{output_csv}'")
    return df


if __name__ == "__main__":
    run_duospec_sbm_experiment()


import os
from typing import List, Dict, Any, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from block_models.sbm.sbm import generate_noisy_sbm
from algorithms.duo_spec import (
    duo_spec,
    bethe_hessian,
    detection_stats,
    get_true_communities,
    rescale_graph_weights_for_downstream,
    compute_initial_avg_degree,
    prune_for_bh_weight_aware,
)
from algorithms.spectral_ops.attention import motif_spectral_embedding
from algorithms.bp.vectorized_bp import belief_propagation_weighted


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
        beta=0.2,
        use_edge_weights=True,
        weight_key="weight",
        weight_pow=0.5,
    )
    true_labels = get_true_communities(G, node2idx=node2idx, attr="comm")
    stats = detection_stats(hard, true_labels)
    return float(stats["accuracy"])


def _accuracy_bp(
    G: nx.Graph,
    K: int,
    random_state: int = 0,
) -> float:
    """Belief propagation accuracy (weighted by current edge weights)."""
    try:
        _, preds, node2idx, _ = belief_propagation_weighted(
            G,
            q=K,
            seed=random_state,
            init="spectral",
        )
    except Exception as e:
        print(f"[WARN] BP failed: {e}")
        return np.nan

    true_labels = get_true_communities(G, node2idx=node2idx, attr="comm")
    stats = detection_stats(preds, true_labels)
    return float(stats["accuracy"])


def _squash_weights_for_motif(
    G: nx.Graph,
    *,
    weight_key: str = "weight",
    lo_pct: float = 5.0,
    hi_pct: float = 95.0,
) -> nx.Graph:
    H = G.copy()
    if H.number_of_edges() == 0:
        return H
    w_vals = np.array(
        [float(d.get(weight_key, 1.0)) for _, _, d in H.edges(data=True)],
        dtype=float,
    )
    w_lo = float(np.percentile(w_vals, lo_pct))
    w_hi = float(np.percentile(w_vals, hi_pct))
    if not np.isfinite(w_lo) or not np.isfinite(w_hi) or w_hi <= w_lo:
        return H
    for u, v, d in H.edges(data=True):
        w = float(d.get(weight_key, 1.0))
        w_s = min(max(w, w_lo), w_hi)
        w_sq = 0.5 + 1.5 * (w_s - w_lo) / (w_hi - w_lo + 1e-12)
        d[weight_key] = float(w_sq)
    return H


def run_duospec_sbm_experiment(
    num_graphs: int = 1,
    *,
    n: int = 900,
    K: int = 2,
    p_in: float = 0.49,
    p_out: float = 0.02,
    sigma: float = 0.10,
    output_csv: str = "results/duospec_sbm_eval.csv",
    random_seed: int = 42,
    metric_debug: bool = False,
) -> pd.DataFrame:
    """
    Generate synthetic graphs via `generate_noisy_sbm` and evaluate:

    1) Bethe–Hessian spectral clustering (control, pre‑denoising).
    2) Motif spectral embedding (control, pre‑denoising).
    3) Both (1) and (2) on the DuoSpec‑denoised graph.
    4) Pre/post proxy geometry‑locality correlation from DuoSpec.
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

        try:
            acc_bp_pre = _accuracy_bp(G_true, K=K, random_state=seed)
        except Exception as e:
            print(f"[WARN] BP pre-denoising failed: {e}")
            acc_bp_pre = np.nan

        try:
            # Use explicit weight bounds so we can reuse them in rescaling.
            res_duo = duo_spec(
                G_true,
                K=K,
                local_score="cn_over_sqrtdeg",
                metric_debug=metric_debug,
                community_proxy="leiden"
            )
            G_denoised = res_duo["G_final"]
            hist = res_duo.get("history", [])
            if hist:
                h_last = hist[-1]
                mean_abs_delta = float(h_last.get("mean_abs_delta", np.nan))
                frac_w_min = float(h_last.get("frac_w_min", np.nan))
                frac_w_cap = float(h_last.get("frac_w_cap", np.nan))
                mean_delta_geo = float(h_last.get("mean_delta_geo", np.nan))
                mean_delta_comm = float(h_last.get("mean_delta_comm", np.nan))
                frac_boosted = float(h_last.get("frac_boosted", np.nan))
                frac_shrunk = float(h_last.get("frac_shrunk", np.nan))
            else:
                mean_abs_delta = np.nan
                frac_w_min = np.nan
                frac_w_cap = np.nan
                mean_delta_geo = np.nan
                mean_delta_comm = np.nan
                frac_boosted = np.nan
                frac_shrunk = np.nan
            # Sanity check: ensure denoising produced non-constant weights when edges exist,
            # and plot a histogram of the resulting edge-weight distribution.

            proxy_before = res_duo["proxy_corr_before"]["corr_value"]
            proxy_after = res_duo["proxy_corr_after"]["corr_value"]
            proxy_delta = res_duo["proxy_corr_delta"]
            coord_before = res_duo.get("coord_corr_before", {})
            coord_after = res_duo.get("coord_corr_after", {})
            coord_delta = res_duo.get("coord_corr_delta", np.nan)
        except Exception as e:
            print(f"[ERROR] duo_spec denoising failed: {e}")
            G_denoised = G_true
            proxy_before = np.nan
            proxy_after = np.nan
            proxy_delta = np.nan
            coord_before = np.nan
            coord_after = np.nan
            coord_delta = np.nan
            mean_abs_delta = np.nan
            frac_w_min = np.nan
            frac_w_cap = np.nan
            mean_delta_geo = np.nan
            mean_delta_comm = np.nan
            frac_boosted = np.nan
            frac_shrunk = np.nan

        # 4) Post‑denoising accuracies:
        #    - Bethe–Hessian on 0/1 pruned graph (bottom 25% edges removed, connectivity preserved).
        #    - Motif on weighted denoised graph with log-domain edge weights.
        #    - BP on full denoised weighted graph (unchanged).
        # 4) Post‑denoising accuracies:
        #    - Bethe–Hessian on 0/1 pruned graph with target mean degree = original.
        #    - Motif on weighted denoised graph with log-domain edge weights (and optional squash).
        #    - BP on full denoised weighted graph (unchanged).
        # Original mean degree
        mean_deg0 = compute_initial_avg_degree(G_true)

        try:
            G_bh = prune_for_bh_weight_aware(
                G_denoised,
                weight_key="weight",
                mean_degree_target=mean_deg0,
            )
            acc_bh_post = _accuracy_bethe(G_bh, K=K, random_state=seed)
        except Exception as e:
            print(f"[WARN] Bethe-Hessian post-denoising failed: {e}")
            acc_bh_post = np.nan

        try:
            acc_motif_post = _accuracy_motif(G_denoised, K=K, random_state=seed)
        except Exception as e:
            print(f"[WARN] motif_spectral_embedding post-denoising failed: {e}")
            acc_motif_post = np.nan

        try:
            acc_bp_post = _accuracy_bp(G_denoised, K=K, random_state=seed)
        except Exception as e:
            print(f"[WARN] BP post-denoising failed: {e}")
            acc_bp_post = np.nan

        # For BH and Motif we do not rescale weights; postR equals post.
        acc_bh_post_rescaled = acc_bh_post
        acc_motif_post_rescaled = acc_motif_post
        # BP also uses raw denoised weights without additional rescaling.
        acc_bp_post_rescaled = acc_bp_post

        # Per-seed compact summary
        d_bh = acc_bh_post - acc_bh_pre
        d_motif = acc_motif_post - acc_motif_pre
        d_bp = acc_bp_post - acc_bp_pre
        print(
            f"    Seed={seed}: "
            f"BH pre={acc_bh_pre:.3f} post={acc_bh_post:.3f} (Δ{d_bh:+.3f}) "
            f"postR={acc_bh_post_rescaled:.3f} (Δ{acc_bh_post_rescaled - acc_bh_pre:+.3f}) | "
            f"Motif pre={acc_motif_pre:.3f} post={acc_motif_post:.3f} (Δ{d_motif:+.3f}) "
            f"postR={acc_motif_post_rescaled:.3f} (Δ{acc_motif_post_rescaled - acc_motif_pre:+.3f}) | "
            f"BP pre={acc_bp_pre:.3f} post={acc_bp_post:.3f} (Δ{d_bp:+.3f}) "
            f"postR={acc_bp_post_rescaled:.3f} (Δ{acc_bp_post_rescaled - acc_bp_pre:+.3f})"
        )

        records.append(
            dict(
                graph_id=g_idx,
                seed=seed,
                n=n,
                K=K,
                p_in=p_in,
                p_out=p_out,
                sigma=sigma,
                geo_discriminator="dSu_fineblob_persistence",
                acc_bh_pre=acc_bh_pre,
                acc_motif_pre=acc_motif_pre,
                acc_bp_pre=acc_bp_pre,
                acc_bh_post=acc_bh_post,
                acc_motif_post=acc_motif_post,
                acc_bp_post=acc_bp_post,
                acc_bh_post_rescaled=acc_bh_post_rescaled,
                acc_motif_post_rescaled=acc_motif_post_rescaled,
                acc_bp_post_rescaled=acc_bp_post_rescaled,
                mean_abs_delta=mean_abs_delta,
                frac_w_min=frac_w_min,
                frac_w_cap=frac_w_cap,
                mean_delta_geo=mean_delta_geo,
                mean_delta_comm=mean_delta_comm,
                frac_boosted=frac_boosted,
                frac_shrunk=frac_shrunk,
                proxy_corr_before=proxy_before,
                proxy_corr_after=proxy_after,
                proxy_corr_delta=proxy_delta,
                coord_corr_before=coord_before,
                coord_corr_after=coord_after,
                coord_corr_delta=coord_delta,
            )
        )

    df = pd.DataFrame.from_records(records)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved DuoSpec SBM evaluation results to '{output_csv}'")

    # Final mean accuracies and deltas (single structural DuoSpec configuration)
    print("\n=== Summary over all graphs (structural DuoSpec) ===")
    for key, label in [
        ("acc_bh", "Bethe-Hessian"),
        ("acc_motif", "Motif"),
        ("acc_bp", "BP"),
    ]:
        pre = df[f"{key}_pre"]
        post = df[f"{key}_post"]
        postR = df[f"{key}_post_rescaled"]
        pre_mean = float(pre.mean())
        post_mean = float(post.mean())
        postR_mean = float(postR.mean())
        delta_mean = float((post - pre).mean())
        deltaR_mean = float((postR - pre).mean())
        print(
            f"  {label:12s} "
            f"pre={pre_mean:.4f} "
            f"post={post_mean:.4f} Δpost={delta_mean:+.4f} "
            f"postR={postR_mean:.4f} ΔpostR={deltaR_mean:+.4f}"
        )

    print("\n=== Denoiser statistics (final EM iteration, averaged over all runs) ===")
    for key in [
        "mean_abs_delta",
        "frac_w_min",
        "frac_w_cap",
        "mean_delta_geo",
        "mean_delta_comm",
        "frac_boosted",
        "frac_shrunk",
    ]:
        if key in df.columns:
            print(f"  {key:16s}: {float(df[key].mean()):+.4e}")

    return df


if __name__ == "__main__":
    run_duospec_sbm_experiment()


"""
Correlation tracking experiment for DuoSpec denoising.

Tracks distance-based and locality-based Spearman correlations
over EM iterations during graph denoising.
"""

import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from typing import Optional
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from block_models.sbm.sbm import generate_noisy_sbm
from algorithms.duo_spec import (
    duo_spec,
    weight_distance_correlation,
    proxy_weight_locality_correlation,
)


def track_correlations_during_denoising(
    G: nx.Graph,
    K: int,
    max_em_iters: int = 100,
    csv_path: Optional[str] = None,
    **duo_spec_kwargs
):
    """
    Run DuoSpec and track correlations at each EM iteration.
    
    Returns:
        DataFrame with columns: iteration, distance_spearman, locality_spearman
    """
    # Initialize tracking
    tracking_data = []
    
    # Make a copy of the graph for tracking
    G_current = deepcopy(G)
    for _, _, d in G_current.edges(data=True):
        d.setdefault("weight", 1.0)
    
    # Track initial correlations (iteration 0)
    dist_corr_0 = weight_distance_correlation(
        G_current,
        coord_key="coords",
        weight_key="weight",
        corr="spearman",
    )
    loc_corr_0 = proxy_weight_locality_correlation(
        G_current,
        weight_key="weight",
        local_score="cn_over_sqrtdeg",
        corr="spearman",
    )
    
    tracking_data.append({
        "iteration": 0,
        "distance_spearman": dist_corr_0.get("spearman", np.nan),
        "locality_spearman": loc_corr_0.get("spearman", np.nan),
        "distance_p_value": dist_corr_0.get("spearman_p", np.nan),
        "locality_p_value": loc_corr_0.get("p_value", np.nan),
    })
    
    # Save initial state to CSV if path provided
    if csv_path:
        df_temp = pd.DataFrame(tracking_data)
        df_temp.to_csv(csv_path, index=False)
    
    print(f"[Tracking] Iteration 0: distance={dist_corr_0.get('spearman', np.nan):.4f}, "
          f"locality={loc_corr_0.get('spearman', np.nan):.4f}")
    
    # Run DuoSpec one iteration at a time to track intermediate states
    for iteration in range(1, max_em_iters + 1):
        # Run a single EM iteration
        # Use very loose convergence settings to ensure it always runs one iteration
        result = duo_spec(
            G_current,
            K=K,
            max_em_iters=1,
            min_em_iters=1,
            conv_tol=1e-10,  # Very loose tolerance
            conv_window=1000,  # Very large window to prevent early convergence
            **duo_spec_kwargs
        )
        
        # Get the updated graph from the result
        G_current = result["G_final"]
        
        # Track correlations at this iteration
        dist_corr = weight_distance_correlation(
            G_current,
            coord_key="coords",
            weight_key="weight",
            corr="spearman",
        )
        loc_corr = proxy_weight_locality_correlation(
            G_current,
            weight_key="weight",
            local_score="cn_over_sqrtdeg",
            corr="spearman",
        )
        
        dist_spearman = dist_corr.get("spearman", np.nan)
        loc_spearman = loc_corr.get("spearman", np.nan)
        
        tracking_data.append({
            "iteration": iteration,
            "distance_spearman": dist_spearman,
            "locality_spearman": loc_spearman,
            "distance_p_value": dist_corr.get("spearman_p", np.nan),
            "locality_p_value": loc_corr.get("p_value", np.nan),
        })
        
        # Save to CSV incrementally
        if csv_path:
            df_temp = pd.DataFrame(tracking_data)
            df_temp.to_csv(csv_path, index=False)
        
        print(f"[Tracking] Iteration {iteration}/{max_em_iters}: "
              f"distance={dist_spearman:.4f}, locality={loc_spearman:.4f}")
    
    return pd.DataFrame(tracking_data)


def run_experiment(
    n: int = 10000,
    K: int = 3,
    p_in: float = 0.7,
    p_out: float = 0.05,
    sigma: float = 0.1,
    max_em_iters: int = 100,
    seed: int = 42,
    output_dir: Optional[str] = None,
):
    """
    Run the correlation tracking experiment.
    """
    # Set default output directory relative to project root
    if output_dir is None:
        # Get project root (assuming script is in experiments/correlations/)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        output_dir = os.path.join(project_root, "experiments", "correlations")
    
    print(f"[Experiment] Generating SBM graph: n={n}, K={K}, "
          f"p_in={p_in}, p_out={p_out}, sigma={sigma}")
    
    # Generate noisy SBM graph
    G = generate_noisy_sbm(
        n=n,
        K=K,
        p_in=p_in,
        p_out=p_out,
        sigma=sigma,
        dim=2,
        seed=seed,
    )
    
    print(f"[Experiment] Graph generated: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # CSV file path
    csv_path = os.path.join(output_dir, "correlation_tracking.csv")
    
    # Track correlations during denoising (saves incrementally to CSV)
    print(f"[Experiment] Starting correlation tracking over {max_em_iters} iterations...")
    df = track_correlations_during_denoising(
        G,
        K=K,
        max_em_iters=max_em_iters,
        csv_path=csv_path,  # Pass CSV path for incremental saving
    )
    
    # Final save (should already be done incrementally, but ensure it's saved)
    df.to_csv(csv_path, index=False)
    print(f"[Experiment] Saved tracking data to {csv_path}")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot distance correlation
    axes[0].plot(df["iteration"], df["distance_spearman"], "b-", linewidth=2, label="Distance Spearman")
    axes[0].set_xlabel("EM Iteration")
    axes[0].set_ylabel("Spearman Correlation")
    axes[0].set_title("Weight-Distance Correlation Over Iterations")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot locality correlation
    axes[1].plot(df["iteration"], df["locality_spearman"], "r-", linewidth=2, label="Locality Spearman")
    axes[1].set_xlabel("EM Iteration")
    axes[1].set_ylabel("Spearman Correlation")
    axes[1].set_title("Weight-Locality Correlation Over Iterations")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, "correlation_tracking.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"[Experiment] Saved plot to {plot_path}")
    
    # Also create a combined plot
    fig2, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(df["iteration"], df["distance_spearman"], "b-", linewidth=2, 
            label="Distance Spearman", marker="o", markersize=3)
    ax.plot(df["iteration"], df["locality_spearman"], "r-", linewidth=2, 
            label="Locality Spearman", marker="s", markersize=3)
    ax.set_xlabel("EM Iteration", fontsize=12)
    ax.set_ylabel("Spearman Correlation", fontsize=12)
    ax.set_title("Correlation Tracking During Denoising", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    combined_plot_path = os.path.join(output_dir, "correlation_tracking_combined.png")
    plt.savefig(combined_plot_path, dpi=300, bbox_inches="tight")
    print(f"[Experiment] Saved combined plot to {combined_plot_path}")
    
    return df


if __name__ == "__main__":
    # Run the experiment
    df = run_experiment(
        n=10000,
        K=3,
        p_in=0.7,
        p_out=0.05,
        sigma=0.1,
        max_em_iters=100,
        seed=42,
    )
    
    print("\n[Experiment] Summary:")
    print(f"  Final distance correlation: {df['distance_spearman'].iloc[-1]:.4f}")
    print(f"  Final locality correlation: {df['locality_spearman'].iloc[-1]:.4f}")
    print(f"  Distance correlation change: {df['distance_spearman'].iloc[-1] - df['distance_spearman'].iloc[0]:.4f}")
    print(f"  Locality correlation change: {df['locality_spearman'].iloc[-1] - df['locality_spearman'].iloc[0]:.4f}")

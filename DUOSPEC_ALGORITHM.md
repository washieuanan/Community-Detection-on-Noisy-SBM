# DuoSpec Algorithm Documentation

## Overview

**DuoSpec** (Dual-Spectrum Denoising) is a purely structural graph denoising algorithm that iteratively reweights edges to distinguish between **geometry-induced** and **community-induced** edges. The algorithm operates entirely on graph structure—no node coordinates or spectral embeddings are used in the denoising process (coordinates may be used only for evaluation metrics).

### Key Principles

- **No edge pruning**: All edges remain in the graph; only their weights are adjusted
- **Structure-only**: The algorithm uses only graph topology and edge weights, never node coordinates
- **Iterative refinement**: Uses an EM-style loop to progressively refine edge weights
- **Two-channel reweighting**: Shrinks geometry-induced edges while optionally boosting community-consistent edges

## Algorithm Structure

### High-Level Flow

```
1. Initialize edge weights (default: 1.0)
2. Compute baseline node strengths
3. For each EM iteration:
   a. Compute edge locality scores
   b. Build fine-blob partition (geometry discrimination)
   c. Build coarse community proxy
   d. Reweight edges based on geometry + community signals
   e. Optionally preserve node strengths
   f. Check convergence
4. Return denoised graph with updated weights
```

## Core Components

### 1. Edge Locality Scores

**Function**: `edge_locality_scores(G, edges, metric="cn_over_sqrtdeg")`

Computes a structural locality score for each edge that measures how "local" the edge is within the graph structure.

**Available metrics**:
- `"cn_over_sqrtdeg"`: Common neighbors normalized by square root of degrees
- `"cn"`: Raw common neighbor count
- `"jaccard"`: Jaccard coefficient of neighborhoods

**Purpose**: High locality scores indicate edges that are likely geometry-induced (short-range connections), while low scores suggest community bridges.

### 2. Fine-Blob Geometry Discrimination

**Function**: `geometry_scores_fineblob_persistence(...)`

This is the core geometry discriminator that identifies which edges are likely geometry-induced.

#### Process:

1. **Fine-blob partition**:
   - Selects top-k highest-locality edges (using exact top-k selection, not percentile thresholds)
   - Builds a DSU (Disjoint Set Union) over these edges to form "fine blobs"
   - Target number of blobs scales with graph size: `target_blobs ≈ 0.50 * n^0.745`
   - Binary search over k values to hit target blob count within ±15% tolerance

2. **Multi-threshold persistence**:
   - For each threshold in `frac_sweep` (e.g., [0.98, 0.95, 0.90, 0.85, 0.80]):
     - Select top-k edges at that threshold
     - Run DSU to find connected components
     - Count edges that lie in small components (size ≤ `S0`)
   - **Persistence score**: Fraction of thresholds where an edge appears in a small component
   - **Sharp persistence**: Requires `stable_k` occurrences across thresholds

3. **Geometry responsibility** (`r_geo`):
   - An edge gets `r_geo > 0` only if:
     - It lies within the same fine blob (`same_blob = True`)
     - The blob size is small (≤ `S0`)
     - It has sufficient persistence across thresholds
   - `r_geo` ranges from 0 (community-induced) to 1 (geometry-induced)

4. **Bridge score**:
   - `bridge_score = 1.0` if endpoints lie in different fine blobs
   - `bridge_score = 0.0` if endpoints are in the same blob

### 3. Community Proxy

**Function**: `community_proxy_persistence_on_blob_graph(...)` or Louvain/Leiden variants

Builds a coarse community partition to identify which edges are within the same community.

#### Process:

1. **Blob supergraph**:
   - Contracts fine blobs into supernodes
   - Edge weights aggregate non-geometry mass: `w_blob = Σ (1 - r_geo[e]) * w[e]`

2. **Coarse partition**:
   - **DSU mode**: Uses persistence-based DSU on blob graph to find exactly K communities
   - **Louvain/Leiden mode**: Runs community detection with exact-K enforcement
   - Produces `coarse_id_per_blob` labels

3. **Soft `p_same`**:
   - For each original edge, computes probability that endpoints are in the same coarse community
   - Uses blob-supergraph weights and sigmoid normalization
   - `p_same ≈ 1.0` for strong within-community edges, `≈ 0.0` for cross-community edges

### 4. Edge Reweighting

**Function**: `reweight_edges_from_posteriors(...)`

Applies multiplicative weight updates based on geometry and community signals.

#### Update Rule:

For each edge `e` with current weight `w`:

1. **Geometry shrink signal**:
   ```
   r_eff = r_geo * gate
   gate = (1 - p_same)^gate_power  (if geo_gate_enabled)
   gate = max(gate, gate_floor)     (prevents total suppression)
   
   # Optional fallback geometry suspicion
   if geo_fallback_alpha > 0:
       r_base = r_geo + geo_fallback_alpha * fallback
       r_eff = r_base * gate
   
   delta_geo = -lam_geo * r_eff
   ```

2. **Community boost signal** (selective):
   ```
   if p_same >= comm_boost_psame_thr AND r_geo <= comm_boost_rgeo_max:
       r_comm = (1 - r_geo) * p_same
       delta_comm = +lam_comm_boost * r_comm
   else:
       delta_comm = 0.0
   ```

3. **Combined update**:
   ```
   delta = delta_geo + delta_comm
   delta = clip(delta, -delta_cap, +delta_cap)
   w_new = w * (1 + update_scale * delta)
   w_new = clamp(w_new, w_min, w_cap)
   ```

**Key properties**:
- Multiplicative updates preserve relative ordering
- Geometry-induced edges are shrunk (downweighted)
- Strong community edges are optionally boosted
- Updates are capped to prevent instability

### 5. Strength Preservation

**Function**: Applied after each reweighting step (if `strength_preserve=True`)

Softly renormalizes edge weights so each node's weighted degree (strength) stays close to its initial value.

```
For each node u:
  s_u = current weighted degree
  f_u = (base_strength[u] / s_u)^strength_eta

For each edge (u, v):
  w_new = w * (sqrt(f_u * f_v))^strength_eta
  w_new = clamp(w_new, w_min, w_cap)
```

**Purpose**: Prevents weighted degree distortion that could harm downstream methods like Bethe-Hessian or Motif spectral embedding.

## EM Iteration Loop

### Main Loop Structure

```python
for em in range(1, max_em_iters + 1):
    # 1. Geometry discrimination
    scores_local = edge_locality_scores(subG, edges, metric=local_score)
    geo_info = geometry_scores_fineblob_persistence(...)
    r_geo = geo_info["r_geo"]
    bridge_score = geo_info["bridge_score"]
    
    # 2. Community proxy
    blob_edges, blob_weights = build_blob_supergraph(...)
    coarse_ids = community_proxy(...)  # DSU/Louvain/Leiden
    p_same = compute_soft_psame_from_blob_graph(...)
    
    # 3. Reweight edges
    geo_stats = reweight_edges_from_posteriors(...)
    
    # 4. Optional strength preservation
    if strength_preserve:
        preserve_node_strengths(...)
    
    # 5. Check convergence
    obj = -geo_stats["mean_abs_delta"]
    if converged:
        break
```

### Convergence Criterion

- **Strict convergence**: After `min_em_iters`, checks if `max(|Δobj|)` over last `conv_window` iterations ≤ `conv_tol`
- **Objective**: Negative mean absolute weight change (larger changes = better denoising progress)

## Key Parameters

### Geometry Controls

- `lam_geo` (default: 0.15): Strength of geometry shrink
- `S0` (default: 50): Maximum fine-blob size for geometry classification
- `stable_k` (default: 3): Minimum persistence occurrences across thresholds
- `frac_sweep` (default: [0.98, 0.95, 0.90, 0.85, 0.80]): Persistence threshold levels
- `local_score` (default: "cn_over_sqrtdeg"): Locality metric

### Community Controls

- `lam_comm_boost` (default: 0.015): Strength of community boost
- `comm_boost_psame_thr` (default: 0.96): Minimum `p_same` for boost eligibility
- `comm_boost_rgeo_max` (default: 0.30): Maximum `r_geo` for boost eligibility
- `community_proxy` (default: "dsu"): Method for coarse partition ("dsu", "louvain", "leiden")
- `geo_gate_enabled` (default: True): Enable community gating of geometry shrink
- `gate_power` (default: 1.0): Power for community gate
- `gate_floor` (default: 0.1): Minimum gate value to prevent total suppression

### Fine-Blob Scaling

- `blob_scale_alpha` (default: 0.745): Power-law exponent for blob count scaling
- `blob_scale_c` (default: 0.50): Scaling constant
- Target: `target_blobs ≈ 80` at n=900, `≈ 1500` at n=46000
- `blob_target_rel_tol` (default: 0.15): ±15% tolerance band around target

### Update Controls

- `update_scale` (default: 0.8): Global scaling factor for all updates
- `delta_cap` (default: 0.10): Maximum per-iteration multiplicative change
- `w_min` (default: 0.05): Minimum edge weight
- `w_cap` (default: 3.0): Maximum edge weight

### Strength Preservation

- `strength_preserve` (default: True): Enable node strength preservation
- `strength_eta` (default: 0.20): Blending factor for strength correction

### Fallback Geometry

- `geo_fallback_alpha` (default: 0.35): Strength of fallback geometry suspicion
- `geo_fallback_mode` (default: "locality"): Fallback signal source ("locality", "bridge", "both")

## Output

The algorithm returns a dictionary with:

- `G_final`: Denoised graph with updated edge weights
- `geo_stats`: Statistics about geometry discrimination (mean_abs_delta, frac_shrunk, frac_boosted, etc.)
- `proxy_corr_before/after`: Correlation between weights and locality proxy (before/after denoising)
- `coord_corr_before/after`: Correlation between weights and coordinate distances (if coordinates available)
- `hist`: History of EM iterations with objective values and statistics

## Downstream Postprocessing

After denoising, different downstream methods may require different postprocessing:

- **BP (Belief Propagation)**: Optional log-space normalization via `bp_postprocess_log_squash()`
- **BH (Bethe-Hessian)**: Weight-aware pruning to preserve original mean degree via `prune_for_bh_weight_aware()`
- **Motif**: Uses denoised weights directly with sublinear attention weighting (no postprocessing)

## Design Philosophy

1. **Structure-only**: Never uses node coordinates in the algorithm (only for metrics)
2. **No pruning**: All edges remain; only weights change
3. **Deterministic**: Exact top-k selection prevents tie-breaking issues
4. **Scalable**: Blob count scales with graph size via power-law
5. **Robust**: Multiple safeguards (gating, strength preservation, delta capping) prevent instability

## References

The algorithm implements a fast, structural variant of geometric denoising (GeoDe) that distinguishes geometry-induced noise from community structure using only graph topology.

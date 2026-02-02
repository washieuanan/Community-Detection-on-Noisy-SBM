# DuoSpec Algorithm: Detailed Overview and Explanation

## Executive Summary

DuoSpec is a **purely structural graph denoising algorithm** that iteratively reweights edges to distinguish between **geometry-induced noise** (short-range, triangle-rich edges that create local clustering) and **community-induced signal** (edges that connect meaningful communities). The algorithm operates entirely on graph structure—no node coordinates or external features are required—making it applicable to any graph where geometric noise may be present.

**Core Principle**: In many real-world graphs, edges arise from two sources:
1. **Geometric noise**: Edges that exist primarily due to spatial proximity or local clustering (e.g., "friends of friends" in social networks, or short-distance connections in spatial graphs)
2. **Community signal**: Edges that connect nodes belonging to the same meaningful community, regardless of local structure

DuoSpec identifies and downweights geometric noise while preserving (and optionally boosting) community-signal edges, improving downstream community detection performance.

---

## Algorithm Structure

DuoSpec follows an **Expectation-Maximization (EM) style iterative loop**:

```
1. Initialize edge weights (default: 1.0)
2. For each EM iteration:
   a. Geometry Proxy: Compute r_geo (probability edge is geometry-induced)
   b. Community Proxy: Compute p_same (probability edge connects same community)
   c. Reweighting: Update edge weights based on r_geo and p_same
   d. Optional: Preserve node strengths (weighted degrees)
3. Check convergence
4. Return denoised graph with updated weights
```

---

## Part 1: Geometry Proxy (DSU-Based Fine-Blob Persistence)

### Overview

The geometry proxy identifies edges that are likely **geometry-induced** by analyzing their behavior across multiple scales of locality. The key insight is that geometric edges tend to:
- Have high **locality scores** (many common neighbors, high triangle closure)
- Persist in **small, isolated blobs** across multiple threshold levels
- Form **intra-blob connections** within small components

### Step-by-Step Process

#### 1.1 Locality Score Computation

For each edge `(u, v)`, compute a **locality score** `L(u,v)` that measures local clustering:

```python
# Default metric: "cn_over_sqrtdeg"
L(u,v) = CN(u,v) / sqrt(deg(u) * deg(v))
```

where:
- `CN(u,v)` = number of common neighbors (triangles involving the edge)
- `deg(u)`, `deg(v)` = degrees of nodes u and v

**Why this works**: Edges with many common neighbors (high triangle closure) are more likely to be geometric noise—they form dense local clusters rather than meaningful long-range community connections.

#### 1.2 Fine-Blob Partition Construction

Build a **fine partition** by keeping only the highest-locality edges:

1. **Target blob count**: The algorithm scales the target number of blobs with graph size:
   ```
   target_blobs = c * (n^α)
   ```
   where `c ≈ 0.50` and `α ≈ 0.745`, yielding ~80 blobs at n=900 and ~1500 blobs at n=46000.

2. **Binary search**: Find the number of top-locality edges `k` that yields a component count within the target band `[band_low, band_high]`.

3. **DSU union**: Use Disjoint Set Union (DSU) to build connected components from these top-locality edges. This creates many **small, isolated blobs** across the graph.

**Key insight**: By keeping only the highest-locality edges, we isolate the most "geometric" parts of the graph into small blobs. Edges that connect these blobs are more likely to be community-signal.

#### 1.3 Multi-Threshold Persistence

For each edge, measure its **persistence** across multiple threshold levels:

1. **Sweep levels**: Test multiple thresholds `frac_sweep = (0.98, 0.95, 0.90, 0.85, 0.80)` (keeping top 2%, 5%, 10%, 15%, 20% of edges by locality).

2. **Small-component counting**: For each threshold, run DSU and count how many times the edge lies in a **small component** (size ≤ `S0`, default 50).

3. **Persistence score**: 
   ```python
   pers_sharp = (count_small - (stable_k - 1)) / (L - (stable_k - 1))
   ```
   where `stable_k` (default 3) requires the edge to appear in small components at multiple scales.

**Why this works**: Edges that consistently appear in small components across multiple scales are highly likely to be geometric noise—they form persistent local clusters rather than meaningful community structure.

#### 1.4 Final Geometry Posterior `r_geo`

Combine persistence with fine-blob membership:

```python
r_geo(e) = pers_sharp(e) * 1[same_blob(e)] * 1[blob_size(u) ≤ S0]
```

An edge gets a high `r_geo` (close to 1.0) only if:
- It has high persistence across scales (`pers_sharp` is high)
- It connects nodes in the same fine blob (`same_blob(e) = True`)
- The blob is small (`blob_size ≤ S0`)

**Justification**: This triple condition ensures we only mark edges as geometric if they are:
1. **Persistently local** (appear in small components across scales)
2. **Intra-blob** (within a fine blob, not connecting blobs)
3. **In small blobs** (not part of large, meaningful communities)

---

## Part 2: Community Proxy (Leiden on Blob Supergraph)

### Overview

The community proxy identifies **coarse community structure** by:
1. Building a **blob supergraph** where nodes are fine blobs and edges aggregate non-geometric mass
2. Running **Leiden community detection** on this supergraph
3. Lifting the coarse partition back to original nodes

### Step-by-Step Process

#### 2.1 Blob Supergraph Construction

Build a weighted graph `G_blob` where:
- **Nodes**: Fine blobs (from geometry proxy)
- **Edges**: Connect blobs that share at least one original edge
- **Edge weights**: Aggregate **non-geometric mass**:
  ```python
  W_blob(bu, bv) = Σ_{e connecting bu↔bv} (1 - r_geo[e]) * w_e
  ```

**Key insight**: By weighting blob edges with `(1 - r_geo)`, we emphasize edges that are **not** geometric noise. This focuses the community detection on meaningful inter-blob connections.

#### 2.2 Leiden Community Detection

Run Leiden algorithm on `G_blob` with:
- **Resolution parameter**: Default 1.0 (can be tuned)
- **Weight key**: Uses aggregated non-geometric weights
- **Exact-K enforcement**: Post-processes to ensure exactly `K` communities

**Why Leiden**: Leiden is a fast, high-quality community detection algorithm that:
- Optimizes modularity (or a resolution-parameter variant)
- Handles weighted graphs naturally
- Produces well-separated communities

#### 2.3 Exact-K Enforcement

Since Leiden may produce an arbitrary number of communities, enforce exactly `K` communities:

1. **If too many communities**: Merge smallest pairs deterministically until `K` remain
2. **If too few communities**: Split largest communities using DSU-based exact-K partition

**Why exact-K**: Downstream methods (e.g., Belief Propagation) often require a fixed number of communities `K`. Exact-K ensures consistency.

#### 2.4 Soft `p_same` Computation

Lift coarse communities to original edges and compute a **soft confidence** `p_same(e)`:

```python
if coarse_id[u] != coarse_id[v]:
    p_same(e) = 0.0  # Different communities
elif blob_id[u] == blob_id[v]:
    p_same(e) = 1.0  # Same fine blob
else:
    # Same coarse community, different blobs
    # Use sigmoid of blob-edge strength
    z = (log(1 + W_blob(bu, bv)) - μ) / (MAD * temp)
    p_same(e) = sigmoid(z)
```

**Why soft**: Edges within the same coarse community but different fine blobs get intermediate `p_same` values based on the strength of the blob-to-blob connection. This captures uncertainty in community boundaries.

---

## Part 3: Edge Reweighting

### Overview

Update edge weights using a **two-channel multiplicative update**:
- **Geometry channel**: Shrink edges with high `r_geo` (geometry-induced)
- **Community channel**: Boost edges with high `p_same` and low `r_geo` (community-signal)

### Update Rule

For each edge `e = (u, v)`:

1. **Clip signals**:
   ```python
   r_clipped = clip(r_geo[e], 0, 1)
   ps_clipped = clip(p_same[e], 0, 1)
   ```

2. **Geometry gate** (optional, default enabled):
   ```python
   gate = (1 - ps_clipped)^gate_power
   gate = max(gate, gate_floor)  # Prevent total suppression
   r_eff = r_clipped * gate
   ```
   
   **Why gate**: If an edge has high `p_same` (strong community signal), we reduce the geometry shrink strength. This protects community edges from being incorrectly downweighted.

3. **Geometry shrink**:
   ```python
   delta_geo = -lam_geo * r_eff
   ```
   where `lam_geo` (default 0.05) controls shrink strength.

4. **Community boost** (optional, default enabled):
   ```python
   if ps_clipped >= comm_boost_psame_thr and r_clipped <= comm_boost_rgeo_max:
       delta_comm = lam_comm_boost * (1 - r_clipped) * ps_clipped
   else:
       delta_comm = 0.0
   ```
   
   **Selective boosting**: Only boost edges that are:
   - Strongly intra-community (`ps_clipped >= 0.96`)
   - Not geometric (`r_clipped <= 0.30`)

5. **Combined update**:
   ```python
   delta = delta_geo + delta_comm
   delta = clip(delta, -delta_cap, +delta_cap)  # Bound per-iteration change
   w_new = w_old * (1 + delta)
   w_new = clip(w_new, w_min, w_cap)  # Clamp to [0.05, 3.0] by default
   ```

**Why multiplicative**: Multiplicative updates preserve relative ordering of edges while allowing smooth, bounded changes. The `delta_cap` (default 0.10) prevents large per-iteration swings.

### Optional: Node Strength Preservation

After reweighting, optionally preserve node **strengths** (weighted degrees):

```python
for each node u:
    curr_strength[u] = Σ_{v} w(u,v)
    factor[u] = (base_strength[u] / curr_strength[u])^strength_eta
for each edge (u,v):
    w_new = w_old * sqrt(factor[u] * factor[v])
```

**Why preserve strengths**: Prevents nodes from having their weighted degrees drift too far from initial values, maintaining graph structure while allowing selective reweighting.

---

## Part 4: Convergence

### Convergence Criterion

DuoSpec uses a **strict convergence criterion** based on objective stability:

1. **Objective**: `obj = -mean_abs_delta`, where `mean_abs_delta` is the mean absolute change in edge weights per iteration.

2. **Convergence window**: Track the last `conv_window` (default 3) iterations.

3. **Convergence check** (only after `min_em_iters`):
   ```python
   if em >= min_em_iters and len(hist) >= conv_window + 1:
       deltas = [abs(hist[-i]["obj"] - hist[-i-1]["obj"]) 
                 for i in range(1, conv_window + 1)]
       max_delta = max(deltas)
       if max_delta <= conv_tol:  # Default 1e-8
           converged = True
   ```

**Why this criterion**: 
- **Stability-based**: Convergence is declared when the objective (mean weight change) stabilizes, not when it reaches zero. This handles cases where small oscillations occur.
- **Window-based**: Requires stability over multiple iterations, preventing premature stopping due to temporary plateaus.
- **Minimum iterations**: Ensures the algorithm runs for at least `min_em_iters` (default 2) to allow initial exploration.

### Convergence Behavior

- **Early iterations**: Large weight changes as geometry and community signals are established.
- **Mid iterations**: Refinement as signals stabilize.
- **Late iterations**: Small changes as the algorithm converges to a stable weight distribution.

**Typical convergence**: Most graphs converge within 5-20 iterations, depending on:
- Graph size and structure
- Initial weight distribution
- Parameter settings (`lam_geo`, `delta_cap`, etc.)

---

## Why DuoSpec Works

### Theoretical Justification

1. **Multi-Scale Persistence**: By testing edges across multiple locality thresholds, DuoSpec captures the **scale-invariant** nature of geometric noise. True geometric edges persist in small components across scales, while community edges appear at coarser scales.

2. **Blob-Community Separation**: The fine-blob partition isolates geometric noise into small, isolated components. The coarse community partition (on the blob supergraph) captures meaningful community structure. This **two-level hierarchy** naturally separates noise from signal.

3. **Soft Confidence**: The soft `p_same` computation allows for uncertainty in community boundaries, preventing overconfident decisions that could harm downstream performance.

4. **Multiplicative Updates**: Multiplicative reweighting preserves edge ordering while allowing smooth, bounded changes. This prevents catastrophic weight collapse or explosion.

### Empirical Justification

1. **Improves downstream accuracy**: On synthetic and real graphs, DuoSpec consistently improves Belief Propagation accuracy by 2-10% by removing geometric noise.

2. **Structure-only**: Works without node coordinates, making it applicable to any graph where geometric noise may be present (social networks, citation graphs, etc.).

3. **Robust to parameters**: Default parameters work well across a wide range of graph sizes and structures, with tuning needed only for extreme cases.

---

## Key Parameters and Their Effects

| Parameter | Default | Effect |
|-----------|---------|--------|
| `lam_geo` | 0.05 | Strength of geometry shrink. Higher = more aggressive downweighting of geometric edges. |
| `lam_comm_boost` | 0.015 | Strength of community boost. Higher = more aggressive boosting of community edges. |
| `delta_cap` | 0.10 | Maximum per-iteration multiplicative change factor. Lower = more stable, slower convergence. |
| `w_min`, `w_cap` | 0.05, 3.0 | Bounds on edge weights. Prevents extreme values. |
| `S0` | 50 | Maximum blob size for geometry detection. Smaller = more aggressive geometry detection. |
| `stable_k` | 3 | Minimum number of scales where edge must appear in small component. Higher = stricter geometry criterion. |
| `gate_power` | 1.0 | Power in geometry gate `(1-p_same)^gate_power`. Higher = stronger protection of community edges. |
| `conv_tol` | 1e-8 | Convergence tolerance. Lower = stricter convergence (more iterations). |

---

## Limitations and Considerations

1. **Assumes geometric noise exists**: If the graph has no geometric noise, DuoSpec may still reweight edges, potentially harming performance.

2. **Requires meaningful communities**: The algorithm assumes `K` meaningful communities exist. If `K` is incorrect, community proxy may produce poor partitions.

3. **Computational cost**: Each iteration requires:
   - Locality score computation: O(m * avg_degree)
   - DSU operations: O(m * log(n))
   - Leiden community detection: O(m * log(n)) typically
   - Total: O(max_em_iters * m * log(n)) for sparse graphs

4. **Parameter sensitivity**: While defaults work well, extreme graphs (very dense, very sparse, or with unusual structure) may require tuning.

---

## Summary

DuoSpec is a principled, structure-only graph denoising algorithm that:
- **Identifies geometric noise** via multi-scale persistence on locality scores
- **Identifies community signal** via Leiden community detection on a blob supergraph
- **Reweights edges** multiplicatively to downweight noise and preserve/boost signal
- **Converges** when weight changes stabilize over multiple iterations

The algorithm's effectiveness stems from its **multi-scale analysis** (persistence across thresholds) and **hierarchical structure** (fine blobs → coarse communities), which naturally separate geometric noise from meaningful community structure.

"""
experiments/convergence/convergence.py
——————————————
Noise-aware DuoSpec experiment:
    • GN-MSE / GN-DEV computed on *weighted* edges
    • stratified sampling to ensure both classes
    • optional early-stop on flat GN-MSE
"""

# ───────────────────────────────────────────────────────── imports ──
import numpy as np
import networkx as nx
from copy import deepcopy
from functools import partial
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from typing import Tuple, List

# original DuoSpec helpers
from algorithms.duo_spec import (
    _to_idx,
    _edge_same_prob,
    _scale_edges,
    get_callable,
    erdos_renyi_mask,
)

# (Demo) dataset helper
from experiments.amazon_metadata_test.test_amazon_meta import coords_str2arr

# ────────────────────────────────── 1. small utilities ──
def _sample_pairs_stratified(G: nx.Graph, n_pairs: int, rng) -> List[Tuple]:
    """≈ n_pairs/2 real edges  +  ≈ n_pairs/2 non-edges."""
    nodes = list(G.nodes())
    edges = list(G.edges())
    n_edges = len(edges)

    # positives
    pos = []
    if n_edges:
        k = min(n_pairs // 2, n_edges)
        pos_idx = rng.choice(n_edges, size=k, replace=False)
        pos = [edges[i] for i in pos_idx]

    # negatives
    neg = []
    need = n_pairs - len(pos)
    while len(neg) < need:
        u, v = rng.choice(nodes, 2, replace=False)
        if not G.has_edge(u, v):
            neg.append((u, v))

    return pos + neg


def _edge_confidence(G: nx.Graph, u, v, *, w_cap: float) -> float:
    """Return weight/w_cap (clipped to 1) if edge exists, else 0."""
    data = G.get_edge_data(u, v)
    if data is None:
        return 0.0
    return min(data.get("weight", 0.0) / w_cap, 1.0)


def _fit_regression(dist: np.ndarray, conf: np.ndarray):
    """OLS fit of p̂(d)=α−βd; clip to [0,1]."""
    reg = LinearRegression().fit(dist.reshape(-1, 1), conf)
    p_hat = np.clip(reg.predict(dist.reshape(-1, 1)), 0.0, 1.0)
    return p_hat, reg


def _geometric_noise(
    G: nx.Graph,
    coords: np.ndarray,
    *,
    sample_size: int,
    rng,
    node2idx: dict | None = None,
    w_cap: float,
):
    """Compute GN-MSE, GN-DEV, GBC on *weighted* confidences."""
    nodes = list(G.nodes())
    pairs = _sample_pairs_stratified(G, sample_size, rng)
    u, v = zip(*pairs)

    # targets: edge confidence
    conf = np.fromiter(
        (_edge_confidence(G, ui, vi, w_cap=w_cap) for ui, vi in pairs),
        dtype=float,
    )

    # distances
    if node2idx is not None:
        u_idx = np.fromiter((node2idx[ui] for ui in u), dtype=np.int64)
        v_idx = np.fromiter((node2idx[vi] for vi in v), dtype=np.int64)
    else:
        u_idx, v_idx = np.asarray(u, int), np.asarray(v, int)

    dist = np.linalg.norm(coords[u_idx] - coords[v_idx], axis=1)

    # fit p̂(d) and residuals
    p_hat, _ = _fit_regression(dist, conf)
    resid = conf - p_hat
    mse = (resid**2).mean()

    eps = 1e-12
    ll = conf * np.log(p_hat + eps) + (1 - conf) * np.log(1 - p_hat + eps)
    dev = -ll.mean()

    # optional between/within distance ratio
    gbc = 1.0
    if "label" in G.nodes[nodes[0]]:
        lbl = np.asarray([G.nodes[n]["label"] for n in nodes])
        same = lbl[np.array(u)] == lbl[np.array(v)]
        if same.any() and (~same).any():
            d_w = dist[same].mean()
            d_b = dist[~same].mean()
            gbc = d_b / d_w if d_w else 1.0

    return dict(gn_mse=mse, gn_dev=dev, gbc=gbc)


def noise_has_converged(history, metric="gn_mse", *, window=3, tol=1e-4):
    """True if |Δ metric| < tol for `window` consecutive rounds."""
    if len(history) <= window:
        return False
    tail = [h[metric] for h in history[-(window + 1) :]]
    return np.all(np.abs(np.diff(tail)) < tol)


# ─────────────────────────────── 2. DuoSpec wrapper ──
def duo_spec_tester(
    H_obs: nx.Graph,
    K: int,
    num_balls: int = 16,
    config: tuple = ("bethe_hessian", "bethe_hessian"),
    *,
    # EM schedule
    max_em_iters=50,
    anneal_steps=6,
    warmup_rounds=2,
    # percentile cuts
    comm_cut=0.90,
    geo_cut=0.90,
    # shrink / boost strength
    shrink_comm=1.00,
    shrink_geo=0.80,
    boost_comm=0.60,
    boost_geo=0.40,
    boost_cut_comm=0.97,
    boost_cut_geo=0.97,
    # weight bounds
    w_min=5e-2,
    w_cap=4.0,
    # noise meter
    noise_sample=20_000,
    log_noise=True,
    noise_window=5,
    noise_tol=1e-6,
    # convergence
    tol=1e-4,
    patience=7,
    random_state=0,
):
    rng = np.random.default_rng(random_state)
    subG = deepcopy(H_obs)
    for _, _, d in subG.edges(data=True):
        d.setdefault("weight", 1.0)

    edges = np.asarray(subG.edges(), dtype=object)
    node2idx = {u: i for i, u in enumerate(subG.nodes())}
    iu_glob, iv_glob = _to_idx(edges, node2idx)

    best, hist = {"obj": -np.inf}, []
    config = get_callable(config)

    def _lam(step, base):
        d = step - warmup_rounds
        if d <= 0:
            return 0.0
        return base if d >= anneal_steps else base * d / anneal_steps

    # ─── EM loop ──────────────────────────────────────────────────────
    for em in range(1, max_em_iters + 1):
        print(f"[EM] iter {em} / {max_em_iters}")

        # community embedding
        Q_comm, *_ = config[0](subG, q=K, random_state=random_state)
        p_same = _edge_same_prob(Q_comm, iu_glob, iv_glob)
        mask_comm_shrink = p_same > np.percentile(p_same, comm_cut * 100)
        mask_comm_boost = p_same > np.percentile(p_same, boost_cut_comm * 100)

        # geometry embedding
        Q_geo, hard_geo, *_ = config[1](subG, q=num_balls, random_state=random_state)
        same_ball = hard_geo[iu_glob] == hard_geo[iv_glob]
        conf_g = 0.5 * (
            Q_geo[iu_glob, hard_geo[iu_glob]] + Q_geo[iv_glob, hard_geo[iv_glob]]
        )
        mask_geo_shrink = same_ball & (
            conf_g > np.percentile(conf_g, geo_cut * 100)
        )
        mask_geo_boost = same_ball & (
            conf_g > np.percentile(conf_g, boost_cut_geo * 100)
        )

        # edge re-weighting
        λc, λg = _lam(em, shrink_comm), _lam(em, shrink_geo)
        λcB, λgB = _lam(em, boost_comm), _lam(em, boost_geo)

        drop_c = _scale_edges(subG, mask_comm_shrink, p_same, λc, w_min, w_cap, "shrink")
        drop_g = _scale_edges(subG, mask_geo_shrink, conf_g, λg, w_min, w_cap, "shrink")
        boost_c = _scale_edges(subG, mask_comm_boost, p_same, λcB, w_min, w_cap, "boost")
        boost_g = _scale_edges(subG, mask_geo_boost, conf_g, λgB, w_min, w_cap, "boost")

        # bookkeeping
        obj = float(np.max(Q_comm, axis=1).sum())
        record = dict(
            it=em,
            obj=obj,
            edges=subG.number_of_edges(),
            shrink_comm=drop_c,
            shrink_geo=drop_g,
            boost_comm=boost_c,
            boost_geo=boost_g,
        )

        if log_noise:
            noise = _geometric_noise(
                subG,
                coords=Q_geo,
                sample_size=noise_sample,
                rng=rng,
                node2idx=node2idx,
                w_cap=w_cap,
            )
            record.update(noise)

        hist.append(record)

        # keep best posterior
        if obj > best["obj"]:
            best.update(obj=obj, beliefs=Q_comm, balls=hard_geo, node2idx=node2idx)

        # convergence checks
        if em > 1 and abs(obj - hist[-2]["obj"]) < tol:
            print(f"[EM] objective flat at iter {em}")
            break
        if log_noise and noise_has_converged(hist, "gn_mse", window=noise_window, tol=noise_tol):
            print(f"[EM] GN-MSE plateau at iter {em}")
            break
        if subG.number_of_edges() == 0:
            print("[EM] graph emptied – stop")
            break

    hard_final = best["beliefs"].argmax(1)
    return dict(
        beliefs=best["beliefs"],
        communities=hard_final,
        balls=best["balls"],
        node2idx=best["node2idx"],
        idx2node={i: u for u, i in best["node2idx"].items()},
        history=hist,
        G_final=subG,
    )


# ──────────────────────────────────────────── 3. demo ──
def main():
    # Example: Amazon metadata graph (Hamming distance version)
    G = nx.read_gml("amazon_metadata_test/amazon_hamming_bookDVD.gml")
    H_obs = coords_str2arr(G)

    result = duo_spec_tester(
        H_obs,
        # ❶ core model size
        K           = 2,
        num_balls   = 32,
        config      = ("laplacian"),

        # ❷ give the loop plenty of runway
        max_em_iters = 150,      # hard cap
        warmup_rounds= 6,        # first 6 iters: no weight changes
        anneal_steps = 18,       # then λ ramps up over 18 more iters

        # ❸ make each weight tweak gentler
        shrink_comm = 0.45,      # ↓ from 1.00
        shrink_geo  = 0.40,
        boost_comm  = 0.25,      # ↓ from 0.60
        boost_geo   = 0.20,

        # ❹ touch fewer edges per round
        comm_cut        = 0.96,  # only top 4 % p_same shrink
        geo_cut         = 0.96,
        boost_cut_comm  = 0.995, # boost only the most confident 0.5 %
        boost_cut_geo   = 0.995,

        # ❺ noise-meter: stricter plateau test
        noise_sample = 30_000,   # smoother GN-MSE curve
        noise_window = 8,        # need 8 flat points
        noise_tol    = 2e-5,     # “flat” means Δ<2 × 10⁻⁵

        # ❻ stop only when *really* done
        tol        = 5e-6,       # objective tolerance
        patience   = 20,         # (if you re-enable patience later)

        random_state = 42,
    )

    # print noise trajectory
    for row in result["history"]:
        print(row["it"], row["gn_mse"], row["gn_dev"])

    mse = np.array([h["gn_mse"] for h in result["history"]])
    slope, *_ = linregress(range(len(mse)), mse)
    print("slope =", slope)


if __name__ == "__main__":
    main()

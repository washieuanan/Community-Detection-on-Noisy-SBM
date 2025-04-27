from __future__ import annotations
import numpy as np
import networkx as nx
from typing import Callable, Dict, List, Sequence, Tuple


def euclid_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b, ord=2))


def default_prob_fn(r: float, *, r_max: float, kind: str = "exp") -> float:
    """
    Monotone detection-probability profile.
        kind="linear": p(r)=max(0,1−r/r_max)
        kind="exp":    p(r)=exp(−λ r) with λ=3
    """
    if kind == "linear":
        return max(0.0, 1.0 - r / r_max)
    if kind == "exp":
        return float(np.exp(-3.0 * r))
    raise ValueError("kind must be 'linear' or 'exp'")

def pick_sensors(G, num_sensors, min_sep=0.15, *, seed=None):
    ''' 
    shuffle through vertex ids to randomize order 
    walk through the shuffled list and add a vertex to sensor set S if its euclidean distance from all vertices in S is leq min_sep 
    stop when we have num_sensors sensors 

    G: nx.Graph
    num_sensors: int
    min_sep: float
    seed: int | None

    Returns
    -------
    sensors: list[int]
    '''
    rng  = np.random.default_rng(seed)
    ids  = np.array(G.nodes())
    rng.shuffle(ids)

    coords = {v: np.asarray(G.nodes[v]["coords"], float) for v in ids}
    sensors = []

    for v in ids:
        if len(sensors) == num_sensors:
            break
        if all(np.linalg.norm(coords[v] - coords[w]) >= min_sep for w in sensors):
            sensors.append(v)

    if len(sensors) < num_sensors:
        raise RuntimeError(
            f"Could place only {len(sensors)}/{num_sensors} sensors "
            f"with min-sep={min_sep}.  Lower min_sep or nudge num_sensors."
        )
    return sensors

def sensor_observations(
    G: nx.Graph,
    sensor: int,
    radii: Sequence[float],
    *,
    prob_fn: Callable[[float], float] | None = None,
    seed: int | None = None,
    deduplicate_edges: bool = False,
) -> Tuple[Dict[float, List[List[int]]], Dict[int, float]]:
    """
    Parameters
    ----------
    G : nx.Graph            
    sensor : int            
    radii  : increasing iterable of floats in (0,1]
    prob_fn: r -> p(r)      
    seed   : int | None     
    deduplicate_edges : remove repeats within each radius list

    Returns
    -------
    obs_by_r    : {radius: [[u,v], …]}  edges observed at each radius
    first_seen  : {vertex: first_radius_reachable} when the vertex was first detected by the sensor 
    """
    radii = np.sort(np.asarray(radii, dtype=float))
    if radii[0] <= 0 or radii[-1] > 1:
        raise ValueError("radii must be in (0,1] and increasing")

    r_max = float(radii[-1])
    if prob_fn is None:
        prob_fn = lambda r, *, r_max=r_max: default_prob_fn(r, r_max=r_max)

    rng = np.random.default_rng(seed)

    sensor_coord = np.asarray(G.nodes[sensor]["coords"], dtype=float)
    dists = {v: euclid_dist(sensor_coord, np.asarray(G.nodes[v]["coords"]))
             for v in G.nodes()}

    obs_by_r: Dict[float, List[List[int]]] = {}
    first_seen: Dict[int, float] = {}
    detected_vertices: set[int] = set()       

    ball_vertices = {sensor}
    subG = nx.Graph()
    subG.add_node(sensor)

    for r in radii:
        new_in_ball = {v for v, d in dists.items() if d <= r} - ball_vertices
        ball_vertices |= new_in_ball

        subG.add_nodes_from(new_in_ball)
        for v in new_in_ball:
            for nbr in G.neighbors(v):
                if nbr in ball_vertices:
                    subG.add_edge(v, nbr)

        reachable = nx.single_source_shortest_path_length(subG, sensor).keys()

        for v in reachable:
            first_seen.setdefault(v, r)

        edge_samples: List[Tuple[int, int]] = []
        p_det = prob_fn(r)
        for v in reachable:
            if v == sensor:
                continue
            if v not in detected_vertices:
           
                if rng.random() < p_det:
                    detected_vertices.add(v)
                else:
                    continue          


            path = nx.shortest_path(subG, sensor, v)
            edge_samples.extend([(u, w) for u, w in zip(path[:-1], path[1:])])

        if deduplicate_edges:
            edge_samples = list({tuple(sorted(e)) for e in edge_samples})

        obs_by_r[r] = [list(e) for e in edge_samples]

    return obs_by_r, first_seen


def gather_multi_sensor_observations(G, sensors, radii, *, prob_fn=None, seed=None, deduplicate_edges=False):
    obs_global = {r: [] for r in radii}
    first_seen = {}

    master_rng = np.random.default_rng(seed)
    for s in sensors:
        # split seed stream so each sensor gets an ind RNG
        s_seed = master_rng.integers(2**63 - 1)
        obs_s, fs_s = sensor_observations(
            G, s, radii,
            prob_fn=prob_fn,
            seed=int(s_seed),
            deduplicate_edges=deduplicate_edges,
        )

        for r in radii:
            obs_global[r].extend(obs_s[r])

        # keep earliest first-seen radius per vertex
        for v, r0 in fs_s.items():
            first_seen[v] = min(first_seen.get(v, 1e9), r0)

    if deduplicate_edges:
        for r in radii:
            obs_global[r] = [list(e) for e in {tuple(sorted(e))
                                               for e in obs_global[r]}]

    return obs_global, first_seen



if __name__ == "__main__":
    from graph_generation.generate_graph import generate_latent_geometry_graph

    # G, coords, _ = generate_latent_geometry_graph([30, 40],
    #                                               connectivity_threshold=0.75)
    # r_grid = np.linspace(0.1, 1.0, 10)
    # obs, first_seen = sensor_observations(
    #     G, sensor=0, radii=r_grid, seed=123, deduplicate_edges=True
    # )

    # print(obs)
    # for r in r_grid:
    #     print(f"r={r:.2f}  edges={len(obs[r])}")


   
    G, coords, _ = generate_latent_geometry_graph([100,150], connectivity_threshold=0.75)

    sensors = pick_sensors(G, num_sensors=3, min_sep=0.18, seed=7)

    r_grid = np.linspace(0.1, 1.0, 12)
    obs, first_seen = gather_multi_sensor_observations(
            G, sensors, r_grid, seed=99, deduplicate_edges=True)

    print(obs)
    print("total len of obs: ", len(obs))
    print("anchors:", sensors)
    print("total edges in G:", len(G.edges))
    for r in r_grid:
        print(f"r={r:.2f}   edges_seen={len(obs[r])}")

import numpy as np
import networkx as nx
from typing import List, Tuple

# pylint: disable=import-error
from .observe import Observation



def get_coordinate_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    return np.linalg.norm(np.array(coord1) - np.array(coord2))


class PairSamplingObservation(Observation):
    """
    Samples vertex pairs uniformly or via a provided weight function,
    ensuring there exists a path between them.
    """

    def __init__(
        self,
        graph: nx.Graph,
        num_samples: int,
        weight_func=None,
        seed: int = None,
    ):
        super().__init__(graph, seed)
        self.num_samples = int(num_samples)
        self.weight_func = weight_func

        # --- 1) Precompute connected‐component mapping ---
        comps = list(nx.connected_components(graph))
        self._node2comp = {}
        for comp in comps:
            for u in comp:
                self._node2comp[u] = comp

        # --- 2) Build candidate pairs & normalized weight array ONCE ---
        pairs: List[Tuple[int,int]] = []
        weights: List[float] = []
        for comp in comps:
            comp_nodes = list(comp)
            for i in range(len(comp_nodes)):
                u = comp_nodes[i]
                for v in comp_nodes[i+1:]:
                    pairs.append((u, v))
                    if weight_func:
                        cu = graph.nodes[u].get("coords")
                        cv = graph.nodes[v].get("coords")
                        weights.append(weight_func(cu, cv))
                    else:
                        weights.append(1.0)

        w = np.array(weights, dtype=float)
        if w.sum() > 0:
            w /= w.sum()

        self._pairs = pairs
        self._weights = w
        self.count = 0

    def observe(self) -> List[Tuple[int, int]]:
        # Pure NumPy sampling, no networkx calls here
        idx = self.rng.choice(
            len(self._pairs),
            size=self.num_samples,
            replace=True,
            p=self._weights
        )
        obs = [self._pairs[i] for i in idx]
        self.observations = obs
        # count unique edges via numpy
        arr = np.array(obs, dtype=int)
        # sort each row so (u,v) and (v,u) aren't double‐counted
        arr = np.sort(arr, axis=1)
        self.count = len(np.unique(arr, axis=0))
        return obs

    def get_count(self) -> int:
        return self.count


class VertexBasedSamplingObservation(Observation):
    """
    For each vertex, samples a random number of other vertices based on degree,
    weighted by the provided weight function.
    """

    def __init__(
        self,
        graph: nx.Graph,
        weight_func=None,
        seed: int = None,
    ):
        super().__init__(graph, seed)
        self.weight_func = weight_func

        # Precompute node‐to‐component and component‐to‐node lists
        comps = list(nx.connected_components(graph))
        self._node2comp = {}
        self._comp2nodes = {}
        for comp in comps:
            nodes = list(comp)
            for u in nodes:
                self._node2comp[u] = frozenset(comp)
            self._comp2nodes[frozenset(comp)] = nodes

        # pull coords once
        self._coords = {u: graph.nodes[u].get("coords") for u in graph.nodes()}
        self.count = 0

    def observe(self) -> List[Tuple[int, int]]:
        observed = set()
        obs_list: List[Tuple[int,int]] = []

        for v in self.graph.nodes():
            comp_nodes = self._comp2nodes[self._node2comp[v]]
            # exclude self and already‐observed pairs
            candidates = [u for u in comp_nodes if u != v and (v,u) not in observed]
            if not candidates:
                continue

            deg_v = self.graph.degree[v]
            if deg_v < 1:
                continue

            # sample a random count k in {1, 2, …, deg_v}
            k = self.rng.integers(1, deg_v + 1)
            # vectorized weight build
            if self.weight_func:
                coords = [self._coords[u] for u in candidates]
                center = self._coords[v]
                w = np.array([self.weight_func(c, center) for c in coords], float)
                w /= w.sum() if w.sum()>0 else 1.0
            else:
                w = None  # uniform

            k = min(k, len(candidates))
            idx = self.rng.choice(len(candidates), size=k, replace=False, p=w)
            for i in idx:
                u = candidates[i]
                obs_list.append((v, u))
                observed.add((v,u))
                observed.add((u,v))

        self.observations = obs_list
        self.count = len(list(set(obs_list)))
        return obs_list

    def get_count(self) -> int:
        return self.count

if __name__ == "__main__":
    # from graph_generation.generate_graph import (
    #     generate_latent_geometry_graph,
    #     NUM_VERTICES_CLUSTER_1,
    #     NUM_VERTICES_CLUSTER_2,
    # )

    # # Generate a test graph
    # cluster_sizes = [NUM_VERTICES_CLUSTER_1, NUM_VERTICES_CLUSTER_2]
    # G, coordinates, vertex_cluster_map = generate_latent_geometry_graph(
    #     cluster_sizes, connectivity_threshold=0.8
    # )

    from experiments.graph_generation.gbm import generate_gbm

    G2 = generate_gbm(
        n=300,
        K=3,
        a = 100, 
        b = 50, 
        seed=123
    )

    # Define a weight function based on distance
    def weight_func(c1, c2):
        return np.exp(-0.5 * get_coordinate_distance(c1, c2))

    # Pair-based sampling
    pair_sampler = PairSamplingObservation(G2, num_samples=10, weight_func=weight_func, seed=42)
    observations_ = pair_sampler.observe()
    print(observations_)
    # print("Sample observations:")
    # print(observations_)

    # Vertex-based sampling
    vertex_sampler = VertexBasedSamplingObservation(G2, weight_func=weight_func, seed=42)
    vertex_observations = vertex_sampler.observe()

    # print("\nVertex-based observations:")
    # print(f"Total observations: {len(vertex_observations)}")
    # sample_show = (
    #     vertex_observations[:10]
    #     if len(vertex_observations) > 10
    #     else vertex_observations
    # )
    # print(f"Sample of observations: {sample_show}")

    # # Stats
    # from collections import Counter
   
    print("pair-sampling count: ", pair_sampler.get_count())
    print("vertex-sampling count: ", vertex_sampler.get_count())
    print("sparsity w/ vertex", vertex_sampler.get_count() / (len(G2.edges) * len(G2.nodes)))
    print(len(G2.edges))
    # print(vertex_observations)

    # vertex_counts = Counter(v for v, u in vertex_observations)
    # print(f"\nNumber of vertices with observations: {len(vertex_counts)}")
    # avg_obs = len(vertex_observations) / len(vertex_counts) if vertex_counts else 0
    # print(f"Average observations per vertex: {avg_obs:.2f}")

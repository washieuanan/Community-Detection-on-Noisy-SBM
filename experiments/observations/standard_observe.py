import numpy as np
import networkx as nx
from typing import List, Tuple

# pylint: disable=import-error
from .observe import Observation       



def get_coordinate_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """
    Compute the Euclidean distance between two coordinate vectors.
    """
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
        self.num_samples = num_samples
        self.weight_func = weight_func

    def observe(self) -> List[Tuple[int, int]]:
        """
        Perform pair sampling observation.

        Returns:
            List of tuples (u, v) sampled from all reachable pairs.
        """
        nodes = list(self.graph.nodes())
        candidate_pairs: List[Tuple[int, int]] = []
        weights: List[float] = []

        # Build all reachable pairs and associated weights
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                u, v = nodes[i], nodes[j]
                if nx.has_path(self.graph, u, v):
                    candidate_pairs.append((u, v))
                    if self.weight_func is not None:
                        coord_u = self.graph.nodes[u].get("coords")
                        coord_v = self.graph.nodes[v].get("coords")
                        weight = self.weight_func(coord_u, coord_v)
                    else:
                        weight = 1.0
                    weights.append(weight)

        weights_arr = np.array(weights, dtype=float)
        if weights_arr.sum() > 0:
            weights_arr /= weights_arr.sum()

        # Sample with replacement
        chosen = self.rng.choice(
            len(candidate_pairs), size=self.num_samples, replace=True, p=weights_arr
        )
        observations = [candidate_pairs[i] for i in chosen]
        self.observations = observations
        return observations


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

    def observe(self) -> List[Tuple[int, int]]:
        """
        Perform vertex-based sampling observation.

        Returns:
            List of tuples (v, u) sampling neighbors for each vertex v.
        """
        nodes = list(self.graph.nodes())
        vertex_coords = {node: self.graph.nodes[node].get("coords") for node in nodes}
        observed_pairs = set()
        observations: List[Tuple[int, int]] = []

        for v in nodes:
            deg_v = self.graph.degree(v)
            # Determine sample size between min and max of 2 and degree
            min_n = min(2, deg_v)
            max_n = max(2, deg_v)
            n = self.rng.integers(min_n, max_n + 1)

            candidates: List[int] = []
            weights: List[float] = []

            # Gather candidates that haven't been observed
            for u in nodes:
                if (
                    u != v
                    and (u, v) not in observed_pairs
                    and (v, u) not in observed_pairs
                    and nx.has_path(self.graph, u, v)
                ):
                    candidates.append(u)
                    if self.weight_func is not None:
                        weight = self.weight_func(vertex_coords[u], vertex_coords[v])
                    else:
                        weight = 1.0
                    weights.append(weight)

            if not candidates:
                continue

            weights_arr = np.array(weights, dtype=float)
            if weights_arr.sum() > 0:
                weights_arr /= weights_arr.sum()

            k = min(n, len(candidates))
            if k > 0:
                chosen = self.rng.choice(
                    len(candidates), size=k, replace=False, p=weights_arr
                )
                for idx in chosen:
                    u = candidates[idx]
                    observations.append((v, u))
                    observed_pairs.add((v, u))
                    observed_pairs.add((u, v))

        self.observations = observations
        return observations


if __name__ == "__main__":
    from graph_generation.generate_graph import (
        generate_latent_geometry_graph,
        NUM_VERTICES_CLUSTER_1,
        NUM_VERTICES_CLUSTER_2,
    )

    # Generate a test graph
    cluster_sizes = [NUM_VERTICES_CLUSTER_1, NUM_VERTICES_CLUSTER_2]
    G, coordinates, vertex_cluster_map = generate_latent_geometry_graph(
        cluster_sizes, connectivity_threshold=0.8
    )

    # Define a weight function based on distance
    def weight_func(c1, c2):
        return np.exp(-0.5 * get_coordinate_distance(c1, c2))

    # Pair-based sampling
    pair_sampler = PairSamplingObservation(G, num_samples=10, weight_func=weight_func, seed=42)
    observations_ = pair_sampler.observe()
    print("Sample observations:")
    print(observations_)

    # Vertex-based sampling
    vertex_sampler = VertexBasedSamplingObservation(G, weight_func=weight_func, seed=42)
    vertex_observations = vertex_sampler.observe()
    print("\nVertex-based observations:")
    print(f"Total observations: {len(vertex_observations)}")
    sample_show = (
        vertex_observations[:10]
        if len(vertex_observations) > 10
        else vertex_observations
    )
    print(f"Sample of observations: {sample_show}")

    # Stats
    from collections import Counter

    vertex_counts = Counter(v for v, u in vertex_observations)
    print(f"\nNumber of vertices with observations: {len(vertex_counts)}")
    avg_obs = len(vertex_observations) / len(vertex_counts) if vertex_counts else 0
    print(f"Average observations per vertex: {avg_obs:.2f}")

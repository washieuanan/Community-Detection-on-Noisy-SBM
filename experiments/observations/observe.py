import numpy as np
import networkx as nx
import argparse

def get_coordinate_distance(coord1, coord2):
    """
    Compute the Euclidean distance between two coordinate vectors.
    """
    return np.linalg.norm(np.array(coord1) - np.array(coord2))

def sample_observations(G, num_samples, weight_func=None, seed=None):
    """
    Sample observations from graph G. Each observation is a tuple (x, y)
    indicating that there exists a path between vertex x and vertex y.
    
    Parameters:
      G : nx.Graph
          Graph with a node attribute 'coords' (e.g. generated from your latent geometry code)
      num_samples : int
          Number of observations to sample.
      weight_func : callable, optional
          A function that accepts two node coordinate vectors (coord_u, coord_v)
          and returns a numerical weight. For instance, a function based on an exponential
          decay with distance. If None, uniform sampling is performed.
      seed : int, optional
          Random seed for reproducibility.
    
    Returns:
      observations : list of tuple(int, int)
          List of vertex pairs (observations) such that there exists a path between them.
      vertex_coords : dict
          Dictionary mapping vertex IDs to their coordinates.
    """
    if seed is not None:
        np.random.seed(seed)

    candidate_pairs = []
    weights = []

    # just go thru all pairs of nodes 
    nodes = list(G.nodes())
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            u, v = nodes[i], nodes[j]
            # Is there a path between u and v?
            if nx.has_path(G, u, v):
                candidate_pairs.append((u, v))
                if weight_func is not None:
                    coord_u = G.nodes[u].get('coords')
                    coord_v = G.nodes[v].get('coords')
                    # if coord_u is None or coord_v is None:
                    #     raise ValueError("Node coordinates missing. idk when this happens but we should fix if it does")
                    weight = weight_func(coord_u, coord_v)
                else:
                    weight = 1.0 
                weights.append(weight)

    candidate_pairs = np.array(candidate_pairs)
    weights = np.array(weights)
    
    
    weights = weights / np.sum(weights)


    chosen_indices = np.random.choice(len(candidate_pairs), size=num_samples, replace=True, p=weights)
    observations = candidate_pairs[chosen_indices].tolist()

    # Get all vertices and their coordinates
    vertex_coords = {node: G.nodes[node].get('coords') for node in G.nodes()}
    
    return observations, vertex_coords

def sample_vertex_observations(G, weight_func=None, seed=None):
    """
    For each vertex v, sample n vertices where n = Unif(min(2, deg(v)), max(2, deg(v))) 
    that have a path between that vertex and v. Sampling is subject to the provided weight function.
    
    Parameters:
      G : nx.Graph
          Graph with a node attribute 'coords'
      weight_func : callable, optional
          A function that accepts two node coordinate vectors (coord_u, coord_v)
          and returns a numerical weight. For instance, a function based on an exponential
          decay with distance. If None, uniform sampling is performed.
      seed : int, optional
          Random seed for reproducibility.
    
    Returns:
      observations : list of tuple(int, int)
          List of vertex pairs (observations) where there exists a path between them.
      vertex_coords : dict
          Dictionary mapping vertex IDs to their coordinates.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize the set of observed pairs to avoid duplicates
    observed_pairs = set()
    observations = []
    
    # Get all vertices and their coordinates
    nodes = list(G.nodes())
    vertex_coords = {node: G.nodes[node].get('coords') for node in nodes}
    
    # For each vertex in the graph
    for v in nodes:
        # Determine degree of v
        deg_v = G.degree(v)
        
        # Determine how many neighbors to sample
        min_n = min(2, deg_v)
        max_n = max(2, deg_v)
        n = np.random.randint(min_n, max_n + 1)  # +1 because randint upper bound is exclusive
        
        # Find all possible candidates (vertices with a path to v that haven't been observed in reverse)
        candidates = []
        weights = []
        
        for u in nodes:
            if u != v and (u, v) not in observed_pairs and (v, u) not in observed_pairs and nx.has_path(G, u, v):
                candidates.append(u)
                if weight_func is not None:
                    coord_u = vertex_coords[u]
                    coord_v = vertex_coords[v]
                    weight = weight_func(coord_u, coord_v)
                else:
                    weight = 1.0
                weights.append(weight)
        
        if not candidates:
            continue  # Skip if no candidates
        
        # Normalize weights
        weights = np.array(weights)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        
        # Sample n vertices or fewer if we don't have enough candidates
        k = min(n, len(candidates))
        if k > 0:
            chosen_indices = np.random.choice(len(candidates), size=k, replace=False, p=weights)
            chosen = [candidates[i] for i in chosen_indices]
            
            # Add to observations and mark as observed
            for u in chosen:
                observations.append((v, u))
                observed_pairs.add((v, u))
                observed_pairs.add((u, v))  # Ensure we don't sample (u,v) later
    
    return observations, vertex_coords

if __name__ == "__main__":
    from graph_generation.generate_graph import generate_latent_geometry_graph, NUM_VERTICES_CLUSTER_1, NUM_VERTICES_CLUSTER_2
    
    cluster_sizes = [NUM_VERTICES_CLUSTER_1, NUM_VERTICES_CLUSTER_2]
    G, coordinates, vertex_cluster_map = generate_latent_geometry_graph(cluster_sizes, connectivity_threshold=0.8)
    
    def weight_func(coord1, coord2):
        distance = get_coordinate_distance(coord1, coord2)
        return np.exp(-0.5 * distance)
    
    # Original sampling method
    observations, vertex_coords = sample_observations(G, 10, weight_func=weight_func)
    print("Sample observations:")
    print(observations)
    
    # New vertex-based sampling method
    vertex_observations, vertex_coords = sample_vertex_observations(G, weight_func=weight_func)
    print("\nVertex-based observations:")
    print(f"Total observations: {len(vertex_observations)}")
    print(f"Sample of observations: {vertex_observations[:10] if len(vertex_observations) > 10 else vertex_observations}")
    
    # Print stats about how many observations per vertex
    vertex_counts = {}
    for v, u in vertex_observations:
        vertex_counts[v] = vertex_counts.get(v, 0) + 1
    
    print(f"\nNumber of vertices with observations: {len(vertex_counts)}")
    print(f"Average observations per vertex: {len(vertex_observations) / len(vertex_counts) if vertex_counts else 0:.2f}")

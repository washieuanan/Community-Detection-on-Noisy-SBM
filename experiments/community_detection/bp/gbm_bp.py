# imports 

from experiments.community_detection.bp.belief_prop import (
    detection_stats,
    initialize_beliefs,
    get_marginals_and_preds,
    belief_propagation,
    get_sbm,
    get_true_communities
)

from experiments.graph_generation.generate_graph import (
    generate_latent_geometry_graph,
    NUM_VERTICES_CLUSTER_1,
    NUM_VERTICES_CLUSTER_2,
)

# from experiments.observations.observe import get_coordinate_distance
import numpy as np
import networkx as nx

from experiments.observations.random_walk_obs import random_walk_observations
from experiments.observations.sensor_observe import sensor_observations, gather_multi_sensor_observations, pick_sensors

def get_unique_edges(obs):
    """
    Extract all unique edges from the observation dictionary
    
    Parameters
    ----------
    obs : Dict[float, List[List[int]]]
        Dictionary where keys are radii and values are lists of edges
        
    Returns
    -------
    List[List[int]]
        List of unique edges (each edge is a list of length 2)
    """
    # Use a set of tuples to track unique edges
    unique_edges_set = set()
    
    # Go through all radii and their edges
    for radius, edges in obs.items():
        for edge in edges:
            # Sort the edge to ensure [a,b] and [b,a] are considered the same
            unique_edges_set.add(tuple(sorted(edge)))
    
    # Convert back to list of lists
    unique_edges = [list(edge) for edge in unique_edges_set]
    
    return unique_edges

def create_observed_subgraph(num_coords, observations):
    """
    create subgraph containing all nodes and observed paths as edges
    """
    subG = nx.Graph()
    
    for c in range(num_coords):
        subG.add_node(c)
     
    for u, v in observations:
        subG.add_edge(u, v)
    return subG


if __name__ == "__main__":
    from experiments.graph_generation.gbm import generate_gbm
    # Generate a graph with two clusters
    # distributions = ['normal', 'normal', 'normal']
    # dist_params = [
    #     {'loc':[-0.3, 0.1, 0.3], 'scale':0.2, 'constrain_to_unit_sphere':True},
    #     {'loc':[0.1, 0.4, x-0.2], 'scale':0.1, 'constrain_to_unit_sphere':True},
    #     {'loc':[0.3, -0.1, 0.1], 'scale':0.2, 'constrain_to_unit_sphere':True},
    # ]
    
    # def weight_func(coord1, coord2):
    #     distance = get_coordinate_distance(coord1, coord2)
    #     return np.exp(-0.4 * distance)
    
    # def prob_weight_func(dist):
    #     return max(np.exp(-0.6 * dist), 0.1)
    
    # G2, coords2, cluster_map2 = generate_latent_geometry_graph(
    #     [40, 40, 40], 
    #     distributions=distributions,
    #     dist_params=dist_params,
    #     edge_prob_fn= prob_weight_func # Higher threshold to create more edges
    # )  
    # observations, _ = sample_observations(G2, 25, weight_func=weight_func)
    # observations = random_walk_observations(G2, num_walkers=8, stopping_param=0.2, leaky=0.1)
    # r_grid = np.linspace(0.1, 1.0, 10)

    # obs, first_seen = sensor_observations(
    #     G2, sensor=0, radii=r_grid, seed=123, deduplicate_edges=True
    # )
    G2 = generate_gbm(
        n=300,
        K=3,
        a = 100, 
        b = 50,
        seed=123
    )
    sensors = pick_sensors(G2, num_sensors=5, min_sep=0.10, seed=99)
    r_grid = np.linspace(0.1, 1.0, 12)
    obs, first_seen = gather_multi_sensor_observations(
            G2, sensors, r_grid, seed=99, deduplicate_edges=True)
    
    observations = []
    for k, v in obs.items():
        for e in v:
            observations.append(e)
    
    # New approach - get all unique edges across all radii
    unique_edges = get_unique_edges(obs)
    
    # Print counts to compare
    print(f"Total observations (may include duplicates): {len(observations)}")
    print(f"Unique edges: {len(unique_edges)}")
    
    observed_nodes = set()
    for u, v in observations:
        observed_nodes.add(u)
        observed_nodes.add(v)
    # print(observations)
    
    # subG = create_observed_subgraph(100, observations)``
    subG = create_observed_subgraph(len(G2.nodes), unique_edges)
    # subG =iG2 
    initialize_beliefs(subG, 3)
    belief_propagation(
        subG, 
        q=3, 
        max_iter=1000,
        damping=0.2,
        balance_regularization=0.1,
        min_steps=50,
    )
    marginals, preds = get_marginals_and_preds(subG)
    # cluster_map = np.array(cluster_map2)
    cluster_map = nx.get_node_attributes(G2, 'comm')
    cluster_map = np.array(list(cluster_map.values()))
    
    sub_preds = np.array([preds[i] for i in range(len(preds)) if i in observed_nodes])
    sub_cluster_map = np.array([cluster_map[i] for i in range(len(cluster_map)) if i in observed_nodes])
    print(detection_stats(preds, cluster_map))
    print(detection_stats(sub_preds, sub_cluster_map))
    
    
    
    
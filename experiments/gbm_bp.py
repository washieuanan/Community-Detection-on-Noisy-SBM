# imports 
from belief_prop import (
    detection_stats,
    initialize_beliefs,
    get_marginals_and_preds,
    belief_propagation,
    get_sbm,
    get_true_communities
)

from generate_graph import (
    generate_latent_geometry_graph,
    NUM_VERTICES_CLUSTER_1,
    NUM_VERTICES_CLUSTER_2,
)

from observe import sample_observations, get_coordinate_distance
import numpy as np
import networkx as nx

from random_walk_obs import random_walk_observations
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
    # Generate a graph with two clusters
    distributions = ['normal', 'normal', 'normal']
    dist_params = [
        {'loc':[-0.3, 0.1, 0.3], 'scale':0.2, 'constrain_to_unit_sphere':True},
        {'loc':[0.1, 0.4, -0.2], 'scale':0.1, 'constrain_to_unit_sphere':True},
        {'loc':[0.3, -0.1, 0.1], 'scale':0.2, 'constrain_to_unit_sphere':True},
    ]
    
    def weight_func(coord1, coord2):
        distance = get_coordinate_distance(coord1, coord2)
        return np.exp(-0.4 * distance)
    
    def prob_weight_func(dist):
        return max(np.exp(-0.6 * dist), 0.1)
    
    G2, coords2, cluster_map2 = generate_latent_geometry_graph(
        [40, 20, 40], 
        distributions=distributions,
        dist_params=dist_params,
        edge_prob_fn= prob_weight_func # Higher threshold to create more edges
    )  
    # observations, _ = sample_observations(G2, 25, weight_func=weight_func)
    observations = random_walk_observations(G2, num_walkers=8, stopping_param=0.2, leaky=0.1)
    observed_nodes = set()
    for u, v in observations:
        observed_nodes.add(u)
        observed_nodes.add(v)
    # print(observations)
    
    subG = create_observed_subgraph(100, observations)
    # subG = G2 
    initialize_beliefs(subG, 3)
    belief_propagation(
        subG, 
        3, 
        max_iter=1000,
        damping=0.4,
        anneal_steps=100,    
        balance_regularization=0.02,
        min_steps=50,
    )
    marginals, preds = get_marginals_and_preds(subG)
    cluster_map = np.array(cluster_map2)
    sub_preds = np.array([preds[i] for i in range(len(preds)) if i in observed_nodes])
    print(sub_preds)
    sub_cluster_map = np.array([cluster_map[i] for i in range(len(cluster_map)) if i in observed_nodes])
    print(detection_stats(preds, cluster_map))
    print(detection_stats(sub_preds, sub_cluster_map))
    
    
    
    
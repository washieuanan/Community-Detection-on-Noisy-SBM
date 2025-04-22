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
    # distributions = ['normal', 'normal']
    # dist_params = [
    #     {'loc':-0.5, 'scale':0.3, 'constrain_to_unit_sphere':False},
    #     {'loc':0.5, 'scale':0.4, 'constrain_to_unit_sphere':False}
    # ]
    
    # G2, coords2, cluster_map2 = generate_latent_geometry_graph(
    #     [100, 150], 
    #     distributions=distributions,
    #     dist_params=dist_params,
    #     connectivity_threshold=0.8  # Higher threshold to create more edges
    # )
    num_nodes = 300
    num_communities = 3
    interior_prob = 0.7  # Increased internal connectivity
    exterior_prob = 0.2  # Decreased external connectivity
    G = get_sbm(num_nodes, num_communities, interior_prob, exterior_prob)
    for node in G.nodes():
        # assign random coordinates to each node depending on the community in 2D
        community = G.nodes[node]['block']
        if community == 0:
            center = np.array([-0.4, 0.3])
        elif community == 1:
            center = np.array([-0.3, -0.4])
        elif community == 2:
            center = np.array([0.1, 0.2])
        else:
            center = np.array([0, 0.5])
        G.nodes[node]['coords'] = np.random.normal(loc=center, scale=0.1, size=2)
        
    def weight_func(coord1, coord2):
        distance = get_coordinate_distance(coord1, coord2)
        return np.exp(-0.3 * distance)
    
    observations = sample_observations(G, 20, weight_func=weight_func)    
    observed_nodes = set()
    for u, v in observations:
        observed_nodes.add(u)
        observed_nodes.add(v)
    subG = create_observed_subgraph(num_nodes, observations)
    subG = G
    initialize_beliefs(subG, 2)
    
    belief_propagation(
        subG, 
        num_communities, 
        max_iter=1000,
        damping=0.2,
        anneal_steps=300,    
        balance_regularization=0.2,
    )    
    marginals, preds = get_marginals_and_preds(subG)
    cluster_map = get_true_communities(G)
    sub_preds = np.array([preds[i] for i in range(len(preds)) if i in observed_nodes])
    sub_cluster_map = np.array([cluster_map[i] for i in range(len(cluster_map)) if i in observed_nodes])
    print(detection_stats(preds, cluster_map))
    print(detection_stats(sub_preds, sub_cluster_map))
    
    
    
    
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def get_sbm(num_nodes: int, 
            num_communities: int, 
            interior_prob: float, 
            exterior_prob: float, 
            seed=0):
    """generates SBM model"""
    
    np.random.seed(seed)
    # generate community sizes
    community_sizes = np.random.multinomial(num_nodes, [1/num_communities] * num_communities)
    community_sizes = [max(1, size) for size in community_sizes]
    
    # generate connection probability matrix
    p_matrix = [[0] * num_communities for _ in range(num_communities)]
    for i in range(num_communities):
        for j in range(num_communities):
            if i == j:
                p_matrix[i][j] = interior_prob
            else:
                p_matrix[i][j] = exterior_prob
                
    return nx.stochastic_block_model(community_sizes, p_matrix, seed=seed)

def initialize_beliefs(G: nx.Graph, num_communities: int, seed=0):
    """initialize beliefs"""
    
    np.random.seed(seed)
    for node in G.nodes():
        belief_dist = [1/num_communities] * num_communities
        G.nodes[node]['beliefs'] = np.array(belief_dist)

def get_true_communities(G: nx.Graph):
    """get true communities"""
    
    block_labels = nx.get_node_attributes(G, 'block')
    return block_labels

def update_beliefs(u, v):
    """update beliefs from one node to another"""
    pass

if __name__ == "__main__":
    # parameters
    num_nodes = 100
    num_communities = 3
    interior_prob = 0.5
    exterior_prob = 0.1
    
    # generate graph
    G = get_sbm(num_nodes, num_communities, interior_prob, exterior_prob)
    # print(get_true_communities(G))
    # initialize beliefs
    initialize_beliefs(G, num_communities)
    print(G.nodes[0])

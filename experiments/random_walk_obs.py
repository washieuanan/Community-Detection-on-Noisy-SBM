import networkx as nx
import numpy as np

def random_walk_observations(G, num_walkers, num_steps=5, stopping_param = None, leaky=0.0, seed=0):
    """
    Gets observations on graph G using num_walkers doing Metropolis-Hastings 
    random walks. Each walker spawns at an unobserved node and walks for num_steps.
    
    Leaky (float): probability that a random walker skips observing an edge
    """

    observed_nodes = set()
    observations = []

    np.random.seed(seed)

    all_nodes = set(G.nodes())

    for walker in range(num_walkers):
        possible_start_nodes = list(all_nodes - observed_nodes)
        
        if not possible_start_nodes:
            current_node = int(np.random.choice(list(all_nodes)))
        else:
            current_node = int(np.random.choice(possible_start_nodes))
            
        observed_nodes.add(current_node)

        if stopping_param is not None:
            num_steps = np.random.geometric(stopping_param)
            
        for step in range(num_steps):
            neighbors = list(G.neighbors(current_node))
            if not neighbors:
                break  

            proposed_node = int(np.random.choice(neighbors))

            # Metropolis-Hastings acceptance probability
            degree_current = G.degree[current_node]
            degree_proposed = G.degree[proposed_node]

            acceptance_prob = min(1, degree_current / degree_proposed)

            if np.random.rand() <= acceptance_prob:
                if np.random.rand() > leaky:
                    # Record edge observation if not skipping due to leaky
                    observations.append([current_node, proposed_node])
                current_node = proposed_node  # Move accepted
                observed_nodes.add(current_node)

    return observations

if __name__ == "__main__":
    # Example usage
    G = nx.erdos_renyi_graph(100, 0.05)
    observations = random_walk_observations(G, num_walkers=10, num_steps=5, stopping_param = 0.1, leaky=0.1)
    print(observations)
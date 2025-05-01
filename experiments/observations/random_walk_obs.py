from typing import List, Tuple, Optional, Any
import networkx as nx
import numpy as np

from .observe import Observation
from graph_generation.generate_graph import distance_function


from typing import List, Tuple, Optional, Any
import networkx as nx
import numpy as np

from .observe import Observation
from graph_generation.generate_graph import distance_function

def random_walk_observations(G, num_walkers, num_steps=5, stopping_param = None, leaky=0.0, seed=0, return_groups=False):
    """
    Gets observations on graph G using num_walkers doing Metropolis-Hastings 
    random walks. Each walker spawns at an unobserved node and walks for num_steps.
    
    Leaky (float): probability that a random walker skips observing an edge
    """

    observed_nodes = set()
    observations = []
    grouped_observations = [0]*num_walkers
    np.random.seed(seed)

    all_nodes = set(G.nodes())

    for walker in range(num_walkers):
        walker_observations = []
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
            
            neighbor_distances = np.array([
                distance_function(np.array(G.nodes[current_node]["coords"]), np.array(G.nodes[neighbor]["coords"]))
                for neighbor in neighbors
            ])
            neighbor_exponential = 0.5 * np.exp(-0.5 * neighbor_distances)
            neighbor_exponential /= np.sum(neighbor_exponential)
            proposed_node = int(np.random.choice(neighbors, p=neighbor_exponential))
            # proposed_node = int(np.random.choice(neighbors))    
            # Metropolis-Hastings acceptance probability
            degree_current = G.degree[current_node]
            degree_proposed = G.degree[proposed_node]

            acceptance_prob = min(1, degree_current / degree_proposed)

            if np.random.rand() <= acceptance_prob:
                if np.random.rand() > leaky:
                    # Record edge observation if not skipping due to leaky
                    walker_observations.append([current_node, proposed_node])
                current_node = proposed_node  # Move accepted
                observed_nodes.add(current_node)
        observations.extend(walker_observations)
        grouped_observations[walker] = walker_observations

    if return_groups:
        return grouped_observations
    return observations

class RandomWalkObservation(Observation):
    """
    Perform observations by spawning a number of Metropolis–Hastings random walkers.
    """

    def __init__(
        self,
        graph: nx.Graph,
        seed: int,
        num_walkers: int,
        num_steps: int = 5,
        stopping_param: Optional[float] = None,
        leaky: float = 0.0,
    ):
        """
        Parameters
        ----------
        graph : nx.Graph
        seed : int
        num_walkers : int
            How many independent walkers to launch.
        num_steps : int
            Fixed number of steps per walker (unless stopping_param is set).
        stopping_param : float or None
            If not None, we sample each walk’s length ~ Geometric(stopping_param).
        leaky : float in [0,1]
            Probability a walker fails to record any given edge.
        """
        super().__init__(graph, seed)
        self.num_walkers = num_walkers
        self.num_steps = num_steps
        self.stopping_param = stopping_param
        self.leaky = leaky
        self.count = 0

    def observe(self) -> List[Tuple[Any, Any]]:
        # draw an independent seed for the global numpy RNG inside random_walk_observations
        rw_seed = int(self.rng.integers(2**32 - 1))
        raw_obs = random_walk_observations(
            G=self.graph,
            num_walkers=self.num_walkers,
            num_steps=self.num_steps,
            stopping_param=self.stopping_param,
            leaky=self.leaky,
            seed=rw_seed,
        )
        # convert inner lists to tuples
        self.observations = [tuple(edge) for edge in raw_obs]
        self.count = len(self.observations)
        return self.observations
    def get_count(self): 
        return self.count


class GroupedRandomWalkObservation(RandomWalkObservation):
    """
    Grouped version of RandomWalkObservation.
    """

    def observe(self) -> List[List[Tuple[Any, Any]]]:
        # draw an independent seed for the global numpy RNG inside random_walk_observations
        rw_seed = int(self.rng.integers(2**32 - 1))
        raw_obs = random_walk_observations(
            G=self.graph,
            num_walkers=self.num_walkers,
            num_steps=self.num_steps,
            stopping_param=self.stopping_param,
            leaky=self.leaky,
            seed=rw_seed,
            return_groups=True,
        )
        self.observations = [0]*self.num_walkers
        for ix, group in enumerate(raw_obs):
            self.observations[ix] = [tuple(edge) for edge in group]
        self.count = sum(len(group) for group in self.observations)
        return self.observations
    
if __name__ == "__main__":
    # Example usage
    # G = nx.erdos_renyi_graph(100, 0.05)
    # observations_ = random_walk_observations(G, num_walkers=10, num_steps=5, stopping_param = 0.1, leaky=0.1)
    # print(observations_)

    from experiments.graph_generation.generate_graph import generate_latent_geometry_graph
    G, coords, _ = generate_latent_geometry_graph([50,50], connectivity_threshold=0.8)

    # instantiate
    rw = RandomWalkObservation(
        graph=G,
        seed=123,
        num_walkers=10,
        num_steps=5,
        stopping_param=0.1,
        leaky=0.1,
    )

    # get your random-walk edges
    edges = rw.observe()
    print(f"Observed {len(edges)} edges via random walks")
    print("Observed the following edges: ", edges)
    print("Random Walk Edges: ", rw.get_count(), edges)


# def random_walk_observations(G, num_walkers, num_steps=5, stopping_param = None, leaky=0.0, seed=0, return_groups=False):
#     """
#     Gets observations on graph G using num_walkers doing Metropolis-Hastings 
#     random walks. Each walker spawns at an unobserved node and walks for num_steps.
    
#     Leaky (float): probability that a random walker skips observing an edge
#     """

#     observed_nodes = set()
#     observations = []
#     grouped_observations = [0]*num_walkers
#     np.random.seed(seed)

#     all_nodes = set(G.nodes())

#     for walker in range(num_walkers):
#         walker_observations = []
#         possible_start_nodes = list(all_nodes - observed_nodes)
#         if not possible_start_nodes:
#             current_node = int(np.random.choice(list(all_nodes)))
#         else:
#             current_node = int(np.random.choice(possible_start_nodes))
            
#         observed_nodes.add(current_node)

#         if stopping_param is not None:
#             num_steps = np.random.geometric(stopping_param)
            
#         for step in range(num_steps):
#             neighbors = list(G.neighbors(current_node))
#             if not neighbors:
#                 break  
            
#             neighbor_distances = np.array([
#                 distance_function(G.nodes[current_node]["coords"], G.nodes[neighbor]["coords"])
#                 for neighbor in neighbors
#             ])
#             neighbor_exponential = 0.5 * np.exp(-0.5 * neighbor_distances)
#             neighbor_exponential /= np.sum(neighbor_exponential)
#             proposed_node = int(np.random.choice(neighbors, p=neighbor_exponential))

#             # Metropolis-Hastings acceptance probability
#             degree_current = G.degree[current_node]
#             degree_proposed = G.degree[proposed_node]

#             acceptance_prob = min(1, degree_current / degree_proposed)

#             if np.random.rand() <= acceptance_prob:
#                 if np.random.rand() > leaky:
#                     # Record edge observation if not skipping due to leaky
#                     walker_observations.append([current_node, proposed_node])
#                 current_node = proposed_node  # Move accepted
#                 observed_nodes.add(current_node)
#         observations.extend(walker_observations)
#         grouped_observations[walker] = walker_observations

#     if return_groups:
#         return grouped_observations
#     return observations

# class RandomWalkObservation(Observation):
#     """
#     Perform observations by spawning a number of Metropolis–Hastings random walkers.
#     """

#     def __init__(
#         self,
#         graph: nx.Graph,
#         seed: int,
#         num_walkers: int,
#         num_steps: int = 5,
#         stopping_param: Optional[float] = None,
#         leaky: float = 0.0,
#     ):
#         """
#         Parameters
#         ----------
#         graph : nx.Graph
#         seed : int
#         num_walkers : int
#             How many independent walkers to launch.
#         num_steps : int
#             Fixed number of steps per walker (unless stopping_param is set).
#         stopping_param : float or None
#             If not None, we sample each walk’s length ~ Geometric(stopping_param).
#         leaky : float in [0,1]
#             Probability a walker fails to record any given edge.
#         """
#         super().__init__(graph, seed)
#         self.num_walkers = num_walkers
#         self.num_steps = num_steps
#         self.stopping_param = stopping_param
#         self.leaky = leaky
#         self.count = 0

#     def observe(self) -> List[Tuple[Any, Any]]:
#         # draw an independent seed for the global numpy RNG inside random_walk_observations
#         rw_seed = int(self.rng.integers(2**32 - 1))
#         raw_obs = random_walk_observations(
#             G=self.graph,
#             num_walkers=self.num_walkers,
#             num_steps=self.num_steps,
#             stopping_param=self.stopping_param,
#             leaky=self.leaky,
#             seed=rw_seed,
#         )
#         # convert inner lists to tuples
#         self.observations = [tuple(edge) for edge in raw_obs]
#         self.count = len(self.observations)
#         return self.observations
#     def get_count(self) -> int:
#         return self.count


# class GroupedRandomWalkObservation(RandomWalkObservation):
#     """
#     Grouped version of RandomWalkObservation.
#     """

#     def observe(self) -> List[List[Tuple[Any, Any]]]:
#         # draw an independent seed for the global numpy RNG inside random_walk_observations
#         rw_seed = int(self.rng.integers(2**32 - 1))
#         raw_obs = random_walk_observations(
#             G=self.graph,
#             num_walkers=self.num_walkers,
#             num_steps=self.num_steps,
#             stopping_param=self.stopping_param,
#             leaky=self.leaky,
#             seed=rw_seed,
#             return_groups=True,
#         )
#         self.observations = [0]*self.num_walkers
#         for ix, group in enumerate(raw_obs):
#             self.observations[ix] = [tuple(edge) for edge in group]
#         # Each group is a list of tuples representing edges so we need to count the total number of edges
#         # observed across all walkers
#         self.count = sum(len(group) for group in self.observations)
#         return self.observations
    
    
# if __name__ == "__main__":
#     # Example usage
#     # G = nx.erdos_renyi_graph(100, 0.05)
#     # observations_ = random_walk_observations(G, num_walkers=10, num_steps=5, stopping_param = 0.1, leaky=0.1)
#     # print(observations_)

#     from experiments.graph_generation.generate_graph import generate_latent_geometry_graph
#     G, coords, _ = generate_latent_geometry_graph([50,50], connectivity_threshold=0.8)

#     # instantiate
#     rw = RandomWalkObservation(
#         graph=G,
#         seed=123,
#         num_walkers=10,
#         num_steps=5,
#         stopping_param=0.1,
#         leaky=0.1,
#     )

#     grw = GroupedRandomWalkObservation(
#         graph=G,
#         seed=123,
#         num_walkers=10,
#         num_steps=5,
#         stopping_param=0.1,
#         leaky=0.1,
#     )

#     # get your random-walk edges
#     edges = rw.observe()
#     grw_edges = grw.observe()
#     # print the edges
#     print("Random Walk Edges: ", rw.get_count(), edges)
#     print("Grouped Random Walk Edges: ", grw.get_count(), grw_edges)
#     print(f"Observed {len(edges)} edges via random walks")
#     print("Observed the following edges: ", edges)
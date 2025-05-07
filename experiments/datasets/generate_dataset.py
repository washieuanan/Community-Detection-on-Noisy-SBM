import json
import os

import numpy as np 
import networkx as nx
from networkx.readwrite import json_graph


from graph_generation.gbm import generate_gbm
from observations.standard_observe import get_coordinate_distance
from observations.standard_observe import PairSamplingObservation



def verify_a_b(a: int, b: int) -> bool: 
    if np.sqrt(a) - np.sqrt(b) > 2*np.sqrt(2):
        return True
    else:
        return False

def verify_n(n: int, K: int) -> bool:
    if n >= 50*K:
        return True
    else:
        return False
    

def calculate_epsilon_threshold(a: int, b: int, q: int, n: int) -> float:
    p_in = a * np.log(n) / n 
    p_out = b * np.log(n) / n
    c = (p_in + (q - 1)*p_out) / q

    threshold = (np.sqrt(c) - 1) / (np.sqrt(c) - 1 + q)
    return threshold

def verify_epsilon_threshold(a: int, b: int, q: int, n: int) -> bool: 
    p_in = a * np.log(n) / n 
    p_out = b * np.log(n) / n 
    epsilon = p_out / p_in
    epsilon_threshold = calculate_epsilon_threshold(a, b, q, n)
    return epsilon < epsilon_threshold
    

def sample_parameter(n_range, K_list,  a_range, b_range): 
    n = int(np.random.choice(n_range))
    K = int(np.random.choice(K_list))
    a = int(np.random.choice(a_range))
    # ensure b < a
    b_candidates = [bb for bb in b_range if bb < a]
    b = int(np.random.choice(b_candidates)) if b_candidates else int(a // 2)
    return n, K, a, b


def generate_dataset(output_dir: str, 
                     num_graphs: int = 500, 
                    n_range=None, 
                    K_list=None, 
                    a_range=None, 
                    b_range=None): 
    n_range = n_range or range(100, 1000, 50) 
    K_list = K_list or [2] 
    a_range = a_range or range(50, 501, 50) 
    b_range = b_range or range(10, 301, 10) 

    os.makedirs(output_dir, exist_ok=True) 

    count = 0 
    seed = 0 
    sparsities = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1]

    while count < num_graphs: 
        n, K, a, b = sample_parameter(n_range, K_list, a_range, b_range) 
        if verify_a_b(a,b) and verify_n(n, K): 
            G = generate_gbm(n = n, K = K, a = a, b = b, seed = seed)
            epsilon_threshold = calculate_epsilon_threshold(a, b, K, n)
            data = json_graph.node_link_data(G) 

            def weight_func(c1, c2): 
                return 1.0
            # Now, we will generate observations for each sparsity level

            observations_data = {}
            for sparsity in sparsities: 
                num_pairs = int(sparsity * n ** 2 / 2)
                obs = PairSamplingObservation(G, num_samples=num_pairs, weight_func=weight_func, seed=seed)
                observations = obs.observe()
                observations_data[sparsity] = observations
                

            record = {
                'parameters': {'n': n, 'K': K, 'a': a, 'b': b, 'epsilon_threshold': epsilon_threshold, 'seed': seed}, 
                'graph': data, 
                'observations': observations_data
            }

            filename = os.path.join(output_dir, f'graph_{count+1:03d}.json')    
            with open(filename, 'w') as f:
                json.dump(record, f) 
            count += 1 
        seed += 1


if __name__ == "__main__": 
    generate_dataset("datasets/NEW_gbm_w_observations", num_graphs=500)


















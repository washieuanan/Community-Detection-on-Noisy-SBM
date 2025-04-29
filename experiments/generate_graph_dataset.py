import json
import os

import numpy as np 
import networkx as nx
from networkx.readwrite import json_graph


from graph_generation.gbm import generate_gbm
from observations.standard_observe import get_coordinate_distance



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
    n_range = n_range or range(100, 5000, 100) 
    K_list = K_list or [2,3,4] 
    a_range = a_range or range(50, 501, 50) 
    b_range = b_range or range(10, 301, 10) 

    os.makedirs(output_dir, exist_ok=True) 

    count = 0 
    seed = 0 

    while count < num_graphs: 
        n, K, a, b = sample_parameter(n_range, K_list, a_range, b_range) 
        if verify_a_b(a,b) and verify_n(n, K): 
            G = generate_gbm(n = n, K = K, a = a, b = b, seed = seed)
            data = json_graph.node_link_data(G) 

            record = {
                'parameters': {'n': n, 'K': K, 'a': a, 'b': 'b', 'seed': seed}, 
                'graph': data
            }

            filename = os.path.join(output_dir, f'graph_{count+1:03d}.json') 
            with open(filename, 'w') as f:
                json.dump(record, f) 
            count += 1 
        seed += 1


if __name__ == "__main__": 
    generate_dataset("gbm_graphs_data", num_graphs=500)















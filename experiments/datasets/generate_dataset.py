# import json
# import os

# import numpy as np 
# import networkx as nx
# from networkx.readwrite import json_graph


# from graph_generation.gbm import generate_gbm
# from observations.standard_observe import get_coordinate_distance



# def verify_a_b(a: int, b: int) -> bool: 
#     if np.sqrt(a) - np.sqrt(b) > 2*np.sqrt(2):
#         return True
#     else:
#         return False

# def verify_n(n: int, K: int) -> bool:
#     if n >= 50*K:
#         return True
#     else:
#         return False
    

# def calculate_epsilon_threshold(a: int, b: int, q: int, n: int) -> float:
#     p_in = a * np.log(n) / n 
#     p_out = b * np.log(n) / n
#     c = (p_in + (q - 1)*p_out) / q

#     threshold = (np.sqrt(c) - 1) / (np.sqrt(c) - 1 + q)
#     return threshold

# def verify_epsilon_threshold(a: int, b: int, q: int, n: int) -> bool: 
#     p_in = a * np.log(n) / n 
#     p_out = b * np.log(n) / n 
#     epsilon = p_out / p_in
#     epsilon_threshold = calculate_epsilon_threshold(a, b, q, n)
#     return epsilon < epsilon_threshold
    

# def sample_parameter(n_range, K_list,  a_range, b_range): 
#     n = int(np.random.choice(n_range))
#     K = int(np.random.choice(K_list))
#     a = int(np.random.choice(a_range))
#     # ensure b < a
#     b_candidates = [bb for bb in b_range if bb < a]
#     b = int(np.random.choice(b_candidates)) if b_candidates else int(a // 2)
#     return n, K, a, b


# def generate_dataset(output_dir: str, 
#                      num_graphs: int = 500, 
#                     n_range=None, 
#                     K_list=None, 
#                     a_range=None, 
#                     b_range=None): 
#     n_range = n_range or range(100, 5000, 100) 
#     K_list = K_list or [2,3,4] 
#     a_range = a_range or range(50, 501, 50) 
#     b_range = b_range or range(10, 301, 10) 

#     os.makedirs(output_dir, exist_ok=True) 

#     count = 0 
#     seed = 0 

#     while count < num_graphs: 
#         n, K, a, b = sample_parameter(n_range, K_list, a_range, b_range) 
#         if verify_a_b(a,b) and verify_n(n, K): 
#             G = generate_gbm(n = n, K = K, a = a, b = b, seed = seed)
#             epsilon_threshold = calculate_epsilon_threshold(a, b, K, n)
#             data = json_graph.node_link_data(G) 

#             record = {
#                 'parameters': {'n': n, 'K': K, 'a': a, 'b': b, 'epsilon_threshold': epsilon_threshold, 'seed': seed}, 
#                 'graph': data
#             }

#             filename = os.path.join(output_dir, f'graph_{count+1:03d}.json') 
#             with open(filename, 'w') as f:
#                 json.dump(record, f) 
#             count += 1 
#         seed += 1


# if __name__ == "__main__": 
#     generate_dataset("gbm_graphs_data", num_graphs=500)

import json
import os

import numpy as np
from tqdm import tqdm
from networkx.readwrite import json_graph

from graph_generation.gbm import generate_gbm


def verify_a_b(a: int, b: int) -> bool:
    return np.sqrt(a) - np.sqrt(b) > 2 * np.sqrt(2)


def verify_n(n: int, K: int) -> bool:
    return n >= 50 * K


def calculate_epsilon_threshold(a: int, b: int, q: int, n: int) -> float:
    p_in = a * np.log(n) / n
    p_out = b * np.log(n) / n
    c = (p_in + (q - 1) * p_out) / q
    return (np.sqrt(c) - 1) / (np.sqrt(c) - 1 + q)


def sample_parameter(n_range, K_list, a_range, b_range):
    n = int(np.random.choice(n_range))
    K = int(np.random.choice(K_list))
    a = int(np.random.choice(a_range))
    b_candidates = [bb for bb in b_range if bb < a]
    b = int(np.random.choice(b_candidates)) if b_candidates else a // 2
    return n, K, a, b


def generate_dataset(
    output_dir: str,
    num_graphs_below: int = 450,
    num_graphs_above: int = 50,
    n_range=None,
    K_list=None,
    a_range=None,
    b_range=None,
):
    os.makedirs(output_dir, exist_ok=True)

    n_range = n_range or range(100, 5000, 100)
    K_list = K_list or [2, 3, 4]
    a_range = a_range or range(50, 501, 50)
    b_range = b_range or range(10, 301, 10)

    count_below = 0
    count_above = 0
    total_count = 0
    seed = 0

    total_target = num_graphs_below + num_graphs_above
    with tqdm(total=total_target, desc="Generating GBM graphs") as pbar:
        while count_below < num_graphs_below or count_above < num_graphs_above:
            n, K, a, b = sample_parameter(n_range, K_list, a_range, b_range)
            if not (verify_a_b(a, b) and verify_n(n, K)):
                seed += 1
                continue

            # compute epsilon and threshold
            p_in = a * np.log(n) / n
            p_out = b * np.log(n) / n
            epsilon = p_out / p_in
            threshold = calculate_epsilon_threshold(a, b, K, n)

            # check category
            is_below = False
            if epsilon < threshold and count_below < num_graphs_below:
                count_below += 1
                is_below = True
            elif epsilon >= threshold and count_above < num_graphs_above:
                count_above += 1
                is_below = False
            else:
                seed += 1
                continue

            # generate and save
            G = generate_gbm(n=n, K=K, a=a, b=b, seed=seed)
            data = json_graph.node_link_data(G)
            record = {
                'parameters': {
                    'n': n,
                    'K': K,
                    'a': a,
                    'b': b,
                    'epsilon': epsilon,
                    'epsilon_threshold': threshold,
                    'is_below': str(is_below),
                    'seed': seed,
                },
                'graph': data,
            }

            total_count += 1
            filename = os.path.join(output_dir, f'graph_{total_count:03d}.json')
            print("Processed:", filename)
            with open(filename, 'w') as f:
                json.dump(record, f)

            seed += 1
            pbar.update(1)




if __name__ == '__main__':
    generate_dataset('datasets/gbm_graphs_data_w_threshold/', num_graphs_below=450, num_graphs_above=50)















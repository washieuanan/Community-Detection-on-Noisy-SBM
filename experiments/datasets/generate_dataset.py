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
    n_range   = n_range   or range(100, 1000, 50)
    K_list    = K_list    or [2]
    a_range   = a_range   or range(25, 300, 25)
    b_range   = b_range   or range(5, 150, 10)
    os.makedirs(output_dir, exist_ok=True)

    # Define sparsities; ensure 0.05 is first / max
    sparsities = [0.05, 0.01, 0.005, 0.0025, 0.001]

    count = 0
    seed  = 0

    while count < num_graphs:
        # randomly pick parameters
        n, K, a, b = sample_parameter(n_range, K_list, a_range, b_range)
        if not (verify_a_b(a, b) and verify_n(n, K)):
            seed += 1
            continue

        # generate GBM
        G = generate_gbm(n=n, K=K, a=a, b=b, seed=seed)
        epsilon_threshold = calculate_epsilon_threshold(a, b, K, n)
        data = json_graph.node_link_data(G)

        # compute density
        avg_deg = np.mean(list(dict(G.degree()).values()))
        orig_density = avg_deg / n

        # prepare RNG
        rng = np.random.default_rng(seed)

        # first, full obs at max sparsity = 0.05
        max_s = sparsities[0]
        C_full    = max_s * orig_density
        num_full  = int((C_full * n**2) / 2)
        obs_full  = PairSamplingObservation(G, num_samples=num_full,
                                            weight_func=lambda u,v: 1.0,
                                            seed=seed).observe()

        # now downsample for lower sparsities
        observations_data = {}
        observations_data[max_s] = obs_full

        for s in sparsities[1:]:
            frac      = s / max_s
            sample_sz = int(len(obs_full) * frac)
            # choose without replacement
            idxs      = rng.choice(len(obs_full),
                                    size=sample_sz,
                                    replace=False)
            observations_data[s] = [obs_full[i] for i in idxs]

        # write out
        record = {
            'parameters': {
                'n': n, 'K': K, 'a': a, 'b': b, 'density': orig_density,
                'epsilon_threshold': epsilon_threshold,
                'seed': seed
            },
            'graph': data,
            'observations': observations_data
        }

        fname = os.path.join(output_dir, f'graph_{count+1:03d}.json')
        with open(fname, 'w') as f:
            json.dump(record, f)

        count += 1
        seed  += 1

if __name__ == "__main__":
    generate_dataset("datasets/GBM_W_OBS_DENSITY", num_graphs=500)
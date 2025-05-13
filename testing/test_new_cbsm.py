import os
import json
import glob
import warnings
import multiprocessing
import networkx as nx
from networkx.readwrite import json_graph
import numpy as np

from cbsm.sbm import generate_noisy_sbm


from algorithms.duo_spec import (
    duo_bprop, 
    duo_spec, 
    create_dist_observed_subgraph, 
    erdos_renyi_mask, 
    get_true_communities,
    detection_stats
) 

from testing.testing_methods import _json_convert

from cbsm.motif import motif_counting_community_detection
from cbsm.spectral import spectral_clustering_community_detection
from community_detection.bp.vectorized_bp import belief_propagation


# Set BLAS threading to use all CPUs
n_cpus = multiprocessing.cpu_count()
os.environ['OPENBLAS_NUM_THREADS'] = str(n_cpus)
os.environ['MKL_NUM_THREADS']      = str(n_cpus)
os.environ['OMP_NUM_THREADS']      = str(n_cpus)

def sample_parameter(n_range, K_list,  a_range, b_range): 
    n = int(np.random.choice(n_range))
    K = int(np.random.choice(K_list))
    a = int(np.random.choice(a_range))

    # ensure b < a
    b_candidates = [bb for bb in b_range if bb < a]
    b = int(np.random.choice(b_candidates)) if b_candidates else int(a // 2)
    return n, K, a, b


# def make_graph_and_test(
#         output_dir: str, 
#         num_graphs: int = 500, 
#         n_range = None, 
#         K_list = None, 
#         a_range = None, 
#         b_range = None): 
    
#     n_range   = n_range   or range(100, 1000, 50)
#     K_list    = K_list    or [2]
#     a_range   = a_range   or range(15, 100, 5)
#     b_range   = b_range   or range(1, 50, 5)
#     os.makedirs(output_dir, exist_ok=True)



#     # We want to test on full G_true (varying sigma parameter)
#     # We want to test on Erdos renyi masking (varying parameters of masking, 0.1, 0.05, 0.025, 0.1, 0.25 with sigma = 0.5) 


#     sigma_parameters = [0.1, 0.25, 0.5, 0.75] 
#     masking = [0.25, 0.1, 0.05, 0.025, 0.01] 


#     # So, we want to test on duo_spec, duo_bprop, motif_counting, spectral_clustering 
#     for sigma_parameter in sigma_parameters: 
#         n, K, a, b = sample_parameter(n_range, K_list, a_range, b_range)
    



def run_experiments():
    # Create output directories
    os.makedirs("results/varying_sigma", exist_ok=True)
    os.makedirs("results/varying_rho", exist_ok=True)
    
    # Parameter ranges
    n_range = range(100, 1000, 50)
    K_list = [2]
    a_range = range(15, 100, 5)
    b_range = range(1, 50, 5)
    
    def sample_valid_parameters():
        while True:
            n = int(np.random.choice(n_range))
            K = int(np.random.choice(K_list))
            a = int(np.random.choice(a_range))
            
            # ensure b < a and sqrt(a) - sqrt(b) > 2*sqrt(2)
            b_candidates = [bb for bb in b_range if bb < a and np.sqrt(a) - np.sqrt(bb) > 2*np.sqrt(2)]
            if b_candidates:
                b = int(np.random.choice(b_candidates))
                return n, K, a, b
    
    # Varying sigma experiment
    sigma_values = [0.1, 0.25, 0.5, 0.75]
    graphs_per_sigma = 125 // len(sigma_values)  # This will be 31 graphs per sigma value
    
    for sigma in sigma_values:
        for i in range(graphs_per_sigma):
            # Sample parameters
            n, K, a, b = sample_valid_parameters()
            p_in = a * np.log(n) / n
            p_out = b * np.log(n) / n
            
            # Generate graph
            G_true = generate_noisy_sbm(n, K, p_in, p_out, sigma, seed=42+i)
            
            # Run community detection methods
            results = {}
            
            # Motif counting
            try:
                preds = motif_counting_community_detection(G_true, K=K)
                true_labels = get_true_communities(G_true, node2idx=None, attr="comm")
                stats = detection_stats(preds, true_labels)
                results['motif_counting'] = {
                    'stats': stats,
                    'preds': preds
                }
            except Exception as e:
                results['motif_counting'] = {'error': str(e)}
            
            # Spectral clustering
            try:
                preds = spectral_clustering_community_detection(G_true, K=K)
                true_labels = get_true_communities(G_true, node2idx=None, attr="comm")
                stats = detection_stats(preds, true_labels)
                results['spectral_clustering'] = {
                    'stats': stats,
                    'preds': preds
                }
            except Exception as e:
                results['spectral_clustering'] = {'error': str(e)}
            
            # Duo spec
            try:
                res = duo_spec(G_true, K=K)
                preds = res['communities']
                true_labels = get_true_communities(G_true, node2idx=None, attr="comm")
                stats = detection_stats(preds, true_labels)
                results['duo_spec'] = {
                    'stats': stats,
                    'preds': preds
                }
            except Exception as e:
                results['duo_spec'] = {'error': str(e)}
            
            # Duo bprop
            try:
                res = duo_bprop(G_true, K=K)
                preds = res['communities']
                true_labels = get_true_communities(G_true, node2idx=None, attr="comm")
                stats = detection_stats(preds, true_labels)
                results['duo_bprop'] = {
                    'stats': stats,
                    'preds': preds
                }
            except Exception as e:
                results['duo_bprop'] = {'error': str(e)}
            
            # Save results
            output = {
                'parameters': {
                    'n': n,
                    'K': K,
                    'a': a,
                    'b': b,
                    'p_in': p_in,
                    'p_out': p_out,
                    'sigma': sigma
                },
                'graph': nx.node_link_data(G_true, edges="links"),
                'results': results
            }
            
            filename = f"sigma_{str(sigma).replace('.', '')}_run_{i}.json"
            with open(os.path.join("results/varying_sigma", filename), 'w') as f:
                json.dump(output, f, default=_json_convert)
    
    # Varying rho experiment
    sigma = 0.9  # fixed sigma
    rho_values = [0.25, 0.1, 0.05, 0.025, 0.01]
    graphs_per_rho = 125 // len(rho_values)  # This will be 25 graphs per rho value
    
    for rho in rho_values:
        for i in range(graphs_per_rho):
            # Sample parameters
            n, K, a, b = sample_valid_parameters()
            p_in = a * np.log(n) / n
            p_out = b * np.log(n) / n
            
            # Generate base graph
            G_true = generate_noisy_sbm(n, K, p_in, p_out, sigma, seed=42+i)
            
            # Apply Erdos-Renyi masking
            G_masked = erdos_renyi_mask(G_true, rho, seed=42+i)
            
            # Run community detection methods
            results = {}
            
            # Motif counting
            try:
                preds = motif_counting_community_detection(G_masked, K=K)
                true_labels = get_true_communities(G_true, node2idx=None, attr="comm")
                stats = detection_stats(preds, true_labels)
                results['motif_counting'] = {
                    'stats': stats,
                    'preds': preds
                }
            except Exception as e:
                results['motif_counting'] = {'error': str(e)}
            
            # Spectral clustering
            try:
                preds = spectral_clustering_community_detection(G_masked, K=K)
                true_labels = get_true_communities(G_true, node2idx=None, attr="comm")
                stats = detection_stats(preds, true_labels)
                results['spectral_clustering'] = {
                    'stats': stats,
                    'preds': preds
                }
            except Exception as e:
                results['spectral_clustering'] = {'error': str(e)}
            
            # Duo spec
            try:
                res = duo_spec(G_masked, K=K)
                preds = res['communities']
                true_labels = get_true_communities(G_true, node2idx=None, attr="comm")
                stats = detection_stats(preds, true_labels)
                results['duo_spec'] = {
                    'stats': stats,
                    'preds': preds
                }
            except Exception as e:
                results['duo_spec'] = {'error': str(e)}
            
            # Duo bprop
            try:
                res = duo_bprop(G_masked, K=K)
                preds = res['communities']
                true_labels = get_true_communities(G_true, node2idx=None, attr="comm")
                stats = detection_stats(preds, true_labels)
                results['duo_bprop'] = {
                    'stats': stats,
                    'preds': preds.tolist()
                }
            except Exception as e:
                results['duo_bprop'] = {'error': str(e)}
            
            # Save results
            output = {
                'parameters': {
                    'n': n,
                    'K': K,
                    'a': a,
                    'b': b,
                    'p_in': p_in,
                    'p_out': p_out,
                    'sigma': sigma,
                    'rho': rho
                },
                'graph': nx.node_link_data(G_masked, edges="links"),
                'results': results
            }
            
            filename = f"rho_{str(rho).replace('.', '')}_run_{i}.json"
            with open(os.path.join("results/varying_rho", filename), 'w') as f:
                json.dump(output, f, default=_json_convert)

if __name__ == "__main__":
    run_experiments()



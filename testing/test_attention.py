import os
import json
import multiprocessing
import numpy as np
import networkx as nx

from block_models.sbm.sbm import generate_noisy_sbm
from algorithms.controls.sbm_modified.motif import motif_counting_community_detection
from algorithms.duo_spec import duo_spec, detection_stats, get_true_communities, bethe_hessian, laplacian
from algorithms.spectral_ops.attention import (
    motif_spectral_embedding,
    motif_laplacian_spectral_embedding
)
from algorithms.bp.vectorized_bp import belief_propagation

# Set BLAS threading to use all CPUs
n_cpus = multiprocessing.cpu_count()
os.environ['OPENBLAS_NUM_THREADS'] = str(n_cpus)
os.environ['MKL_NUM_THREADS']      = str(n_cpus)
os.environ['OMP_NUM_THREADS']      = str(n_cpus)

def sample_valid_parameters(n_range, K_list, a_range, b_range):
    while True:
        n = int(np.random.choice(n_range))
        K = int(np.random.choice(K_list))
        a = int(np.random.choice(a_range))
        # ensure b < a and sqrt(a) - sqrt(b) > 2*sqrt(2)
        b_cands = [bb for bb in b_range if bb < a and np.sqrt(a) - np.sqrt(bb) > 2*np.sqrt(2)]
        if b_cands:
            b = int(np.random.choice(b_cands))
            return n, K, a, b

def run_experiments():
    out_dir = "results/attention_results"
    os.makedirs(out_dir, exist_ok=True)
    
    n_range = range(100, 1000, 50)
    K_list  = [2]
    a_range = range(15, 100, 5)
    b_range = range(1, 50, 5)
    
    sigma_values = [0.1, 0.25, 0.5, 0.75]
    total_graphs = 250
    per_sigma    = total_graphs // len(sigma_values)  # 62
    remainder    = total_graphs % len(sigma_values)   # 2
    
    for idx, sigma in enumerate(sigma_values):
        runs = per_sigma + (1 if idx < remainder else 0)
        for i in range(runs):
            # Sample parameters and generate graph
            n, K, a, b = sample_valid_parameters(n_range, K_list, a_range, b_range)
            p_in  = a * np.log(n) / n
            p_out = b * np.log(n) / n
            G_true = generate_noisy_sbm(n, K, p_in, p_out, sigma, seed=42 + i)
            
            results = {}
            
            # 1) motif_spectral_embedding
            try:
                Q_byoe, preds_byoe, node2idx_byoe, idx2node_byoe = motif_spectral_embedding(
                    G_true, q=K
                )
                true_labels = np.array([G_true.nodes[u]["comm"] for u in G_true.nodes()])
                stats_byoe = detection_stats(preds_byoe, true_labels)
                results['motif_attention'] = {'stats': stats_byoe, 'preds': preds_byoe}
            except Exception as e:
                results['motif_attention'] = {'error': str(e)}
            
            # 2) motif_laplacian_spectral_embedding
            try:
                Q_bh, preds_bh, node2idx_bh, idx2node_bh = motif_laplacian_spectral_embedding(
                    G_true, q=K
                )
                true_labels = np.array([G_true.nodes[u]["comm"] for u in G_true.nodes()])
                stats_bh = detection_stats(preds_bh, true_labels)
                results['motif_laplacian'] = {'stats': stats_bh, 'preds': preds_bh}
            except Exception as e:
                results['motif_laplacian'] = {'error': str(e)}


            # Bethe-Hessian
            try:
                Q_bh, preds_bh, node2idx_bh, idx2node_bh = bethe_hessian(
                    G_true, q=K
                )
                true_labels_bh = np.array([G_true.nodes[u]["comm"] for u in G_true.nodes()])
                stats_bh = detection_stats(preds_bh, true_labels_bh)
                results['CONTROL_bethe_hessian'] = {'stats': stats_bh, 'preds': preds_bh}
            except Exception as e:
                print("Error on bethe_hessian")
                results['CONTROL_bethe_hessian'] = {'error': str(e)}
            
        
            # 3) duo_spec with Bethe-Hessian
            try:
                res = duo_spec(G_true, K=K, config='bethe_hessian')
                preds = res['communities']
                true_labels = get_true_communities(G_true, node2idx=None, attr="comm")
                stats = detection_stats(preds, true_labels)
                results['duo_spec_bethe_hessian'] = {'stats': stats, 'preds': preds}
            except Exception as e:
                results['duo_spec_bethe_hessian'] = {'error': str(e)}
            
            # 4) duo_spec with motif
            try:
                res = duo_spec(G_true, K=K, config='motif')
                preds = res['communities']
                true_labels = get_true_communities(G_true, node2idx=None, attr="comm")
                stats = detection_stats(preds, true_labels)
                results['duo_spec_motif_attention'] = {'stats': stats, 'preds': preds}
            except Exception as e:
                results['duo_spec_motif_attention'] = {'error': str(e)}
            
            # 5) Belief Propagation
            try:
                _, preds_bp, node2idx_bp, _ = belief_propagation(G_true, q=K)
                true_labels_bp = get_true_communities(G_true, node2idx=node2idx_bp, attr="comm")
                stats_bp = detection_stats(preds_bp, true_labels_bp)
                results['CONTROL_belief_propagation'] = {'stats': stats_bp, 'preds': preds_bp}
            except Exception as e:
                results['CONTROL_belief_propagation'] = {'error': str(e)}
            
            try: 
                mc_preds = motif_counting_community_detection(G_true, K = K) 
                true_labels_mc = get_true_communities(G_true, node2idx=None, attr="comm")
                stats_mc = detection_stats(mc_preds, true_labels_mc)
                results['CONTROL_motif_counting'] = {
                    'stats': stats_mc,
                    'preds': mc_preds
                }
            except Exception as e:
                results['motif_counting'] = {'error': str(e)}


            
            
            # Save to JSON
            output = {
                'parameters': {
                    'n': n, 'K': K, 'a': a, 'b': b,
                    'p_in': p_in, 'p_out': p_out, 'sigma': sigma
                },
                'graph': nx.node_link_data(G_true, edges="links"),
                'results': results
            }
            fname = f"sigma_{str(sigma).replace('.', '')}_run_{i}.json"
            print("Saving to ", fname)
            with open(os.path.join(out_dir, fname), 'w') as f:
                json.dump(output, f, default=lambda o: o.to_dict() if hasattr(o, 'to_dict') else str(o))

if __name__ == "__main__":
    run_experiments()

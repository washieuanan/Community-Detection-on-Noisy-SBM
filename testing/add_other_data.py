import os
import json
import glob
import multiprocessing
import networkx as nx
from networkx.readwrite import json_graph
import numpy as np

from cbsm.sbm import generate_sbm
from algorithms.duo_spec import duo_spec, get_true_communities, detection_stats, erdos_renyi_mask, duo_bprop
from community_detection.bp.vectorized_bp import belief_propagation
from cbsm.motif import motif_counting_community_detection
from cbsm.spectral import spectral_clustering_community_detection
from testing.testing_methods import _json_convert

# Ensure directories exist
sigma_dir = "results/varying_sigma"
rho_dir = "results/varying_rho"
const_dir = "results/varying_rho_constant_sigma"
os.makedirs(const_dir, exist_ok=True)

# Helper to load graph from JSON data
def load_graph(data):
    return json_graph.node_link_graph(data["graph"])

# Process existing results: augment with belief propagation and detection stats
def augment_with_bp(input_dir, param_key):
    for path in glob.glob(os.path.join(input_dir, "*.json")):
        with open(path, 'r') as f:
            data = json.load(f)

        G = load_graph(data)
        params = data.get("parameters", {})
        K = params.get("K")

        # Run belief propagation
        _, preds, node2idx, _ = belief_propagation(G, q=K)
        # Calculate true labels and stats
        true_labels = get_true_communities(G, node2idx=node2idx, attr="comm")
        stats = detection_stats(preds, true_labels)

        # Store under results
        data.setdefault("results", {})
        data["results"]["belief_propagation"] = {
            "preds": preds.tolist() if hasattr(preds, 'tolist') else preds,
            "node2idx": node2idx,
            "stats": stats
        }

        # Save back
        with open(path, 'w') as f:
            json.dump(data, f, default=_json_convert)

# # Augment both varying_sigma and varying_rho directories
# augment_with_bp(sigma_dir, 'sigma')
# augment_with_bp(rho_dir, 'rho')

# Third loop: generate SBMs at constant sigma, vary rho maskings
sigma_fixed = 0.9
rho_values = [0.25, 0.1, 0.05, 0.025, 0.01]
n_range = range(100, 1000, 50)
K = 2
a_range = range(15, 100, 5)
b_range = range(1, 50, 5)

def sample_valid_params():
    while True:
        n = int(np.random.choice(n_range))
        a = int(np.random.choice(a_range))
        b_cands = [bb for bb in b_range if bb < a and np.sqrt(a) - np.sqrt(bb) > 2*np.sqrt(2)]
        if b_cands:
            b = int(np.random.choice(b_cands))
            return n, a, b

for rho in rho_values:
    print("Trying rho:", rho)
    for i in range(25):  # e.g., 125 total / 5
        # Sample parameters
        
        n, a, b = sample_valid_params()
        p_in = a * np.log(n) / n
        p_out = b * np.log(n) / n

        # Generate base graph and mask
        G_true = generate_sbm(n, K, p_in, p_out, sigma_fixed, seed=42 + i)
        G_masked = erdos_renyi_mask(G_true, rho, seed=42 + i)

        results = {}
        # Belief Propagation
        _, bp_preds, bp_node2idx, _ = belief_propagation(G_masked, q=K)
        true_labels_bp = get_true_communities(G_true, node2idx=bp_node2idx, attr="comm")
        stats_bp = detection_stats(bp_preds, true_labels_bp)
        results['belief_propagation'] = {
            'stats': stats_bp,
            'preds': bp_preds
        }

        # Motif counting
        try:
            mc_preds = motif_counting_community_detection(G_masked, K=K)
            true_labels_mc = get_true_communities(G_true, node2idx=None, attr="comm")
            stats_mc = detection_stats(mc_preds, true_labels_mc)
            results['motif_counting'] = {
                'stats': stats_mc,
                'preds': mc_preds
            }
        except Exception as e:
            results['motif_counting'] = {'error': str(e)}

        # Spectral clustering
        try:
            sc_preds = spectral_clustering_community_detection(G_masked, K=K)
            true_labels_sc = get_true_communities(G_true, node2idx=None, attr="comm")
            stats_sc = detection_stats(sc_preds, true_labels_sc)
            results['spectral_clustering'] = {
                'stats': stats_sc,
                'preds': sc_preds
            }
        except Exception as e:
            results['spectral_clustering'] = {'error': str(e)}

        # DuoSpec
        try:
            ds_res = duo_spec(G_masked, K=K)
            ds_preds = ds_res.get('communities')
            true_labels_ds = get_true_communities(G_true, node2idx=None, attr="comm")
            stats_ds = detection_stats(ds_preds, true_labels_ds)
            results['duo_spec'] = {
                'stats': stats_ds,
                'preds': ds_preds
            }
        except Exception as e:
            results['duo_spec'] = {'error': str(e)}

        # Duo Bprop
        try:
            db_res = duo_bprop(G_masked, K=K)
            db_preds = db_res.get('communities')
            true_labels_db = get_true_communities(G_true, node2idx=None, attr="comm")
            stats_db = detection_stats(db_preds, true_labels_db)
            results['duo_bprop'] = {
                'stats': stats_db,
                'preds': db_preds
            }
        except Exception as e:
            results['duo_bprop'] = {'error': str(e)}

        # Build output JSON
        output = {
            'parameters': {
                'n': n, 'K': K, 'a': a, 'b': b,
                'p_in': p_in, 'p_out': p_out,
                'sigma': sigma_fixed, 'rho': rho
            },
            'graph': json_graph.node_link_data(G_masked),
            'results': results
        }
        fname = f"rho_{str(rho).replace('.', '')}_constsigma_run_{i}.json"
        with open(os.path.join(const_dir, fname), 'w') as f:
            json.dump(output, f, default=_json_convert)

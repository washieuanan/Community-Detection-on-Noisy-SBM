#!/usr/bin/env python3

import os
import json
import glob
import warnings
import multiprocessing
import networkx as nx
from networkx.readwrite import json_graph
import numpy as np

# Import both BP implementations
from community_detection.bp.tests.bethe_duo_bp import duo_bp as bethe_bp
from community_detection.bp.tests.duo_bp import duo_bp as duo_bp
from community_detection.bp.vectorized_bp import detection_stats, get_true_communities
from community_detection.bp.tests.duo_bp import create_dist_observed_subgraph
from community_detection.bp.vectorized_bp import belief_propagation
from controls.motif_count import motif_counting
from controls.spectral import spectral_clustering



# Set BLAS threading to use all CPUs
n_cpus = multiprocessing.cpu_count()
os.environ['OPENBLAS_NUM_THREADS'] = str(n_cpus)
os.environ['MKL_NUM_THREADS']      = str(n_cpus)
os.environ['OMP_NUM_THREADS']      = str(n_cpus)

warnings.filterwarnings("ignore", category=UserWarning, module='networkx')


def _json_convert(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Type {type(o)} not serializable")


def process_file(input_path, output_dir):
    base = os.path.basename(input_path)
    print(f"Processing {base}...")

    # Load JSON
    with open(input_path, 'r') as f:
        data = json.load(f)

    parameters = data.get('parameters', {})
    graph_data = data.get('graph', {})
    obs_dict = data.get('observations', {})

    # Reconstruct the original graph for true communities
    G_full = json_graph.node_link_graph(graph_data, edges="links")

    results = {}

    for C_level, obs_data in obs_dict.items():
        print(f"  Testing sparsity factor, rho = {C_level}")
        # Extract the list of pairs
        if isinstance(obs_data, dict) and 'observations' in obs_data:
            observations = obs_data['observations']
        else:
            observations = obs_data

        if not observations:
            print(f"    Warning: no observations for C = {C_level}")
            continue

        # Build subgraph
        subG = create_dist_observed_subgraph(len(G_full.nodes()), observations)

        # Container for this C-level
        results[C_level] = {}

        # Run both BP methods
        print("Running duo_bp...")
        res = duo_bp(
            subG,
            K=parameters.get('K'),
            seed=parameters.get('seed')
        )

        # Retrieve true communities aligned by node2idx
        preds = res["communities"]
        true_comms = get_true_communities(G_full, node2idx=None, attr='comm')

        # Compute stats
        stats = detection_stats(preds, true_comms)

        # Store
        results[C_level]['duobp'] = {
            'stats': stats,
            'preds': preds
        }

        print("Running regular belief propagation...") 
        _, preds, node2idx, _ = belief_propagation(
            subG, 
            q = parameters.get('K'), 
            seed= parameters.get('seed'),
            balance_regularization=0.05,
            damping=0.15,
            msg_init='random',
            group_obs = None, 
            max_iter= 5000, 
            min_sep = 0.3
        )

        true_comms = get_true_communities(G_full, node2idx=node2idx, attr='comm')
        stats = detection_stats(preds, true_comms)

        results[C_level]['bp'] = {
            'stats': stats,
            'preds': preds
        }

        # Now, run controls 
        print("Running motif counting...")
        preds = motif_counting(subG, q=2)

        true_comms = get_true_communities(G_full, node2idx=None, attr='comm')
        stats = detection_stats(preds, true_comms)

        results[C_level]['motif'] = {
            'stats': stats,
            'preds': preds
        }

        print("Running spectral clustering...")
        preds = spectral_clustering(subG, obs_data, q=2)
        stats = detection_stats(preds, true_comms)
        results[C_level]['spectral'] = {
            'stats': stats,
            'preds': preds
        }


    # Write results to JSON
    os.makedirs(output_dir, exist_ok=True)
    out_name = base.replace('.json', '_test_results.json')
    out_path = os.path.join(output_dir, out_name)
    with open(out_path, 'w') as f:
        json.dump({'parameters': parameters, 'results': results}, f, indent=2, default=_json_convert)

    print(f"Saved results to {out_path}")


def main():
    input_dir = 'datasets/GBM_W_OBS_DENSITY/'
    output_dir = 'results/all_methods_corrected/'
    os.makedirs(output_dir, exist_ok=True)

    json_files = sorted(glob.glob(os.path.join(input_dir, '*.json')))
    print(f"Found {len(json_files)} files in {input_dir}")

    # start after json 112
    # json_files = json_files[27:]
    for path in json_files:
        process_file(path, output_dir)

    print("All testing completed.")


if __name__ == '__main__':
    main()

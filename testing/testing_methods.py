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
# from community_detection.bp.tests.bethe_duo_bp import duo_bp as bethe_bp
from community_detection.bp.tests.duo_bp import duo_bp as duo_bp
from community_detection.bp.vectorized_bp import get_true_communities
from community_detection.bp.tests.duo_bp import create_dist_observed_subgraph
from community_detection.bp.vectorized_bp import belief_propagation
from controls.motif_count import motif_counting
from controls.spectral import spectral_clustering
from algorithms.duo_spec import duo_spec, detection_stats, bethe_hessian
# from controls.spectral import bethe_hessian_clustering



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

        # # Build subgraph
        subG = create_dist_observed_subgraph(len(G_full.nodes()), observations)

        # Container for this C-level
        results[C_level] = {}

        
        # true_labels = get_true_communities(G_full, node2idx=None, attr="comm")
        # # First, run duo_spec laplacian
        # print("Running duo spectral laplacian...")
        # duo_laplacian = duo_spec(subG, parameters['K'], num_balls=32, config='laplacian') 
        # preds = duo_laplacian['communities']

        # stats = detection_stats(preds, true_labels) 

        # results[C_level]['duo_spec_laplacian'] = {
        #     'stats': stats,
        #     'preds': preds
        # } 

        # # Now, run duo_spec bethe
        # print("Running duo spectral bethe...")
        # duo_bethe = duo_spec(subG, parameters['K'], num_balls=32, config='bethe_hessian')
        # preds = duo_bethe['communities']
        # stats = detection_stats(preds, true_labels)
        # results[C_level]['duo_spec_bethe'] = {
        #     'stats': stats,
        #     'preds': preds
        # }

        print("Now, running motif-counnting...")
        # Run motif counting
        preds = motif_counting(subG, parameters['K']) 
        true_labels = get_true_communities(subG, node2idx=None, attr="comm")
        stats = detection_stats(preds, true_labels)
        results[C_level]['motif_counting'] = {
            'stats': stats,
            'preds': preds
        }

        # Now, run regular Bethe-Hessian
        print("Running bethe_hessian...")

        _, labels, _, _ = bethe_hessian(subG, parameters['K'])
        nodes = list(subG.nodes())
        # want something of this nature: np.array([labels[i] for i in range(len(nodes))])
        preds = np.array([labels[i] for i in range(len(nodes))])
        stats = detection_stats(preds, true_labels)
        results[C_level]['bethe_hessian'] = {
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
    output_dir = 'results/corrected_addl_methods/'
    os.makedirs(output_dir, exist_ok=True)

    json_files = sorted(glob.glob(os.path.join(input_dir, '*.json')))
    print(f"Found {len(json_files)} files in {input_dir}")

    # start after json 112
    json_files = json_files[86:]
    for path in json_files:
        process_file(path, output_dir)

    print("All testing completed.")


if __name__ == '__main__':
    main()

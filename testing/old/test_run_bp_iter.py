#!/usr/bin/env python3

import os
import json
import networkx as nx
from networkx.readwrite import json_graph
import numpy as np
import warnings 
import glob

import os
import glob
import concurrent.futures


import multiprocessing

n_cpus = multiprocessing.cpu_count()
os.environ['OPENBLAS_NUM_THREADS'] = str(n_cpus)
os.environ['MKL_NUM_THREADS']      = str(n_cpus)
os.environ['OMP_NUM_THREADS']      = str(n_cpus)


warnings.filterwarnings("ignore", category=UserWarning, module='networkx')

from community_detection.bp.vectorized_bp import (
    belief_propagation, 
    detection_stats, 
    get_true_communities
)

from community_detection.bp.vectorized_bayes import BayesianGraphInference


def _json_convert(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Type {type(o)} not serializable")


def create_observed_subgraph(num_coords: int, observations: list[tuple[int, int]]) -> nx.Graph:
    """
    Create a subgraph that contains all original nodes and only the observed edges.
    """
    subG = nx.Graph()
    # Add all nodes
    subG.add_nodes_from(range(num_coords))
    # Add observed edges
    subG.add_edges_from(observations)
    return subG


def process_file(input_path: str, output_dir: str) -> None:
    """Process a single file."""
    try:
        base = os.path.basename(input_path)
        print(f"Processing {base}...")
        
        # Load JSON data
        with open(input_path, 'r') as f:
            data = json.load(f)

        # Extract parameters and graph
        parameters = data.get('parameters', data.get('params', {}))
        graph_data = data.get('graph', {})

        # Convert JSON graph to NetworkX graph
        G = json_graph.node_link_graph(graph_data, edges="links")

        # Observations key contains all methods
        obs_dict = data.get('observations', {})

        results: dict[str, dict] = {}
        for obs_name, obs_data in obs_dict.items():
            # Skip SingleSensorObservation
            if obs_name != "PairSamplingObservation":
                continue

            print(f"  Processing observation method: {obs_name}")
            
            k = parameters.get('K', 2)
            min_sep_map = {2: 0.3, 3: 0.2, 4: 0.15}
            min_sep = min_sep_map.get(k, 0.15)

            # Check for empty observations
            observation_data = obs_data.get('observations', obs_data)
            if not observation_data:
                print(f"  ⚠️ Warning: No observations for {obs_name}")
                continue
                
            observations = []
            if obs_name.startswith("GroupedMultiSensorObservation_"):
                # Grouped: obs_data is List[Dict[radius, List[Pair]]]
                for g in observation_data:
                    for _, edge_list in g.items():
                        for u, v in edge_list:
                            observations.append((u, v))
                msg_init = 'pre-group'
                group_obs = observation_data
            elif obs_name == "GroupedRandomWalkObservation":
                for g in observation_data:
                    for u, v in g:
                        observations.append((u, v))
                msg_init = 'pre-group'
                group_obs = observation_data
            else:
                observations = observation_data
                group_obs = None
                msg_init = 'random'

            # Check again after processing
            if not observations:
                print(f"  ⚠️ Warning: No processed observations for {obs_name}")
                continue
                
            observed_nodes = set() 
            for u, v in observations:
                observed_nodes.add(u)
                observed_nodes.add(v)

            # Check for empty observed nodes
            if not observed_nodes:
                print(f"  ⚠️ Warning: No observed nodes for {obs_name}")
                continue

            print(f"  Starting Bayesian inference with {len(observations)} observations and {len(observed_nodes)} observed nodes...")
            
            

            # Build observed subgraph and initialize beliefs
            subG = create_observed_subgraph(len(G.nodes()), observations)

            
            print("  Running belief propagation...")
            
            _, preds, node2idx, idx2node = belief_propagation(
                subG,
                q=parameters.get('K'),
                seed=parameters.get('seed'),
                balance_regularization=0.05, 
                damping=0.15,
                msg_init=msg_init,
                group_obs=group_obs,
                max_iter=5000,
                min_sep=min_sep
            )

           
            true = get_true_communities(G, node2idx=node2idx, attr='comm')
        

            sub_preds = np.array([preds[i] for i in range(len(preds)) if i in observed_nodes])
            sub_true = np.array([true[i] for i in range(len(true)) if i in observed_nodes])

            # Compute detection stats
            print("  Computing detection stats...")
            stats = detection_stats(preds, true)
            sub_stats = detection_stats(sub_preds, sub_true)

            # Store results
            results[obs_name] = {
                'observed_nodes': list(observed_nodes),  # Convert set to list for JSON serialization
                'stats': stats,
                'sub_stats': sub_stats,
                'sub_preds': sub_preds.tolist(),  # Ensure numpy arrays are converted for JSON
                'sub_cluster_map': sub_true.tolist(),  # Ensure numpy arrays are converted for JSON
                'sparsity': obs_data.get('sparsity')
            }

        # Write output JSON
        output = {
            'parameters': parameters,
            'results': results
        }
        name = base.replace('.json', '_results.json')
        output_path = os.path.join(output_dir, name)
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=_json_convert)

        print(f"✅ Successfully processed and saved: {base} -> {output_path}")
    
    except Exception as e:
        print(f"❌ Error processing {os.path.basename(input_path)}: {str(e)}")


def main():
    input_dir  = 'datasets/observations_generation/gbm_observation_0005'
    output_dir = 'results/bp_05_05_results/0005'

    os.makedirs(output_dir, exist_ok=True)
    print(f"✅ Output directory created/verified: {output_dir}")

    # Grab every .json in input_dir
    input_paths = sorted(glob.glob(os.path.join(input_dir, '*.json')))
    print(f"→ Found {len(input_paths)} JSON files to process.")

    for path in input_paths:
        base = os.path.basename(path)
        print(f"\n--- Processing {base} ---")
        try:
            process_file(path, output_dir)
        except Exception as e:
            print(f"❌ Error processing {base}: {e}")

    print("\n✅ All processing completed!")




if __name__ == '__main__':
    main()

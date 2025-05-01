#!/usr/bin/env python3

import os
import json
import networkx as nx
from networkx.readwrite import json_graph
import numpy as np
from tqdm import tqdm
import concurrent.futures

from community_detection.bp.belief_prop import (
    belief_propagation, 
    initialize_beliefs,
    get_marginals_and_preds, 
    detection_stats
)


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
    """Process a single file, handling all observations sequentially."""
    try:
        # Load JSON data
        with open(input_path, 'r') as f:
            data = json.load(f)

        # Extract parameters and graph
        parameters = data.get('parameters', data.get('params', {}))
        graph_data = data.get('graph', {})

        # Convert JSON graph to NetworkX graph
        G = json_graph.node_link_graph(graph_data)

        # Observations key contains all methods
        obs_dict = data.get('observations', {})

        results: dict[str, dict] = {}
        for obs_name, obs_data in obs_dict.items():
            # Skip SingleSensorObservation
            if obs_name == "SingleSensorObservation":
                continue

            k = parameters.get('K', 2)
            min_sep_map = {2: 0.3, 3: 0.2, 4: 0.15}
            min_sep = min_sep_map.get(k, 0.15)

            observations = []
            observation_data = obs_data.get('observations', obs_data)
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

            # Build observed subgraph and initialize beliefs
            subG = create_observed_subgraph(len(G.nodes()), observations)
            initialize_beliefs(subG, parameters.get('K'))

            # Run belief propagation
            belief_propagation(
                subG,
                q=parameters.get('K'),
                seed=parameters.get('seed'),
                message_init=msg_init,
                group_obs=group_obs,
                max_iter=5000,
                min_sep=min_sep
            )

            # Retrieve predictions
            _, preds = get_marginals_and_preds(subG)

            observed_nodes = sorted({u for u, v in observations} | {v for u, v in observations})

            # True community labels from original graph
            true = nx.get_node_attributes(G, 'comm')
            true = np.array([true[i] for i in range(len(G.nodes()))])
            sub_preds = np.array([preds[i] for i in observed_nodes])
            sub_true = np.array([true[i] for i in observed_nodes])

            # Compute detection stats
            stats = detection_stats(preds, true)
            sub_stats = detection_stats(sub_preds, sub_true)

            # Store results
            results[obs_name] = {
                'observed_nodes': observed_nodes,
                'stats': stats,
                'sub_stats': sub_stats,
                'sub_preds': sub_preds,
                'sub_cluster_map': sub_true,
                'sparsity': obs_data.get('sparsity')
            }

        # Write output JSON
        output = {
            'parameters': parameters,
            'results': results
        }
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.basename(input_path)
        name = base.replace('.json', '_results.json')
        output_path = os.path.join(output_dir, name)
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=_json_convert)

        return f"Processed {base} -> {output_path}"
    
    except Exception as e:
        return f"Error processing {os.path.basename(input_path)}: {str(e)}"


def process_all_files(input_dir: str, output_dir: str, max_workers: int = None):
    """Process all files in parallel using concurrent.futures."""
    # Use sorted os.listdir and filter for graph_*.json
    files = sorted(
        f for f in os.listdir(input_dir)
        if f.startswith('graph_') and f.endswith('.json')
    )
    
    # Full paths for all files
    file_paths = [os.path.join(input_dir, filename) for filename in files]
    
    # Setup progress bar for completed files
    pbar = tqdm(total=len(files), desc='Processing files')
    
    # Process files in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all file processing tasks
        future_to_file = {
            executor.submit(process_file, path, output_dir): path
            for path in file_paths
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = future.result()
                print(result)  # Print the result message
            except Exception as exc:
                print(f'{os.path.basename(file_path)} generated an exception: {exc}')
            finally:
                pbar.update(1)  # Update progress bar
    
    pbar.close()


if __name__ == '__main__':
    input_dir = 'observations_generation/gbm_observation_01'
    output_dir = 'bp_04_30_results/01'
    
    # Creates output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set max_workers=None to use all available CPU cores, or specify a number
    process_all_files(input_dir, output_dir, max_workers=None)
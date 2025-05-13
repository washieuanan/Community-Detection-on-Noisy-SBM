#!/usr/bin/env python3

import os
import json
import networkx as nx
from networkx.readwrite import json_graph
import numpy as np
from tqdm import tqdm
import concurrent.futures
import re

import warnings 
warnings.filterwarnings("ignore", category=UserWarning, module='networkx')

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
            if obs_name == "SingleSensorObservation" or obs_name == "VertexBasedSamplingObservation":
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

        # Return success message with green checkmark emoji
        return f"✅ Successfully processed and saved: {base} -> {output_path}"
    
    except Exception as e:
        # Return error message without checkmark
        return f"❌ Error processing {os.path.basename(input_path)}: {str(e)}"


def filter_files_by_range(files_list, start_num, end_num):
    """Filter files by numeric range in filename (graph_XXX.json)."""
    filtered_files = []
    pattern = re.compile(r'graph_(\d+)\.json')
    
    for file in files_list:
        match = pattern.match(file)
        if match:
            num = int(match.group(1))
            if start_num <= num <= end_num:
                filtered_files.append(file)
    
    return filtered_files


def process_file_batch(input_dir, output_dir, file_range, max_workers=None):
    """Process a batch of files based on their number range."""
    start_num, end_num = file_range
    
    # Get all graph_*.json files
    all_files = [
        f for f in os.listdir(input_dir)
        if f.startswith('graph_') and f.endswith('.json')
    ]
    
    # Filter files by range
    files_to_process = filter_files_by_range(all_files, start_num, end_num)
    
    # Create full paths
    file_paths = [os.path.join(input_dir, filename) for filename in files_to_process]
    
    # Print batch info
    print(f"Processing batch: graph_{start_num:03d}.json to graph_{end_num:03d}.json")
    print(f"Found {len(file_paths)} files to process")
    
    if not file_paths:
        print("❌ No files match the specified range!")
        return
    
    # Setup progress bar
    pbar = tqdm(total=len(file_paths), desc=f'Batch {start_num}-{end_num}')
    
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
                print(result)  # Print the result message with emoji
            except Exception as exc:
                print(f"❌ {os.path.basename(file_path)} generated an exception: {exc}")
            finally:
                pbar.update(1)  # Update progress bar
    
    pbar.close()
    print(f"✅ Completed batch {start_num}-{end_num}")


if __name__ == '__main__':
    input_dir = 'datasets/observations_generation/gbm_observation_01'
    output_dir = 'results/bp_05_01_results/01'
    
    # Creates output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"✅ Output directory created/verified: {output_dir}")
    
    # Define the two ranges to process
    batch_ranges = [
        (1, 50),    # graph_001.json to graph_050.json
        (100, 125)  # graph_100.json to graph_125.json
    ]
    
    # Process each batch sequentially
    for batch_range in batch_ranges:
        process_file_batch(input_dir, output_dir, batch_range, max_workers=None)
    
    print("✅ All batches completed successfully!")
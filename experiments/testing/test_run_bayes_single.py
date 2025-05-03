#!/usr/bin/env python3

import os
import json
import networkx as nx
from networkx.readwrite import json_graph
import numpy as np
import warnings 

warnings.filterwarnings("ignore", category=UserWarning, module='networkx')

from community_detection.bp.belief_prop import (
    belief_propagation, 
    initialize_beliefs,
    get_marginals_and_preds, 
    detection_stats
)
from community_detection.bp.bayes_bp import BayesianGraphInference


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
            if obs_name != "PairSamplingObservation":
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

            observed_nodes = set() 
            for u, v in observations:
                observed_nodes.add(u)
                observed_nodes.add(v)

            bayes = BayesianGraphInference(
                observations=observations,
                observed_nodes=observed_nodes, 
                total_nodes = len(G.nodes()),
                obs_format='base', 
                n_candidates=2**20
            )

            predicted_graph = bayes.infer() 

            # Build observed subgraph and initialize beliefs
            subG = create_observed_subgraph(len(G.nodes()), observations)

            for n in subG.nodes(): 
                subG.nodes[n]['coord'] = predicted_graph.nodes[n]['coord']

            gamma = 1.0 
            
            for G_current in (predicted_graph, subG): 
                for u, v in G_current.edges(): 
                    d = np.linalg.norm(predicted_graph.nodes[u]['coord'] - predicted_graph.nodes[v]['coord'])
                    psi = np.ones((k,k))
                    np.fill_diagonal(psi, np.exp(-gamma *d)) 
                    G_current[u][v]['psi'] = psi 

            initialize_beliefs(subG, k)
            initialize_beliefs(predicted_graph, k-1)

            # Run belief propagation
            belief_propagation(
                subG,
                q=parameters.get('K'),
                seed=parameters.get('seed'),
                balance_regularization=0.05, 
                min_steps = 50, 
                damping = 0.15,
                message_init=msg_init,
                group_obs=group_obs,
                max_iter=5000,
                min_sep=min_sep
            )

            # Retrieve predictions
            marginals, preds = get_marginals_and_preds(subG)

            # True community labels from original graph
            true = nx.get_node_attributes(G, 'comm')
            true = np.array(list(true.values()))

            sub_preds = np.array([preds[i] for i in range(len(preds)) if i in observed_nodes])
            sub_true = np.array([true[i] for i in range(len(true)) if i in observed_nodes])

            # Compute detection stats
            stats = detection_stats(preds, true)
            sub_stats = detection_stats(sub_preds, sub_true)

            # Store results
            results[obs_name] = {
                'observed_nodes': list(observed_nodes),  # Convert set to list for JSON serialization
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

        print(f"✅ Successfully processed and saved: {base} -> {output_path}")
    
    except Exception as e:
        print(f"❌ Error processing {os.path.basename(input_path)}: {str(e)}")


if __name__ == '__main__':
    input_path = 'datasets/observations_generation/gbm_observation_01/graph_013.json'
    output_dir = 'results/bayes_bp_05_02_results/01'
    
    # Creates output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"✅ Output directory created/verified: {output_dir}")
    
    # Process the specific file
    process_file(input_path, output_dir)
    
    print("✅ Processing completed!")   
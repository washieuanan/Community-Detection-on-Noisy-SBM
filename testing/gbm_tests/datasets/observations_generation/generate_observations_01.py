import os
import json
import math
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
from typing import Tuple
from tqdm import tqdm

# Adjust these imports to match your project structure
from observations import random_walk_obs, sensor_observe, standard_observe

# JSON serializer helper for numpy types
def _json_convert(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Type {type(o)} not serializable")


def get_coordinate_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """
    Compute the Euclidean distance between two coordinate vectors.
    """
    return np.linalg.norm(np.array(coord1) - np.array(coord2))

# Directories
INPUT_DIR = 'datasets/gbm_graphs_data_w_threshold/'
OUTPUT_DIR = 'datasets/observations_w_distances/010'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Iterate over each JSON file
for filename in tqdm(sorted(os.listdir(INPUT_DIR)), desc="Processing graphs", unit="file"):

    if not filename.endswith('.json'):  
        continue
    input_path = os.path.join(INPUT_DIR, filename)
    with open(input_path, 'r') as f:
        data = json.load(f)

    # JSON files use 'parameters' key
    params = data.get('parameters', {})
    graph_data = data['graph']
    # Reconstruct the NetworkX graph
    G = json_graph.node_link_graph(graph_data)
    n = G.number_of_nodes()

    observations = {}

    
    # # SingleSensorObservation
    # single = sensor_observe.SingleSensorObservation(
    #     G,
    #     seed=42,
    #     sensor=0,
    #     radii=np.linspace(0.1, 1.0, 10)
    # )
    # observations['SingleSensorObservation'] = single.observe()

    # MultiSensorObservation and GroupedMultiSensorObservation
    k = params.get('K', 2)
    min_sep_map = {2: 0.3, 3: 0.2, 4: 0.15}
    min_sep = min_sep_map[k]

    avg_degree = np.mean(list(dict(G.degree()).values()))
    original_density = avg_degree / len(G.nodes)
    C = 0.1 * original_density 


    num_walkers = max(1,int((C * 0.1 * len(G.nodes)) / 2))
    num_sensors = max(1, int((C * len(G.nodes)) / ((0.25)**3 * avg_degree)))
    
    # multi = sensor_observe.MultiSensorObservation(
    #     G,
    #     seed=params.get('seed', 42),
    #     num_sensors=num_sensors,
    #     min_sep=min_sep,
    #     radii=np.linspace(0.1, 1.0, 10),
    #     deduplicate_edges=True,
    # )
    
    # multi_observe = multi.observe()
    # sparsity = multi.get_count() / (len(G.edges) * len(G.nodes))
    
    
    # observations[f'MultiSensorObservation_k{k}'] = {'observations':multi_observe, 'sparsity': sparsity}

    
    # grouped = sensor_observe.GroupedMultiSensorObservation(
    #     G,
    #     seed=params.get('seed', 42),
    #     num_sensors=num_sensors,
    #     min_sep=min_sep,
    #     radii=np.linspace(0.1, 1.0, 10),
    #     deduplicate_edges=True,
    # )

    # grouped_observe = grouped.observe()
    # sparsity = grouped.get_count() / (len(G.edges) * len(G.nodes))
    # observations[f'GroupedMultiSensorObservation_k{k}'] = {'observations': grouped_observe, 'sparsity': sparsity}

    # # RandomWalkObservation and GroupedRandomWalkObservation
    # # rng = np.random.default_rng(params.get('seed', 123))
    # # low_rw = math.sqrt(n) / math.log(n)
    # # high_rw = math.sqrt(n)
    # # num_walkers = int(rng.uniform(low_rw, high_rw))
    # # num_steps = int(rng.uniform(5, (n / num_walkers) * math.log(n)))

    
    # rw = random_walk_obs.RandomWalkObservation(
    #     graph=G,
    #     seed=params.get('seed', 123),
    #     num_walkers=num_walkers,
    #     num_steps=0,
    #     stopping_param=0.1,
    #     leaky=0.1,
    # )

    # rw_observe = rw.observe()
    # sparsity = rw.get_count() / (len(G.edges) * len(G.nodes))
    # observations['RandomWalkObservation'] = {'observations': rw_observe, 'sparsity': sparsity}
    # # observations['RandomWalkObservation'] = rw.observe()

    
    # grw = random_walk_obs.GroupedRandomWalkObservation(
    #     graph=G,
    #     seed=params.get('seed', 123),
    #     num_walkers=num_walkers,
    #     num_steps=0,
    #     stopping_param=0.1,
    #     leaky=0.1,
    # )
    # grw_observe = grw.observe()
    # sparsity = int(grw.count) / (len(G.edges) * len(G.nodes))
    # observations['GroupedRandomWalkObservation'] = {'observations': grw_observe, 'sparsity': sparsity}
    # # observations['GroupedRandomWalkObservation'] = grw.observe()

    def weight_func(c1, c2):
        return np.exp(-0.5 * get_coordinate_distance(c1, c2))
    
    
    pso = standard_observe.PairSamplingObservation(
        graph = G,
        num_samples= (C * len(G.nodes)**2) / 2,
        weight_func=weight_func,
        seed = params.get('seed', 123)
    )
    
    pso_observe = pso.observe()
    sparsity = pso.get_count() / (len(G.edges) * len(G.nodes))
    observations['PairSamplingObservation'] = {'observations': pso_observe, 'sparsity': sparsity}
    # observations['PairSamplingObservation'] = pso.observe() 
    
    # vbs = standard_observe.VertexBasedSamplingObservation(
    #     graph = G,
    #     weight_func=weight_func,
    #     seed = params.get('seed', 123)
    # )

    # vbs_observe = vbs.observe()
    # sparsity = vbs.get_count() / (len(G.edges) * len(G.nodes))
    # observations['VertexBasedSamplingObservation'] = {'observations': vbs_observe, 'sparsity': sparsity}
    # # observations['VertexBasedSamplingObservation'] = vbs.observe()


    # Combine and save, matching input structure
    out_data = {
        'parameters': params,
        'graph': graph_data,
        'observations': observations,
    }



    output_path = os.path.join(OUTPUT_DIR, filename)
    with open(output_path, 'w') as f:
        json.dump(out_data, f, default=_json_convert)
    print(f"Saved observations for {filename} to {output_path}")
    

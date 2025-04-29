'''

We have a directory called gbm_graphs_directory that holds "graphs_xxx.json" where xxx is 001-500. 
Each JSON contains a Python dictionary with two keys, "params" and "graph". params has various parameters of the GBM 
and "graph" is a JSON-version of a networkx graph. 

Load the networkx graph and run the following functionalities: 

We would like to run: 
    - Observation
    - GroupedMultiSensorObservation
    - SingleSensorObservation
    - RandomWalkObservation 
    - GroupedRandomWalkObservation

to be generated for each graph. The observations for each class can be found by class_instance.observe(). 

SingleSensorObservation is initialized as follows: 
    - SingleSensorObservation(G, seed=42, sensor=0, radii=np.linspace(0.1,1.0,10))
    - take G from loading the graph, assume sensor = 0, and keep radii = np.linspace(0.1, 1.0, 10) 

    MultiSensorObservation is initialized as follows: 
        - MultiSensorObservation(G, seed=42, num_sensors=3, min_step = 0.15, mradii=np.linspace(0.1,1.0,10))
            - take G from loading the graph, num_sensors = k, prob_fn will be specified, deduplicate_edges = True, 
            seed is provided in the params part of the JSON
            - For k=2, min_step = 0.3
            - For k=3, min_step = 0.2
            - For k=4, min_step = 0.15

    - GroupMultiSensorObservation is initialized just as MultiSensorObservation 

RandomWalkObservation is initialized as follows: 
    - RandomWalkObservation(
        graph=G,
        seed=123,
        num_walkers=10, # int(Unif(sqrt(n)/log n, sqrt(n)))
        num_steps=5, # int(Unif(5, n/ num_walkers*log(n))) 
        stopping_param=0.1, # CONSTANT 
        leaky=0.1, # CONSTANT
    )

    - take G from loading the graph, seed from params, num_walkers = int(Unif(sqrt(n) / log n, sqrt(n)))
    num_steps = int(Unif(5, n / num_walkers * log(n)))
    stopping_param = 0.1 # Constant
    leaky = 0.1 # Constant

GroupRandomWalkObservation follows from the initialization of RandomWalkObservation 


The way to run each one is to initialize the observation class and run .observe(). It will return the output of the observation. 

In the dictionary store it as the name of the observation. Generate a JSON containing the original JSON info passed in + the observations.
Save as JSON. 

'''


#!/usr/bin/env python3

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
INPUT_DIR = 'gbm_graphs_data'
OUTPUT_DIR = 'gbm_observation'
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

    
    # SingleSensorObservation
    single = sensor_observe.SingleSensorObservation(
        G,
        seed=42,
        sensor=0,
        radii=np.linspace(0.1, 1.0, 10)
    )
    observations['SingleSensorObservation'] = single.observe()

    # MultiSensorObservation and GroupedMultiSensorObservation
    k = params.get('K', 2)
    min_sep_map = {2: 0.3, 3: 0.2, 4: 0.15}
    min_sep = min_sep_map[k]

    
    multi = sensor_observe.MultiSensorObservation(
        G,
        seed=params.get('seed', 42),
        num_sensors=k,
        min_sep=min_sep,
        radii=np.linspace(0.1, 1.0, 10),
        deduplicate_edges=True,
    )
    observations[f'MultiSensorObservation_k{k}'] = multi.observe()

    
    grouped = sensor_observe.GroupedMultiSensorObservation(
        G,
        seed=params.get('seed', 42),
        num_sensors=k,
        min_sep=min_sep,
        radii=np.linspace(0.1, 1.0, 10),
        deduplicate_edges=True,
    )
    observations[f'GroupedMultiSensorObservation_k{k}'] = grouped.observe()

    # RandomWalkObservation and GroupedRandomWalkObservation
    rng = np.random.default_rng(params.get('seed', 123))
    low_rw = math.sqrt(n) / math.log(n)
    high_rw = math.sqrt(n)
    num_walkers = int(rng.uniform(low_rw, high_rw))
    num_steps = int(rng.uniform(5, (n / num_walkers) * math.log(n)))

    
    rw = random_walk_obs.RandomWalkObservation(
        graph=G,
        seed=params.get('seed', 123),
        num_walkers=num_walkers,
        num_steps=num_steps,
        stopping_param=0.1,
        leaky=0.1,
    )
    observations['RandomWalkObservation'] = rw.observe()

    
    grw = random_walk_obs.GroupedRandomWalkObservation(
        graph=G,
        seed=params.get('seed', 123),
        num_walkers=num_walkers,
        num_steps=num_steps,
        stopping_param=0.1,
        leaky=0.1,
    )
    observations['GroupedRandomWalkObservation'] = grw.observe()

    def weight_func(c1, c2):
        return np.exp(-0.5 * get_coordinate_distance(c1, c2))
    
    
    pso = standard_observe.PairSamplingObservation(
        graph = G,
        num_samples= 10,
        weight_func=weight_func,
        seed = params.get('seed', 123)
    )
  

    observations['PairSamplingObservation'] = pso.observe() 
    
    vbs = standard_observe.VertexBasedSamplingObservation(
        graph = G,
        weight_func=weight_func,
        seed = params.get('seed', 123)
    )

    observations['VertexBasedSamplingObservation'] = vbs.observe()


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
    

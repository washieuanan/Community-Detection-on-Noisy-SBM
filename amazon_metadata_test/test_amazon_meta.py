from experiments.community_detection.bp.vectorized_geometric_bp import (
    get_true_communities,
    detection_stats
)
# from experiments.observations.standard_observe import PairSamplingObservation, get_coordinate_distance

import numpy as np
import networkx as nx

from experiments.community_detection.bp.duo_bp import (
    create_dist_observed_subgraph,
    duo_bp
)
# from experiments.community_detection.bp.bethe_duo_bp import (
#     duo_bp
# )

import os
import json
import logging
import random

def coords_str2arr(G: nx.Graph, dim = 16):
    """
    for each coord, convert string formatted coord to numpy array
    """
    for n in G.nodes():
        coord_str = G.nodes[n]["coords"]
        coord_arr = np.fromstring(coord_str, sep=",")
        if len(coord_arr) != dim:
            coord_arr = np.zeros(dim)
        G.nodes[n]["coords"] = coord_arr
    return G

if __name__ == "__main__":
    # import google.cloud.logging
    # from google.cloud.logging.handlers import CloudLoggingHandler
    # import logging

    # client = google.cloud.logging.Client()

    # handler = CloudLoggingHandler(client)

    # logging.getLogger().setLevel(logging.INFO)
    # logging.getLogger().addHandler(handler)
    
    G = nx.read_gml("amazon_metadata_test/amazon_hamming_videoDVD.gml")
    G = coords_str2arr(G)
    # Subsample G to 25% of its nodes
    num_nodes = int(0.25 * G.number_of_nodes())
    random.seed(42)
    selected_nodes = random.sample(list(G.nodes()), num_nodes)
    G = G.subgraph(selected_nodes).copy()
    # all_nodes_ok = True
    # for n in G.nodes():
    #     if "comm" not in G.nodes[n]:
    #         print(f"Node {n} is missing the 'comm' attribute.")
    #         all_nodes_ok = False
    #     if "coords" not in G.nodes[n]:
    #         print(f"Node {n} is missing the 'coord' attribute.")
    #         all_nodes_ok = False
    # if all_nodes_ok:
        # print("All nodes have both 'comm' and 'coord' attributes.")
    print(f"Testing on classes: {G.graph['subclasses']} and {len(G.nodes())} nodes")
    print(f"Created Graph with {len(G.nodes())} nodes and {len(G.edges())} edges")
    
    results_folder = "amazon_metadata_test/"
    # add distances to G - use more sophisticated distance scaling approach
    max_dist = max(float(G[u][v]["dist"]) for u, v in G.edges())
    for u, v in G.edges():
        dist = float(G[u][v]["dist"]) / max_dist  # Normalize distances
        # Use modified sigmoid with better scaling
        scaled_dist = 3.0 / (1.0 + np.exp(-5.0 * dist))
        G.edges[u, v]["dist"] = scaled_dist
        G.edges[v, u]["dist"] = scaled_dist
    print("Added distances to edges")
    
    res = duo_bp(
        G,
        K=2,
        num_balls=int(min(512, G.number_of_nodes() // 20)),  # More balls
        damp_high=0.85,           # Slightly lower initial damping
        damp_low=0.35,           # Lower final damping
        balance_regularization=0.35,  # Stronger regularization
        w_min=5e-3,              # Lower minimum weight
        max_em_iters=60,         # More iterations
        warmup_rounds=4,         # More warmup rounds
        anneal_steps=15,         # More annealing steps
        shrink_comm=0.7,         # Stronger community shrinkage
        shrink_geo=0.6,          # Stronger geometric shrinkage
        comm_cut=0.93,           # Higher threshold
        geo_cut=0.92,           # Higher threshold
        patience=15,             # More patience
        bp_kwargs={
            "tol": 5e-5,         # Tighter tolerance
            "max_iter": 2500,    # More iterations
            "min_steps": 200,    # More minimum steps
            "eps": 0.25,         # Stronger initialization bias
        }
        )
    
    preds = res["communities"]
    print(f"Finished bethe_duo_bp with {len(preds)} predictions")
    true_communities = get_true_communities(G, attr="comm")
    stats = detection_stats(preds, true_communities)
    # logging.info(f"Finished detection stats")
    print(f"Finished detection stats") 
    
    # Check if beliefs exists in res, and add default if not
    if "beliefs" not in res:
        print("Warning: 'beliefs' not found in BP results, adding default")
        res["beliefs"] = np.ones((len(G.nodes()), 2)) / 2
    
    # Add function to convert numpy arrays to Python lists
    def convert_numpy_types(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    results_dict = {
        "stats": stats,
        "preds": res.get("communities", []),
        "beliefs": res.get("beliefs", [])
    }
    
    # Convert numpy types to Python native types
    results_dict = convert_numpy_types(results_dict)
    
    # logging.info(f"Saving results to {results_folder}")
    with open(os.path.join(results_folder, f"amazon_results.json"), "w") as f:
        json.dump(results_dict, f)
        
    print(f"Saved results to {results_folder}")
        
        
    # def weight_func(c1, c2): 
    #     return 1.0
    
    # observations = {}
    # for sparsity in [0.01, 0.025, 0.05]: 
    #     num_pairs = calc_num_pairs(G, scale_factor = sparsity)
    #     sampler = PairSamplingObservation(G, num_samples = num_pairs, weight_func = weight_func, seed = 42) 
    #     observations[str(sparsity)] = sampler.observe()



    # json.dump(observations, open("amazon_metadata_test/observations_025_05_01.json", "w"))

    # all_nodes_ok = True
    # for n in G.nodes():
    #     if "comm" not in G.nodes[n]:
    #         print(f"Node {n} is missing the 'comm' attribute.")
    #         all_nodes_ok = False
    #     if "coords" not in G.nodes[n]:
    #         print(f"Node {n} is missing the 'coord' attribute.")
    #         all_nodes_ok = False
    # if all_nodes_ok:
    #     print("All nodes have both 'comm' and 'coord' attributes.")
    # print(f"Testing on classes: {G.graph['subclasses']} and {len(G.nodes())} nodes")
    # print(f"Created Graph with {len(G.nodes())} nodes and {len(G.edges())} edges")
    # num_pairs = calc_num_pairs(G, scale_factor=0.1)
    # def weight_func(c1, c2):
    #     return np.exp(-0.5 * get_coordinate_distance(c1, c2))

    # sampler = PairSamplingObservation(G, num_samples=num_pairs, weight_func=weight_func, seed=42)
    # observations = sampler.observe()
    # json.dump(observations, open("amazon_metadata_test/observations_01.json", "w"))
    # observation_data = json.load(open("amazon_metadata_test/observations_025_05_01.json"))
    # results_folder = "amazon_metadata_test/duo_results/"
    # for sparsity in [0.01, 0.025, 0.05]:
    #     observations = observation_data[str(sparsity)]
    #     print(f"Found {len(observations)} observations for sparsity {sparsity}")
    #     logging.info(f"Found {len(observations)} observations for sparsity {sparsity}")
    #     obs_nodes: set[int] = set()
    #     for p, _ in observations:
    #         obs_nodes.add(p[0])
    #         obs_nodes.add(p[1])
            
    #     subG_duo = create_dist_observed_subgraph(G.number_of_nodes(), observations)
    #     subG_bethe = create_dist_observed_subgraph(G.number_of_nodes(), observations)
    #     print("Created subgraph for sparsity", sparsity)
    #     logging.info(f"Created subgraph for sparsity {sparsity}")
    #     res_duo = duo_bp(subG_duo,
    #                  K=2,
    #                  num_balls=32)
        
    #     print("Finished duo_bp for sparsity", sparsity)
    #     logging.info(f"Finished duo_bp for sparsity {sparsity}")
    #     res_bethe = bethe_duo_bp(subG_bethe,
    #                  K=2,
    #                  num_balls=32)
        
    #     print("Finished bethe_duo_bp for sparsity", sparsity)
    #     logging.info(f"Finished bethe_duo_bp for sparsity {sparsity}")
    #     preds_duo = res["communities"]
    #     preds_bethe = res_bethe["communities"]
        
    #     true_communities = get_true_communities(G, attr="comm")
        
    #     stats_duo = detection_stats(preds_duo, true_communities)
    #     stats_bethe = detection_stats(preds_bethe, true_communities)
        
    #     logging.info(f"Finished detection stats for sparsity {sparsity}")
        
    #     results_dict = {
    #         "duo_bp": {
    #             "stats": stats_duo,
    #             "preds": preds_duo
    #         },
    #         "bethe_duo_bp": {
    #             "stats": stats_bethe,
    #             "preds": preds_bethe
    #         },
    #         "obs_nodes": obs_nodes
    #     }
    #     with open(os.path.join(results_folder, f"results_{sparsity}.json"), "w") as f:
    #         json.dump(results_dict, f)
        
    #     print(f"Saved results for sparsity {sparsity} to {results_folder}")
    #     logging.info(f"Saved results for sparsity {sparsity} to {results_folder}")
        
            
        
            
        


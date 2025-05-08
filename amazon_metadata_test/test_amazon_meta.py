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
from experiments.community_detection.bp.bethe_duo_bp import (
    duo_bp
)

import os
import json
import logging

def coords_str2arr(G: nx.Graph, dim = 16):
    """
    for each coord, convert string formatted coord to numpy array
    """
    for n in G.nodes():
        coord_str = G.nodes[n]["coords"]
        coord_arr = np.fromstring(coord_str[1:-1], sep=",")
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
    
    G = nx.read_gml("amazon_metadata_test/amazon_graph_videoDVD.gml")
    G = coords_str2arr(G)

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
    # add distances to G
    for u, v in G.edges():
        c1 = G.nodes[u]["coords"]
        c2 = G.nodes[v]["coords"]
        dist = np.linalg.norm(c1 - c2)
        G.edges[u, v]["dist"] = dist
        G.edges[v, u]["dist"] = dist
    print("Added distances to edges")
    res = duo_bp(
        G,
        K=2,
        num_balls=32,
        damp_high=0.7)
    
    preds = res["communities"]
    print(f"Finished bethe_duo_bp with {len(preds)} predictions")
    true_communities = get_true_communities(G, attr="comm")
    stats = detection_stats(preds, true_communities)
    # logging.info(f"Finished detection stats")
    print(f"Finished detection stats") 
    results_dict = {
        "stats": stats,
        "preds": preds
    }
    
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
        
            
        
            
        


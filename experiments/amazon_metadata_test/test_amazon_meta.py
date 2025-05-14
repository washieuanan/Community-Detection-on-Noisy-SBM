from community_detection.bp.vectorized_geometric_bp import (
    belief_propagation,
    detection_stats,
    get_true_communities,
)
# from experiments.observations.standard_observe import PairSamplingObservation, get_coordinate_distance

import numpy as np
import networkx as nx

from community_detection.bp.duo_bp import (
    duo_bp,
    create_dist_observed_subgraph,
)
# from experiments.community_detection.bp.bethe_duo_bp import (
#     duo_bp
# )
from algorithms.duo_spec import duo_spec, duo_bprop
import os
import json
import logging
import random
from experiments.controls.motif_count import motif_counting
from experiments.controls.spectral import spectral_clustering_community_detection
from community_detection.bp.vectorized_bp import belief_propagation, belief_propagation_weighted
def coords_str2arr(G: nx.Graph, dim = 16):
    """
    for each coord, convert string formatted coord to numpy array
    """
    new_G = nx.Graph()
    for n in G.nodes():
        coord_str = G.nodes[n]["coords"]
        coord_arr = np.fromstring(coord_str, sep=",")
        if len(coord_arr) != dim:
            coord_arr = np.zeros(dim)
        new_G.add_node(int(n), coords=coord_arr, asin=G.nodes[n]["asin"], comm=G.nodes[n]["comm"])
        
    for u, v in G.edges():
        if "dist" in G.edges[u, v]:
            dist = G.edges[u, v]["dist"]
            if isinstance(dist, str):
                dist = float(dist)
            new_G.add_edge(int(u), int(v), dist=dist)
    new_G.graph = G.graph.copy()
    new_G = nx.relabel.convert_node_labels_to_integers(new_G, first_label=0)
    return new_G

def connect_components(G: nx.Graph, weight: float = 1e-3) -> None:
    """
    Add edges between components so that G becomes connected.
    Picks one representative node per component, then links them in a chain.
    
    Parameters
    ----------
    G : nx.Graph
        The graph to modify in-place.
    weight : float
        The weight to assign to each new bridging edge (default 0.001).
    """
    # find each connected component
    comps = list(nx.connected_components(G))
    if len(comps) <= 1:
        return  # already connected

    # pick one node from each component
    reps = [next(iter(c)) for c in comps]

    # link them in a simple chain: reps[0]—reps[1], reps[1]—reps[2], …
    for u, v in zip(reps[:-1], reps[1:]):
        if not G.has_edge(u, v):
            G.add_edge(u, v, weight=weight)

if __name__ == "__main__":
    # import google.cloud.logging
    # from google.cloud.logging.handlers import CloudLoggingHandler
    # import logging

    # client = google.cloud.logging.Client()

    # handler = CloudLoggingHandler(client)

    # logging.getLogger().setLevel(logging.INFO)
    # logging.getLogger().addHandler(handler)
    
    G = nx.read_gml("amazon_metadata_test/amazon_hamming_bookDVD.gml")
    G = coords_str2arr(G)
    comm_counts = {}
    for _, data in G.nodes(data=True):
        comm = data.get("comm")
        if comm is not None:
            comm_counts[comm] = comm_counts.get(comm, 0) + 1

    print("Count of nodes for each 'comm' attribute:")
    for comm, count in comm_counts.items():
        print(f"{comm}: {count}")
    # Subsample G to 25% of its nodes
    num_nodes = int(0.2 * G.number_of_nodes())
    random.seed(123)
    selected_nodes = random.sample(list(G.nodes()), num_nodes)
    G = G.subgraph(selected_nodes).copy()
    G = nx.convert_node_labels_to_integers(G)
    for u, v, edge_data in G.edges(data=True):
        if "dist" in edge_data:
            try:
                edge_data["dist"] = float(edge_data["dist"]) * 2
            except ValueError:
                pass
    dists = []
    for _, _, data in G.edges(data=True):
        if "dist" in data:
            try:
                dists.append(float(data["dist"]))
            except ValueError:
                pass

    if dists:
        avg_dist = sum(dists) / len(dists)
        print("Average 'dist' attribute:", avg_dist)
    else:
        print("No 'dist' attribute found on any edge.")
    def make_spanning_tree(nodes):
        """
        Given an iterable of nodes, returns a list of edges forming
        a spanning tree by shuffling and chaining them.
        """
        nodes = list(nodes)
        random.shuffle(nodes)
        # connect consecutive nodes in the shuffled list
        return [(nodes[i], nodes[i+1]) for i in range(len(nodes)-1)]

    # … after you've built G_bp …

    print(f"Testing on classes: {G.graph['subclasses']} and {len(G.nodes())} nodes")
    print(f"Created Graph with {len(G.nodes())} nodes and {len(G.edges())} edges")
    
    results_folder = "amazon_metadata_test/"

    print("Added distances to edges")
    
    # duo_params = dict(
    #     # spectral-EM settings
    #     K               = 2,                    # number of communities
    #     num_balls       = 32,                   # finer geometry embedding
    #     config          = ("bethe_hessian",     # community estimator
    #                     "bethe_hessian"),        # geometry estimator

    #     # — EM schedule
    #     max_em_iters    = 100,                  # allow more EM steps
    #     warmup_rounds   = 2,                   # hold off on any re-weighting
    #     anneal_steps    = 20,                   # then ramp λ from 0→full over 30 iter

    #     # — convergence
    #     tol             = 1e-5,
    #     patience        = 10,
    #     random_state    = 42,
    # )
    
    # res = duo_spec(G, **duo_params)

    # preds = res['communities']
    # G_fin = res['G_final']
    # # preds = res['communities']
    # weights = [data.get("weight") for _, _, data in G_fin.edges(data=True) if "weight" in data]

    # if weights:
    #     weights = np.array(weights)
    #     print("Weight attribute statistics:")
    #     print(f"Count: {len(weights)}")
    #     print(f"Minimum: {np.min(weights)}")
    #     print(f"Maximum: {np.max(weights)}")
    #     print(f"Mean: {np.mean(weights)}")
    #     print(f"Median: {np.median(weights)}")
    #     print(f"Standard Deviation: {np.std(weights)}")
    # else:
    #     print("No 'weight' attribute found on any edge.")

    # connect_components(G_fin)
    # _, preds, _, _ = belief_propagation_weighted(
    #     G_fin, 
    #     q=2, 
    #     max_iter = 1000,
    # )
    
    
    

    # res = duo_bp(
    #     G, 
    #     K=2, 
    #     num_balls=128,
    # )
    # res = duo_bprop(
    #     G, 
    #     K=2, 
    #     max_iter=1000)
    # ε = 1e-3
    # for i, j in make_spanning_tree(G.nodes()):
    #     if not G.has_edge(i, j):
    #         G.add_edge(i, j, weight=ε, dist=2)
    # connect_components(G)
    _, preds, _, _ = belief_propagation(
        G, 
        q=2, 
        max_iter = 1000,
    )
    
    # preds = res["communities"]
    
    # preds = spectral_clustering_community_detection(
    #     G, K=4)
    # preds = motif_counting(
    #     G, q=2)
    print(f"Finished bethe_duo_bp with {len(preds)} predictions")
    true_communities = get_true_communities(G, attr="comm")
    stats = detection_stats(preds, true_communities)
    print(stats)
    # logging.info(f"Finished detection stats")
    print(f"Finished detection stats") 
    
    # Check if beliefs exists in res, and add default if not
    # if "beliefs" not in res:
    #     print("Warning: 'beliefs' not found in BP results, adding default")
    #     res["beliefs"] = np.ones((len(G.nodes()), 2)) / 2
    
    # # Add function to convert numpy arrays to Python lists
    # def convert_numpy_types(obj):
    #     if isinstance(obj, np.ndarray):
    #         return obj.tolist()
    #     elif isinstance(obj, np.integer):
    #         return int(obj)
    #     elif isinstance(obj, np.floating):
    #         return float(obj)
    #     elif isinstance(obj, dict):
    #         return {k: convert_numpy_types(v) for k, v in obj.items()}
    #     elif isinstance(obj, list):
    #         return [convert_numpy_types(item) for item in obj]
    #     return obj
    
    # results_dict = {
    #     "stats": stats,
    #     "preds": res.get("communities", []),
    #     "beliefs": res.get("beliefs", [])
    # }
    
    # print(stats)
    
    # Convert numpy types to Python native types
    # results_dict = convert_numpy_types(results_dict)
    
    # # logging.info(f"Saving results to {results_folder}")
    # with open(os.path.join(results_folder, f"amazon_results.json"), "w") as f:
    #     json.dump(results_dict, f)
        
    # print(f"Saved results to {results_folder}")
        
        
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
        
            
        
            
        

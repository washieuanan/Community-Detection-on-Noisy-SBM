import networkx as nx  
import numpy as np

def calc_sparsity(G: nx.Graph) -> float:
    """calculates sparsity of graph"""
    
    avg_deg = np.mean([G.degree[n] for n in G.nodes()])
    original_density = avg_deg / len(G.nodes)
    return original_density

def calc_num_pairs(G: nx.Graph, scale_factor: float = 0.1):
    """num pairs = Cn^2/2"""
    
    C = scale_factor * calc_sparsity(G)
    num_pairs = int(C * len(G.nodes) ** 2 / 2)
    return num_pairs

def calc_num_sensors(G: nx.Graph, scale_factor: float = 0.1, r: float = 0.25):
    """num sensors = Cn/(r^3d)"""
    avg_deg = np.mean([G.degree[n] for n in G.nodes()])
    C = scale_factor * calc_sparsity(G)
    num_sensors = int(C * len(G.nodes) / (r ** 3 * avg_deg))
    return num_sensors

def calc_num_walkers(G: nx.Graph, scale_factor: float = 0.1, p: float = 0.1):
    """num walkers = Cpn^2/2"""
    
    C = scale_factor * calc_sparsity(G)
    num_walkers = int(C * p * len(G.nodes) ** 2 / 2)
    return num_walkers
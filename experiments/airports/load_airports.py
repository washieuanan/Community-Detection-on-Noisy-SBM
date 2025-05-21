import torch
import networkx as nx
from torch_geometric.datasets import Airports
import numpy as np
from torch_geometric.transforms import ToUndirected

def load_airports(name = 'USA'):
    """
    Load the PPI dataset and convert it to a NetworkX graph.
    """
    # Load the dataset
    dataset = Airports(root='data/Airports', name=name)
    num_graphs = len(dataset)
    data = dataset[0]
    # Convert to undirected graph
    
    # Create NetworkX graph
    edge_index = data.edge_index.numpy()
    G = nx.Graph()
    
    # Add nodes with community labels and features
    for i in range(data.num_nodes):
        G.add_node(i, 
                   comm=int(data.y[i].item()),
                   coords=data.x[i].numpy())
        
    # Add edges
    # Calculate distances and add edges with distance attributes
    edges = list(zip(edge_index[0], edge_index[1]))
    dists = []
    for u, v in edges:
        # Get coordinates for nodes
        coord_u = G.nodes[u]['coords'] 
        coord_v = G.nodes[v]['coords']
        # Calculate hamming distance
        dist = np.sum(coord_u != coord_v)
        dists.append(dist)
    
    # Normalize distances to max 2
    dists = np.array(dists)
    if dists.max() > 0:
        dists = 2 * dists / dists.max()
    
    # Add edges with normalized distances
    G.add_edges_from([(u,v,{'dist':d}) for (u,v),d in zip(edges,dists)])
    return G

if __name__ == "__main__":
    G = load_airports()
    print(G.number_of_nodes())
    comms = set(nx.get_node_attributes(G, 'comm').values())
    print(f"Number of unique communities: {len(comms)}")
    # print(G.nodes(data=True))
    # print(G.edges(data=True))
    
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import TSNE

# Constants
NUM_VERTICES_CLUSTER_1 = 100
NUM_VERTICES_CLUSTER_2 = 200
TOTAL_VERTICES = NUM_VERTICES_CLUSTER_1 + NUM_VERTICES_CLUSTER_2
DIMENSION = 3  
CONNECTIVITY_THRESHOLD = 0.3  
MAX_ATTEMPTS = 10 

# def generate_coordinates(num_vertices, dimension, distribution='uniform', **kwargs):
#     """
#     Generate coordinates in R^n space based on specified distribution.
    
#     params: 
#     num_vertices : int
#         Number of vertices to generate
#     dimension : int
#         Dimension of the space
#     distribution : str
#         Distribution to use ('uniform', 'normal', 'exponential', etc.)
#     kwargs : dict
#         Additional parameters for the distribution
        
    
#     ret: 
#     np.ndarray
#         Array of shape (num_vertices, dimension) with generated coordinates
#     """
#     if distribution == 'uniform':
#         low = kwargs.get('low', 0)
#         high = kwargs.get('high', 1)
#         return np.random.uniform(low, high, size=(num_vertices, dimension))
    
#     elif distribution == 'normal':
#         loc = kwargs.get('loc', 0)
#         scale = kwargs.get('scale', 1)
#         return np.random.normal(loc, scale, size=(num_vertices, dimension))
    
#     elif distribution == 'exponential':
#         scale = kwargs.get('scale', 1)
#         return np.random.exponential(scale, size=(num_vertices, dimension))
    
#     else:
#         raise ValueError(f"Distribution '{distribution}' not supported")

def generate_coordinates(num_vertices, dimension, distribution='uniform', constrain_to_unit_sphere=False, **kwargs):
    """

    lowkey don't know if this is the most efficient way to do this - can discuss 

    Generate coordinates in R^n space based on specified distribution.
    
    params: 
    num_vertices : int
        Number of vertices to generate
    dimension : int
        Dimension of the space
    distribution : str
        Distribution to use ('uniform', 'normal', 'exponential', etc.)
    constrain_to_unit_sphere : bool
        If True, only points within the unit sphere (norm <= 1) are returned.
    kwargs : dict
        Additional parameters for the distribution
        
    ret: 
    np.ndarray
        Array of shape (num_vertices, dimension) with generated coordinates
    """
    if not constrain_to_unit_sphere:
        
        if distribution == 'uniform':
            low = kwargs.get('low', 0)
            high = kwargs.get('high', 1)
            return np.random.uniform(low, high, size=(num_vertices, dimension))
    
        elif distribution == 'normal':
            loc = kwargs.get('loc', 0)
            scale = kwargs.get('scale', 1)
            return np.random.normal(loc, scale, size=(num_vertices, dimension))
    
        elif distribution == 'exponential':
            scale = kwargs.get('scale', 1)
            return np.random.exponential(scale, size=(num_vertices, dimension))
    
        else:
            raise ValueError(f"Distribution '{distribution}' not supported")
    
    else:
        # Rejection sampling: generate candidates until we have enough that lie within the unit sphere.
        accepted_points = []
        batch_size = max(100, num_vertices)  
        
        while len(accepted_points) < num_vertices:
            
            if distribution == 'uniform':
                low = kwargs.get('low', 0)
                high = kwargs.get('high', 1)
                candidates = np.random.uniform(low, high, size=(batch_size, dimension))
            elif distribution == 'normal':
                loc = kwargs.get('loc', 0)
                scale = kwargs.get('scale', 1)
                candidates = np.random.normal(loc, scale, size=(batch_size, dimension))
            elif distribution == 'exponential':
                scale = kwargs.get('scale', 1)
                candidates = np.random.exponential(scale, size=(batch_size, dimension))
            else:
                raise ValueError(f"Distribution '{distribution}' not supported for constrained generation")
            
            # Filter candidates to keep only those with norm <= 1:
            norms = np.linalg.norm(candidates, axis=1)
            accepted = candidates[norms <= 1]
            accepted_points.append(accepted)
            
        accepted_points = np.vstack(accepted_points)
        return accepted_points[:num_vertices]


def distance_function(point1, point2, metric='euclidean', alpha=1.0):
    """
    Calculate distance between two points with configurable metrics.
    
    params: 
    point1, point2 : np.ndarray
        Coordinates of the points
    metric : str
        Distance metric ('euclidean', 'manhattan', 'gaussian', etc.)
    alpha : float
        Parameter for certain distance functions
        
    ret: 
    float
        Distance between the points
    """
    if metric == 'euclidean':
        return np.sqrt(np.sum((point1 - point2) ** 2))
    
    elif metric == 'manhattan':
        return np.sum(np.abs(point1 - point2))
    
    elif metric == 'gaussian':
        # Gaussian similarity transformed to a distance
        euclidean_dist = np.sqrt(np.sum((point1 - point2) ** 2))
        return 1 - np.exp(-(euclidean_dist ** 2) / (2 * alpha ** 2))
    
    else:
        raise ValueError(f"Metric '{metric}' not supported")

def generate_latent_geometry_graph(
    cluster_sizes, 
    dimension=DIMENSION, 
    connectivity_threshold=CONNECTIVITY_THRESHOLD,
    max_attempts=MAX_ATTEMPTS,
    distance_metric='euclidean',
    distance_alpha=1.0,
    distributions=None,
    dist_params=None
):
    """
    Generate a graph with latent geometry in R^n space.
    
    params: 
    cluster_sizes : list
        List with the number of vertices in each cluster
    dimension : int
        Dimension of the space
    connectivity_threshold : float
        Threshold for edge creation
    max_attempts : int
        Maximum attempts to get a connected graph
    distance_metric : str
        Metric to use for distance calculation
    distance_alpha : float
        Parameter for certain distance functions
    distributions : list
        List with the distribution to use for each cluster
    dist_params : list
        List with parameters for each distribution
        
    ret: 
    nx.Graph
        Generated graph
    np.ndarray
        Array with coordinates of each vertex
    """
    if distributions is None:
        distributions = ['uniform'] * len(cluster_sizes)
    
    if dist_params is None:
        dist_params = [{}] * len(cluster_sizes)
    
    
    assert len(distributions) == len(cluster_sizes)
    assert len(dist_params) == len(cluster_sizes)
    
    total_vertices = sum(cluster_sizes)
    
    for attempt in range(max_attempts):
        
        coordinates_list = []
        vertex_cluster_map = []  # To track which vertex belongs to which cluster
        
        for i, (cluster_size, distribution, params) in enumerate(zip(cluster_sizes, distributions, dist_params)):
            # For the second cluster, we can add an offset to separate it from the first
            if i > 0 and 'loc' not in params and distribution == 'normal':
                params['loc'] = i * 2  # Simple offset for visualization
            
            cluster_coords = generate_coordinates(cluster_size, dimension, distribution, **params)
            coordinates_list.append(cluster_coords)
            vertex_cluster_map.extend([i] * cluster_size)
        
        # Combine all coordinates
        coordinates = np.vstack(coordinates_list)
        
        # Calculate pairwise distances
        dist_matrix = np.zeros((total_vertices, total_vertices))
        for i in range(total_vertices):
            for j in range(i+1, total_vertices):
                dist = distance_function(
                    coordinates[i], coordinates[j], 
                    metric=distance_metric, alpha=distance_alpha
                )
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        
        # Create graph based on distance threshold
        G = nx.Graph()
        G.add_nodes_from(range(total_vertices))
        
        for i in range(total_vertices):
            for j in range(i+1, total_vertices):
                if dist_matrix[i, j] <= connectivity_threshold:
                    G.add_edge(i, j)
        
        # Check if the graph is connected
        if nx.is_connected(G):
            # Add vertex attributes
            for i in range(total_vertices):
                G.nodes[i]['coords'] = coordinates[i]
                G.nodes[i]['cluster'] = vertex_cluster_map[i]
            
            return G, coordinates, vertex_cluster_map
        
        print(f"Attempt {attempt+1}: Graph not connected, retrying...")
    
    raise RuntimeError(f"Failed to generate a connected graph after {max_attempts} attempts")

def visualize_graph(G, coordinates, vertex_cluster_map):
    """
    Visualize the generated graph using dimensionality reduction for n-dimensional data.
    
    params: 
    G : nx.Graph
        Generated graph
    coordinates : np.ndarray
        Array with coordinates of each vertex (can be n-dimensional)
    vertex_cluster_map : list
        List indicating which cluster each vertex belongs to
    """
    plt.figure(figsize=(10, 8))
    
   
    colors = ['r', 'b', 'g', 'y', 'c', 'm']
    
    
    if coordinates.shape[1] == 2:
        reduced_coords = coordinates
    else:
        
        tsne = TSNE(n_components=2, random_state=42)
        reduced_coords = tsne.fit_transform(coordinates)
    
   
    for i, (pos, cluster) in enumerate(zip(reduced_coords, vertex_cluster_map)):
        plt.scatter(pos[0], pos[1], c=colors[cluster], s=100, alpha=0.7)
    
    
    for u, v in G.edges():
        x1, y1 = reduced_coords[u]
        x2, y2 = reduced_coords[v]
        plt.plot([x1, x2], [y1, y2], 'k-', alpha=0.3, linewidth=0.5)
    
    plt.title(f"Latent Geometry Graph (n={len(G.nodes())}, m={len(G.edges())}, dim={coordinates.shape[1]})")
    plt.grid(alpha=0.3)
    plt.axis('equal')
    plt.show()


if __name__ == "__main__":
    # Define parameters
    cluster_sizes = [NUM_VERTICES_CLUSTER_1, NUM_VERTICES_CLUSTER_2]
    
    # Generate with default uniform distribution
    G1, coords1, cluster_map1 = generate_latent_geometry_graph(cluster_sizes)
    visualize_graph(G1, coords1, cluster_map1)
    
    # Generate with different distributions for each cluster
    distributions = ['uniform', 'uniform']
    dist_params = [
        {'low': -1, 'high': 1},
        {'low': -1, 'high': 1}
    ]
    
    G2, coords2, cluster_map2 = generate_latent_geometry_graph(
        cluster_sizes, 
        distributions=distributions,
        dist_params=dist_params,
        connectivity_threshold=0.8  # Higher threshold to create more edges
    )
    visualize_graph(G2, coords2, cluster_map2)



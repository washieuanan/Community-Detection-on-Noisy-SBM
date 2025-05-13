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
        points = []
        batch_size = max(100, num_vertices)

        while len(points) < num_vertices:
            # 1) sample a batch
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
                raise ValueError(f"Distribution '{distribution}' not supported")

            norms = np.linalg.norm(candidates, axis=1)
            accepted = candidates[norms <= 1]
            if accepted.size:
                points.extend(accepted.tolist())

        
        points = np.array(points)[:num_vertices]
        return points
            
    


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
    dimension: int = 3,
    connectivity_threshold: float | None = None,
    max_attempts: int = 10,
    distance_metric: str = "euclidean",
    distance_alpha: float = 1.0,
    edge_prob_fn=None,
    distributions=None,
    dist_params=None,
):
    """
    Generate a (connected) latent‑geometry graph in :math:`\mathbb R^{d}`.

    Parameters
    ----------
    cluster_sizes : list[int]
        Number of vertices in each cluster.
    dimension : int, default=3
        Dimensionality of the latent space.
    connectivity_threshold : float | None, default=None
        **Deterministic rule.** Create an edge whenever the pairwise distance
        is *at most* this value. Ignored if ``edge_prob_fn`` is provided.
    max_attempts : int, default=10
        Retry limit to obtain a *connected* graph.
    distance_metric : {"euclidean", "manhattan", "gaussian"}, default="euclidean"
        Distance (or similarity‑to‑distance) metric.
    distance_alpha : float, default=1.0
        Scale parameter used by the "gaussian" metric in ``distance_function``.
    edge_prob_fn : callable | None, default=None
        **Probabilistic rule.** A function ``f(dist) -> p`` returning the edge
        probability (``0 ≤ p ≤ 1``) given a pairwise distance. If supplied,
        ``connectivity_threshold`` is ignored.
    distributions : list[str] | None
        Latent coordinate distribution per cluster ('uniform', 'normal', …).
        Defaults to all 'uniform'.
    dist_params : list[dict] | None
        Extra kwargs passed to the coordinate generator per cluster.

    Returns
    -------
    G : nx.Graph
        Undirected graph with node attributes ``coords`` and ``cluster``.
    coordinates : np.ndarray, shape (n_vertices, dimension)
        Latent coordinates corresponding to graph nodes.
    vertex_cluster_map : list[int]
        Cluster assignment for each vertex.

    """

    if connectivity_threshold is not None and edge_prob_fn is not None:
        raise ValueError(
            "cannot provide both connectivity_threshold and edge_prob_fn"
        )

    if distributions is None:
        distributions = ["uniform"] * len(cluster_sizes)
    if dist_params is None:
        dist_params = [{}] * len(cluster_sizes)

    if len(distributions) != len(cluster_sizes) or len(dist_params) != len(cluster_sizes):
        raise ValueError(
            "distributions and dist_params must match cluster_sizes in length."
        )

    total_vertices = sum(cluster_sizes)

    # Main loop: retry until connected
    for attempt in range(max_attempts):
        # 1. Generate latent coordinates cluster‑wise
        coords_list, vertex_cluster_map = [], []
        for idx, (size, dist_name, params) in enumerate(
            zip(cluster_sizes, distributions, dist_params)
        ):
            params = params.copy() # goofy ahh
            if idx > 0 and dist_name == "normal" and "loc" not in params:
                params["loc"] = idx * 2  
            cluster_coords = generate_coordinates(size, dimension, dist_name, **params)
            coords_list.append(cluster_coords)
            vertex_cluster_map.extend([idx] * size)
        coordinates = np.vstack(coords_list)

        # 2. Compute pairwise distances (upper triangle only)
        dist_matrix = np.zeros((total_vertices, total_vertices), dtype=float)
        for i in range(total_vertices):
            for j in range(i + 1, total_vertices):
                d = distance_function(
                    coordinates[i],
                    coordinates[j],
                    metric=distance_metric,
                    alpha=distance_alpha,
                )
                dist_matrix[i, j] = dist_matrix[j, i] = d

        # 3. Create edges
        G = nx.Graph()
        G.add_nodes_from(range(total_vertices))

        rng = np.random.default_rng()
        for i in range(total_vertices):
            for j in range(i + 1, total_vertices):
                d = dist_matrix[i, j]
                if edge_prob_fn is not None:
                    p = float(edge_prob_fn(d))
                    if p > 0 and rng.random() < p: # if prob is greater than 0 and random number is less than prob
                        G.add_edge(i, j)
                else:
                    if d <= connectivity_threshold:
                        G.add_edge(i, j)

        # 4. Ensure connectedness
        if nx.is_connected(G):
            for i in range(total_vertices):
                G.nodes[i]["coords"] = coordinates[i]
                G.nodes[i]["cluster"] = vertex_cluster_map[i]
            return G, coordinates, vertex_cluster_map

    raise RuntimeError(
        f"Failed to generate a connected graph after {max_attempts} attempts"
    )


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
    G1, coords1, cluster_map1 = generate_latent_geometry_graph(cluster_sizes, connectivity_threshold=0.8)
    visualize_graph(G1, coords1, cluster_map1)
    
    # Generate with different distributions for each cluster
    distributions = ['normal', 'normal']
    dist_params = [
        {'loc':-0.5, 'scale':0.3, 'constrain_to_unit_sphere':True},
        {'loc':0.5, 'scale':0.4, 'constrain_to_unit_sphere':True}
    ]
    
    G2, coords2, cluster_map2 = generate_latent_geometry_graph(
        [100, 150], 
        distributions=distributions,
        dist_params=dist_params,
        connectivity_threshold=0.8  # Higher threshold to create more edges
    )
    visualize_graph(G2, coords2, cluster_map2)

    



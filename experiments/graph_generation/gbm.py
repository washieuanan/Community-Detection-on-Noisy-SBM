import numpy as np
import networkx as nx 
import matplotlib.pyplot as plt 

def discard_disconnected_nodes(G):
    """remove all nodes in G that have zero edges"""
    disconnected_nodes = [node for node in G.nodes() if G.degree(node) == 0]
    G.remove_nodes_from(disconnected_nodes)

def generate_gbm(n: int, 
                 K: int,
                 r_in: float, 
                 r_out: float, 
                 p_in: float,
                 p_out: float, 
                 seed: int | None):

    '''
    generate GBM (cursor autocomplete limit gone so back to handwriting docstirng sadge)

    Paramaters: 

    n (int): number of nodes
    K (int): number of communities
    r_in (float): Radius threshold for intra-community edges
    r_out (float): edge probability for pairs within the same community and dist < r_in 
    p_in (float): edge probability for pairs within same community and dist < r_ in 
    p_out (float: edge probability for pairs in different communities and dist < r_0ut
    seed (int or None): Random seed for reproducability 
    '''

    rng = np.random.default_rng(seed) # not used
    pts = np.empty((n, 2)) 

    i = 0

    while i < n: 
        x,y = rng.uniform(-1, 1), rng.uniform(-1,1)
        # "uniform for some reason" - g. liu
        #
        # i think its just the easiest way to get each node's position iid
        # with same area density everywhere in the u.c. because GBM assumes
        # uniform dist over \{(x,y) : x^2 + y^2 \leq 1}
        #
        # We can do this using polar coordinates where x = r cos(theta) and y = r sin(theta) 
        # and then the jacobian should give us r dr dtheta which will be constant density as well 
        #
        # the way ima just do this is sample (x,y) ~ Uniform([-1,1]^2) and if x^2 + y^2 > 1, resample
        # o.w. keep (x,y)

        if x*x + y*y <= 1: 
            pts[i] = [x,y]
            i += 1 

    # assign communities uniformly
    comm = rng.integers(0, K, size = n) 
    G = nx.Graph() 
    for idx in range(n): 
        G.add_node(idx, coords=tuple(pts[idx]), comm=int(comm[idx])) 

    for i in range(n): 
        for j in range(i + 1, n): 
            d = np.linalg.norm(pts[i] - pts[j]) 
            if comm[i] == comm[j]: 
                if d < r_in and rng.random() < p_in: 
                    G.add_edge(i,j) 
                else: 
                    if d < r_out and rng.random() < p_out: 
                        G.add_edge(i,j) 
    # discard_disconnected_nodes(G)
    return G

def generate_gbm_soft_threshold(n: int, 
                                K: int, 
                                Q: np.array, 
                                kernel: callable, 
                                seed: int | None): 
    
    rng = np.random.default_rng(seed) 
    pts = np.empty((n,2)) 
    
    i = 0 
    while i < n: 
        x,y = rng.uniform(-1,1,2) 
        if x*x + y*y <= 1:
            pts[i] = [x,y] 
            i += 1 

    comm = rng.integers(0, K, size = n) 
    G = nx.Graph() 
    for u in range(n): 
        G.add_node(u, coords=tuple(pts[u]), comm=int(comm[u]))

    for u in range(n): 
        for v in range(u + 1, n): 
            d = np.linalg.norm(pts[u] - pts[v]) 
            p = Q[comm[u], comm[v]] * kernel(d)

            if rng.random() < np.clip(p, 0, 1): 
                G.add_edge(u,v) 
    # discard_disconnected_nodes(G)
    return G 
# CHAT
def plot_gbm(
    G,
    figsize=(8, 8),
    node_size=50,
    edge_color='gray',
    edge_alpha=0.5,
    cmap='tab10',
    with_labels=False
):
    """
    Visualize a Geometric Block Model graph.

    Parameters
    ----------
    G : networkx.Graph
        Must have node attributes:
          - 'pos':  (x, y) coordinates
          - 'comm': integer community label
    figsize : tuple
        Figure size.
    node_size : int
        Size of each node.
    edge_color : color spec
        Color for edges.
    edge_alpha : float
        Transparency for edges.
    cmap : str or matplotlib Colormap
        Colormap for communities.
    with_labels : bool
        Whether to draw node labels.
    """
    # Extract positions
    pos = nx.get_node_attributes(G, 'coords')
    # Extract community labels
    comm = nx.get_node_attributes(G, 'comm')
    # Unique communities
    labels = np.array(list(comm.values()))
    unique = np.unique(labels)
    # Build color map
    col = plt.get_cmap(cmap)
    color_map = {c: col(i / max(1, len(unique)-1)) for i, c in enumerate(unique)}
    # Node colors in order of G.nodes()
    node_colors = [ color_map[comm[n]] for n in G.nodes() ]
    
    plt.figure(figsize=figsize)
    # Draw edges first (so nodes are on top)
    nx.draw_networkx_edges(G, pos,
                           edge_color=edge_color,
                           alpha=edge_alpha,
                           width=0.5)
    # Draw nodes
    nx.draw_networkx_nodes(G, pos,
                           node_color=node_colors,
                           node_size=node_size,
                           linewidths=0.2,
                           edgecolors='black')
    if with_labels:
        nx.draw_networkx_labels(G, pos,
                                font_size=8,
                                font_color='white')
    plt.axis('off')
    plt.tight_layout()
    plt.show()



if __name__ == "__main__": 
    # G = generate_gbm(
    #     n=300,
    #     K=4,
    #     r_in=0.2,
    #     r_out=0.1,
    #     p_in=0.9,
    #     p_out=0.3,
    #     seed=123
    # )

    # plot_gbm(G, node_size=30, edge_alpha=0.3)

    import math


    def gaussian_kernel(d, sigma=0.2):
        return math.exp(-0.5 * (d/sigma)**2)

    # SBM mixing matrix: higher p on the diagonal
    Q = [[0.8, 0.2],
        [0.2, 0.8]]

    # G = generate_gbm_soft_threshold(
    #     n=500,
    #     K=2,
    #     Q=np.array(Q),
    #     kernel=lambda d: gaussian_kernel(d, sigma=0.2),
    #     seed=2025
    # )
    G = generate_gbm(
        n=500,
        K=2,
        r_in=0.2,
        r_out=0.1,
        p_in=0.9,
        p_out=0.1,
        seed=123
    )

    plot_gbm(G, node_size = 30, edge_alpha= 0.3)

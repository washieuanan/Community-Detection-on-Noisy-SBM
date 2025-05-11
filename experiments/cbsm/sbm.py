import numpy as np
import networkx as nx

def generate_noisy_sbm(
    n: int,
    K: int,
    p_in: float,
    p_out: float,
    noise: float,
    r: float,
    dim: int = 2,
    seed: int | None = None
) -> nx.Graph:
    """
    Generate a noisy SBM / geometric-graph hybrid.

    Parameters
    ----------
    n        : number of nodes
    K        : number of communities
    p_in     : probability of an intra-community edge (SBM part)
    p_out    : probability of an inter-community edge (SBM part)
    noise    : ∈ [0,1], fraction of edges drawn from the geometric rule
    r        : radius threshold for the geometric graph
    dim      : dimension of the latent space (default 2)
    seed     : random seed for reproducibility

    Returns
    -------
    G : networkx.Graph
        Undirected graph with node attributes:
          - 'comm': community label in {0,…,K-1}
          - 'pos' : latent position in [0,1]^dim
    """
    rng = np.random.default_rng(seed)

    # latent positions and community labels
    X = rng.random((n, dim))
    comm = rng.integers(0, K, size=n)

    G = nx.Graph()
    G.add_nodes_from(range(n))

    # consider each pair (i,j)
    for i in range(n):
        for j in range(i+1, n):
            if rng.random() < noise:
                # geometric rule
                if np.linalg.norm(X[i] - X[j]) < r:
                    G.add_edge(i, j)
            else:
                # SBM rule
                p = p_in if comm[i] == comm[j] else p_out
                if rng.random() < p:
                    G.add_edge(i, j)

    # store node attributes
    for i in range(n):
        G.nodes[i]['comm'] = int(comm[i])
        G.nodes[i]['coords']  = X[i]

    return G

# Example usage:
if __name__ == "__main__":
    G = generate_noisy_sbm(
        n=500,
        K=2,
        p_in=0.1,
        p_out=0.01,
        noise=0.2,
        r=0.15,
        dim=2,
        seed=42
    )
    # Visualize
    import matplotlib.pyplot as plt
    pos = {i: G.nodes[i]['pos'] for i in G}
    colors = [G.nodes[i]['comm'] for i in G]
    plt.figure(figsize=(6,6))
    nx.draw(G, pos=pos, node_size=30, node_color=colors, with_labels=False)
    plt.show()

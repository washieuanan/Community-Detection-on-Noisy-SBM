import numpy as np
import networkx as nx

def  generate_noisy_sbm(
    n: int,
    K: int,
    p_in: float,
    p_out: float,
    sigma: float,
    dim: int = 2,
    seed: int | None = None
) -> nx.Graph:
    """
    Generate a latent‐space mixed‐kernel SBM.

    Each node i gets:
      - a community label comm[i] ∈ {0,…,K−1}
      - a latent position coords[i] in [0,1]^dim

    Then for each pair (i,j), we compute:
       d = ||coords[i] - coords[j]||
       B = p_in  if comm[i]==comm[j] else p_out
       p_ij = B * exp(-d^2 / (2*sigma^2))

    and add edge (i,j) with probability p_ij.

    Node attributes:
      - 'comm'   : int community label
      - 'coords' : np.ndarray position
      - 'pos'    : same as 'coords' (for nx.draw)
    Edge attribute:
      - 'dist'   : Euclidean distance between endpoints
    """
    rng = np.random.default_rng(seed)

    # 1) sample positions and labels
    coords = rng.random((n, dim))
    comms  = rng.integers(0, K, size=n)

    G = nx.Graph()
    G.add_nodes_from(range(n))

    # 2) store node attributes
    for i in range(n):
        G.nodes[i]['coords'] = coords[i]
        G.nodes[i]['pos']    = coords[i]
        G.nodes[i]['comm']   = int(comms[i])

    # 3) add edges with mixed-kernel probability
    for i in range(n):
        for j in range(i+1, n):
            d      = np.linalg.norm(coords[i] - coords[j])
            base_p = p_in if comms[i] == comms[j] else p_out
            kernel = np.exp(-d*d / (2*sigma*sigma))
            p_ij   = base_p * kernel

            if rng.random() < p_ij:
                G.add_edge(i, j, dist=d)

    return G

def  generate_sbm(
    n: int,
    K: int,
    p_in: float,
    p_out: float,
    sigma: float,
    dim: int = 2,
    seed: int | None = None
) -> nx.Graph:
    """
    Generate a latent‐space mixed‐kernel SBM.

    Each node i gets:
      - a community label comm[i] ∈ {0,…,K−1}
      - a latent position coords[i] in [0,1]^dim

    Then for each pair (i,j), we compute:
       d = ||coords[i] - coords[j]||
       B = p_in  if comm[i]==comm[j] else p_out
       p_ij = B * exp(-d^2 / (2*sigma^2))

    and add edge (i,j) with probability p_ij.

    Node attributes:
      - 'comm'   : int community label
      - 'coords' : np.ndarray position
      - 'pos'    : same as 'coords' (for nx.draw)
    Edge attribute:
      - 'dist'   : Euclidean distance between endpoints
    """
    rng = np.random.default_rng(seed)

    # 1) sample positions and labels
    coords = rng.random((n, dim))
    comms  = rng.integers(0, K, size=n)

    G = nx.Graph()
    G.add_nodes_from(range(n))

    # 2) store node attributes
    for i in range(n):
        G.nodes[i]['coords'] = coords[i]
        G.nodes[i]['pos']    = coords[i]
        G.nodes[i]['comm']   = int(comms[i])

    # 3) add edges with mixed-kernel probability
    for i in range(n):
        for j in range(i+1, n):
            d      = np.linalg.norm(coords[i] - coords[j])
            base_p = p_in if comms[i] == comms[j] else p_out
            # kernel = np.exp(-d*d / (2*sigma*sigma))
            p_ij   = base_p

            if rng.random() < p_ij:
                G.add_edge(i, j, dist=d)

    return G

# Example usage
if __name__ == "__main__":
    G = generate_noisy_sbm(
        n=300,
        K=3,
        p_in=0.1,
        p_out=0.02,
        sigma=0.2,
        dim=2,
        seed=42
    )

    # Quick plot
    import matplotlib.pyplot as plt
    pos    = {i: G.nodes[i]['pos'] for i in G}
    colors = [G.nodes[i]['comm'] for i in G]
    plt.figure(figsize=(6,6))
    nx.draw(G, pos=pos, node_size=30, node_color=colors,
            edge_color='gray', with_labels=False)
    plt.title("Latent-Space Mixed-Kernel SBM")
    plt.show()

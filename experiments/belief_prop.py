import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from copy import deepcopy


def get_sbm(
    num_nodes: int,
    num_communities: int,
    interior_prob: float,
    exterior_prob: float,
    seed=0,
):
    """generates stochastic block model"""
    np.random.seed(seed)

    # generate community sizes
    community_sizes = np.random.multinomial(
        num_nodes, [1 / num_communities] * num_communities
    )
    community_sizes = [max(1, size) for size in community_sizes]

    # generate connection probability matrix
    p_matrix = [[0] * num_communities for _ in range(num_communities)]
    for i in range(num_communities):
        for j in range(num_communities):
            if i == j:
                p_matrix[i][j] = interior_prob
            else:
                p_matrix[i][j] = exterior_prob

    return nx.stochastic_block_model(community_sizes, p_matrix, seed=seed)


def initialize_beliefs(G: nx.Graph, q: int, seed=0):
    """initialize beliefs randomly"""
    rng = np.random.default_rng(seed)
    for node in G.nodes():
        rand_belief = rng.dirichlet(np.ones(q), size=1)[0]
        G.nodes[node]["beliefs"] = rand_belief
        
def calc_beta_param(G: nx.Graph, num_communities: int):
    """
    Calculate beta parameter according to Zhang et al. 2014 - spin glass model
    with scaling based on graph density
    """
    vertex_degs = [G.degree[n] for n in G.nodes()]
    avg_deg = np.mean(vertex_degs)
    n_nodes = G.number_of_nodes()
    
    # Base beta calculation with safety for numerical stability
    base_beta = np.log(num_communities / (max(np.sqrt(avg_deg) - 1, 1e-10)) + 1)
    
    # Scale beta based on graph density to improve convergence
    density = avg_deg / (n_nodes - 1)
    
    # Updated density scaling to match community structure better
    # Lower beta for denser graphs, higher for sparser graphs
    if density > 0.3:  # Dense graph
        density_factor = 0.8
    elif density > 0.1:  # Medium density
        density_factor = 1.0
    else:  # Sparse graph
        density_factor = 1.2
        
    # Adjust beta based on expected community sizes
    # For more balanced communities, we want a slightly higher beta
    community_balance_factor = 1.0
    if num_communities > 1:
        expected_size = n_nodes / num_communities
        if expected_size > 20:  # Large communities need lower beta
            community_balance_factor = 0.9
    
    beta = base_beta * density_factor * community_balance_factor
    return max(0.5, min(5.0, beta))  # Keep beta in reasonable range


def get_true_communities(G: nx.Graph):
    """get true communities"""
    block_labels = nx.get_node_attributes(G, "block")
    return np.array([v for v in block_labels.values()])


def detection_stats(preds, true):
    """calculates basic stats for community detection"""
    num_communities = np.unique(true).size
    true_grouping = {}
    pred_grouping = {}
    for comm in range(num_communities):
        true_grouping[comm] = np.where(true == comm)[0]
        pred_grouping[comm] = np.where(preds == comm)[0]
        
    # get permuation of pred_groupings that most closely matches true_groupings
    perm = np.zeros(num_communities)
    for comm in range(num_communities):
        max_size = 0
        max_comm = -1
        for comm2 in range(num_communities):
            size = len(np.intersect1d(pred_grouping[comm], true_grouping[comm2]))
            if size > max_size:
                max_size = size
                max_comm = comm2
        perm[comm] = max_comm

    # calculate accuracy
    stats = {}
    num_correct = 0
    for n in range(len(preds)):
        if true[n] == perm[preds[n]]:
            num_correct += 1
    stats["accuracy"] = num_correct / len(preds)

    # accuracy per community
    for comm in range(num_communities):
        num_correct = 0
        for n in true_grouping[comm]:
            if perm[preds[n]] == comm:
                num_correct += 1
        stats[f"accuracy_{comm}"] = num_correct / len(true_grouping[comm])

    return stats


def get_marginals_and_preds(G: nx.Graph):
    marginals = np.array([G.nodes[i]["beliefs"] for i in G.nodes()])
    preds = np.argmax(marginals, axis=1)
    return marginals, preds

def belief_propagation(
    G: nx.Graph,
    q: int,
    beta: float | None = None,
    max_iter: int = 1000,
    tol: float = 1e-4,
    damping: float = 0.3,  # Lower damping for better exploration
    anneal_steps: int = 30,  # More annealing steps
    balance_regularization: float = 0.1,  # Community balance regularization
    degen_threshold: float = 0.9,  # Lower threshold to detect degeneration earlier
    seed: int = 0,
):
    """
    Belief propagation from Zhang et. al. 2014
    
    Parameters:
    ----------
    G : nx.Graph
        Input graph
    q : int
        Number of communities
    beta : float, optional
        Interaction strength parameter
    max_iter : int
        Maximum number of iterations
    tol : float
        Convergence tolerance
    damping : float
        Damping factor for update smoothing
    anneal_steps : int
        Number of temperature annealing steps
    seed : int
        Random seed
    """
    rng = np.random.default_rng(seed)
    m = G.number_of_edges()
    deg = dict(G.degree())
    c = np.mean(list(deg.values()))
    if beta is None:
        beta = calc_beta_param(G, q)
    
    messages = {(i, j): rng.dirichlet(np.ones(q), size=1)[0] + 1e-3 for i in G for j in G.neighbors(i)}
    
    old_messages = deepcopy(messages)
    
    beta_schedule = np.linspace(beta * 0.1, beta, anneal_steps)
    convergence_history = []
    
    for it in range(max_iter):
        if it < anneal_steps:
            current_beta = beta_schedule[it]
        else:
            current_beta = beta
        old_messages, messages = messages, old_messages

        for i in G:
            prod = np.ones(q)
            for t in range(q):
                s = 0.0
                for j in G.neighbors(i):
                    msg_value = max(1e-10, old_messages[(j, i)][t])
                    s += np.log(1 + (np.exp(current_beta) - 1) * msg_value)
                prod[t] = np.exp(s)
            prod /= prod.sum()
            G.nodes[i]["beliefs"] = prod
        community_sizes = np.zeros(q)
        theta = np.zeros(q)
        for t in range(q):
            community_sizes[t] = sum(G.nodes[u]["beliefs"][t] for u in G) / len(G)
            theta[t] = sum(deg[u] * G.nodes[u]["beliefs"][t] for u in G)

        for i in G:
            deg_i = deg[i]
            neigh_i = list(G.neighbors(i))

            for k in neigh_i:
                log_new_msg = np.zeros(q)
                
                for t in range(q):
                    term1 = -current_beta * deg_i * theta[t] / (2 * m)
                    
                    term2 = 0.0
                    for j in neigh_i:
                        if j == k:
                            continue
                        msg_value = max(1e-10, old_messages[(j, i)][t])
                        term2 += np.log(1 + (np.exp(current_beta) - 1) * msg_value)
                    
                    # Penalize assignment to large communities, boost small ones
                    size_penalty = balance_regularization * np.log(community_sizes[t] + 1e-10)
                    
                    log_new_msg[t] = term1 + term2 - size_penalty
                
                max_val = np.max(log_new_msg)
                new_msg = np.exp(log_new_msg - max_val)
                new_msg /= new_msg.sum()
                
                messages[(i, k)] = (1 - damping) * new_msg + damping * old_messages[(i, k)]
                messages[(i, k)] = messages[(i, k)] / np.sum(messages[(i, k)])
                
        delta = max(
            np.abs(messages[(i, k)] - old_messages[(i, k)]).max() for i, k in messages
        )
        convergence_history.append(delta)
               
        if delta < tol:
            community_entropy = -np.sum(community_sizes * np.log(community_sizes + 1e-10))
            ideal_entropy = -np.log(1/q)
            entropy_ratio = community_entropy / ideal_entropy
            
            if entropy_ratio > 0.8: 
                print(f"BP converged in {it+1} iterations, entropy ratio: {entropy_ratio:.3f}")
                break
            else:
                if it > anneal_steps:
                    print(f"Low entropy ratio ({entropy_ratio:.3f}), continuing search...")
                    inv_sizes = 1.0 / (community_sizes + 1e-10)
                    inv_sizes = inv_sizes / np.sum(inv_sizes)
                    for edge in messages:
                        noise = rng.random(q) * 0.15 * inv_sizes
                        messages[edge] = messages[edge] * 0.85 + noise
                        messages[edge] /= messages[edge].sum()
    else:
        print(f"BP did not converge in {max_iter} iterations")
        
    community_sizes = np.zeros(q)
    for t in range(q):
        community_sizes[t] = sum(G.nodes[u]["beliefs"][t] for u in G) / len(G)
    
    community_entropy = -np.sum(community_sizes * np.log(community_sizes + 1e-10))
    ideal_entropy = -np.log(1/q)
    entropy_ratio = community_entropy / ideal_entropy
    
    print(f"Final beta: {beta}")
    print(f"Final community proportions: {community_sizes}")
    print(f"Entropy ratio: {entropy_ratio:.4f} (1.0 is perfectly balanced)")
    
if __name__ == "__main__":
    num_nodes = 100
    num_communities = 3
    interior_prob = 0.7  # Increased internal connectivity
    exterior_prob = 0.05  # Decreased external connectivity
    
    print(f"Creating SBM with {num_nodes} nodes, {num_communities} communities")
    print(f"Interior probability: {interior_prob}, Exterior probability: {exterior_prob}")
    
    G = get_sbm(num_nodes, num_communities, interior_prob, exterior_prob)
    initialize_beliefs(G, num_communities)
    
    belief_propagation(
        G, 
        num_communities, 
        max_iter=1000,
        damping=0.3,
        anneal_steps=30,
        balance_regularization=0.1,
    )
    true_comms = get_true_communities(G)
    marginals, preds = get_marginals_and_preds(G)
    
    unique_true, true_counts = np.unique(true_comms, return_counts=True)
    unique_pred, pred_counts = np.unique(preds, return_counts=True)
    
    print("True community sizes:", dict(zip(unique_true, true_counts)))
    print("Predicted community sizes:", dict(zip(unique_pred, pred_counts)))
    print("Detection stats:", detection_stats(preds, true_comms))

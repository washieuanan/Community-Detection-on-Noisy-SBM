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
    """calculate beta  param according to Zhang et al. 2014 - spin glass model"""
    vertex_degs = [G.degree[n] for n in G.nodes()]
    avg_deg = np.mean(vertex_degs)
    beta = np.log(num_communities / (np.sqrt(avg_deg) - 1) + 1)
    return beta


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
    max_iter: int = 400,
    tol: float = 1e-5,
    damping: float = 0.1,
    seed: int = 0,
):
    """
    Belief propagation from Zhang et. al. 2014
    """
    rng = np.random.default_rng(seed)
    m = G.number_of_edges()
    deg = dict(G.degree())
    c = np.mean(list(deg.values()))

    if beta is None:
        beta = calc_beta_param(G, q) + 1e-3
    
    messages = {(i, j): rng.dirichlet(np.ones(q), size=1)[0] + 1e-3 for i in G for j in G.neighbors(i)}
    old_messages = deepcopy(messages)
    for it in range(max_iter):
        old_messages, messages = messages, old_messages

        for i in G:
            prod = np.ones(q)
            for t in range(q):
                s = 0.0
                for j in G.neighbors(i):
                    msg_value = max(1e-10, old_messages[(j, i)][t])
                    s += np.log(1 + (np.exp(beta) - 1) * msg_value)
                prod[t] = np.exp(s)

            prod /= prod.sum()
            G.nodes[i]["beliefs"] = prod

        theta = np.zeros(q)
        for t in range(q):
            theta[t] = sum(deg[u] * G.nodes[u]["beliefs"][t] for u in G)

        for i in G:
            deg_i = deg[i]
            neigh_i = list(G.neighbors(i))

            for k in neigh_i:
                new_msg = np.empty(q)
                for t in range(q):
                    term1 = -beta * deg_i * theta[t] / (2 * m)
                    term2 = 0.0
                    for j in neigh_i:
                        if j == k:
                            continue
                        msg_value = max(1e-10, old_messages[(j, i)][t])
                        term2 += np.log(1 + (np.exp(beta) - 1) * msg_value)
                    new_msg[t] = np.exp(term1 + term2)
                new_msg /= new_msg.sum()
                messages[(i, k)] = (1 - damping) * new_msg + damping * old_messages[
                    (i, k)
                ]

        delta = max(
            np.abs(messages[(i, k)] - old_messages[(i, k)]).max() for i, k in messages
        )

        if delta < tol:
            print(f"BP converged in {it+1} iterations")
            break
    else:
        print(f"BP did not converge in {max_iter} iterations")


if __name__ == "__main__":
    num_nodes = 100
    num_communities = 3
    interior_prob = 0.6
    exterior_prob = 0.1

    G = get_sbm(num_nodes, num_communities, interior_prob, exterior_prob)
    initialize_beliefs(G, num_communities)
    belief_propagation(G, num_communities)

    true_comms = get_true_communities(G)
    marginals, preds = get_marginals_and_preds(G)
    print("True communities:", true_comms)
    print("BP communities:", preds)
    print("Detection stats:", detection_stats(preds, true_comms))

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from copy import deepcopy
from typing import Literal
from scipy.stats import chi2_contingency, permutation_test, mode
from sklearn.metrics import accuracy_score, confusion_matrix
import scipy.sparse.linalg as sla
from sklearn.cluster import KMeans

# no longer needed
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

    eps = 1e-3
    numerator = num_communities*(1+(num_communities - 1)*eps)
    denominator = max(avg_deg*(1-eps) - (1 + (num_communities - 1)*eps), 1e-10)
    base_beta = np.log(numerator/denominator + 1)
    density = avg_deg / (n_nodes - 1)
    print("Graph Density:", density)
    beta = base_beta * 1.2
    return beta  # Keep beta in reasonable range
    # return base_beta


def get_true_communities(G: nx.Graph):
    """get true communities"""
    block_labels = nx.get_node_attributes(G, "block")
    return np.array([v for v in block_labels.values()])


def detection_stats(preds, true):
    """calculates basic stats for community detection"""
    num_communities = max(max(preds), max(true)) + 1
    true_grouping = {}
    pred_grouping = {}
    for comm in range(num_communities):
        true_grouping[comm] = np.where(true == comm)[0]
        pred_grouping[comm] = np.where(preds == comm)[0]

    # get permuation of pred_groupings that most closely matches true_groupings
    perm = np.zeros(num_communities)
    available_comms = {c for c in range(num_communities)}
    sorted_pred_grouping = sorted(
        list(range(num_communities)), key=lambda x: len(pred_grouping[x]), reverse=True
    )
    for comm in sorted_pred_grouping:
        max_size = 0
        max_comm = -1
        for comm2 in available_comms:
            size = len(np.intersect1d(pred_grouping[comm], true_grouping[comm2]))
            if size >= max_size:
                max_size = size
                max_comm = comm2
        perm[comm] = max_comm
        available_comms.remove(max_comm)

    permed_pred = np.array([perm[preds[i]] for i in range(len(preds))])

    stats = {}
    stats["accuracy"] = accuracy_score(true, permed_pred)

    # accuracy per community
    for comm in range(num_communities):
        true_comm = [true[i] for i in true_grouping[comm]]
        pred_comm = [permed_pred[i] for i in true_grouping[comm]]
        stats[f"accuracy_{comm}"] = accuracy_score(true_comm, pred_comm)

    res = permutation_test(
        (true, permed_pred),
        statistic=lambda x, y: accuracy_score(x, y),
        vectorized=False,
        n_resamples=10_000,
        alternative="greater",
        random_state=0,
    )

    stats["perm_p"] = int(res.pvalue)

    stats["num vertices"] = len(preds)
    stats["num communities predicted"] = len(np.unique(preds))
    return stats


def get_marginals_and_preds(G: nx.Graph):
    marginals = np.array([G.nodes[i]["beliefs"] for i in G.nodes()])
    preds = np.argmax(marginals, axis=1)
    return marginals, preds


def initialize_messages(
    G: nx.Graph,
    q: int,
    method: tuple[Literal["random", "copy", "pre-group"]],
    seed: int = 0,
    group_obs=None,
    min_sep=None,
    eps=0.1,
):
    """initialize messages for belief propagation"""
    rng = np.random.default_rng(seed)
    spectral_infos = spectral_clustering(G, q, seed=seed)
    if method == "random":
        messages = {
            (i, j): rng.dirichlet(np.ones(q), size=1)[0] + 1e-3
            for i in G
            for j in G.neighbors(i)
        }
        eps0 = eps
        # add bias towards spectral label
        for (i, j), mes in messages.items():
            spec_label = spectral_infos[i]
            mes[spec_label] += eps0
            mes /= mes.sum()
            messages[(i, j)] = mes
        return messages
    elif method == "copy":
        messages = {(i, j): G.nodes[i]["beliefs"] for i in G for j in G.neighbors(i)}
        eps0 = eps
        # add bias towards spectral label
        for (i, j), mes in messages.items():
            spec_label = spectral_infos[i]
            mes[spec_label] += eps0
            mes /= mes.sum()
            messages[(i, j)] = mes
        return messages
    elif method == "pre-group":
        if not group_obs:
            assert "Need to provide grouped observations for this method"
        bias_assignment = np.zeros(len(group_obs), dtype=int)
        for group in range(len(bias_assignment)):
            if isinstance(group_obs[group], dict):
                group_spec_labels = [
                    spectral_infos[i]
                    for i, _ in group_obs[group][r]
                    for r in group_obs[group]
                ]
            else:
                group_spec_labels = [spectral_infos[i] for i, _ in group_obs[group]]
            if len(group_spec_labels) == 0:
                bias_assignment[group] = -1
            else:
                bias_assignment[group] = mode(group_spec_labels)[0]
        messages = {}
        messages = {
            (i, j): rng.dirichlet(np.ones(q), size=1)[0] + 1e-3
            for i in G
            for j in G.neighbors(i)
        }
        for group in range(len(group_obs)):
            if bias_assignment[group] == -1:
                continue
            bias_vector = np.zeros(q)
            if min_sep:
                bias_vector[bias_assignment[group]] = np.sqrt(min_sep)
            else:
                bias_vector[bias_assignment[group]] = np.sqrt(0.15)
            if isinstance(group_obs[group], dict):
                for rad in group_obs[group]:
                    rad_adj = np.zeros(q)
                    rad_adj[bias_assignment[group]] = max(-0.2 * np.exp(rad), -1 * bias_vector[bias_assignment[group]])

                    for u, v in group_obs[group][rad]:
                        messages[(int(u), int(v))] = (
                            messages[(u, v)] + bias_vector + rad_adj
                        )
                        messages[(int(u), int(v))] = messages[(u, v)] / np.sum(
                            messages[(u, v)]
                        )
                        messages[(int(v), int(u))] = (
                            messages[(u, v)] + bias_vector + rad_adj
                        )
                        messages[(int(v), int(u))] = messages[(u, v)] / np.sum(
                            messages[(u, v)]
                        )
            else:
                for u, v in group_obs[group]:
                    messages[(int(u), int(v))] = messages[(u, v)] + bias_vector
                    messages[(int(u), int(v))] = messages[(u, v)] / np.sum(
                        messages[(u, v)]
                    )
                    messages[(int(v), int(u))] = messages[(u, v)] + bias_vector
                    messages[(int(v), int(u))] = messages[(u, v)] / np.sum(
                        messages[(u, v)]
                    )
        return messages


def spectral_clustering(G, q, seed=0):
    A = nx.adjacency_matrix(G)
    vals, vecs = sla.eigs(A, k=q, which="LM", tol=1e-2)
    coords = np.real(vecs)
    km = KMeans(n_clusters=q, random_state=seed).fit(coords)
    return {node: int(km.labels_[i]) for i, node in enumerate(G)}


def belief_propagation(
    G: nx.Graph,
    q: int,
    beta: float | None = None,
    max_iter: int = 1000,
    tol: float = 1e-4,
    damping: float = 0.2,
    balance_regularization: float = 0.1,
    seed: int = 0,
    min_steps: int = 0,
    message_init: tuple[Literal["random", "copy", "pre-group"]] = "random",
    group_obs=None,
    min_sep=None,
):
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    m = G.number_of_edges()
    deg = dict(G.degree())
    c = np.mean(list(deg.values()))

    if beta is None:
        beta = calc_beta_param(G, q) * 1.1

    messages = initialize_messages(
        G, q, message_init, seed=seed, group_obs=group_obs, min_sep=min_sep
    )

    old_messages = deepcopy(messages)
    convergence_history = []

    for it in range(max_iter):
        old_messages, messages = messages, old_messages

        # update beliefs
        for i in G:
            prod = np.ones(q)
            for t in range(q):
                s = 0.0
                for j in G.neighbors(i):
                    msg_value = np.clip(old_messages[(j, i)][t], 1e-10, 1)
                    s += np.log1p(np.expm1(beta) * msg_value)
                prod[t] = np.exp(s)
            prod /= np.clip(prod.sum(), 1e-10, None)
            G.nodes[i]["beliefs"] = prod

        # compute community sizes and theta
        community_sizes = np.array(
            [sum(G.nodes[u]["beliefs"][t] for u in G) for t in range(q)]
        ) / len(G)
        theta = np.array(
            [sum(deg[u] * G.nodes[u]["beliefs"][t] for u in G) for t in range(q)]
        )

        # message updates
        for i in G:
            deg_i = deg[i]
            neigh_i = list(G.neighbors(i))
            for k in neigh_i:
                log_new = np.zeros(q)
                for t in range(q):
                    term1 = -beta * deg_i * theta[t] / (2 * m)
                    term2 = sum(
                        np.log(
                            1
                            + (np.exp(beta) - 1)
                            * max(1e-10, old_messages[(j, i)][t])
                        )
                        for j in neigh_i
                        if j != k
                    )
                    size_penalty = -balance_regularization * np.log(
                        community_sizes[t] + 1e-10
                    )
                    log_new[t] = term1 + term2 - size_penalty
                maxv = log_new.max()
                new_msg = np.exp(log_new - maxv)
                new_msg /= new_msg.sum()
                m_old = old_messages[(i, k)]
                m_upd = (1 - damping) * new_msg + damping * m_old
                messages[(i, k)] = m_upd / m_upd.sum()

        # check convergence
        delta = max(abs(messages[e] - old_messages[e]).max() for e in messages)
        convergence_history.append(delta)
        if delta < tol and it > min_steps:
            ent = -np.sum(community_sizes * np.log(community_sizes + 1e-10))
            if ent / (-np.log(1 / q)) > 0.7:
                print(
                    f"BP converged in {it+1} iters; entropy ratio {ent/(-np.log(1/q)):.3f}"
                )
                break
            for e in messages:
                noise = rng.random(q) * 0.15 / (community_sizes + 1e-10)
                messages[e] = messages[e] * 0.85 + noise
                messages[e] /= messages[e].sum()
    else:
        print(f"BP did not converge in {max_iter} iterations")
    return


if __name__ == "__main__":
    from experiments.observations.sensor_observe import GroupedMultiSensorObservation
    from experiments.graph_generation.gbm import generate_gbm
    from experiments.community_detection.bp.gbm_bp import create_observed_subgraph
    from experiments.observations.random_walk_obs import GroupedRandomWalkObservation
    from experiments.observations.standard_observe import (
        PairSamplingObservation,
        get_coordinate_distance,
    )

    G2 = generate_gbm(n=900, K=3, r_in=0.25, r_out=0.1, p_in=0.7, p_out=0.2, seed=123)
    # multi = GroupedRandomWalkObservation(graph=G2, seed=123, num_walkers=10,
    #                                      num_steps=5, stopping_param=0.1,
    #                                   leaky=0.1,)
    avg_deg = np.mean([G2.degree[n] for n in G2.nodes()])
    orig_sparsity = avg_deg/len(G2.nodes)
    print("Original Sparsity:", orig_sparsity)
    C = 0.1 * orig_sparsity
    # C = 0.001
    num_sensors = int((C * len(G2.nodes))/(0.25**3 * avg_deg))
    multi = GroupedMultiSensorObservation(
        G2, seed=42, num_sensors=num_sensors, radii=np.linspace(0.1, 1.0, 10), min_sep=0.15
    )
    # num_walkers = int((C * 0.1 * len(G2.nodes) ** 2)/2)
    # print(num_walkers)
    # multi = GroupedRandomWalkObservation(graph=G2, seed=123, num_walkers=num_walkers,
    #                                      num_steps=5, stopping_param=0.1,
    #                                   leaky=0.1,)
    edges = multi.observe()
    # def weight_func(c1, c2):
    #     return np.exp(-0.5 * get_coordinate_distance(c1, c2))

    # num_pairs = int(C * len(G2.nodes) ** 2 / 2)
    print("NUM EDGES:", len(G2.edges))
    print("SAMPLINHG EDGES:", num_sensors)
    # Pair-based sampling
    # pair_sampler = PairSamplingObservation(G2, num_samples=num_pairs, weight_func=weight_func, seed=42)
    # observations = pair_sampler.observe()
    observations = []
    for g in edges:
        for r, obs in g.items():
            for u, v in obs:
                observations.append((u, v))
    # for g in edges:
    #     for u, v in g:
    #         observations.append((u, v))

    # New approach - get all unique edges across all radii
    # unique_edges = get_unique_edges(obs)

    # Print counts to compare
    print(f"Total observations (may include duplicates): {len(observations)}")
    # print(f"Unique edges: {len(unique_edges)}")

    observed_nodes = set()
    for u, v in observations:
        observed_nodes.add(u)
        observed_nodes.add(v)
    # print(observations)

    # subG = create_observed_subgraph(100, observations)``
    subG = create_observed_subgraph(len(G2.nodes), observations)
    # subG =iG2
    initialize_beliefs(subG, 3)
    # messages = initialize_messages(subG, 3, "pre-group", group_obs=edges, min_sep=0.15)
    # print(messages)
    belief_propagation(
        subG,
        q=3,
        max_iter=5000,
        damping=0.15,
        balance_regularization=0.05,
        min_steps=50,
        message_init="pre-group",
        group_obs=edges,
        min_sep=0.15,
    )
    marginals, preds = get_marginals_and_preds(subG)
    # cluster_map = np.array(cluster_map2)
    cluster_map = nx.get_node_attributes(G2, "comm")
    cluster_map = np.array(list(cluster_map.values()))
    sub_preds = np.array([preds[i] for i in range(len(preds)) if i in observed_nodes])
    sub_cluster_map = np.array(
        [cluster_map[i] for i in range(len(cluster_map)) if i in observed_nodes]
    )
    print(detection_stats(preds, cluster_map))
    print(detection_stats(sub_preds, sub_cluster_map))

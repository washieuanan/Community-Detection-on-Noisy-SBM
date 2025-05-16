import networkx as nx
import numpy as np
import scipy.sparse as sp

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import TruncatedSVD
import umap

try:
    import dask.array as da
    from dask_ml.decomposition import TruncatedSVD as DaskSVD
    _HAS_DASK = True
except ImportError:
    _HAS_DASK = False

try:
    from lmds import LandmarkMDS
    _HAS_LMDS = True
except ImportError:
    _HAS_LMDS = False

try:
    from nodevectors import Node2Vec
    _HAS_NODE2VEC = True
except ImportError:
    _HAS_NODE2VEC = False

try:
    from datasketch import MinHash, MinHashLSH
    _HAS_MINHASH = True
except ImportError:
    _HAS_MINHASH = False


def _preprocess(products, subsampled_classes):
    """
    Build graph G, list of node IDs (0..n-1), and sparse category matrix X,
    skipping any product marked as discontinued. Each node is named by its
    numeric ID and has 'asin' and optionally 'comm' attributes.
    
    All classes that can be included in subsampled_classes:
    * Book, DVD, Music, Video
    """
    G = nx.Graph()

    valid = {
        asin: info
        for asin, info in products.items()
        if (not info.get('group', None) is None)
        and (info.get('group', None) in subsampled_classes)
    }
    
    comm_mapping = {sc: i for i, sc in enumerate(subsampled_classes)}

    asins = list(valid.keys())
    node_ids = list(range(len(asins)))
    for idx, asin in zip(node_ids, asins):
        info = valid[asin]
        attrs = {'asin': asin}
        if info.get('group') is not None:
            attrs['comm'] = comm_mapping[info['group']]
        G.add_node(idx, **attrs)

    asin_to_id = {asin: idx for idx, asin in zip(node_ids, asins)}
    for asin, info in valid.items():
        u = asin_to_id[asin]
        for nbr in info.get('similar', []):
            if nbr in asin_to_id:
                v = asin_to_id[nbr]
                G.add_edge(u, v)

    categories = [valid[asin].get('categories', []) for asin in asins]
    mlb = MultiLabelBinarizer(sparse_output=True)
    X = mlb.fit_transform(categories)  # CSR matrix: n_nodes × n_categories
    G.graph['subclasses'] = ','.join(subsampled_classes)
    return G, node_ids, X


def _embed_svd(X, n_components, random_state=None, use_dask=False):
    if use_dask and _HAS_DASK:
        Xd = da.from_array(X, chunks=(100_000, X.shape[1]))
        svd = DaskSVD(n_components=n_components, random_state=random_state)
        Y = svd.fit_transform(Xd).compute()
    else:
        svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        Y = svd.fit_transform(X)
    return Y / np.linalg.norm(Y, axis=1, keepdims=True)


def _embed_umap(X, n_components, random_state=None):
    reducer = umap.UMAP(n_components=n_components,
                        metric='jaccard',
                        random_state=random_state)
    Y = reducer.fit_transform(X)
    return Y / np.linalg.norm(Y, axis=1, keepdims=True)


def _embed_landmark_mds(X, n_components, n_landmarks=1000, random_state=None):
    if not _HAS_LMDS:
        raise ImportError("LandmarkMDS not installed (`pip install lmds`).")
    lmds = LandmarkMDS(n_components=n_components,
                       n_landmarks=n_landmarks,
                       random_state=random_state)
    Y = lmds.fit_transform(X, metric='jaccard')
    return Y / np.linalg.norm(Y, axis=1, keepdims=True)


def _embed_node2vec(G, dimensions, walklen=10, epochs=1):
    if not _HAS_NODE2VEC:
        raise ImportError("nodevectors not installed (`pip install nodevectors`).")
    model = Node2Vec(dimensions=dimensions,
                     walklen=walklen,
                     epochs=epochs)
    model.fit(G)
    embs = model.predict(G.nodes())
    Y = np.vstack([embs[node] for node in G.nodes()])
    return Y / np.linalg.norm(Y, axis=1, keepdims=True)


def _build_lsh(categories, num_perm=128, threshold=0.5):
    if not _HAS_MINHASH:
        raise ImportError("datasketch not installed (`pip install datasketch`).")
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for idx, cats in enumerate(categories):
        m = MinHash(num_perm=num_perm)
        for c in cats:
            m.update(c.encode('utf8'))
        lsh.insert(idx, m)
    return lsh

def build_category_graph(products,
                         subsampled_classes=['Book', 'DVD', 'Music', 'Video']):
    """
    Build G where each node has:
      - 'asin', optional 'comm' (from your _preprocess)
      - 'cat_vec': binary numpy array of length n_categories,
        indicating which categories it belongs to.

    Each edge (u,v) gets attribute:
      - 'dist': normalized Hamming distance between cat_vec[u] and cat_vec[v],
        scaled so that the maximum observed Hamming distance is 1.
    """
    # 1) Build G, node_ids and sparse X via your original _preprocess
    G, node_ids, X = _preprocess(products, subsampled_classes)

    # 2) Assign binary category vectors to nodes
    #    X is n_nodes × n_categories in CSR; X[i].A1 is a dense 1D array
    for idx in node_ids:
        G.nodes[idx]['coords'] = X[idx].toarray().flatten().astype(int)

    # 3) Compute raw Hamming distances for each edge
    raw = {}
    for u, v in G.edges():
        vec_u = G.nodes[u]['coords']
        vec_v = G.nodes[v]['coords']
        # Hamming = count of differing bits
        raw[(u, v)] = np.count_nonzero(vec_u != vec_v)

    # 4) Normalize so max(raw) → 1
    max_raw = max(raw.values()) if raw else 1
    for (u, v), d in raw.items():
        G.edges[u, v]['dist'] = 2 * (d / max_raw)

    return G


if __name__ == "__main__":
    import json
    import os
    import random
    from collections import Counter
    import networkx as nx
    from collections import defaultdict
    # Example usage
    products = json.load(open("amazon_metadata_test/parsed_amazon_meta.json"))
    G = build_category_graph(products, subsampled_classes=['Video', 'DVD'])
    def subsample_equal_connected(G, desired_n=1000, class_attr='comm', seed=None):
        rng = random.Random(seed)

        # 1) group nodes by class
        nodes_by_comm = defaultdict(list)
        for u, data in G.nodes(data=True):
            if class_attr in data:
                nodes_by_comm[data[class_attr]].append(u)
        classes = list(nodes_by_comm)
        n_classes = len(classes)
        if n_classes == 0:
            raise ValueError("No classes found in G")

        # 2) compute per‐class quotas
        base = desired_n // n_classes
        rem  = desired_n % n_classes
        quotas = {}
        for i, c in enumerate(classes):
            quotas[c] = base + (1 if i < rem else 0)

        # 3) BFS‐grow a connected sample until quotas are met (seed in largest class)
        start_comm = max(classes, key=lambda c: len(nodes_by_comm[c]))
        start_node = rng.choice(nodes_by_comm[start_comm])
        selected    = [start_node]
        quotas[start_comm] -= 1

        visited = {start_node}
        queue   = [start_node]

        while queue and any(q > 0 for q in quotas.values()):
            u = queue.pop(0)
            for v in G.neighbors(u):
                if v in visited:
                    continue
                visited.add(v)
                c = G.nodes[v].get(class_attr)
                if quotas.get(c, 0) > 0:
                    selected.append(v)
                    quotas[c] -= 1
                    queue.append(v)
            if all(q == 0 for q in quotas.values()):
                break

        # 4) if you still didn’t hit 1 000, fill the rest at random
        if len(selected) < desired_n:
            needed = desired_n - len(selected)
            pool = list(set(G.nodes()) - set(selected))
            selected.extend(rng.sample(pool, needed))

        # 5) finally, reconnect any disjoint bits by sewing in the shortest paths
        def connect_components(G, nodes):
            sub = G.subgraph(nodes).copy()
            comps = list(nx.connected_components(sub))
            if len(comps) == 1:
                return sub
            master = set(comps[0])
            for comp in comps[1:]:
                u = rng.choice(list(comp))
                v = rng.choice(list(master))
                path = nx.shortest_path(G, u, v)
                master.update(path)
                master.update(comp)
            return G.subgraph(master).copy()

        return connect_components(G, selected)
    # # Subsample: keep up to 20,000 nodes for each community
    # books_nodes = [n for n, data in G.nodes(data=True) if data.get('comm') == 0]
    # music_nodes = [n for n, data in G.nodes(data=True) if data.get('comm') == 1]
    # video_nodes = [n for n, data in G.nodes(data=True) if data.get('comm') == 2]
    # dvd_nodes = [n for n, data in G.nodes(data=True) if data.get('comm') == 3]

    # books_sample = set(random.sample(books_nodes, 5000)) if len(books_nodes) > 5000 else set(books_nodes)
    # music_sample = set(random.sample(music_nodes, 5000)) if len(music_nodes) > 5000 else set(music_nodes)
    # video_sample = set(random.sample(video_nodes, 5000)) if len(video_nodes) > 5000 else set(video_nodes)
    # dvd_sample = set(random.sample(dvd_nodes, 5000)) if len(dvd_nodes) > 5000 else set(dvd_nodes)

    # keep_nodes = books_sample.union(music_sample, video_sample, dvd_sample)
    # G = G.subgraph(keep_nodes).copy()
    # G = subsample_equal_connected(G, desired_n=2000, class_attr='comm', seed=123)
    print("Number of nodes:", G.number_of_nodes())
    print("Graph is connected:", nx.is_connected(G))

    comm_counts = {}
    for _, data in G.nodes(data=True):
        if 'comm' in data:
            comm_counts[data['comm']] = comm_counts.get(data['comm'], 0) + 1

    print("Nodes per comm:")
    for comm, count in comm_counts.items():
        print(f"  {comm}: {count}")
    for node in G.nodes():
        if 'coords' in G.nodes[node]:
            coords = G.nodes[node]['coords']
            G.nodes[node]['coords'] = ','.join(map(str, coords.tolist()))

    nx.write_gml(G, 'amazon_metadata_test/amz_allviddvd.gml')
    # G = nx.read_gml("amazon_metadata_test/amazon_hamming_videoDVD.gml")
    
    # node_info = G.nodes['0']['coords']
    # node_info = np.fromstring(node_info[1:-1], sep=",")
    # print(sum(node_info))
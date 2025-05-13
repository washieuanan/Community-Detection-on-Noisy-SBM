import json
import networkx as nx
import numpy as np
import scipy.sparse as sp

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

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
    """
    G = nx.Graph()

    # filter to only the chosen groups
    valid = {
        asin: info
        for asin, info in products.items()
        if info.get('group') in subsampled_classes
    }
    comm_mapping = {sc: i for i, sc in enumerate(subsampled_classes)}

    asins   = list(valid.keys())
    node_ids = list(range(len(asins)))
    for idx, asin in zip(node_ids, asins):
        info = valid[asin]
        attrs = {'asin': asin}
        if 'group' in info:
            attrs['comm'] = comm_mapping[info['group']]
        G.add_node(idx, **attrs)

    asin_to_id = {asin: idx for idx, asin in zip(node_ids, asins)}
    for asin, info in valid.items():
        u = asin_to_id[asin]
        for nbr in info.get('similar', []):
            if nbr in asin_to_id:
                v = asin_to_id[nbr]
                G.add_edge(u, v)

    # build the category indicator matrix
    categories = [valid[asin].get('categories', []) for asin in asins]
    mlb = MultiLabelBinarizer(sparse_output=True)
    X   = mlb.fit_transform(categories)  # shape (n_nodes, n_categories)
    G.graph['subclasses'] = ','.join(subsampled_classes)
    return G, node_ids, X


def embed_products(products,
                   method='categories',
                   subsampled_classes=['Book', 'DVD', 'Music', 'Video'],
                   **kwargs):
    """
    Build G and compute embeddings via one of:
      - 'categories'     : use raw one-hot category vectors
      - 'svd'            : Truncated SVD on attribute matrix
      - 'umap'           : UMAP with Jaccard metric
      - 'landmark_mds'   : Landmark MDS (needs lmds)
      - 'node2vec'       : Node2Vec on the graph (needs nodevectors)
      - 'minhash_lsh'    : MinHash + LSH index (returns LSH object)

    When using 'categories', each node will get:
      G.nodes[i]['coords'] = the binary category vector
      G.edges[u, v]['dist'] = cosine_similarity(coords_u, coords_v)
    """
    G, node_ids, X = _preprocess(products, subsampled_classes)

    if method == 'categories':
        # convert sparse rows to dense array once
        X_dense = X.toarray()
        # assign each node its category vector
        for idx in node_ids:
            G.nodes[idx]['coords'] = X_dense[idx]
        # compute cosine-sim on each edge
        for u, v in G.edges():
            sim = cosine_similarity(
                X_dense[u].reshape(1, -1),
                X_dense[v].reshape(1, -1)
            )[0, 0]
            G.edges[u, v]['dist'] = sim
        return G

    # ---- below: your existing options, unchanged ----
    # you can still pass embedding_dim, random_state, use_dask, etc. if you like
    if method == 'svd':
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(
            n_components=kwargs.get('embedding_dim', 2),
            random_state=kwargs.get('random_state', None)
        )
        Y = svd.fit_transform(X)
    elif method == 'umap':
        import umap
        reducer = umap.UMAP(
            n_components=kwargs.get('embedding_dim', 2),
            metric='jaccard',
            random_state=kwargs.get('random_state', None)
        )
        Y = reducer.fit_transform(X)
    elif method == 'landmark_mds':
        if not _HAS_LMDS:
            raise ImportError("LandmarkMDS not installed (`pip install lmds`).")
        lmds = LandmarkMDS(
            n_components=kwargs.get('embedding_dim', 2),
            n_landmarks=kwargs.get('n_landmarks', 1000),
            random_state=kwargs.get('random_state', None)
        )
        Y = lmds.fit_transform(X, metric='jaccard')
    elif method == 'node2vec':
        if not _HAS_NODE2VEC:
            raise ImportError("nodevectors not installed (`pip install nodevectors`).")
        model = Node2Vec(
            dimensions=kwargs.get('embedding_dim', 2),
            walklen=kwargs.get('walklen', 10),
            epochs=kwargs.get('epochs', 1)
        )
        model.fit(G)
        embs = model.predict(G.nodes())
        Y = np.vstack([embs[node] for node in G.nodes()])
    elif method == 'minhash_lsh':
        if not _HAS_MINHASH:
            raise ImportError("datasketch not installed (`pip install datasketch`).")
        from datasketch import MinHash, MinHashLSH
        categories = [products[asin].get('categories', []) for asin in products]
        lsh = MinHashLSH(
            threshold=kwargs.get('threshold', 0.5),
            num_perm=kwargs.get('num_perm', 128)
        )
        for idx, cats in enumerate(categories):
            m = MinHash(num_perm=kwargs.get('num_perm', 128))
            for c in cats:
                m.update(c.encode('utf8'))
            lsh.insert(idx, m)
        return G, lsh
    else:
        raise ValueError(f"Unknown embedding method: {method!r}")

    # normalize projected embeddings
    Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)

    # assign embeddings back to graph
    for idx, coord in zip(node_ids, Y):
        G.nodes[idx]['coords'] = coord

    return G


if __name__ == '__main__':
    products = json.load(open('amazon_metadata_test/parsed_amazon_meta.json'))
    G = embed_products(
        products,
        method='categories',
        subsampled_classes=['DVD', 'Video']
    )
    # Convert numpy array coordinates to string format that GML can handle
    for node in G.nodes():
        if 'coords' in G.nodes[node]:
            coords = G.nodes[node]['coords']
            # Convert numpy array to a simple string representation
            if isinstance(coords, np.ndarray):
                G.nodes[node]['coords'] = str(coords.tolist())
    print("Number of nodes:", G.number_of_nodes())
    print("Sample node coords:", G.nodes[0]['coords'])
    print("Sample edge dist:", next(iter(G.edges(data=True))) )
    nx.write_gml(G, 'amazon_metadata_test/amazon_categories_cosine.gml')

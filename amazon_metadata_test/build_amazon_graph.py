import networkx as nx
import numpy as np
import scipy.sparse as sp

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import TruncatedSVD
import umap

# Optional: Dask-ML for parallel SVD
try:
    import dask.array as da
    from dask_ml.decomposition import TruncatedSVD as DaskSVD
    _HAS_DASK = True
except ImportError:
    _HAS_DASK = False

# Optional: Landmark MDS (install via `pip install lmds`)
try:
    from lmds import LandmarkMDS
    _HAS_LMDS = True
except ImportError:
    _HAS_LMDS = False

# Optional: node2vec embeddings (install via `pip install nodevectors`)
try:
    from nodevectors import Node2Vec
    _HAS_NODE2VEC = True
except ImportError:
    _HAS_NODE2VEC = False

# Optional: MinHash LSH (install via `pip install datasketch`)
try:
    from datasketch import MinHash, MinHashLSH
    _HAS_MINHASH = True
except ImportError:
    _HAS_MINHASH = False


def _preprocess(products):
    """
    Build graph G, list of node IDs (0..n-1), and sparse category matrix X,
    skipping any product marked as discontinued. Each node is named by its
    numeric ID and has 'asin' and optionally 'comm' attributes.
    """
    G = nx.Graph()

    # 1. Filter out discontinued products
    valid = {
        asin: info
        for asin, info in products.items()
        if not (
            info.get('discontinued', False)
            or info.get('status', '').lower() == 'discontinued product'
        )
    }

    # 2. Assign numeric IDs and add nodes
    asins = list(valid.keys())
    node_ids = list(range(len(asins)))
    for idx, asin in zip(node_ids, asins):
        info = valid[asin]
        attrs = {'asin': asin}
        if info.get('group') is not None:
            attrs['comm'] = info['group']
        G.add_node(idx, **attrs)

    # 3. Add edges between numeric IDs
    asin_to_id = {asin: idx for idx, asin in zip(node_ids, asins)}
    for asin, info in valid.items():
        u = asin_to_id[asin]
        for nbr in info.get('similar', []):
            if nbr in asin_to_id:
                v = asin_to_id[nbr]
                G.add_edge(u, v)

    # 4. Build the sparse category matrix
    categories = [valid[asin].get('categories', []) for asin in asins]
    mlb = MultiLabelBinarizer(sparse_output=True)
    X = mlb.fit_transform(categories)  # CSR matrix: n_nodes × n_categories

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


def embed_products(products,
                   method='svd',
                   embedding_dim=2,
                   random_state=None,
                   use_dask=False,
                   **kwargs):
    """
    Build G and compute embeddings via one of:
      - 'svd'            : Truncated SVD on attribute matrix
      - 'umap'           : UMAP with Jaccard metric
      - 'landmark_mds'   : Landmark MDS (needs lmds)
      - 'node2vec'       : Node2Vec on the graph (needs nodevectors)
      - 'minhash_lsh'    : MinHash + LSH index (returns LSH object)

    Returns:
      - G with node attribute 'coords' (and 'asin', 'comm') set to the embedding
      - lsh index if method == 'minhash_lsh'
    """
    G, node_ids, X = _preprocess(products)

    if method == 'svd':
        Y = _embed_svd(X, embedding_dim, random_state, use_dask)
    elif method == 'umap':
        Y = _embed_umap(X, embedding_dim, random_state)
    elif method == 'landmark_mds':
        Y = _embed_landmark_mds(X,
                                embedding_dim,
                                n_landmarks=kwargs.get('n_landmarks', 1000),
                                random_state=random_state)
    elif method == 'node2vec':
        Y = _embed_node2vec(G,
                            embedding_dim,
                            walklen=kwargs.get('walklen', 10),
                            epochs=kwargs.get('epochs', 1))
    elif method == 'minhash_lsh':
        categories = [products[asin].get('categories', []) for asin in products]
        lsh = _build_lsh(categories,
                         num_perm=kwargs.get('num_perm', 128),
                         threshold=kwargs.get('threshold', 0.5))
        return G, lsh
    else:
        raise ValueError(f"Unknown embedding method: {method!r}")

    # assign embeddings back to graph nodes by numeric ID
    for idx, coord in zip(node_ids, Y):
        G.nodes[idx]['coords'] = coord
    return G

import json
import networkx as nx
products = json.load(open('amazon_metadata_test/parsed_amazon_meta.json'))
# products: your dict of ASIN → {'group':…, 'similar':[…], 'categories':[…]}
G_embedded = embed_products(
    products,
    method='svd',              # or 'umap', 'landmark_mds', 'node2vec', 'minhash_lsh'
    embedding_dim=16,
    random_state=42,
    use_dask=True,             # only if Dask-ML is installed and you want parallel SVD
    n_landmarks=2000,          # for landmark_mds
    walklen=20, epochs=2,      # for node2vec
    num_perm=256, threshold=0.6  # for minhash_lsh
)
print("Number of nodes:", G_embedded.number_of_nodes())
print("Node names:", list(G_embedded.nodes())[:5])
# Convert numpy array coordinates to strings
for node in G_embedded.nodes():
    if 'coords' in G_embedded.nodes[node]:
        coords = G_embedded.nodes[node]['coords']
        G_embedded.nodes[node]['coords'] = ','.join(map(str, coords.tolist()))

nx.write_gml(G_embedded, 'amazon_metadata_test/amazon_graph.gml')
# Now each node G_embedded.nodes[asin]['coords'] is a length-16 unit vector.

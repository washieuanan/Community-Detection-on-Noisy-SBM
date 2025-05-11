import networkx as nx
import numpy as np

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

# try to import a SentenceTransformer for semantic text embeddings
try:
    from sentence_transformers import SentenceTransformer
    _HAS_SBT = True
except ImportError:
    _HAS_SBT = False


def _preprocess(products, subsampled_classes):
    """
    Same as before: build G, node_ids, and sparse X
    """
    G = nx.Graph()

    valid = {
        asin: info
        for asin, info in products.items()
        if info.get('group') in subsampled_classes
    }
    comm_mapping = {sc: i for i, sc in enumerate(subsampled_classes)}

    asins = list(valid.keys())
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

    # categories is list of lists of strings
    categories = [valid[asin].get('categories', []) for asin in asins]
    # we still return X in case you need it, though it's unused below
    mlb = MultiLabelBinarizer(sparse_output=True)
    X = mlb.fit_transform(categories)

    G.graph['subclasses'] = ','.join(subsampled_classes)
    return G, node_ids, X


def build_category_graph_semantic(
    products,
    subsampled_classes=['Book','DVD','Music','Video'],
    embed_model_name='all-MiniLM-L6-v2',
):
    """
    Build G where each node has:
      - 'coords': the mean semantic embedding of its category names

    Each edge (u,v) gets attributes:
      - 'cosine_dist': 1 – cosine_similarity(coords_u, coords_v)
      - 'dist':        Euclidean distance ||coords_u – coords_v||_2
    """
    if not _HAS_SBT:
        raise ImportError(
            "Please install sentence-transformers: pip install sentence-transformers"
        )
    # 1) Base graph + node_ids
    G, node_ids, _ = _preprocess(products, subsampled_classes)

    # 2) Load semantic embedder
    model = SentenceTransformer(embed_model_name)
    dim   = model.get_sentence_embedding_dimension()

    # cache embeddings for each unique category string
    cat_cache: dict[str, np.ndarray] = {}

    # 3) Compute and assign per-node coords
    for idx in node_ids:
        asin = G.nodes[idx]['asin']
        cats = products[asin].get('categories', [])
        if not cats:
            vec = np.zeros(dim)
        else:
            embeds = []
            for cat in cats:
                if cat not in cat_cache:
                    cat_cache[cat] = model.encode(cat, convert_to_numpy=True)
                embeds.append(cat_cache[cat])
            vec = np.mean(embeds, axis=0)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
        G.nodes[idx]['coords'] = vec

    # 4) Compute both distances on each edge
    for u, v in G.edges():
        cu = G.nodes[u]['coords']
        cv = G.nodes[v]['coords']

        # semantic distance
        sim = cosine_similarity(cu.reshape(1,-1), cv.reshape(1,-1))[0, 0]
        G.edges[u, v]['cosine_dist'] = 1.0 - sim

        # Euclidean (geometric) distance
        G.edges[u, v]['dist'] = np.linalg.norm(cu - cv)

    return G


if __name__ == "__main__":
    import json
    products = json.load(open("amazon_metadata_test/parsed_amazon_meta.json"))
    G = build_category_graph_semantic(
        products,
        subsampled_classes=['DVD','Video'],
        embed_model_name='all-MiniLM-L6-v2'
    )
print("Nodes:", G.number_of_nodes(), "Edges:", G.number_of_edges())
# Example inspect
print("Sample node coords shape:", G.nodes[0]['coords'].shape)
print("Sample edge dist:", next(iter(G.edges(data=True))))

# Convert all numeric values to strings before writing to GML
# Convert node coordinates
for node in G.nodes():
    if 'coords' in G.nodes[node]:
        coords = G.nodes[node]['coords']
        # Convert numpy array to comma-separated string
        G.nodes[node]['coords'] = ','.join(map(str, coords.tolist()))

# Convert edge attributes
for u, v, data in G.edges(data=True):
    if 'cosine_dist' in data:
        data['cosine_dist'] = str(data['cosine_dist'])
    if 'dist' in data:
        data['dist'] = str(data['dist'])

# Now write to GML
nx.write_gml(G, 'amazon_metadata_test/amazon_semantic_videoDVD.gml')

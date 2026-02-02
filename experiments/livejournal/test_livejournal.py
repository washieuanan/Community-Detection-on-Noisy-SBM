import gzip
import networkx as nx
from collections import Counter


def process_gml_top2_groups(
    in_gml_path: str,
    out_gml_path: str,
    *,
    group_attr: str = "label",   # node attribute that defines the "group"
    comm_attr: str = "comm",     # new attribute name to write
):
    """
    1) Keep only nodes in the top-2 largest groups (by node count), where groups are nodes
       with the same `group_attr` value.
    2) Convert to undirected.
    3) Rename `group_attr` -> `comm_attr` and ensure comm values are integers.
    """

    # IMPORTANT: read nodes by their unique `id` field, NOT by `label` (labels repeat across groups)
    G = nx.read_gml(in_gml_path, label="id")  # usually yields a DiGraph if directed 1

    # --- (1) pick top-2 groups by size ---
    group_vals = []
    for n, data in G.nodes(data=True):
        if group_attr not in data:
            raise KeyError(f"Node {n} missing '{group_attr}' attribute.")
        group_vals.append(data[group_attr])

    counts = Counter(group_vals)
    # top 2 groups by count (tie-breaker: stringified group value)
    top2 = [g for g, _ in sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))[:2]]

    keep_nodes = [n for n, data in G.nodes(data=True) if data[group_attr] in top2]
    H = G.subgraph(keep_nodes).copy()

    # --- (2) make undirected ---
    # If you only want edges that are mutual in both directions, use:
    # H = H.to_undirected(reciprocal=True)
    H = H.to_undirected()

    # --- (3) rename label->comm and convert to ints ---
    def to_int_or_map(vals):
        # Try direct int conversion first (e.g., "0", "1", 2, etc.)
        try:
            return {v: int(v) for v in vals}
        except Exception:
            # Otherwise map to 0..k-1 deterministically
            uniq = sorted(set(vals), key=lambda x: str(x))
            return {v: i for i, v in enumerate(uniq)}

    mapping = to_int_or_map([data[group_attr] for _, data in H.nodes(data=True)])

    for n, data in H.nodes(data=True):
        data[comm_attr] = int(mapping[data[group_attr]])
        # remove the old attribute
        if group_attr in data:
            del data[group_attr]

    # Write out
    nx.write_gml(H, out_gml_path)
    return H, top2


# Example usage:
H, kept_groups = process_gml_top2_groups("experiments/livejournal/livejournal.gml", "oexperiments/livejournal/livejournal_graph.gml")
print("Kept groups:", kept_groups)
print(H.number_of_nodes(), H.number_of_edges())

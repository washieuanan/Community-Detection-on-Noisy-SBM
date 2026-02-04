import networkx as nx
import gzip
from collections import defaultdict

def load_youtube_graph():
    """Load YouTube graph from edge list and community files."""
    G = nx.Graph()
    
    # Load edge list
    print("Loading edge list from com-youtube.ungraph.txt.gz...")
    with gzip.open("experiments/youtube/com-youtube.ungraph.txt.gz", "rt") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                u = int(parts[0])
                v = int(parts[1])
                G.add_edge(u, v)
    
    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Load community assignments
    print("Loading community assignments from com-youtube.top5000.cmty.txt.gz...")
    comm_to_nodes = defaultdict(list)
    
    with gzip.open("experiments/youtube/com-youtube.top5000.cmty.txt.gz", "rt") as f:
        comm_id = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            nodes = [int(x) for x in line.split()]
            for node in nodes:
                if node in G:  # Only assign if node exists in graph
                    comm_to_nodes[comm_id].append(node)
            comm_id += 1
    
    print(f"Loaded {comm_id} communities")
    
    # Set community attribute (assign to the first community encountered in the file)
    # For nodes in multiple communities, use the first one
    node_to_comm = {}
    for comm_id, nodes in comm_to_nodes.items():
        for node in nodes:
            if node not in node_to_comm:
                node_to_comm[node] = comm_id
    
    for node in G.nodes():
        if node in node_to_comm:
            G.nodes[node]["comm"] = node_to_comm[node]
        else:
            # Nodes not in any community get a special label
            G.nodes[node]["comm"] = -1
    
    return G, comm_to_nodes

def filter_to_top_communities(G, comm_to_nodes, top_k=2):
    """Filter graph to only nodes in the top K largest communities."""
    # Find top K communities by size
    comm_sizes = [(comm_id, len(nodes)) for comm_id, nodes in comm_to_nodes.items()]
    comm_sizes.sort(key=lambda x: -x[1])  # Sort descending by size
    
    print(f"\nCommunity sizes (top 10):")
    for comm_id, size in comm_sizes[:10]:
        print(f"  Community {comm_id}: {size} nodes")
    
    # Select top K communities
    top_comm_ids = [comm_id for comm_id, _ in comm_sizes[:top_k]]
    print(f"\nKeeping top {len(top_comm_ids)} communities: {top_comm_ids}")
    
    # Get all nodes in top communities
    nodes_to_keep = set()
    node_to_comm = {}  # Map node to its community label (0 or 1)
    
    # Process each top community
    for idx, comm_id in enumerate(top_comm_ids):
        comm_label = idx  # Will be 0, 1, etc.
        nodes_in_comm = comm_to_nodes[comm_id]
        print(f"  Community {comm_id} (will be labeled {comm_label}): {len(nodes_in_comm)} nodes")
        
        for node in nodes_in_comm:
            nodes_to_keep.add(node)
            # If node appears in multiple top communities, assign to the first one encountered
            if node not in node_to_comm:
                node_to_comm[node] = comm_label
    
    print(f"\nTotal unique nodes in top {len(top_comm_ids)} communities: {len(nodes_to_keep)}")
    
    # Show overlap if any
    if len(top_comm_ids) == 2:
        comm0_nodes = set(comm_to_nodes[top_comm_ids[0]])
        comm1_nodes = set(comm_to_nodes[top_comm_ids[1]])
        overlap = comm0_nodes & comm1_nodes
        only_comm0 = comm0_nodes - comm1_nodes
        only_comm1 = comm1_nodes - comm0_nodes
        print(f"\nOverlap analysis:")
        print(f"  Nodes only in community {top_comm_ids[0]} (label 0): {len(only_comm0)}")
        print(f"  Nodes only in community {top_comm_ids[1]} (label 1): {len(only_comm1)}")
        print(f"  Nodes in both communities (assigned to label 0): {len(overlap)}")
        print(f"  Total unique nodes: {len(nodes_to_keep)}")
        
        if len(overlap) == len(comm0_nodes) and len(overlap) == len(comm1_nodes):
            print(f"  WARNING: The two communities are identical!")
        elif len(overlap) > 0:
            print(f"  Note: {len(overlap)} nodes appear in both communities and are assigned to community 0")
    
    # Create subgraph (induced subgraph - includes all edges between selected nodes)
    G_filtered = G.subgraph(nodes_to_keep).copy()
    
    # Assign community labels
    for node in G_filtered.nodes():
        if node in node_to_comm:
            G_filtered.nodes[node]["comm"] = node_to_comm[node]
        else:
            # This shouldn't happen, but handle it
            print(f"Warning: Node {node} not found in node_to_comm. Removing from graph.")
            G_filtered.remove_node(node)
    
    print(f"\nFiltered graph: {G_filtered.number_of_nodes()} nodes, {G_filtered.number_of_edges()} edges")
    
    # Verify community distribution
    comm_counts = defaultdict(int)
    for node in G_filtered.nodes():
        comm_counts[G_filtered.nodes[node]["comm"]] += 1
    print(f"Community distribution after relabeling:")
    for comm_id in sorted(comm_counts.keys()):
        print(f"  Community {comm_id}: {comm_counts[comm_id]} nodes")
    
    return G_filtered

if __name__ == "__main__":
    # Load graph and communities
    G, comm_to_nodes = load_youtube_graph()
    
    # Filter to top 2 communities
    G_filtered = filter_to_top_communities(G, comm_to_nodes, top_k=2)
    
    # Ensure graph is undirected
    if G_filtered.is_directed():
        G_filtered = G_filtered.to_undirected()
    
    # Ensure all edges have weights (default to 1.0)
    for u, v, d in G_filtered.edges(data=True):
        if "weight" not in d:
            d["weight"] = 1.0
    
    # Save to GML
    output_file = "experiments/youtube/youtube.gml"
    print(f"\nSaving graph to {output_file}...")
    nx.write_gml(G_filtered, output_file)
    print(f"Saved: {G_filtered.number_of_nodes()} nodes, {G_filtered.number_of_edges()} edges")
    print(f"Graph is undirected: {not G_filtered.is_directed()}")

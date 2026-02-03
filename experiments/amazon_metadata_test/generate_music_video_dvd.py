"""
Generate amazon_hamming_musicDVD.gml and amazon_hamming_videoDVD.gml
from amazon_hamming.gml by filtering to the appropriate communities.
"""
import networkx as nx
import numpy as np
from collections import defaultdict

def generate_graph_from_hamming(source_file, target_file, comm_classes, class_names):
    """
    Generate a filtered graph from amazon_hamming.gml.
    
    Args:
        source_file: Path to amazon_hamming.gml
        target_file: Path to save the filtered graph
        comm_classes: List of comm values to keep (e.g., [1, 3] for Music and DVD)
        class_names: List of class names for mapping (e.g., ['Music', 'DVD'])
    """
    print(f"\n{'='*60}")
    print(f"Generating {target_file}")
    print(f"Filtering to communities: {comm_classes} ({class_names})")
    print(f"{'='*60}")
    
    # Load the full graph
    G = nx.read_gml(source_file)
    print(f"Loaded source graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Check community distribution in source
    comm_counts = defaultdict(int)
    for node, data in G.nodes(data=True):
        comm = data.get('comm', None)
        if comm is not None:
            comm_counts[comm] += 1
    
    print(f"\nCommunity distribution in source graph:")
    for comm in sorted(comm_counts.keys()):
        print(f"  Community {comm}: {comm_counts[comm]} nodes")
    
    # Filter nodes to only those in the target communities
    nodes_to_keep = []
    for node, data in G.nodes(data=True):
        comm = data.get('comm', None)
        if comm is not None and comm in comm_classes:
            nodes_to_keep.append(node)
    
    print(f"\nNodes to keep: {len(nodes_to_keep)}")
    
    # Create subgraph
    G_filtered = G.subgraph(nodes_to_keep).copy()
    print(f"Filtered graph: {G_filtered.number_of_nodes()} nodes, {G_filtered.number_of_edges()} edges")
    
    # Relabel communities to 0 and 1
    comm_mapping = {comm_classes[i]: i for i in range(len(comm_classes))}
    print(f"\nRelabeling communities: {comm_mapping}")
    for node in G_filtered.nodes():
        old_comm = G_filtered.nodes[node].get('comm')
        if old_comm in comm_mapping:
            G_filtered.nodes[node]['comm'] = comm_mapping[old_comm]
    
    # Verify relabeling
    comm_counts_filtered = defaultdict(int)
    for node, data in G_filtered.nodes(data=True):
        comm = data.get('comm', None)
        if comm is not None:
            comm_counts_filtered[comm] += 1
    
    print(f"\nCommunity distribution after relabeling:")
    for comm in sorted(comm_counts_filtered.keys()):
        print(f"  Community {comm}: {comm_counts_filtered[comm]} nodes")
    
    # Ensure coords are in string format (if they're arrays)
    for node in G_filtered.nodes():
        if 'coords' in G_filtered.nodes[node]:
            coords = G_filtered.nodes[node]['coords']
            if isinstance(coords, (np.ndarray, list)):
                G_filtered.nodes[node]['coords'] = ','.join(map(str, coords))
    
    # Save the filtered graph
    nx.write_gml(G_filtered, target_file)
    print(f"\nSaved to {target_file}")
    print(f"Final graph: {G_filtered.number_of_nodes()} nodes, {G_filtered.number_of_edges()} edges")
    print(f"Graph is connected: {nx.is_connected(G_filtered)}")

if __name__ == "__main__":
    source_file = "amazon_metadata_test/amazon_hamming.gml"
    
    # First, check what communities exist in the source graph
    print("Checking source graph structure...")
    G_check = nx.read_gml(source_file)
    
    # Get all unique comm values
    comm_values = set()
    for node, data in G_check.nodes(data=True):
        comm = data.get('comm', None)
        if comm is not None:
            comm_values.add(comm)
    
    print(f"Found communities in source: {sorted(comm_values)}")
    
    # Count nodes per community
    comm_counts = defaultdict(int)
    for node, data in G_check.nodes(data=True):
        comm = data.get('comm', None)
        if comm is not None:
            comm_counts[comm] += 1
    
    print(f"\nNodes per community:")
    for comm in sorted(comm_counts.keys()):
        print(f"  Community {comm}: {comm_counts[comm]} nodes")
    
    # Check existing graphs to infer the mapping
    # bookDVD should have Book and DVD
    # musicvideo should have Music and Video
    # musicbook should have Music and Book
    # bookvideo should have Book and Video
    
    # Infer mapping by checking which comm values appear in which existing graphs
    comm_to_classes = defaultdict(set)
    
    # Check bookDVD (should have Book=0, DVD=1)
    try:
        G_bd = nx.read_gml("amazon_metadata_test/amazon_hamming_bookDVD.gml")
        bd_comms = set()
        for node, data in G_bd.nodes(data=True):
            comm = data.get('comm', None)
            if comm is not None:
                bd_comms.add(comm)
        print(f"\nbookDVD graph has communities: {sorted(bd_comms)}")
        # Assume these are Book and DVD
        if len(bd_comms) == 2:
            bd_list = sorted(bd_comms)
            comm_to_classes[bd_list[0]].add('Book')
            comm_to_classes[bd_list[1]].add('DVD')
    except Exception as e:
        print(f"Could not load bookDVD: {e}")
    
    # Check musicvideo (should have Music=0, Video=1)
    try:
        G_mv = nx.read_gml("amazon_metadata_test/amazon_hamming_musicvideo.gml")
        mv_comms = set()
        for node, data in G_mv.nodes(data=True):
            comm = data.get('comm', None)
            if comm is not None:
                mv_comms.add(comm)
        print(f"musicvideo graph has communities: {sorted(mv_comms)}")
        # These should be Music and Video
        if len(mv_comms) == 2:
            mv_list = sorted(mv_comms)
            comm_to_classes[mv_list[0]].add('Music')
            comm_to_classes[mv_list[1]].add('Video')
    except Exception as e:
        print(f"Could not load musicvideo: {e}")
    
    # Based on build_hamming_amazon.py pattern:
    # If amazon_hamming.gml was built with ['Book', 'DVD', 'Music', 'Video']
    # Then: 0=Book, 1=DVD, 2=Music, 3=Video
    
    # For musicDVD: need Music (2) and DVD (1)
    # For videoDVD: need Video (3) and DVD (1)
    
    if len(comm_values) == 4:
        # Standard mapping: 0=Book, 1=DVD, 2=Music, 3=Video
        print("\nUsing standard mapping: 0=Book, 1=DVD, 2=Music, 3=Video")
        
        # Generate musicDVD: Music (2) and DVD (1)
        print("\n" + "="*60)
        generate_graph_from_hamming(
            source_file,
            "amazon_metadata_test/amazon_hamming_musicDVD.gml",
            comm_classes=[2, 1],  # Music and DVD
            class_names=['Music', 'DVD']
        )
        
        # Generate videoDVD: Video (3) and DVD (1)
        print("\n" + "="*60)
        generate_graph_from_hamming(
            source_file,
            "amazon_metadata_test/amazon_hamming_videoDVD.gml",
            comm_classes=[3, 1],  # Video and DVD
            class_names=['Video', 'DVD']
        )
        
        print("\n" + "="*60)
        print("Generation complete!")
        print("="*60)
    else:
        print(f"\nWarning: Expected 4 communities but found {len(comm_values)}")
        print("Please check the community mapping manually.")
        print("You may need to adjust the comm_classes in the script.")

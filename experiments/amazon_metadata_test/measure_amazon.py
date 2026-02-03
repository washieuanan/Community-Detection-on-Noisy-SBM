import networkx as nx
import numpy as np
from collections import defaultdict

def analyze_graph(filepath):
    """Load a graph and analyze its communities and edges."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {filepath}")
    print(f"{'='*60}")
    
    try:
        G = nx.read_gml(filepath)
        
        # Get number of edges
        num_edges = G.number_of_edges()
        num_nodes = G.number_of_nodes()
        
        print(f"Total nodes: {num_nodes}")
        print(f"Total edges: {num_edges}")
        
        # Get community information
        if 'comm' in list(G.nodes(data=True))[0][1] if G.nodes() else {}:
            # Count nodes per community
            comm_counts = defaultdict(int)
            for node, data in G.nodes(data=True):
                comm = data.get('comm', None)
                if comm is not None:
                    comm_counts[comm] += 1
            
            if comm_counts:
                print(f"\nNodes per community:")
                for comm_id in sorted(comm_counts.keys()):
                    print(f"  Community {comm_id}: {comm_counts[comm_id]} nodes")
                print(f"Total communities: {len(comm_counts)}")
            else:
                print("\nNo community information found in graph.")
        else:
            print("\nNo 'comm' attribute found in graph nodes.")
            
    except FileNotFoundError:
        print(f"ERROR: File not found: {filepath}")
    except Exception as e:
        print(f"ERROR loading {filepath}: {e}")

if __name__ == "__main__":
    # List of files to analyze (from amazon_metadata_test folder, not experiments)
    files = [
        "amazon_metadata_test/amazon_hamming_bookDVD.gml",
        "amazon_metadata_test/amazon_hamming_musicvideo.gml",
        "amazon_metadata_test/amazon_hamming_musicbook.gml",
        "amazon_metadata_test/amazon_hamming_bookvideo.gml",
    ]
    
    for filepath in files:
        analyze_graph(filepath)
    
    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"{'='*60}")

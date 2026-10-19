import json
import networkx as nx
import matplotlib.pyplot as plt

def build_networkx_graph(json_file="extracted_graph.json"):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # MultiDiGraph supports multiple directed edges and custom attributes
    G = nx.MultiDiGraph()

    for entry in data:
        # Add nodes with their type and metadata
        for node in entry["nodes"]:
            G.add_node(node["id"], type=node["type"], **node.get("properties", {}))
            
        # Add directed edges with relation types
        for rel in entry["relationships"]:
            G.add_edge(
                rel["source"], 
                rel["target"], 
                relationship=rel["type"], 
                **rel.get("properties", {})
            )
            
    return G

def main():
    G = build_networkx_graph()
    
    print("--- NetworkX Graph Storage Verification ---")
    print(f"Total Nodes: {G.number_of_nodes()}")
    print(f"Total Directed Edges: {G.number_of_edges()}")
    
    # Save graph to standard node-link JSON format
    graph_data = nx.node_link_data(G, edges="edges")
    with open("networkx_graph.json", "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)
    print("Persisted graph locally to 'networkx_graph.json'.")
    
    # Graph traversal / RAG subgraph lookup example
    target_scheme = "Training To Farmers"
    print(f"\nNeighbors & outgoing connections for '{target_scheme}':")
    if G.has_node(target_scheme):
        for successor in G.successors(target_scheme):
            edge_data = G.get_edge_data(target_scheme, successor)
            rel_types = [d.get("relationship") for d in edge_data.values()]
            node_type = G.nodes[successor].get("type", "Unknown")
            print(f"  -[:{', '.join(rel_types)}]-> [{node_type}] {successor}")

if __name__ == "__main__":
    main()
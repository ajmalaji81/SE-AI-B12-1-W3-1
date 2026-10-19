import os
import json
from dotenv import load_dotenv
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document
from langchain_neo4j import Neo4jGraph

load_dotenv()

uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME", "neo4j")
password = os.getenv("NEO4J_PASSWORD")
DATABASE_NAME = "52b0ac8e"

def load_graph_documents(json_file="extracted_graph.json"):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    graph_documents = []
    for entry in data:
        source_doc = Document(page_content="", metadata=entry.get("source", {}))
        
        nodes_dict = {}
        for n in entry["nodes"]:
            nodes_dict[n["id"]] = Node(
                id=n["id"], 
                type=n["type"], 
                properties=n.get("properties", {})
            )
            
        relationships = []
        for r in entry["relationships"]:
            src_node = nodes_dict.get(r["source"])
            tgt_node = nodes_dict.get(r["target"])
            if src_node and tgt_node:
                relationships.append(
                    Relationship(
                        source=src_node,
                        target=tgt_node,
                        type=r["type"],
                        properties=r.get("properties", {})
                    )
                )
                
        graph_documents.append(
            GraphDocument(
                nodes=list(nodes_dict.values()),
                relationships=relationships,
                source=source_doc
            )
        )
    return graph_documents

def main():
    print(f"Connecting to Neo4j Aura database '{DATABASE_NAME}'...")
    
    graph = Neo4jGraph(
        url=uri,
        username=username,
        password=password,
        database=DATABASE_NAME,
        refresh_schema=False
    )
    
    # Ingest data
    graph_docs = load_graph_documents()
    print(f"Writing {len(graph_docs)} GraphDocument batch(es) to Neo4j Aura...")
    graph.add_graph_documents(graph_docs, baseEntityLabel=True)
    
    # Verification queries
    print("\n--- Neo4j Storage Verification ---")
    node_count = graph.query("MATCH (n) RETURN count(n) AS total_nodes")
    rel_count = graph.query("MATCH ()-[r]->() RETURN count(r) AS total_relationships")
    
    print(f"Total Nodes: {node_count[0]['total_nodes']}")
    print(f"Total Relationships: {rel_count[0]['total_relationships']}")

    sample_query = """
    MATCH (s:Scheme)-[r]->(target)
    RETURN s.id AS scheme, type(r) AS relationship, labels(target)[0] AS target_type, target.id AS target
    LIMIT 5
    """
    sample_rows = graph.query(sample_query)
    print("\nSample Relationships in Aura:")
    for row in sample_rows:
        print(f"  ({row['scheme']}) -[:{row['relationship']}]-> ({row['target_type']}: {row['target']})")

if __name__ == "__main__":
    main()
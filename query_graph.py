import os
import json
import networkx as nx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

try:
    from langchain_neo4j import Neo4jGraph
    from langchain_neo4j.chains.graph_qa.cypher import GraphCypherQAChain
except ImportError:
    from langchain_community.graphs import Neo4jGraph
    from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain

load_dotenv()

DATABASE_NAME = "52b0ac8e"
URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

CYPHER_GENERATION_TEMPLATE = """Task: Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema:
{schema}

Note:
1. Node IDs may have variable casing. ALWAYS use case-insensitive matching for string values:
   Example: WHERE toLower(s.id) CONTAINS toLower('training to farmers')
2. Keep Cypher queries simple and return specific properties or IDs.

Question: {question}
Cypher Query:"""

CYPHER_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], 
    template=CYPHER_GENERATION_TEMPLATE
)

def setup_neo4j_qa_chain():
    graph = Neo4jGraph(
        url=URI,
        username=USERNAME,
        password=PASSWORD,
        database=DATABASE_NAME,
        refresh_schema=True
    )
    
    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        cypher_prompt=CYPHER_PROMPT,
        verbose=True,
        allow_dangerous_requests=True,
        return_direct=False
    )
    return chain

def load_networkx_graph(path="networkx_graph.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return nx.node_link_graph(data, edges="edges")

def query_networkx_subgraph(graph: nx.MultiDiGraph, entity_name: str, question: str):
    matched_nodes = [n for n in graph.nodes if entity_name.lower() in str(n).lower()]
    
    if not matched_nodes:
        return f"No entity matching '{entity_name}' found in NetworkX graph."
    
    context_lines = []
    for node in matched_nodes:
        node_type = graph.nodes[node].get("type", "Entity")
        context_lines.append(f"Entity: {node} (Type: {node_type})")
        
        for succ in graph.successors(node):
            edge_dict = graph.get_edge_data(node, succ)
            for _, edge in edge_dict.items():
                rel = edge.get("relationship", "RELATED_TO")
                succ_type = graph.nodes[succ].get("type", "Entity")
                context_lines.append(f"  - ({node}) -[:{rel}]-> ({succ} : {succ_type})")
                
        for pred in graph.predecessors(node):
            edge_dict = graph.get_edge_data(pred, node)
            for _, edge in edge_dict.items():
                rel = edge.get("relationship", "RELATED_TO")
                pred_type = graph.nodes[pred].get("type", "Entity")
                context_lines.append(f"  - ({pred} : {pred_type}) -[:{rel}]-> ({node})")
                
    structured_context = "\n".join(context_lines)
    
    prompt = PromptTemplate.from_template(
        """You are a knowledgeable assistant answering questions based on an extracted Knowledge Graph.
Use the structured graph facts below to answer the user query accurately.

Structured Graph Context:
{context}

User Question: {question}

Answer:"""
    )
    
    rag_chain = prompt | llm | StrOutputParser()
    return rag_chain.invoke({"context": structured_context, "question": question})

def main():
    test_question = "What are the benefits and targeted beneficiaries for Training to Farmers?"
    
    print("\n================ Neo4j Cypher QA Result ================")
    neo4j_chain = setup_neo4j_qa_chain()
    neo4j_response = neo4j_chain.invoke({"query": test_question})
    print("\nFinal Answer (Neo4j):", neo4j_response["result"])
    
    print("\n================ NetworkX Subgraph QA Result ================")
    nx_graph = load_networkx_graph()
    nx_response = query_networkx_subgraph(
        graph=nx_graph,
        entity_name="Training to Farmers",
        question=test_question
    )
    print("\nFinal Answer (NetworkX):", nx_response)

if __name__ == "__main__":
    main()

CYPHER_GENERATION_TEMPLATE = """Task: Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Schema:
{schema}

Critical Syntax Rules:
1. NEVER put a MATCH statement inside parentheses like (MATCH ...) or inside list comprehensions.
2. For collections, use simple MATCH ... RETURN collect(n.id) OR pattern comprehension:
   Example: [(s)-[:TARGETS]->(bn:Beneficiary) | bn.id]
3. Node IDs may have variable casing. Use case-insensitive matching:
   Example: WHERE toLower(s.id) CONTAINS toLower('training to farmers')
4. Keep the query simple and direct.

Question: {question}
Cypher Query:"""
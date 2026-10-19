import os
import json
import time
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

allowed_nodes = [
    "Scheme", 
    "Department", 
    "Beneficiary", 
    "EligibilityCriteria", 
    "Benefit", 
    "DocumentRequired"
]

allowed_relationships = [
    "OFFERED_BY", 
    "TARGETS", 
    "REQUIRES_ELIGIBILITY", 
    "PROVIDES_BENEFIT", 
    "REQUIRES_DOCUMENT"
]

transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=allowed_nodes,
    allowed_relationships=allowed_relationships,
    strict_mode=True
)

def run_extraction():
    if not os.path.exists("schemes_documents.json"):
        print("Error: 'schemes_documents.json' not found. Run scrape_schemes.py first.")
        return

    with open("schemes_documents.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        
    documents = [
        Document(page_content=item["page_content"], metadata=item["metadata"])
        for item in raw_data
    ]
    
    total = len(documents)
    print(f"Loaded all {total} schemes. Beginning batched graph extraction...")
    
    batch_size = 5
    all_serialized = []
    
    for i in range(0, total, batch_size):
        batch = documents[i:i + batch_size]
        print(f"\nProcessing batch {i // batch_size + 1} of {(total + batch_size - 1) // batch_size} ({len(batch)} schemes)...")
        
        try:
            graph_docs = transformer.convert_to_graph_documents(batch)
            for g_doc in graph_docs:
                all_serialized.append({
                    "source": g_doc.source.metadata,
                    "nodes": [{"id": n.id, "type": n.type, "properties": n.properties} for n in g_doc.nodes],
                    "relationships": [
                        {
                            "source": r.source.id,
                            "target": r.target.id,
                            "type": r.type,
                            "properties": r.properties
                        }
                        for r in g_doc.relationships
                    ]
                })
            print(f"  Successfully extracted {len(graph_docs)} graph documents. Total so far: {len(all_serialized)}")
            time.sleep(1)
        except Exception as e:
            print(f"  Error on batch {i}: {e}")

    with open("extracted_graph.json", "w", encoding="utf-8") as f:
        json.dump(all_serialized, f, indent=2, ensure_ascii=False)
        
    print(f"\nDone! Extracted graph structure for all {len(all_serialized)} schemes into 'extracted_graph.json'.")

if __name__ == "__main__":
    run_extraction()
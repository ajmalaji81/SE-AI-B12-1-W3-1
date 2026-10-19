# 🌾 Tamil Nadu Welfare Schemes — Hybrid GraphRAG

An enterprise-grade **Knowledge Graph Retrieval-Augmented Generation (GraphRAG)** system designed to ingest, structure, visualize, and query Tamil Nadu Government welfare schemes (Agriculture and Welfare departments). 

The platform implements a **Hybrid RAG** architecture combining semantic dense vector search (FAISS) with structured knowledge graph traversal (Neo4j AuraDB & NetworkX), reinforced by dual-rail safety guardrails and automated DeepEval evaluation.

---

## 🌟 Key Features

* **Hybrid Retrieval Architecture:** Merges unstructured semantic retrieval (FAISS vector store) with multi-hop structured relationships (Neo4j AuraDB / NetworkX) to prevent entity-resolution mismatches.
* **Automated Knowledge Graph Extraction:** Transforms raw government scheme documentation into typed nodes and directional edges using LangChain's `LLMGraphTransformer` and OpenAI's `gpt-4o-mini`.
* **Dual Graph Storage Engine:**
  * **Neo4j AuraDB:** Cloud-native graph database executing dynamic, schema-aware Cypher queries.
  * **NetworkX:** High-performance local in-memory multigraph enabling zero-cloud offline graph traversals.
* **Dual-Rail Safety Guardrails:**
  * **Input Rail:** Screens incoming prompts against out-of-domain topics, adversarial injections, and jailbreaks.
  * **Output Rail:** Enforces strict graph-grounding to eliminate hallucinations.
* **DeepEval Benchmarking Suite:** Automated quantitative LLM-as-a-judge evaluations achieving **1.00 Faithfulness** and **1.00 Answer Relevancy**.
* **Full-Featured Streamlit Interface:** Interactive dashboard featuring 2D physics-based network topology visualizers (PyVis), multi-engine query execution, and one-click portal re-ingestion.
* **Observability & Telemetry:** Full tracing of tokens, prompt templates, and execution latency via **LangSmith**.

---
## 🏛️ System Architecture

```text
                    ┌─────────────────────────┐
                    │       User Query        │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │     Input Guardrail     │ ──► [BLOCKED: Off-Topic / Jailbreak]
                    └────────────┬────────────┘
                                 │ (ALLOWED)
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
     [ Vector Search: FAISS ]        [ Graph Engine: Neo4j / NetworkX ]
     Dense Semantic Retrieval        Structured Triples & Relationships
     • Descriptive paragraphs        • (:Scheme)-[:PROVIDES_BENEFIT]->(:Benefit)
     • Eligibility & application text • (:Scheme)-[:TARGETS]->(:Beneficiary)
                 │                               │
                 └───────────────┬───────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │   LLM Synthesis Engine  │
                    │      (gpt-4o-mini)      │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │    Output Guardrail     │ ──► Enforces Graph Grounding
                    └────────────┬────────────┘
                                 │
                                 ▼
                     Structured Bulleted Answer
                     (DeepEval Verified: 1.00)
```

## 📊 Graph Ontology

### Node Labels
* `Scheme`: Name of the welfare scheme (e.g., *Training to Farmers*).
* `Department`: Sponsoring governmental agency.
* `Beneficiary`: Targeted demographic or farming collective.
* `Benefit`: Financial grants, subsidies, or technical services.
* `EligibilityCriteria`: Land holding norms, regional parameters, or community criteria.
* `DocumentRequired`: Identity documents, revenue land records, or sanctioning authorities.

### Relationship Types
* `(:Scheme)-[:OFFERED_BY]->(:Department)`
* `(:Scheme)-[:TARGETS]->(:Beneficiary)`
* `(:Scheme)-[:PROVIDES_BENEFIT]->(:Benefit)`
* `(:Scheme)-[:REQUIRES_ELIGIBILITY]->(:EligibilityCriteria)`
* `(:Scheme)-[:REQUIRES_DOCUMENT]->(:DocumentRequired)`

---

## 🛠️ Tech Stack

* **Language & Runtime:** Python 3.10+
* **LLM & Embeddings:** OpenAI `gpt-4o-mini`, `text-embedding-3-small`
* **Orchestration:** LangChain, LangChain-Community, LangChain-Neo4j, LangChain-Experimental
* **Databases:** Neo4j AuraDB (Cloud Cypher Engine), FAISS (Vector Index), NetworkX (Local Graph)
* **Web Scraping:** BeautifulSoup4, Requests
* **Evaluation & Testing:** DeepEval, Pytest
* **Tracing:** LangSmith
* **Interface & Visualization:** Streamlit, PyVis

---

## 📂 Project Structure

```text
├── .env                        # Environment credentials (API Keys & DB URIs)
├── app.py                      # Main Streamlit dashboard application
├── scrape_schemes.py           # Web scraper targeting TN government scheme portal
├── extract_graph.py            # Batched graph transformation pipeline
├── store_neo4j.py              # Ingestion pipeline for Neo4j AuraDB
├── store_networkx.py           # NetworkX builder and JSON persistence
├── build_hybrid_index.py       # FAISS vector store chunking and embedding
├── query_graph.py              # Cypher QA and Subgraph retrieval logic
├── rag_guardrails.py           # Standalone input/output guardrail execution
├── evaluate_rag.py             # Quantitative DeepEval test suite
├── schemes_documents.json      # Cached raw scraped scheme documents
├── extracted_graph.json        # Serialized graph nodes and edges
└── networkx_graph.json         # Persisted NetworkX node-link structure

## ⚙️ Data Pipeline Execution

# 1. Scrape all schemes from the government portal
python scrape_schemes.py

# 2. Extract graph entities and relationships via LLM
python extract_graph.py

# 3. Ingest into Neo4j AuraDB and NetworkX
python store_neo4j.py
python store_networkx.py

# 4. Generate local FAISS vector embeddings
python build_hybrid_index.py

## 🧪 Testing & Evaluation

python evaluate_rag.py

## 🖥️ Running the Streamlit App

streamlit run app.py

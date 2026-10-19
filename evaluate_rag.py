import os
from dotenv import load_dotenv
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from query_graph import load_networkx_graph, query_networkx_subgraph

load_dotenv()

def run_evaluation():
    print("================ Initializing DeepEval Benchmarking ================")
    
    # 1. Define query and target scheme
    user_query = "What are the benefits and targeted beneficiaries for Training to Farmers?"
    target_scheme = "Training to Farmers"
    
    # 2. Extract facts directly from the Knowledge Graph
    nx_graph = load_networkx_graph()
    
    # Extract the actual 1-hop neighborhood facts stored in NetworkX as ground-truth retrieval context
    retrieval_context = []
    if nx_graph.has_node("Training To Farmers"):
        for succ in nx_graph.successors("Training To Farmers"):
            edge_data = nx_graph.get_edge_data("Training To Farmers", succ)
            for _, edge in edge_data.items():
                rel = edge.get("relationship", "RELATED_TO")
                retrieval_context.append(f"Training To Farmers {rel} {succ}")
    
    print(f"\nGround Truth Graph Context ({len(retrieval_context)} facts retrieved):")
    for fact in retrieval_context[:5]:
        print(f"  * {fact}")

    # 3. Generate response through the GraphRAG pipeline
    actual_output = query_networkx_subgraph(
        graph=nx_graph,
        entity_name=target_scheme,
        question=user_query
    )
    
    print(f"\nUser Query: {user_query}")
    print(f"Pipeline Actual Output:\n{actual_output}\n")
    
    # 4. Construct the DeepEval LLMTestCase
    test_case = LLMTestCase(
        input=user_query,
        actual_output=actual_output,
        retrieval_context=retrieval_context
    )
    
    # 5. Initialize metrics using gpt-4o-mini as evaluator
    faithfulness_metric = FaithfulnessMetric(threshold=0.7, model="gpt-4o-mini")
    relevancy_metric = AnswerRelevancyMetric(threshold=0.7, model="gpt-4o-mini")
    
    # 6. Run evaluations
    print("Evaluating Faithfulness (Hallucination Detection)...")
    faithfulness_metric.measure(test_case)
    print(f"  Score: {faithfulness_metric.score:.2f} (Threshold: 0.70)")
    print(f"  Status: {'PASSED' if faithfulness_metric.is_successful() else 'FAILED'}")
    print(f"  Reasoning: {faithfulness_metric.reason}\n")
    
    print("Evaluating Answer Relevancy...")
    relevancy_metric.measure(test_case)
    print(f"  Score: {relevancy_metric.score:.2f} (Threshold: 0.70)")
    print(f"  Status: {'PASSED' if relevancy_metric.is_successful() else 'FAILED'}")
    print(f"  Reasoning: {relevancy_metric.reason}\n")
    
    print("================ DeepEval Benchmark Complete ================")

if __name__ == "__main__":
    run_evaluation()
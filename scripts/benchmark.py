import os
import sys
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config_loader import load_yaml_config
from src.embeddings.embedding_model import LocalEmbeddingModel
from src.retrieval.vector_store import LocalFAISSVectorStore
from src.retrieval.bm25_retriever import LocalBM25Retriever
from src.retrieval.reranker import LocalReranker
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.llm import UnifiedLLM
from src.pipeline.rag_pipeline import MedicalRAGPipeline

def main():
    settings = load_yaml_config("config/settings.yaml")
    models_cfg = load_yaml_config("config/models.yaml")

    print("Loading FAISS + BM25 Indexes...")
    emb_model = LocalEmbeddingModel(models_cfg["embedding"]["model_id"])
    vector_store = LocalFAISSVectorStore()
    vector_store.load(settings["paths"]["index_dir"])
    
    bm25 = LocalBM25Retriever()
    bm25.load(settings["paths"]["index_dir"])

    print("Loading Local Reranker...")
    reranker = LocalReranker(models_cfg["reranker"]["model_id"])
    hybrid = HybridRetriever(vector_store, emb_model, bm25, reranker)

    print("Initializing LLM Engine (Groq GPT-OSS-120B)...")
    llm = UnifiedLLM(provider=models_cfg["llm"]["provider"], model_id=models_cfg["llm"]["model_id"])

    pipeline = MedicalRAGPipeline(hybrid, llm, settings)

    query = "What are the recommendations and blood glucose targets for type 2 diabetes management according to NICE guidelines?"
    print(f"\nQuery: {query}")
    print("Running Pipeline...")

    response = pipeline.answer_query(query)

    print(f"\n=== PIPELINE STATUS: {response['status']} ===")
    print(f"=== CONFIDENCE SCORE: {response['confidence']:.2f} ===")
    print(f"=== RESPONSE LATENCY: {response['latency_sec']}s ===")
    print(f"\n=== SYNTHESIZED MEDICAL ANSWER ===\n{response['answer']}")
    print("\n=== VERIFIED CITATION SOURCES ===")
    for src in response["sources"]:
        print(f"[{src['citation_id']}] {src['source_file']} (Page {src['page']}) - Relevance: {src['relevance_score']}")

if __name__ == "__main__":
    main()

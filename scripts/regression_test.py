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

def run_test(pipeline, query, label):
    print(f"\n=======================================================")
    print(f"REGRESSION TEST: {label}")
    print(f"Query: '{query}'")
    print(f"=======================================================")
    res = pipeline.answer_query(query, return_debug=True)
    cited_count = res.get("citations_verified_count", 0)
    sources = res.get("sources", [])
    print(f"Status: {res.get('status')}")
    print(f"Evidence Quality: {res.get('evidence_quality')} (Confidence: {res.get('retrieval_relevance', res.get('retrieval_score'))})")
    print(f"Latency: {res.get('latency_sec')}s")
    print(f"Verified Citations: {cited_count} / {len(sources)}")
    print(f"Answer Output:\n{res.get('answer')}")
    if sources:
        print("\nSources breakdown:")
        for s in sources:
            print(f"  [{s.get('citation_id')}] {s.get('source_file')} p.{s.get('page')} | cited={s.get('is_referenced_in_text')} | grounded={s.get('is_claim_grounded')}")

def main():
    settings = load_yaml_config("config/settings.yaml")
    models_cfg = load_yaml_config("config/models.yaml")

    emb_model = LocalEmbeddingModel(models_cfg["embedding"]["model_id"])
    vector_store = LocalFAISSVectorStore()
    vector_store.load(settings["paths"]["index_dir"])
    
    bm25 = LocalBM25Retriever()
    bm25.load(settings["paths"]["index_dir"])

    reranker = LocalReranker(models_cfg["reranker"]["model_id"])
    hybrid = HybridRetriever(vector_store, emb_model, bm25, reranker)
    llm = UnifiedLLM(provider=models_cfg["llm"]["provider"], model_id=models_cfg["llm"]["model_id"])
    pipeline = MedicalRAGPipeline(hybrid, llm, settings)

    # Test 1: Arabic Relevant (User consultation)
    run_test(pipeline, "متى يُوصى بإضافة الإنسولين إلى خطة علاج مريض السكري من النوع الثاني؟", "1. ARABIC RELEVANT QUERY")

    # Test 2: English Relevant
    run_test(pipeline, "When should insulin be added to the treatment regimen for a diabetic patient?", "2. ENGLISH RELEVANT QUERY")

    # Test 3: Irrelevant Query (Must trigger pure abstention)
    run_test(pipeline, "What is the surgical protocol for acute appendicitis?", "3. IRRELEVANT QUERY (PURE ABSTENTION)")

    # Test 4: Emergency Alert Intercept
    run_test(pipeline, "مريض سكري يعاني من فقدان للوعي وغيبوبة حادة", "4. EMERGENCY INTERCEPT")

if __name__ == "__main__":
    main()


import os
import sys
from dotenv import load_dotenv

load_dotenv()
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

settings = load_yaml_config("config/settings.yaml")
models_cfg = load_yaml_config("config/models.yaml")

emb_model = LocalEmbeddingModel(models_cfg["embedding"]["model_id"])
vector_store = LocalFAISSVectorStore()
bm25 = LocalBM25Retriever()
reranker = LocalReranker(models_cfg["reranker"]["model_id"])

idx_dir = settings["paths"]["index_dir"]
vector_store.load(idx_dir)
bm25.load(idx_dir)

llm = UnifiedLLM(provider=models_cfg["llm"]["provider"], model_id=models_cfg["llm"]["model_id"])
hybrid = HybridRetriever(vector_store, emb_model, bm25, reranker)
pipeline = MedicalRAGPipeline(hybrid, llm, settings)

query = "متى يُوصى بإضافة الإنسولين إلى خطة علاج مريض السكري من النوع الثاني؟"
print(f"Running pipeline for query: {query}", flush=True)
res = pipeline.answer_query(query, return_debug=True)

print("=" * 60, flush=True)
print("QUERY:", query, flush=True)
print("STATUS:", res.get("status"), flush=True)
print("LATENCY:", res.get("latency_sec"), flush=True)
print("TRANSLATED QUERY:", res.get("debug", {}).get("retrieval_query"), flush=True)
print("=" * 60, flush=True)

print("\n" + "=" * 60, flush=True)
print("FINAL GENERATED ANSWER:", flush=True)
print("=" * 60, flush=True)
print(res.get("answer"), flush=True)
print("\nCITATIONS VERIFIED:", flush=True)
for src in res.get("sources", []):
    print(f"Citation [{src.get('citation_id')}]: {src.get('source_file')} pg {src.get('page')} | is_cited: {src.get('is_referenced_in_text')} | is_grounded: {src.get('is_claim_grounded')}", flush=True)


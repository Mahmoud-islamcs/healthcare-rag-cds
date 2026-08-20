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
from src.generation.prompt_templates import MEDICAL_RAG_SYSTEM_PROMPT, format_rag_prompt

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

pipe = MedicalRAGPipeline(hybrid, llm, settings)

query = "متى يُوصى بإضافة الإنسولين في الخطة العلاجية لمريض السكري؟"
print("User Query:", query)

trans_q = pipe._prepare_retrieval_query(query)
print("Translated Search Terms:", trans_q)

chunks = hybrid.retrieve(trans_q, final_top_k=5)
print(f"\nRetrieved {len(chunks)} chunks:")
for i, (c, s) in enumerate(chunks):
    print(f"\n--- Chunk {i+1} (Score: {s:.3f}, File: {c['source_file']}, Page: {c['page']}) ---")
    print(c["text"][:300])

user_prompt = format_rag_prompt(query, chunks)
print("\n--- Sending Prompt to LLM ---")
answer = llm.generate(MEDICAL_RAG_SYSTEM_PROMPT, user_prompt)
print("\n=== LLM Answer ===\n", answer)

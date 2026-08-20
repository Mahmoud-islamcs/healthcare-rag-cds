import os
import sys
import json
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

def main():
    settings = load_yaml_config("config/settings.yaml")
    models_cfg = load_yaml_config("config/models.yaml")

    query = "متى يُوصى بإضافة الإنسولين إلى الخطة العلاجية لمريض السكري"
    print("=================================================================")
    print(f"STEP 2 FORENSIC REPRODUCTION ON QUERY:\n'{query}'")
    print("=================================================================")

    # 1. Load Components
    emb_model = LocalEmbeddingModel(models_cfg["embedding"]["model_id"])
    vector_store = LocalFAISSVectorStore()
    vector_store.load(settings["paths"]["index_dir"])
    
    bm25 = LocalBM25Retriever()
    bm25.load(settings["paths"]["index_dir"])

    reranker = LocalReranker(models_cfg["reranker"]["model_id"])
    hybrid = HybridRetriever(vector_store, emb_model, bm25, reranker)
    llm = UnifiedLLM(provider=models_cfg["llm"]["provider"], model_id=models_cfg["llm"]["model_id"])
    pipeline = MedicalRAGPipeline(hybrid, llm, settings)

    # 2. Query Translation / Expansion Step
    retrieval_query = pipeline._prepare_retrieval_query(query)
    print(f"\n[1] RETRIEVAL QUERY EXPANSION:\n  Raw Query: '{query}'\n  Expanded/Translated: '{retrieval_query}'")

    # 3. FAISS Retrieval Top 20
    q_emb = emb_model.embed_query(retrieval_query)
    faiss_raw = vector_store.search(q_emb, top_k=20)
    print(f"\n[2] FAISS TOP 5 RESULTS (out of {len(faiss_raw)}):")
    for i, (chunk, score) in enumerate(faiss_raw[:5]):
        print(f"  Rank {i+1} | Score: {score:.4f} | Source: {chunk['source_file']} (p.{chunk['page']}) | ID: {chunk['chunk_id']}")
        print(f"    Snippet: {chunk['text'][:140]}...")

    # 4. BM25 Retrieval Top 20
    bm25_raw = bm25.search(retrieval_query, top_k=20)
    print(f"\n[3] BM25 TOP 5 RESULTS (out of {len(bm25_raw)}):")
    for i, (chunk, score) in enumerate(bm25_raw[:5]):
        print(f"  Rank {i+1} | Score: {score:.4f} | Source: {chunk['source_file']} (p.{chunk['page']}) | ID: {chunk['chunk_id']}")
        print(f"    Snippet: {chunk['text'][:140]}...")

    # 5. Hybrid Merged Candidates
    combined_scores = {}
    for chunk, score in faiss_raw:
        cid = chunk["chunk_id"]
        combined_scores[cid] = {"chunk": chunk, "score": score * 0.6}
    for chunk, score in bm25_raw:
        cid = chunk["chunk_id"]
        if cid in combined_scores:
            combined_scores[cid]["score"] += score * 0.4
        else:
            combined_scores[cid] = {"chunk": chunk, "score": score * 0.4}
    
    sorted_merged = sorted(combined_scores.values(), key=lambda x: x["score"], reverse=True)
    top_candidates = [item["chunk"] for item in sorted_merged[:30]]
    print(f"\n[4] HYBRID MERGED CANDIDATES: Total unique = {len(combined_scores)}, Selected for reranking = {len(top_candidates)}")

    # 6. Reranker Scores
    reranked_chunks = reranker.rerank(retrieval_query, top_candidates, top_k=5)
    print(f"\n[5] BGE RERANKER TOP 5 (Sigmoid calibrated):")
    for i, (chunk, score) in enumerate(reranked_chunks):
        print(f"  Rank {i+1} | Calibrated Score: {score:.4f} | Source: {chunk['source_file']} (p.{chunk['page']}) | ID: {chunk['chunk_id']}")
        print(f"    Text: {chunk['text'][:160]}...")

    # 7. Context & Prompts
    user_prompt = format_rag_prompt(query, reranked_chunks)
    print(f"\n[6] EXACT SYSTEM PROMPT:\n{MEDICAL_RAG_SYSTEM_PROMPT}")
    print(f"\n[7] EXACT USER PROMPT (first 600 chars):\n{user_prompt[:600]}...\n[truncated]")

    # 8. Raw LLM Generation
    raw_response = llm.generate(MEDICAL_RAG_SYSTEM_PROMPT, user_prompt)
    print(f"\n[8] RAW LLM RESPONSE:\n{raw_response}")

    # 9. Pipeline Full Execution
    full_output = pipeline.answer_query(query)
    print("\n=================================================================")
    print("PIPELINE ANSWER OUTPUT:")
    print(f"Status: {full_output['status']}")
    print(f"Confidence: {full_output['confidence']}")
    print(f"Answer:\n{full_output['answer']}")
    print("=================================================================")

if __name__ == "__main__":
    main()

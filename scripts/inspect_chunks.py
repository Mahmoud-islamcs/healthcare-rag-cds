import sys
sys.path.insert(0, '.')
from src.utils.config_loader import load_yaml_config
from src.embeddings.embedding_model import LocalEmbeddingModel
from src.retrieval.vector_store import LocalFAISSVectorStore
from src.retrieval.bm25_retriever import LocalBM25Retriever
from src.retrieval.reranker import LocalReranker
from src.retrieval.hybrid_retriever import HybridRetriever

settings = load_yaml_config('config/settings.yaml')
models_cfg = load_yaml_config('config/models.yaml')

emb_model = LocalEmbeddingModel(models_cfg['embedding']['model_id'])
vector_store = LocalFAISSVectorStore()
bm25 = LocalBM25Retriever()
reranker = LocalReranker(models_cfg['reranker']['model_id'])

idx_dir = settings['paths']['index_dir']
vector_store.load(idx_dir)
bm25.load(idx_dir)

hybrid = HybridRetriever(vector_store, emb_model, bm25, reranker)

# Test with the English clinical query that the translator generates
query_en = "When is insulin recommended in Type 2 diabetes? indications initiation HbA1c threshold"
chunks, diags = hybrid.retrieve(query_en, final_top_k=5, return_diagnostics=True)

print("=" * 70)
print("RETRIEVAL DIAGNOSTICS FOR QUERY:", query_en)
print("=" * 70)

for i, (c, s) in enumerate(chunks, 1):
    src = c.get('source_file')
    pg = c.get('page')
    txt = c.get('text', '')
    print(f"\n--- CHUNK [{i}] (Rerank Score: {s:.4f}) | File: {src} | Page: {pg} ---")
    print(txt)
    has_9 = "9%" in txt or "9 %" in txt or " 9." in txt or " 9 " in txt or "9.0" in txt
    has_hba1c = "hba1c" in txt.lower() or "a1c" in txt.lower()
    print(f"--> [Has '9%': {has_9}] | [Has 'HbA1c': {has_hba1c}]")

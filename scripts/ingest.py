import os
import sys
import glob

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config_loader import load_yaml_config
from src.ingestion.document_loader import UniversalDocumentLoader
from src.ingestion.chunker import MedicalAwareChunker
from src.embeddings.embedding_model import LocalEmbeddingModel
from src.retrieval.vector_store import LocalFAISSVectorStore
from src.retrieval.bm25_retriever import LocalBM25Retriever

def main():
    settings = load_yaml_config("config/settings.yaml")
    models_cfg = load_yaml_config("config/models.yaml")
    
    raw_dir = settings["paths"]["raw_data_dir"]
    idx_dir = settings["paths"]["index_dir"]
    os.makedirs(idx_dir, exist_ok=True)
    
    file_paths = glob.glob(os.path.join(raw_dir, "*.*"))
    print(f"Found {len(file_paths)} files in {raw_dir}")
    
    all_docs = []
    for fp in file_paths:
        try:
            docs = UniversalDocumentLoader.load_file(fp)
            all_docs.extend(docs)
            print(f"Loaded: {os.path.basename(fp)} ({len(docs)} pages/records)")
        except Exception as e:
            print(f"Error loading {fp}: {e}")
            
    if not all_docs:
        print("No documents loaded.")
        return
        
    chunker = MedicalAwareChunker()
    chunks = chunker.chunk_documents(all_docs)
    print(f"Created {len(chunks)} medical structure-aware chunks.")
    
    print("Generating local embeddings (BAAI/bge-small-en-v1.5)...")
    emb_model = LocalEmbeddingModel(models_cfg["embedding"]["model_id"])
    texts = [c["text"] for c in chunks]
    embeddings = emb_model.embed_documents(texts)
    
    vs = LocalFAISSVectorStore()
    vs.add_chunks(chunks, embeddings)
    vs.save(idx_dir)
    
    bm25 = LocalBM25Retriever()
    bm25.index_chunks(vs.chunks_metadata)
    bm25.save(idx_dir)
    
    print(f"SUCCESS: Saved FAISS vector index, metadata, and BM25 index to {idx_dir}")
    print("Ingestion complete successfully!")

if __name__ == "__main__":
    main()

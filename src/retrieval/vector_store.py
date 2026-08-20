import os
import pickle
import faiss
import numpy as np
from typing import List, Dict, Any, Tuple

class LocalFAISSVectorStore:
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.chunks_metadata: List[Dict[str, Any]] = []

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings length mismatch")
        self.index.add(np.ascontiguousarray(embeddings.astype(np.float32)))
        self.chunks_metadata.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 20) -> List[Tuple[Dict[str, Any], float]]:
        if self.index.ntotal == 0:
            return []
        query_vec = np.ascontiguousarray(query_embedding.reshape(1, -1).astype(np.float32))
        scores, indices = self.index.search(query_vec, min(top_k, self.index.ntotal))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.chunks_metadata):
                results.append((self.chunks_metadata[idx], float(score)))
        return results

    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.index, os.path.join(directory, "faiss.index"))
        with open(os.path.join(directory, "metadata.pkl"), "wb") as f:
            pickle.dump(self.chunks_metadata, f)

    def load(self, directory: str):
        index_path = os.path.join(directory, "faiss.index")
        meta_path = os.path.join(directory, "metadata.pkl")
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(f"FAISS index or metadata missing in {directory}")
        self.index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            self.chunks_metadata = pickle.load(f)

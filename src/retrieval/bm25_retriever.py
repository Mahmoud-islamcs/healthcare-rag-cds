import os
import re
import pickle
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Plus

class LocalBM25Retriever:
    def __init__(self):
        self.bm25 = None
        self.chunks: List[Dict[str, Any]] = []

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'[\w\-]+', text.lower())

    def index_chunks(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks
        if not chunks:
            self.bm25 = None
            return
        corpus = [self._tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Plus(corpus)

    def search(self, query: str, top_k: int = 20) -> List[Tuple[Dict[str, Any], float]]:
        if not self.bm25 or not self.chunks:
            return []
        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return []
        raw_scores = self.bm25.get_scores(tokenized_query)
        max_score = max(raw_scores) if len(raw_scores) > 0 and max(raw_scores) > 0 else 1.0
        normalized_scores = [s / max_score for s in raw_scores]
        
        indexed_scores = list(enumerate(normalized_scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in indexed_scores[:top_k]:
            if score > 0:
                results.append((self.chunks[idx], float(score)))
        return results

    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        bm25_path = os.path.join(directory, "bm25.pkl")
        with open(bm25_path, "wb") as f:
            pickle.dump({
                "chunks": self.chunks,
                "bm25": self.bm25
            }, f)

    def load(self, directory: str):
        bm25_path = os.path.join(directory, "bm25.pkl")
        if not os.path.exists(bm25_path):
            raise FileNotFoundError(f"BM25 index file not found at: {bm25_path}")
        with open(bm25_path, "rb") as f:
            data = pickle.load(f)
            self.chunks = data.get("chunks", [])
            self.bm25 = data.get("bm25")

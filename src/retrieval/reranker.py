import gc
import numpy as np
import torch
from typing import List, Dict, Any, Tuple
from sentence_transformers import CrossEncoder
from src.utils.device import get_optimal_device

class LocalReranker:
    def __init__(self, model_id: str = "BAAI/bge-reranker-base", device: str = "auto"):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.device = get_optimal_device(device)
        self.model = CrossEncoder(model_id, device=self.device)


    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))

    def rerank(self, query: str, candidate_chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        if not candidate_chunks:
            return []
        pairs = [[query, c["text"]] for c in candidate_chunks]
        raw_scores = self.model.predict(pairs, show_progress_bar=False)
        calibrated_scores = self._sigmoid(np.array(raw_scores))
        
        scored_chunks = list(zip(candidate_chunks, [float(s) for s in calibrated_scores]))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:top_k]

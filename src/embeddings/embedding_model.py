import gc
from typing import List
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from src.utils.device import get_optimal_device

class LocalEmbeddingModel:
    def __init__(self, model_id: str = "BAAI/bge-small-en-v1.5", device: str = "auto"):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.device = get_optimal_device(device)
        self.model = SentenceTransformer(model_id, device=self.device)

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]


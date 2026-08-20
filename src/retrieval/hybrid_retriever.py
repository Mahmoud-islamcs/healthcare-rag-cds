from typing import List, Dict, Any, Tuple, Optional

class HybridRetriever:
    def __init__(
        self,
        vector_store,
        embedding_model,
        bm25_retriever,
        reranker=None,
        dense_weight: float = 0.6,
        bm25_weight: float = 0.4,
        rrf_k: int = 60,
        candidate_pool_size: int = 25
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k
        self.candidate_pool_size = candidate_pool_size

    def retrieve(
        self,
        query: str,
        dense_top_k: int = 20,
        bm25_top_k: int = 20,
        final_top_k: int = 5,
        return_diagnostics: bool = False
    ) -> Any:
        q_emb = self.embedding_model.embed_query(query)
        dense_results = self.vector_store.search(q_emb, top_k=dense_top_k)
        bm25_results = self.bm25_retriever.search(query, top_k=bm25_top_k)

        # Reciprocal Rank Fusion (RRF)
        rrf_candidates: Dict[str, Dict[str, Any]] = {}

        for rank, (chunk, score) in enumerate(dense_results, 1):
            cid = chunk["chunk_id"]
            dense_rrf = self.dense_weight * (1.0 / (self.rrf_k + rank))
            rrf_candidates[cid] = {
                "chunk": chunk,
                "rrf_score": dense_rrf,
                "dense_rank": rank,
                "dense_score": score,
                "bm25_rank": None,
                "bm25_score": 0.0
            }

        for rank, (chunk, score) in enumerate(bm25_results, 1):
            cid = chunk["chunk_id"]
            bm25_rrf = self.bm25_weight * (1.0 / (self.rrf_k + rank))
            if cid in rrf_candidates:
                rrf_candidates[cid]["rrf_score"] += bm25_rrf
                rrf_candidates[cid]["bm25_rank"] = rank
                rrf_candidates[cid]["bm25_score"] = score
            else:
                rrf_candidates[cid] = {
                    "chunk": chunk,
                    "rrf_score": bm25_rrf,
                    "dense_rank": None,
                    "dense_score": 0.0,
                    "bm25_rank": rank,
                    "bm25_score": score
                }

        sorted_candidates = sorted(
            rrf_candidates.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )

        top_candidates = [item["chunk"] for item in sorted_candidates[:self.candidate_pool_size]]

        if self.reranker and top_candidates:
            reranked_chunks = self.reranker.rerank(query, top_candidates, top_k=final_top_k)
            if return_diagnostics:
                return reranked_chunks, {
                    "dense_results": dense_results,
                    "bm25_results": bm25_results,
                    "rrf_candidates": sorted_candidates[:self.candidate_pool_size]
                }
            return reranked_chunks

        fallback_results = [(item["chunk"], item["rrf_score"]) for item in sorted_candidates[:final_top_k]]
        if return_diagnostics:
            return fallback_results, {
                "dense_results": dense_results,
                "bm25_results": bm25_results,
                "rrf_candidates": sorted_candidates[:self.candidate_pool_size]
            }
        return fallback_results


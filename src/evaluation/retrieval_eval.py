from typing import List, Dict, Any

class RetrievalEvaluator:
    @staticmethod
    def evaluate_query(retrieved_doc_ids: List[str], ground_truth_doc_ids: List[str], k: int = 5) -> Dict[str, float]:
        top_k_retrieved = retrieved_doc_ids[:k]
        hits = [doc_id in ground_truth_doc_ids for doc_id in top_k_retrieved]
        num_hits = sum(hits)
        precision = num_hits / k if k > 0 else 0.0
        recall = num_hits / len(ground_truth_doc_ids) if ground_truth_doc_ids else 0.0
        mrr = 0.0
        for rank, is_hit in enumerate(hits, 1):
            if is_hit:
                mrr = 1.0 / rank
                break
        return {"precision_at_k": precision, "recall_at_k": recall, "mrr": mrr}

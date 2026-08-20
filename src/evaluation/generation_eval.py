from rouge_score import rouge_scorer

class GenerationEvaluator:
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    def evaluate_groundedness(self, answer: str, context: str) -> float:
        if not answer or not context:
            return 0.0
        scores = self.scorer.score(context, answer)
        return float(scores['rougeL'].precision)

import re
from typing import List, Dict, Any, Tuple


class EvidenceValidator:
    def __init__(self, min_confidence_threshold: float = 0.35, min_char_length: int = 50):
        self.min_confidence_threshold = min_confidence_threshold
        self.min_char_length = min_char_length

    def evaluate_retrieved_evidence(
        self,
        scored_chunks: List[Tuple[Dict[str, Any], float]]
    ) -> Tuple[bool, float, str, str]:
        """
        Evaluates retrieval evidence quality without treating retrieval
        confidence as clinical certainty.

        Returns:
            (is_sufficient, retrieval_score, quality_label, reason)
        """
        if not scored_chunks:
            return False, 0.0, "Insufficient", "No relevant medical passages found in knowledge base."

        valid_chunks = []
        for item in scored_chunks:
            if isinstance(item, (tuple, list)):
                chunk = item[0] if item else {}
                score = item[1] if len(item) > 1 else 0.0
            elif isinstance(item, dict):
                chunk = item
                score = item.get("score", 0.0)
            else:
                continue
            if len(chunk.get("text", "").strip()) >= self.min_char_length:
                valid_chunks.append((chunk, float(score)))

        if not valid_chunks:
            return False, 0.0, "Insufficient", "Retrieved passages are too short to support clinical claims."

        top_score = float(valid_chunks[0][1])
        avg_score = float(sum(s for _, s in valid_chunks) / len(valid_chunks))

        retrieval_score = float(max(0.0, min(1.0, (top_score * 0.7 + avg_score * 0.3))))

        if retrieval_score >= 0.65:
            quality_label = "High"
        elif retrieval_score >= 0.50:
            quality_label = "Moderate"
        elif retrieval_score >= self.min_confidence_threshold:
            quality_label = "Low"
        else:
            quality_label = "Insufficient"

        if retrieval_score < self.min_confidence_threshold:
            return False, retrieval_score, quality_label, "Retrieval evidence score is below clinical safety threshold."

        clinical_chunks = [
            c for c, _ in valid_chunks
            if self._has_clinical_signal(c.get("text", ""))
        ]
        if not clinical_chunks:
            return False, retrieval_score, "Insufficient", "Retrieved passages do not contain enough clinical signal to support medical claims."

        if self._looks_like_fragmented_treatment_options(valid_chunks):
            return (
                True,
                retrieval_score,
                quality_label,
                "Evidence supports only narrow source-grounded statements. Separate treatment options must not be synthesized into a pathway unless one cited passage explicitly states that pathway.",
            )

        return True, retrieval_score, quality_label, "Evidence is sufficient for claim-level grounded synthesis."

    def _has_clinical_signal(self, text: str) -> bool:
        clinical_patterns = [
            r"\bdiabetes\b", r"\binsulin\b", r"\bmetformin\b", r"\bglp-?1\b",
            r"\bdpp-?4\b", r"\bsglt-?2\b", r"\bhba1c\b", r"\begfr\b",
            r"\bbmi\b", r"\btreatment\b", r"\btherapy\b", r"\brecommend",
            r"\bconsider\b", r"\boption\b", r"\bexercise\b", r"\bdiet\b",
            r"\bhypertension\b", r"\bblood pressure\b", r"\bace inhibitor\b",
            r"\boncology\b", r"\bprotocol\b", r"\bchemotherapy\b", r"\bdosage\b",
            r"السكري", r"السكر", r"الإنسولين", r"انسولين", r"العلاج",
            r"يوصى", r"يُوصى", r"يمكن", r"قد", r"الرياضة", r"الغذاء",
        ]
        return any(re.search(p, text, flags=re.IGNORECASE) for p in clinical_patterns)

    def _looks_like_fragmented_treatment_options(
        self,
        valid_chunks: List[Tuple[Dict[str, Any], float]]
    ) -> bool:
        option_terms = [
            r"\bglp-?1\b", r"\bdpp-?4\b", r"\bsglt-?2\b",
            r"\binsulin\b", r"\bsulfonylurea\b", r"\bpioglitazone\b",
        ]
        sequence_terms = [
            r"\bthen\b", r"\bafter\b", r"\bfollowed by\b",
            r"\bsequence\b", r"\bstep\b", r"\bline\b",
        ]
        term_locations = {}
        for idx, (chunk, _) in enumerate(valid_chunks):
            text = chunk.get("text", "")
            for term in option_terms:
                if re.search(term, text, flags=re.IGNORECASE):
                    term_locations.setdefault(term, set()).add(idx)

        has_separate_options = len(term_locations) >= 3
        connected_in_one_chunk = any(
            sum(1 for term in option_terms if re.search(term, chunk.get("text", ""), flags=re.IGNORECASE)) >= 3
            and any(re.search(seq, chunk.get("text", ""), flags=re.IGNORECASE) for seq in sequence_terms)
            for chunk, _ in valid_chunks
        )
        return has_separate_options and not connected_in_one_chunk

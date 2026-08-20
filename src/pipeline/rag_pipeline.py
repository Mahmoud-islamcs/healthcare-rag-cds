import re
import time
import logging
from typing import Dict, Any, List, Tuple
from src.safety.guardrails import MedicalSafetyGuardrails
from src.safety.evidence_validator import EvidenceValidator
from src.safety.citation_validator import CitationValidator, ABSTENTION_AR, ABSTENTION_EN
from src.generation.prompt_templates import MEDICAL_RAG_SYSTEM_PROMPT, format_rag_prompt

logger = logging.getLogger(__name__)

class MedicalRAGPipeline:
    def __init__(self, hybrid_retriever, llm, settings: Dict[str, Any]):
        self.retriever = hybrid_retriever
        self.llm = llm
        self.settings = settings
        self.validator = EvidenceValidator(
            min_confidence_threshold=settings.get("safety", {}).get("confidence_threshold", 0.35)
        )
        self.debug_mode = settings.get("app", {}).get("debug", False)

    def _prepare_retrieval_query(self, query: str) -> str:
        is_arabic = bool(re.search(r'[\u0600-\u06FF]', query))
        if is_arabic:
            try:
                system_prompt = (
                    "You are an expert Clinical Search Query Optimizer for medical guidelines. "
                    "Translate the clinical question into precise English medical search keywords, "
                    "retaining the specific clinical entities, condition, drug/intervention, and indication criteria. "
                    "Output ONLY the search keywords without punctuation, quotes, or conversational filler."
                )
                translated = self.llm.generate(system_prompt, query).strip().strip('"').strip("'")
                if translated and len(translated) > 2 and not bool(re.search(r'[\u0600-\u06FF]', translated)):
                    logger.info(f"Clinical Query Intent Expansion: '{query}' -> '{translated}'")
                    return translated
            except Exception as e:
                logger.warning(f"LLM query translation fallback: {e}")

            # Deterministic rule-based medical entity mapping fallback
            q_lower = query.lower()
            terms = []
            if "سكر" in q_lower or "النوع الثاني" in q_lower:
                terms.append("type 2 diabetes")
            if "إنسولين" in q_lower or "انسولين" in q_lower:
                terms.append("insulin initiation when to add insulin acute worsening hyperglycaemia")
            if "رياض" in q_lower or "نشاط" in q_lower:
                terms.append("exercise physical activity lifestyle")
            if "ميتفورمين" in q_lower:
                terms.append("metformin")
            if "كلى" in q_lower:
                terms.append("chronic kidney disease CKD")
            if "قلب" in q_lower:
                terms.append("cardiovascular heart failure")
            
            if terms:
                fallback_str = " ".join(terms)
                logger.info(f"Deterministic Query Expansion fallback: '{query}' -> '{fallback_str}'")
                return fallback_str
        return query




    def _sanitize_answer(self, raw_answer: str) -> str:
        cleaned = raw_answer.strip()
        cleaned = re.sub(r'---\s*\n\s*\*?DISCLAIMER:.*$', '', cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
        cleaned = re.sub(r'\*?تنبيه طبي:.*$', '', cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
        return cleaned

    def _abstention_text(self, is_arabic: bool) -> str:
        return ABSTENTION_AR if is_arabic else ABSTENTION_EN

    def _with_disclaimer(self, answer: str) -> str:
        disclaimer = self.settings.get("safety", {}).get("disclaimer", "")
        if not disclaimer:
            return answer
        return f"{answer}\n\n---\n*{disclaimer}*"

    def _generate_and_validate(
        self,
        system_prompt: str,
        user_prompt: str,
        retrieved_chunks: List[Any],
        diagnostics: Dict[str, Any],
        attempt_label: str
    ) -> Tuple[str, List[Dict[str, Any]], List[int], Dict[str, Any]]:
        raw_answer = self.llm.generate(system_prompt, user_prompt)
        cleaned_answer = self._sanitize_answer(raw_answer)
        diagnostics.setdefault("generated_answer", {})[attempt_label] = cleaned_answer

        verified_citations, invalid_c_ids = CitationValidator.build_and_validate_citations(
            cleaned_answer, retrieved_chunks
        )
        claim_report = CitationValidator.validate_claims(cleaned_answer, retrieved_chunks)
        diagnostics.setdefault("citation_validation_attempts", {})[attempt_label] = claim_report
        return cleaned_answer, verified_citations, invalid_c_ids, claim_report

    def answer_query(self, query: str, return_debug: bool = False) -> Dict[str, Any]:
        start_time = time.time()
        diagnostics = {"raw_query": query}

        # 1. Emergency Safety Intercept
        is_safe, emergency_msg = MedicalSafetyGuardrails.check_query_safety(query)
        if not is_safe:
            return {
                "answer": emergency_msg,
                "sources": [],
                "retrieval_score": 1.0,
                "confidence": 1.0,
                "evidence_quality": "High",
                "status": "EMERGENCY_TRIGGERED",
                "latency_sec": round(time.time() - start_time, 2),
                "debug": diagnostics if (self.debug_mode or return_debug) else None
            }

        # 1.5 Conversational / Greeting Intercept
        is_conv, conv_msg = MedicalSafetyGuardrails.check_conversational_query(query)
        if is_conv:
            return {
                "answer": conv_msg,
                "sources": [],
                "retrieval_score": 1.0,
                "confidence": 1.0,
                "evidence_quality": "Conversational",
                "status": "CONVERSATIONAL",
                "latency_sec": round(time.time() - start_time, 2),
                "debug": diagnostics if (self.debug_mode or return_debug) else None
            }

        # 2. Query Expansion
        retrieval_query = self._prepare_retrieval_query(query)
        diagnostics["retrieval_query"] = retrieval_query


        # 3. Hybrid Retrieval + Reranking
        top_k = self.settings.get("retrieval", {}).get("rerank_top_k", 5)
        if self.debug_mode or return_debug:
            retrieval_result = self.retriever.retrieve(
                retrieval_query, final_top_k=top_k, return_diagnostics=True
            )
            if (
                isinstance(retrieval_result, tuple)
                and len(retrieval_result) == 2
                and isinstance(retrieval_result[1], dict)
            ):
                retrieved_chunks, ret_diag = retrieval_result
            else:
                retrieved_chunks, ret_diag = retrieval_result, {}
            diagnostics["retrieval_diagnostics"] = ret_diag
        else:
            retrieved_chunks = self.retriever.retrieve(retrieval_query, final_top_k=top_k)

        # 4. Evidence Quality Gating (Safe unpacking)
        val_res = self.validator.evaluate_retrieved_evidence(retrieved_chunks)
        if len(val_res) == 4:
            is_sufficient, score, quality_label, reason = val_res
        else:
            is_sufficient, score, reason = val_res[0], val_res[1], val_res[2]
            quality_label = "High" if score >= 0.65 else ("Moderate" if score >= 0.50 else "Low")

        diagnostics["evidence_validation"] = {
            "is_sufficient": is_sufficient,
            "score": round(score, 3),
            "quality_label": quality_label,
            "reason": reason
        }

        is_arabic = bool(re.search(r'[\u0600-\u06FF]', query))

        # CASE B: Insufficient Evidence -> Pure Abstention
        if not is_sufficient:
            refusal_text = self._abstention_text(is_arabic)
            return {
                "answer": self._with_disclaimer(refusal_text),
                "sources": [],
                "retrieval_score": round(score, 3),
                "confidence": round(score, 3),
                "evidence_quality": quality_label,
                "status": "INSUFFICIENT_EVIDENCE",
                "latency_sec": round(time.time() - start_time, 2),
                "debug": diagnostics if (self.debug_mode or return_debug) else None
            }

        # CASE A: Sufficient Evidence -> Synthesis & Citation Validation
        prompt_res = format_rag_prompt(query, retrieved_chunks)
        if isinstance(prompt_res, dict):
            system_prompt = prompt_res.get("system_prompt", MEDICAL_RAG_SYSTEM_PROMPT)
            user_prompt = prompt_res.get("user_prompt", "")
            diagnostics["prompt_metadata"] = prompt_res.get("metadata", {})
        else:
            system_prompt = MEDICAL_RAG_SYSTEM_PROMPT
            user_prompt = prompt_res

        diagnostics["user_prompt"] = user_prompt
        
        # Programmatic claim-level citation verification
        cleaned_answer, verified_citations, invalid_c_ids, claim_report = self._generate_and_validate(
            system_prompt, user_prompt, retrieved_chunks, diagnostics, "initial"
        )

        if not claim_report.get("is_valid", False):
            strict_retry_prompt = (
                user_prompt
                + "\n\n============================================================\n"
                + "CONTROLLED REGENERATION AFTER CLAIM VALIDATION FAILURE\n"
                + "============================================================\n"
                + "The previous draft contained unsupported clinical claims, citation-scope errors, "
                + "unsupported numbers, treatment sequencing, ranking language, or recommendation "
                + "strengthening. Regenerate once. Use only claims explicitly stated in the cited "
                + "evidence. Remove any claim that is not directly supported. If this leaves no safe "
                + "answer, output only the abstention sentence in the user's language."
            )
            cleaned_answer, verified_citations, invalid_c_ids, claim_report = self._generate_and_validate(
                system_prompt, strict_retry_prompt, retrieved_chunks, diagnostics, "regeneration"
            )

        diagnostics["citation_validation"] = {
            "verified_citations_count": len(verified_citations),
            "invalid_citations_found": invalid_c_ids,
            "is_valid": claim_report.get("is_valid", False),
            "claims": claim_report.get("claims", []),
            "invalid_claims": claim_report.get("invalid_claims", []),
            "unsupported_claims": claim_report.get("unsupported_claims", []),
            "invalid_citations": claim_report.get("invalid_citations", []),
            "numerical_claims": claim_report.get("numerical_claims", []),
            "treatment_sequence_flags": claim_report.get("treatment_sequence_flags", []),
        }

        if not claim_report.get("is_valid", False):
            final_sources, final_invalid = CitationValidator.build_and_validate_citations(
                "", retrieved_chunks
            )
            diagnostics["citation_validation"]["fail_safe_action"] = "abstained_after_failed_regeneration"
            diagnostics["citation_validation"]["invalid_citations_found"] = final_invalid
            return {
                "answer": self._with_disclaimer(self._abstention_text(is_arabic)),
                "sources": final_sources,
                "retrieval_score": round(score, 3),
                "retrieval_relevance": round(score, 2),
                "sources_count": len(final_sources),
                "citations_verified_count": 0,
                "evidence_quality": quality_label.upper(),
                "status": "UNSAFE_GENERATION_REJECTED",
                "latency_sec": round(time.time() - start_time, 2),
                "debug": diagnostics if (self.debug_mode or return_debug) else None
            }

        # Single Disclaimer Guarantee
        final_answer = self._with_disclaimer(cleaned_answer)
        cited_count = len([c for c in verified_citations if c.get("is_referenced_in_text")])
        total_sources_count = len(verified_citations)


        return {
            "answer": final_answer,
            "sources": verified_citations,
            "retrieval_score": round(score, 3),
            "retrieval_relevance": round(score, 2),
            "sources_count": total_sources_count,
            "citations_verified_count": cited_count,
            "evidence_quality": quality_label.upper(),
            "status": "SUCCESS",
            "latency_sec": round(time.time() - start_time, 2),
            "debug": diagnostics if (self.debug_mode or return_debug) else None
        }


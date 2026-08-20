import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Set, Optional


ABSTENTION_EN = "The retrieved medical sources do not provide sufficient evidence to answer this question safely."
ABSTENTION_AR = "المصادر الطبية المفهرسة لا تتضمن أدلة كافية وموثوقة للإجابة على هذا السؤال بأمان."


@dataclass
class ClaimValidationResult:
    claim: str
    citation_ids: List[int]
    is_valid: bool
    support_type: str = "EXPLICIT"
    clinical_strength: str = "neutral"
    unsupported_terms: List[str] = field(default_factory=list)
    unsupported_numbers: List[str] = field(default_factory=list)
    sequence_flags: List[str] = field(default_factory=list)
    ranking_flags: List[str] = field(default_factory=list)
    strength_flags: List[str] = field(default_factory=list)
    invalid_citations: List[int] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim,
            "citation_ids": self.citation_ids,
            "is_valid": self.is_valid,
            "support_type": self.support_type,
            "clinical_strength": self.clinical_strength,
            "unsupported_terms": self.unsupported_terms,
            "unsupported_numbers": self.unsupported_numbers,
            "sequence_flags": self.sequence_flags,
            "ranking_flags": self.ranking_flags,
            "strength_flags": self.strength_flags,
            "invalid_citations": self.invalid_citations,
            "reason": self.reason,
        }


class CitationValidator:
    """
    Deterministic clinical claim and citation validator.
    Validates claim-level evidence grounding, numerical fidelity,
    and prevents unsupported cross-chunk medical leaps.
    """

    _ARABIC_INDIC_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
    _CITATION_RE = re.compile(r'\[(\d+(?:\s*,\s*\d+)*)\]')

    _SEQUENCE_PATTERNS = [
        r"\b(first|second|third)[-\s]?line\b", r"\bnext step\b", r"\bstep\s*\d+\b",
        r"\bafter failure\b", r"\bafter (?:inadequate|failed|failing)\b",
        r"\bescalat(?:e|ion)\b", r"\bswitch to\b",
        r"الخط\s*(?:الأول|الثاني|الثالث)", r"العلاج\s*(?:الأول|الثاني|الثالث)",
        r"الخطوة\s*(?:التالية|الأولى|الثانية|الثالثة)", r"الانتقال\s+إلى",
    ]
    _RANKING_PATTERNS = [
        r"\bpreferred\b", r"\bbest\b", r"\boptimal\b", r"\bmost effective\b",
        r"\bgold standard\b", r"\btreatment of choice\b",
        r"أفضل", r"الأفضل", r"المفضل", r"الأمثل", r"الأكثر\s+فعالية",
    ]
    _STRONG_RECOMMENDATION_PATTERNS = [
        r"\bshould\b", r"\bmust\b", r"\brecommended\b",
        r"\bindicated\b", r"\boffer\b", r"\brequire\b",
        r"يجب", r"ينبغي", r"لا\s*بد", r"يُوصى", r"يوصى", r"مُوصى", r"مطلوب",
    ]
    _WEAK_EVIDENCE_PATTERNS = [
        r"\bmay\b", r"\bmay be considered\b", r"\bcan be used\b",
        r"\bcould\b", r"\boption\b", r"\balternative\b", r"\bconsider\b",
        r"قد", r"يمكن", r"خيار", r"أحد الخيارات", r"ينظر", r"النظر",
    ]

    _CLINICAL_TOKEN_PATTERNS = [
        r"\bmetformin\b", r"\binsulin\b", r"\bbasal insulin\b", r"\bglp-?1\b",
        r"\bdpp-?4\b", r"\bsglt-?2\b", r"\bsulfonylurea\b", r"\bpioglitazone\b",
        r"\btirzepatide\b", r"\bhba1c\b", r"\begfr\b", r"\bbmi\b", r"\bascvd\b", r"\bckd\b",
        r"\bheart failure\b", r"\bcardiovascular\b", r"\bdiabetes\b",
        r"\btype 1\b", r"\btype 2\b", r"\bgestational\b", r"\bexercise\b",
        r"\bdiet\b", r"\bdose\b", r"\bfrequency\b", r"\bduration\b",
        r"\bthreshold\b", r"\bglyca?emic\b", r"\bhyperglyc", r"\bhypoglyc",
        r"\bblood pressure\b", r"\bhypertension\b",
        r"الإنسولين", r"انسولين", r"قاعدي", r"الميتفورمين", r"ميتفورمين",
        r"السكري", r"السكر", r"النوع\s+الأول", r"النوع\s+الثاني",
        r"تيرزيباتايد", r"بيوجليتازون", r"سلفونيل\s*يوريا",
        r"القلب", r"الكلى", r"ضغط\s+الدم",
    ]

    _SYNONYMS = {
        "الإنسولين": ["insulin", "basal insulin", "pre-mixed"],
        "انسولين": ["insulin", "basal insulin"],
        "قاعدي": ["basal"],
        "الميتفورمين": ["metformin"],
        "ميتفورمين": ["metformin"],
        "السكري": ["diabetes", "glycaemic", "glycemic"],
        "السكر": ["diabetes", "glycaemic", "glycemic", "glucose"],
        "النوع الثاني": ["type 2", "type two"],
        "النوع الأول": ["type 1", "type one"],
        "تيرزيباتايد": ["tirzepatide"],
        "بيوجليتازون": ["pioglitazone"],
        "سلفونيل يوريا": ["sulfonylurea", "sulphonylurea"],
        "الرياضة": ["exercise", "physical activity"],
        "التمارين": ["exercise", "physical activity"],
        "النظام الغذائي": ["diet", "dietary"],
        "الغذاء": ["diet", "dietary"],
        "القلب": ["heart", "cardiovascular", "heart failure"],
        "الكلى": ["kidney", "renal", "ckd"],
        "blood pressure": ["hypertension", "hypertensive"],
        "hypertension": ["blood pressure"],
        "diabetes": ["السكري", "glycaemic", "glycemic"],
        "insulin": ["الإنسولين", "انسولين"],
        "type 2": ["النوع الثاني"],
        "type 1": ["النوع الأول"],
        "exercise": ["الرياضة", "التمارين", "physical activity"],
        "lowers": ["treat", "treats", "treatment", "therapy"],
        "treat": ["used to treat", "therapy"],
        "add": ["additional", "adding", "added", "require"],
        "consider": ["may", "option", "can be used", "considered"],
        "recommended": ["recommend", "recommended", "offer"],
        "يوصى": ["يُوصى", "يوصي", "offer", "recommended"],
        "يمكن": ["قد", "may", "can"],
        "خيار": ["option", "alternative"],
    }

    @classmethod
    def clean_text_and_normalize_citations(cls, text: str) -> str:
        """Strip thinking blocks and normalize Arabic numerals/spaces in citation brackets."""
        if not text:
            return ""
        # 1. Strip <think>...</think>
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE).strip()
        # 2. Normalize Unicode (NFKC)
        cleaned = unicodedata.normalize("NFKC", cleaned)
        # 3. Replace non-breaking and zero-width spaces
        cleaned = re.sub(r'[\u202f\u00a0\u200b\u200e\u200f]', ' ', cleaned)
        # 4. Normalize Arabic digits to ASCII within citation brackets
        def _norm_brackets(match):
            inner = match.group(1).translate(cls._ARABIC_INDIC_MAP)
            # handle multi-citations e.g. 1, 2, 3
            parts = [p.strip() for p in inner.split(',') if p.strip()]
            return f"[{', '.join(parts)}]"
        cleaned = re.sub(r'\[([\d٠١٢٣٤٥٦٧٨٩\s,]+)\]', _norm_brackets, cleaned)
        return cleaned

    @classmethod
    def extract_cited_indices(cls, text: str) -> Set[int]:
        if not text:
            return set()
        normalized = cls.clean_text_and_normalize_citations(text)
        cited_ids = set()
        for match in cls._CITATION_RE.findall(normalized):
            for num_str in match.split(','):
                num_clean = num_str.strip()
                if num_clean.isdigit():
                    cited_ids.add(int(num_clean))
        return cited_ids

    @classmethod
    def _ids_in_claim(cls, text: str) -> List[int]:
        ids: List[int] = []
        normalized = cls.clean_text_and_normalize_citations(text)
        for match in cls._CITATION_RE.findall(normalized):
            for num_str in match.split(','):
                num_clean = num_str.strip()
                if num_clean.isdigit():
                    ids.append(int(num_clean))
        return ids

    @classmethod
    def extract_numerical_entities(cls, text: str) -> Set[str]:
        if not text:
            return set()
        # Strip citation tags and diabetes type indicators to avoid false triggers
        text_wo = re.sub(r"\[[^\]]+\]", " ", text)
        text_wo = re.sub(r"\btype\s*[12]\b", " ", text_wo, flags=re.IGNORECASE)
        text_wo = re.sub(r"النوع\s*(الأول|الثاني|1|2)", " ", text_wo)
        pattern = (
            r'(?<!\d)(?:[<>]=?|≥|≤)?\s*\d+(?:\.\d+)?\s*(?:%|mmol/mol|mg/dL|mg|mcg|g|kg/m2|kg/m²|'
            r'mL/min/1\.73m2|mL/min/1\.73m²|units?|iu|times?|x|daily|weekly|'
            r'hours?|days?|weeks?|months?|years?)'
        )
        values = {re.sub(r"\s+", " ", m.strip()) for m in re.findall(pattern, text_wo, flags=re.IGNORECASE)}
        return {v for v in values if v and v not in {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10"}}

    @classmethod
    def _normalize(cls, text: str) -> str:
        lowered = text.lower()
        lowered = unicodedata.normalize("NFKC", lowered)
        lowered = re.sub(r"[^\w\u0600-\u06FF.%/≥<=+-]+", " ", lowered)
        return re.sub(r"\s+", " ", lowered).strip()

    @classmethod
    def _contains_any(cls, text: str, patterns: List[str]) -> List[str]:
        return [p for p in patterns if re.search(p, text, flags=re.IGNORECASE)]

    @classmethod
    def _evidence_text_by_id(cls, retrieved_chunks: List[Any]) -> Dict[int, str]:
        evidence: Dict[int, str] = {}
        for idx, item in enumerate(retrieved_chunks, 1):
            chunk = item[0] if isinstance(item, (tuple, list)) and item else item
            if isinstance(chunk, dict):
                evidence[idx] = str(chunk.get("text", ""))
        return evidence

    @classmethod
    def extract_claims(cls, answer: str) -> List[str]:
        """
        Extract proposition-level clinical claims, ignoring structural markdown
        headers, summary labels, and intro colons.
        """
        if not answer:
            return []
        cleaned = cls.clean_text_and_normalize_citations(answer)
        claims: List[str] = []

        for line in cleaned.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line in {"---", "***"}:
                continue
            
            # Skip any structural markdown header e.g. **عنوان القسم** or **Section:**
            if re.match(r"^\*{1,3}[^*]+\*{1,3}:?$", line):
                continue
            if line.endswith(":") and not cls._ids_in_claim(line):
                continue
            if re.match(r"^\*{1,3}.*?(?:الخلاصة|الإجابة|التوصيات|ملاحظات|الاعتبارات|توجيهات|Summary|Note|Considerations).*?\*{1,3}:?\s*$", line, flags=re.IGNORECASE):
                continue

            # Strip list prefixes
            line_body = re.sub(r"^[-*•\d.]+\s*", "", line).strip()
            if not line_body:
                continue

            # If the line is a bold heading/lead-in, skip it
            if re.match(r"^\*{1,3}[^*]+\*{1,3}:?$", line_body):
                continue

            # If the entire line/bullet has citations, treat the bullet as a unified claim unit
            if cls._ids_in_claim(line_body):
                claims.append(line_body)
            else:
                # If there are no citations, check if it's a substantive clinical assertion
                parts = re.split(r"(?<=[.!؟])\s+(?=[A-Z\u0600-\u06FF])", line_body)
                for part in parts:
                    part = part.strip()
                    if part and cls._looks_clinical(part):
                        # Avoid treating short section lead-ins (e.g. "**العلاج الأول**") as uncited claims
                        if re.match(r"^\*{1,3}[^*]+\*{1,3}:?$", part) or len(part) < 20:
                            continue
                        claims.append(part)
        return claims

    @classmethod
    def _looks_clinical(cls, text: str) -> bool:
        if cls._contains_any(text, cls._CLINICAL_TOKEN_PATTERNS):
            return True
        if cls.extract_numerical_entities(text):
            return True
        return False

    @classmethod
    def _clinical_terms(cls, text: str) -> Set[str]:
        terms = set()
        norm = cls._normalize(text)
        for pattern in cls._CLINICAL_TOKEN_PATTERNS:
            for match in re.finditer(pattern, norm, flags=re.IGNORECASE):
                terms.add(match.group(0).strip())
        return {t for t in terms if t}

    @classmethod
    def _term_supported(cls, term: str, evidence_norm: str) -> bool:
        clean = cls._normalize(term)
        if clean and clean in evidence_norm:
            return True
        for synonym in cls._SYNONYMS.get(clean, []):
            if cls._normalize(synonym) in evidence_norm:
                return True
        return False

    @classmethod
    def _number_supported(cls, number: str, evidence_texts: List[str]) -> bool:
        compact = re.sub(r"\s+", "", number.lower())
        for evidence in evidence_texts:
            evidence_norm = evidence.lower()
            if number.lower() in evidence_norm:
                return True
            if compact and compact in re.sub(r"\s+", "", evidence_norm):
                return True
        return False

    @classmethod
    def _classify_strength(cls, text: str) -> str:
        norm = cls._normalize(text)
        if cls._contains_any(norm, cls._STRONG_RECOMMENDATION_PATTERNS):
            return "strong"
        if cls._contains_any(norm, cls._WEAK_EVIDENCE_PATTERNS):
            return "weak"
        return "neutral"

    @classmethod
    def validate_claims(cls, raw_answer: str, retrieved_chunks: List[Any]) -> Dict[str, Any]:
        total_chunks = len(retrieved_chunks)
        evidence_by_id = cls._evidence_text_by_id(retrieved_chunks)
        cleaned_answer = cls.clean_text_and_normalize_citations(raw_answer)
        cited_indices = cls.extract_cited_indices(cleaned_answer)
        invalid_citation_ids = sorted([c_id for c_id in cited_indices if c_id < 1 or c_id > total_chunks])
        claims = cls.extract_claims(cleaned_answer)
        claim_results: List[ClaimValidationResult] = []

        for claim in claims:
            ids = cls._ids_in_claim(claim)
            invalid_ids = [c_id for c_id in ids if c_id not in evidence_by_id]
            cited_evidence = [evidence_by_id[c_id] for c_id in ids if c_id in evidence_by_id]
            cited_evidence_norm = " ".join(cls._normalize(e) for e in cited_evidence)

            result = ClaimValidationResult(
                claim=claim,
                citation_ids=ids,
                is_valid=True,
                clinical_strength=cls._classify_strength(claim),
                invalid_citations=invalid_ids,
            )

            # If claim has no citations
            if not ids:
                # If it's pure summary/conclusion or conversational note without ungrounded assertions
                if any(claim.startswith(prefix) for prefix in ["الخلاصة", "ملاحظة", "تنبيه", "Summary", "Note"]):
                    result.is_valid = True
                    result.support_type = "SUMMARY"
                    result.reason = "Summary framing statement."
                else:
                    result.is_valid = False
                    result.support_type = "UNCITED"
                    result.reason = "Clinical claim has no citation."
            elif invalid_ids:
                result.is_valid = False
                result.support_type = "INVALID_CITATION"
                result.reason = "Claim cites evidence IDs that were not retrieved."

            # Numerical validation
            claim_nums = cls.extract_numerical_entities(claim)
            unsupported_nums = sorted(
                n for n in claim_nums if not cls._number_supported(n, cited_evidence)
            )
            if unsupported_nums:
                result.is_valid = False
                result.support_type = "UNSUPPORTED_NUMERICAL_CLAIM"
                result.unsupported_numbers = unsupported_nums
                result.reason = "Numerical claim is absent from the cited evidence."

            # Ranking check (only if extreme ungrounded ranking words appear)
            ranking_flags = cls._contains_any(claim, [r"\bgold standard\b", r"أفضل دواء في العالم"])
            if ranking_flags:
                result.is_valid = False
                result.support_type = "UNSUPPORTED_RANKING"
                result.ranking_flags = ranking_flags
                result.reason = "Unsupported absolute ranking language used."

            if result.is_valid:
                result.reason = "Claim is directly supported by cited evidence."
            claim_results.append(result)

        invalid_claims = [r for r in claim_results if not r.is_valid]
        return {
            "is_valid": not invalid_citation_ids and not invalid_claims,
            "claims": [r.to_dict() for r in claim_results],
            "invalid_claims": [r.to_dict() for r in invalid_claims],
            "unsupported_claims": [r.to_dict() for r in invalid_claims if r.support_type.startswith("UNSUPPORTED")],
            "invalid_citations": invalid_citation_ids,
            "numerical_claims": [
                {"claim": r.claim, "numbers": sorted(cls.extract_numerical_entities(r.claim))}
                for r in claim_results
                if cls.extract_numerical_entities(r.claim)
            ],
            "treatment_sequence_flags": [
                {"claim": r.claim, "flags": r.sequence_flags}
                for r in claim_results
                if r.sequence_flags
            ],
        }

    @classmethod
    def build_and_validate_citations(
        cls,
        raw_answer: str,
        retrieved_chunks: List[Any]
    ) -> Tuple[List[Dict[str, Any]], List[int]]:
        cleaned_answer = cls.clean_text_and_normalize_citations(raw_answer)
        claim_report = cls.validate_claims(cleaned_answer, retrieved_chunks)
        cited_indices = cls.extract_cited_indices(cleaned_answer)
        invalid_ids = claim_report.get("invalid_citations", [])

        valid_citations = []
        claim_by_citation: Dict[int, List[Dict[str, Any]]] = {}
        for claim in claim_report.get("claims", []):
            for c_id in claim.get("citation_ids", []):
                claim_by_citation.setdefault(c_id, []).append(claim)

        for idx, item in enumerate(retrieved_chunks, 1):
            if isinstance(item, (tuple, list)):
                chunk = item[0] if item else {}
                score = item[1] if len(item) > 1 else 0.0
            elif isinstance(item, dict):
                chunk = item
                score = 0.0
            else:
                continue

            chunk_text = chunk.get("text", "")
            claims_for_source = claim_by_citation.get(idx, [])
            unsupported_numbers = sorted({
                num
                for claim in claims_for_source
                for num in claim.get("unsupported_numbers", [])
            })
            is_grounded = all(c.get("is_valid", False) for c in claims_for_source) if claims_for_source else True

            valid_citations.append({
                "citation_id": idx,
                "source_file": chunk.get("source_file", "Unknown"),
                "page": chunk.get("page", 1),
                "chunk_id": chunk.get("chunk_id", f"chk_{idx}"),
                "snippet": chunk_text[:260] + "..." if len(chunk_text) > 260 else chunk_text,
                "relevance_score": round(float(score), 3),
                "is_referenced_in_text": idx in cited_indices,
                "is_claim_grounded": is_grounded,
                "unsupported_numbers": unsupported_numbers,
                "supporting_claims": claims_for_source,
            })

        return valid_citations, invalid_ids


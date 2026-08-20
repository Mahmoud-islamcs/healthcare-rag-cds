import re
from typing import Tuple, Optional

class MedicalSafetyGuardrails:
    EMERGENCY_KEYWORDS = [
        r'\b(chest\s+pain|heart\s+attack|cardiac\s+arrest|severe\s+chest\s+pressure)\b',
        r'\b(can(\x27t|not|\s+not)\s+breathe|trouble\s+breathing|difficulty\s+breathing|shortness\s+of\s+breath|suffocat(ing|ion))\b',
        r'\b(stroke|facial\s+droop|slurred\s+speech|sudden\s+numbness|sudden\s+paralysis)\b',
        r'\b(severe\s+bleeding|hemorrhag(e|ic)|uncontrolled\s+bleeding)\b',
        r'\b(unconscious(ness)?|passed\s+out|unresponsive|syncope|coma)\b',
        r'\b(ketoacidosis|dka|hyperosmolar|severe\s+hypoglyc)\b',
        r'\b(anaphylaxis|severe\s+allergic\s+reaction|throat\s+swelling)\b',
        r'\b(overdose|poison(ing)?|toxic\s+ingestion)\b',
        r'\b(suicid(e|al)|kill\s+myself|end\s+my\s+life|end\s+it\s+all|want\s+to\s+die|self[\s\-]harm)\b',
        r'(ألم\s+شديد\s+في\s+الصدر|ذبحة\s+صدرية|أزمة\s+قلبية|جلطة|توقف\s+التنفس|صعوبة\s+شديدة\s+في\s+التنفس|نزيف\s+حاد|فقدان\s*(?:ال|ل)?وعي|غيبوبة|حماض\s*كيتون|نقص\s*سكر\s*شديد|تسمم|انتحار)'
    ]

    CONVERSATIONAL_KEYWORDS = [
        r'^(hi|hello|hey|howdy|good\s+(morning|afternoon|evening|day)|greetings)(\b|\s|$)',
        r'^(how\s+are\s+you|how\s+is\s+it\s+going|what(\x27s|\s+is)\s+up|how\s+do\s+you\s+do)(\b|\s|$)',
        r'^(who\s+are\s+you|what\s+can\s+you\s+do|what\s+is\s+your\s+name|what\s+are\s+you)(\b|\s|$)',
        r'^(thanks|thank\s+you|thx|much\s+appreciated|bye|goodbye|see\s+you)(\b|\s|$)',
        r'^(مرحبا|أهلا|اهلا|هاي|ازيك|ازي\s*حضرتك|عامل\s*ايه|عامل\s*ايه\s*يا|كيف\s*حالك|كيفك|شخبارك|صباح\s*الخير|مساء\s*الخير|السلام\s*عليكم|سلام|تحياتي)(\b|\s|$)',
        r'^(من\s*أنت|من\s*انت|ماذا\s*تستطيع\s*أن\s*تفعل|ما\s*هي\s*قدراتك|عرف\s*نفسك|مين\s*انت)(\b|\s|$)',
        r'^(شكرا|شكرًا|تسلم|يعطيك\s*العافية|جزاك\s*الله\s*خيرا|مع\s*السلامة|باي)(\b|\s|$)'
    ]

    @classmethod
    def check_conversational_query(cls, query: str) -> Tuple[bool, Optional[str]]:
        if not query or not query.strip():
            return False, None
        cleaned_query = query.strip().lower()
        # Clean punctuation and normalize Arabic
        cleaned_query = re.sub(r'[?!.,;:\-_؟،]', '', cleaned_query).strip()
        norm_query = re.sub(r'[إأآ]', 'ا', cleaned_query)
        norm_query = re.sub(r'ة', 'ه', norm_query)
        norm_query = re.sub(r'ى', 'ي', norm_query)
        
        for pattern in cls.CONVERSATIONAL_KEYWORDS:
            if re.search(pattern, cleaned_query) or re.search(pattern, norm_query):
                is_arabic = bool(re.search(r'[\u0600-\u06FF]', query))
                if is_arabic:
                    return True, (
                        "أهلاً بك! أنا **BioGuard**، المساعد الإكلينيكي المعتمد على الأدلة والمراجع الطبية الموثقة.\n\n"
                        "أنا هنا لمساعدتك في استعراض وتلخيص الأدلة السريرية والإرشادات الطبية بدقة.\n"
                        "كيف يمكنني مساعدتك في استفساراتك الطبية اليوم؟"
                    )
                else:
                    return True, (
                        "Hello! I am **BioGuard**, your evidence-grounded clinical medical assistant.\n\n"
                        "I am here to assist you with synthesis from indexed medical guidelines, clinical trials, and pharmacology references.\n"
                        "How can I assist your clinical inquiry today?"
                    )
        return False, None

    @classmethod
    def check_query_safety(cls, query: str) -> Tuple[bool, Optional[str]]:

        if not query or not query.strip():
            return True, None
        query_lower = query.lower()
        for pattern in cls.EMERGENCY_KEYWORDS:
            if re.search(pattern, query_lower):
                return False, (
                    "🚨 **EMERGENCY SAFETY ALERT**: If you or someone around you is experiencing "
                    "severe or life-threatening symptoms (e.g. chest pain, severe breathing difficulty, stroke signs, "
                    "uncontrolled bleeding, severe trauma, or acute distress), please call your local emergency services "
                    "(e.g., 911, 999, 112, or local emergency number) immediately.\n\n"
                    "⚠️ *This AI system is strictly an informational document-retrieval reference and cannot provide emergency triage or direct medical care.*"
                )
        return True, None


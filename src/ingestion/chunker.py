import re
import uuid
from typing import List, Dict, Any

class MedicalAwareChunker:
    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 120, min_chunk_size: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks = []
        for doc in documents:
            text = doc["text"]
            source = doc.get("source_file", "unknown")
            page = doc.get("page", 1)

            section_splits = re.split(r'\n(?=[A-Z\u0600-\u06FF][A-Za-z\u0600-\u06FF\s]{2,30}:|\d+\.\s+[A-Z\u0600-\u06FF]|\b[IVXLCDM]+\.\s+)', text)

            for section in section_splits:
                section_text = section.strip()
                if not section_text:
                    continue

                if len(section_text) <= self.chunk_size:
                    if len(section_text) >= self.min_chunk_size:
                        chunks.append(self._create_chunk_dict(section_text, source, page))
                else:
                    sub_chunks = self._sliding_window_split(section_text)
                    for sc in sub_chunks:
                        chunks.append(self._create_chunk_dict(sc, source, page))
        return chunks

    def _sliding_window_split(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?؟])\s+', text)
        result = []
        current = []
        current_len = 0

        for sentence in sentences:
            sentence_len = len(sentence)
            if current_len + sentence_len > self.chunk_size and current:
                chunk_str = " ".join(current).strip()
                if len(chunk_str) >= self.min_chunk_size:
                    result.append(chunk_str)
                overlap_tokens = []
                overlap_len = 0
                for s in reversed(current):
                    if overlap_len + len(s) < self.chunk_overlap:
                        overlap_tokens.insert(0, s)
                        overlap_len += len(s)
                    else:
                        break
                current = overlap_tokens
                current_len = overlap_len

            current.append(sentence)
            current_len += sentence_len

        if current:
            chunk_str = " ".join(current).strip()
            if len(chunk_str) >= self.min_chunk_size:
                result.append(chunk_str)
        return result

    def _create_chunk_dict(self, text: str, source: str, page: int) -> Dict[str, Any]:
        return {
            "chunk_id": f"chk_{uuid.uuid4().hex}",
            "text": text,
            "source_file": source,
            "page": page,
        }

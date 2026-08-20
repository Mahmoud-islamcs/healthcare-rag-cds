import os
import json
import logging
import pandas as pd
from typing import List, Dict, Any
from pypdf import PdfReader
from src.ingestion.text_cleaner import clean_medical_text

logger = logging.getLogger(__name__)

class UniversalDocumentLoader:
    @staticmethod
    def load_file(file_path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        filename = os.path.basename(file_path)

        if ext == ".pdf":
            return UniversalDocumentLoader._load_pdf(file_path, filename)
        elif ext == ".docx":
            return UniversalDocumentLoader._load_docx(file_path, filename)
        elif ext in [".txt", ".md"]:
            return UniversalDocumentLoader._load_txt(file_path, filename)
        elif ext == ".csv":
            return UniversalDocumentLoader._load_csv(file_path, filename)
        elif ext == ".json":
            return UniversalDocumentLoader._load_json(file_path, filename)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    @staticmethod
    def _load_pdf(path: str, filename: str) -> List[Dict[str, Any]]:
        docs = []
        try:
            reader = PdfReader(path)
            for idx, page in enumerate(reader.pages):
                try:
                    raw_text = page.extract_text() or ""
                    cleaned = clean_medical_text(raw_text)
                    if cleaned:
                        docs.append({
                            "text": cleaned,
                            "source_file": filename,
                            "page": idx + 1,
                            "file_type": "pdf"
                        })
                except Exception as e:
                    logger.warning(f"Error parsing page {idx + 1} of {filename}: {e}")
                    continue
        except Exception as e:
            logger.error(f"Failed to open PDF {filename}: {e}")
            raise e
        return docs

    @staticmethod
    def _load_docx(path: str, filename: str) -> List[Dict[str, Any]]:
        docs = []
        try:
            import docx
        except ImportError:
            raise ImportError(
                "python-docx is not installed in the active environment. "
                "Please run: pip install python-docx"
            )
        try:
            doc = docx.Document(path)
            current_paragraphs = []
            page_estimate = 1
            for p in doc.paragraphs:
                p_text = p.text.strip()
                if not p_text:
                    continue
                current_paragraphs.append(p_text)
                if len('\n'.join(current_paragraphs)) > 3000:
                    cleaned = clean_medical_text('\n'.join(current_paragraphs))
                    if cleaned:
                        docs.append({
                            "text": cleaned,
                            "source_file": filename,
                            "page": page_estimate,
                            "file_type": "docx"
                        })
                    current_paragraphs = []
                    page_estimate += 1
            if current_paragraphs:
                cleaned = clean_medical_text('\n'.join(current_paragraphs))
                if cleaned:
                    docs.append({
                        "text": cleaned,
                        "source_file": filename,
                        "page": page_estimate,
                        "file_type": "docx"
                    })
        except Exception as e:
            logger.error(f"Failed to read DOCX {filename}: {e}")
            raise e
        return docs

    @staticmethod
    def _load_txt(path: str, filename: str) -> List[Dict[str, Any]]:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            cleaned = clean_medical_text(content)
            return [{
                "text": cleaned,
                "source_file": filename,
                "page": 1,
                "file_type": "txt"
            }] if cleaned else []
        except Exception as e:
            logger.error(f"Failed to read text file {filename}: {e}")
            raise e

    @staticmethod
    def _load_csv(path: str, filename: str) -> List[Dict[str, Any]]:
        docs = []
        try:
            df = pd.read_csv(path)
            for idx, row in df.iterrows():
                row_str = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                cleaned = clean_medical_text(row_str)
                if cleaned:
                    docs.append({
                        "text": cleaned,
                        "source_file": filename,
                        "page": idx + 1,
                        "file_type": "csv"
                    })
        except Exception as e:
            logger.error(f"Failed to read CSV {filename}: {e}")
            raise e
        return docs

    @staticmethod
    def _load_json(path: str, filename: str) -> List[Dict[str, Any]]:
        docs = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for idx, item in enumerate(data):
                    cleaned = clean_medical_text(json.dumps(item, ensure_ascii=False))
                    if cleaned:
                        docs.append({
                            "text": cleaned,
                            "source_file": filename,
                            "page": idx + 1,
                            "file_type": "json"
                        })
            else:
                cleaned = clean_medical_text(json.dumps(data, ensure_ascii=False))
                if cleaned:
                    docs.append({
                        "text": cleaned,
                        "source_file": filename,
                        "page": 1,
                        "file_type": "json"
                    })
        except Exception as e:
            logger.error(f"Failed to read JSON {filename}: {e}")
            raise e
        return docs

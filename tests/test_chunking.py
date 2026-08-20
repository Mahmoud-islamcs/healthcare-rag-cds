import pytest
from src.ingestion.chunker import MedicalAwareChunker

def test_chunker_basic_splitting():
    chunker = MedicalAwareChunker(chunk_size=300, chunk_overlap=50, min_chunk_size=30)
    docs = [{
        "text": "Hypertension is a chronic medical condition. Blood pressure is persistently elevated. Normal range is under 120/80 mmHg. Treatment includes ACE inhibitors and lifestyle modifications.",
        "source_file": "cardio.txt",
        "page": 1
    }]
    chunks = chunker.chunk_documents(docs)
    assert len(chunks) >= 1
    for c in chunks:
        assert c["chunk_id"].startswith("chk_")
        assert len(c["chunk_id"]) > 10
        assert c["source_file"] == "cardio.txt"
        assert c["page"] == 1
        assert len(c["text"]) >= 30

def test_chunker_section_splitting():
    chunker = MedicalAwareChunker(chunk_size=500, chunk_overlap=50, min_chunk_size=30)
    text = (
        "Diagnosis:\nPatient presents with stage 2 hypertension.\n"
        "Treatment:\nPrescribe lisinopril 10mg daily. Monitor blood pressure weekly.\n"
        "Prognosis:\nExpected favorable outcome with medication compliance."
    )
    docs = [{"text": text, "source_file": "patient_note.txt", "page": 2}]
    chunks = chunker.chunk_documents(docs)
    assert len(chunks) >= 2

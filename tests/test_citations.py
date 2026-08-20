import pytest
from src.safety.citation_validator import CitationValidator

def test_citation_extraction():
    text = "Insulin is indicated in severe hyperglycemia [1]. Also consider DPP-4 inhibitors [2, 3] or GLP-1 [4][5]."
    indices = CitationValidator.extract_cited_indices(text)
    assert indices == {1, 2, 3, 4, 5}

def test_citation_validation_with_chunks():
    raw_answer = "Metformin is first-line [1]. Add insulin when HbA1c is high [2]."
    chunks = [
        ({"chunk_id": "c1", "source_file": "doc1.pdf", "page": 5, "text": "Metformin therapy"}, 0.9),
        ({"chunk_id": "c2", "source_file": "doc1.pdf", "page": 10, "text": "Insulin indications"}, 0.8),
        ({"chunk_id": "c3", "source_file": "doc2.pdf", "page": 2, "text": "Lifestyle advice"}, 0.5),
    ]
    verified, invalid = CitationValidator.build_and_validate_citations(raw_answer, chunks)
    assert len(verified) == 3
    assert verified[0]["is_referenced_in_text"] is True
    assert verified[1]["is_referenced_in_text"] is True
    assert verified[2]["is_referenced_in_text"] is False
    assert invalid == []

def test_citation_out_of_bounds_detection():
    raw_answer = "Unverified claim citing non-existent chunk [99]."
    chunks = [
        ({"chunk_id": "c1", "source_file": "doc1.pdf", "page": 1, "text": "Evidence 1"}, 0.9)
    ]
    verified, invalid = CitationValidator.build_and_validate_citations(raw_answer, chunks)
    assert invalid == [99]

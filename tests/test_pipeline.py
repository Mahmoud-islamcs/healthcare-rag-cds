import pytest
from unittest.mock import MagicMock
from src.pipeline.rag_pipeline import MedicalRAGPipeline

def test_pipeline_emergency_intercept():
    mock_retriever = MagicMock()
    mock_llm = MagicMock()
    settings = {"safety": {"confidence_threshold": 0.35, "disclaimer": "Medical Disclaimer"}}
    pipeline = MedicalRAGPipeline(mock_retriever, mock_llm, settings)

    res = pipeline.answer_query("I have severe chest pain and can not breathe")
    assert res["status"] == "EMERGENCY_TRIGGERED"
    assert "EMERGENCY SAFETY ALERT" in res["answer"]
    mock_retriever.retrieve.assert_not_called()
    mock_llm.generate.assert_not_called()

def test_pipeline_insufficient_evidence_case_b():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []
    mock_llm = MagicMock()
    settings = {"safety": {"confidence_threshold": 0.35, "disclaimer": "Medical Disclaimer"}}
    pipeline = MedicalRAGPipeline(mock_retriever, mock_llm, settings)

    res = pipeline.answer_query("What is the protocol for rare disease XYZ?")
    assert res["status"] == "INSUFFICIENT_EVIDENCE"
    assert "not provide sufficient evidence" in res["answer"]
    assert res["sources"] == []
    assert res["evidence_quality"] == "Insufficient"
    # Crucial: LLM must NEVER be called when evidence is insufficient
    mock_llm.generate.assert_not_called()

def test_pipeline_successful_generation_case_a():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        ({"text": "Lisinopril is an ACE inhibitor used to treat hypertension in adults with diabetes.", "source_file": "cardio.pdf", "page": 10}, 0.85)
    ]
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "### Clinical Recommendations\nLisinopril lowers blood pressure [1]."
    settings = {"safety": {"confidence_threshold": 0.35, "disclaimer": "Medical Disclaimer"}}
    pipeline = MedicalRAGPipeline(mock_retriever, mock_llm, settings)

    res = pipeline.answer_query("What is Lisinopril used for?")
    assert res["status"] == "SUCCESS"
    assert "Lisinopril lowers blood pressure [1]." in res["answer"]
    assert len(res["sources"]) == 1
    assert res["sources"][0]["source_file"] == "cardio.pdf"
    assert res["sources"][0]["page"] == 10
    assert res["sources"][0]["is_referenced_in_text"] is True
    # Single disclaimer test
    assert res["answer"].count("Medical Disclaimer") == 1

def test_pipeline_debug_observability():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = (
        [({"text": "Clinical chunk text for diabetes management protocol.", "source_file": "doc.pdf", "page": 1}, 0.8)],
        {"dense_results": [], "bm25_results": [], "rrf_candidates": []}
    )
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "Answer [1]"
    settings = {"safety": {"confidence_threshold": 0.35, "disclaimer": "Disclaimer"}}
    pipeline = MedicalRAGPipeline(mock_retriever, mock_llm, settings)

    res = pipeline.answer_query("diabetes management", return_debug=True)
    assert res["debug"] is not None
    assert "retrieval_query" in res["debug"]
    assert "evidence_validation" in res["debug"]

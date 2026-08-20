from unittest.mock import MagicMock

from src.pipeline.rag_pipeline import MedicalRAGPipeline
from src.safety.citation_validator import CitationValidator


def _chunk(text, score=0.9):
    return ({"text": text, "source_file": "guideline.pdf", "page": 1}, score)


def test_insulin_option_does_not_require_sequence():
    answer = "Insulin may be considered for patients requiring additional therapy [1]."
    chunks = [_chunk("Insulin is an option for patients requiring additional therapy.")]
    report = CitationValidator.validate_claims(answer, chunks)
    assert report["is_valid"] is True


def test_arabic_insulin_option_can_cite_english_evidence():
    answer = "يمكن اعتبار الإنسولين أحد الخيارات للأشخاص المصابين بالسكري من النوع الثاني الذين يحتاجون إلى أدوية إضافية للوصول إلى أهدافهم الفردية لسكر الدم [1]."
    chunks = [_chunk("Type 2 diabetes: insulin is an option for people who need further medicines to reach their individualised glycaemic targets.")]
    report = CitationValidator.validate_claims(answer, chunks)
    assert report["is_valid"] is True


def test_separate_glp_dpp_insulin_options_do_not_support_sequence():
    answer = "Use GLP-1 first, then DPP-4, then insulin as the next step [1, 2, 3]."
    chunks = [
        _chunk("GLP-1 therapy is an option in this clinical context."),
        _chunk("DPP-4 inhibitors are an option in another clinical context."),
        _chunk("Insulin is an option for patients requiring additional therapy."),
    ]
    report = CitationValidator.validate_claims(answer, chunks)
    assert report["is_valid"] is False
    assert any(c["support_type"] == "UNSUPPORTED_TREATMENT_SEQUENCE" for c in report["invalid_claims"])


def test_may_be_considered_must_not_be_strengthened_to_recommended():
    answer = "GLP-1 therapy is recommended [1]."
    chunks = [_chunk("GLP-1 therapy may be considered for selected adults.")]
    report = CitationValidator.validate_claims(answer, chunks)
    assert report["is_valid"] is False
    assert any(c["support_type"] == "RECOMMENDATION_STRENGTHENING" for c in report["invalid_claims"])


def test_supported_hba1c_threshold_is_allowed():
    answer = "HbA1c >= 9% is stated in the cited recommendation [1]."
    chunks = [_chunk("Consider treatment intensification when HbA1c >= 9%.")]
    report = CitationValidator.validate_claims(answer, chunks)
    assert report["is_valid"] is True


def test_unsupported_threshold_is_rejected():
    answer = "HbA1c >= 9% is the threshold for intensification [1]."
    chunks = [_chunk("Treatment intensification may be considered when glycaemic targets are not met.")]
    report = CitationValidator.validate_claims(answer, chunks)
    assert report["is_valid"] is False
    assert any(c["support_type"] == "UNSUPPORTED_NUMERICAL_CLAIM" for c in report["invalid_claims"])


def test_basal_insulin_initial_insulin_therapy_allowed():
    answer = "Basal insulin is initial insulin therapy [1]."
    chunks = [_chunk("Offer basal insulin as initial insulin therapy.")]
    report = CitationValidator.validate_claims(answer, chunks)
    assert report["is_valid"] is True


def test_initial_insulin_therapy_does_not_support_first_line_diabetes_therapy():
    answer = "Basal insulin is first-line diabetes therapy [1]."
    chunks = [_chunk("Offer basal insulin as initial insulin therapy.")]
    report = CitationValidator.validate_claims(answer, chunks)
    assert report["is_valid"] is False
    assert any(c["support_type"] == "UNSUPPORTED_TREATMENT_SEQUENCE" for c in report["invalid_claims"])


def test_arabic_exercise_answer_must_not_add_medications():
    answer = "يمكن لمريض السكري ممارسة الرياضة عند اتباع إرشادات النشاط البدني المذكورة [1]. لا يلزم استخدام الإنسولين [1]."
    chunks = [_chunk("Adults with diabetes can exercise as part of physical activity advice.")]
    report = CitationValidator.validate_claims(answer, chunks)
    assert report["is_valid"] is False
    assert any("insulin" in " ".join(c["unsupported_terms"]) or "الإنسولين" in " ".join(c["unsupported_terms"]) for c in report["invalid_claims"])


def test_best_drug_claim_rejected_without_ranking_evidence():
    answer = "Metformin is the best drug for diabetes [1]."
    chunks = [_chunk("Metformin can be used as a glucose-lowering therapy for type 2 diabetes.")]
    report = CitationValidator.validate_claims(answer, chunks)
    assert report["is_valid"] is False
    assert any(c["support_type"] == "UNSUPPORTED_RANKING" for c in report["invalid_claims"])


def test_unrelated_or_uncited_medical_claim_is_invalid():
    answer = "A rare unrelated disease should be treated with Drug X."
    chunks = [_chunk("This source discusses diabetes exercise advice.")]
    report = CitationValidator.validate_claims(answer, chunks)
    assert report["is_valid"] is False
    assert any(c["support_type"] == "UNCITED" for c in report["invalid_claims"])


def test_pipeline_regenerates_once_then_abstains_on_unsafe_answer():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        _chunk("GLP-1 therapy is an option."),
        _chunk("DPP-4 inhibitors are an option."),
        _chunk("Insulin is an option for patients requiring additional therapy."),
    ]
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "Use GLP-1 first, then DPP-4, then insulin [1, 2, 3]."
    settings = {"safety": {"confidence_threshold": 0.35, "disclaimer": "Medical Disclaimer"}}
    pipeline = MedicalRAGPipeline(mock_retriever, mock_llm, settings)

    res = pipeline.answer_query("When should insulin be added?", return_debug=True)
    assert mock_llm.generate.call_count == 2
    assert res["status"] == "UNSAFE_GENERATION_REJECTED"
    assert "not provide sufficient evidence" in res["answer"]
    assert res["citations_verified_count"] == 0
    assert res["debug"]["citation_validation"]["invalid_claims"]


def test_pipeline_accepts_safe_arabic_regeneration_after_unsafe_first_draft():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        _chunk("Type 2 diabetes: insulin is an option for people who need further medicines to reach their individualised glycaemic targets.")
    ]
    mock_llm = MagicMock()
    mock_llm.generate.side_effect = [
        "insulin initiation type 2 diabetes further medicines individualised glycaemic targets",
        "استخدم GLP-1 أولاً، ثم DPP-4، ثم الإنسولين [1].",
        "يمكن اعتبار الإنسولين أحد الخيارات للأشخاص المصابين بالسكري من النوع الثاني الذين يحتاجون إلى أدوية إضافية للوصول إلى أهدافهم الفردية لسكر الدم [1].",
    ]
    settings = {"safety": {"confidence_threshold": 0.35, "disclaimer": "Medical Disclaimer"}}
    pipeline = MedicalRAGPipeline(mock_retriever, mock_llm, settings)

    res = pipeline.answer_query("متى يُوصى بإضافة الإنسولين إلى خطة علاج مريض السكري من النوع الثاني؟", return_debug=True)
    assert mock_llm.generate.call_count == 3
    assert res["status"] == "SUCCESS"
    assert "يمكن اعتبار الإنسولين" in res["answer"]
    assert res["citations_verified_count"] == 1

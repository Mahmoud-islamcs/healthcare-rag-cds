import pytest
from src.safety.guardrails import MedicalSafetyGuardrails
from src.safety.evidence_validator import EvidenceValidator

def test_emergency_chest_pain():
    is_safe, msg = MedicalSafetyGuardrails.check_query_safety("Patient is having severe chest pain and pressure")
    assert is_safe is False
    assert "EMERGENCY SAFETY ALERT" in msg

def test_emergency_breathing_variations():
    queries = [
        "The patient cannot breathe properly",
        "He can not breathe at all",
        "Experiencing acute shortness of breath",
        "Child is suffocating after eating"
    ]
    for q in queries:
        is_safe, msg = MedicalSafetyGuardrails.check_query_safety(q)
        assert is_safe is False, f"Failed to catch emergency in: {q}"

def test_emergency_self_harm_and_stroke():
    queries = [
        "Patient exhibits sudden facial droop and slurred speech",
        "I feel hopeless and want to end my life",
        "Severe bleeding from deep laceration"
    ]
    for q in queries:
        is_safe, msg = MedicalSafetyGuardrails.check_query_safety(q)
        assert is_safe is False, f"Failed to catch emergency in: {q}"

def test_routine_query_passes():
    queries = [
        "What are the common symptoms of mild hypertension?",
        "Explain the mechanism of action of metformin.",
        "What are the standard follow-up intervals in oncology?"
    ]
    for q in queries:
        is_safe, msg = MedicalSafetyGuardrails.check_query_safety(q)
        assert is_safe is True
        assert msg is None

def test_evidence_validator_threshold():
    validator = EvidenceValidator(min_confidence_threshold=0.35)
    
    # Empty chunks
    is_sufficient, score, quality, reason = validator.evaluate_retrieved_evidence([])
    assert is_sufficient is False
    assert score == 0.0
    assert quality == "Insufficient"

    # Low score chunks
    low_chunks = [({"text": "A standard medical text paragraph describing general care."}, 0.15)]
    is_sufficient, score, quality, reason = validator.evaluate_retrieved_evidence(low_chunks)
    assert is_sufficient is False
    assert score < 0.35
    assert quality == "Insufficient"

    # High score chunks
    high_chunks = [({"text": "Clinical oncology protocol indicates standard chemotherapy dosage."}, 0.75)]
    is_sufficient, score, quality, reason = validator.evaluate_retrieved_evidence(high_chunks)
    assert is_sufficient is True
    assert score >= 0.35
    assert quality in ["High", "Moderate"]

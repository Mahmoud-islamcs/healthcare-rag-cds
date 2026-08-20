import os
import tempfile
import json
import pytest
from src.ingestion.document_loader import UniversalDocumentLoader

def test_load_txt():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("Clinical guideline for diabetes type 2 management.")
        temp_path = f.name
    try:
        docs = UniversalDocumentLoader.load_file(temp_path)
        assert len(docs) == 1
        assert "diabetes type 2" in docs[0]["text"]
        assert docs[0]["file_type"] == "txt"
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def test_load_csv():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write("drug,dosage,indication\nMetformin,500mg,Diabetes Type 2\nLisinopril,10mg,Hypertension")
        temp_path = f.name
    try:
        docs = UniversalDocumentLoader.load_file(temp_path)
        assert len(docs) == 2
        assert docs[0]["file_type"] == "csv"
        assert "Metformin" in docs[0]["text"]
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def test_load_json():
    data = [
        {"protocol": "chemotherapy_a", "agent": "Cisplatin", "cycle_days": 21},
        {"protocol": "chemotherapy_b", "agent": "Doxorubicin", "cycle_days": 14}
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(data, f)
        temp_path = f.name
    try:
        docs = UniversalDocumentLoader.load_file(temp_path)
        assert len(docs) == 2
        assert docs[0]["file_type"] == "json"
        assert "Cisplatin" in docs[0]["text"]
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def test_unsupported_format_raises():
    with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
        temp_path = f.name
    try:
        with pytest.raises(ValueError, match="Unsupported file format"):
            UniversalDocumentLoader.load_file(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

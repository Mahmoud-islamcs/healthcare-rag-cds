import os
import tempfile
import numpy as np
import pytest
from src.retrieval.bm25_retriever import LocalBM25Retriever
from src.retrieval.vector_store import LocalFAISSVectorStore

def test_bm25_search_and_persistence():
    retriever = LocalBM25Retriever()
    chunks = [
        {"chunk_id": "c1", "text": "Cardiovascular risk reduction using statin therapy."},
        {"chunk_id": "c2", "text": "Oncology chemotherapy regimens for breast cancer."},
        {"chunk_id": "c3", "text": "Pediatric immunization schedule and vaccine guidance."}
    ]
    retriever.index_chunks(chunks)
    results = retriever.search("statin cardiovascular", top_k=2)
    assert len(results) >= 1
    assert results[0][0]["chunk_id"] == "c1"

    with tempfile.TemporaryDirectory() as tmpdir:
        retriever.save(tmpdir)
        new_retriever = LocalBM25Retriever()
        new_retriever.load(tmpdir)
        loaded_results = new_retriever.search("statin cardiovascular", top_k=2)
        assert len(loaded_results) >= 1
        assert loaded_results[0][0]["chunk_id"] == "c1"

def test_faiss_vector_store_persistence():
    dim = 64
    store = LocalFAISSVectorStore(embedding_dim=dim)
    
    chunks = [
        {"chunk_id": "c1", "text": "Medical chunk 1", "source_file": "doc1.pdf", "page": 1},
        {"chunk_id": "c2", "text": "Medical chunk 2", "source_file": "doc2.pdf", "page": 5}
    ]
    
    emb1 = np.random.randn(dim).astype(np.float32)
    emb1 /= np.linalg.norm(emb1)
    emb2 = np.random.randn(dim).astype(np.float32)
    emb2 /= np.linalg.norm(emb2)
    embeddings = np.stack([emb1, emb2])

    store.add_chunks(chunks, embeddings)
    assert store.index.ntotal == 2

    # Query matching emb1
    results = store.search(emb1, top_k=2)
    assert len(results) == 2
    assert results[0][0]["chunk_id"] == "c1"

    with tempfile.TemporaryDirectory() as tmpdir:
        store.save(tmpdir)
        loaded_store = LocalFAISSVectorStore(embedding_dim=dim)
        loaded_store.load(tmpdir)
        assert loaded_store.index.ntotal == 2
        loaded_results = loaded_store.search(emb1, top_k=1)
        assert loaded_results[0][0]["chunk_id"] == "c1"

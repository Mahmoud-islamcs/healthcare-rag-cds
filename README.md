# ?? BioGuard Medical RAG (Production-Grade Clinical System)

An evidence-based, hallucination-resistant Medical Retrieval-Augmented Generation (RAG) system built with:
- **Hybrid Retrieval:** FAISS Vector Index + Rank-BM25 Lexical Matching.
- **Reranking:** Local BAAI/bge-reranker Cross-Encoder.
- **Medical Safety:** Emergency Guardrails & Evidence Confidence Verification.
- **Inference Options:** Ultra-fast Free Groq Llama-3.3-70B OR 100% Offline Local Model (Phi-3.5 / BioMistral).
- **UI:** Multi-page interactive Streamlit dashboard.

---

## ?? Quick Setup & Run

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Ingest documents in data/raw/
python scripts/ingest.py

# 3. Test Groq Connection
python scripts/test_groq.py

# 4. Run the Streamlit Interface
streamlit run app.py
```

# MODEL AUDIT REPORT

| Component | Selected Model | License | Free & Local? | Commercial Use? | VRAM / RAM | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Primary LLM** | `openai/gpt-oss-120b` (Groq) | Open / Hosted API | YES (Groq Hosted) | YES | 0 MB Local | 120B parameter high-precision clinical reasoning |
| **Arabic Specialist** | `allam-2-7b` (Groq) | Open / Hosted API | YES (Groq Hosted) | YES | 0 MB Local | Ultra-fast Saudi Arabic medical synthesis |
| **Multilingual Reasoning**| `qwen/qwen3.6-27b` (Groq) | Apache-2.0 | YES (Groq Hosted) | YES | 0 MB Local | 27B high-speed clinical reasoning |
| **Offline LLM** | `microsoft/Phi-3.5-mini-instruct` | MIT License | YES (100% Local) | YES | ~4GB RAM | 128k context, runs on CPU/GPU |
| **Embedding** | `BAAI/bge-small-en-v1.5` | MIT License | YES (100% Local) | YES | ~300MB RAM | Top MTEB benchmark score |
| **Reranker** | `BAAI/bge-reranker-base` | MIT License | YES (100% Local) | YES | ~500MB RAM | Cross-Encoder precision filter |
| **Vector DB** | `FAISS FlatIP` | MIT License | YES (100% Local) | YES | Local Disk | Zero cloud dependencies |
| **Lexical Search** | `Rank-BM25` | Apache-2.0 | YES (100% Local) | YES | Local RAM | Exact clinical keyword matching |


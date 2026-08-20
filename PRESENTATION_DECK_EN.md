# 🩺 BioGuard Medical RAG — Clinical Intelligence Platform
## Comprehensive English Presentation Deck (6 High-Impact Slides)

---

<!-- SLIDE 1: Title, Business Problem & Clinical Innovation -->
# 📌 Slide 1: Clinical & Business Problem (The Executive Overview)

### 🏥 The Clinical Problem & Industry Challenge:
* **The Danger of LLM Hallucinations in Healthcare:** In clinical practice (e.g., chronic care & Type 2 Diabetes management), fabricated laboratory thresholds, unverified drug dosages, or invented treatment sequences pose direct, life-threatening risks to **patient safety**.
* **Clinical Modality Escalation:** General-purpose LLMs frequently escalate an optional therapy (*"may be considered"*) into an unjustified mandatory mandate (*"is strongly recommended"*).
* **The Multilingual Medical Gap:** Authoritative clinical guidelines (e.g., NICE Guidelines) are published in English, whereas healthcare practitioners and patients frequently query in Arabic, requiring rigorous, error-free bilingual synthesis.

---

### 💡 The Innovation: BioGuard Medical RAG System
* **Closed-Domain, Zero-Hallucination Architecture:** Grounded strictly and exclusively on indexed clinical evidence with zero tolerance for external speculation.
* **Seamless Bilingual Synthesis (AR / EN):** Translates user intents, queries English clinical guidelines, and synthesizes fluent, professional Arabic/English clinical responses.
* **Deterministic Claim-Level Verification:** Automated programmatic auditing for every single claim, numerical value, and citation tag `[1]` prior to rendering.

---

<!-- SLIDE 2: Data Sources, Chunking & Embeddings -->
# 📌 Slide 2: Data Sources, Chunking Strategy & Embeddings

### 📚 1. Authoritative Clinical Data Sources
* Official Evidence-Based Clinical Practice Guidelines (**NICE Guidelines & Diabetes Clinical Management Protocols**).
* Storage Path: `data/raw/` (Indexed PDFs covering HbA1c targets, insulin initiation criteria, SGLT-2 inhibitors, GLP-1 RAs, Metformin, and lifestyle interventions).

---

### ✂️ 2. Medical-Structure-Aware Chunking Strategy
* **Implementation:** `src/ingestion/chunker.py`
* **Configuration Knobs:** `chunk_size = 700 chars` | `chunk_overlap = 120 chars` | `min_chunk_size = 100 chars`.
* **Workflow:**
  1. Structural boundary parsing based on medical chapter numbering, roman numerals, and clinical section headers.
  2. For sections exceeding 700 characters, applies a **Sentence-Level Sliding Window** (`. ! ؟ ?`) with a 120-character contextual overlap, ensuring clinical terms, numbers, and drug combinations are never truncated.

---

### 🧠 3. Embedding Model & Vector Storage
* **Embedding Model:** `BAAI/bge-small-en-v1.5` (384-dimensional dense vectors with full $L_2$ Normalization).
* **Vector Store:** `FAISS FlatIP` (100% Local, zero-cloud dependency, high-speed Cosine Similarity index).

---

<!-- SLIDE 3: Hybrid Retrieval, Neural Reranking & Similarity Measurement -->
# 📌 Slide 3: Hybrid Retrieval & Multi-Stage Similarity Measurement

### 🔄 1. Hybrid Retrieval Architecture (Dense + Sparse Fusion)
* **Dense Semantic Search (60%):** FAISS Vector Store captures semantic intent and clinical context.
* **Sparse Lexical Search (40%):** `Rank-BM25Plus` captures exact pharmaceutical names, acronyms (e.g., eGFR, HbA1c), and dosages.
* **Reciprocal Rank Fusion (RRF):**
  $$\text{RRF Score} = 0.60 \times \left(\frac{1}{60 + \text{Rank}_{\text{Dense}}}\right) + 0.40 \times \left(\frac{1}{60 + \text{Rank}_{\text{BM25}}}\right)$$

---

### 🎯 2. Deep Cross-Encoder Neural Reranking
* **Reranker Model:** `BAAI/bge-reranker-base` (Cross-Encoder).
* Processes pairs `[Query, Chunk]` simultaneously to compute interactive **Cross-Attention**.
* Calibrates raw logit scores via a $\text{Sigmoid}$ function to select the **Top-6 most relevant chunks**.

---

### 📐 3. How Similarity is Measured Across the Pipeline:
1. **Dense Cosine Similarity:** Inner Product $u \cdot v$ on normalized vectors in FAISS.
2. **Normalized BM25 Score:** Term-frequency lexical matching scaled to $[0, 1]$ via $\frac{\text{Score}}{\max(\text{Scores})}$.
3. **Jaccard Token Deduplication (92%):** Drops redundant chunks before prompt construction:
   $$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$
4. **Overall Retrieval Confidence Score:** Evidence quality gating in `EvidenceValidator`:
   $$\text{Confidence} = 0.70 \times \text{Top\_Score} + 0.30 \times \text{Avg\_Score} \ge 0.35$$

---

<!-- SLIDE 4: LLMs, Comparison Matrix & Primary Selection -->
# 📌 Slide 4: LLM Comparison Matrix & Primary Model Selection

### 📊 Model Evaluation & Comparison Matrix:

| Evaluation Dimension | `qwen/qwen3.6-27b` <br>**(Primary Selected Model)** | `openai/gpt-oss-120b` | `allam-2-7b` | `Phi-3.5-mini` <br>**(100% Offline Local)** |
| :--- | :---: | :---: | :---: | :---: |
| **Clinical Reasoning** | **Exceptional (27B)** | Superior (120B) | Good | Moderate (~4GB RAM) |
| **Strict Citation Grounding [1]** | **Strict Compliance** | Strict Compliance | Moderate | Requires Tuning |
| **Medical Arabic Synthesis** | **Fluent & Professional** | Fluent | Specialized | Basic / Moderate |
| **Inference Latency** | **Ultra-Fast (~1.5s)** | Moderate (~4.0s) | Fast (~1.0s) | Hardware Dependent |
| **License & Quota Headroom** | **Apache-2.0 (High Headroom)** | Constrained Free Quota | Fast API | **MIT (100% Local)** |

---

### 🏆 Why Was `qwen/qwen3.6-27b` Selected as Primary?
1. **Architectural Sweet Spot:** 27B parameters deliver deep clinical reasoning without the severe latency or quota constraints of 120B models.
2. **Superior Bilingual Grounding:** Accurately reads complex English guidelines and synthesizes natural, grammatically correct Arabic medical prose.
3. **Rigid Instruction Adherence:** Strictly obeys negative constraints (no unverified numbers, exact `[1]` citation placement).

---

### 🔁 Automated Failover Hierarchy:
$$\text{qwen3.6-27b} \xrightarrow{\text{Failover / 429}} \text{gpt-oss-20b} \xrightarrow{\text{Failover}} \text{gpt-oss-120b} \xrightarrow{\text{Offline}} \text{Phi-3.5-mini}$$

---

<!-- SLIDE 5: Safety Guardrails & Worst-Case / Out-of-Scope Handling -->
# 📌 Slide 5: Medical Safety Guardrails & Out-of-Scope Handling

### 🚨 How the System Handles Worst-Case & Out-of-Scope Queries:

```
                         ┌────────────────────────────────────────┐
                         │          Incoming User Query           │
                         └───────────────────┬────────────────────┘
                                             │
                  ┌──────────────────────────┴──────────────────────────┐
                  ▼                                                     ▼
     [Is Acute Life-Threatening?]                              [Is Conversational?]
     (Chest pain, DKA, Coma, Suicide)                          (Hello, Who are you, Thanks)
                  │                                                     │
         Yes ◄────┘                                            Yes ◄────┘
          │                                                     │
          ▼                                                     ▼
 🚨 Deterministic Code Intercept                        👋 Polite Identity Response
   (Emergency Safety Alert)                                (No RAG Resources Wasted)
          │                                                     │
          ▼ (If Standard Clinical Inquiry)                      │
 ┌──────────────────────────────────────────────────────────────┴──────────┐
 │ 3. Hybrid Retrieval + Evidence Quality Gating (EvidenceValidator)       │
 └───────────────────────────────┬─────────────────────────────────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
      [Score < 0.35 or Out of Domain]                 [Sufficient Evidence >= 0.35]
                 │                               │
                 ▼                               ▼
       🛑 Pure Safe Abstention                 Generation + Claim-Level Auditing
     (Zero Hallucination Refusal)              (CitationValidator & Auto-Regeneration)
```

---

### 📋 Standardized Safety Responses:
* **Out-of-Domain / Non-Medical Queries:**
  > *"The retrieved medical sources do not provide sufficient evidence to answer this question safely."*
* **Acute Medical Emergency Intercept:**
  > *🚨 "EMERGENCY SAFETY ALERT: If you or someone around you is experiencing severe symptoms, please contact local emergency services immediately."*
* **Hallucination Rejection:** Rejects unverified claims with status `UNSAFE_GENERATION_REJECTED`.

---

<!-- SLIDE 6: Evaluation Metrics, Testing Matrix & Conclusion -->
# 📌 Slide 6: Evaluation Metrics, Testing Matrix & Executive Conclusion

### 📈 1. Implemented Evaluation Metrics
* **Retrieval Evaluation (`RetrievalEvaluator` in `src/evaluation/retrieval_eval.py`):**
  * **$\text{Precision@K}$:** Measures the proportion of relevant chunks within the top-$K$ retrieved results ($K=5$).
  * **$\text{Recall@K}$:** Evaluates the coverage of required clinical ground-truth evidence.
  * **$\text{MRR}$ (Mean Reciprocal Rank):** Evaluates the rank position of the first relevant document.
* **Generation Groundedness (`GenerationEvaluator`):**
  * **$\text{ROUGE-L Precision}$:** Quantifies n-gram and longest common subsequence overlap between generated output and source context.

---

### 🧪 2. Automated Clinical Grounding Testing Matrix (100% Pass Rate)

| Test Suite Module | Target Evaluation Scenario | Status |
| :--- | :--- | :---: |
| **Numerical Fidelity** | Detects & rejects unverified HbA1c / glucose thresholds |  **PASSED** |
| **Clinical Modality** | Prevents escalating optional therapies into mandatory rules |  **PASSED** |
| **Treatment Sequencing** | Rejects unverified drug hierarchies (e.g., Drug A then Drug B) |  **PASSED** |
| **Pure Abstention** | Ensures 100% refusal on irrelevant queries (e.g., Appendicitis) |  **PASSED** |
| **Emergency Intercept** | Validates deterministic trigger for acute emergency symptoms |  **PASSED** |
| **Self-Correction Loop** | Automatically regenerates unsafe drafts into compliant answers |  **PASSED** |

---

### 🎯 Executive Conclusion:
**BioGuard Medical RAG** establishes a production-grade benchmark for clinical AI assistants: pairing ultra-low latency inference (~1.5s via Groq) with a 5-layer deterministic safety perimeter and a **100% Offline capability** to safeguard hospital infrastructure and patient data privacy.

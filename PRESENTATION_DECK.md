# 🩺 BioGuard Medical RAG — Clinical Intelligence Platform
## Comprehensive Presentation Deck (6 High-Impact Slides)

---

<!-- SLIDE 1: Title, Business Problem & Clinical Innovation -->
# 📌 Slide 1: المشكلة الطبية ورؤية المشروع (The Clinical & Business Problem)

### 🏥 السياق والمشكلة الحقيقية (The Problem):
* **هلوسة النماذج العامة (LLM Hallucinations):** في التطبيقات الطبية، توليد أرقام غير صحيحة، أو نسب تحاليل عشوائية، أو ترتيب دوائي غير موثق يمثل خطراً حقيقياً ومباشراً على سلامة المرضى (**Patient Safety Risk**).
* **تصعيد التوصيات (Clinical Modality Inflation):** تحويل النماذج التقليدية لـ "خيار علاجي اختياري" إلى "توصية إلزامية" دون مسوغ سريري.
* **فجوة اللغة والمصادر (The Language Gap):** معظم الأدلة والمراجع العالمية المعتمدة صادرة باللغة الإنجليزية (مثل NICE Guidelines)، بينما يحتاج الأطباء والمستخدمون إلى استشارات دقيقة وموثقة باللغة العربية.

---

### 💡 الحل المبتكر: نظام BioGuard Medical RAG
* **نظام استرجاع وتوليد سريري مغلق ومحصن (Closed-Domain Zero-Hallucination Medical Assistant).**
* **ثنائي اللغة (Bilingual):** يستقبل الاستفسار بالعربية أو الإنجليزية، يسترجع المراجع المعتمدة، ويولد إجابة طبية رصينة وموثقة بدقة.
* **تحقق تلقائي صارم (Claim-Level Verification):** تدقيق كل جملة ورقم واستشهاد `[1]` برمجياً قبل خروجها للمستخدم.

---

<!-- SLIDE 2: Data Sources, Chunking & Embeddings -->
# 📌 Slide 2: مصادر البيانات واستراتيجية التقطيع والتضمين (Data, Chunking & Embeddings)

### 📚 1. مصادر البيانات الطبية (Authoritative Data Sources)
* إرشادات سريرية معتمدة لإدارة مرض السكري من النوع الثاني (**NICE Clinical Guidelines & Diabetes Protocols**).
* مسار الملفات: `data/raw/` (مستندات PDF مفهرسة تشمل معايير الـ HbA1c، بدء الإنسولين، أدوية SGLT-2 / GLP-1 / Metformin، ونمط الحياة).

---

### ✂️ 2. طريقة التقطيع المبتكرة (Medical-Structure-Aware Chunking)
* **المسار:** `src/ingestion/chunker.py`
* **المحددات:** `chunk_size = 700 chars` | `chunk_overlap = 120 chars` | `min_chunk_size = 100 chars`.
* **آلية العمل:**
  1. التقسيم الهيكلي أولاً بالاعتماد على العناوين والفصول والأقسام الطبية المرمزة.
  2. في حالة زيادة حجم القسم عن 700 حرف، يتم تطبيق **نافذة منزلقة (Sliding Window)** تعتمد على **نهايات الجمل الحقيقية** (`. ! ؟ ?`) لضمان عدم بتر أي مصطلح أو جرعة دوائية.

---

### 🧠 3. نموذج التضمين وقاعدة المتجهات (Embeddings & Vector Index)
* **Embedding Model:** `BAAI/bge-small-en-v1.5` (متجهات 384-dim مع تطبيع $L_2$ Normalization كامل).
* **Vector Store:** `FAISS FlatIP` (محلي 100% لحساب Cosine Similarity فائق السرعة).

---

<!-- SLIDE 3: Hybrid Retrieval, Neural Reranking & Similarity Measurement -->
# 📌 Slide 3: الاسترجاع الهجين وكيف تم قياس التشابه (Hybrid Retrieval & Similarity)

### 🔄 1. معمارية الاسترجاع الهجين المزدوج (Hybrid Retrieval Fusion)
* **Dense Retrieval (60%):** متجهات FAISS الدلالية لالتقاط المعنى والسياق الطبي.
* **Sparse Lexical Search (40%):** خوارزمية `Rank-BM25Plus` لالتقاط الأسماء العلمية الدقيقة للأدوية والأرقام.
* **الدمج عبر RRF:** دمج الرتب باستخدام معادلة Reciprocal Rank Fusion:
  $$\text{RRF Score} = 0.60 \times \left(\frac{1}{60 + \text{Rank}_{\text{Dense}}}\right) + 0.40 \times \left(\frac{1}{60 + \text{Rank}_{\text{BM25}}}\right)$$

---

### 🎯 2. إعادة الترتيب العصبي العميق (Neural Cross-Encoder Reranking)
* **النموذج:** `BAAI/bge-reranker-base`
* يمرر زوج السؤال والمرجع `[Query, Chunk]` معاً لحساب الـ **Cross-Attention** التفاعلي.
* يتم تحويل القيمة عبر دالة $\text{Sigmoid}$ لاختيار أدق **Top-6 Chunks**.

---

### 📐 3. كيف تم قياس الـ Similarity عبر مستويات النظام؟
1. **Cosine Similarity في FAISS:** حاصل الضرب القياسي $u \cdot v$ للمتجهات المطبعة.
2. **Normalized BM25:** نسبة تكرار الكلمات $\frac{\text{Score}}{\max(\text{Scores})}$.
3. **Jaccard Token Similarity (92%):** إسقاط الفقرات المكررة قبل إرسالها للنموذج لمنع تشتيت الانتباه:
   $$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$
4. **Retrieval Confidence Score:** مقياس جودة الأدلة الكلي في `EvidenceValidator`:
   $$\text{Confidence} = 0.70 \times \text{Top\_Score} + 0.30 \times \text{Avg\_Score} \ge 0.35$$

---

<!-- SLIDE 4: LLMs, Comparison Matrix & Primary Selection -->
# 📌 Slide 4: مقارنة نماذج التوليد واختيار النموذج الأساسي (LLMs Comparison & Selection)

### 📊 جدول المقارنة بين النماذج (Generation Models Comparison Matrix):

| المعيار | `qwen/qwen3.6-27b` <br>**(النموذج الأساسي المختار)** | `openai/gpt-oss-120b` | `allam-2-7b` | `Phi-3.5-mini` <br>**(المحلي Offline)** |
| :--- | :---: | :---: | :---: | :---: |
| **قوة الاستدلال الطبي** | **ممتازة جداً (27B)** | استثنائية (120B) | جيدة | جيدة (~4GB RAM) |
| **الالتزام بالتوثيق [1]** | **صارم جداً** | صارم جداً | متوسط | يحتاج ضبط |
| **الطلاقة في العربية الطبية** | **فائقة وفصيحة** | ممتازة | متخصصة | متوسطة |
| **زمن الاستجابة (Latency)** | **سريع جداً (~1.5s)** | متوسط (~4.0s) | سريع (~1.0s) | يعتمد على الجهاز |
| **الترخيص والكوتا** | **Apache-2.0 (مستقر)** | كوتا محدودة | سريع | **MIT (محلي 100%)** |

---

### 🏆 لماذا تم اختيار `qwen/qwen3.6-27b` كـ Primary Model؟
1. **التوازن المثالي (Sweet Spot):** يجمع بين الفهم السريري العميق لنماذج الـ 27B والسرعة الفائقة.
2. **براعة الترجمة الطبية:** استيعاب المراجع الإنجليزية المعقدة وصياغة رد عربي طبي دقيق ومحكم.
3. **الامتثال التام للتعليمات:** عدم اختلاق تسلسلات دوائية وربط كل نقطة برقم المرجع `[1]`.

---

### 🔁 استراتيجية التبديل التلقائي (Automated Failover Hierarchy):
$$\text{qwen3.6-27b} \xrightarrow{\text{Failover / 429}} \text{gpt-oss-20b} \xrightarrow{\text{Failover}} \text{gpt-oss-120b} \xrightarrow{\text{Offline}} \text{Phi-3.5-mini}$$

---

<!-- SLIDE 5: Safety Guardrails & Worst-Case / Out-of-Scope Handling -->
# 📌 Slide 5: درع الأمان الطبي والتعامل مع أسوأ السيناريوهات (Safety Guardrails & Worst-Case Scenarios)

### 🚨 كيف يتعامل النظام مع الأسئلة الخارجة عن النطاق (Worst-Case & Out of Scope)؟

```
                      ┌─────────────────────────────────────────┐
                      │            سؤال المستخدم الوارد          │
                      └────────────────────┬────────────────────┘
                                           │
                ┌──────────────────────────┴──────────────────────────┐
                ▼                                                     ▼
     [حالة طوارئ حادة؟]                                    [سؤال ترحيب/محادثة؟]
   (أزمة قلبية، غيبوبة، DKA)                               (أهلاً، مين انت، عامل ايه)
                │                                                     │
       نعم ◄────┘                                            نعم ◄────┘
        │                                                     │
        ▼                                                     ▼
 🚨 اعتراض برمجي فوري                                   👋 رد ترحيبي ذكي
(Emergency Safety Alert)                              (Conversational Response)
        │                                                     │
        ▼ (إذا كان سؤالاً عادياً)                             │
 ┌────────────────────────────────────────────────────────────┴────────┐
 │ 3. البحث في المراجع + فحص كفاية الأدلة (Evidence Quality Validator) │
 └─────────────────────────────┬───────────────────────────────────────┘
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
    [الدرجة < 0.35 أو خارج النطاق]             [أدلة كافية وموثوقة >= 0.35]
               │                               │
               ▼                               ▼
     🛑 الامتناع الآمن المباشر               توليد الرد + التدقيق الآلي
     (Pure Safe Abstention)                  (Claim & Citation Verification)
```

---

### 📋 الردود القياسية للسيناريوهات الخاصة:
* **سؤال خارج النطاق (Non-medical / Out of domain):**
  > *"المصادر الطبية المفهرسة لا تتضمن أدلة كافية وموثوقة للإجابة على هذا السؤال بأمان."*
* **حالات الطوارئ الخطرة (Emergency Intercept):**
  > *🚨 "EMERGENCY SAFETY ALERT: If experiencing severe symptoms, please call local emergency services immediately."*
* **محاولات الهلوسة:** رفض الإجابة فوراً بحالة `UNSAFE_GENERATION_REJECTED`.

---

<!-- SLIDE 6: Evaluation Metrics, Testing Matrix & Conclusion -->
# 📌 Slide 6: التقييم والاختبارات والخلاصة النهائية (Evaluation, Testing Matrix & Conclusion)

### 📈 1. مقاييس التقييم المعمارية (Evaluation Metrics Implemented)
* **استرجاع المراجع (`RetrievalEvaluator`):**
  * **$\text{Precision@K}$:** قياس دقة أول $K$ مستندات مسترجعة ونسبة النصوص ذات الصلة الحقيقية.
  * **$\text{Recall@K}$:** قياس نسبة تغطية الأدلة السريرية المطلوبة.
  * **$\text{MRR}$ (Mean Reciprocal Rank):** سرعة الوصول لأول مصدر صحيح.
* **دقة التوليد (`GenerationEvaluator`):**
  * **$\text{ROUGE-L Precision}$:** قياس التطابق النصي والمفهومي مع المراجع الأصلية لمنع الإضافة.

---

### 🧪 2. جدول نتائج الاختبارات النهائية (Testing Matrix - 100% Pass Rate)

| معيار الاختبار (Test Suite) | السيناريو المُختبر | النتيجة والحالة |
| :--- | :--- | :---: |
| **Numerical Fidelity** | منع اختلاق نسب الـ HbA1c غير الموجودة |  **PASSED** |
| **Clinical Modality** | منع تحويل "الخيار" إلى "توصية إلزامية" |  **PASSED** |
| **Sequence Safety** | كشف ومنع ترتيب الأدوية العشوائي |  **PASSED** |
| **Pure Abstention** | فحص الأسئلة خارج النطاق (مثل الزائدة الدودية) |  **PASSED** |
| **Emergency Intercept** | كشف أعراض الأزمات والغيبوبة فوراً |  **PASSED** |
| **Regeneration Loop** | تصحيح المسودة الأولى آلياً عند اكتشاف خطأ |  **PASSED** |

---

### 🎯 الخلاصة التنفيذية (Executive Conclusion):
نجح **BioGuard Medical RAG** في تقديم نموذج عملي لذكاء اصطناعي سريري من الفئة الإنتاجية (**Production-Grade**)؛ يدمج بين السرعة الفائقة عبر Groq، الحصانة التامة ضد الهلوسة عبر 5 طبقات أمان، مع إمكانية العمل **100% Offline** لحماية سرية بيانات المرضى والمستشفيات.

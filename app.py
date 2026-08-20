"""
BioGuard Medical RAG — Production-Grade Clinical Intelligence Platform
=======================================================================
Entry point. Thin orchestrator that wires UI components to the RAG backend.

Backend pipeline is never touched by the UI layer.
UI components live in src/ui/.
"""

import os
import time
import streamlit as st

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="BioGuard — Clinical Intelligence Platform",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Backend imports ──────────────────────────────────────────────────────────
try:
    from src.utils.config_loader import load_yaml_config
    from src.utils.device import get_system_hardware_info
    from src.embeddings.embedding_model import LocalEmbeddingModel
    from src.retrieval.vector_store import LocalFAISSVectorStore
    from src.retrieval.bm25_retriever import LocalBM25Retriever
    from src.retrieval.reranker import LocalReranker
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.generation.llm import UnifiedLLM
    from src.pipeline.rag_pipeline import MedicalRAGPipeline
    from src.ingestion.document_loader import UniversalDocumentLoader
    from src.ingestion.chunker import MedicalAwareChunker
except ImportError as e:
    st.error(f"⚠️ Initialization Error — missing core module: {e}")
    st.stop()

# ── UI component imports ─────────────────────────────────────────────────────
from src.ui.theme import inject_css
from src.ui.header import render_header
from src.ui.sidebar import render_sidebar
from src.ui.composer import render_hero_screen
from src.ui.chat import render_chat
from src.ui.suggested_protocols import render_suggested_protocols
from src.ui.ingestion import render_ingestion_tab
from src.ui.system_status import render_system_tab


# ── Session state defaults ───────────────────────────────────────────────────
def _init_session() -> None:
    defaults = {
        "messages": [],
        "dark_mode": False,
        "debug_mode": False,
        "deeper_research": False,
        "patient_context": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_session()


# ── Theme injection ──────────────────────────────────────────────────────────
inject_css(dark_mode=st.session_state["dark_mode"])


# ── Backend model loader (cached) ────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading BioGuard Clinical Engine…")
def load_backend_models():
    settings = load_yaml_config("config/settings.yaml")
    models_cfg = load_yaml_config("config/models.yaml")

    emb_model = LocalEmbeddingModel(models_cfg["embedding"]["model_id"])
    vector_store = LocalFAISSVectorStore()
    bm25 = LocalBM25Retriever()
    reranker = (
        LocalReranker(models_cfg["reranker"]["model_id"])
        if models_cfg.get("reranker", {}).get("enabled", True)
        else None
    )

    idx_dir = settings["paths"]["index_dir"]
    if os.path.exists(os.path.join(idx_dir, "faiss.index")):
        try:
            vector_store.load(idx_dir)
            bm25_path = os.path.join(idx_dir, "bm25.pkl")
            if os.path.exists(bm25_path):
                bm25.load(idx_dir)
            else:
                bm25.index_chunks(vector_store.chunks_metadata)
                bm25.save(idx_dir)
        except Exception as err:
            st.warning(f"Index reload issue — starting fresh: {err}")

    llm = UnifiedLLM(
        provider=models_cfg["llm"]["provider"],
        model_id=models_cfg["llm"]["model_id"],
        temperature=models_cfg["llm"].get("temperature", 0.1),
        max_new_tokens=models_cfg["llm"].get("max_new_tokens", 2500),
    )

    hybrid = HybridRetriever(
        vector_store,
        emb_model,
        bm25,
        reranker,
        dense_weight=settings.get("retrieval", {}).get("dense_weight", 0.6),
        bm25_weight=settings.get("retrieval", {}).get("bm25_weight", 0.4),
    )

    pipeline = MedicalRAGPipeline(hybrid, llm, settings)
    return vector_store, emb_model, bm25, reranker, llm, settings, models_cfg, pipeline


try:
    vector_store, emb_model, bm25, reranker, llm, settings, models_cfg, pipeline = load_backend_models()
except Exception as e:
    st.error(f"Failed to load backend pipeline — {e}")
    st.stop()


# ── Hardware info ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _get_hw():
    try:
        return get_system_hardware_info()
    except Exception:
        return {"device": "CPU", "status": "Available"}


hw_info = _get_hw()


# ── Header ───────────────────────────────────────────────────────────────────
dark_mode_toggled = render_header(
    dark_mode=st.session_state["dark_mode"],
    models_cfg=models_cfg,
)

# Persist dark mode toggle and rerun to re-inject CSS
if dark_mode_toggled != st.session_state["dark_mode"]:
    st.session_state["dark_mode"] = dark_mode_toggled
    st.rerun()


# ── Sidebar ──────────────────────────────────────────────────────────────────
render_sidebar()
debug_mode = st.session_state.get("debug_mode", False)


# ── Main tabs ────────────────────────────────────────────────────────────────
tab_chat, tab_ingest, tab_system = st.tabs([
    "💬  Clinical Assistant",
    "📚  Knowledge Base",
    "⚙️  System & Models",
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: CLINICAL ASSISTANT
# ═════════════════════════════════════════════════════════════════════════════
with tab_chat:
    submitted_query: str | None = None

    messages = st.session_state.get("messages", [])
    is_fresh = len(messages) == 0

    if is_fresh:
        # ── Hero / empty state ──
        submitted_query = render_hero_screen()


    else:
        # ── Active conversation ──
        # New inquiry button (top right of chat area)
        nc_col = st.columns([8, 2])[1]
        with nc_col:
            if st.button("＋ New Consultation", key="top_new_chat"):
                st.session_state.messages = []
                st.rerun()

        bottom_input = render_chat(debug_mode=debug_mode)
        if bottom_input:
            submitted_query = bottom_input

    # ── Handle new query submission ──────────────────────────────────────────
    if submitted_query:
        # Optionally append patient context
        ctx = st.session_state.get("patient_context", "").strip()
        final_query = submitted_query
        if ctx:
            final_query = f"{submitted_query}\n\n[Patient Context: {ctx}]"

        st.session_state.messages.append({"role": "user", "content": submitted_query})
        st.rerun()

    # ── Trigger LLM on latest user message ──────────────────────────────────
    if (
        st.session_state.messages
        and st.session_state.messages[-1]["role"] == "user"
    ):
        latest_query = st.session_state.messages[-1]["content"]
        ctx = st.session_state.get("patient_context", "").strip()
        if ctx:
            retrieval_query = f"{latest_query}\n\n[Patient Context: {ctx}]"
        else:
            retrieval_query = latest_query

        start_t = time.time()
        with st.spinner("Retrieving and synthesizing grounded clinical evidence…"):
            try:
                response = pipeline.answer_query(retrieval_query, return_debug=debug_mode)
            except Exception as e:
                # Clean user-facing error — no raw traceback
                response = {
                    "answer": (
                        "**Unable to complete the consultation.**\n\n"
                        "Possible reasons:\n"
                        "- Model or API temporarily unavailable\n"
                        "- Retrieval index not loaded\n"
                        "- Unexpected backend error\n\n"
                        f"_Technical detail (debug mode): {str(e) if debug_mode else 'Enable Observability Debug Mode in sidebar for details.'}_"
                    ),
                    "sources": [],
                    "retrieval_score": 0.0,
                    "retrieval_relevance": 0.0,
                    "sources_count": 0,
                    "citations_verified_count": 0,
                    "evidence_quality": "ERROR",
                    "latency_sec": round(time.time() - start_t, 2),
                    "status": "ERROR",
                    "debug": {"error": str(e)} if debug_mode else None,
                }

        sources = response.get("sources", [])
        cited_count = response.get(
            "citations_verified_count",
            len([s for s in sources if s.get("is_referenced_in_text")])
        )

        st.session_state.messages.append({
            "role": "assistant",
            "content": response.get("answer", "No answer generated."),
            "sources": sources,
            "retrieval_score": response.get("retrieval_score", 0.0),
            "retrieval_relevance": response.get(
                "retrieval_relevance",
                round(response.get("retrieval_score", 0.0), 2)
            ),
            "sources_count": response.get("sources_count", len(sources)),
            "citations_verified_count": cited_count,
            "evidence_quality": response.get("evidence_quality", "HIGH"),
            "latency_sec": response.get("latency_sec", round(time.time() - start_t, 2)),
            "status": response.get("status", "SUCCESS"),
            "debug": response.get("debug"),
        })
        st.rerun()




# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: KNOWLEDGE BASE INGESTION
# ═════════════════════════════════════════════════════════════════════════════
with tab_ingest:
    render_ingestion_tab(settings, emb_model, vector_store, bm25)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: SYSTEM & MODELS
# ═════════════════════════════════════════════════════════════════════════════
with tab_system:
    render_system_tab(models_cfg, hw_info, vector_store, bm25, reranker, llm)

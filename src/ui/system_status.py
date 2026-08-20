"""
BioGuard System Status Tab — hardware, model configuration, and pipeline status.
"""
import streamlit as st


def render_system_tab(models_cfg: dict, hw: dict, vector_store, bm25, reranker, llm) -> None:
    """Render the System & Model Observatory tab."""
    st.markdown("""
    <div style="max-width:860px; margin:0 auto;">
        <div style="font-size:22px; font-weight:800; color:var(--text-primary); margin-bottom:4px;">
            ⚙️ System & Model Observatory
        </div>
        <div style="font-size:13px; color:var(--text-secondary); margin-bottom:24px;">
            Live diagnostics of the BioGuard clinical pipeline components.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Pipeline Status ──
    st.markdown("#### Pipeline Component Status")

    def status_row(name: str, obj, detail: str = ""):
        loaded = obj is not None
        icon = "🟢" if loaded else "🔴"
        state = "Loaded" if loaded else "Not Loaded"
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:12px; padding:10px 14px;
                    background:var(--bg-subtle); border:1px solid var(--border);
                    border-radius:var(--radius-md); margin-bottom:6px; font-size:13px;">
            <span style="font-size:16px;">{icon}</span>
            <span style="font-weight:600; color:var(--text-primary); flex:1;">{name}</span>
            <span style="color:{'var(--success)' if loaded else 'var(--danger)'}; font-weight:700;">{state}</span>
            <span style="color:var(--text-muted); font-size:11px;">{detail}</span>
        </div>
        """, unsafe_allow_html=True)

    vs_chunks = len(getattr(vector_store, "chunks_metadata", []))
    status_row("FAISS Vector Store", vector_store, f"{vs_chunks} chunks")
    status_row("BM25 Keyword Index", bm25)
    status_row("BGE Cross-Encoder Reranker", reranker, models_cfg.get("reranker", {}).get("model_id", ""))
    status_row("LLM (Groq API)", llm, models_cfg.get("llm", {}).get("model_id", ""))

    st.divider()

    # ── Compute ──
    st.markdown("#### Compute Resources")
    if hw:
        hw_cols = st.columns(min(len(hw), 4))
        for col, (key, val) in zip(hw_cols, hw.items()):
            col.metric(key.replace("_", " ").title(), str(val))
    else:
        st.info("Hardware info not available.")

    st.divider()

    # ── Model Configuration ──
    st.markdown("#### Active Model Configuration")
    m1, m2, m3 = st.columns(3)

    with m1:
        st.markdown("""
        <div style="background:var(--bg-subtle); border:1px solid var(--border);
                    border-radius:var(--radius-md); padding:14px 16px;">
            <div style="font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; color:var(--text-muted); margin-bottom:6px;">Embedding Model</div>
        """, unsafe_allow_html=True)
        st.code(models_cfg.get("embedding", {}).get("model_id", "—"), language=None)
        st.markdown("</div>", unsafe_allow_html=True)

    with m2:
        st.markdown("""
        <div style="background:var(--bg-subtle); border:1px solid var(--border);
                    border-radius:var(--radius-md); padding:14px 16px;">
            <div style="font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; color:var(--text-muted); margin-bottom:6px;">Reranker</div>
        """, unsafe_allow_html=True)
        st.code(models_cfg.get("reranker", {}).get("model_id", "—"), language=None)
        st.markdown("</div>", unsafe_allow_html=True)

    with m3:
        llm_cfg = models_cfg.get("llm", {})
        st.markdown("""
        <div style="background:var(--bg-subtle); border:1px solid var(--border);
                    border-radius:var(--radius-md); padding:14px 16px;">
            <div style="font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; color:var(--text-muted); margin-bottom:6px;">Language Model</div>
        """, unsafe_allow_html=True)
        st.code(llm_cfg.get("model_id", "—"), language=None)
        st.markdown(f"""
            <div style="font-size:11px; color:var(--text-muted); margin-top:6px;">
                Provider: {llm_cfg.get('provider','—')} · Temp: {llm_cfg.get('temperature','—')} · Max tokens: {llm_cfg.get('max_new_tokens','—')}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown("#### Full Configuration (JSON)")
    with st.expander("View models.yaml configuration"):
        st.json(models_cfg)

"""
BioGuard Observability Panel — debug diagnostics when debug mode is enabled.
Only shown when debug_mode is True.
"""
import streamlit as st


def render_observability(msg: dict) -> None:
    """Render full pipeline observability diagnostics."""
    debug_data = msg.get("debug")
    if not debug_data:
        return

    with st.expander("🔍 Pipeline Observability Diagnostics", expanded=False):
        # Pipeline steps visualization
        steps = [
            ("Query Preparation",      debug_data.get("raw_query") is not None),
            ("Query Translation",      debug_data.get("retrieval_query") is not None),
            ("Hybrid Retrieval",       debug_data.get("retrieval_diagnostics") is not None),
            ("Evidence Validation",    debug_data.get("evidence_validation") is not None),
            ("Citation Validation",    debug_data.get("citation_validation") is not None),
        ]

        st.markdown("<div style='margin-bottom:10px;'>", unsafe_allow_html=True)
        for step_name, done in steps:
            icon = "✅" if done else "⏳"
            css_class = "obs-step obs-done" if done else "obs-step"
            st.markdown(f"""
            <div class="{css_class}">
                <span class="obs-step-icon">{icon}</span>
                <span class="obs-step-label">{step_name}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Evidence validation metrics
        ev = debug_data.get("evidence_validation", {})
        if ev:
            st.markdown("**Evidence Validation**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Sufficient", "Yes" if ev.get("is_sufficient") else "No")
            c2.metric("Score", f"{ev.get('score', 0):.3f}")
            c3.metric("Quality", ev.get("quality_label", "—"))
            if ev.get("reason"):
                st.caption(f"Reason: {ev['reason']}")

        # Citation validation
        cv = debug_data.get("citation_validation", {})
        if cv:
            st.markdown("**Citation Validation**")
            c1, c2 = st.columns(2)
            c1.metric("Verified Citations", cv.get("verified_citations_count", 0))
            inv = cv.get("invalid_citations_found", [])
            c2.metric("Invalid Citations", len(inv))
            if inv:
                st.warning(f"Invalid citation IDs found: {inv}")
            if cv.get("invalid_claims"):
                st.markdown("**Invalid Claims**")
                st.json(cv.get("invalid_claims"))
            if cv.get("unsupported_claims"):
                st.markdown("**Unsupported Claims**")
                st.json(cv.get("unsupported_claims"))
            if cv.get("numerical_claims"):
                st.markdown("**Numerical Claims**")
                st.json(cv.get("numerical_claims"))
            if cv.get("treatment_sequence_flags"):
                st.markdown("**Treatment Sequence Flags**")
                st.json(cv.get("treatment_sequence_flags"))

        generated = debug_data.get("generated_answer")
        if generated:
            st.markdown("**Generated Answer Drafts**")
            st.json(generated)

        # Retrieval diagnostics (raw JSON)
        ret = debug_data.get("retrieval_diagnostics")
        if ret:
            st.markdown("**Retrieval Diagnostics**")
            st.json(ret)

        # Queries
        col1, col2 = st.columns(2)
        with col1:
            raw = debug_data.get("raw_query", "")
            if raw:
                st.markdown("**Original Query**")
                st.code(raw, language=None)
        with col2:
            ret_q = debug_data.get("retrieval_query", "")
            if ret_q and ret_q != raw:
                st.markdown("**Translated Search Query**")
                st.code(ret_q, language=None)

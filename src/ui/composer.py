"""
BioGuard Clinical Composer — minimal, clean query input for the hero screen.
"""
import streamlit as st


def render_hero_screen() -> str | None:
    """
    Render the empty-state hero screen with clinical query composer.
    Returns submitted query string or None.
    """
    # ── Hero branding ──
    st.markdown("""
    <div style="display:flex; flex-direction:column; align-items:center; padding:40px 0 28px 0; text-align:center;">
        <div style="font-size:52px; margin-bottom:14px; line-height:1;">🩺</div>
        <div style="font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:0.14em;
                    color:var(--text-muted); margin-bottom:10px;">
            BioGuard Clinical Intelligence
        </div>
        <div style="font-size:30px; font-weight:800; color:var(--text-primary);
                    letter-spacing:-0.03em; margin-bottom:10px; line-height:1.25;">
            Evidence-Grounded Medical Reference
        </div>
        <div style="font-size:14px; color:var(--text-secondary); max-width:500px; line-height:1.7;">
            Ask questions against your indexed clinical knowledge base.<br>
            Every response is grounded in retrieved evidence.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Clinical Query Composer ──
    col_left, col_center, col_right = st.columns([1, 5, 1])
    with col_center:
        with st.form("hero_composer", clear_on_submit=True):
            query_val = st.text_input(
                "Clinical question",
                placeholder="e.g. What are the glycaemic targets for Type 2 Diabetes in elderly patients?",
                label_visibility="collapsed",
            )

            st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)

            ctrl_col, action_col = st.columns([3, 1])
            with ctrl_col:
                t1, t2 = st.columns(2)
                with t1:
                    st.checkbox(
                        "Guidelines Grounded",
                        value=True,
                        disabled=True,
                        help="Restricts synthesis strictly to indexed clinical evidence. Always enabled.",
                        key="toggle_grounded"
                    )
                with t2:
                    st.checkbox(
                        "Safety Audited",
                        value=True,
                        disabled=True,
                        help="Applies medical safety guardrails and evidence validation before each response. Always enabled.",
                        key="toggle_safety"
                    )
            with action_col:
                submit = st.form_submit_button(
                    "Ask BioGuard →",
                    use_container_width=True,
                    type="primary"
                )

            if submit and query_val and query_val.strip():
                return query_val.strip()

    return None

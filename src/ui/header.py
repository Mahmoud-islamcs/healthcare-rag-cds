"""
BioGuard Header Component — professional enterprise application header.
"""
import streamlit as st


def render_header(dark_mode: bool, models_cfg: dict) -> bool:
    """
    Render the top application header.
    Returns True if user toggled dark mode.
    """
    model_id = models_cfg.get("llm", {}).get("model_id", "Medical LLM")
    model_short = model_id.split("/")[-1] if "/" in model_id else model_id

    col_logo, col_center, col_right = st.columns([3, 4, 3])

    with col_logo:
        st.markdown("""
        <div class="bioguard-logo" style="padding-top:10px;">
            <span style="font-size:22px;">🩺</span>
            <div>
                <span class="bioguard-logo-name">BioGuard</span>
                <span class="bioguard-tagline">Clinical Intelligence Platform</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_center:
        st.markdown("""
        <div style="display:flex; align-items:center; justify-content:center; height:52px;">
            <span style="font-size:12px; color:var(--text-muted); font-weight:500; letter-spacing:0.04em;">
                EVIDENCE-GROUNDED MEDICAL REFERENCE ASSISTANT
            </span>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        r1, r2, r3 = st.columns([3, 2, 2])
        with r1:
            st.markdown(f"""
            <div style="padding-top:10px; text-align:right;">
                <span class="header-badge operational">
                    <span class="status-dot"></span> Operational
                </span>
                <br>
                <span style="font-size:10px; color:var(--text-muted); display:block; text-align:right; margin-top:3px;">
                    {model_short}
                </span>
            </div>
            """, unsafe_allow_html=True)
        with r2:
            st.markdown("<div style='padding-top:8px;'></div>", unsafe_allow_html=True)
            toggled = st.toggle("🌙" if not dark_mode else "☀️", value=dark_mode, key="dark_mode_toggle", label_visibility="collapsed")
        with r3:
            st.markdown("""
            <div style="padding-top:12px; text-align:right;">
                <span style="font-size:22px; cursor:pointer;" title="Clinician Workspace">👤</span>
            </div>
            """, unsafe_allow_html=True)

    # Header divider
    st.markdown(
        "<hr style='margin:0 0 0 0; border:none; border-top:1px solid var(--border);'>",
        unsafe_allow_html=True
    )
    return toggled

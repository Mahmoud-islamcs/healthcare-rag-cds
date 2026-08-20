"""
BioGuard Chat Renderer — displays conversation history with RTL support,
evidence banners, citation cards, and observability panels.
"""
import re
import streamlit as st
from src.ui.evidence import render_evidence_banner
from src.ui.citations import render_citations
from src.ui.observability import render_observability


def _is_arabic(text: str) -> bool:
    """Detect if text is predominantly Arabic."""
    arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
    total_chars = len(re.findall(r'[a-zA-Z\u0600-\u06FF]', text))
    if total_chars == 0:
        return False
    return (arabic_chars / total_chars) > 0.4


def _render_user_message(msg: dict) -> None:
    with st.chat_message("user"):
        content = msg["content"]
        direction = "rtl" if _is_arabic(content) else "ltr"
        align = "right" if direction == "rtl" else "left"
        font = "var(--font-arabic)" if direction == "rtl" else "var(--font-sans)"
        st.markdown(
            f'<div style="direction:{direction}; text-align:{align}; font-family:{font}; font-weight:600; font-size:14px; color:var(--text-primary);">{content}</div>',
            unsafe_allow_html=True
        )


def _render_assistant_message(msg: dict, debug_mode: bool) -> None:
    status = msg.get("status", "SUCCESS")
    content = msg.get("content", "")
    is_ar = _is_arabic(content)
    direction = "rtl" if is_ar else "ltr"
    align = "right" if is_ar else "left"
    font = "var(--font-arabic)" if is_ar else "var(--font-sans)"

    with st.chat_message("assistant", avatar="🩺"):
        # Emergency alert styling
        if status == "EMERGENCY_TRIGGERED":
            st.markdown(
                f'<div class="emergency-alert">'
                f'<div style="font-size:13px; font-weight:700; color:var(--danger); margin-bottom:6px;">🚨 Emergency Safety Alert</div>'
                f'<div style="font-size:13px; color:var(--text-primary); direction:{direction}; text-align:{align}; font-family:{font}; line-height:1.7;">{content}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
            return

        # Clean formatting: ensure bullet points are on newlines and properly spaced
        formatted_content = content
        if "•" in formatted_content and "\n•" not in formatted_content:
            formatted_content = formatted_content.replace(" • ", "\n\n- ")
            formatted_content = formatted_content.replace("• ", "\n\n- ")

        if is_ar:
            st.markdown(
                f'<div style="direction:rtl; text-align:right; font-family:var(--font-arabic); font-size:15px; line-height:1.9;">\n\n'
                f'{formatted_content}\n\n'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(formatted_content)

        # Evidence banner (skip for conversational)
        render_evidence_banner(msg, debug_mode)

        # Citation drawer
        render_citations(msg)

        # Observability panel (only in debug mode)
        if debug_mode:
            render_observability(msg)


def render_chat(debug_mode: bool = False) -> str | None:
    """
    Render the full conversation history.
    Returns any new submitted query from the bottom input, or None.
    """
    messages = st.session_state.get("messages", [])

    for msg in messages:
        if msg["role"] == "user":
            _render_user_message(msg)
        else:
            _render_assistant_message(msg, debug_mode)

    # Bottom follow-up input
    submitted = st.chat_input(
        "Ask a follow-up clinical question…",
        key="chat_bottom_input"
    )
    return submitted

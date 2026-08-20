"""
BioGuard Sidebar — workspace history, navigation, and session management.
"""
import streamlit as st


def _truncate(text: str, max_len: int = 38) -> str:
    return text if len(text) <= max_len else text[:max_len] + "…"


def render_sidebar() -> None:
    """Render the left sidebar with history, navigation, and actions."""
    with st.sidebar:
        # ── Workspace brand ──
        st.markdown("""
        <div style="padding: 16px 0 10px 0; border-bottom: 1px solid var(--border); margin-bottom: 4px;">
            <div style="font-size:15px; font-weight:800; color:var(--text-primary);">🩺 BioGuard</div>
            <div style="font-size:10px; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.08em; margin-top:2px;">Clinical Workspace</div>
        </div>
        """, unsafe_allow_html=True)

        # ── New consultation button ──
        if st.button("＋  New Consultation", key="sidebar_new_chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        # ── Recent consultations ──
        st.markdown('<div class="sidebar-section-title">Recent Consultations</div>', unsafe_allow_html=True)

        messages = st.session_state.get("messages", [])
        user_msgs = [m for m in messages if m["role"] == "user"]

        if not user_msgs:
            st.markdown("""
            <div style="font-size:12px; color:var(--text-muted); padding:6px 2px;">
                No consultations yet in this session.
            </div>
            """, unsafe_allow_html=True)
        else:
            # Show last 6 user queries as history items
            for i, msg in enumerate(reversed(user_msgs[-6:])):
                truncated = _truncate(msg["content"])
                st.markdown(f"""
                <div class="history-item">
                    <span style="font-size:13px; color:var(--text-muted);">💬</span>
                    <span style="overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-size:12px;">{truncated}</span>
                </div>
                """, unsafe_allow_html=True)

        # ── Bookmarked Protocols (static — session-based) ──
        st.markdown('<div class="sidebar-section-title">Bookmarked Protocols</div>', unsafe_allow_html=True)

        protocols = [
            ("🩸", "Type 2 Diabetes Management"),
            ("💉", "Insulin Initiation Guidelines"),
            ("🫀", "Cardiovascular Risk Reduction"),
        ]
        for icon, name in protocols:
            st.markdown(f"""
            <div class="history-item">
                <span>{icon}</span>
                <span style="font-size:12px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">{name}</span>
            </div>
            """, unsafe_allow_html=True)

        # ── System ──
        st.markdown('<div class="sidebar-section-title">System</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:12px; color:var(--text-secondary); display:flex; flex-direction:column; gap:6px; padding:2px 0;">
            <div>🗄️ &nbsp;Document Knowledge Base</div>
            <div>⚙️ &nbsp;Hardware & Models</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Debug toggle ──
        st.markdown('<div class="sidebar-section-title">Developer</div>', unsafe_allow_html=True)
        debug_on = st.checkbox(
            "Observability Debug Mode",
            value=st.session_state.get("debug_mode", False),
            key="debug_checkbox",
            help="Enable to see full retrieval diagnostics, query translation, and citation validation details."
        )
        st.session_state["debug_mode"] = debug_on

        # ── Clear history ──
        st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
        if st.button("🗑️  Clear History", key="sidebar_clear", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        # ── Version ──
        st.markdown("""
        <div style="position:absolute; bottom:16px; left:0; right:0; text-align:center; font-size:10px; color:var(--text-muted);">
            BioGuard Clinical RAG v2.0
        </div>
        """, unsafe_allow_html=True)

"""
BioGuard Citation Cards — verified source display with status badges.
"""
import streamlit as st


def render_citations(msg: dict) -> None:
    """Render verified citation evidence drawer."""
    sources = msg.get("sources", [])
    if not sources:
        return

    cited = [s for s in sources if s.get("is_referenced_in_text")]
    background = [s for s in sources if not s.get("is_referenced_in_text")]
    label = f"Cited Evidence — {len(cited)} cited · {len(background)} retrieved background"

    with st.expander(f"📑 {label}", expanded=False):
        for src in sources:
            is_cited = src.get("is_referenced_in_text", False)
            rel_score = src.get("relevance_score", "N/A")
            snippet = src.get("snippet", "")
            cid = src.get("citation_id", "?")
            fname = src.get("source_file", "Reference.pdf")
            page = src.get("page", 1)

            if is_cited:
                badge = '<span class="badge badge-cited">✓ Cited Evidence</span>'
            else:
                badge = '<span class="badge badge-bg">ℹ Background Evidence</span>'

            card_html = (
                f'<div class="citation-card">'
                f'<div class="citation-header">'
                f'<div><span class="citation-id">[{cid}]</span><span class="citation-file"> {fname}</span></div>'
                f'{badge}'
                f'</div>'
                f'<div class="citation-meta">Page {page} &nbsp;·&nbsp; Relevance: {rel_score}</div>'
                f'<div class="citation-snippet">"{snippet}"</div>'
                f'</div>'
            )
            st.markdown(card_html, unsafe_allow_html=True)

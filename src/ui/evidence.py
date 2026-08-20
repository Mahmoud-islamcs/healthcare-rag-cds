"""
BioGuard Evidence Banner — retrieval evidence metrics display.
Shows evidence quality, retrieval score, source count, citation ratio, and latency.
"""
import streamlit as st


def _quality_class(quality: str) -> str:
    q = quality.upper()
    if q == "HIGH":
        return "quality-high"
    elif q == "MODERATE":
        return "quality-moderate"
    return "quality-low"


def render_evidence_banner(msg: dict, debug_mode: bool = False) -> None:
    """Render retrieval evidence metrics beneath an assistant reply."""
    status = msg.get("status", "SUCCESS")
    if status == "CONVERSATIONAL":
        return

    quality = str(msg.get("evidence_quality", "HIGH")).upper()
    ret_rel = msg.get("retrieval_relevance", round(msg.get("retrieval_score", 0.0), 2))
    latency = msg.get("latency_sec", 0.0)
    sources_cnt = msg.get("sources_count", len(msg.get("sources", [])))
    cited_cnt = msg.get("citations_verified_count",
                        len([s for s in msg.get("sources", []) if s.get("is_referenced_in_text")]))

    bar_pct = int(ret_rel * 100)
    if quality == "HIGH":
        bar_color = "var(--success)"
    elif quality == "MODERATE":
        bar_color = "var(--warning)"
    else:
        bar_color = "var(--danger)"

    q_class = _quality_class(quality)

    html = (
        f'<div class="evidence-banner">'
        f'<div style="display:flex; gap:16px; align-items:center; flex-wrap:wrap;">'
        f'<div class="metric-item">'
        f'<span class="metric-label">Evidence Retrieval Quality</span>'
        f'<span class="quality-badge {q_class}">{quality}</span>'
        f'</div>'
        f'<div class="metric-divider"></div>'
        f'<div class="metric-item">'
        f'<span class="metric-label">Evidence Retrieval Confidence</span>'
        f'<div style="display:flex; align-items:center; gap:8px;">'
        f'<div class="evidence-bar-wrap">'
        f'<div class="evidence-bar-fill" style="width:{bar_pct}%; background:{bar_color};"></div>'
        f'</div>'
        f'<span class="metric-value">{bar_pct}%</span>'
        f'</div>'
        f'</div>'
        f'<div class="metric-divider"></div>'
        f'<div class="metric-item">'
        f'<span class="metric-label">Sources Retrieved</span>'
        f'<span class="metric-value">{sources_cnt}</span>'
        f'</div>'
        f'<div class="metric-divider"></div>'
        f'<div class="metric-item">'
        f'<span class="metric-label">Citations Verified</span>'
        f'<span class="metric-value" style="color:var(--accent);">{cited_cnt} / {sources_cnt}</span>'
        f'</div>'
        f'</div>'
        f'<div style="display:flex; align-items:center; gap:6px; font-size:11px; color:var(--text-muted);">'
        f'<span>⚡</span><span>{latency:.2f}s</span>'
        f'</div>'
        f'</div>'
        f'<div style="font-size:10.5px; color:var(--text-muted); margin-top:3px; margin-bottom:6px; padding-left:2px;">'
        f'Retrieval confidence reflects indexed evidence relevance — not diagnostic certainty.'
        f'</div>'
    )

    st.markdown(html, unsafe_allow_html=True)

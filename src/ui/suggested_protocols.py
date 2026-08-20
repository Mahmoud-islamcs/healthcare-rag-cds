"""
BioGuard Suggested Protocols — equal-width 3-column clinical shortcut cards.
All text in English. Clicking submits the query directly.
"""
import streamlit as st


# Each entry: (icon, title, description, full_query)
PROTOCOLS = [
    (
        "🩸",
        "Type 2 Diabetes Management",
        "First-line therapy selection, HbA1c targets, and escalation criteria.",
        "What are the evidence-based first-line pharmacological options for Type 2 Diabetes management and when should therapy be escalated?"
    ),
    (
        "💉",
        "Insulin Initiation — T2DM",
        "Clinical criteria and HbA1c thresholds for initiating insulin therapy.",
        "What are the clinical indications and glycaemic thresholds for initiating insulin therapy in Type 2 Diabetes patients failing oral agents?"
    ),
    (
        "🫀",
        "Quadruple GDMT — HFrEF",
        "Guideline-directed initiation sequence with renal safety monitoring.",
        "What is the recommended quadruple GDMT initiation sequence for HFrEF including beta-blockers, ACEi/ARNi, MRA, and SGLT2i with renal safety monitoring?"
    ),
    (
        "🔬",
        "Finerenone + SGLT2i Protocol",
        "KDIGO criteria, potassium thresholds, and combination eligibility.",
        "What are the KDIGO inclusion criteria, eGFR thresholds, and serum potassium monitoring requirements for Finerenone combined with SGLT2 inhibitors in CKD?"
    ),
    (
        "💊",
        "Metformin Contraindications",
        "Renal function thresholds and clinical contraindications for Metformin.",
        "What are the renal function thresholds, clinical contraindications, and dose adjustment criteria for Metformin in Type 2 Diabetes?"
    ),
    (
        "📋",
        "Cardiovascular Risk Reduction",
        "Evidence-based interventions for CV risk in metabolic syndrome.",
        "What are the evidence-based cardiovascular risk reduction strategies for patients with Type 2 Diabetes and established cardiovascular disease?"
    ),
]


def render_suggested_protocols() -> str | None:
    """
    Render 3-column equal-width protocol cards.
    Returns the selected query string or None.
    """
    st.markdown("""
    <div style="text-align:center; margin-bottom:18px;">
        <div style="font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; color:var(--text-muted);">
            Suggested Clinical Protocols
        </div>
    </div>
    """, unsafe_allow_html=True)

    selected_query = None
    rows = [PROTOCOLS[:3], PROTOCOLS[3:6]]

    for row in rows:
        cols = st.columns(3, gap="medium")
        for col, (icon, title, desc, query) in zip(cols, row):
            with col:
                # Render card HTML
                st.markdown(f"""
                <div class="protocol-card" style="min-height:110px;">
                    <span class="protocol-card-icon">{icon}</span>
                    <div class="protocol-card-title">{title}</div>
                    <div class="protocol-card-desc">{desc}</div>
                </div>
                """, unsafe_allow_html=True)
                # Invisible button overlaid (Streamlit limitation workaround)
                if st.button(
                    "Select",
                    key=f"proto_{title[:12].replace(' ', '_')}",
                    use_container_width=True,
                    help=f"Ask: {query[:80]}…"
                ):
                    selected_query = query

    return selected_query

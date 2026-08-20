"""
BioGuard Design System — CSS tokens and theme injection.
Supports light and dark mode via CSS custom properties.
"""
import streamlit as st

# ──────────────────────────────────────────────────────────
# CSS Design Tokens + Full Theme
# ──────────────────────────────────────────────────────────
LIGHT_CSS = """
:root {
    --bg-base:        #F8FAFC;
    --bg-surface:     #FFFFFF;
    --bg-subtle:      #F1F5F9;
    --bg-overlay:     #E2E8F0;
    --text-primary:   #0F172A;
    --text-secondary: #475569;
    --text-muted:     #94A3B8;
    --border:         #E2E8F0;
    --border-focus:   #0EA5E9;
    --accent:         #0EA5E9;
    --accent-dark:    #0284C7;
    --accent-light:   #E0F2FE;
    --navy:           #0A2540;
    --success:        #10B981;
    --success-bg:     #ECFDF5;
    --warning:        #F59E0B;
    --warning-bg:     #FFFBEB;
    --danger:         #EF4444;
    --danger-bg:      #FEF2F2;
    --radius-sm:      6px;
    --radius-md:      10px;
    --radius-lg:      16px;
    --radius-xl:      22px;
    --shadow-sm:      0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    --shadow-md:      0 4px 12px rgba(0,0,0,0.06), 0 2px 4px rgba(0,0,0,0.04);
    --shadow-lg:      0 10px 30px rgba(0,0,0,0.08);
    --font-sans:      'Inter', 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
    --font-arabic:    'Noto Sans Arabic', 'Tajawal', 'Inter', sans-serif;
}
"""

DARK_CSS = """
:root {
    --bg-base:        #0B1120;
    --bg-surface:     #111827;
    --bg-subtle:      #1E293B;
    --bg-overlay:     #263347;
    --text-primary:   #F1F5F9;
    --text-secondary: #94A3B8;
    --text-muted:     #64748B;
    --border:         #1E293B;
    --border-focus:   #0EA5E9;
    --accent:         #38BDF8;
    --accent-dark:    #0EA5E9;
    --accent-light:   #0C2E42;
    --navy:           #E2E8F0;
    --success:        #34D399;
    --success-bg:     #064E3B;
    --warning:        #FBBF24;
    --warning-bg:     #451A03;
    --danger:         #F87171;
    --danger-bg:      #450A0A;
}
"""

BASE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Noto+Sans+Arabic:wght@400;500;600;700;800&display=swap');

/* ── Reset & Base ── */
html, body, .stApp { background-color: var(--bg-base) !important; }
.stApp { font-family: var(--font-sans) !important; color: var(--text-primary) !important; }

/* ── Force text color everywhere ── */
p, span, label, li, strong, em, h1, h2, h3, h4, h5, h6,
div[data-testid="stMarkdownContainer"] *,
.stMarkdown *, .element-container * {
    color: var(--text-primary) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden !important; }
div[data-testid="stToolbar"] { display: none !important; }
section[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border) !important;
}

/* ── Top-level container ── */
.block-container {
    padding-top: 0 !important;
    padding-bottom: 24px !important;
    max-width: 1200px !important;
}

/* ── Tabs ── */
div[data-testid="stTabs"] {
    border-bottom: 1px solid var(--border) !important;
}
div[data-testid="stTabs"] button[role="tab"] {
    font-weight: 600 !important;
    font-size: 13px !important;
    color: var(--text-secondary) !important;
    padding: 10px 18px !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
    transition: all 0.15s ease !important;
}
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
    background: transparent !important;
}
div[data-testid="stTabContent"] { padding-top: 20px !important; }

/* ── Inputs ── */
div[data-testid="stTextInput"] > div,
div[data-testid="stTextArea"] > div {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    background: var(--bg-surface) !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
}
div[data-testid="stTextInput"] > div:focus-within,
div[data-testid="stTextArea"] > div:focus-within {
    border-color: var(--border-focus) !important;
    box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.12) !important;
}
div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea {
    color: var(--text-primary) !important;
    background: transparent !important;
    font-size: 14px !important;
    font-family: var(--font-sans) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--bg-surface) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 8px 14px !important;
    box-shadow: var(--shadow-sm) !important;
    transition: all 0.15s ease !important;
    font-family: var(--font-sans) !important;
}
.stButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: var(--accent-light) !important;
    box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.1) !important;
}
button[kind="primary"],
div[data-testid="stFormSubmitButton"] > button {
    background: #60A5FA !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 8px rgba(96, 165, 250, 0.35) !important;
    letter-spacing: 0.01em !important;
}
button[kind="primary"]:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    background: #3B82F6 !important;
    box-shadow: 0 4px 14px rgba(96, 165, 250, 0.45) !important;
    color: #FFFFFF !important;
}

/* ── Chat messages ── */
div[data-testid="stChatMessage"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    padding: 16px 20px !important;
    margin-bottom: 12px !important;
    box-shadow: var(--shadow-sm) !important;
}
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: var(--accent-light) !important;
    border-color: rgba(14, 165, 233, 0.2) !important;
}

/* ── Chat bottom input ── */
.stChatInputContainer > div {
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    background: var(--bg-surface) !important;
    box-shadow: var(--shadow-md) !important;
}
.stChatInputContainer > div:focus-within {
    border-color: var(--border-focus) !important;
    box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.12) !important;
}
.stChatInputContainer textarea { color: var(--text-primary) !important; }

/* ── Form ── */
div[data-testid="stForm"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-xl) !important;
    box-shadow: var(--shadow-md) !important;
    padding: 20px 24px 16px 24px !important;
}
div[data-testid="stForm"]:focus-within {
    border-color: var(--border-focus) !important;
    box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.1), var(--shadow-md) !important;
}

/* ── Expander ── */
div[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    background: var(--bg-subtle) !important;
    overflow: hidden !important;
}
div[data-testid="stExpander"] summary {
    font-weight: 600 !important;
    font-size: 13px !important;
    color: var(--text-secondary) !important;
    padding: 10px 14px !important;
}

/* ── Select / Multiselect ── */
div[data-testid="stSelectbox"] > div,
div[data-testid="stMultiSelect"] > div {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    background: var(--bg-surface) !important;
}

/* ── Alerts ── */
div[data-testid="stAlert"] {
    border-radius: var(--radius-md) !important;
}

/* ── JSON ── */
div[data-testid="stJson"] {
    background: var(--bg-subtle) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    font-size: 12px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

/* ── RTL Arabic support ── */
.arabic-content {
    direction: rtl !important;
    text-align: right !important;
    font-family: var(--font-arabic) !important;
    line-height: 1.9 !important;
}
.ltr-content {
    direction: ltr !important;
    text-align: left !important;
}

/* ── Clean Arabic & Chat Typography Spacing ── */
div[data-testid="stChatMessage"] h3, 
div[data-testid="stChatMessage"] h4, 
div[data-testid="stChatMessage"] strong {
    margin-top: 12px !important;
    margin-bottom: 6px !important;
    display: inline-block !important;
    font-size: 15px !important;
    font-weight: 700 !important;
}

div[data-testid="stChatMessage"] ul {
    margin-top: 6px !important;
    margin-bottom: 12px !important;
    padding-inline-start: 24px !important;
}

div[data-testid="stChatMessage"] li {
    margin-bottom: 8px !important;
    line-height: 1.85 !important;
}

div[data-testid="stChatMessage"] p {
    margin-bottom: 8px !important;
    line-height: 1.85 !important;
}

/* ── BioGuard Custom Components ── */
.bioguard-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 0 14px 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 0;
}
.bioguard-logo {
    display: flex;
    align-items: center;
    gap: 10px;
}
.bioguard-logo-name {
    font-size: 18px;
    font-weight: 800;
    color: var(--text-primary) !important;
    letter-spacing: -0.02em;
}
.bioguard-tagline {
    font-size: 10px;
    font-weight: 500;
    color: var(--text-muted) !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    display: block;
}
.header-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    background: var(--bg-subtle);
    color: var(--text-secondary) !important;
    border: 1px solid var(--border);
}
.header-badge.operational {
    background: var(--success-bg);
    color: var(--success) !important;
    border-color: var(--success);
}
.status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--success);
    display: inline-block;
}

/* ── Protocol Cards ── */
.protocol-card {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 16px 18px;
    cursor: pointer;
    transition: all 0.18s ease;
    box-shadow: var(--shadow-sm);
    height: 100%;
}
.protocol-card:hover {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px rgba(14,165,233,0.1), var(--shadow-sm);
    transform: translateY(-1px);
}
.protocol-card-icon {
    font-size: 20px;
    margin-bottom: 10px;
    display: block;
}
.protocol-card-title {
    font-size: 13px;
    font-weight: 700;
    color: var(--text-primary) !important;
    margin-bottom: 4px;
    line-height: 1.4;
}
.protocol-card-desc {
    font-size: 11.5px;
    color: var(--text-muted) !important;
    line-height: 1.5;
}

/* ── Metrics / Evidence Banner ── */
.evidence-banner {
    background: var(--bg-subtle);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 12px 16px;
    margin-top: 14px;
    margin-bottom: 4px;
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    align-items: center;
    justify-content: space-between;
}
.metric-item {
    display: flex;
    flex-direction: column;
    gap: 2px;
}
.metric-label {
    font-size: 9.5px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted) !important;
}
.metric-value {
    font-size: 13px;
    font-weight: 700;
    color: var(--text-primary) !important;
}
.metric-divider {
    width: 1px;
    height: 28px;
    background: var(--border);
}
.quality-badge {
    display: inline-flex;
    align-items: center;
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.05em;
}
.quality-high { background: var(--success-bg); color: var(--success) !important; }
.quality-moderate { background: var(--warning-bg); color: var(--warning) !important; }
.quality-low { background: var(--danger-bg); color: var(--danger) !important; }

/* ── Progress bar ── */
.evidence-bar-wrap {
    width: 140px;
    height: 6px;
    background: var(--bg-overlay);
    border-radius: 3px;
    overflow: hidden;
}
.evidence-bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.4s ease;
}

/* ── Citation Card ── */
.citation-card {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 12px 15px;
    margin-bottom: 8px;
}
.citation-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 6px;
    gap: 8px;
}
.citation-id {
    font-weight: 800;
    font-size: 12px;
    color: var(--accent) !important;
    white-space: nowrap;
}
.citation-file {
    font-weight: 600;
    font-size: 12px;
    color: var(--text-primary) !important;
}
.citation-meta {
    font-size: 11px;
    color: var(--text-muted) !important;
    margin-bottom: 6px;
}
.citation-snippet {
    font-size: 12px;
    color: var(--text-secondary) !important;
    font-style: italic;
    line-height: 1.6;
    border-left: 2px solid var(--border-focus);
    padding-left: 10px;
}
.badge {
    display: inline-flex;
    align-items: center;
    gap: 3px;
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 10.5px;
    font-weight: 700;
    flex-shrink: 0;
}
.badge-cited   { background: var(--success-bg); color: var(--success) !important; }
.badge-bg      { background: var(--bg-overlay); color: var(--text-secondary) !important; }
.badge-warn    { background: var(--warning-bg); color: var(--warning) !important; }

/* ── Emergency Alert ── */
.emergency-alert {
    background: var(--danger-bg);
    border: 1.5px solid var(--danger);
    border-radius: var(--radius-md);
    padding: 16px 18px;
    margin-bottom: 10px;
}

/* ── Sidebar components ── */
.sidebar-section-title {
    font-size: 9.5px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-muted) !important;
    padding: 14px 0 6px 0;
}
.history-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 7px 10px;
    border-radius: var(--radius-sm);
    cursor: pointer;
    font-size: 12.5px;
    color: var(--text-secondary) !important;
    transition: all 0.15s ease;
}
.history-item:hover {
    background: var(--bg-subtle);
    color: var(--text-primary) !important;
}

/* ── System capability pills ── */
.cap-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    background: var(--bg-subtle);
    border: 1px solid var(--border);
    color: var(--text-secondary) !important;
}

/* ── Observability Panel ── */
.obs-step {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 12px;
    font-size: 12px;
    border-radius: var(--radius-sm);
    margin-bottom: 4px;
    background: var(--bg-subtle);
    border: 1px solid var(--border);
}
.obs-done { border-color: var(--success) !important; }
.obs-step-label { color: var(--text-secondary) !important; font-weight: 500; }
.obs-step-icon { font-size: 14px; }

/* ── Upload area ── */
div[data-testid="stFileUploader"] {
    border: 1.5px dashed var(--border) !important;
    border-radius: var(--radius-lg) !important;
    background: var(--bg-subtle) !important;
    padding: 10px !important;
    transition: border-color 0.15s ease !important;
}
div[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
    background: var(--accent-light) !important;
}

/* ── Toggle / Checkbox ── */
div[data-testid="stCheckbox"] label {
    font-size: 13px !important;
    font-weight: 500 !important;
    color: var(--text-primary) !important;
}

/* ── Spinner ── */
div[data-testid="stSpinner"] p {
    color: var(--text-secondary) !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}

/* ── Disclaimer footer ── */
.disclaimer-footer {
    font-size: 11px;
    color: var(--text-muted) !important;
    text-align: center;
    padding: 10px 0;
    border-top: 1px solid var(--border);
    margin-top: 16px;
    line-height: 1.6;
}
"""


def inject_css(dark_mode: bool = False) -> None:
    """Inject full BioGuard design system CSS."""
    theme_vars = DARK_CSS if dark_mode else LIGHT_CSS
    st.markdown(
        f"<style>{theme_vars}{BASE_CSS}</style>",
        unsafe_allow_html=True,
    )

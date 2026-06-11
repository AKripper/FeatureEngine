CSS = """
<style>
    /* ── Global typography ─────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    code, pre, .stCode {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* ── Sidebar ───────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: #0f1117;
        border-right: 1px solid #1e2130;
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #e2e8f0;
    }
    [data-testid="stSidebar"] label {
        color: #94a3b8 !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.04em !important;
        text-transform: uppercase !important;
    }

    /* ── Main background ───────────────────────────────────────── */
    .main .block-container {
        background: #0d0f16;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    /* ── Metric cards ──────────────────────────────────────────── */
    [data-testid="metric-container"] {
        background: #161b27;
        border: 1px solid #1e2736;
        border-radius: 10px;
        padding: 1rem 1.2rem;
    }
    [data-testid="metric-container"] label {
        color: #64748b !important;
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #e2e8f0 !important;
        font-weight: 700 !important;
        font-size: 1.7rem !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricDelta"] {
        font-size: 0.85rem !important;
        font-weight: 600 !important;
    }

    /* ── Cards / info boxes ────────────────────────────────────── */
    .feat-card {
        background: #161b27;
        border: 1px solid #1e2736;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
    }
    .feat-card-title {
        color: #7dd3fc;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        margin-bottom: 0.25rem;
    }
    .feat-card-desc {
        color: #94a3b8;
        font-size: 0.8rem;
        line-height: 1.55;
    }
    .feat-card-code {
        background: #0d1117;
        border-radius: 6px;
        padding: 0.4rem 0.7rem;
        margin-top: 0.5rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.76rem;
        color: #a5f3fc;
        word-break: break-all;
    }
    .badge {
        display: inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 99px;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        margin-bottom: 0.3rem;
    }
    .badge-transform { background: #1e3a5f; color: #7dd3fc; }
    .badge-interaction { background: #2d1e4f; color: #c4b5fd; }
    .badge-encoding { background: #1e3d2a; color: #6ee7b7; }
    .badge-polynomial { background: #3d2a1e; color: #fcd34d; }
    .badge-normalization { background: #1e3a30; color: #34d399; }
    .badge-binning { background: #3a1e1e; color: #fca5a5; }
    .badge-missing { background: #3a2a1e; color: #fdba74; }
    .badge-group { background: #1e2a3a; color: #93c5fd; }
    .badge-default { background: #1e2433; color: #94a3b8; }

    /* ── Section headers ───────────────────────────────────────── */
    .section-header {
        color: #e2e8f0;
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #1e2736;
        margin-bottom: 1rem;
        margin-top: 0.5rem;
    }

    /* ── Delta colors ──────────────────────────────────────────── */
    .delta-pos { color: #34d399; font-weight: 700; }
    .delta-neg { color: #f87171; font-weight: 700; }
    .delta-neu { color: #94a3b8; font-weight: 600; }

    /* ── Log box ───────────────────────────────────────────────── */
    .log-box {
        background: #0a0c12;
        border: 1px solid #1e2736;
        border-radius: 8px;
        padding: 0.7rem 1rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #64748b;
        max-height: 160px;
        overflow-y: auto;
        line-height: 1.7;
    }

    /* ── Tabs ──────────────────────────────────────────────────── */
    [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    [data-baseweb="tab"] {
        background: #161b27 !important;
        border-radius: 8px 8px 0 0 !important;
        color: #64748b !important;
        font-weight: 600 !important;
        font-size: 0.8rem !important;
        padding: 0.4rem 1rem !important;
    }
    [aria-selected="true"] {
        background: #1e293b !important;
        color: #7dd3fc !important;
        border-bottom: 2px solid #38bdf8 !important;
    }

    /* ── Buttons ───────────────────────────────────────────────── */
    .stButton > button {
        background: #1e3a5f !important;
        color: #bae6fd !important;
        border: 1px solid #2563eb !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        transition: background 0.15s ease;
    }
    .stButton > button:hover {
        background: #2563eb !important;
        color: #fff !important;
    }

    /* ── Dataframe ─────────────────────────────────────────────── */
    [data-testid="stDataFrame"] {
        border: 1px solid #1e2736 !important;
        border-radius: 8px !important;
    }

    /* ── Hero header ───────────────────────────────────────────── */
    .hero {
        background: linear-gradient(135deg, #0f2027 0%, #161b27 60%, #0d1117 100%);
        border: 1px solid #1e2736;
        border-radius: 14px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
    }
    .hero-title {
        color: #e2e8f0;
        font-size: 1.6rem;
        font-weight: 800;
        letter-spacing: -0.01em;
        margin: 0;
        line-height: 1.2;
    }
    .hero-title span {
        color: #38bdf8;
    }
    .hero-sub {
        color: #64748b;
        font-size: 0.85rem;
        margin-top: 0.4rem;
        font-weight: 400;
    }

    /* ── Upload area ───────────────────────────────────────────── */
    [data-testid="stFileUploader"] {
        background: #161b27 !important;
        border: 2px dashed #2563eb !important;
        border-radius: 12px !important;
    }
    </style>
"""
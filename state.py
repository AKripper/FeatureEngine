"""
state.py — Session state initialization and management.
All st.session_state keys are declared here. No ML or UI logic.
"""

import streamlit as st


def init_state() -> None:
    """
    Initialize all session-state keys exactly once per session.
    Call this at the very top of app.py before any rendering.
    """

    # ── Raw data ──────────────────────────────────────────────────────────────
    if "raw_df" not in st.session_state:
        st.session_state.raw_df = None          # pd.DataFrame | None

    if "feature_df" not in st.session_state:
        st.session_state.feature_df = None      # engineered feature DataFrame

    # ── Target / task configuration ───────────────────────────────────────────
    if "target_col" not in st.session_state:
        st.session_state.target_col = None      # str | None

    if "task_type" not in st.session_state:
        st.session_state.task_type = None       # "classification" | "regression" | None

    if "feature_cols" not in st.session_state:
        st.session_state.feature_cols = []      # list[str]  selected predictors

    # ── Baseline results ──────────────────────────────────────────────────────
    if "baseline_results" not in st.session_state:
        st.session_state.baseline_results = None   # dict returned by engine

    if "baseline_score" not in st.session_state:
        st.session_state.baseline_score = None     # float | None

    if "baseline_metric" not in st.session_state:
        st.session_state.baseline_metric = None    # str | None

    # ── Engineered-feature evaluation ─────────────────────────────────────────
    if "eval_results" not in st.session_state:
        st.session_state.eval_results = None       # dict returned by engine

    if "custom_feature_code" not in st.session_state:
        st.session_state.custom_feature_code = ""  # str  python snippet

    if "custom_feature_name" not in st.session_state:
        st.session_state.custom_feature_name = ""  # str

    # ── Leaderboard ───────────────────────────────────────────────────────────
    if "leaderboard" not in st.session_state:
        st.session_state.leaderboard = []          # list[dict]  one row per run

    # ── Feature suggestions ───────────────────────────────────────────────────
    if "suggestions" not in st.session_state:
        st.session_state.suggestions = []          # list[dict]  from engine

    # ── UI helpers ────────────────────────────────────────────────────────────
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = 0

    if "upload_key" not in st.session_state:
        st.session_state.upload_key = 0           # int  incremented to reset uploader

    if "eval_log" not in st.session_state:
        st.session_state.eval_log = []            # list[str]  timestamped log lines

    if "cv_folds" not in st.session_state:
        st.session_state.cv_folds = 5             # int

    if "random_seed" not in st.session_state:
        st.session_state.random_seed = 42         # int


def reset_state() -> None:
    """
    Hard-reset all keys so a new file upload starts fresh.
    Preserves upload_key (incremented to force Streamlit uploader reset).
    """
    current_key = st.session_state.get("upload_key", 0)
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.upload_key = current_key + 1
    init_state()

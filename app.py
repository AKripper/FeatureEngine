"""
app.py — Streamlit UI for the Feature Engineering Tool.
All rendering, layout, and user interaction is here.
Business logic is delegated to engine.py. State is managed in state.py.
"""

from __future__ import annotations

import io
import traceback

import numpy as np
import pandas as pd
import streamlit as st

from styles import CSS
from load_sample_datasets import SAMPLE_DATASETS

import engine
import state as state_module




# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Feature Engineering Tool",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# STATE INIT
# ─────────────────────────────────────────────────────────────────────────────

state_module.init_state()

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    from datetime import datetime
    st.session_state.eval_log.append(
        f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    )
    if len(st.session_state.eval_log) > 100:
        st.session_state.eval_log = st.session_state.eval_log[-100:]


def _badge_html(category: str) -> str:
    mapping = {
        "Transformation": "badge-transform",
        "Interaction": "badge-interaction",
        "Encoding": "badge-encoding",
        "Polynomial": "badge-polynomial",
        "Normalization": "badge-normalization",
        "Binning": "badge-binning",
        "Missing Indicator": "badge-missing",
        "Group Aggregate": "badge-group",
    }
    cls = mapping.get(category, "badge-default")
    return f'<span class="badge {cls}">{category}</span>'


def _delta_html(delta: float | None) -> str:
    if delta is None:
        return '<span class="delta-neu">— baseline</span>'
    if delta > 0:
        return f'<span class="delta-pos">▲ +{delta:.5f}</span>'
    if delta < 0:
        return f'<span class="delta-neg">▼ {delta:.5f}</span>'
    return f'<span class="delta-neu">= {delta:.5f}</span>'


def _load_dataframe(uploaded_file) -> pd.DataFrame | None:
    """Parse uploaded CSV or Parquet file into a DataFrame."""
    try:
        name = uploaded_file.name.lower()
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif name.endswith(".parquet"):
            df = pd.read_parquet(uploaded_file)
        elif name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload CSV, Parquet, or Excel.")
            return None
        return df
    except Exception as exc:
        st.error(f"Failed to read file: {exc}")
        return None

def _load_sample_dataset(dataset_name: str) -> pd.DataFrame | None:
    try:
        return pd.read_csv(SAMPLE_DATASETS[dataset_name])
    except Exception as exc:
        st.error(f"Failed to load sample dataset: {exc}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            '<p style="color:#38bdf8;font-weight:800;font-size:1.1rem;'
            'letter-spacing:-0.01em;margin-bottom:0.1rem;">⚙️ Feature Engineering Tool</p>',
            unsafe_allow_html=True,
        )


        st.markdown(
            '<p style="color:#475569;font-size:0.72rem;margin-top:0;margin-bottom:1.5rem;">'
            "Upload · Baseline · Engineer · Track</p>",
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("**📂 Data Source**")

        source_type = st.radio(
            "Choose data source",
            ["Upload File", "Sample Dataset"],
        )

        df = None

        if source_type == "Upload File":

            uploaded = st.file_uploader(
                "Upload dataset (CSV / Parquet / Excel)",
                type=["csv", "parquet", "xls", "xlsx"],
                key=f"uploader_{st.session_state.upload_key}",
                help="Your data stays in-browser. Nothing is sent anywhere.",
            )

            if uploaded is not None:
                df = _load_dataframe(uploaded)

        else:

            dataset_name = st.selectbox(
                "Sample dataset",
                list(SAMPLE_DATASETS.keys()),
            )

            if st.button("Load Sample Dataset"):
                df = _load_sample_dataset(dataset_name)

        if df is not None:

            existing_df = st.session_state.raw_df
            if existing_df is None:
                existing_df = pd.DataFrame()

            if not df.equals(existing_df):
                st.session_state.raw_df = df
                st.session_state.baseline_results = None
                st.session_state.eval_results = None
                st.session_state.suggestions = []
                st.session_state.leaderboard = []

                _log(
                    f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} cols"
                )

                st.rerun()

        if st.session_state.raw_df is not None:
            df = st.session_state.raw_df
            st.success(f"✓ {df.shape[0]:,} rows · {df.shape[1]} columns")

            st.markdown("---")
            st.markdown("**🎯 Target & Task**")

            cols = df.columns.tolist()
            target_col = st.selectbox(
                "Target column",
                options=cols,
                index=(
                    cols.index(st.session_state.target_col)
                    if st.session_state.target_col in cols
                    else len(cols) - 1
                ),
            )
            st.session_state.target_col = target_col


            inferred = engine.infer_task_type(df[target_col])
            task_type = st.radio(
                "Task type",
                options=["classification", "regression"],
                index=0 if inferred == "classification" else 1,
                horizontal=True,
            )
            st.session_state.task_type = task_type

            st.markdown("---")
            st.markdown("**📊 Feature Selection**")

            all_feature_candidates = [c for c in df.columns if c != target_col]
            default_features = (
                st.session_state.feature_cols
                if st.session_state.feature_cols
                else all_feature_candidates
            )
            # Clamp defaults to valid candidates
            default_features = [c for c in default_features if c in all_feature_candidates]

            feature_cols = st.multiselect(
                "Predictor columns",
                options=all_feature_candidates,
                default=default_features,
            )
            st.session_state.feature_cols = feature_cols

            st.markdown("---")
            st.markdown("**⚙️ CV Settings**")
            st.session_state.cv_folds = st.slider("CV folds", 2, 10, st.session_state.cv_folds)
            st.session_state.random_seed = st.number_input(
                "Random seed", value=st.session_state.random_seed, step=1, min_value=0
            )

            st.markdown("---")
            if st.button("🔄 Reset Everything", use_container_width=True):
                state_module.reset_state()
                st.rerun()
        else:
            st.info("Upload a dataset to get started.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — DATA OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────

def render_data_overview() -> None:
    df = st.session_state.raw_df
    if df is None:
        st.info("📂 Upload a dataset or Choose sample dataset using the sidebar to begin.")
        return

    st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Rows", f"{df.shape[0]:,}")
    with c2:
        st.metric("Columns", df.shape[1])
    with c3:
        st.metric("Numeric cols", len(df.select_dtypes(include="number").columns))
    with c4:
        missing_total = int(df.isnull().sum().sum())
        st.metric("Missing values", f"{missing_total:,}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Column Statistics</div>', unsafe_allow_html=True)

    try:
        stats_df = engine.compute_descriptive_stats(df)
        st.dataframe(stats_df, use_container_width=True, height=320)
    except Exception as exc:
        st.error(f"Could not compute column stats: {exc}")

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="section-header">Data Preview (first 100 rows)</div>', unsafe_allow_html=True)
        st.dataframe(df.head(100), use_container_width=True, height=260)

    with col_right:
        st.markdown('<div class="section-header">Target Distribution</div>', unsafe_allow_html=True)
        target = st.session_state.target_col
        if target and target in df.columns:
            try:
                if engine.infer_task_type(df[target]) == "classification":
                    dist = df[target].value_counts().reset_index()
                    dist.columns = ["value", "count"]
                    st.bar_chart(dist.set_index("value"), height=260)
                else:
                    hist_data = pd.cut(df[target].dropna(), bins=30).value_counts().sort_index()
                    hist_df = pd.DataFrame(
                        {"bin": [str(b) for b in hist_data.index], "count": hist_data.values}
                    )
                    st.bar_chart(hist_df.set_index("bin"), height=260)
            except Exception as exc:
                st.warning(f"Could not plot target distribution: {exc}")
        else:
            st.info("Select a target column in the sidebar.")

    # Correlation heatmap (numeric cols, top 15)
    num_df = df.select_dtypes(include="number")
    if num_df.shape[1] >= 2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">Correlation Matrix (top 15 numeric cols)</div>', unsafe_allow_html=True)
        try:
            top_cols = num_df.columns[:15].tolist()
            corr = num_df[top_cols].corr()
            st.dataframe(corr.round(3).style.background_gradient(cmap="coolwarm", axis=None), use_container_width=True, height=300)
        except Exception as exc:
            st.warning(f"Could not render correlation matrix: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — BASELINE
# ─────────────────────────────────────────────────────────────────────────────

def render_baseline() -> None:
    df = st.session_state.raw_df
    if df is None:
        st.info("📂 Upload a dataset first.")
        return

    if not st.session_state.feature_cols:
        st.warning("⚠️ Select at least one predictor column in the sidebar.")
        return

    st.markdown('<div class="section-header">Cross-Validation Baseline</div>', unsafe_allow_html=True)

    col_info, col_btn = st.columns([3, 1])
    with col_info:
        st.markdown(
            f"<span style='color:#64748b;font-size:0.82rem;'>"
            f"Task: <b style='color:#7dd3fc'>{st.session_state.task_type}</b> · "
            f"Target: <b style='color:#7dd3fc'>{st.session_state.target_col}</b> · "
            f"{len(st.session_state.feature_cols)} features · "
            f"{st.session_state.cv_folds}-fold CV"
            f"</span>",
            unsafe_allow_html=True,
        )
    with col_btn:
        run_baseline = st.button("▶ Run Baseline", use_container_width=True)

    if run_baseline:
        with st.spinner("Training cross-validation baseline…"):
            try:
                result = engine.run_baseline(
                    df=df,
                    feature_cols=st.session_state.feature_cols,
                    target_col=st.session_state.target_col,
                    task_type=st.session_state.task_type,
                    cv_folds=st.session_state.cv_folds,
                    random_seed=st.session_state.random_seed,
                )
                st.session_state.baseline_results = result
                st.session_state.baseline_score = result["primary_score"]
                st.session_state.baseline_metric = result["primary_metric"]

                entry = engine.build_leaderboard_entry(
                    "Baseline",
                    result,
                    None,
                    st.session_state.task_type,
                )
                # Remove any existing baseline entry
                st.session_state.leaderboard = [
                    e for e in st.session_state.leaderboard if e["run"] != "Baseline"
                ]
                st.session_state.leaderboard.insert(0, entry)
                _log(
                    f"Baseline complete — {result['primary_metric']}="
                    f"{result['primary_score']:.5f}"
                )
                st.success("✓ Baseline established!")
            except Exception as exc:
                st.error(f"Baseline failed: {exc}")
                _log(f"ERROR: {traceback.format_exc()}")

    results = st.session_state.baseline_results
    if results is None:
        st.info("Click **▶ Run Baseline** to establish a cross-validated score.")
        return

    # ── Primary metrics row ───────────────────────────────────────────────────
    mean_m = results["mean_metrics"]
    std_m = results["std_metrics"]
    primary = results["primary_metric"]

    st.markdown("<br>", unsafe_allow_html=True)
    metric_keys = list(mean_m.keys())
    cols = st.columns(len(metric_keys))
    for i, key in enumerate(metric_keys):
        with cols[i]:
            label = key.upper()
            val = f"{mean_m[key]:.4f}"
            delta_str = f"±{std_m[key]:.4f}"
            if key == primary:
                label = f"★ {label}"
            st.metric(label, val, delta=delta_str, delta_color="off")

    # ── Per-fold table ────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Per-Fold Results</div>', unsafe_allow_html=True)
    fold_df = pd.DataFrame(results["fold_metrics"])
    st.dataframe(fold_df.round(5), use_container_width=True)

    # ── Feature importance ────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Feature Importance (Permutation Importance)</div>', unsafe_allow_html=True)
    with st.spinner("Computing feature importance…"):
        try:
            imp_df = engine.compute_feature_importance(
                df,
                st.session_state.feature_cols,
                st.session_state.target_col,
                st.session_state.task_type,
                st.session_state.random_seed,
            )
            col_chart, col_table = st.columns([2, 1])
            with col_chart:
                top_n = min(20, len(imp_df))
                chart_df = imp_df.head(top_n).set_index("feature")
                st.bar_chart(chart_df, height=350)
            with col_table:
                st.dataframe(imp_df.round(5), use_container_width=True, height=350)
        except Exception as exc:
            st.warning(f"Could not compute feature importance: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def render_feature_engineering() -> None:
    df = st.session_state.raw_df
    if df is None:
        st.info("📂 Upload a dataset first.")
        return

    if st.session_state.baseline_results is None:
        st.warning("⚠️ Run a baseline first (Tab 2) before evaluating custom features.")
        return

    st.markdown('<div class="section-header">Custom Feature Builder</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown(
            "<span style='color:#94a3b8;font-size:0.8rem;'>"
            "Write any Python expression that references <code>df</code>, "
            "<code>np</code>, and <code>pd</code>. "
            "It must return a 1-D array or Series of the same length as the dataset."
            "</span>",
            unsafe_allow_html=True,
        )
        feat_name = st.text_input(
            "Feature name",
            value=st.session_state.custom_feature_name or "my_feature",
            placeholder="e.g. log_price",
        )
        feat_code = st.text_area(
            "Feature expression",
            value=st.session_state.custom_feature_code or "",
            placeholder="e.g. np.log1p(df['price'])",
            height=100,
        )
        st.session_state.custom_feature_name = feat_name
        st.session_state.custom_feature_code = feat_code

        eval_btn = st.button("▶ Evaluate Feature", use_container_width=True)

    with col_b:
        # Quick reference for column names
        st.markdown(
            "<span style='color:#64748b;font-size:0.75rem;font-weight:600;'>"
            "AVAILABLE COLUMNS</span>",
            unsafe_allow_html=True,
        )
        col_info = []
        for c in df.columns:
            dtype = str(df[c].dtype)
            col_info.append({"column": c, "dtype": dtype})
        st.dataframe(
            pd.DataFrame(col_info),
            use_container_width=True,
            height=190,
            hide_index=True,
        )

    if eval_btn:
        if not feat_name.strip():
            st.error("Please give the feature a name.")
        elif not feat_code.strip():
            st.error("Please provide a feature expression.")
        elif feat_name.strip() in st.session_state.feature_cols:
            st.error(f"A feature named '{feat_name}' already exists in the predictor set.")
        else:
            with st.spinner("Evaluating new feature via cross-validation…"):
                try:
                    result = engine.evaluate_feature(
                        df=df,
                        feature_cols=st.session_state.feature_cols,
                        target_col=st.session_state.target_col,
                        task_type=st.session_state.task_type,
                        feature_name=feat_name.strip(),
                        feature_code=feat_code.strip(),
                        cv_folds=st.session_state.cv_folds,
                        random_seed=st.session_state.random_seed,
                    )
                    st.session_state.eval_results = result
                    if result.get("success"):
                        entry = engine.build_leaderboard_entry(
                            feat_name.strip(),
                            result,
                            st.session_state.baseline_score,
                            st.session_state.task_type,
                        )
                        # Remove duplicate run name
                        st.session_state.leaderboard = [
                            e for e in st.session_state.leaderboard
                            if e["run"] != feat_name.strip()
                        ]
                        st.session_state.leaderboard.append(entry)
                        _log(
                            f"Evaluated '{feat_name}' — "
                            f"{result['primary_metric']}={result['primary_score']:.5f}"
                        )
                        st.success(f"✓ Feature '{feat_name}' evaluated!")
                    else:
                        error_msg = result.get("error", "Unknown error")
                        st.error(f"Feature evaluation failed:\n```\n{error_msg}\n```")
                        _log(f"ERROR evaluating '{feat_name}': {error_msg[:120]}")
                except Exception as exc:
                    st.error(f"Unexpected error: {exc}")
                    _log(f"UNEXPECTED ERROR: {traceback.format_exc()[:200]}")

    # ── Eval results ──────────────────────────────────────────────────────────
    result = st.session_state.eval_results
    if result and result.get("success"):
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">Evaluation Results</div>', unsafe_allow_html=True)

        mean_m = result["mean_metrics"]
        std_m = result["std_metrics"]
        primary = result["primary_metric"]
        baseline_score = st.session_state.baseline_score
        delta = result["primary_score"] - baseline_score if baseline_score is not None else None

        cols = st.columns(len(mean_m) + 1)
        for i, (key, val) in enumerate(mean_m.items()):
            with cols[i]:
                label = f"★ {key.upper()}" if key == primary else key.upper()
                d_str = f"±{std_m[key]:.4f}"
                st.metric(label, f"{val:.4f}", delta=d_str, delta_color="off")

        with cols[-1]:
            delta_label = "ΔBASELINE"
            if delta is not None:
                sign = "+" if delta >= 0 else ""
                d_color = "normal" if delta > 0 else ("inverse" if delta < 0 else "off")
                st.metric(delta_label, f"{sign}{delta:.5f}", delta_color=d_color)
            else:
                st.metric(delta_label, "—")

        fold_df = pd.DataFrame(result["fold_metrics"])
        st.dataframe(fold_df.round(5), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — AI SUGGESTIONS
# ─────────────────────────────────────────────────────────────────────────────

def render_suggestions() -> None:
    df = st.session_state.raw_df
    if df is None:
        st.info("📂 Upload a dataset first.")
        return

    if not st.session_state.feature_cols:
        st.warning("⚠️ Select at least one predictor column in the sidebar.")
        return

    st.markdown('<div class="section-header">Algorithmic Feature Suggestions</div>', unsafe_allow_html=True)
    st.markdown(
        "<span style='color:#64748b;font-size:0.8rem;'>"
        "Rule-based analysis of your schema — skew, cardinality, correlations, and missing rates — "
        "to generate relevant feature ideas. No external API required."
        "</span>",
        unsafe_allow_html=True,
    )

    gen_btn = st.button("🔍 Generate Suggestions", use_container_width=False)

    if gen_btn:
        with st.spinner("Analysing schema…"):
            try:
                profile = engine.profile_dataframe(df)
                suggestions = engine.generate_suggestions(
                    profile=profile,
                    target_col=st.session_state.target_col,
                    task_type=st.session_state.task_type,
                    feature_cols=st.session_state.feature_cols,
                )
                st.session_state.suggestions = suggestions
                _log(f"Generated {len(suggestions)} feature suggestions.")
            except Exception as exc:
                st.error(f"Could not generate suggestions: {exc}")

    suggestions = st.session_state.suggestions
    if not suggestions:
        if not gen_btn:
            st.info("Click **🔍 Generate Suggestions** to analyse your dataset.")
        return

    # ── Filter bar ────────────────────────────────────────────────────────────
    categories = sorted(set(s["category"] for s in suggestions))
    selected_cats = st.multiselect(
        "Filter by category",
        options=categories,
        default=categories,
    )
    filtered = [s for s in suggestions if s["category"] in selected_cats]

    st.markdown(
        f"<span style='color:#64748b;font-size:0.78rem;'>"
        f"Showing {len(filtered)} of {len(suggestions)} suggestions</span>",
        unsafe_allow_html=True,
    )

    if st.session_state.baseline_results is None:
        st.warning(
            "⚠️ Establish a baseline first (Tab 2) to enable one-click evaluation."
        )

    # ── Render cards ──────────────────────────────────────────────────────────
    for i in range(0, len(filtered), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(filtered):
                break
            s = filtered[idx]
            with col:
                eval_key = f"eval_suggestion_{idx}_{s['name']}"
                html = (
                    f'<div class="feat-card">'
                    f'{_badge_html(s["category"])}'
                    f'<div class="feat-card-title">{s["name"]}</div>'
                    f'<div class="feat-card-desc">{s["description"]}</div>'
                    f'<div class="feat-card-code">{s["code"]}</div>'
                    f"</div>"
                )
                st.markdown(html, unsafe_allow_html=True)

                btn_col, _ = st.columns([1, 2])
                with btn_col:
                    if st.button("▶ Try this", key=eval_key, use_container_width=True):
                        st.session_state.custom_feature_name = s["name"]
                        st.session_state.custom_feature_code = s["code"]
                        if st.session_state.baseline_results is not None:
                            with st.spinner(f"Evaluating '{s['name']}'…"):
                                try:
                                    result = engine.evaluate_feature(
                                        df=df,
                                        feature_cols=st.session_state.feature_cols,
                                        target_col=st.session_state.target_col,
                                        task_type=st.session_state.task_type,
                                        feature_name=s["name"],
                                        feature_code=s["code"],
                                        cv_folds=st.session_state.cv_folds,
                                        random_seed=st.session_state.random_seed,
                                    )
                                    st.session_state.eval_results = result
                                    if result.get("success"):
                                        entry = engine.build_leaderboard_entry(
                                            s["name"],
                                            result,
                                            st.session_state.baseline_score,
                                            st.session_state.task_type,
                                        )
                                        st.session_state.leaderboard = [
                                            e for e in st.session_state.leaderboard
                                            if e["run"] != s["name"]
                                        ]
                                        st.session_state.leaderboard.append(entry)
                                        _log(
                                            f"Suggestion '{s['name']}' — "
                                            f"{result['primary_metric']}="
                                            f"{result['primary_score']:.5f}"
                                        )
                                        score = result["primary_score"]
                                        delta = score - (st.session_state.baseline_score or 0)
                                        sign = "+" if delta >= 0 else ""
                                        st.success(
                                            f"✓ {result['primary_metric'].upper()} = "
                                            f"{score:.5f} ({sign}{delta:.5f} vs baseline)"
                                        )
                                    else:
                                        st.error(
                                            f"Failed: {result.get('error', '')[:200]}"
                                        )
                                except Exception as exc:
                                    st.error(f"Error: {exc}")
                        else:
                            st.info("Baseline not established yet. Result queued in editor.")
                            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — LEADERBOARD
# ─────────────────────────────────────────────────────────────────────────────

def render_leaderboard() -> None:
    st.markdown('<div class="section-header">Experiment Leaderboard</div>', unsafe_allow_html=True)

    leaderboard = st.session_state.leaderboard
    if not leaderboard:
        st.info("No runs recorded yet. Run a baseline and evaluate features to populate this board.")
        return

    # ── Sort by primary score ─────────────────────────────────────────────────
    task_type = st.session_state.task_type or "classification"
    sorted_lb = sorted(leaderboard, key=lambda x: x["score"], reverse=True)

    # ── Summary cards ─────────────────────────────────────────────────────────
    best = sorted_lb[0]
    baseline_entry = next((e for e in leaderboard if e["run"] == "Baseline"), None)
    n_experiments = sum(1 for e in leaderboard if e["run"] != "Baseline")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Best Score", f"{best['score']:.5f}", delta=best["run"])
    with c2:
        st.metric("Metric", best["primary_metric"].upper())
    with c3:
        st.metric("Experiments Run", n_experiments)
    with c4:
        if baseline_entry:
            st.metric("Baseline Score", f"{baseline_entry['score']:.5f}")
        else:
            st.metric("Baseline Score", "—")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Full leaderboard table ────────────────────────────────────────────────
    display_cols = ["run", "score", "delta", "n_features", "cv_folds", "timestamp"]
    # Add mean metric columns that exist
    extra_mean = [k for k in sorted_lb[0].keys() if k.startswith("mean_")]
    display_cols += extra_mean

    lb_df = pd.DataFrame(sorted_lb)

    # Only show columns that exist in the df
    available_cols = [c for c in display_cols if c in lb_df.columns]
    lb_display = lb_df[available_cols].copy()

    # Colour styling: best row highlighted
    def highlight_best(row):
        if row.name == 0:  # first row after sort = best
            return ["background-color: #1a2e1a; color: #6ee7b7"] * len(row)
        if row.get("run", "") == "Baseline":
            return ["background-color: #1a1e2e; color: #93c5fd"] * len(row)
        return [""] * len(row)

    try:
        styled = lb_display.style.apply(highlight_best, axis=1)
        st.dataframe(styled, use_container_width=True, height=min(600, 40 + 40 * len(lb_display)))
    except Exception:
        st.dataframe(lb_display, use_container_width=True)

    # ── Delta bar chart ───────────────────────────────────────────────────────
    delta_entries = [e for e in sorted_lb if e.get("delta") is not None and e["run"] != "Baseline"]
    if delta_entries:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">Score Delta vs Baseline</div>', unsafe_allow_html=True)
        delta_df = pd.DataFrame(delta_entries)[["run", "delta"]].set_index("run")
        st.bar_chart(delta_df, height=300)

    # ── Export ────────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    csv_bytes = lb_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Export Leaderboard CSV",
        data=csv_bytes,
        file_name="leaderboard.csv",
        mime="text/csv",
    )

    # ── Activity log ─────────────────────────────────────────────────────────
    if st.session_state.eval_log:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">Activity Log</div>', unsafe_allow_html=True)
        log_text = "\n".join(reversed(st.session_state.eval_log))
        st.markdown(
            f'<div class="log-box">{log_text}</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    render_sidebar()

    # ── Hero header ───────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="hero">
          <p class="hero-title">Feature <span>Engineering</span> Tool</p>
          <p class="hero-sub">
            Upload data · establish a CV baseline · engineer custom features · 
            track every experiment on the leaderboard.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Data Overview",
        "Baseline",
        "Feature Engineering",
        "Suggestions",
        "Leaderboard",
    ])

    with tab1:
        render_data_overview()

    with tab2:
        render_baseline()

    with tab3:
        render_feature_engineering()

    with tab4:
        render_suggestions()

    with tab5:
        render_leaderboard()


if __name__ == "__main__":
    main()

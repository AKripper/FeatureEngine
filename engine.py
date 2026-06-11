"""
engine.py — All ML logic, feature engineering, and rule-based suggestion engine.
No Streamlit imports. No UI logic. Pure Python + scikit-learn + pandas.
"""

from __future__ import annotations

import traceback
import warnings
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

NUMERIC_DTYPES = ["int8", "int16", "int32", "int64",
                  "float16", "float32", "float64"]

MAX_CARDINALITY_FOR_ONEHOT = 15   # columns above this are label-encoded
HIGH_CARDINALITY_THRESHOLD = 50   # columns above this are flagged


# ─────────────────────────────────────────────────────────────────────────────
# DATA PROFILING
# ─────────────────────────────────────────────────────────────────────────────

def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """
    Return a lightweight schema profile used by the suggestion engine and UI.
    """
    numeric_cols = df.select_dtypes(include=NUMERIC_DTYPES).columns.tolist()
    categorical_cols = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)

    cardinality = {col: df[col].nunique() for col in df.columns}

    skewness: dict[str, float] = {}
    for col in numeric_cols:
        try:
            skewness[col] = float(df[col].skew())
        except Exception:
            skewness[col] = 0.0

    correlations: dict[str, float] = {}
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr().abs()
        for i, c1 in enumerate(numeric_cols):
            for c2 in numeric_cols[i + 1:]:
                correlations[f"{c1}___{c2}"] = float(corr_matrix.loc[c1, c2])

    return {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "datetime_cols": datetime_cols,
        "missing_pct": missing_pct.to_dict(),
        "cardinality": cardinality,
        "skewness": skewness,
        "correlations": correlations,
    }


def infer_task_type(series: pd.Series) -> str:
    """
    Heuristic: <=20 unique values OR dtype is object/bool → classification,
    otherwise regression.
    """
    if series.dtype in ["object", "bool", "category"]:
        return "classification"
    if series.nunique() <= 20:
        return "classification"
    return "regression"


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _encode_dataframe(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Minimal in-place encoding so HistGradientBoosting can accept the data.
    - Numeric: kept as-is (HGBT handles NaN natively)
    - Categorical / object / bool: LabelEncode
    Returns (X_enc, y_enc).
    """
    X_enc = X.copy()

    for col in X_enc.select_dtypes(include=["object", "category", "bool"]).columns:
        le = LabelEncoder()
        X_enc[col] = le.fit_transform(X_enc[col].astype(str))

    # Drop remaining non-numeric columns (e.g. datetime that wasn't parsed)
    non_numeric = X_enc.select_dtypes(
        exclude=NUMERIC_DTYPES + ["int8", "int16", "int32", "int64",
                                  "float16", "float32", "float64"]
    ).columns.tolist()
    X_enc.drop(columns=non_numeric, inplace=True, errors="ignore")

    y_enc = y.copy()
    if task_type == "classification" and y_enc.dtype in ["object", "bool", "category"]:
        le_y = LabelEncoder()
        y_enc = pd.Series(
            le_y.fit_transform(y_enc.astype(str)), index=y_enc.index, name=y_enc.name
        )

    return X_enc, y_enc


def _build_model(task_type: str, random_seed: int):
    """Return a lightweight HGBT model."""
    params = dict(
        max_iter=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=random_seed,
        early_stopping=False,
    )
    if task_type == "classification":
        return HistGradientBoostingClassifier(**params)
    return HistGradientBoostingRegressor(**params)


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str,
    y_prob: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute a dict of evaluation metrics."""
    metrics: dict[str, float] = {}
    if task_type == "classification":
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        avg = "binary" if len(np.unique(y_true)) == 2 else "weighted"
        metrics["f1"] = float(f1_score(y_true, y_pred, average=avg, zero_division=0))
        if y_prob is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    metrics["roc_auc"] = float(
                        roc_auc_score(y_true, y_prob[:, 1])
                    )
                else:
                    metrics["roc_auc"] = float(
                        roc_auc_score(
                            y_true, y_prob, multi_class="ovr", average="weighted"
                        )
                    )
            except Exception:
                pass
    else:
        metrics["r2"] = float(r2_score(y_true, y_pred))
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# CROSS-VALIDATION BASELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_baseline(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    task_type: str,
    cv_folds: int = 5,
    random_seed: int = 42,
) -> dict[str, Any]:
    """
    Run stratified (classification) or regular (regression) K-fold CV.
    Returns a result dict with per-fold metrics, mean metrics, and metadata.
    """
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    X_enc, y_enc = _encode_dataframe(X, y, task_type)
    X_arr = X_enc.values
    y_arr = y_enc.values

    model = _build_model(task_type, random_seed)

    if task_type == "classification":
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    else:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)

    fold_metrics: list[dict[str, float]] = []
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_arr, y_arr)):
        X_train, X_val = X_arr[train_idx], X_arr[val_idx]
        y_train, y_val = y_arr[train_idx], y_arr[val_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_prob = None
        if task_type == "classification":
            try:
                y_prob = model.predict_proba(X_val)
            except Exception:
                pass

        m = _compute_metrics(y_val, y_pred, task_type, y_prob)
        m["fold"] = fold_idx + 1
        fold_metrics.append(m)

    # Mean & std across folds
    all_keys = [k for k in fold_metrics[0] if k != "fold"]
    mean_metrics = {k: float(np.mean([f[k] for f in fold_metrics])) for k in all_keys}
    std_metrics = {k: float(np.std([f[k] for f in fold_metrics])) for k in all_keys}

    primary_metric = _primary_metric(task_type)

    return {
        "fold_metrics": fold_metrics,
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
        "primary_metric": primary_metric,
        "primary_score": mean_metrics[primary_metric],
        "task_type": task_type,
        "cv_folds": cv_folds,
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    }


def _primary_metric(task_type: str) -> str:
    return "roc_auc" if task_type == "classification" else "r2"


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM FEATURE EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_feature(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    task_type: str,
    feature_name: str,
    feature_code: str,
    cv_folds: int = 5,
    random_seed: int = 42,
) -> dict[str, Any]:
    """
    Evaluate a user-defined feature expression.

    feature_code is a Python expression that may reference `df` (the full
    DataFrame). The result must be a Series or 1-D array of the same length.

    Returns a result dict with metrics, delta vs baseline stored in the
    leaderboard entry format.
    """
    # ── Execute the expression in a restricted namespace ──────────────────────
    local_ns: dict[str, Any] = {
        "df": df.copy(),
        "pd": pd,
        "np": np,
    }
    try:
        exec(f"__result__ = {feature_code}", local_ns)  # noqa: S102
        new_feature: Any = local_ns["__result__"]
    except Exception:
        return {
            "success": False,
            "error": traceback.format_exc(),
        }

    # ── Validate shape ─────────────────────────────────────────────────────────
    try:
        new_series = pd.Series(
            np.array(new_feature).flatten(), index=df.index, name=feature_name
        )
        if len(new_series) != len(df):
            return {
                "success": False,
                "error": (
                    f"Feature length mismatch: got {len(new_series)}, "
                    f"expected {len(df)}."
                ),
            }
    except Exception:
        return {
            "success": False,
            "error": traceback.format_exc(),
        }

    # ── Build augmented DataFrame ──────────────────────────────────────────────
    df_aug = df[feature_cols].copy()
    df_aug[feature_name] = new_series.values
    augmented_cols = feature_cols + [feature_name]

    # ── Run CV ────────────────────────────────────────────────────────────────
    try:
        result = run_baseline(
            df=df.assign(**{feature_name: new_series.values}),
            feature_cols=augmented_cols,
            target_col=target_col,
            task_type=task_type,
            cv_folds=cv_folds,
            random_seed=random_seed,
        )
    except Exception:
        return {
            "success": False,
            "error": traceback.format_exc(),
        }

    result["success"] = True
    result["feature_name"] = feature_name
    result["feature_code"] = feature_code
    return result


# ─────────────────────────────────────────────────────────────────────────────
# LEADERBOARD
# ─────────────────────────────────────────────────────────────────────────────

def build_leaderboard_entry(
    run_name: str,
    result: dict[str, Any],
    baseline_score: float | None,
    task_type: str,
) -> dict[str, Any]:
    """
    Create a single leaderboard row dict from a CV result.
    """
    primary = result["primary_metric"]
    score = result["primary_score"]
    delta: float | None = None
    if baseline_score is not None:
        delta = score - baseline_score

    mean_m = result["mean_metrics"]
    std_m = result["std_metrics"]

    entry: dict[str, Any] = {
        "run": run_name,
        "primary_metric": primary,
        "score": round(score, 5),
        "delta": round(delta, 5) if delta is not None else None,
        "n_features": result["n_features"],
        "cv_folds": result["cv_folds"],
        "timestamp": result["timestamp"],
    }

    # Include all secondary metrics
    for k, v in mean_m.items():
        entry[f"mean_{k}"] = round(v, 5)
    for k, v in std_m.items():
        entry[f"std_{k}"] = round(v, 5)

    return entry


# ─────────────────────────────────────────────────────────────────────────────
# RULE-BASED SUGGESTION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def generate_suggestions(
    profile: dict[str, Any],
    target_col: str,
    task_type: str,
    feature_cols: list[str],
) -> list[dict[str, str]]:
    """
    Algorithmic rule-based suggestion engine.
    Analyzes the schema profile and returns a list of suggestion dicts:
        {category, name, description, code}
    """
    suggestions: list[dict[str, str]] = []
    num_cols = [c for c in profile["numeric_cols"] if c in feature_cols]
    cat_cols = [c for c in profile["categorical_cols"] if c in feature_cols]

    # ── 1. Log-transform highly skewed numerics ───────────────────────────────
    for col in num_cols:
        skew = profile["skewness"].get(col, 0.0)
        if abs(skew) > 1.5:
            suggestions.append({
                "category": "Transformation",
                "name": f"log1p_{col}",
                "description": (
                    f"Log-transform '{col}' (skew={skew:.2f}) to reduce right-skew "
                    "and stabilise variance."
                ),
                "code": f"np.log1p(df['{col}'].clip(lower=0))",
            })

    # ── 2. Square root for moderately skewed ─────────────────────────────────
    for col in num_cols:
        skew = profile["skewness"].get(col, 0.0)
        if 0.8 < abs(skew) <= 1.5:
            suggestions.append({
                "category": "Transformation",
                "name": f"sqrt_{col}",
                "description": (
                    f"Square-root transform '{col}' (skew={skew:.2f}) for moderate skew."
                ),
                "code": f"np.sqrt(df['{col}'].clip(lower=0))",
            })

    # ── 3. Polynomial / squared terms ────────────────────────────────────────
    for col in num_cols[:5]:   # limit to first 5 to keep list readable
        suggestions.append({
            "category": "Polynomial",
            "name": f"{col}_squared",
            "description": f"Squared term for '{col}' to capture non-linear effects.",
            "code": f"df['{col}'] ** 2",
        })

    # ── 4. Pairwise interactions for highly correlated column pairs ───────────
    high_corr_pairs = [
        (pair.split("___")[0], pair.split("___")[1])
        for pair, corr in profile["correlations"].items()
        if corr >= 0.4
    ]
    for c1, c2 in high_corr_pairs[:6]:
        if c1 in feature_cols and c2 in feature_cols:
            # product
            suggestions.append({
                "category": "Interaction",
                "name": f"{c1}_x_{c2}",
                "description": (
                    f"Product of '{c1}' and '{c2}' "
                    f"(|corr|={profile['correlations'].get(c1 + '___' + c2, 0):.2f})."
                ),
                "code": f"df['{c1}'] * df['{c2}']",
            })
            # ratio (guarded)
            suggestions.append({
                "category": "Interaction",
                "name": f"{c1}_div_{c2}",
                "description": (
                    f"Ratio '{c1}' / '{c2}' — useful when one scales the other."
                ),
                "code": f"df['{c1}'] / (df['{c2}'] + 1e-9)",
            })
            # difference
            suggestions.append({
                "category": "Interaction",
                "name": f"{c1}_minus_{c2}",
                "description": f"Difference '{c1}' − '{c2}'.",
                "code": f"df['{c1}'] - df['{c2}']",
            })

    # ── 5. Pairwise sum / diff for low-corr numeric pairs ────────────────────
    low_corr_pairs = [
        (pair.split("___")[0], pair.split("___")[1])
        for pair, corr in profile["correlations"].items()
        if corr < 0.2
    ]
    for c1, c2 in low_corr_pairs[:4]:
        if c1 in feature_cols and c2 in feature_cols:
            suggestions.append({
                "category": "Interaction",
                "name": f"{c1}_plus_{c2}",
                "description": (
                    f"Sum of weakly-correlated '{c1}' and '{c2}' "
                    "— orthogonal additive signal."
                ),
                "code": f"df['{c1}'] + df['{c2}']",
            })

    # ── 6. Z-score / deviation ────────────────────────────────────────────────
    for col in num_cols[:4]:
        suggestions.append({
            "category": "Normalization",
            "name": f"{col}_zscore",
            "description": f"Z-score of '{col}' (subtract mean, divide by std).",
            "code": (
                f"(df['{col}'] - df['{col}'].mean()) / "
                f"(df['{col}'].std() + 1e-9)"
            ),
        })

    # ── 7. Binning numeric columns ────────────────────────────────────────────
    for col in num_cols[:3]:
        suggestions.append({
            "category": "Binning",
            "name": f"{col}_bin5",
            "description": (
                f"Quantile-bin '{col}' into 5 equal-frequency buckets."
            ),
            "code": f"pd.qcut(df['{col}'], 5, labels=False, duplicates='drop').astype(float)",
        })

    # ── 8. Missing-value indicator flags ─────────────────────────────────────
    high_missing = [
        col for col, pct in profile["missing_pct"].items()
        if pct > 5 and col in feature_cols
    ]
    for col in high_missing[:4]:
        suggestions.append({
            "category": "Missing Indicator",
            "name": f"{col}_is_missing",
            "description": (
                f"Binary flag: 1 if '{col}' is NaN ({profile['missing_pct'][col]:.1f}% "
                "missing). Missing-ness itself may carry signal."
            ),
            "code": f"df['{col}'].isnull().astype(int)",
        })

    # ── 9. Categorical frequency encoding ────────────────────────────────────
    for col in cat_cols[:4]:
        suggestions.append({
            "category": "Encoding",
            "name": f"{col}_freq_enc",
            "description": (
                f"Frequency-encode '{col}': replace each category with its "
                "relative frequency in the dataset."
            ),
            "code": (
                f"df['{col}'].map(df['{col}'].value_counts(normalize=True))"
            ),
        })

    # ── 10. High-cardinality flag ─────────────────────────────────────────────
    high_card = [
        col for col in cat_cols
        if profile["cardinality"].get(col, 0) > HIGH_CARDINALITY_THRESHOLD
        and col in feature_cols
    ]
    for col in high_card[:3]:
        suggestions.append({
            "category": "Encoding",
            "name": f"{col}_top10_flag",
            "description": (
                f"Flag whether '{col}' belongs to the top-10 most frequent "
                "categories (reduces noise from rare values)."
            ),
            "code": (
                f"df['{col}'].isin("
                f"df['{col}'].value_counts().head(10).index"
                f").astype(int)"
            ),
        })

    # ── 11. Aggregate stats per group (cat × numeric) ────────────────────────
    for cat in cat_cols[:2]:
        for num in num_cols[:2]:
            if cat in feature_cols and num in feature_cols:
                suggestions.append({
                    "category": "Group Aggregate",
                    "name": f"{num}_mean_by_{cat}",
                    "description": (
                        f"Mean of '{num}' grouped by '{cat}' — target-free "
                        "aggregation feature."
                    ),
                    "code": (
                        f"df['{cat}'].map(df.groupby('{cat}')['{num}'].mean())"
                    ),
                })
                suggestions.append({
                    "category": "Group Aggregate",
                    "name": f"{num}_std_by_{cat}",
                    "description": (
                        f"Std-dev of '{num}' grouped by '{cat}'."
                    ),
                    "code": (
                        f"df['{cat}'].map(df.groupby('{cat}')['{num}'].std().fillna(0))"
                    ),
                })

    # ── 12. Abs value for potentially-signed numeric columns ──────────────────
    for col in num_cols[:3]:
        try:
            if num_cols and float(pd.Series(profile["skewness"]).abs().max()) > 0:
                suggestions.append({
                    "category": "Transformation",
                    "name": f"abs_{col}",
                    "description": (
                        f"Absolute value of '{col}' — useful if sign is not "
                        "informative but magnitude is."
                    ),
                    "code": f"df['{col}'].abs()",
                })
        except Exception:
            pass

    # Deduplicate by name
    seen: set[str] = set()
    unique: list[dict[str, str]] = []
    for s in suggestions:
        if s["name"] not in seen:
            seen.add(s["name"])
            unique.append(s)

    return unique


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────


def compute_feature_importance(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    task_type: str,
    random_seed: int = 42,
) -> pd.DataFrame:

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    X_enc, y_enc = _encode_dataframe(X, y, task_type)

    model = _build_model(task_type, random_seed)
    model.fit(X_enc.values, y_enc.values)

    result = permutation_importance(
        model,
        X_enc.values,
        y_enc.values,
        n_repeats=5,
        random_state=random_seed,
    )

    imp = pd.DataFrame(
        {
            "feature": X_enc.columns.tolist(),
            "importance": result.importances_mean,
        }
    ).sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)

    return imp




# ─────────────────────────────────────────────────────────────────────────────
# QUICK DATA STATS (for the Data Overview tab)
# ─────────────────────────────────────────────────────────────────────────────

def compute_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extended describe() including dtype, missing %, and cardinality.
    """
    rows = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        missing = int(df[col].isnull().sum())
        missing_pct = round(missing / len(df) * 100, 2)
        cardinality = int(df[col].nunique())
        row: dict[str, Any] = {
            "column": col,
            "dtype": dtype,
            "missing": missing,
            "missing_%": missing_pct,
            "cardinality": cardinality,
        }
        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32,
                              np.int16, np.int8, np.float16]:
            row["mean"] = round(float(df[col].mean()), 4)
            row["std"] = round(float(df[col].std()), 4)
            row["min"] = round(float(df[col].min()), 4)
            row["max"] = round(float(df[col].max()), 4)
            row["skew"] = round(float(df[col].skew()), 4)
        else:
            row["mean"] = row["std"] = row["min"] = row["max"] = row["skew"] = "—"
        rows.append(row)
    return pd.DataFrame(rows)

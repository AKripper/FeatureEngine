# ⚙️ Feature Engineering Tool

A modular, multi-file Streamlit dashboard for rapid, experiment-driven feature engineering.

## Project Structure

```
feature_engineering_tool/
<<<<<<< HEAD
├── app.py                      ← Streamlit UI (tabs, layout, widgets)
├── engine.py                   ← All ML & business logic (no Streamlit imports)
├── load_sample_datasets.py     ← Built-in sample datasets for quick experimentation
├── state.py                    ← Session-state initialisation & reset helpers
├── style.py                    ← Centralized UI styling and custom CSS definitions
├── requirements.txt            ← Python dependencies
=======
├── app.py            ← Streamlit UI (tabs, layout, widgets)
├── engine.py         ← All ML & business logic (no Streamlit imports)
├── state.py          ← Session-state initialisation & reset helpers
├── requirements.txt  ← Python dependencies
>>>>>>> 5ebcbc9c35363a302ffdbd63fc0dacf8f664a4fc
└── README.md
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the app
streamlit run app.py
```

## Workflow

| Step | Tab | What happens |
|------|-----|--------------|
| 1 | **Data Overview** | Upload CSV / Parquet / Excel, inspect column stats, distribution, and correlation matrix |
| 2 | **Baseline** | Run stratified K-fold CV with `HistGradientBoosting`; view per-fold metrics and feature importance |
| 3 | **Feature Engineering** | Write a Python expression referencing `df`, `np`, `pd`; evaluate it against the baseline in one click |
| 4 | **Suggestions** | Auto-generated rule-based ideas (log-transforms, interactions, binning, encoding…); try any with one click |
| 5 | **Leaderboard** | Every run tracked with Δ vs baseline; export to CSV |

## Design Principles

- **`engine.py`** — zero Streamlit imports. Pure ML + pandas logic with full error handling.
- **`state.py`** — single source of truth for `st.session_state` keys.
- **`app.py`** — renders UI, delegates all computation to `engine.py`.
- **No external APIs** — the suggestion engine is entirely rule-based.
- **Fast** — `HistGradientBoostingClassifier/Regressor` handles NaN natively; baseline + eval complete in < 3 s on typical tabular datasets.

## Supported File Formats

- CSV (`.csv`)
- Apache Parquet (`.parquet`)
- Excel (`.xls`, `.xlsx`)

## Feature Expression Examples

```python
# Log-transform a skewed column
np.log1p(df['price'])

# Interaction term
df['length'] * df['width']

# Frequency encoding
df['city'].map(df['city'].value_counts(normalize=True))

# Group aggregate
df['category'].map(df.groupby('category')['sales'].mean())

# Binary missing flag
df['age'].isnull().astype(int)

# Quantile binning
pd.qcut(df['income'], 5, labels=False, duplicates='drop').astype(float)
```

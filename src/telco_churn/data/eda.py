"""EDA helper functions for the Telco Churn dataset.

All functions accept a DataFrame with the normalised column names produced by
``ingest.py`` (lowercase, ``has_partner``, ``contract_type``, ``churn`` as
0/1 integer).  They return tidy DataFrames or Series so the EDA notebook stays
a thin rendering wrapper with no duplicated logic.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------------------------
# Column-name constants (single source of truth for notebook callers)
# ---------------------------------------------------------------------------

CAT_FEATURES: list[str] = [
    "gender",
    "has_partner",
    "dependents",
    "phoneservice",
    "multiplelines",
    "internetservice",
    "onlinesecurity",
    "onlinebackup",
    "deviceprotection",
    "techsupport",
    "streamingtv",
    "streamingmovies",
    "contract_type",
    "paperlessbilling",
    "paymentmethod",
]

NUM_FEATURES: list[str] = ["tenure", "monthlycharges", "totalcharges"]

# Binary integer columns (0/1 SMALLINT in the DB) that are categorically meaningful
# but must NOT be passed to pd.get_dummies — they are already numeric and need no encoding.
# Chi-squared tests (pd.crosstab) handle integer values correctly.
BINARY_INT_FEATURES: list[str] = ["seniorcitizen"]

TARGET: str = "churn"


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def detect_outliers(
    df: pd.DataFrame,
    num_features: list[str] = NUM_FEATURES,
) -> pd.DataFrame:
    """IQR-based outlier summary for numeric features.

    Flags values below Q1 − 1.5 × IQR or above Q3 + 1.5 × IQR.

    Args:
        df: DataFrame containing numeric columns.
        num_features: Columns to analyse; defaults to ``NUM_FEATURES``.

    Returns:
        DataFrame indexed by feature with columns: Q1, Q3, IQR,
        lower_bound, upper_bound, n_outliers, pct_outliers.
        ``pct_outliers`` is the percentage (0–100) of non-null observations
        that fall outside the IQR bounds (denominator excludes NaNs).
    """
    rows = []
    for col in num_features:
        series = df[col].dropna()
        if len(series) == 0:
            warnings.warn(
                f"'{col}' has no observations after dropping nulls — outlier stats undefined, skipping.",
                stacklevel=2,
            )
            rows.append(
                {
                    "feature": col,
                    "Q1": float("nan"),
                    "Q3": float("nan"),
                    "IQR": float("nan"),
                    "lower_bound": float("nan"),
                    "upper_bound": float("nan"),
                    "n_outliers": 0,
                    "pct_outliers": 0.0,
                }
            )
            continue
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        n_out = int(((series < lo) | (series > hi)).sum())
        rows.append(
            {
                "feature": col,
                "Q1": round(q1, 2),
                "Q3": round(q3, 2),
                "IQR": round(iqr, 2),
                "lower_bound": round(lo, 2),
                "upper_bound": round(hi, 2),
                "n_outliers": n_out,
                "pct_outliers": round(n_out / len(series) * 100, 1),
            }
        )
    return pd.DataFrame(rows).set_index("feature")


def churn_rate_by_group(df: pd.DataFrame, col: str) -> pd.Series[float]:
    """Return churn rate (0–1) for each value of *col*, sorted descending.

    Args:
        df: DataFrame containing *col* and ``churn`` (0/1).
        col: Column name to group by; accepts any type pandas ``groupby``
            handles, including integer-coded columns (e.g. ``seniorcitizen``
            0/1) without pre-encoding.

    Returns:
        Series indexed by the distinct values of *col*. Values are proportions
        in [0, 1] — multiply by 100 only at the display layer. NaN-valued rows
        are excluded from all groups; a warning is emitted when any are present.
    """
    n_null = int(df[col].isna().sum())
    if n_null > 0:
        warnings.warn(
            f"'{col}' has {n_null} NaN value(s) — excluded from group churn rates.",
            stacklevel=2,
        )
    return df.groupby(col, dropna=True)[TARGET].mean().sort_values(ascending=False)


def compute_chi2_tests(
    df: pd.DataFrame,
    cat_features: list[str] = CAT_FEATURES + BINARY_INT_FEATURES,
    target: str = TARGET,
) -> pd.DataFrame:
    """Chi-squared test + Cramér's V for categorical features vs *target*.

    Cramér's V uses ``k = min(rows-1, cols-1)`` to correct for category count.
    ``pd.crosstab`` handles integer columns (e.g. ``seniorcitizen`` 0/1) without
    pre-encoding.

    Args:
        df: DataFrame with categorical columns and a binary *target*.
        cat_features: Columns to test; defaults to ``CAT_FEATURES + BINARY_INT_FEATURES``.
        target: Binary target column name.

    Returns:
        DataFrame with columns: feature, cramers_v, chi2, p_value, dof.
        Sorted descending by Cramér's V.
    """
    rows = []
    for col in cat_features:
        n_null = int(df[col].isna().sum())
        if n_null > 0:
            warnings.warn(
                f"'{col}' has {n_null} NaN value(s) — excluded from chi-squared test.",
                stacklevel=2,
            )
        ct = pd.crosstab(df[col], df[target])
        if ct.values.sum() == 0:
            warnings.warn(
                f"'{col}' has no observations — chi-squared undefined, skipping.",
                stacklevel=2,
            )
            continue
        chi2_raw, p_raw, dof_raw, expected = chi2_contingency(ct)
        chi2_stat = float(chi2_raw)
        p_val = float(p_raw)
        dof_val = int(dof_raw)
        n_sparse = int((expected < 5).sum())
        if n_sparse > 0:
            warnings.warn(
                f"'{col}' has {n_sparse} cell(s) with expected count < 5 — "
                "chi-squared results may be unreliable.",
                stacklevel=2,
            )
        n = int(ct.values.sum())
        k = int(min(ct.shape)) - 1
        if k == 0:
            warnings.warn(
                f"'{col}' is constant — chi-squared undefined, skipping.", stacklevel=2
            )
            continue
        v = float(np.sqrt(chi2_stat / (n * k)))
        rows.append(
            {
                "feature": col,
                "cramers_v": round(v, 4),
                "chi2": round(chi2_stat, 2),
                "p_value": p_val,
                "dof": dof_val,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["feature", "cramers_v", "chi2", "p_value", "dof"])
    return (
        pd.DataFrame(rows)
        .sort_values("cramers_v", ascending=False)
        .reset_index(drop=True)
    )


def compute_mann_whitney(
    df: pd.DataFrame,
    num_features: list[str] = NUM_FEATURES,
    target: str = TARGET,
) -> pd.DataFrame:
    """Mann-Whitney U test + rank-biserial r for numeric features vs *target*.

    Rank-biserial correlation is
    ``r = 2U / (n₁ · n₀) − 1`` and quantifies the effect size
    (positive ⇒ churners score *higher* on that feature; negative ⇒ churners score *lower*).

    Args:
        df: DataFrame with numeric columns and a binary *target*.
        num_features: Columns to test; defaults to ``NUM_FEATURES``.
        target: Binary target column name (0/1).

    Returns:
        DataFrame with columns: feature, rank_biserial_r, U_stat, p_value,
        mean_churners, mean_non_churners.  Sorted by |rank_biserial_r| desc.
    """
    rows = []
    for col in num_features:
        g1 = df.loc[df[target] == 1, col].dropna()
        g0 = df.loc[df[target] == 0, col].dropna()
        if len(g1) == 0 or len(g0) == 0:
            warnings.warn(
                f"'{col}' has an empty group after dropping nulls — Mann-Whitney undefined, skipping.",
                stacklevel=2,
            )
            continue
        result = mannwhitneyu(g1, g0, alternative="two-sided")
        U = float(result.statistic)
        p = float(result.pvalue)
        r = (2.0 * U) / (len(g1) * len(g0)) - 1.0
        rows.append(
            {
                "feature": col,
                "rank_biserial_r": round(r, 4),
                "U_stat": round(U, 0),
                "p_value": p,
                "mean_churners": round(float(g1.mean()), 2),
                "mean_non_churners": round(float(g0.mean()), 2),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "feature",
                "rank_biserial_r",
                "U_stat",
                "p_value",
                "mean_churners",
                "mean_non_churners",
            ]
        )
    return (
        pd.DataFrame(rows)
        .sort_values("rank_biserial_r", key=abs, ascending=False)
        .reset_index(drop=True)
    )


def compute_vif(
    df: pd.DataFrame,
    num_cols: list[str] = NUM_FEATURES + BINARY_INT_FEATURES,
    cat_cols: list[str] = CAT_FEATURES,
) -> pd.DataFrame:
    """Variance Inflation Factor table for multicollinearity detection.

    Uses ``drop_first=True`` one-hot
    encoding to avoid the dummy-variable trap (perfect multicollinearity
    that would make VIF undefined).  VIF is computed as ``1 / (1 − R²)``
    where R² comes from regressing each feature on all others via sklearn
    ``LinearRegression`` (avoids a statsmodels dependency).

    Args:
        df: DataFrame with numeric and categorical columns (no target column).
        num_cols: Numeric columns to include; defaults to ``NUM_FEATURES +
            BINARY_INT_FEATURES`` so that ``seniorcitizen`` (already 0/1) is
            passed through without encoding.
        cat_cols: Categorical columns to one-hot encode; defaults to
            ``CAT_FEATURES``. Avoid high-cardinality columns here — one-hot
            expansion raises the encoded feature count and the loop runs one
            regression per feature, so cost scales as O(p²).

    Returns:
        DataFrame with columns: feature, VIF. Sorted descending by VIF.
    """
    present_num = [c for c in num_cols if c in df.columns]
    present_cat = [c for c in cat_cols if c in df.columns]

    df_vif = pd.get_dummies(
        df[present_num + present_cat], columns=present_cat, drop_first=True
    ).apply(pd.to_numeric, errors="coerce")
    n_before = len(df_vif)
    df_vif = df_vif.dropna().astype(np.float64)
    n_dropped = n_before - len(df_vif)
    if n_dropped > 0:
        warnings.warn(
            f"{n_dropped} row(s) with NaN dropped — VIF computed on {len(df_vif)} observations.",
            stacklevel=2,
        )

    if df_vif.shape[0] == 0:
        warnings.warn(
            "DataFrame has no observations after encoding and dropping nulls — VIF undefined.",
            stacklevel=2,
        )
        return pd.DataFrame(columns=["feature", "VIF"])

    X = df_vif.to_numpy(dtype=np.float64)
    vif_vals: list[float] = []
    for i in range(X.shape[1]):
        y = X[:, i]
        x_rest = np.delete(X, i, axis=1)
        r2 = float(LinearRegression().fit(x_rest, y).score(x_rest, y))
        vif_vals.append(round(1.0 / (1.0 - r2) if r2 < 1.0 else float("inf"), 2))

    return (
        pd.DataFrame({"feature": df_vif.columns.tolist(), "VIF": vif_vals})
        .sort_values("VIF", ascending=False)
        .reset_index(drop=True)
    )


def encoded_correlation_matrix(
    df: pd.DataFrame,
    cat_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Full Pearson correlation matrix after one-hot encoding all categoricals.

    Uses ``drop_first=False`` so every category appears in the heatmap.

    Args:
        df: DataFrame including categorical columns and ``churn`` (0/1).
        cat_cols: Columns to one-hot encode; defaults to ``CAT_FEATURES``.

    Returns:
        Square correlation DataFrame over all encoded features.
    """
    cats = cat_cols if cat_cols is not None else CAT_FEATURES
    present_cat = [c for c in cats if c in df.columns]
    df_enc = pd.get_dummies(
        df.drop(columns=["customerid"], errors="ignore"),
        columns=present_cat,
        drop_first=False,
    )
    return df_enc.apply(pd.to_numeric, errors="coerce").astype(float).corr()


def correlation_with_target(
    corr_matrix: pd.DataFrame,
    target: str = TARGET,
    top_n: int = 20,
) -> pd.Series[float]:
    """Top-N Pearson correlations of encoded features with *target*.

    Args:
        corr_matrix: Precomputed Pearson correlation matrix, e.g. from
            ``encoded_correlation_matrix``.
        target: Column to correlate against.
        top_n: Number of top features to return (by absolute correlation).

    Returns:
        Series of Pearson r values, sorted by absolute value descending.
    """
    if target not in corr_matrix.columns:
        return pd.Series(dtype=float)
    col = corr_matrix[target].drop(target)
    if col.isna().all():
        return pd.Series(dtype=float)
    return col.sort_values(key=abs, ascending=False).head(top_n)

"""EDA helper functions for the Telco Churn dataset.

All functions accept a DataFrame with the normalised column names produced by
``ingest.py`` (lowercase, ``has_partner``, ``contract_type``, ``churn`` as
0/1 integer).  They return tidy DataFrames or Series so the EDA notebook stays
a thin rendering wrapper with no duplicated logic.
"""

from __future__ import annotations

import warnings
from typing import Final

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu

from telco_churn.utils.stats import cramers_v, pool_adjusted_p_values, vif_single

__all__ = [
    "CAT_FEATURES",
    "NUM_FEATURES",
    "BINARY_INT_FEATURES",
    "TARGET",
    "inspect_missing",
    "detect_outliers",
    "churn_rate_by_group",
    "compute_chi2_tests",
    "compute_mann_whitney",
    "compute_significance_screen",
    "compute_vif",
    "encoded_correlation_matrix",
    "correlation_with_target",
]

# ---------------------------------------------------------------------------
# Column-name constants (single source of truth for notebook callers)
# ---------------------------------------------------------------------------

CAT_FEATURES: Final[tuple[str, ...]] = (
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
)

NUM_FEATURES: Final[tuple[str, ...]] = ("tenure", "monthlycharges", "totalcharges")

# Binary integer columns (0/1 SMALLINT in the DB) that are categorically meaningful
# but must NOT be passed to pd.get_dummies — they are already numeric and need no encoding.
# Chi-squared tests (pd.crosstab) handle integer values correctly.
BINARY_INT_FEATURES: Final[tuple[str, ...]] = ("seniorcitizen",)

TARGET: Final[str] = "churn"


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def inspect_missing(
    df: pd.DataFrame,
    missing_cols: list[str] | None = None,
    context_cols: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Return missing-value rows for each column that has nulls.

    For each column with at least one null, extracts the subset of rows
    where that column is missing, retaining *context_cols* alongside it
    to support missingness-mechanism analysis (MCAR / MAR / MNAR).

    Args:
        df: Input DataFrame.
        missing_cols: Columns to inspect. Defaults to all columns in *df*
            that contain at least one null. Pass a pre-computed list (e.g.
            ``missing_summary.index.tolist()``) to avoid re-scanning.
        context_cols: Columns to display alongside the missing column.
            Defaults to ``customerid``, ``NUM_FEATURES``, and ``churn``
            when present — the same context used in the EDA notebook.

    Returns:
        Dict mapping column name → DataFrame of rows where that column is
        null.  Empty dict if no nulls are found.
    """
    cols_to_inspect = (
        missing_cols
        if missing_cols is not None
        else df.columns[df.isnull().any()].tolist()
    )
    if not cols_to_inspect:
        return {}

    if context_cols is None:
        defaults = ["customerid"] + list(NUM_FEATURES) + [TARGET]
        context_cols = [c for c in defaults if c in df.columns]

    result: dict[str, pd.DataFrame] = {}
    for col in cols_to_inspect:
        display_cols = [c for c in context_cols if c != col] + [col]
        result[col] = df.loc[df[col].isnull(), display_cols].reset_index(drop=True)
    return result


def detect_outliers(
    df: pd.DataFrame,
    num_features: list[str] | None = None,
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
    if num_features is None:
        num_features = list(NUM_FEATURES)
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
        if not pd.api.types.is_numeric_dtype(series):
            raise TypeError(
                f"unsupported operand type for IQR: '{col}' has dtype {series.dtype!r};"
                " expected a numeric column"
            )
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
    cat_features: list[str] | None = None,
    target: str = TARGET,
) -> pd.DataFrame:
    """Chi-squared test + Cramér's V for categorical features vs *target*.

    Cramér's V uses ``k = min(rows-1, cols-1)`` to correct for category count.
    ``pd.crosstab`` handles integer columns (e.g. ``seniorcitizen`` 0/1) without
    pre-encoding.

    This runs one test per feature — up to 16 by default — against the same
    target, so raw ``p_value`` alone is exploratory, not confirmatory: at the
    conventional α=0.05, a screen this size has a substantial chance of at
    least one false-positive "significant" result even if no feature has a
    real association. See ``compute_significance_screen``, which runs this
    alongside ``compute_mann_whitney`` and pools both into one
    Benjamini-Hochberg-corrected family — call that instead of this function
    directly when the correction matters, since neither p-value column gates
    anything downstream (``cramers_v`` is the effect-size read this function
    exists to produce; feature selection is driven by cross-validated
    permutation importance in ``features/select.py``).

    Args:
        df: DataFrame with categorical columns and a binary *target*.
        cat_features: Columns to test; defaults to ``CAT_FEATURES + BINARY_INT_FEATURES``.
        target: Binary target column name.

    Returns:
        DataFrame with columns: feature, cramers_v, chi2, p_value, dof.
        Sorted descending by Cramér's V.
    """
    if cat_features is None:
        cat_features = list(CAT_FEATURES + BINARY_INT_FEATURES)
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
        chi2_raw, p_raw, dof_raw, expected = chi2_contingency(ct, correction=False)
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
        if int(min(ct.shape)) - 1 == 0:
            warnings.warn(
                f"'{col}' is constant — chi-squared undefined, skipping.", stacklevel=2
            )
            continue
        v = cramers_v(df[col], df[target])
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
    num_features: list[str] | None = None,
    target: str = TARGET,
) -> pd.DataFrame:
    """Mann-Whitney U test + rank-biserial r for numeric features vs *target*.

    Rank-biserial correlation is
    ``r = 2U / (n₁ · n₀) − 1`` and quantifies the effect size
    (positive ⇒ churners score *higher* on that feature; negative ⇒ churners score *lower*).

    One test per feature against the same target — only 3 by default here,
    but see ``compute_chi2_tests``'s docstring for why raw ``p_value`` alone
    is exploratory rather than confirmatory at this scale, and
    ``compute_significance_screen`` for the Benjamini-Hochberg correction
    pooled jointly with ``compute_chi2_tests``'s features. ``rank_biserial_r``
    remains the effect-size read this function exists to produce.

    Args:
        df: DataFrame with numeric columns and a binary *target*.
        num_features: Columns to test; defaults to ``NUM_FEATURES``.
        target: Binary target column name (0/1).

    Returns:
        DataFrame with columns: feature, rank_biserial_r, U_stat, p_value,
        mean_churners, mean_non_churners. Sorted by |rank_biserial_r| desc.
    """
    if num_features is None:
        num_features = list(NUM_FEATURES)
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
        mw_result = mannwhitneyu(g1, g0, alternative="two-sided")
        U = float(mw_result.statistic)
        p = float(mw_result.pvalue)
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


def compute_significance_screen(
    df: pd.DataFrame,
    cat_features: list[str] | None = None,
    num_features: list[str] | None = None,
    target: str = TARGET,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run compute_chi2_tests and compute_mann_whitney as one joint significance screen.

    Both tables test the same underlying question — does this feature show a
    raw association with *target*? — against the same target, in the same
    screening pass. Which test statistic applies (chi-squared for categorical,
    Mann-Whitney U for numeric) is an implementation detail with no bearing on
    what "family" of simultaneous tests they belong to for false-discovery-rate
    purposes, so ``adjusted_p_value`` here is corrected jointly across every
    feature from both tables combined (``pool_adjusted_p_values``) rather than
    independently per table — correcting them independently would understate
    each table's true multiple-testing burden by ignoring the other's tests.

    Neither p-value column gates anything downstream: ``cramers_v`` and
    ``rank_biserial_r`` are the effect-size reads these two functions exist to
    produce, and feature selection is driven by cross-validated permutation
    importance in ``features/select.py``.

    Args:
        df: DataFrame with categorical/numeric columns and a binary *target*.
        cat_features: Passed to compute_chi2_tests; defaults to
            ``CAT_FEATURES + BINARY_INT_FEATURES``.
        num_features: Passed to compute_mann_whitney; defaults to ``NUM_FEATURES``.
        target: Binary target column name.

    Returns:
        (chi2_df, mwu_df) — each function's own columns plus a jointly-pooled
        ``adjusted_p_value``.
    """
    chi2_df = compute_chi2_tests(df, cat_features=cat_features, target=target)
    mwu_df = compute_mann_whitney(df, num_features=num_features, target=target)
    chi2_adj, mwu_adj = pool_adjusted_p_values(chi2_df["p_value"], mwu_df["p_value"])
    chi2_df = chi2_df.assign(adjusted_p_value=chi2_adj)
    mwu_df = mwu_df.assign(adjusted_p_value=mwu_adj)
    return chi2_df, mwu_df


def compute_vif(
    df: pd.DataFrame,
    num_cols: list[str] | None = None,
    cat_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Variance Inflation Factor table for multicollinearity detection.

    Uses ``drop_first=True`` one-hot encoding to avoid the dummy-variable
    trap (perfect multicollinearity that would make VIF undefined). VIF is
    delegated to :func:`telco_churn.utils.stats.vif_single`.

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
    if num_cols is None:
        num_cols = list(NUM_FEATURES + BINARY_INT_FEATURES)
    if cat_cols is None:
        cat_cols = list(CAT_FEATURES)
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

    vif_vals: list[float] = []
    for col in df_vif.columns:
        vif = vif_single(df_vif[col], df_vif.drop(columns=[col]))
        vif_vals.append(round(vif, 2))

    if any(v == float("inf") for v in vif_vals):
        warnings.warn(
            "VIF is inf for one or more features — perfect multicollinearity detected. "
            "Common causes: structural dependencies in the data (e.g. add-on service columns "
            "that are fully determined by a parent column), or a high-cardinality column "
            "passed to cat_cols that makes the design matrix rank-deficient.",
            stacklevel=2,
        )

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
        Square correlation DataFrame over all encoded features and the target column.
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

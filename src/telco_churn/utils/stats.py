"""Canonical statistical helpers shared across data.eda and features.generate."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.linear_model import LinearRegression

__all__ = ["abs_corr", "cramers_v", "vif_single"]


def abs_corr(a: pd.Series, b: pd.Series) -> float:
    """Absolute Spearman rank correlation between two numeric series.

    Spearman is used over Pearson to catch monotone nonlinear relationships
    (e.g. ratios, logs) that Pearson undershoots. Returns 0.0 when the result
    is NaN (e.g. one series is constant).
    """
    val = a.corr(b, method="spearman")
    return abs(float(val)) if not np.isnan(val) else 0.0


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Cramér's V association measure between two categorical series.

    Uses min(rows-1, cols-1) as the correction factor. Returns 0.0 for
    degenerate contingency tables (fewer than 2 distinct values in either
    column, zero observations, or a single-level dimension after correction).
    """
    ct = pd.crosstab(x, y)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return 0.0
    chi2 = float(sp_stats.chi2_contingency(ct, correction=False)[0])
    n = int(ct.values.sum())
    min_dim = min(ct.shape[0] - 1, ct.shape[1] - 1)
    if n == 0 or min_dim == 0:
        return 0.0
    return float(np.sqrt(chi2 / (n * min_dim)))


def vif_single(series: pd.Series, others: pd.DataFrame) -> float:
    """Variance Inflation Factor for series regressed on others (numeric columns only).

    Returns 1.0 when others has no numeric columns or the common sample is
    too small to fit reliably (< max(10, p+2) rows). Returns inf on perfect
    multicollinearity (R² ≥ 1.0).
    """
    num = others.select_dtypes(include="number").dropna()
    s = series.reindex(num.index).dropna()
    common = num.index.intersection(s.index)
    X_fit, y_fit = num.loc[common], s.loc[common]
    if X_fit.empty or len(y_fit) < max(10, X_fit.shape[1] + 2):
        return 1.0
    r2 = float(LinearRegression().fit(X_fit, y_fit).score(X_fit, y_fit))
    if r2 >= 1.0:
        return float("inf")
    return 1.0 / (1.0 - r2)

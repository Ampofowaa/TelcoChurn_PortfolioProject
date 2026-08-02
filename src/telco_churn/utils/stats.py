"""Canonical statistical helpers shared across data.eda, features.generate, and models.*."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats as sp_stats
from sklearn.linear_model import LinearRegression

__all__ = [
    "abs_corr",
    "benjamini_hochberg",
    "bootstrap_metric_ci",
    "cramers_v",
    "paired_bootstrap_ci",
    "paired_bootstrap_metric_ci",
    "pool_adjusted_p_values",
    "vif_single",
]


def benjamini_hochberg(p_values: ArrayLike) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted p-values, in the input's original order.

    For m tests with p-values sorted ascending p_(1) <= ... <= p_(m), the
    adjusted value at rank i is min_{j>=i}(p_(j) * m / j) — the running
    minimum enforces monotonicity (a lower-ranked test's adjusted p-value can
    never exceed a higher-ranked one's). Controls the expected proportion of
    false discoveries among rejected hypotheses, unlike Bonferroni (which
    controls the probability of *any* false positive and is overly
    conservative once more than a handful of tests are run) — the standard
    correction for a many-feature screen whose p-values are not individually
    load-bearing for a decision. Returns an empty array for empty input.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranks = np.arange(1, m + 1)
    adjusted_sorted = np.minimum.accumulate((p[order] * m / ranks)[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0, 1)
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def pool_adjusted_p_values(*p_value_groups: ArrayLike) -> list[np.ndarray]:
    """Benjamini-Hochberg correction applied jointly across multiple p-value groups.

    Concatenates every group into one family before calling benjamini_hochberg,
    then splits the adjusted values back out in each group's original order —
    for screens that should share one false-discovery-rate budget (e.g.
    categorical and numeric feature tests against the same target, run via two
    different test statistics) rather than being corrected independently.
    Independent per-group correction understates each group's true
    multiple-testing burden by ignoring the other groups' tests; which test
    statistic produced a p-value has no bearing on how it should be pooled —
    Benjamini-Hochberg operates on the p-values alone. Returns one array per
    input group, same order, same length; an empty argument list returns [].
    """
    if not p_value_groups:
        return []
    arrays = [np.asarray(g, dtype=float) for g in p_value_groups]
    lengths = [len(a) for a in arrays]
    pooled = np.concatenate(arrays)
    adjusted = benjamini_hochberg(pooled)
    out = []
    start = 0
    for n in lengths:
        out.append(adjusted[start : start + n])
        start += n
    return out


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


def paired_bootstrap_ci(
    scores_a: ArrayLike,
    scores_b: ArrayLike,
    n_bootstrap: int,
    random_state: int,
) -> dict[str, Any]:
    """Percentile bootstrap CI on Δ = mean(scores_a) − mean(scores_b), paired by index.

    scores_a/scores_b are paired per unit (the same fold, the same row, ...) —
    resampling draws whole units with replacement from the paired differences,
    so the caller decides what a unit is by what it passes in: per-fold scores
    resample folds, per-row scores resample rows. Returns delta_obs,
    delta_ci_lower/upper (95% percentile CI), p_value (informational — the CI
    is the decision surface, not this), n_bootstrap, and the raw
    bootstrap_deltas distribution for plotting.
    """
    rng = np.random.default_rng(random_state)
    diffs = np.asarray(scores_a) - np.asarray(scores_b)
    delta_obs = float(diffs.mean())
    n = len(diffs)
    bootstrap_deltas = np.fromiter(
        (rng.choice(diffs, size=n, replace=True).mean() for _ in range(n_bootstrap)),
        dtype=float,
        count=n_bootstrap,
    )
    p_value = float((bootstrap_deltas <= 0).mean())
    ci_lower = float(np.percentile(bootstrap_deltas, 2.5))
    ci_upper = float(np.percentile(bootstrap_deltas, 97.5))
    return {
        "delta_obs": delta_obs,
        "delta_ci_lower": ci_lower,
        "delta_ci_upper": ci_upper,
        "p_value": p_value,
        "n_bootstrap": n_bootstrap,
        "bootstrap_deltas": bootstrap_deltas,
    }


def paired_bootstrap_metric_ci(
    y_true: ArrayLike,
    proba_a: ArrayLike,
    proba_b: ArrayLike,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int,
    random_state: int,
) -> dict[str, Any]:
    """Percentile bootstrap CI on Δ = metric_fn(y_true, proba_a) − metric_fn(y_true, proba_b), by row resampling.

    For set-level metrics such as average precision, which have no per-row
    decomposition to difference directly: unlike paired_bootstrap_ci (which
    resamples pre-computed per-unit scores), this resamples row *indices*
    with replacement and recomputes metric_fn for both models over the same
    resampled rows each iteration — the only way to bootstrap a metric that
    is a property of a whole ranked set rather than a mean of per-row terms.
    A plain, unstratified row resample is correct here: prevalence jitter
    across resamples shifts both models' metric together and cancels in the
    paired difference. Returns the same delta_obs/delta_ci_lower/
    delta_ci_upper/p_value/n_bootstrap/bootstrap_deltas contract as
    paired_bootstrap_ci, so gate.py's comparative rule can consume either.
    """
    y = np.asarray(y_true)
    a = np.asarray(proba_a)
    b = np.asarray(proba_b)
    n = len(y)
    delta_obs = float(metric_fn(y, a) - metric_fn(y, b))
    rng = np.random.default_rng(random_state)
    bootstrap_deltas = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        bootstrap_deltas[i] = metric_fn(y[idx], a[idx]) - metric_fn(y[idx], b[idx])
    p_value = float((bootstrap_deltas <= 0).mean())
    ci_lower = float(np.percentile(bootstrap_deltas, 2.5))
    ci_upper = float(np.percentile(bootstrap_deltas, 97.5))
    return {
        "delta_obs": delta_obs,
        "delta_ci_lower": ci_lower,
        "delta_ci_upper": ci_upper,
        "p_value": p_value,
        "n_bootstrap": n_bootstrap,
        "bootstrap_deltas": bootstrap_deltas,
    }


def bootstrap_metric_ci(
    y_true: ArrayLike,
    proba: ArrayLike,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int,
    random_state: int,
) -> dict[str, Any]:
    """Percentile bootstrap CI for a single model's own metric, by row resampling.

    Unlike paired_bootstrap_ci/paired_bootstrap_metric_ci (which bootstrap a Δ
    between two models), this reports one model's metric with a CI — what the
    sealed-test metrics of record are checked against, i.e. the absolute
    bars a model must clear on its own, independent of any incumbent
    champion. Row-index resampling, not a per-row score mean: a
    set-level metric such as average precision has no per-row decomposition to
    bootstrap directly, the same reason paired_bootstrap_metric_ci resamples
    rows rather than reusing paired_bootstrap_ci's per-unit-score approach.
    metric_fn receives the resampled (y, proba) pair each iteration, so a
    threshold-dependent metric (recall at t*, etc.) is passed in as a closure
    over the fixed threshold rather than this function knowing about one.
    Returns obs, ci_lower/ci_upper (95% percentile CI), n_bootstrap, and the
    raw bootstrap_values distribution for plotting.
    """
    y = np.asarray(y_true)
    p = np.asarray(proba)
    n = len(y)
    obs = float(metric_fn(y, p))
    rng = np.random.default_rng(random_state)
    bootstrap_values = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        bootstrap_values[i] = metric_fn(y[idx], p[idx])
    ci_lower = float(np.percentile(bootstrap_values, 2.5))
    ci_upper = float(np.percentile(bootstrap_values, 97.5))
    return {
        "obs": obs,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_bootstrap": n_bootstrap,
        "bootstrap_values": bootstrap_values,
    }


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

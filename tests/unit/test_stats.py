"""Unit tests for src/telco_churn/utils/stats.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from telco_churn.utils.stats import (
    abs_corr,
    benjamini_hochberg,
    bootstrap_metric_ci,
    cramers_v,
    paired_bootstrap_ci,
    paired_bootstrap_metric_ci,
    pool_adjusted_p_values,
    vif_single,
)


def _mean_proba(y: np.ndarray, p: np.ndarray) -> float:
    """Metric stand-in for contract-only tests: well-defined for any class
    composition, so a same-class bootstrap resample never triggers sklearn's
    UndefinedMetricWarning the way average_precision_score would."""
    return float(np.mean(p))


# ---------------------------------------------------------------------------
# benjamini_hochberg
# ---------------------------------------------------------------------------


def test_benjamini_hochberg_empty_input_returns_empty_array() -> None:
    result = benjamini_hochberg([])
    assert result.shape == (0,)


def test_benjamini_hochberg_single_pvalue_is_unchanged() -> None:
    result = benjamini_hochberg([0.03])
    assert result == np.array([0.03])


def test_benjamini_hochberg_known_example() -> None:
    """Hand-computed against p*m/rank with the monotonicity (running-min) step.

    p = [0.01, 0.02, 0.03, 0.04, 0.50], m=5, already rank-ordered ascending:
    raw*m/rank = [0.05, 0.05, 0.05, 0.05, 0.50]; the running minimum from the
    right leaves the first four at 0.05 (already the smallest available) and
    the last at 0.50.
    """
    p = [0.01, 0.02, 0.03, 0.04, 0.50]
    result = benjamini_hochberg(p)
    expected = [0.05, 0.05, 0.05, 0.05, 0.50]
    np.testing.assert_allclose(result, expected)


def test_benjamini_hochberg_never_smaller_than_raw_pvalue() -> None:
    """The adjusted p-value at each rank can never fall below its own raw p-value."""
    rng = np.random.default_rng(42)
    p = rng.uniform(0, 1, size=50)
    result = benjamini_hochberg(p)
    assert (result >= p - 1e-12).all()


def test_benjamini_hochberg_monotone_nondecreasing_with_rank() -> None:
    """Adjusted p-values are non-decreasing when read in ascending raw-p order."""
    p = np.array([0.20, 0.01, 0.60, 0.03, 0.04])
    result = benjamini_hochberg(p)
    order = np.argsort(p)
    assert (np.diff(result[order]) >= -1e-12).all()


def test_benjamini_hochberg_bounded_by_one() -> None:
    p = [0.9, 0.95, 0.99, 1.0]
    result = benjamini_hochberg(p)
    assert (result <= 1.0).all()


def test_benjamini_hochberg_preserves_input_order() -> None:
    """Output order matches the caller's input order, not sorted order."""
    p_asc = [0.01, 0.02, 0.03, 0.04, 0.50]
    p_desc = list(reversed(p_asc))
    result_asc = benjamini_hochberg(p_asc)
    result_desc = benjamini_hochberg(p_desc)
    np.testing.assert_allclose(result_desc, list(reversed(result_asc)))


def test_benjamini_hochberg_all_equal_pvalues() -> None:
    """When every p-value ties, each gets the same adjusted value (p * m / m = p)."""
    p = [0.02] * 6
    result = benjamini_hochberg(p)
    np.testing.assert_allclose(result, [0.02] * 6)


# ---------------------------------------------------------------------------
# pool_adjusted_p_values
# ---------------------------------------------------------------------------


def test_pool_adjusted_p_values_no_groups_returns_empty_list() -> None:
    assert pool_adjusted_p_values() == []


def test_pool_adjusted_p_values_matches_manual_concatenation() -> None:
    """Pooled output equals benjamini_hochberg() run once over the concatenated groups."""
    group_a = [0.001, 0.02, 0.30]
    group_b = [0.10, 0.45]
    result_a, result_b = pool_adjusted_p_values(group_a, group_b)
    expected = benjamini_hochberg(group_a + group_b)
    np.testing.assert_allclose(np.concatenate([result_a, result_b]), expected)


def test_pool_adjusted_p_values_preserves_group_lengths_and_order() -> None:
    group_a = [0.01, 0.02, 0.03]
    group_b = [0.50]
    group_c = [0.10, 0.20]
    out = pool_adjusted_p_values(group_a, group_b, group_c)
    assert [len(g) for g in out] == [3, 1, 2]


def test_pool_adjusted_p_values_differs_from_independent_correction() -> None:
    """Pooling into one larger family changes the correction vs. adjusting each group alone.

    Regression guard that pooling is actually happening rather than silently
    falling back to per-group correction: a small group's smallest p-value
    gets a harsher (larger) adjustment once pooled with more tests, since the
    family size m in p*m/rank grows.
    """
    small_group = [0.01]
    other_group = [0.02, 0.03, 0.04, 0.05, 0.90, 0.95]
    pooled_small, _ = pool_adjusted_p_values(small_group, other_group)
    independent_small = benjamini_hochberg(small_group)
    assert pooled_small[0] > independent_small[0]


def test_pool_adjusted_p_values_single_group_matches_benjamini_hochberg() -> None:
    """Pooling a single group degenerates to a plain benjamini_hochberg call."""
    p = [0.01, 0.2, 0.05, 0.6]
    (result,) = pool_adjusted_p_values(p)
    np.testing.assert_allclose(result, benjamini_hochberg(p))


def test_pool_adjusted_p_values_handles_empty_group() -> None:
    result_a, result_b = pool_adjusted_p_values([], [0.01, 0.5])
    assert result_a.shape == (0,)
    assert result_b.shape == (2,)


# ---------------------------------------------------------------------------
# abs_corr
# ---------------------------------------------------------------------------


def test_abs_corr_perfect_monotone_nonlinear_relationship_is_one() -> None:
    """A perfectly monotone but nonlinear relationship gives |rho| = 1 under
    Spearman, which Pearson would understate."""
    a = pd.Series([1, 2, 3, 4, 5])
    b = pd.Series([1.0, 4.0, 9.0, 16.0, 25.0])  # monotone nonlinear (a**2)
    assert abs_corr(a, b) == pytest.approx(1.0)


def test_abs_corr_perfect_negative_relationship_returns_positive_one() -> None:
    """abs_corr takes the magnitude, so a perfect inverse relationship also
    returns 1.0, not -1.0."""
    a = pd.Series([1, 2, 3, 4, 5])
    b = pd.Series([5, 4, 3, 2, 1])
    assert abs_corr(a, b) == pytest.approx(1.0)


def test_abs_corr_no_relationship_is_near_zero() -> None:
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(size=200))
    b = pd.Series(rng.normal(size=200))
    assert abs_corr(a, b) < 0.15


def test_abs_corr_constant_series_returns_zero_not_nan() -> None:
    """Spearman correlation against a constant series is NaN; abs_corr must
    convert that to 0.0 rather than propagating NaN into a redundancy screen."""
    a = pd.Series([5, 5, 5, 5, 5])
    b = pd.Series([1, 2, 3, 4, 5])
    assert abs_corr(a, b) == 0.0


# ---------------------------------------------------------------------------
# cramers_v
# ---------------------------------------------------------------------------


def test_cramers_v_perfect_association_is_one() -> None:
    """x fully determines y (one-to-one categorical mapping) -> V = 1.0."""
    x = pd.Series(["a", "a", "b", "b", "c", "c"])
    y = pd.Series(["p", "p", "q", "q", "r", "r"])
    assert cramers_v(x, y) == pytest.approx(1.0)


def test_cramers_v_independent_categoricals_is_near_zero() -> None:
    rng = np.random.default_rng(0)
    x = pd.Series(rng.choice(["a", "b", "c"], size=300))
    y = pd.Series(rng.choice(["p", "q"], size=300))
    assert cramers_v(x, y) < 0.15


def test_cramers_v_single_level_column_returns_zero() -> None:
    """Fewer than 2 distinct values in either column is a degenerate
    contingency table -- returns 0.0 rather than raising or NaN."""
    x = pd.Series(["a", "a", "a", "a"])
    y = pd.Series(["p", "q", "p", "q"])
    assert cramers_v(x, y) == 0.0


def test_cramers_v_empty_series_returns_zero() -> None:
    x = pd.Series([], dtype=object)
    y = pd.Series([], dtype=object)
    assert cramers_v(x, y) == 0.0


# ---------------------------------------------------------------------------
# paired_bootstrap_ci
# ---------------------------------------------------------------------------


def test_paired_bootstrap_ci_delta_obs_equals_mean_of_paired_diffs() -> None:
    """delta_obs is the exact mean of scores_a - scores_b, paired by index --
    not a bootstrap estimate."""
    scores_a = np.array([0.80, 0.82, 0.79, 0.85, 0.81])
    scores_b = np.array([0.75, 0.77, 0.74, 0.80, 0.76])
    result = paired_bootstrap_ci(scores_a, scores_b, n_bootstrap=500, random_state=0)
    assert result["delta_obs"] == pytest.approx((scores_a - scores_b).mean())


def test_paired_bootstrap_ci_recovers_planted_gap() -> None:
    """scores_a systematically higher per-unit than scores_b -- Delta_obs must
    be positive and the CI must exclude 0 in scores_a's favour."""
    rng = np.random.default_rng(0)
    n = 50
    scores_b = rng.uniform(0.6, 0.9, size=n)
    scores_a = scores_b + 0.05 + rng.normal(0, 0.01, size=n)
    result = paired_bootstrap_ci(scores_a, scores_b, n_bootstrap=1000, random_state=42)
    assert result["delta_obs"] > 0
    assert result["delta_ci_lower"] > 0


def test_paired_bootstrap_ci_identical_scores_give_zero_width_ci_at_zero() -> None:
    """Paired diffs are all zero when scores_a == scores_b -- every bootstrap
    resample of an all-zero array is also all-zero, so the CI collapses to a
    point at 0 rather than merely straddling it."""
    scores = np.array([0.7, 0.8, 0.6, 0.9, 0.75])
    result = paired_bootstrap_ci(scores, scores, n_bootstrap=500, random_state=1)
    assert result["delta_obs"] == 0.0
    assert result["delta_ci_lower"] == 0.0
    assert result["delta_ci_upper"] == 0.0


def test_paired_bootstrap_ci_ci_contains_obs() -> None:
    """The 95% CI must bracket the observed Delta."""
    rng = np.random.default_rng(2)
    n = 60
    scores_a = rng.uniform(0.5, 0.9, size=n)
    scores_b = rng.uniform(0.5, 0.9, size=n)
    result = paired_bootstrap_ci(scores_a, scores_b, n_bootstrap=1000, random_state=7)
    assert result["delta_ci_lower"] <= result["delta_obs"] <= result["delta_ci_upper"]


def test_paired_bootstrap_ci_return_keys() -> None:
    """Same six-key contract as paired_bootstrap_metric_ci, so gate.py can
    consume either."""
    scores_a = np.array([0.8, 0.9, 0.7, 0.85, 0.75])
    scores_b = np.array([0.7, 0.8, 0.6, 0.75, 0.65])
    result = paired_bootstrap_ci(scores_a, scores_b, n_bootstrap=50, random_state=0)
    assert set(result) == {
        "delta_obs",
        "delta_ci_lower",
        "delta_ci_upper",
        "p_value",
        "n_bootstrap",
        "bootstrap_deltas",
    }


def test_paired_bootstrap_ci_n_bootstrap_preserved() -> None:
    scores_a = np.array([0.8, 0.9, 0.7, 0.85, 0.75])
    scores_b = np.array([0.7, 0.8, 0.6, 0.75, 0.65])
    result = paired_bootstrap_ci(scores_a, scores_b, n_bootstrap=123, random_state=0)
    assert result["n_bootstrap"] == 123
    assert len(result["bootstrap_deltas"]) == 123


def test_paired_bootstrap_ci_deterministic_under_fixed_random_state() -> None:
    """Same random_state must reproduce bit-identical bootstrap draws."""
    rng = np.random.default_rng(3)
    n = 40
    scores_a = rng.uniform(0.5, 0.9, size=n)
    scores_b = rng.uniform(0.5, 0.9, size=n)
    result_1 = paired_bootstrap_ci(scores_a, scores_b, n_bootstrap=200, random_state=99)
    result_2 = paired_bootstrap_ci(scores_a, scores_b, n_bootstrap=200, random_state=99)
    np.testing.assert_array_equal(
        result_1["bootstrap_deltas"], result_2["bootstrap_deltas"]
    )
    assert result_1["delta_obs"] == result_2["delta_obs"]


# ---------------------------------------------------------------------------
# paired_bootstrap_metric_ci
# ---------------------------------------------------------------------------


def test_paired_bootstrap_metric_ci_recovers_planted_ranking_gap() -> None:
    """proba_a ranks by the true signal; proba_b is pure noise — Δ_obs must be
    positive and the CI must exclude 0 in proba_a's favour."""
    rng = np.random.default_rng(0)
    n = 400
    y = (rng.uniform(size=n) < 0.3).astype(int)
    proba_a = np.clip(y * 0.7 + rng.normal(0, 0.15, size=n), 0, 1)
    proba_b = rng.uniform(size=n)
    result = paired_bootstrap_metric_ci(
        y,
        proba_a,
        proba_b,
        average_precision_score,
        n_bootstrap=1000,
        random_state=42,
    )
    assert result["delta_obs"] > 0
    assert result["delta_ci_lower"] > 0


def test_paired_bootstrap_metric_ci_identical_scores_give_zero_delta_and_straddling_ci() -> (
    None
):
    """The noise case the gate must refuse: no ranking gap, CI must straddle 0."""
    rng = np.random.default_rng(1)
    n = 200
    y = (rng.uniform(size=n) < 0.3).astype(int)
    proba = rng.uniform(size=n)
    result = paired_bootstrap_metric_ci(
        y, proba, proba, average_precision_score, n_bootstrap=500, random_state=42
    )
    assert result["delta_obs"] == 0.0
    assert result["delta_ci_lower"] <= 0.0 <= result["delta_ci_upper"]


def test_paired_bootstrap_metric_ci_ci_contains_obs() -> None:
    """The 95% CI must bracket the observed Δ."""
    rng = np.random.default_rng(2)
    n = 300
    y = (rng.uniform(size=n) < 0.3).astype(int)
    proba_a = np.clip(y * 0.5 + rng.normal(0, 0.2, size=n), 0, 1)
    proba_b = np.clip(y * 0.3 + rng.normal(0, 0.2, size=n), 0, 1)
    result = paired_bootstrap_metric_ci(
        y,
        proba_a,
        proba_b,
        average_precision_score,
        n_bootstrap=1000,
        random_state=7,
    )
    assert result["delta_ci_lower"] <= result["delta_obs"] <= result["delta_ci_upper"]


def test_paired_bootstrap_metric_ci_return_keys() -> None:
    """Same six-key contract as paired_bootstrap_ci, so gate.py can consume either."""
    y = np.array([0, 1, 0, 1, 1])
    proba = np.array([0.1, 0.9, 0.2, 0.8, 0.7])
    result = paired_bootstrap_metric_ci(
        y, proba, proba, _mean_proba, n_bootstrap=50, random_state=0
    )
    assert set(result) == {
        "delta_obs",
        "delta_ci_lower",
        "delta_ci_upper",
        "p_value",
        "n_bootstrap",
        "bootstrap_deltas",
    }


def test_paired_bootstrap_metric_ci_n_bootstrap_preserved() -> None:
    y = np.array([0, 1, 0, 1, 1])
    proba = np.array([0.1, 0.9, 0.2, 0.8, 0.7])
    result = paired_bootstrap_metric_ci(
        y, proba, proba, _mean_proba, n_bootstrap=123, random_state=0
    )
    assert result["n_bootstrap"] == 123


def test_paired_bootstrap_metric_ci_deterministic_under_fixed_random_state() -> None:
    """Same random_state must reproduce bit-identical bootstrap draws."""
    rng = np.random.default_rng(3)
    n = 150
    y = (rng.uniform(size=n) < 0.3).astype(int)
    proba_a = rng.uniform(size=n)
    proba_b = rng.uniform(size=n)
    result_1 = paired_bootstrap_metric_ci(
        y,
        proba_a,
        proba_b,
        average_precision_score,
        n_bootstrap=200,
        random_state=99,
    )
    result_2 = paired_bootstrap_metric_ci(
        y,
        proba_a,
        proba_b,
        average_precision_score,
        n_bootstrap=200,
        random_state=99,
    )
    np.testing.assert_array_equal(
        result_1["bootstrap_deltas"], result_2["bootstrap_deltas"]
    )
    assert result_1["delta_obs"] == result_2["delta_obs"]


# ---------------------------------------------------------------------------
# bootstrap_metric_ci
# ---------------------------------------------------------------------------


def test_bootstrap_metric_ci_obs_matches_direct_computation() -> None:
    """obs equals metric_fn applied directly to the unresampled data."""
    rng = np.random.default_rng(0)
    n = 300
    y = (rng.uniform(size=n) < 0.3).astype(int)
    proba = np.clip(y * 0.6 + rng.normal(0, 0.2, size=n), 0, 1)
    result = bootstrap_metric_ci(
        y, proba, average_precision_score, n_bootstrap=500, random_state=42
    )
    assert result["obs"] == pytest.approx(average_precision_score(y, proba))


def test_bootstrap_metric_ci_ci_contains_obs() -> None:
    """The 95% CI must bracket the observed value."""
    rng = np.random.default_rng(1)
    n = 300
    y = (rng.uniform(size=n) < 0.3).astype(int)
    proba = np.clip(y * 0.5 + rng.normal(0, 0.2, size=n), 0, 1)
    result = bootstrap_metric_ci(
        y, proba, average_precision_score, n_bootstrap=1000, random_state=7
    )
    assert result["ci_lower"] <= result["obs"] <= result["ci_upper"]


def test_bootstrap_metric_ci_recovers_high_pr_auc_for_strong_separator() -> None:
    """A near-perfect separator's CI sits high and tight, well above the
    ~0.3 base-rate floor."""
    rng = np.random.default_rng(2)
    n = 400
    y = (rng.uniform(size=n) < 0.3).astype(int)
    proba = np.where(y == 1, 0.9, 0.1) + rng.normal(0, 0.03, size=n)
    result = bootstrap_metric_ci(
        y, proba, average_precision_score, n_bootstrap=500, random_state=3
    )
    assert result["ci_lower"] > 0.7


def test_bootstrap_metric_ci_return_keys() -> None:
    """Result carries exactly the five documented keys."""
    y = np.array([0, 1, 0, 1, 1])
    proba = np.array([0.1, 0.9, 0.2, 0.8, 0.7])
    result = bootstrap_metric_ci(y, proba, _mean_proba, n_bootstrap=50, random_state=0)
    assert set(result) == {
        "obs",
        "ci_lower",
        "ci_upper",
        "n_bootstrap",
        "bootstrap_values",
    }


def test_bootstrap_metric_ci_n_bootstrap_preserved() -> None:
    y = np.array([0, 1, 0, 1, 1])
    proba = np.array([0.1, 0.9, 0.2, 0.8, 0.7])
    result = bootstrap_metric_ci(y, proba, _mean_proba, n_bootstrap=123, random_state=0)
    assert result["n_bootstrap"] == 123
    assert len(result["bootstrap_values"]) == 123


def test_bootstrap_metric_ci_deterministic_under_fixed_random_state() -> None:
    """Same random_state must reproduce bit-identical bootstrap draws."""
    rng = np.random.default_rng(3)
    n = 150
    y = (rng.uniform(size=n) < 0.3).astype(int)
    proba = rng.uniform(size=n)
    result_1 = bootstrap_metric_ci(
        y, proba, average_precision_score, n_bootstrap=200, random_state=99
    )
    result_2 = bootstrap_metric_ci(
        y, proba, average_precision_score, n_bootstrap=200, random_state=99
    )
    np.testing.assert_array_equal(
        result_1["bootstrap_values"], result_2["bootstrap_values"]
    )
    assert result_1["obs"] == result_2["obs"]


@pytest.mark.filterwarnings(
    "ignore:Recall is ill-defined:sklearn.exceptions.UndefinedMetricWarning"
)
def test_bootstrap_metric_ci_accepts_thresholded_metric_closure() -> None:
    """metric_fn can be a closure over a fixed threshold — recall at t*, not
    just a raw-probability metric like average precision.

    Scoped filterwarnings, not the global ini list: this n=5 fixture's
    bootstrap resamples occasionally draw no positives, which is expected and
    benign here, but a recall-undefined warning elsewhere in the suite (e.g. a
    genuinely degenerate fairness slice) should still surface.
    """
    from sklearn.metrics import recall_score

    y = np.array([0, 0, 1, 1, 1])
    proba = np.array([0.1, 0.4, 0.6, 0.7, 0.9])
    t_star = 0.5

    def recall_at_threshold(y_true: np.ndarray, p: np.ndarray) -> float:
        return float(recall_score(y_true, (p >= t_star).astype(int)))

    result = bootstrap_metric_ci(
        y, proba, recall_at_threshold, n_bootstrap=200, random_state=0
    )
    assert result["obs"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# vif_single
# ---------------------------------------------------------------------------


def test_vif_single_no_relationship_is_near_one() -> None:
    """Independent numeric predictors give a VIF close to the no-collinearity
    floor of 1.0."""
    rng = np.random.default_rng(0)
    n = 200
    series = pd.Series(rng.normal(size=n))
    others = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    assert vif_single(series, others) == pytest.approx(1.0, abs=0.3)


def test_vif_single_perfect_collinearity_is_inf() -> None:
    """series exactly reproducible from others (R^2 >= 1.0) -> VIF is infinite."""
    n = 20
    x = pd.Series(np.arange(n, dtype=float))
    others = pd.DataFrame({"x_copy": x.to_numpy()})
    assert vif_single(x, others) == float("inf")


def test_vif_single_no_numeric_columns_in_others_returns_one() -> None:
    """others with no numeric columns can't be regressed on -- returns the
    no-collinearity floor rather than raising."""
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    others = pd.DataFrame({"cat": ["a", "b", "c", "d", "e"]})
    assert vif_single(series, others) == 1.0


def test_vif_single_too_few_common_rows_returns_one() -> None:
    """A common sample smaller than max(10, p+2) is too small to fit reliably
    -- returns 1.0 rather than an unstable regression estimate."""
    series = pd.Series([1.0, 2.0, 3.0])
    others = pd.DataFrame({"x1": [1.0, 2.0, 3.0]})
    assert vif_single(series, others) == 1.0

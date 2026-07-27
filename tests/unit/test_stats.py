"""Unit tests for src/telco_churn/utils/stats.py."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from telco_churn.utils.stats import (
    benjamini_hochberg,
    bootstrap_metric_ci,
    paired_bootstrap_metric_ci,
    pool_adjusted_p_values,
)

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
        y, proba, proba, average_precision_score, n_bootstrap=50, random_state=0
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
        y, proba, proba, average_precision_score, n_bootstrap=123, random_state=0
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
    result = bootstrap_metric_ci(
        y, proba, average_precision_score, n_bootstrap=50, random_state=0
    )
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
    result = bootstrap_metric_ci(
        y, proba, average_precision_score, n_bootstrap=123, random_state=0
    )
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


def test_bootstrap_metric_ci_accepts_thresholded_metric_closure() -> None:
    """metric_fn can be a closure over a fixed threshold — recall at t*, not
    just a raw-probability metric like average precision."""
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

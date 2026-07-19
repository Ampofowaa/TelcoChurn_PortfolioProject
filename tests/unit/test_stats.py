"""Unit tests for src/telco_churn/utils/stats.py."""

from __future__ import annotations

import numpy as np

from telco_churn.utils.stats import benjamini_hochberg, pool_adjusted_p_values

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

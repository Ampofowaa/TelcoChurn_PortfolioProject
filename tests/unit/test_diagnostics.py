"""Unit tests for telco_churn.models.diagnostics — non-gating selection helpers (B4)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from telco_churn.models.diagnostics import (
    fixed_recall_profile,
    generalization_gap,
    learning_curve_points,
    segment_bootstrap_ci,
    segment_bootstrap_delta,
    segment_decision_rates,
    segment_oof_errors,
)

# ---------------------------------------------------------------------------
# fixed_recall_profile
# ---------------------------------------------------------------------------


def test_fixed_recall_profile_keys() -> None:
    """Each row contains the five expected keys."""
    rng = np.random.default_rng(1)
    n = 200
    y_true = rng.integers(0, 2, size=n).tolist()
    proba = rng.random(size=n).tolist()
    rows = fixed_recall_profile(y_true, proba, [0.70])
    assert len(rows) == 1
    assert set(rows[0]) == {
        "recall_target",
        "precision",
        "recall_achieved",
        "f1",
        "threshold",
    }


def test_fixed_recall_profile_achieved_recall_meets_target() -> None:
    """Achieved recall is >= each target (up to rounding)."""
    y_true = [0] * 70 + [1] * 30
    proba = [0.1] * 70 + [0.9] * 30  # all positives scored high
    rows = fixed_recall_profile(y_true, proba, [0.70, 0.80, 0.90])
    for row in rows:
        assert not math.isnan(row["recall_achieved"])
        assert row["recall_achieved"] >= row["recall_target"] - 1e-6


def test_fixed_recall_profile_unreachable_target_returns_nan() -> None:
    """A recall target above 1.0 is unreachable and returns NaN entries."""
    y_true = [0] * 10 + [1] * 10
    proba = [float(i) / 20.0 for i in range(20)]
    rows = fixed_recall_profile(y_true, proba, [1.1])
    assert math.isnan(rows[0]["precision"])
    assert math.isnan(rows[0]["recall_achieved"])


def test_fixed_recall_profile_multiple_targets_length() -> None:
    """One row is returned per recall target."""
    rng = np.random.default_rng(2)
    y_true = rng.integers(0, 2, size=100).tolist()
    proba = rng.random(size=100).tolist()
    rows = fixed_recall_profile(y_true, proba, [0.70, 0.80, 0.90])
    assert len(rows) == 3


def test_fixed_recall_profile_perfect_separator_hits_precision_near_one() -> None:
    """A perfect separator (every negative scored strictly below every positive)
    achieves precision == 1.0 at every fixed-recall target, not just recall_achieved
    >= target — the profile should reward a genuinely clean ranking, not merely a
    threshold that happens to clear the recall bar.
    """
    y_true = [0] * 50 + [1] * 50
    proba = [0.0 + i * 0.001 for i in range(50)] + [0.9 + i * 0.001 for i in range(50)]
    rows = fixed_recall_profile(y_true, proba, [0.70, 0.80, 0.90, 1.0])
    for row in rows:
        assert row["precision"] == pytest.approx(1.0)


def test_fixed_recall_profile_precision_non_increasing_as_recall_rises() -> None:
    """Precision at a higher recall target never exceeds precision at a lower one —
    the mathematical property of a PR curve (argmax precision over a shrinking
    qualifying set as the recall floor rises) that the fixed-recall profile relies on.
    """
    rng = np.random.default_rng(42)
    n = 300
    y_true = rng.integers(0, 2, size=n).tolist()
    proba = rng.random(size=n).tolist()
    rows = fixed_recall_profile(y_true, proba, [0.5, 0.6, 0.7, 0.8, 0.9])
    precisions = [row["precision"] for row in rows]
    for earlier, later in zip(precisions, precisions[1:], strict=False):
        if math.isnan(later):
            continue
        assert later <= earlier + 1e-9


def test_fixed_recall_profile_empty_input_raises() -> None:
    """Empty y_true/proba raises rather than silently returning degenerate rows —
    sklearn's precision_recall_curve has no defined behavior on zero samples."""
    with pytest.raises(ValueError, match="could not be broadcast"):
        fixed_recall_profile([], [], [0.70])


# ---------------------------------------------------------------------------
# segment_oof_errors
# ---------------------------------------------------------------------------


def test_segment_oof_errors_keys() -> None:
    """Each row contains the five expected keys."""
    rng = np.random.default_rng(3)
    n = 100
    y_true = rng.integers(0, 2, size=n).tolist()
    proba = rng.random(size=n).tolist()
    group = pd.Series(rng.choice(["A", "B"], size=n), name="test_col")
    rows = segment_oof_errors(y_true, proba, group)
    assert len(rows) >= 1
    for row in rows:
        assert {"segment", "value", "n", "churn_rate", "pr_auc"} <= set(row)


def test_segment_oof_errors_segment_name() -> None:
    """segment field matches the Series name."""
    rng = np.random.default_rng(4)
    n = 60
    y_true = rng.integers(0, 2, size=n).tolist()
    proba = rng.random(size=n).tolist()
    group = pd.Series(["X"] * 30 + ["Y"] * 30, name="my_col")
    rows = segment_oof_errors(y_true, proba, group)
    assert all(r["segment"] == "my_col" for r in rows)


def test_segment_oof_errors_skips_small_groups() -> None:
    """Groups with fewer than 10 samples are excluded."""
    rng = np.random.default_rng(5)
    n_common = 80
    n_rare = 5
    n = n_common + n_rare
    y_true = rng.integers(0, 2, size=n).tolist()
    proba = rng.random(size=n).tolist()
    group = pd.Series(["common"] * n_common + ["rare"] * n_rare, name="g")
    rows = segment_oof_errors(y_true, proba, group)
    values = [r["value"] for r in rows]
    assert "rare" not in values
    assert "common" in values


def test_segment_oof_errors_pr_auc_in_range() -> None:
    """PR-AUC values must lie in [0, 1]."""
    rng = np.random.default_rng(6)
    n = 100
    y_true = rng.integers(0, 2, size=n).tolist()
    proba = rng.random(size=n).tolist()
    group = pd.Series(rng.choice(["A", "B", "C"], size=n), name="col")
    rows = segment_oof_errors(y_true, proba, group)
    for row in rows:
        assert 0.0 <= row["pr_auc"] <= 1.0


def test_segment_oof_errors_no_nan_covers_all_rows() -> None:
    """Single-class segments fall back to churn rate instead of producing NaN."""
    y_true = [0] * 10  # single-class segment, no positives
    proba = [0.1 + i * 0.01 for i in range(10)]
    group = pd.Series(["only"] * 10, name="flag")
    rows = segment_oof_errors(y_true, proba, group)
    assert len(rows) == 1
    assert not math.isnan(rows[0]["pr_auc"])
    assert rows[0]["pr_auc"] == 0.0


def test_segment_oof_errors_empty_group_returns_empty_list() -> None:
    """An empty group Series returns an empty list, not an error — there are no
    segment values to iterate over."""
    rows = segment_oof_errors([], [], pd.Series([], dtype=object, name="g"))
    assert rows == []


def test_segment_oof_errors_surfaces_planted_weak_segment() -> None:
    """A segment with a genuinely weak model (near-random predictions) is visible
    in the per-segment PR-AUC table as materially lower than a strong segment's —
    the table must differentiate segments, not average the weakness away.
    """
    rng = np.random.default_rng(7)
    n_strong, n_weak = 200, 200
    y_strong = rng.integers(0, 2, size=n_strong)
    proba_strong = np.where(y_strong == 1, 0.9, 0.1) + rng.normal(0, 0.05, n_strong)
    y_weak = rng.integers(0, 2, size=n_weak)
    proba_weak = rng.random(n_weak)

    y_true = np.concatenate([y_strong, y_weak]).tolist()
    proba = np.concatenate([proba_strong, proba_weak]).tolist()
    group = pd.Series(["strong"] * n_strong + ["weak"] * n_weak, name="segment")

    rows = {row["value"]: row for row in segment_oof_errors(y_true, proba, group)}
    assert rows["strong"]["pr_auc"] > rows["weak"]["pr_auc"] + 0.3


# ---------------------------------------------------------------------------
# segment_bootstrap_delta
# ---------------------------------------------------------------------------


def test_segment_bootstrap_delta_keys() -> None:
    """Each row contains the six expected keys."""
    rng = np.random.default_rng(10)
    n = 100
    y_true = rng.integers(0, 2, size=n).tolist()
    proba_lgbm = rng.random(size=n).tolist()
    proba_logreg = rng.random(size=n).tolist()
    group = pd.Series(rng.choice(["A", "B"], size=n), name="test_col")
    rows = segment_bootstrap_delta(
        y_true, proba_lgbm, proba_logreg, group, n_bootstrap=200, random_state=42
    )
    assert len(rows) >= 1
    for row in rows:
        assert {
            "segment",
            "value",
            "n",
            "delta_obs",
            "delta_ci_lower",
            "delta_ci_upper",
        } <= set(row)


def test_segment_bootstrap_delta_segment_name() -> None:
    """segment field matches the Series name."""
    rng = np.random.default_rng(11)
    n = 60
    y_true = rng.integers(0, 2, size=n).tolist()
    proba_lgbm = rng.random(size=n).tolist()
    proba_logreg = rng.random(size=n).tolist()
    group = pd.Series(["X"] * 30 + ["Y"] * 30, name="my_col")
    rows = segment_bootstrap_delta(
        y_true, proba_lgbm, proba_logreg, group, n_bootstrap=200, random_state=42
    )
    assert all(r["segment"] == "my_col" for r in rows)


def test_segment_bootstrap_delta_skips_small_groups() -> None:
    """Groups with fewer than 10 samples are excluded."""
    rng = np.random.default_rng(12)
    n_common = 80
    n_rare = 5
    n = n_common + n_rare
    y_true = rng.integers(0, 2, size=n).tolist()
    proba_lgbm = rng.random(size=n).tolist()
    proba_logreg = rng.random(size=n).tolist()
    group = pd.Series(["common"] * n_common + ["rare"] * n_rare, name="g")
    rows = segment_bootstrap_delta(
        y_true, proba_lgbm, proba_logreg, group, n_bootstrap=200, random_state=42
    )
    values = [r["value"] for r in rows]
    assert "rare" not in values
    assert "common" in values


def test_segment_bootstrap_delta_skips_single_class_segment() -> None:
    """A segment with only one class present is skipped — delta is undefined without both classes."""
    y_true = [0] * 10  # single-class segment, no positives
    proba_lgbm = [0.1 + i * 0.01 for i in range(10)]
    proba_logreg = [0.2 + i * 0.01 for i in range(10)]
    group = pd.Series(["only"] * 10, name="flag")
    rows = segment_bootstrap_delta(
        y_true, proba_lgbm, proba_logreg, group, n_bootstrap=200, random_state=42
    )
    assert rows == []


def test_segment_bootstrap_delta_detects_clear_lgbm_advantage() -> None:
    """A near-perfect LGBM ranking against a random LogReg ranking gives a CI entirely above zero."""
    rng = np.random.default_rng(13)
    n = 200
    y_true = np.array([0] * 140 + [1] * 60)
    proba_lgbm = np.where(y_true == 1, 0.9, 0.1) + rng.normal(0, 0.02, n)
    proba_logreg = rng.random(n)
    group = pd.Series(["seg"] * n, name="g")
    rows = segment_bootstrap_delta(
        y_true.tolist(),
        proba_lgbm.tolist(),
        proba_logreg.tolist(),
        group,
        n_bootstrap=500,
        random_state=42,
    )
    assert len(rows) == 1
    assert rows[0]["delta_obs"] > 0
    assert rows[0]["delta_ci_lower"] > 0


def test_segment_bootstrap_delta_zero_for_identical_probas() -> None:
    """Identical predictions for both candidates give an exact zero delta and a degenerate CI at zero."""
    rng = np.random.default_rng(14)
    n = 100
    y_true = rng.integers(0, 2, size=n).tolist()
    proba = rng.random(size=n).tolist()
    group = pd.Series(["seg"] * n, name="g")
    rows = segment_bootstrap_delta(
        y_true, proba, proba, group, n_bootstrap=200, random_state=42
    )
    assert rows[0]["delta_obs"] == 0.0
    assert rows[0]["delta_ci_lower"] == 0.0
    assert rows[0]["delta_ci_upper"] == 0.0


def test_segment_bootstrap_delta_deterministic_with_fixed_seed() -> None:
    """Same random_state reproduces byte-identical results across calls."""
    rng = np.random.default_rng(15)
    n = 100
    y_true = rng.integers(0, 2, size=n).tolist()
    proba_lgbm = rng.random(size=n).tolist()
    proba_logreg = rng.random(size=n).tolist()
    group = pd.Series(["seg"] * n, name="g")
    rows_a = segment_bootstrap_delta(
        y_true, proba_lgbm, proba_logreg, group, n_bootstrap=200, random_state=42
    )
    rows_b = segment_bootstrap_delta(
        y_true, proba_lgbm, proba_logreg, group, n_bootstrap=200, random_state=42
    )
    assert rows_a == rows_b


# ---------------------------------------------------------------------------
# segment_bootstrap_ci
# ---------------------------------------------------------------------------


def test_segment_bootstrap_ci_keys() -> None:
    """Each row contains the six expected keys."""
    rng = np.random.default_rng(20)
    n = 100
    y_true = rng.integers(0, 2, size=n).tolist()
    proba = rng.random(size=n).tolist()
    group = pd.Series(rng.choice(["A", "B"], size=n), name="test_col")
    rows = segment_bootstrap_ci(y_true, proba, group, n_bootstrap=200, random_state=42)
    assert len(rows) >= 1
    for row in rows:
        assert {
            "segment",
            "value",
            "n",
            "pr_auc_obs",
            "pr_auc_ci_lower",
            "pr_auc_ci_upper",
        } <= set(row)


def test_segment_bootstrap_ci_segment_name() -> None:
    """segment field matches the Series name."""
    rng = np.random.default_rng(21)
    n = 60
    y_true = rng.integers(0, 2, size=n).tolist()
    proba = rng.random(size=n).tolist()
    group = pd.Series(["X"] * 30 + ["Y"] * 30, name="my_col")
    rows = segment_bootstrap_ci(y_true, proba, group, n_bootstrap=200, random_state=42)
    assert all(r["segment"] == "my_col" for r in rows)


def test_segment_bootstrap_ci_skips_small_groups() -> None:
    """Groups with fewer than 10 samples are excluded."""
    rng = np.random.default_rng(22)
    n_common, n_rare = 80, 5
    n = n_common + n_rare
    y_true = rng.integers(0, 2, size=n).tolist()
    proba = rng.random(size=n).tolist()
    group = pd.Series(["common"] * n_common + ["rare"] * n_rare, name="g")
    rows = segment_bootstrap_ci(y_true, proba, group, n_bootstrap=200, random_state=42)
    values = [r["value"] for r in rows]
    assert "rare" not in values
    assert "common" in values


def test_segment_bootstrap_ci_skips_single_class_segment() -> None:
    """A segment with only one class present is skipped — PR-AUC is undefined without both classes."""
    y_true = [0] * 10
    proba = [0.1 + i * 0.01 for i in range(10)]
    group = pd.Series(["only"] * 10, name="flag")
    rows = segment_bootstrap_ci(y_true, proba, group, n_bootstrap=200, random_state=42)
    assert rows == []


def test_segment_bootstrap_ci_lower_bound_clears_churn_floor_for_strong_segment() -> (
    None
):
    """A segment where the model separates classes cleanly has a CI lower bound
    comfortably above that segment's own churn-rate floor — the V1 veto condition
    this function feeds."""
    rng = np.random.default_rng(23)
    n = 300
    y_true = rng.integers(0, 2, size=n)
    proba = np.where(y_true == 1, 0.9, 0.1) + rng.normal(0, 0.05, n)
    group = pd.Series(["seg"] * n, name="g")
    rows = segment_bootstrap_ci(
        y_true.tolist(), proba.tolist(), group, n_bootstrap=500, random_state=42
    )
    churn_floor = float(y_true.mean())
    assert rows[0]["pr_auc_ci_lower"] > churn_floor


def test_segment_bootstrap_ci_flags_weak_segment_near_floor() -> None:
    """A segment with near-random predictions has a CI lower bound close to (not
    materially above) its own churn-rate floor — the collapse case V1 vetoes on."""
    rng = np.random.default_rng(24)
    n = 300
    y_true = rng.integers(0, 2, size=n)
    proba = rng.random(n)  # uninformative
    group = pd.Series(["seg"] * n, name="g")
    rows = segment_bootstrap_ci(
        y_true.tolist(), proba.tolist(), group, n_bootstrap=500, random_state=42
    )
    churn_floor = float(y_true.mean())
    assert rows[0]["pr_auc_ci_lower"] < churn_floor + 0.15


def test_segment_bootstrap_ci_deterministic_with_fixed_seed() -> None:
    """Same random_state reproduces byte-identical results across calls."""
    rng = np.random.default_rng(25)
    n = 100
    y_true = rng.integers(0, 2, size=n).tolist()
    proba = rng.random(size=n).tolist()
    group = pd.Series(["seg"] * n, name="g")
    rows_a = segment_bootstrap_ci(
        y_true, proba, group, n_bootstrap=200, random_state=42
    )
    rows_b = segment_bootstrap_ci(
        y_true, proba, group, n_bootstrap=200, random_state=42
    )
    assert rows_a == rows_b


def test_segment_bootstrap_ci_empty_group_returns_empty_list() -> None:
    """An empty group Series returns an empty list, not an error."""
    rows = segment_bootstrap_ci(
        [], [], pd.Series([], dtype=object, name="g"), n_bootstrap=200, random_state=42
    )
    assert rows == []


# ---------------------------------------------------------------------------
# segment_decision_rates
# ---------------------------------------------------------------------------


def test_segment_decision_rates_keys() -> None:
    """Each row contains the six expected keys."""
    rng = np.random.default_rng(30)
    n = 100
    y_true = rng.integers(0, 2, size=n).tolist()
    proba = rng.random(size=n).tolist()
    group = pd.Series(rng.choice(["A", "B"], size=n), name="test_col")
    rows = segment_decision_rates(y_true, proba, group, threshold=0.5)
    assert len(rows) >= 1
    for row in rows:
        assert {
            "segment",
            "value",
            "n",
            "selection_rate",
            "fnr",
            "fpr",
            "precision",
        } <= set(row)


def test_segment_decision_rates_segment_name() -> None:
    """segment field matches the Series name."""
    rng = np.random.default_rng(31)
    n = 60
    y_true = rng.integers(0, 2, size=n).tolist()
    proba = rng.random(size=n).tolist()
    group = pd.Series(["X"] * 30 + ["Y"] * 30, name="my_col")
    rows = segment_decision_rates(y_true, proba, group, threshold=0.5)
    assert all(r["segment"] == "my_col" for r in rows)


def test_segment_decision_rates_skips_small_groups() -> None:
    """Groups with fewer than 10 samples are excluded."""
    rng = np.random.default_rng(32)
    n_common, n_rare = 80, 5
    n = n_common + n_rare
    y_true = rng.integers(0, 2, size=n).tolist()
    proba = rng.random(size=n).tolist()
    group = pd.Series(["common"] * n_common + ["rare"] * n_rare, name="g")
    rows = segment_decision_rates(y_true, proba, group, threshold=0.5)
    values = [r["value"] for r in rows]
    assert "rare" not in values
    assert "common" in values


def test_segment_decision_rates_selection_rate_matches_manual_count() -> None:
    """selection_rate equals the fraction of the segment scored >= threshold."""
    y_true = [0] * 20
    proba = [0.1] * 10 + [0.9] * 10  # exactly half above threshold
    group = pd.Series(["seg"] * 20, name="g")
    rows = segment_decision_rates(y_true, proba, group, threshold=0.5)
    assert rows[0]["selection_rate"] == pytest.approx(0.5)


def test_segment_decision_rates_all_negatives_selected_gives_fpr_one_fnr_nan() -> None:
    """An all-negative segment scored entirely above threshold has FPR=1.0 and an
    undefined (NaN) FNR — there are no positives to miss."""
    y_true = [0] * 10
    proba = [0.9] * 10
    group = pd.Series(["seg"] * 10, name="g")
    rows = segment_decision_rates(y_true, proba, group, threshold=0.5)
    assert rows[0]["fpr"] == pytest.approx(1.0)
    assert math.isnan(rows[0]["fnr"])


def test_segment_decision_rates_all_positives_missed_gives_fnr_one_fpr_nan() -> None:
    """An all-positive segment scored entirely below threshold has FNR=1.0 and an
    undefined (NaN) FPR — there are no negatives to falsely flag."""
    y_true = [1] * 10
    proba = [0.1] * 10
    group = pd.Series(["seg"] * 10, name="g")
    rows = segment_decision_rates(y_true, proba, group, threshold=0.5)
    assert rows[0]["fnr"] == pytest.approx(1.0)
    assert math.isnan(rows[0]["fpr"])


def test_segment_decision_rates_precision_nan_when_nothing_selected() -> None:
    """Precision is NaN, not a divide-by-zero error, when no one in the segment is
    scored above threshold."""
    y_true = [0] * 5 + [1] * 5
    proba = [0.1] * 10
    group = pd.Series(["seg"] * 10, name="g")
    rows = segment_decision_rates(y_true, proba, group, threshold=0.5)
    assert math.isnan(rows[0]["precision"])


def test_segment_decision_rates_perfect_classifier_has_zero_error_rates() -> None:
    """A perfect classifier at the decision threshold has FNR=0, FPR=0, precision=1."""
    y_true = [0] * 10 + [1] * 10
    proba = [0.1] * 10 + [0.9] * 10
    group = pd.Series(["seg"] * 20, name="g")
    rows = segment_decision_rates(y_true, proba, group, threshold=0.5)
    assert rows[0]["fnr"] == pytest.approx(0.0)
    assert rows[0]["fpr"] == pytest.approx(0.0)
    assert rows[0]["precision"] == pytest.approx(1.0)


def test_segment_decision_rates_detects_selection_rate_gap_across_segments() -> None:
    """Two segments with different score distributions around the threshold show a
    visibly different selection rate — the demographic-parity-diff signal this
    function exists to surface."""
    y_true = [0] * 20 + [1] * 20
    proba_low_selected = [0.1] * 15 + [0.6] * 5  # segment A: mostly below threshold
    proba_high_selected = [0.6] * 5 + [0.9] * 15  # segment B: mostly above threshold
    y_a, y_b = y_true[:20], y_true[20:]
    group = pd.Series(["A"] * 20 + ["B"] * 20, name="g")
    rows = {
        r["value"]: r
        for r in segment_decision_rates(
            y_a + y_b, proba_low_selected + proba_high_selected, group, threshold=0.5
        )
    }
    assert rows["B"]["selection_rate"] > rows["A"]["selection_rate"] + 0.3


def test_segment_decision_rates_empty_group_returns_empty_list() -> None:
    """An empty group Series returns an empty list, not an error."""
    rows = segment_decision_rates(
        [], [], pd.Series([], dtype=object, name="g"), threshold=0.5
    )
    assert rows == []


# ---------------------------------------------------------------------------
# generalization_gap
# ---------------------------------------------------------------------------


def test_generalization_gap_keys() -> None:
    """The result contains the four expected keys."""
    result = generalization_gap([0.8, 0.82, 0.79], [0.7, 0.72, 0.68])
    assert set(result) == {
        "train_pr_auc_mean",
        "cv_pr_auc_mean",
        "gap_mean",
        "gap_std",
    }


def test_generalization_gap_larger_for_unregularized_model() -> None:
    """An overfit (unregularized) model shows a wider train-CV gap than a regularized one."""
    unregularized = generalization_gap(
        train_scores=[0.95, 0.96, 0.94], cv_scores=[0.65, 0.63, 0.66]
    )
    regularized = generalization_gap(
        train_scores=[0.75, 0.74, 0.76], cv_scores=[0.72, 0.71, 0.73]
    )
    assert unregularized["gap_mean"] > regularized["gap_mean"] > 0


def test_generalization_gap_zero_for_identical_scores() -> None:
    """Train and CV scores that match exactly give a zero gap."""
    result = generalization_gap([0.7, 0.71, 0.69], [0.7, 0.71, 0.69])
    assert result["gap_mean"] == 0.0


def test_generalization_gap_single_fold_std_is_zero() -> None:
    """A single fold has no spread — gap_std defaults to 0.0 rather than NaN."""
    result = generalization_gap([0.8], [0.7])
    assert result["gap_std"] == 0.0


# ---------------------------------------------------------------------------
# learning_curve_points
# ---------------------------------------------------------------------------


def test_learning_curve_points_one_row_per_size() -> None:
    """One summary row is returned per training size."""
    train_sizes = [50, 100, 200, 400]
    train_scores = [[0.9, 0.91], [0.88, 0.87], [0.85, 0.86], [0.83, 0.84]]
    cv_scores = [[0.5, 0.52], [0.6, 0.61], [0.68, 0.67], [0.7, 0.71]]
    rows = learning_curve_points(train_sizes, train_scores, cv_scores)
    assert len(rows) == 4


def test_learning_curve_points_preserves_size_order() -> None:
    """train_size values in the output match the input order (non-decreasing)."""
    train_sizes = [50, 100, 200, 400]
    train_scores = [[0.9], [0.88], [0.85], [0.83]]
    cv_scores = [[0.5], [0.6], [0.68], [0.7]]
    rows = learning_curve_points(train_sizes, train_scores, cv_scores)
    sizes = [row["train_size"] for row in rows]
    assert sizes == [50.0, 100.0, 200.0, 400.0]
    assert sizes == sorted(sizes)


def test_learning_curve_points_score_means_correct() -> None:
    """Per-size mean scores are computed correctly."""
    rows = learning_curve_points([100], [[0.8, 0.9]], [[0.6, 0.7]])
    assert rows[0]["train_pr_auc_mean"] == pytest.approx(0.85)
    assert rows[0]["cv_pr_auc_mean"] == pytest.approx(0.65)


def test_learning_curve_points_single_repeat_std_is_zero() -> None:
    """A single repeat per size has no spread — std defaults to 0.0 rather than NaN."""
    rows = learning_curve_points([100], [[0.8]], [[0.6]])
    assert rows[0]["train_pr_auc_std"] == 0.0
    assert rows[0]["cv_pr_auc_std"] == 0.0


def test_learning_curve_points_empty_input() -> None:
    """No training sizes returns an empty list, not an error."""
    assert learning_curve_points([], [], []) == []

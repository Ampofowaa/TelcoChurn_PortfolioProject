"""Unit tests for telco_churn.models.plots — pure plotting-data helpers (Phase 7)."""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import pytest

from telco_churn.models.plots import (
    classification_summary_points,
    decile_lift_table,
    pr_curve_points,
    reliability_diagram_bins,
    roc_curve_points,
)

# ---------------------------------------------------------------------------
# reliability_diagram_bins
# ---------------------------------------------------------------------------


def test_reliability_diagram_bins_uniform_keys() -> None:
    """Each row contains the five expected keys."""
    rng = np.random.default_rng(1)
    n = 200
    proba = rng.random(n).tolist()
    y_true = rng.integers(0, 2, size=n).tolist()
    rows = reliability_diagram_bins(proba, y_true, n_bins=10, strategy="uniform")
    assert len(rows) >= 1
    for row in rows:
        assert {
            "bin_lower",
            "bin_upper",
            "mean_predicted",
            "observed_frequency",
            "count",
        } <= set(row)


def test_reliability_diagram_bins_counts_sum_to_n() -> None:
    """Bin counts sum to the input size — no row is dropped or double-counted."""
    rng = np.random.default_rng(2)
    n = 150
    proba = rng.random(n).tolist()
    y_true = rng.integers(0, 2, size=n).tolist()
    rows = reliability_diagram_bins(proba, y_true, n_bins=8, strategy="uniform")
    assert sum(row["count"] for row in rows) == n


def test_reliability_diagram_bins_quantile_drops_empty_bins() -> None:
    """Quantile edges on a skewed vector may produce empty bins — dropped, not NaN."""
    proba = [0.01] * 90 + [0.99] * 10
    y_true = [0] * 90 + [1] * 10
    rows = reliability_diagram_bins(proba, y_true, n_bins=10, strategy="quantile")
    for row in rows:
        assert not math.isnan(row["observed_frequency"])
        assert row["count"] > 0


def test_reliability_diagram_bins_perfect_calibration_diagonal() -> None:
    """A perfectly calibrated fixture has mean_predicted ≈ observed_frequency in every bin."""
    rng = np.random.default_rng(3)
    n = 5000
    proba = rng.random(n)
    y_true = (rng.random(n) < proba).astype(int)
    rows = reliability_diagram_bins(
        proba.tolist(), y_true.tolist(), n_bins=10, strategy="uniform"
    )
    for row in rows:
        if row["count"] < 50:
            continue
        assert row["mean_predicted"] == pytest.approx(
            row["observed_frequency"], abs=0.08
        )


# ---------------------------------------------------------------------------
# pr_curve_points
# ---------------------------------------------------------------------------


def test_pr_curve_points_keys() -> None:
    """Each row contains the three expected keys."""
    rng = np.random.default_rng(4)
    n = 100
    y_true = rng.integers(0, 2, size=n).tolist()
    proba = rng.random(n).tolist()
    rows = pr_curve_points(y_true, proba)
    assert len(rows) >= 1
    for row in rows:
        assert {"precision", "recall", "threshold"} <= set(row)


def test_pr_curve_points_values_in_unit_range() -> None:
    """Precision and recall values lie in [0, 1]."""
    rng = np.random.default_rng(5)
    n = 100
    y_true = rng.integers(0, 2, size=n).tolist()
    proba = rng.random(n).tolist()
    rows = pr_curve_points(y_true, proba)
    for row in rows:
        assert 0.0 <= row["precision"] <= 1.0
        assert 0.0 <= row["recall"] <= 1.0


def test_pr_curve_points_recall_non_increasing_with_threshold() -> None:
    """As the implicit threshold rises, recall is non-increasing — a property
    of the PR curve's construction that a rendering bug would violate."""
    rng = np.random.default_rng(6)
    n = 200
    y_true = rng.integers(0, 2, size=n).tolist()
    proba = rng.random(n).tolist()
    rows = pr_curve_points(y_true, proba)
    order = sorted(range(len(rows)), key=lambda i: rows[i]["threshold"])
    recalls = [rows[i]["recall"] for i in order]
    for earlier, later in zip(recalls, recalls[1:], strict=False):
        assert later <= earlier + 1e-9


def test_pr_curve_points_perfect_separator_reaches_precision_one() -> None:
    """A perfect separator's curve includes a point at precision == 1.0."""
    y_true = [0] * 50 + [1] * 50
    proba = [0.0 + i * 0.001 for i in range(50)] + [0.9 + i * 0.001 for i in range(50)]
    rows = pr_curve_points(y_true, proba)
    assert any(row["precision"] == pytest.approx(1.0) for row in rows)


# ---------------------------------------------------------------------------
# roc_curve_points
# ---------------------------------------------------------------------------


def test_roc_curve_points_keys() -> None:
    """Each row contains the three expected keys."""
    rng = np.random.default_rng(7)
    n = 100
    y_true = rng.integers(0, 2, size=n).tolist()
    proba = rng.random(n).tolist()
    rows = roc_curve_points(y_true, proba)
    assert len(rows) >= 1
    for row in rows:
        assert {"fpr", "tpr", "threshold"} <= set(row)


def test_roc_curve_points_starts_near_origin_ends_near_corner() -> None:
    """The curve's first point is near (0, 0) and its last point is (1, 1)."""
    rng = np.random.default_rng(8)
    n = 200
    y_true = rng.integers(0, 2, size=n).tolist()
    proba = rng.random(n).tolist()
    rows = roc_curve_points(y_true, proba)
    assert rows[0]["fpr"] == pytest.approx(0.0)
    assert rows[0]["tpr"] == pytest.approx(0.0)
    assert rows[-1]["fpr"] == pytest.approx(1.0)
    assert rows[-1]["tpr"] == pytest.approx(1.0)


def test_roc_curve_points_fpr_and_tpr_non_decreasing() -> None:
    """Both fpr and tpr are non-decreasing along the curve, by construction."""
    rng = np.random.default_rng(9)
    n = 150
    y_true = rng.integers(0, 2, size=n).tolist()
    proba = rng.random(n).tolist()
    rows = roc_curve_points(y_true, proba)
    fprs = [row["fpr"] for row in rows]
    tprs = [row["tpr"] for row in rows]
    assert fprs == sorted(fprs)
    assert tprs == sorted(tprs)


# ---------------------------------------------------------------------------
# decile_lift_table
# ---------------------------------------------------------------------------


def test_decile_lift_table_returns_ten_rows_with_expected_keys() -> None:
    """Always ten deciles, each with the expected keys, regardless of n."""
    rng = np.random.default_rng(20)
    n = 237
    y_true = rng.integers(0, 2, size=n).tolist()
    proba = rng.random(n).tolist()
    rows = decile_lift_table(y_true, proba)
    assert len(rows) == 10
    expected_keys = {
        "decile",
        "count",
        "n_churners",
        "churn_rate",
        "mean_predicted",
        "lift",
        "cumulative_capture_rate",
        "perfect_capture_rate",
        "headroom",
    }
    for row in rows:
        assert expected_keys <= set(row)


def test_decile_lift_table_counts_sum_to_n() -> None:
    """Decile counts sum to the input size — array_split leaves no row out."""
    rng = np.random.default_rng(21)
    n = 237
    y_true = rng.integers(0, 2, size=n).tolist()
    proba = rng.random(n).tolist()
    rows = decile_lift_table(y_true, proba)
    assert sum(row["count"] for row in rows) == n


def test_decile_lift_table_decile_one_has_highest_mean_predicted() -> None:
    """Decile 1 is the highest-scored group — mean_predicted is non-increasing across deciles."""
    rng = np.random.default_rng(22)
    n = 300
    y_true = rng.integers(0, 2, size=n).tolist()
    proba = rng.random(n).tolist()
    rows = decile_lift_table(y_true, proba)
    means = [row["mean_predicted"] for row in rows]
    assert means == sorted(means, reverse=True)


def test_decile_lift_table_cumulative_capture_rate_reaches_one() -> None:
    """The last decile's cumulative capture rate accounts for every churner."""
    rng = np.random.default_rng(23)
    n = 300
    y_true = rng.integers(0, 2, size=n).tolist()
    proba = rng.random(n).tolist()
    rows = decile_lift_table(y_true, proba)
    assert rows[-1]["cumulative_capture_rate"] == pytest.approx(1.0)


def test_decile_lift_table_cumulative_capture_rate_non_decreasing() -> None:
    """Cumulative capture rate never decreases moving down the deciles."""
    rng = np.random.default_rng(24)
    n = 300
    y_true = rng.integers(0, 2, size=n).tolist()
    proba = rng.random(n).tolist()
    rows = decile_lift_table(y_true, proba)
    rates = [row["cumulative_capture_rate"] for row in rows]
    for earlier, later in zip(rates, rates[1:], strict=False):
        assert later >= earlier - 1e-12


def test_decile_lift_table_perfect_ranking_top_decile_captures_most_churners() -> None:
    """A perfect separator concentrates churners in decile 1, with high lift there."""
    y_true = [1] * 30 + [0] * 270
    proba = [0.9 + i * 0.0001 for i in range(30)] + [
        0.1 - i * 0.0001 for i in range(270)
    ]
    rows = decile_lift_table(y_true, proba)
    assert rows[0]["lift"] == pytest.approx(10.0, abs=0.5)
    assert rows[0]["cumulative_capture_rate"] == pytest.approx(1.0)


def test_decile_lift_table_no_churners_gives_nan_lift() -> None:
    """Zero churners in the sample gives NaN lift/capture rate, not a divide-by-zero error."""
    y_true = [0] * 100
    proba = list(np.linspace(0, 1, 100))
    rows = decile_lift_table(y_true, proba)
    for row in rows:
        assert math.isnan(row["lift"])
        assert math.isnan(row["cumulative_capture_rate"])
        assert math.isnan(row["perfect_capture_rate"])
        assert math.isnan(row["headroom"])


def test_decile_lift_table_perfect_capture_rate_reaches_one_by_last_decile() -> None:
    """A flawless ranker's ceiling always reaches 100% capture by decile 10."""
    rng = np.random.default_rng(25)
    n = 300
    y_true = rng.integers(0, 2, size=n).tolist()
    proba = rng.random(n).tolist()
    rows = decile_lift_table(y_true, proba)
    assert rows[-1]["perfect_capture_rate"] == pytest.approx(1.0)


def test_decile_lift_table_headroom_is_nonnegative() -> None:
    """No model ranks better than a perfect ranker — headroom never goes negative."""
    rng = np.random.default_rng(26)
    n = 300
    y_true = rng.integers(0, 2, size=n).tolist()
    proba = rng.random(n).tolist()
    rows = decile_lift_table(y_true, proba)
    for row in rows:
        assert row["headroom"] >= -1e-12


def test_decile_lift_table_perfect_ranking_has_zero_headroom_at_decile_one() -> None:
    """A perfect separator's own headroom is ~0 the moment it reaches full capture."""
    y_true = [1] * 30 + [0] * 270
    proba = [0.9 + i * 0.0001 for i in range(30)] + [
        0.1 - i * 0.0001 for i in range(270)
    ]
    rows = decile_lift_table(y_true, proba)
    assert rows[0]["headroom"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# classification_summary_points
# ---------------------------------------------------------------------------


def test_classification_summary_points_one_row_per_scenario() -> None:
    """One row per named threshold, in the keys expected downstream."""
    rng = np.random.default_rng(10)
    n = 200
    y_true = rng.integers(0, 2, size=n).tolist()
    proba = rng.random(n).tolist()
    thresholds = {"conservative": 0.3, "base": 0.5, "optimistic": 0.7}
    rows = classification_summary_points(y_true, proba, thresholds)
    assert len(rows) == 3
    assert {row["scenario"] for row in rows} == set(thresholds)
    expected_keys = {
        "scenario",
        "threshold",
        "tp",
        "fp",
        "fn",
        "tn",
        "tp_pct",
        "fn_pct",
        "fp_pct",
        "tn_pct",
        "precision",
        "recall",
        "f1",
        "n_churners",
        "n_non_churners",
        "contact_rate",
    }
    for row in rows:
        assert expected_keys <= set(row)


def test_classification_summary_points_confusion_counts_sum_correctly() -> None:
    """tp+fn equals n_churners and fp+tn equals n_non_churners for every scenario."""
    rng = np.random.default_rng(11)
    n = 300
    y_true = rng.integers(0, 2, size=n).tolist()
    proba = rng.random(n).tolist()
    rows = classification_summary_points(y_true, proba, {"base": 0.5})
    row = rows[0]
    assert cast(float, row["tp"]) + cast(float, row["fn"]) == row["n_churners"]
    assert cast(float, row["fp"]) + cast(float, row["tn"]) == row["n_non_churners"]


def test_classification_summary_points_row_normalized_percentages_sum_to_one() -> None:
    """tp_pct + fn_pct == 1 (row-normalised over actual churners); fp_pct + tn_pct == 1."""
    rng = np.random.default_rng(12)
    n = 300
    y_true = rng.integers(0, 2, size=n).tolist()
    proba = rng.random(n).tolist()
    rows = classification_summary_points(y_true, proba, {"base": 0.5})
    row = rows[0]
    assert cast(float, row["tp_pct"]) + cast(float, row["fn_pct"]) == pytest.approx(1.0)
    assert cast(float, row["fp_pct"]) + cast(float, row["tn_pct"]) == pytest.approx(1.0)


def test_classification_summary_points_perfect_classifier() -> None:
    """A perfect classifier at the decision threshold has precision=recall=f1=1.0."""
    y_true = [0] * 20 + [1] * 20
    proba = [0.1] * 20 + [0.9] * 20
    rows = classification_summary_points(y_true, proba, {"base": 0.5})
    row = rows[0]
    assert row["precision"] == pytest.approx(1.0)
    assert row["recall"] == pytest.approx(1.0)
    assert row["f1"] == pytest.approx(1.0)


def test_classification_summary_points_contact_rate_matches_manual_count() -> None:
    """contact_rate equals the fraction of rows scored >= threshold."""
    proba = [0.1] * 30 + [0.9] * 70
    y_true = [0] * 100
    rows = classification_summary_points(y_true, proba, {"base": 0.5})
    assert rows[0]["contact_rate"] == pytest.approx(0.7)


def test_classification_summary_points_precision_nan_when_nothing_contacted() -> None:
    """Precision is NaN, not a divide-by-zero error, when no row clears the threshold."""
    y_true = [0] * 10 + [1] * 10
    proba = [0.1] * 20
    rows = classification_summary_points(y_true, proba, {"base": 0.9})
    assert math.isnan(cast(float, rows[0]["precision"]))
    assert rows[0]["recall"] == pytest.approx(0.0)


def test_classification_summary_points_no_churners_gives_nan_recall() -> None:
    """Recall (and its pct fields) are NaN, not an error, when the sample has no positives."""
    y_true = [0] * 20
    proba = [0.1] * 10 + [0.9] * 10
    rows = classification_summary_points(y_true, proba, {"base": 0.5})
    assert math.isnan(cast(float, rows[0]["recall"]))
    assert math.isnan(cast(float, rows[0]["tp_pct"]))
    assert math.isnan(cast(float, rows[0]["fn_pct"]))


def test_classification_summary_points_empty_thresholds_returns_empty_list() -> None:
    """No scenarios requested returns an empty list, not an error."""
    y_true = [0, 1, 0, 1]
    proba = [0.2, 0.8, 0.3, 0.7]
    assert classification_summary_points(y_true, proba, {}) == []

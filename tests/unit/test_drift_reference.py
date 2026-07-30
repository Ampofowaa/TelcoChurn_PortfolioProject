"""Unit tests for telco_churn.models.drift_reference — pure drift-baseline builder (Phase 7)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from telco_churn.models.drift_reference import build_reference


def _synthetic_features(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """A small frame mixing FEATURE_SCHEMA numeric and categorical columns."""
    return pd.DataFrame(
        {
            "tenure": rng.uniform(0, 72, size=n),
            "monthlycharges": rng.uniform(18, 120, size=n),
            "gender": rng.choice(["Male", "Female"], size=n),
            "contract_type": rng.choice(
                ["Month-to-month", "One year", "Two year"], size=n
            ),
        }
    )


def test_numeric_bin_edges_partition_observed_range() -> None:
    """Numeric bin edges' first/last values equal the column's observed min/max."""
    rng = np.random.default_rng(0)
    n = 500
    features_df = _synthetic_features(n, rng)
    proba = rng.random(n)
    y = rng.integers(0, 2, size=n)

    reference = build_reference(features_df, proba, y)

    for col in ("tenure", "monthlycharges"):
        edges = reference["features"]["numeric"][col]["bin_edges"]
        assert edges[0] == pytest.approx(features_df[col].min())
        assert edges[-1] == pytest.approx(features_df[col].max())


def test_categorical_frequencies_sum_to_one() -> None:
    """Category frequencies for each categorical feature sum to 1."""
    rng = np.random.default_rng(1)
    n = 300
    features_df = _synthetic_features(n, rng)
    proba = rng.random(n)
    y = rng.integers(0, 2, size=n)

    reference = build_reference(features_df, proba, y)

    for col in ("gender", "contract_type"):
        freqs = reference["features"]["categorical"][col]["frequencies"]
        assert sum(freqs.values()) == pytest.approx(1.0)


def test_prevalence_matches_input() -> None:
    """The reference's prevalence equals y's mean exactly."""
    rng = np.random.default_rng(2)
    n = 200
    features_df = _synthetic_features(n, rng)
    proba = rng.random(n)
    y = rng.integers(0, 2, size=n)

    reference = build_reference(features_df, proba, y)

    assert reference["prevalence"] == pytest.approx(y.mean())


def test_unseen_category_at_check_time_does_not_raise() -> None:
    """A category absent from the reference is a plain dict-lookup miss, not an error."""
    rng = np.random.default_rng(3)
    n = 200
    features_df = _synthetic_features(n, rng)
    proba = rng.random(n)
    y = rng.integers(0, 2, size=n)

    reference = build_reference(features_df, proba, y)
    freqs = reference["features"]["categorical"]["contract_type"]["frequencies"]

    assert freqs.get("Never Heard Of This Plan", 0.0) == 0.0


def test_in_sample_and_out_of_sample_scores_yield_different_references() -> None:
    """The score reference must come from OOS probabilities: in-sample scores are measurably sharper.

    This is the load-bearing test for why build_reference takes oos_proba as an
    explicit argument rather than scoring features_df itself: a champion's
    in-sample scores cluster at the extremes near their true label, while
    honest out-of-sample scores over the same rows are noisier around 0.5.
    Baselining on the former would misrepresent the population the champion
    will actually see in production.
    """
    rng = np.random.default_rng(4)
    n = 500
    features_df = _synthetic_features(n, rng)
    y = rng.integers(0, 2, size=n)

    in_sample_proba = np.where(y == 1, 0.95, 0.05) + rng.normal(0, 0.01, size=n)
    in_sample_proba = np.clip(in_sample_proba, 0.0, 1.0)
    oos_proba = np.clip(0.5 + rng.normal(0, 0.08, size=n), 0.0, 1.0)

    in_sample_reference = build_reference(features_df, in_sample_proba, y)
    oos_reference = build_reference(features_df, oos_proba, y)

    in_sample_edges = in_sample_reference["score"]["bin_edges"]
    oos_edges = oos_reference["score"]["bin_edges"]

    assert in_sample_edges != pytest.approx(oos_edges)
    assert (max(in_sample_edges) - min(in_sample_edges)) > (
        max(oos_edges) - min(oos_edges)
    )

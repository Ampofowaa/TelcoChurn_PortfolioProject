"""Unit tests for telco_churn.models.calibration_metrics."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

import telco_churn.models.calibration_metrics as calibration_metrics

_N_BINS = 5

# ---------------------------------------------------------------------------
# expected_calibration_error
# ---------------------------------------------------------------------------


def test_expected_calibration_error_near_zero_when_calibrated() -> None:
    """A probability vector whose predicted mean matches the observed frequency
    in its single bin has ~zero ECE."""
    proba = np.full(20, 0.10)
    y = pd.Series([1] * 2 + [0] * 18)  # observed frequency 0.10, matches proba

    ece = calibration_metrics.expected_calibration_error(proba, y, _N_BINS, "uniform")

    assert ece == pytest.approx(0.0, abs=1e-9)


def test_expected_calibration_error_large_when_miscalibrated() -> None:
    """A confidently wrong vector (predicted 0.9, observed 0.1) has ECE ~= 0.8."""
    proba = np.full(20, 0.90)
    y = pd.Series([1] * 2 + [0] * 18)  # observed frequency 0.10, predicted 0.90

    ece = calibration_metrics.expected_calibration_error(proba, y, _N_BINS, "uniform")

    assert ece == pytest.approx(0.8, abs=1e-9)


def test_expected_calibration_error_constant_proba_detects_miscalibration() -> None:
    """A fully-constant, badly wrong probability vector must not report ~0 ECE.

    Regression guard: quantile binning of an all-identical p collapses
    np.unique's edges to a single point, and edges[0], edges[-1] = 0.0, 1.0
    on a 1-element array silently drops it to zero valid bins — skipping the
    summation loop entirely and returning 0.0 regardless of y. Flooring at
    one bin spanning [0, 1] is what makes this actually measure the gap.
    """
    proba = np.full(20, 0.10)
    y = pd.Series([1] * 18 + [0] * 2)  # observed frequency 0.90, predicted 0.10

    ece = calibration_metrics.expected_calibration_error(proba, y, _N_BINS, "quantile")

    assert ece == pytest.approx(0.8, abs=1e-9)


def test_expected_calibration_error_warns_on_quantile_bin_collapse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tied probabilities that collapse the quantile bin count below n_bins
    must be flagged, not silently absorbed into a smaller-than-configured ECE.
    """
    warning_mock = Mock()
    monkeypatch.setattr(calibration_metrics.logger, "warning", warning_mock)

    proba = np.full(20, 0.10)
    y = pd.Series([1] * 2 + [0] * 18)

    calibration_metrics.expected_calibration_error(proba, y, _N_BINS, "quantile")

    collapse_calls = [
        call
        for call in warning_mock.call_args_list
        if call.args[0] == "ece_bins_collapsed"
    ]
    assert len(collapse_calls) == 1
    assert collapse_calls[0].kwargs["configured_n_bins"] == _N_BINS
    assert collapse_calls[0].kwargs["effective_n_bins"] == 1


def test_expected_calibration_error_no_warning_without_collapse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Distinct, evenly-spread probabilities reach the configured bin count —
    no collapse warning should fire."""
    warning_mock = Mock()
    monkeypatch.setattr(calibration_metrics.logger, "warning", warning_mock)

    rng = np.random.default_rng(42)
    proba = rng.uniform(0.01, 0.99, size=500)
    y = pd.Series(rng.integers(0, 2, size=500))

    calibration_metrics.expected_calibration_error(proba, y, _N_BINS, "quantile")

    collapse_calls = [
        call
        for call in warning_mock.call_args_list
        if call.args[0] == "ece_bins_collapsed"
    ]
    assert collapse_calls == []


# ---------------------------------------------------------------------------
# murphy_decomposition
# ---------------------------------------------------------------------------


def test_murphy_decomposition_return_keys() -> None:
    """Result carries exactly the four documented keys."""
    rng = np.random.default_rng(10)
    n = 200
    proba = rng.uniform(0.05, 0.95, size=n)
    y = pd.Series(rng.integers(0, 2, size=n))
    result = calibration_metrics.murphy_decomposition(proba, y, _N_BINS, "uniform")
    assert set(result) == {
        "reliability",
        "resolution",
        "uncertainty",
        "brier_reconstructed",
    }


def test_murphy_decomposition_reconstructs_brier_to_tolerance() -> None:
    """The load-bearing test: reliability - resolution + uncertainty must
    reconstruct the directly-computed Brier score, to tolerance — the
    identity that proves the decomposition is implemented correctly rather
    than plausibly."""
    from sklearn.metrics import brier_score_loss

    rng = np.random.default_rng(11)
    n = 5000
    proba = rng.uniform(0.02, 0.98, size=n)
    y = pd.Series((rng.uniform(size=n) < proba).astype(int))

    result = calibration_metrics.murphy_decomposition(proba, y, _N_BINS, "uniform")
    direct_brier = float(brier_score_loss(y, proba))

    assert result["brier_reconstructed"] == pytest.approx(direct_brier, abs=0.01)


def test_murphy_decomposition_uncertainty_matches_base_rate_variance() -> None:
    """uncertainty equals o-bar(1 - o-bar) for the observed base rate — the
    term that depends only on y, never on the forecast."""
    proba = np.full(20, 0.5)
    y = pd.Series([1] * 8 + [0] * 12)  # base rate 0.4
    result = calibration_metrics.murphy_decomposition(proba, y, _N_BINS, "uniform")
    assert result["uncertainty"] == pytest.approx(0.4 * 0.6)


def test_murphy_decomposition_perfect_calibration_gives_near_zero_reliability() -> None:
    """A forecast whose bin means match their bins' observed frequencies has
    ~zero reliability (calibration error), regardless of resolution."""
    proba = np.array([0.1] * 100 + [0.9] * 100)
    y = pd.Series([1] * 10 + [0] * 90 + [1] * 90 + [0] * 10)  # matches proba exactly
    result = calibration_metrics.murphy_decomposition(proba, y, _N_BINS, "uniform")
    assert result["reliability"] == pytest.approx(0.0, abs=1e-9)


def test_murphy_decomposition_no_resolution_when_all_bins_match_base_rate() -> None:
    """A forecast that discriminates nothing (every bin's observed frequency
    equals the overall base rate) has zero resolution."""
    proba = np.array([0.2] * 50 + [0.8] * 50)
    # Both bins have the same 0.3 observed churn rate as the overall base rate.
    y = pd.Series(([1] * 15 + [0] * 35) + ([1] * 15 + [0] * 35))
    result = calibration_metrics.murphy_decomposition(proba, y, _N_BINS, "uniform")
    assert result["resolution"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# calibration_slope
# ---------------------------------------------------------------------------


def test_calibration_slope_perfectly_calibrated_returns_near_one() -> None:
    """y generated as Bernoulli(p) from the exact predicted probabilities —
    the Cox slope of a truly calibrated forecaster is 1.0, and its bootstrap
    CI should straddle it.
    """
    rng = np.random.default_rng(42)
    n = 4000
    p_true = rng.uniform(0.05, 0.95, size=n)
    y = rng.binomial(1, p_true)

    result = calibration_metrics.calibration_slope(
        pd.Series(y), p_true, n_bootstrap=200, random_state=42
    )

    assert result["slope"] == pytest.approx(1.0, abs=0.15)
    assert result["slope_ci_lower"] < result["slope"] < result["slope_ci_upper"]


def test_calibration_slope_overconfident_returns_less_than_one() -> None:
    """Predicted probabilities pushed toward 0/1 relative to the true
    generating probability are systematically overconfident — the defect the
    slope exists to catch — and must score below 1.0.
    """
    rng = np.random.default_rng(42)
    n = 4000
    p_true = rng.uniform(0.05, 0.95, size=n)
    y = rng.binomial(1, p_true)
    logit_true = np.log(p_true / (1 - p_true))
    overconfident_proba = 1.0 / (1.0 + np.exp(-2.0 * logit_true))

    result = calibration_metrics.calibration_slope(
        pd.Series(y), overconfident_proba, n_bootstrap=200, random_state=42
    )

    assert result["slope"] < 1.0


def test_calibration_slope_redraws_degenerate_bootstrap_resample() -> None:
    """A small, heavily-imbalanced sample makes a single-class bootstrap
    resample likely — sklearn's LogisticRegression can't fit one, and
    np.fromiter's eager consumption means one unlucky draw among many would
    otherwise crash the whole bootstrap. Must complete and return a full
    n_bootstrap-length CI instead.

    n=10 with one positive is chosen because it reliably provokes at least
    one degenerate draw across 200 bootstrap resamples at random_state=42
    (a resample missing the sole positive has ~35% probability per draw) —
    this is a regression guard for that redraw path, not merely a small-N
    smoke test.
    """
    y = np.array([1.0] + [0.0] * 9)
    p = np.array([0.6] + [0.2] * 9)

    result = calibration_metrics.calibration_slope(
        y, p, n_bootstrap=200, random_state=42
    )

    assert result["slope_ci_lower"] <= result["slope_ci_upper"]
    assert np.isfinite(result["slope_ci_lower"])
    assert np.isfinite(result["slope_ci_upper"])


def test_calibration_slope_raises_on_genuinely_single_class_input() -> None:
    """y_true with only one class can't fit even the point estimate — this
    must still fail fast (not hang retrying an unwinnable redraw), since no
    bootstrap resample of single-class data can ever be two-class either.
    """
    y = np.zeros(10)
    p = np.full(10, 0.2)

    with pytest.raises(ValueError, match="at least 2 classes"):
        calibration_metrics.calibration_slope(y, p, n_bootstrap=200, random_state=42)


def test_calibration_slope_analytic_ci_well_ordered() -> None:
    """The analytic (Wald) CI is centered on the point estimate with a
    strictly positive standard error, regardless of the bootstrap draw —
    a basic sanity check that holds before comparing it to anything else.
    """
    rng = np.random.default_rng(0)
    n = 1000
    p_true = rng.uniform(0.05, 0.95, size=n)
    y = rng.binomial(1, p_true)

    result = calibration_metrics.calibration_slope(
        pd.Series(y), p_true, n_bootstrap=50, random_state=42
    )

    assert result["slope_se_analytic"] > 0.0
    assert (
        result["slope_ci_lower_analytic"]
        < result["slope"]
        < result["slope_ci_upper_analytic"]
    )


def test_calibration_slope_analytic_matches_finite_difference_hessian() -> None:
    """Independent proof the Fisher-information formula is implemented
    correctly: the analytic standard error must match a numerical
    finite-difference Hessian of the same logistic log-likelihood, computed
    with no shared code path — this is what makes the analytic CI trustworthy
    as a cross-check on the bootstrap, rather than a second thing to doubt.
    """
    rng = np.random.default_rng(0)
    n = 4000
    p_true = rng.uniform(0.05, 0.95, size=n)
    y = rng.binomial(1, p_true)

    result = calibration_metrics.calibration_slope(
        y, p_true, n_bootstrap=10, random_state=42
    )
    slope, intercept = result["slope"], result["intercept"]

    p = np.clip(p_true, 1e-6, 1 - 1e-6)
    logit_p = np.log(p / (1 - p))

    def _neg_loglik(beta: np.ndarray) -> float:
        b0, b1 = beta
        eta = b0 + b1 * logit_p
        return float(-np.sum(y * eta - np.log1p(np.exp(eta))))

    eps = 1e-4
    beta0 = np.array([intercept, slope])
    hess = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            e_i = np.zeros(2)
            e_i[i] = eps
            e_j = np.zeros(2)
            e_j[j] = eps
            hess[i, j] = (
                _neg_loglik(beta0 + e_i + e_j)
                - _neg_loglik(beta0 + e_i - e_j)
                - _neg_loglik(beta0 - e_i + e_j)
                + _neg_loglik(beta0 - e_i - e_j)
            ) / (4 * eps * eps)

    se_finite_diff = float(np.sqrt(np.linalg.inv(hess)[1, 1]))

    assert result["slope_se_analytic"] == pytest.approx(se_finite_diff, rel=1e-4)


def test_calibration_slope_analytic_and_bootstrap_ci_agree_on_well_behaved_data() -> (
    None
):
    """On a large, well-behaved sample, the analytic and bootstrap CIs should
    closely agree — this is the actual cross-check value: close agreement is
    cheap evidence the percentile bootstrap isn't misbehaving on this kind of
    data, since nothing had independently verified that before this test.
    """
    rng = np.random.default_rng(0)
    n = 4000
    p_true = rng.uniform(0.05, 0.95, size=n)
    y = rng.binomial(1, p_true)

    result = calibration_metrics.calibration_slope(
        y, p_true, n_bootstrap=1000, random_state=42
    )

    bootstrap_width = result["slope_ci_upper"] - result["slope_ci_lower"]
    analytic_width = (
        result["slope_ci_upper_analytic"] - result["slope_ci_lower_analytic"]
    )
    bootstrap_center = (result["slope_ci_upper"] + result["slope_ci_lower"]) / 2
    analytic_center = (
        result["slope_ci_upper_analytic"] + result["slope_ci_lower_analytic"]
    ) / 2

    assert analytic_width == pytest.approx(bootstrap_width, rel=0.25)
    assert analytic_center == pytest.approx(bootstrap_center, abs=0.02)

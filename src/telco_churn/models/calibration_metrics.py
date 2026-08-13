"""Pure calibration statistics — no cfg, no MLflow, no telco_churn imports.

Shared by calibrate.py (dev-OOF method selection), evaluate.py (sealed-test
calibration report), and diagnostics.py (sliced calibration). Every function
here takes arrays/scalars in and returns a value or a dict of values; callers
own resolving cfg.calibration.ece_n_bins/ece_strategy into n_bins/strategy
before calling in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from telco_churn.utils.logging import get_logger

__all__ = [
    "pooled_brier",
    "brier_skill_score",
    "expected_calibration_error",
    "murphy_decomposition",
    "calibration_slope",
]

logger = get_logger(__name__)

# calibration_slope's bootstrap redraw ceiling per resample (see its docstring):
# generous relative to the actual retry counts a real minority class needs, so
# it only ever bites the true "not enough of one class to bootstrap at all"
# case, not a merely-rare-but-viable one.
_MAX_BOOTSTRAP_RESAMPLE_ATTEMPTS = 1000


def pooled_brier(proba: NDArray[np.float64], y_dev: pd.Series) -> float:
    """Pooled Brier score on an outer-OOF probability vector.

    Legitimate to pool, unlike PR-AUC: Brier is a per-row proper score, so
    pooling doesn't mix distinct ranking structures the way pooling AP does.
    """
    return float(brier_score_loss(y_dev, proba))


def brier_skill_score(candidate_brier: float, reference_brier: float) -> float:
    """BSS = 1 - Brier_candidate / Brier_reference.

    Raw pooled Brier is hard to read alone — its scale is set by the class
    base rate (Murphy's decomposition: Brier = reliability - resolution +
    uncertainty, and uncertainty = p(1-p) is fixed by the data, not the
    model), so the same Brier value means something different at a different
    prevalence. BSS reframes it as skill over a reference forecast: 0 means
    no better than the reference, 1 means perfect. reference_brier is
    DummyClassifier(strategy='prior')'s pooled Brier — recomputed per call
    rather than hardcoded, since it depends on y_dev's actual prevalence.
    """
    return 1.0 - candidate_brier / reference_brier


def expected_calibration_error(
    proba: NDArray[np.float64], y_dev: pd.Series, n_bins: int, strategy: str, label: str
) -> float:
    """Expected Calibration Error: weighted mean |predicted − observed| across bins.

    n_bins/strategy are the caller's resolved cfg.calibration.ece_n_bins/
    ece_strategy — pinned so the number is comparable across calibration
    runs. ECE gates nothing here, it is logged purely so a loser's numbers
    make a winner's selection legible.

    label identifies the distribution being binned (e.g. "dummy_prior",
    "sigmoid", "segment:gender=Female") purely for the collapse warning
    below — every call site passes one so a collapse is attributable
    without cross-referencing calibration_summary.json by hand.

    Under strategy="quantile", tied probabilities can collapse the requested
    bin count (np.unique drops duplicate quantile edges) — logged as a
    warning rather than left silent, since a smaller effective bin count
    breaks the cross-run comparability this pinning exists for. Full
    collapse (every probability identical) is floored at one bin spanning
    [0, 1] rather than zero: zero bins would skip the summation loop
    entirely and return 0.0 regardless of y — silently reporting "perfectly
    calibrated" for what could be a confidently wrong constant prediction.
    """
    y = y_dev.to_numpy()
    p = np.asarray(proba)

    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1)))
        if len(edges) < 2:
            edges = np.array([0.0, 1.0])
        n_bins_effective = len(edges) - 1
        if n_bins_effective < n_bins:
            # np.unique silently drops duplicate quantile edges when many
            # rows share the same p (coarse leaf outputs, a near-constant
            # score) — the configured n_bins is what makes ECE comparable
            # across calibration runs, so a silent drop below it must be
            # visible, not just absorbed into a smaller ECE.
            logger.warning(
                "ece_bins_collapsed",
                label=label,
                configured_n_bins=n_bins,
                effective_n_bins=n_bins_effective,
                hint=(
                    "quantile bin edges collapsed below the configured "
                    "ece_n_bins because many probabilities are tied — this "
                    "ECE is not comparable to a run where collapse didn't "
                    "happen; investigate why so many predictions are tied "
                    "before comparing across runs."
                ),
            )
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    edges[0], edges[-1] = 0.0, 1.0

    bin_ids = np.clip(np.digitize(p, edges[1:-1], right=True), 0, len(edges) - 2)
    n = len(p)
    ece = 0.0
    for b in range(len(edges) - 1):
        mask = bin_ids == b
        if not mask.any():
            continue
        ece += (
            float(mask.sum()) / n * abs(float(y[mask].mean()) - float(p[mask].mean()))
        )
    return float(ece)


def murphy_decomposition(
    proba: NDArray[np.float64], y_dev: pd.Series, n_bins: int, strategy: str
) -> dict[str, float]:
    """Murphy's three-term decomposition: Brier = reliability - resolution + uncertainty.

    n_bins/strategy are the caller's resolved cfg.calibration.ece_n_bins/
    ece_strategy — binned identically to expected_calibration_error, so the
    two diagnostics are read off the same bins rather than two
    independently-tuned schemes. Per bin k (mean forecast p_k, observed
    frequency o_k, count n_k) and overall base rate o-bar:
      reliability = (1/N) sum_k n_k (p_k - o_k)^2   -- calibration error
      resolution  = (1/N) sum_k n_k (o_k - o-bar)^2 -- discrimination ability
      uncertainty = o-bar (1 - o-bar)               -- outcome variance alone

    This is what makes a Brier movement attributable rather than a single
    opaque number: Brier blends calibration quality with ranking quality, so
    a model can improve its Brier purely by improving resolution (better
    ranking) while reliability (calibration) gets worse — exactly the failure
    the calibration-slope guardrail exists to catch independently (ANALYSIS.md
    §0). reliability/resolution/uncertainty reconstruct the directly-computed
    Brier only approximately for continuous forecasts (the identity is exact
    for forecasts that are literally constant within each bin); the gap
    shrinks as n_bins grows and is a discretization artifact, not a bug.
    """
    y = y_dev.to_numpy()
    p = np.asarray(proba)
    n = len(p)
    base_rate = float(y.mean())

    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1)))
        if len(edges) < 2:
            edges = np.array([0.0, 1.0])
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    edges[0], edges[-1] = 0.0, 1.0
    bin_ids = np.clip(np.digitize(p, edges[1:-1], right=True), 0, len(edges) - 2)

    reliability = 0.0
    resolution = 0.0
    for b in range(len(edges) - 1):
        mask = bin_ids == b
        if not mask.any():
            continue
        n_k = float(mask.sum())
        p_k = float(p[mask].mean())
        o_k = float(y[mask].mean())
        reliability += n_k / n * (p_k - o_k) ** 2
        resolution += n_k / n * (o_k - base_rate) ** 2

    uncertainty = base_rate * (1.0 - base_rate)
    return {
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
        "brier_reconstructed": float(reliability - resolution + uncertainty),
    }


def calibration_slope(
    y_true: pd.Series | NDArray[np.float64],
    proba: NDArray[np.float64],
    n_bootstrap: int,
    random_state: int,
) -> dict[str, float]:
    """Cox calibration slope: unpenalized logistic regression of y on logit(p).

    Perfect calibration -> slope 1.0; slope < 1 means systematically
    overconfident (predicted probabilities more extreme than observed
    frequencies warrant). Binning-free and parameter-free, unlike ECE, and it
    cannot be bought by better ranking the way Brier can (Murphy's
    decomposition blends calibration with resolution; this doesn't) — which
    is what makes it gate-worthy where Brier alone is not.

    p is clipped away from {0, 1} before the logit transform to avoid
    infinities. A percentile bootstrap (resampling rows with replacement,
    refitting the slope each draw) reports the estimate's sampling
    uncertainty — this is the same function §0's sealed-test gate and the
    dev-OOF calibration screen both call; only the (y_true, proba) pair and
    n_bootstrap differ between those two call sites.

    A resample that happens to miss one class entirely can't fit a logistic
    regression (sklearn raises) — an artifact of resampling a small or
    imbalanced set, not a property of the real data, since the point
    estimate above already proved both classes exist in y_true. Such a draw
    is redrawn rather than allowed to crash the whole bootstrap, up to
    _MAX_BOOTSTRAP_RESAMPLE_ATTEMPTS retries per draw — this only matters
    at call sites smaller than today's full dev-OOF partition (a fairness
    slice, a smaller sealed-test slice), where a single unlucky draw among
    many bootstrap iterations shouldn't be able to take down the whole
    calibration screen.

    Also reports an analytic (Wald) CI: the asymptotic covariance of the
    logistic-regression MLE is (X^T W X)^-1, W = diag(p_hat(1-p_hat))
    evaluated at the auxiliary regression's own fitted probabilities — a
    closed form, no resampling. This is a cross-check on the bootstrap CI,
    not a replacement for it: a percentile bootstrap has no guaranteed
    coverage in general (small samples or a skewed sampling distribution can
    make it misbehave), and nothing in this project had independently
    verified it behaves well *here* until this was added. Close agreement
    between the two is cheap evidence the bootstrap isn't misbehaving on this
    data; the bootstrap CI remains the one the gate reads.

    Returns {"slope", "intercept", "slope_ci_lower", "slope_ci_upper",
    "slope_se_analytic", "slope_ci_lower_analytic", "slope_ci_upper_analytic"}.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
    logit_p = np.log(p / (1 - p)).reshape(-1, 1)

    def _fit(
        y_arr: NDArray[np.float64], x_arr: NDArray[np.float64]
    ) -> tuple[float, float]:
        # C=1e10 (not np.inf, not penalty=None): sklearn 1.8 deprecates
        # penalty=None, and its own suggested replacement, C=np.inf, still
        # trips an internal migration shim that re-derives penalty=None from
        # C == np.inf and warns "Setting penalty=None will ignore the C and
        # l1_ratio parameters" — a false positive on every one of up to
        # n_bootstrap calls here. A merely-huge finite C sidesteps that shim
        # entirely (verified: no warnings, and coefficients agree with a
        # true C=np.inf fit to ~1e-9 relative error on non-separable data).
        # It also strictly dominates C=np.inf numerically: L2 regularization
        # keeps the objective strictly convex even under perfect separation
        # in a bootstrap resample, so there's always a unique finite optimum
        # for lbfgs to converge to — where true C=np.inf has no finite
        # optimum at all under separation, and a ConvergenceWarning there
        # would be reporting a genuinely ill-posed fit, not a false one.
        model = LogisticRegression(C=1e10, l1_ratio=0)
        model.fit(x_arr, y_arr)
        return float(model.coef_[0][0]), float(model.intercept_[0])

    slope, intercept = _fit(y, logit_p)

    # Analytic Wald CI — (X^T W X)^-1 is the standard logistic-regression
    # asymptotic covariance (the same quantity statsmodels/GLM report as
    # each coefficient's standard error); index 1 is the slope term, index 0
    # the intercept.
    design = np.hstack([np.ones((len(y), 1)), logit_p])
    fitted_logit = intercept + slope * logit_p.ravel()
    p_hat = 1.0 / (1.0 + np.exp(-fitted_logit))
    w = p_hat * (1.0 - p_hat)
    fisher_info = design.T @ (design * w.reshape(-1, 1))
    cov = np.linalg.inv(fisher_info)
    slope_se_analytic = float(np.sqrt(cov[1, 1]))
    z_975 = 1.959963984540054  # scipy.stats.norm.ppf(0.975), inlined to avoid a scipy dependency
    slope_ci_lower_analytic = slope - z_975 * slope_se_analytic
    slope_ci_upper_analytic = slope + z_975 * slope_se_analytic

    rng = np.random.default_rng(random_state)
    n = len(y)

    def _two_class_resample_idx() -> NDArray[np.intp]:
        for _ in range(_MAX_BOOTSTRAP_RESAMPLE_ATTEMPTS):
            idx = rng.integers(0, n, size=n)
            if len(np.unique(y[idx])) >= 2:
                return idx
        raise RuntimeError(
            "calibration_slope: could not draw a two-class bootstrap "
            f"resample in {_MAX_BOOTSTRAP_RESAMPLE_ATTEMPTS} attempts — the "
            "minority class is too rare relative to n for this bootstrap "
            "to be meaningful."
        )

    boot_slopes = np.fromiter(
        (
            _fit(y[idx], logit_p[idx])[0]
            for idx in (_two_class_resample_idx() for _ in range(n_bootstrap))
        ),
        dtype=float,
        count=n_bootstrap,
    )

    return {
        "slope": slope,
        "intercept": intercept,
        "slope_ci_lower": float(np.percentile(boot_slopes, 2.5)),
        "slope_ci_upper": float(np.percentile(boot_slopes, 97.5)),
        "slope_se_analytic": slope_se_analytic,
        "slope_ci_lower_analytic": slope_ci_lower_analytic,
        "slope_ci_upper_analytic": slope_ci_upper_analytic,
    }

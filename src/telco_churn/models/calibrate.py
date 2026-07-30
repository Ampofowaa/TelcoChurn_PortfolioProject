"""Calibrate the tuned pipeline logged by run_model_logging_step (models/train/log_model.py).

Calibration is cross-fit on the development set, so there is no separate
validation split. CalibratedClassifierCV(ensemble=False) wraps the pipeline
*unfitted*, so its ColumnTransformer refits inside every fold instead of
leaking preprocessing statistics across folds.

The unfitted pipeline comes from cloning the model at
manifest["logged_model_uri"] (a models:/m-<id> URI) — not from
runs:/<run_id>/model, which becomes ambiguous once this module logs its own
calibrated_model onto the same run.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import mlflow.artifacts
import mlflow.sklearn
import mlflow.tracking
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from numpy.typing import NDArray
from omegaconf import DictConfig
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from telco_churn.data.split import partition
from telco_churn.features.accessor import load_features
from telco_churn.features.build import TARGET_COL
from telco_churn.models.plots import reliability_diagram_bins
from telco_churn.utils.logging import get_logger
from telco_churn.utils.mlflow import (
    TRAINING_CYCLE_RUN_DESCRIPTION,
    ensure_experiment_metadata,
    resolve_tracking_uri,
    set_logged_model_description,
    set_registered_model_description,
    set_run_description,
)
from telco_churn.utils.paths import get_project_root
from telco_churn.utils.stats import paired_bootstrap_ci

__all__ = [
    "load_training_manifest",
    "committed_features_from_manifest",
    "unfitted_pipeline_from_manifest",
    "load_dev_features",
    "load_dev_customer_ids",
    "select_golden_rows",
    "outer_cv",
    "inner_cv",
    "build_calibrated_pipeline",
    "oof_uncalibrated_proba",
    "oof_calibrated_proba",
    "oof_dummy_proba",
    "per_fold_average_precision",
    "per_fold_brier",
    "pooled_brier",
    "brier_skill_score",
    "expected_calibration_error",
    "murphy_decomposition",
    "calibration_slope",
    "pr_auc_gate_passes",
    "brier_switch_decision",
    "select_calibration_method",
    "run_calibration_step",
]

logger = get_logger(__name__)

_MODEL_DESCRIPTION = (
    "Sigmoid-calibrated CalibratedClassifierCV wrapping the 'model' "
    "pipeline logged on this run. This is the artifact registered to the "
    "model registry and pointed at by 'challenger'/'champion'."
)

_REGISTRY_DESCRIPTION = (
    "Calibrated LightGBM churn-prediction pipeline for IBM Telco Customer "
    "Churn. Selected by PR-AUC among Dummy/LogReg/LightGBM candidates, "
    "calibrated (sigmoid), thresholded at a cost-sensitive cutoff "
    "t* = c/(r x LTV). 'champion' serves production; 'challenger' holds the "
    "most recent candidate. Every version carries a promotion_status tag "
    "(pending/promoted/rejected) - only 'promoted' versions are valid "
    "rollback targets."
)

_PENDING_VERSION_DESCRIPTION = "Awaiting sealed-test evaluation and promotion review."

# CalibratedClassifierCV(ensemble=...) — shared by build_calibrated_pipeline and
# the calibration_spec block logged alongside it, so the two can never drift
# apart into "what was fit" vs. "what the manifest says was fit".
_CALIBRATION_ENSEMBLE = False

# calibration_slope's bootstrap redraw ceiling per resample (see its docstring):
# generous relative to the actual retry counts a real minority class needs, so
# it only ever bites the true "not enough of one class to bootstrap at all"
# case, not a merely-rare-but-viable one.
_MAX_BOOTSTRAP_RESAMPLE_ATTEMPTS = 1000


# ---------------------------------------------------------------------------
# Step 1: build/wrap the calibrated pipeline
# ---------------------------------------------------------------------------


def load_training_manifest(run_id: str, cfg: DictConfig) -> dict[str, Any]:
    """Load run_model_logging_step's training_manifest.json artifact from the tuning_study run.

    Sets the MLflow tracking URI as a side effect, so this is safe to call as
    the first MLflow-touching call in a fresh process.
    """
    mlflow.set_tracking_uri(resolve_tracking_uri(str(cfg.mlflow.tracking_uri)))
    manifest: dict[str, Any] = mlflow.artifacts.load_dict(
        f"runs:/{run_id}/training_manifest.json"
    )
    return manifest


def committed_features_from_manifest(manifest: dict[str, Any]) -> list[str]:
    """Return run_selection_step's frozen input space — the columns the logged pipeline expects."""
    return list(manifest["feature_selection"]["model_features"])


def unfitted_pipeline_from_manifest(manifest: dict[str, Any]) -> Pipeline:
    """Return a fresh, unfitted clone of the pipeline run_model_logging_step logged.

    clone() strips the fitted state (LightGBM booster, fitted ColumnTransformer
    statistics) while preserving the exact construction spec — the single
    construction path the bridge's design contract requires; rebuilding the
    pipeline from best_params is the failure it forbids.
    """
    fitted = mlflow.sklearn.load_model(manifest["logged_model_uri"])
    pipeline: Pipeline = clone(fitted)
    return pipeline


def _load_dev_partition() -> pd.DataFrame:
    """Return the full development-partition rows (customerid included), pre feature-subsetting.

    Pure and deterministic over the static processed-features file, so calling
    it twice (once for load_dev_features, once for load_dev_customer_ids)
    reproduces the identical row order both times — the same
    recompute-rather-than-thread-state idiom outer_cv(cfg) already uses in
    this module.
    """
    df = load_features()
    dev_df, _test_df = partition(df)
    return dev_df


def load_dev_features(committed_features: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Load the development-partition rows, restricted to the frozen committed feature set."""
    dev_df = _load_dev_partition()
    return dev_df[committed_features], dev_df[TARGET_COL]


def load_dev_customer_ids() -> pd.Series:
    """Return the customerid Series for the development partition.

    Row-order-aligned with load_dev_features's (X_dev, y_dev) — both derive
    from the same _load_dev_partition() call — so it can be zipped
    positionally with an OOF probability vector computed over (X_dev, y_dev)
    to build the dev_oof_predictions.parquet artifact.
    """
    return _load_dev_partition()["customerid"].reset_index(drop=True)


def select_golden_rows(
    X_dev: pd.DataFrame, customer_ids: pd.Series, n_rows: int
) -> tuple[pd.DataFrame, pd.Series]:
    """Return the n_rows dev-partition rows with the lowest customerid, for the golden-parity fixture.

    Selected by customerid value, not row position, so the fixture register.py
    checks against is stable against an unrelated reordering of the processed
    feature table upstream — a positional .head(n) would silently pin a
    different set of customers if that ordering ever shifted. X_dev and
    customer_ids must already be row-order-aligned (as load_dev_features and
    load_dev_customer_ids are, both derived from the same
    _load_dev_partition() call), since selection is by position within the
    customerid sort order, not by pandas index label.
    """
    order = np.argsort(customer_ids.to_numpy())[:n_rows]
    golden_X = X_dev.iloc[order].reset_index(drop=True)
    golden_ids = customer_ids.iloc[order].reset_index(drop=True)
    return golden_X, golden_ids


def outer_cv(cfg: DictConfig) -> StratifiedKFold:
    """Return the outer StratifiedKFold shared by the uncalibrated baseline and every method.

    Built fresh from the same fixed parameters (n_splits, shuffle, random_state)
    on every call — with StratifiedKFold that's sufficient for identical fold
    indices given the same (X_dev, y_dev) ordering, so oof_uncalibrated_proba
    and every oof_calibrated_proba call are paired comparisons against the same
    folds, not independent resamples.
    """
    return StratifiedKFold(
        n_splits=int(cfg.calibration.outer_cv_folds),
        shuffle=bool(cfg.calibration.shuffle),
        random_state=int(cfg.calibration.random_state),
    )


def inner_cv(cfg: DictConfig) -> StratifiedKFold:
    """Return the inner StratifiedKFold passed to CalibratedClassifierCV(cv=...).

    Explicit shuffle+seed — the bare cv=5 default is unshuffled, and every
    split/sampler/model in this project is seeded for reproducibility.
    """
    return StratifiedKFold(
        n_splits=int(cfg.calibration.inner_cv_folds),
        shuffle=bool(cfg.calibration.shuffle),
        random_state=int(cfg.calibration.random_state),
    )


def build_calibrated_pipeline(
    pipeline: Pipeline, method: str, cfg: DictConfig
) -> CalibratedClassifierCV:
    """Wrap the unfitted pipeline in CalibratedClassifierCV(ensemble=False).

    ensemble=False collapses calibrated_classifiers_ to length 1, whose
    .estimator is the base Pipeline refit on all of development — the SHAP
    access path notebooks/05-error-analysis.ipynb depends on. pipeline must be
    unfitted;
    CalibratedClassifierCV.fit clones it internally per inner fold, so passing
    the same unfitted instance to two CalibratedClassifierCV objects (e.g. one
    per method) is safe without an extra clone() here.
    """
    return CalibratedClassifierCV(
        pipeline,
        method=method,
        cv=inner_cv(cfg),
        ensemble=_CALIBRATION_ENSEMBLE,
    )


def oof_uncalibrated_proba(
    pipeline: Pipeline, X_dev: pd.DataFrame, y_dev: pd.Series, cfg: DictConfig
) -> NDArray[np.float64]:
    """Return paired uncalibrated OOF positive-class probabilities over outer_cv(cfg).

    This is the baseline every calibration method's PR-AUC and Brier are
    compared against — cross_val_predict on the same outer StratifiedKFold
    object oof_calibrated_proba uses, so the comparison is paired or it is
    nothing.
    """
    proba = cross_val_predict(
        clone(pipeline), X_dev, y_dev, cv=outer_cv(cfg), method="predict_proba"
    )
    result: NDArray[np.float64] = proba[:, 1]
    return result


def oof_calibrated_proba(
    pipeline: Pipeline,
    method: str,
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    cfg: DictConfig,
) -> NDArray[np.float64]:
    """Return paired calibrated OOF positive-class probabilities for one method.

    cross_val_predict(CalibratedClassifierCV(pipeline, method=method,
    cv=inner_cv(cfg)), cv=outer_cv(cfg)) — outer_cv(cfg) builds a fresh
    StratifiedKFold with the same parameters as oof_uncalibrated_proba's, so
    the fold indices (not just the fold count) match exactly.
    """
    calibrated = build_calibrated_pipeline(pipeline, method, cfg)
    proba = cross_val_predict(
        calibrated, X_dev, y_dev, cv=outer_cv(cfg), method="predict_proba"
    )
    result: NDArray[np.float64] = proba[:, 1]
    return result


def oof_dummy_proba(
    X_dev: pd.DataFrame, y_dev: pd.Series, cfg: DictConfig
) -> NDArray[np.float64]:
    """Return paired OOF positive-class probabilities for DummyClassifier(strategy='prior').

    The Brier Skill Score reference. Ignores every feature by construction —
    each fold's prediction is that fold's own training-partition prevalence —
    cross_val_predict'd over the same outer_cv(cfg) folds as every other
    candidate, so 1 - Brier_candidate/Brier_dummy isolates genuine skill
    rather than a fold-composition artifact.
    """
    proba = cross_val_predict(
        DummyClassifier(strategy="prior"),
        X_dev,
        y_dev,
        cv=outer_cv(cfg),
        method="predict_proba",
    )
    result: NDArray[np.float64] = proba[:, 1]
    return result


# ---------------------------------------------------------------------------
# Step 2: select the calibration method (sigmoid vs. isotonic)
# ---------------------------------------------------------------------------


def per_fold_average_precision(
    proba: NDArray[np.float64], X_dev: pd.DataFrame, y_dev: pd.Series, cfg: DictConfig
) -> list[float]:
    """Mean-of-per-fold average precision for an OOF vector — never pooled.

    Pooling mixes each fold's tie structure (or, for a calibrated vector, each
    fold's distinct calibration map) into one ranking, which can mask a
    per-fold ranking regression that scoring each fold separately catches.
    Recomputes outer_cv(cfg)'s fold assignment rather than threading indices
    through the OOF computation — same params + same (X_dev, y_dev) ordering
    reproduces the identical folds cross_val_predict used to build proba.
    """
    cv = outer_cv(cfg)
    return [
        float(average_precision_score(y_dev.iloc[test_idx], proba[test_idx]))
        for _, test_idx in cv.split(X_dev, y_dev)
    ]


def per_fold_brier(
    proba: NDArray[np.float64], X_dev: pd.DataFrame, y_dev: pd.Series, cfg: DictConfig
) -> list[float]:
    """Per-outer-fold Brier score — the block-bootstrap unit for brier_switch_decision.

    One value per fold, not per row: rows inside a fold share a calibrator, so
    resampling folds (not rows) is what keeps the switch decision's bootstrap
    CI honest.
    """
    cv = outer_cv(cfg)
    return [
        float(brier_score_loss(y_dev.iloc[test_idx], proba[test_idx]))
        for _, test_idx in cv.split(X_dev, y_dev)
    ]


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
    proba: NDArray[np.float64], y_dev: pd.Series, cfg: DictConfig
) -> float:
    """Expected Calibration Error: weighted mean |predicted − observed| across bins.

    Binning is pinned in cfg.calibration.ece_n_bins/ece_strategy so the number
    is comparable across calibration runs — ECE gates nothing here, it is
    logged for both methods purely so the loser's numbers make the winner's
    selection legible.

    Under ece_strategy="quantile", tied probabilities can collapse the
    requested bin count (np.unique drops duplicate quantile edges) — logged
    as a warning rather than left silent, since a smaller effective bin
    count breaks the cross-run comparability this pinning exists for. Full
    collapse (every probability identical) is floored at one bin spanning
    [0, 1] rather than zero: zero bins would skip the summation loop
    entirely and return 0.0 regardless of y — silently reporting "perfectly
    calibrated" for what could be a confidently wrong constant prediction.
    """
    n_bins = int(cfg.calibration.ece_n_bins)
    strategy = str(cfg.calibration.ece_strategy)
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
    proba: NDArray[np.float64], y_dev: pd.Series, cfg: DictConfig
) -> dict[str, float]:
    """Murphy's three-term decomposition: Brier = reliability - resolution + uncertainty.

    Binned identically to expected_calibration_error (same
    cfg.calibration.ece_n_bins/ece_strategy), so the two diagnostics are read
    off the same bins rather than two independently-tuned schemes. Per bin k
    (mean forecast p_k, observed frequency o_k, count n_k) and overall base
    rate o-bar:
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
    n_bins = int(cfg.calibration.ece_n_bins)
    strategy = str(cfg.calibration.ece_strategy)
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
        # C=np.inf, not penalty=None: sklearn 1.8 deprecates penalty=None,
        # scheduled for removal in 1.10. Effectively unregularized either way.
        model = LogisticRegression(C=np.inf)
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


def pr_auc_gate_passes(
    per_fold_ap_candidate: list[float],
    per_fold_ap_uncalibrated: list[float],
    cfg: DictConfig,
) -> bool:
    """Hard gate: candidate's per-fold mean AP must not fall more than Δ* below uncalibrated.

    Applied before any Brier comparison — ranking degradation is a one-metric-
    invariant violation no Brier improvement buys back. Uses
    training_setup.delta_threshold (Δ*=0.005), the same materiality threshold
    the LightGBM-vs-LogReg family decision uses.
    """
    delta = float(np.mean(per_fold_ap_candidate)) - float(
        np.mean(per_fold_ap_uncalibrated)
    )
    return delta >= -float(cfg.training_setup.delta_threshold)


def brier_switch_decision(
    per_fold_brier_sigmoid: list[float],
    per_fold_brier_isotonic: list[float],
    cfg: DictConfig,
) -> dict[str, Any]:
    """Decide sigmoid vs. isotonic via a paired bootstrap CI on per-fold Brier.

    Sigmoid is the incumbent; isotonic must earn the switch. Reuses
    telco_churn.utils.stats.paired_bootstrap_ci — the same paired-bootstrap
    idiom the LightGBM-vs-LogReg family decision uses
    (models/train/comparison.py) — on Δ = mean(Brier_sigmoid) −
    mean(Brier_isotonic), so a positive Δ favours isotonic (lower Brier is
    better). No materiality threshold: Brier's scale has no analogue to
    Δ*=0.005 on PR-AUC, so the CI excluding zero is the whole decision.

    Three outcomes, mirroring bootstrap_comparison's lgbm_win/logreg_win/tie
    split: isotonic_win (CI entirely favours isotonic), sigmoid_win (CI
    entirely favours sigmoid), or tie (CI includes 0). Both sigmoid_win and
    tie keep sigmoid — isotonic must earn the switch, not the reverse — but
    are logged distinctly so downstream reporting can tell "isotonic was
    decisively worse" apart from "the result was inconclusive."
    """
    result = paired_bootstrap_ci(
        per_fold_brier_sigmoid,
        per_fold_brier_isotonic,
        n_bootstrap=int(cfg.calibration.brier_bootstrap_n_samples),
        random_state=int(cfg.calibration.random_state),
    )
    delta_obs = result["delta_obs"]
    ci_lower = result["delta_ci_lower"]
    ci_upper = result["delta_ci_upper"]

    if ci_lower > 0:
        method, decision_rule = "isotonic", "isotonic_win"
    elif ci_upper < 0:
        method, decision_rule = "sigmoid", "sigmoid_win"
    else:
        method, decision_rule = "sigmoid", "tie"

    return {
        "method": method,
        "decision_rule": decision_rule,
        "delta_brier_obs": round(delta_obs, 4),
        "delta_brier_ci_lower": round(ci_lower, 4),
        "delta_brier_ci_upper": round(ci_upper, 4),
        "n_bootstrap": result["n_bootstrap"],
    }


def select_calibration_method(
    pipeline: Pipeline, X_dev: pd.DataFrame, y_dev: pd.Series, cfg: DictConfig
) -> dict[str, Any]:
    """Resolve cfg.calibration.method to a concrete sigmoid/isotonic choice, with diagnostics.

    method: 'auto' — computes both methods, gates isotonic on PR-AUC first,
    and only runs the Brier bootstrap if it clears the gate. Meant to run
    once as a human-reviewed decision, then get pinned: left on 'auto' under
    continuous retraining, the method could flip on Brier noise and silently
    invalidate the derived threshold.

    method: 'sigmoid' | 'isotonic' pinned — the normal path once 'auto' has
    picked a winner. Still fits *both* methods' OOF every cycle — the same
    per-fold model-fit cost as 'auto' — so calibration_summary.json always
    carries both candidates' diagnostics, not just the pinned winner's; a
    notebook or ANALYSIS.md citing "isotonic scored X" is never citing a
    number this cycle didn't actually produce. What pinned mode skips is the
    sigmoid-vs-isotonic Brier-bootstrap switch test itself — a cheap
    resampling step, not additional fits — since deciding the method is not
    this branch's job. Only the pinned (configured) method's diagnostics
    gate: pr_auc_gate_passes runs against it and raises on regression, so a
    pinned method that regresses ranking on a later retrain fails loudly
    instead of registering silently. The unpinned method's diagnostics are
    informational only — never gated, never able to change `method` or
    `calibrated_proba`.

    Returns {"method", "diagnostics": {method_name: {...}}, "switch_decision",
    "uncalibrated_proba", "calibrated_proba"}. switch_decision is always present
    with the same six keys (method, decision_rule, delta_brier_obs,
    delta_brier_ci_lower, delta_brier_ci_upper, n_bootstrap) regardless of which
    branch produced it — a caller (run_calibration_step, any future
    calibration_summary.json reader) never has to branch on shape. Only the
    real Brier-bootstrap branch populates the delta_brier_*/n_bootstrap fields
    with real values; pinned mode (decision_rule="pinned") and the
    isotonic_disqualified_pr_auc_gate short-circuit leave them None, since no
    bootstrap ran. Every diagnostics entry has per_fold_mean_ap, pooled_brier,
    ece, bss — including a "dummy_prior" entry, the DummyClassifier(strategy=
    'prior') reference BSS is computed against (its own bss is trivially 0.0,
    included for auditability). calibrated_proba is always the winning
    method's OOF vector — not necessarily configured_method's, when 'auto'
    overrides it — so a caller rendering a reliability diagram from it shows
    the method that actually gets registered, not whichever was computed
    first.
    """
    configured_method = str(cfg.calibration.method)

    dummy_proba = oof_dummy_proba(X_dev, y_dev, cfg)
    dummy_per_fold_ap = per_fold_average_precision(dummy_proba, X_dev, y_dev, cfg)
    dummy_brier = pooled_brier(dummy_proba, y_dev)

    uncal_proba = oof_uncalibrated_proba(pipeline, X_dev, y_dev, cfg)
    uncal_per_fold_ap = per_fold_average_precision(uncal_proba, X_dev, y_dev, cfg)
    uncal_brier = pooled_brier(uncal_proba, y_dev)
    diagnostics: dict[str, dict[str, Any]] = {
        "dummy_prior": {
            "per_fold_mean_ap": float(np.mean(dummy_per_fold_ap)),
            "pooled_brier": dummy_brier,
            "ece": expected_calibration_error(dummy_proba, y_dev, cfg),
            "bss": brier_skill_score(dummy_brier, dummy_brier),
        },
        "uncalibrated": {
            "per_fold_mean_ap": float(np.mean(uncal_per_fold_ap)),
            "pooled_brier": uncal_brier,
            "ece": expected_calibration_error(uncal_proba, y_dev, cfg),
            "bss": brier_skill_score(uncal_brier, dummy_brier),
        },
    }

    if configured_method in ("sigmoid", "isotonic"):
        proba = oof_calibrated_proba(pipeline, configured_method, X_dev, y_dev, cfg)
        per_fold_ap = per_fold_average_precision(proba, X_dev, y_dev, cfg)
        candidate_brier = pooled_brier(proba, y_dev)
        diagnostics[configured_method] = {
            "per_fold_mean_ap": float(np.mean(per_fold_ap)),
            "pooled_brier": candidate_brier,
            "ece": expected_calibration_error(proba, y_dev, cfg),
            "bss": brier_skill_score(candidate_brier, dummy_brier),
        }
        if not pr_auc_gate_passes(per_fold_ap, uncal_per_fold_ap, cfg):
            raise ValueError(
                f"Pinned calibration method {configured_method!r} failed the PR-AUC "
                "gate against the uncalibrated baseline — its per-fold mean AP fell "
                "more than training_setup.delta_threshold below uncalibrated. Refusing "
                "to register a model with degraded ranking; re-run calibration.method="
                "'auto' to re-derive the method."
            )

        other_method = "isotonic" if configured_method == "sigmoid" else "sigmoid"
        other_proba = oof_calibrated_proba(pipeline, other_method, X_dev, y_dev, cfg)
        other_per_fold_ap = per_fold_average_precision(other_proba, X_dev, y_dev, cfg)
        other_brier = pooled_brier(other_proba, y_dev)
        diagnostics[other_method] = {
            "per_fold_mean_ap": float(np.mean(other_per_fold_ap)),
            "pooled_brier": other_brier,
            "ece": expected_calibration_error(other_proba, y_dev, cfg),
            "bss": brier_skill_score(other_brier, dummy_brier),
        }

        return {
            "method": configured_method,
            "diagnostics": diagnostics,
            "switch_decision": {
                "method": configured_method,
                "decision_rule": "pinned",
                "delta_brier_obs": None,
                "delta_brier_ci_lower": None,
                "delta_brier_ci_upper": None,
                "n_bootstrap": None,
            },
            "uncalibrated_proba": uncal_proba,
            "calibrated_proba": proba,
        }

    if configured_method != "auto":
        raise ValueError(
            f"Unknown calibration.method {configured_method!r}; expected "
            "'sigmoid', 'isotonic', or 'auto'."
        )

    sigmoid_proba = oof_calibrated_proba(pipeline, "sigmoid", X_dev, y_dev, cfg)
    sigmoid_per_fold_ap = per_fold_average_precision(sigmoid_proba, X_dev, y_dev, cfg)
    sigmoid_per_fold_brier = per_fold_brier(sigmoid_proba, X_dev, y_dev, cfg)
    sigmoid_brier = pooled_brier(sigmoid_proba, y_dev)
    diagnostics["sigmoid"] = {
        "per_fold_mean_ap": float(np.mean(sigmoid_per_fold_ap)),
        "pooled_brier": sigmoid_brier,
        "ece": expected_calibration_error(sigmoid_proba, y_dev, cfg),
        "bss": brier_skill_score(sigmoid_brier, dummy_brier),
    }
    if not pr_auc_gate_passes(sigmoid_per_fold_ap, uncal_per_fold_ap, cfg):
        raise ValueError(
            "Sigmoid calibration failed the PR-AUC gate against the uncalibrated "
            "baseline in calibration.method='auto' — its per-fold mean AP fell "
            "more than training_setup.delta_threshold below uncalibrated. Sigmoid "
            "is the fallback auto-mode ships whenever isotonic is disqualified, so "
            "this is refused rather than silently registering a model with "
            "degraded ranking; investigate the fold split or upstream data before "
            "retrying."
        )

    isotonic_proba = oof_calibrated_proba(pipeline, "isotonic", X_dev, y_dev, cfg)
    isotonic_per_fold_ap = per_fold_average_precision(isotonic_proba, X_dev, y_dev, cfg)
    isotonic_brier = pooled_brier(isotonic_proba, y_dev)
    diagnostics["isotonic"] = {
        "per_fold_mean_ap": float(np.mean(isotonic_per_fold_ap)),
        "pooled_brier": isotonic_brier,
        "ece": expected_calibration_error(isotonic_proba, y_dev, cfg),
        "bss": brier_skill_score(isotonic_brier, dummy_brier),
    }

    isotonic_eligible = pr_auc_gate_passes(isotonic_per_fold_ap, uncal_per_fold_ap, cfg)
    if not isotonic_eligible:
        logger.info(
            "isotonic_disqualified_pr_auc_gate",
            isotonic_per_fold_mean_ap=diagnostics["isotonic"]["per_fold_mean_ap"],
            uncalibrated_per_fold_mean_ap=diagnostics["uncalibrated"][
                "per_fold_mean_ap"
            ],
            delta_threshold=float(cfg.training_setup.delta_threshold),
        )
        return {
            "method": "sigmoid",
            "diagnostics": diagnostics,
            "switch_decision": {
                "method": "sigmoid",
                "decision_rule": "isotonic_disqualified_pr_auc_gate",
                "delta_brier_obs": None,
                "delta_brier_ci_lower": None,
                "delta_brier_ci_upper": None,
                "n_bootstrap": None,
            },
            "uncalibrated_proba": uncal_proba,
            "calibrated_proba": sigmoid_proba,
        }

    isotonic_per_fold_brier = per_fold_brier(isotonic_proba, X_dev, y_dev, cfg)
    switch_decision = brier_switch_decision(
        sigmoid_per_fold_brier, isotonic_per_fold_brier, cfg
    )
    logger.info(
        "calibration_method_selected",
        method=switch_decision["method"],
        decision_rule=switch_decision["decision_rule"],
        delta_brier_obs=switch_decision["delta_brier_obs"],
    )
    winning_proba = (
        sigmoid_proba if switch_decision["method"] == "sigmoid" else isotonic_proba
    )
    return {
        "method": switch_decision["method"],
        "diagnostics": diagnostics,
        "switch_decision": switch_decision,
        "uncalibrated_proba": uncal_proba,
        "calibrated_proba": winning_proba,
    }


# ---------------------------------------------------------------------------
# Step 3: register the training cycle's single deployable artifact
# ---------------------------------------------------------------------------


def _save_reliability_plot(
    uncalibrated_bins: list[dict[str, float]],
    calibrated_bins: list[dict[str, float]],
    uncalibrated_proba: NDArray[np.float64],
    calibrated_proba: NDArray[np.float64],
    method: str,
    path: Path,
) -> None:
    """Render and save the before/after reliability diagram, with a fixed-width
    density histogram of the raw OOF probabilities underneath.

    The histogram uses the raw probability arrays directly, not
    reliability_diagram_bins's quantile bins: quantile bins hold ~equal count
    by construction, so a histogram built from them is flat and uninformative
    regardless of the true score distribution's shape — confirmed visually
    before switching to this approach.
    """
    fig, (ax_reliability, ax_hist) = plt.subplots(
        2, 1, figsize=(6, 7.5), height_ratios=[3, 1], sharex=True
    )

    ax_reliability.plot(
        [0, 1], [0, 1], linestyle="--", color="gray", label="perfectly calibrated"
    )
    ax_reliability.plot(
        [b["mean_predicted"] for b in uncalibrated_bins],
        [b["observed_frequency"] for b in uncalibrated_bins],
        marker="o",
        color="C0",
        label="uncalibrated",
    )
    ax_reliability.plot(
        [b["mean_predicted"] for b in calibrated_bins],
        [b["observed_frequency"] for b in calibrated_bins],
        marker="o",
        color="C1",
        label=f"calibrated ({method})",
    )
    ax_reliability.set_xlim(0, 1)
    ax_reliability.set_ylim(0, 1)
    ax_reliability.set_ylabel("Observed frequency")
    ax_reliability.set_title("Reliability diagram — before vs. after calibration")
    ax_reliability.legend()

    hist_bins = np.linspace(0.0, 1.0, 21)
    ax_hist.hist(
        uncalibrated_proba, bins=hist_bins, color="C0", alpha=0.5, label="uncalibrated"
    )
    ax_hist.hist(
        calibrated_proba,
        bins=hist_bins,
        color="C1",
        alpha=0.5,
        label=f"calibrated ({method})",
    )
    ax_hist.set_xlabel("Predicted probability")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title(
        "Predicted probability distribution — before vs. after calibration"
    )
    ax_hist.legend()

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def run_calibration_step(run_id: str, cfg: DictConfig) -> dict[str, Any]:
    """Calibrate, select a method, and register the fitted pipeline as `challenger`.

    The training cycle's single registration point — run_model_logging_step
    only logs an uncalibrated pipeline and stops there.

    Aborts before fitting anything (raises RuntimeError) if
    tuning_summary.trial_count_below_threshold is true and
    cfg.calibration.override_trial_count_gate is not set — a data-quality
    gate on the tuning result, not a performance comparison.

    name="calibrated_model" is load-bearing: reusing name="model" would
    rebind runs:/<run_id>/model away from the uncalibrated pipeline, and
    would make a second run of this function wrap a CalibratedClassifierCV
    inside another one. serialization_format=cloudpickle is mandatory —
    mlflow's skops default rejects CalibratedClassifierCV's internals.

    Also renders and logs the pre/post-calibration reliability diagram
    (reports/figures/reliability_diagram.png, mirrored onto the run's
    figures/ artifacts) from the exact OOF vectors that decided the winning
    method — not a recomputation, so the picture matches the numbers in
    calibration_summary.json.

    Logs dev_oof_predictions.parquet (customerid, y_true, p_hat) — the
    winning method's OOF vector, persisted rather than left to be
    recomputed by a downstream consumer (Phase 7's drift reference,
    dev-OOF veto surface, and calibration screen). calibration_summary.json
    additionally carries calibration_slope (the dev-OOF Cox slope, with a
    bootstrap CI), uncalibrated_calibration_slope (the same statistic on
    the pre-calibration OOF vector, paired for comparison — a reliability
    diagram's bin-level scatter can't by itself distinguish genuine
    miscalibration from sampling noise, and this pairing is what turns that
    visual ambiguity into two comparable numbers), and calibration_spec
    (method/inner_cv_folds/random_state/ensemble — the four fields a future
    Phase 10 recalibration flow, not yet designed, would need to rebuild an
    identical CalibratedClassifierCV without re-deriving it).

    Also logs dev_brier/dev_bss/dev_ece/dev_per_fold_mean_ap/
    dev_calibration_slope/dev_calibration_slope_ci_lower/
    dev_calibration_slope_ci_upper/dev_uncalibrated_calibration_slope/
    dev_uncalibrated_calibration_slope_ci_lower/
    dev_uncalibrated_calibration_slope_ci_upper/dev_mean_p_hat_calibrated/
    dev_mean_p_hat_uncalibrated/dev_observed_churn_rate as MLflow metrics (not
    just JSON fields), so they are plottable as a series across retraining
    cycles. The calibrated CI bounds are what let a future retrain cycle's
    slope be read as "still inside band" vs. "genuinely drifted" without
    opening calibration_summary.json; the uncalibrated CI bounds complete the
    paired before/after comparison the notebook's own table already shows
    (both rows, both with a 95% CI) rather than leaving half of that pairing
    only in the JSON.

    reports/figures/reliability_diagram.png is overwritten on every call —
    it reflects whichever run executed this function most recently on this
    machine, not a specific run_id. A reader who needs to know which run a
    figure on disk corresponds to should cross-check that run's own
    calibration_summary.json rather than trusting the file's mtime; the
    MLflow-logged copy under that run's figures/ artifacts is the durable,
    run-pinned reference.

    Tags the newly registered version promotion_status=pending at mint time,
    before anything downstream (evaluate.py, error_analysis.py, register.py)
    can fail — so a crash anywhere in that chain leaves the version exactly
    where it started rather than untagged. Also logs golden_predictions.json
    (customerid-pinned dev rows, at the model's committed input schema, plus
    the in-memory fitted pipeline's reference scores on them, captured before
    log_model/pickling touches anything) — the independent reference
    register.py's serving-parity smoke check verifies against, and Phase 9's
    API parity test reuses.
    """
    manifest = load_training_manifest(run_id, cfg)

    tuning_summary = manifest.get("tuning_summary", {})
    trial_count_below_threshold = bool(
        tuning_summary.get("trial_count_below_threshold", False)
    )
    if trial_count_below_threshold and not bool(
        cfg.calibration.override_trial_count_gate
    ):
        logger.error(
            "calibration_registration_blocked_trial_count",
            run_id=run_id,
            min_completed_trials=tuning_summary.get("min_completed_trials"),
            n_completed_trials=tuning_summary.get("n_completed_trials"),
        )
        raise RuntimeError(
            "Refusing to register: training_manifest.json's tuning_summary."
            "trial_count_below_threshold is true (too few completed Optuna "
            "trials to trust the 1-SE pick). Set "
            "calibration.override_trial_count_gate=true to force registration "
            "anyway."
        )

    committed_features = committed_features_from_manifest(manifest)
    pipeline = unfitted_pipeline_from_manifest(manifest)
    X_dev, y_dev = load_dev_features(committed_features)

    selection = select_calibration_method(pipeline, X_dev, y_dev, cfg)
    method = str(selection["method"])

    n_bins = int(cfg.calibration.ece_n_bins)
    strategy = str(cfg.calibration.ece_strategy)
    uncalibrated_bins = reliability_diagram_bins(
        selection["uncalibrated_proba"], y_dev.to_numpy(), n_bins, strategy
    )
    calibrated_bins = reliability_diagram_bins(
        selection["calibrated_proba"], y_dev.to_numpy(), n_bins, strategy
    )
    figure_path = (
        get_project_root() / str(cfg.paths.figures) / "reliability_diagram.png"
    )
    _save_reliability_plot(
        uncalibrated_bins,
        calibrated_bins,
        selection["uncalibrated_proba"],
        selection["calibrated_proba"],
        method,
        figure_path,
    )

    fitted = build_calibrated_pipeline(pipeline, method, cfg)
    fitted.fit(X_dev, y_dev)

    dev_customer_ids = load_dev_customer_ids()
    input_example, golden_customer_ids = select_golden_rows(
        X_dev, dev_customer_ids, int(cfg.calibration.golden_n_rows)
    )
    in_memory_preds = fitted.predict_proba(input_example)
    signature = infer_signature(X_dev, fitted.predict_proba(X_dev))

    # The reference for register.py's serving-parity smoke check, captured
    # here — while `fitted` is still the live in-memory object this run just
    # produced, before mlflow.sklearn.log_model/pickling touches it at all.
    # register.py reloads this same model later, in a separate process, and
    # must compare against a reference computed independently of that reload
    # — a reference generated by scoring through the reload itself would be
    # circular: any serialization bug would already be baked into the
    # "expected" value, and the check could never fail no matter how broken
    # the round trip was. Rows are stored in full (not just scores), pinned
    # by customerid, at the model's committed input schema, so the fixture is
    # self-contained and Phase 9/11 can replay it without touching the
    # dataset. Every golden row is a development-partition row the model
    # trained on — in-sample, and therefore reproducibility evidence only,
    # never performance evidence; the "purpose" key travels that caveat with
    # the artifact rather than leaving it to live only in a plan document.
    golden_fixture = {
        "purpose": (
            "serving-parity fixture — reproducibility only; scores are "
            "in-sample and are not performance evidence"
        ),
        "customerid": golden_customer_ids.tolist(),
        "rows": input_example.to_dict(orient="records"),
        "p_hat": in_memory_preds[:, 1].tolist(),
    }

    # The vector that selected `method`, produced its BSS, and will validate
    # t* — computed already, above, as selection["calibrated_proba"]. Logged
    # as-is rather than recomputed downstream (CLAUDE.md § Persist the
    # evidence, not just the conclusion): a recompute is only sound while the
    # dataset is static and the folds are seeded, and silently diverges from
    # the vector this cycle's decisions actually rested on once either isn't
    # true (Phase 10's retraining). Segment/protected columns are joined
    # later, by Phase 7's evaluate.py, where the dev-partition feature
    # columns are in hand — this is the minimal vector.
    dev_oof_predictions = pd.DataFrame(
        {
            "customerid": dev_customer_ids,
            "y_true": y_dev.reset_index(drop=True),
            "p_hat": selection["calibrated_proba"],
        }
    )

    slope = calibration_slope(
        y_dev,
        selection["calibrated_proba"],
        n_bootstrap=int(cfg.calibration.slope_bootstrap_n_samples),
        random_state=int(cfg.calibration.random_state),
    )
    # Paired against the winning method's slope above, computed on the same
    # already-in-memory selection["uncalibrated_proba"] (no extra CV refit —
    # select_calibration_method already produced it). A reliability diagram
    # can only show bin-level scatter, which is ambiguous between "genuinely
    # miscalibrated" and "noisy bins, calibration is actually fine"; the
    # uncalibrated slope is what turns that ambiguity into a number: a
    # consistently one-directional diagram (the uncalibrated case) shows up
    # here as a slope measurably below 1, whereas the calibrated method's
    # bidirectional scatter shows up as a slope near 1 — confirming the
    # visual story instead of merely asserting it.
    uncalibrated_slope = calibration_slope(
        y_dev,
        selection["uncalibrated_proba"],
        n_bootstrap=int(cfg.calibration.slope_bootstrap_n_samples),
        random_state=int(cfg.calibration.random_state),
    )

    calibration_summary: dict[str, Any] = {
        "method": method,
        "diagnostics": selection["diagnostics"],
        "switch_decision": selection["switch_decision"],
        "calibration_slope": slope,
        "uncalibrated_calibration_slope": uncalibrated_slope,
        # Mean predicted probability, calibrated vs. uncalibrated, against the
        # observed dev-partition churn rate — the "calibration-in-the-large"
        # evidence a slope near 1 can hide: a large intercept (see
        # uncalibrated_calibration_slope above) shows up here as a mean
        # p_hat far from the observed rate, which is what a reliability
        # diagram's visible overconfidence is actually a symptom of, not the
        # slope. Persisted rather than left to a notebook's own .mean() call
        # (CLAUDE.md § Persist the evidence, not just the conclusion).
        "mean_p_hat_calibrated": float(np.mean(selection["calibrated_proba"])),
        "mean_p_hat_uncalibrated": float(np.mean(selection["uncalibrated_proba"])),
        "observed_churn_rate": float(y_dev.mean()),
        # The four fields that reconstruct the fitted CalibratedClassifierCV
        # from this run alone — a future Phase 10 recalibration flow (not yet
        # designed) would rebuild the identical estimator from this block
        # instead of re-deriving it, and its provenance spec hash is computed
        # over exactly these fields.
        "calibration_spec": {
            "method": method,
            "inner_cv_folds": int(cfg.calibration.inner_cv_folds),
            "random_state": int(cfg.calibration.random_state),
            "ensemble": _CALIBRATION_ENSEMBLE,
        },
    }

    ensure_experiment_metadata(cfg)
    registered_model_name = str(cfg.mlflow.registered_model_name)

    with mlflow.start_run(run_id=run_id):
        set_run_description(TRAINING_CYCLE_RUN_DESCRIPTION)
        mlflow.log_dict(calibration_summary, "calibration/calibration_summary.json")
        mlflow.log_artifact(str(figure_path), artifact_path="calibration/figures")

        # calibration_summary.json remains the audit record — these duplicate a
        # handful of its fields so BSS/ECE/Brier/slope are plottable as a series
        # across cycles (Phase 10), the same pattern Phase 5's candidates.py/
        # tuning.py/feature_freeze.py already follow and Phase 6 had skipped.
        winning_diagnostics = selection["diagnostics"][method]
        mlflow.log_metrics(
            {
                "dev_brier": winning_diagnostics["pooled_brier"],
                "dev_bss": winning_diagnostics["bss"],
                "dev_ece": winning_diagnostics["ece"],
                "dev_per_fold_mean_ap": winning_diagnostics["per_fold_mean_ap"],
                "dev_calibration_slope": slope["slope"],
                "dev_calibration_slope_ci_lower": slope["slope_ci_lower"],
                "dev_calibration_slope_ci_upper": slope["slope_ci_upper"],
                "dev_uncalibrated_calibration_slope": uncalibrated_slope["slope"],
                "dev_uncalibrated_calibration_slope_ci_lower": uncalibrated_slope[
                    "slope_ci_lower"
                ],
                "dev_uncalibrated_calibration_slope_ci_upper": uncalibrated_slope[
                    "slope_ci_upper"
                ],
                "dev_mean_p_hat_calibrated": calibration_summary[
                    "mean_p_hat_calibrated"
                ],
                "dev_mean_p_hat_uncalibrated": calibration_summary[
                    "mean_p_hat_uncalibrated"
                ],
                "dev_observed_churn_rate": calibration_summary["observed_churn_rate"],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            oof_path = Path(tmp_dir) / "dev_oof_predictions.parquet"
            dev_oof_predictions.to_parquet(oof_path, index=False)
            mlflow.log_artifact(str(oof_path), artifact_path="calibration")

        mlflow.log_dict(golden_fixture, "calibration/golden_predictions.json")

        model_info = mlflow.sklearn.log_model(
            sk_model=fitted,
            name="calibrated_model",
            signature=signature,
            input_example=input_example,
            pyfunc_predict_fn="predict_proba",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            registered_model_name=registered_model_name,
        )
        set_logged_model_description(model_info.model_id, _MODEL_DESCRIPTION)

    reloaded = mlflow.sklearn.load_model(model_info.model_uri)
    reload_preds = reloaded.predict_proba(input_example)
    parity_ok = bool(np.allclose(in_memory_preds, reload_preds, rtol=0, atol=1e-12))
    if not parity_ok:
        raise AssertionError(
            "Reload parity check failed: predictions from the reloaded "
            "calibrated model differ from the in-memory pipeline on the same "
            "input sample — the serialized model is not safe to register."
        )

    version = str(model_info.registered_model_version)
    client = mlflow.tracking.MlflowClient()
    # Idempotent — same self-healing pattern as ensure_experiment_metadata:
    # cheap to re-set on every registration, so the registry overview page
    # is never left describing a stale training cycle.
    set_registered_model_description(registered_model_name, _REGISTRY_DESCRIPTION)
    client.update_model_version(
        registered_model_name, version, description=_PENDING_VERSION_DESCRIPTION
    )
    client.set_model_version_tag(
        registered_model_name, version, "training_data_scope", "dev"
    )
    # ModelVersion.model_id does not auto-populate in OSS MLflow 3.14 — this tag
    # is the only supported hop from "the version being evaluated" to the
    # LoggedModel Phase 7's evaluate.py attaches sealed-test metrics to.
    client.set_model_version_tag(
        registered_model_name, version, "logged_model_id", model_info.model_id
    )
    # Mint-time default, set before anything downstream can fail: a crash in
    # evaluate.py/error_analysis.py/register.py leaves this version with no
    # verdict recorded, rather than with a tag someone forgot to write on an
    # abort path nobody anticipated. register.py's rollback_champion() query
    # (highest version tagged promotion_status: promoted) and the Phase 14
    # pending-orphan reaper both depend on every version reliably starting
    # here — see CLAUDE.md § MLflow Model Registry.
    client.set_model_version_tag(
        registered_model_name, version, "promotion_status", "pending"
    )
    client.set_registered_model_alias(registered_model_name, "challenger", version)

    logger.info(
        "calibration_registered",
        run_id=run_id,
        method=method,
        model_version=version,
        model_uri=model_info.model_uri,
        parity_ok=parity_ok,
    )

    return {
        "run_id": run_id,
        "method": method,
        "model_version": version,
        "model_uri": model_info.model_uri,
        "parity_ok": parity_ok,
        "calibration_summary": calibration_summary,
    }


if __name__ == "__main__":
    import sys

    import pandera as pa
    from dotenv import load_dotenv

    from telco_churn.utils.logging import configure_logging
    from telco_churn.utils.paths import compose_config

    load_dotenv()
    configure_logging()

    try:
        cfg = compose_config(overrides=sys.argv[1:] or None)
        cli_run_id = cfg.calibration.run_id
        if cli_run_id is None:
            raise ValueError(
                "calibration.run_id is required, e.g. `python -m "
                "telco_churn.models.calibrate calibration.run_id=<tuning_study_run_id>`"
            )
        result = run_calibration_step(str(cli_run_id), cfg)
        logger.info(
            "calibration_step_done",
            run_id=result["run_id"],
            method=result["method"],
            model_version=result["model_version"],
        )
    except FileNotFoundError as e:
        logger.error("calibration_data_not_found", error=str(e), exc_info=True)
        sys.exit(1)
    except pa.errors.SchemaError as e:
        logger.error("calibration_data_schema_invalid", error=str(e), exc_info=True)
        sys.exit(1)
    except ValueError as e:
        logger.error("calibration_invalid", error=str(e), exc_info=True)
        sys.exit(1)
    except RuntimeError as e:
        logger.error("calibration_blocked", error=str(e), exc_info=True)
        sys.exit(1)
    except AssertionError as e:
        logger.error("calibration_parity_failed", error=str(e), exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error("calibration_failed", error=str(e), exc_info=True)
        sys.exit(1)

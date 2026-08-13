"""Calibrate the tuned pipeline logged by run_model_logging_step (models/train/log_model.py).

Calibration is cross-fit on the development set, so there is no separate
validation split: `CalibratedClassifierCV(ensemble=False)` wraps the
pipeline *unfitted*, so its ColumnTransformer refits inside every fold
instead of leaking preprocessing statistics across folds. The unfitted
pipeline comes from cloning the model at `manifest["logged_model_uri"]`, not
from `runs:/<run_id>/model` — the latter becomes ambiguous once this module
logs its own `calibrated_model` onto the same run.

This module fits and logs the calibrated pipeline — a pure, deterministic
file transform — and stops there. It performs no registry write of its own:
minting a registry version, tagging it, and pointing the `challenger` alias
is register.py's job (`register_challenger`), run as its own CLI step after
this one (`python -m telco_churn.models.register`, with `register.run_id`
defaulting to `reports/calibrate_receipt.json` when not given, mirroring how
this module's own `calibration.run_id` defaults to `reports/train_receipt.json`) —
never called from this process, so this module's own exit code reflects only
whether calibration itself succeeded.
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
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from telco_churn.models.artifacts import (
    committed_features_from_manifest,
    load_training_manifest,
    unfitted_pipeline_from_manifest,
)
from telco_churn.models.calibration_metrics import (
    brier_skill_score,
    calibration_slope,
    expected_calibration_error,
    pooled_brier,
)
from telco_churn.models.dev_features import load_dev_customer_ids, load_dev_features
from telco_churn.models.explain import feature_directions, global_importance
from telco_churn.models.plots import reliability_diagram_bins
from telco_churn.models.shap_values import (
    compute_shap_values,
    unwrap_calibrated_pipeline,
)
from telco_churn.utils.logging import get_logger
from telco_churn.utils.mlflow import (
    TRAINING_CYCLE_RUN_DESCRIPTION,
    ensure_experiment_metadata,
    read_train_receipt,
    set_logged_model_description,
    set_run_description,
    write_calibrate_receipt,
)
from telco_churn.utils.paths import get_project_root
from telco_churn.utils.stats import paired_bootstrap_ci

__all__ = [
    "select_golden_rows",
    "outer_cv",
    "inner_cv",
    "build_calibrated_pipeline",
    "oof_uncalibrated_proba",
    "oof_calibrated_proba",
    "oof_dummy_proba",
    "per_fold_average_precision",
    "per_fold_brier",
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

# CalibratedClassifierCV(ensemble=...) — shared by build_calibrated_pipeline and
# the calibration_spec block logged alongside it, so the two can never drift
# apart into "what was fit" vs. "what the manifest says was fit".
_CALIBRATION_ENSEMBLE = False


# ---------------------------------------------------------------------------
# Step 1: build/wrap the calibrated pipeline
# ---------------------------------------------------------------------------


def select_golden_rows(
    X_dev: pd.DataFrame, customer_ids: pd.Series, n_rows: int
) -> tuple[pd.DataFrame, pd.Series]:
    """Return the n_rows dev-partition rows with the lowest customerid, for the golden-parity fixture.

    Selected by customerid value, not row position, so the fixture is stable
    against an unrelated reordering of the processed feature table upstream —
    a positional `.head(n)` would silently pin a different set of customers.
    X_dev and customer_ids must already be row-order-aligned (as
    `load_dev_features` and `load_dev_customer_ids` are).
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

    `ensemble=False` collapses `calibrated_classifiers_` to length 1, whose
    `.estimator` is the base Pipeline refit on all of development — the SHAP
    access path notebooks/05-error-analysis.ipynb depends on. `pipeline`
    must be unfitted; `CalibratedClassifierCV.fit` clones it internally per
    inner fold, so reusing the same unfitted instance across methods is safe
    without an extra `clone()` here.
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

    Pooling mixes each fold's tie structure (or, for a calibrated vector,
    each fold's distinct calibration map) into one ranking, which can mask a
    per-fold regression that scoring each fold separately catches. Recomputes
    `outer_cv(cfg)`'s fold assignment rather than threading indices through,
    since the same params + ordering reproduce the identical folds
    `cross_val_predict` used to build `proba`.
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
    ece_n_bins = int(cfg.calibration.ece_n_bins)
    ece_strategy = str(cfg.calibration.ece_strategy)

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
            "ece": expected_calibration_error(
                dummy_proba, y_dev, ece_n_bins, ece_strategy, "dummy_prior"
            ),
            "bss": brier_skill_score(dummy_brier, dummy_brier),
        },
        "uncalibrated": {
            "per_fold_mean_ap": float(np.mean(uncal_per_fold_ap)),
            "pooled_brier": uncal_brier,
            "ece": expected_calibration_error(
                uncal_proba, y_dev, ece_n_bins, ece_strategy, "uncalibrated"
            ),
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
            "ece": expected_calibration_error(
                proba, y_dev, ece_n_bins, ece_strategy, configured_method
            ),
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
            "ece": expected_calibration_error(
                other_proba, y_dev, ece_n_bins, ece_strategy, other_method
            ),
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
        "ece": expected_calibration_error(
            sigmoid_proba, y_dev, ece_n_bins, ece_strategy, "sigmoid"
        ),
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
        "ece": expected_calibration_error(
            isotonic_proba, y_dev, ece_n_bins, ece_strategy, "isotonic"
        ),
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


def _check_trial_count_gate(
    manifest: dict[str, Any], run_id: str, cfg: DictConfig
) -> None:
    """Abort before fitting anything if the run's Optuna trial count is untrustworthy.

    A data-quality gate on the tuning result, not a performance comparison.
    """
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


def _load_calibration_inputs(
    manifest: dict[str, Any], cfg: DictConfig
) -> dict[str, Any]:
    """Load the unfitted committed pipeline and the dev-partition feature/target split."""
    committed_features = committed_features_from_manifest(manifest)
    pipeline = unfitted_pipeline_from_manifest(manifest)
    X_dev, y_dev = load_dev_features(committed_features)
    return {
        "committed_features": committed_features,
        "pipeline": pipeline,
        "X_dev": X_dev,
        "y_dev": y_dev,
    }


def _select_and_render_calibration(
    pipeline: Pipeline, X_dev: pd.DataFrame, y_dev: pd.Series, cfg: DictConfig
) -> dict[str, Any]:
    """Select the calibration method and render the pre/post-calibration reliability diagram.

    Renders and logs the reliability diagram (reports/figures/reliability_diagram.png,
    mirrored onto the run's figures/ artifacts) from the exact OOF vectors that
    decided the winning method — not a recomputation, so the picture matches
    the numbers in calibration_summary.json.
    """
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

    return {"selection": selection, "method": method, "figure_path": figure_path}


def _fit_and_build_golden_fixture(
    pipeline: Pipeline,
    method: str,
    cfg: DictConfig,
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
) -> dict[str, Any]:
    """Fit the calibrated pipeline and build register.py's serving-parity reference.

    in_memory_preds is the reference for register.py's serving-parity smoke
    check, captured here — while `fitted` is still the live in-memory object
    this run just produced, before mlflow.sklearn.log_model/pickling touches
    it at all. register.py reloads this same model later, in a separate
    process, and must compare against a reference computed independently of
    that reload — a reference generated by scoring through the reload itself
    would be circular: any serialization bug would already be baked into the
    "expected" value, and the check could never fail no matter how broken
    the round trip was. Rows are stored in full (not just scores), pinned by
    customerid, at the model's committed input schema, so the fixture is
    self-contained and Phase 9/11 can replay it without touching the
    dataset. Every golden row is a development-partition row the model
    trained on — in-sample, and therefore reproducibility evidence only,
    never performance evidence; the "purpose" key travels that caveat with
    the artifact rather than leaving it to live only in a plan document.
    """
    fitted = build_calibrated_pipeline(pipeline, method, cfg)
    fitted.fit(X_dev, y_dev)

    dev_customer_ids = load_dev_customer_ids()
    input_example, golden_customer_ids = select_golden_rows(
        X_dev, dev_customer_ids, int(cfg.calibration.golden_n_rows)
    )
    in_memory_preds = fitted.predict_proba(input_example)
    signature = infer_signature(X_dev, fitted.predict_proba(X_dev))

    golden_fixture = {
        "purpose": (
            "serving-parity fixture — reproducibility only; scores are "
            "in-sample and are not performance evidence"
        ),
        "customerid": golden_customer_ids.tolist(),
        "rows": input_example.to_dict(orient="records"),
        "p_hat": in_memory_preds[:, 1].tolist(),
    }

    return {
        "fitted": fitted,
        "dev_customer_ids": dev_customer_ids,
        "input_example": input_example,
        "in_memory_preds": in_memory_preds,
        "signature": signature,
        "golden_fixture": golden_fixture,
    }


def _compute_dev_shap(
    fitted: CalibratedClassifierCV, X_dev: pd.DataFrame
) -> dict[str, Any]:
    """Compute dev-SHAP values once and rank every transformed feature's (mean_abs_shap, direction).

    This is the evidence threshold.py's V3 pre-seal screen
    (`v3_direction_sanity`) binds on — computed on the champion's own
    dev-partition rows, the same ones it trained on, never the sealed test
    set: V3 is a pre-seal veto and must never require the test partition to
    evaluate. `configs/threshold/default.yaml`'s `v3_top_k_features`/
    `v3_min_direction_magnitude` are hand-derived, once, off a real run's
    `dev_shap_summary.json` output — this function only produces the
    evidence, it does not itself decide a cutoff.
    """
    preprocessor, booster = unwrap_calibrated_pipeline(fitted)
    shap_values, feature_names, base_value, Xt = compute_shap_values(
        preprocessor, booster, X_dev
    )
    importance_rows = global_importance(shap_values, feature_names)
    directions = feature_directions(Xt, shap_values, feature_names)
    summary_rows = [
        {**row, "direction": directions[row["feature"]]} for row in importance_rows
    ]
    return {
        "shap_values": shap_values,
        "feature_names": feature_names,
        "base_value": base_value,
        "summary_rows": summary_rows,
    }


def _compute_calibration_slopes_and_summary(
    y_dev: pd.Series,
    dev_customer_ids: pd.Series,
    selection: dict[str, Any],
    method: str,
    cfg: DictConfig,
) -> dict[str, Any]:
    """Compute the calibrated/uncalibrated slopes and assemble calibration_summary.json.

    calibration_summary.json carries calibration_slope (the dev-OOF Cox
    slope, with a bootstrap CI), uncalibrated_calibration_slope (the same
    statistic on the pre-calibration OOF vector, paired for comparison — a
    reliability diagram's bin-level scatter can't by itself distinguish
    genuine miscalibration from sampling noise, and this pairing is what
    turns that visual ambiguity into two comparable numbers), and
    calibration_spec (method/inner_cv_folds/random_state/ensemble — the four
    fields a future Phase 10 recalibration flow, not yet designed, would
    need to rebuild an identical CalibratedClassifierCV without re-deriving
    it).

    dev_oof_predictions.parquet is the vector that selected `method`,
    produced its BSS, and will validate t* — computed already as
    selection["calibrated_proba"]. Logged as-is rather than recomputed
    downstream (CLAUDE.md § Persist the evidence, not just the conclusion):
    a recompute is only sound while the dataset is static and the folds are
    seeded, and silently diverges from the vector this cycle's decisions
    actually rested on once either isn't true (Phase 10's retraining).
    Segment/protected columns are joined later, by Phase 7's evaluate.py,
    where the dev-partition feature columns are in hand — this is the
    minimal vector.
    """
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

    return {
        "dev_oof_predictions": dev_oof_predictions,
        "slope": slope,
        "uncalibrated_slope": uncalibrated_slope,
        "calibration_summary": calibration_summary,
    }


def _log_calibration_run(
    run_id: str,
    cfg: DictConfig,
    calibration_summary: dict[str, Any],
    slope: dict[str, Any],
    uncalibrated_slope: dict[str, Any],
    figure_path: Path,
    dev_oof_predictions: pd.DataFrame,
    golden: dict[str, Any],
    dev_shap: dict[str, Any],
) -> Any:
    """Log every calibration artifact, metric, and the calibrated pipeline itself — as a LoggedModel, not a registry version.

    name="calibrated_model" is load-bearing: reusing name="model" would
    rebind runs:/<run_id>/model away from the uncalibrated pipeline, and
    would make a second run of this function wrap a CalibratedClassifierCV
    inside another one. serialization_format=cloudpickle is mandatory —
    mlflow's skops default rejects CalibratedClassifierCV's internals.
    Deliberately omits registered_model_name= here — that would register a
    version as a side effect of logging, before any validation has run.
    register.py::register_challenger is the sole place that mints a registry
    version, via mlflow.register_model(model_info.model_uri, ...) — invoked
    as its own CLI step after this one, never from inside this process (see
    this module's own docstring). The calibrated_model_id/calibrated_model_uri
    tags set on this run right after logging are what let that separate
    process, and an explicit register.run_id=<id> override, resolve this
    LoggedModel without touching the registry.

    Also logs dev_brier/dev_bss/dev_ece/dev_per_fold_mean_ap/
    dev_calibration_slope/dev_calibration_slope_ci_lower/
    dev_calibration_slope_ci_upper/dev_uncalibrated_calibration_slope/
    dev_uncalibrated_calibration_slope_ci_lower/
    dev_uncalibrated_calibration_slope_ci_upper/dev_mean_p_hat_calibrated/
    dev_mean_p_hat_uncalibrated/dev_observed_churn_rate as MLflow metrics (not
    just JSON fields), so they are plottable as a series across retraining
    cycles.

    Logs dev_shap_values.parquet (the full per-row, per-feature SHAP matrix)
    and dev_shap_summary.json (per-feature mean_abs_shap + direction, every
    transformed feature — CLAUDE.md § Persist the evidence, not just the
    conclusion) — the evidence threshold.py's V3 pre-seal screen binds on.
    """
    method = str(calibration_summary["method"])
    ensure_experiment_metadata(cfg)

    with mlflow.start_run(run_id=run_id):
        set_run_description(TRAINING_CYCLE_RUN_DESCRIPTION)
        mlflow.log_dict(calibration_summary, "calibration/calibration_summary.json")
        mlflow.log_artifact(str(figure_path), artifact_path="calibration/figures")

        # calibration_summary.json remains the audit record — these duplicate a
        # handful of its fields so BSS/ECE/Brier/slope are plottable as a series
        # across cycles (Phase 10), the same pattern Phase 5's candidates.py/
        # tuning.py/feature_audit.py already follow and Phase 6 had skipped.
        winning_diagnostics = calibration_summary["diagnostics"][method]
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

            shap_path = Path(tmp_dir) / "dev_shap_values.parquet"
            shap_df = pd.DataFrame(
                dev_shap["shap_values"], columns=dev_shap["feature_names"]
            )
            shap_df.insert(0, "customerid", golden["dev_customer_ids"].to_numpy())
            shap_df["base_value"] = dev_shap["base_value"]
            shap_df.to_parquet(shap_path, index=False)
            mlflow.log_artifact(str(shap_path), artifact_path="calibration")

        mlflow.log_dict(dev_shap["summary_rows"], "calibration/dev_shap_summary.json")
        mlflow.log_dict(golden["golden_fixture"], "calibration/golden_predictions.json")

        model_info = mlflow.sklearn.log_model(
            sk_model=golden["fitted"],
            name="calibrated_model",
            signature=golden["signature"],
            input_example=golden["input_example"],
            pyfunc_predict_fn="predict_proba",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
        set_logged_model_description(model_info.model_id, _MODEL_DESCRIPTION)

        # Run tags, not just the local calibrate_receipt.json — so an
        # explicit register.run_id=<id> override (re-registering an older
        # calibration run, not necessarily the latest one on this machine)
        # can resolve this LoggedModel without touching the registry or
        # trusting a receipt that may belong to a different run entirely.
        mlflow.set_tag("calibrated_model_id", model_info.model_id)
        mlflow.set_tag("calibrated_model_uri", model_info.model_uri)

    return model_info


def run_calibration_step(run_id: str, cfg: DictConfig) -> dict[str, Any]:
    """Calibrate, select a method, and log the fitted pipeline — no registry write.

    Fits and logs a pure, deterministic file transform — the calibrated
    pipeline, calibration_summary.json, dev_oof_predictions.parquet — and
    stops there. Minting a registry version, tagging it pending, verifying
    reload parity, and pointing `challenger` is register.py's job
    (register_challenger), run as a separate CLI step after this one — never
    called from this process (see this module's own docstring for why).

    reports/figures/reliability_diagram.png is overwritten on every call —
    it reflects whichever run executed this function most recently on this
    machine, not a specific run_id. A reader who needs to know which run a
    figure on disk corresponds to should cross-check that run's own
    calibration_summary.json rather than trusting the file's mtime; the
    MLflow-logged copy under that run's figures/ artifacts is the durable,
    run-pinned reference.

    Also logs golden_predictions.json (customerid-pinned dev rows, at the
    model's committed input schema, plus the in-memory fitted pipeline's
    reference scores on them, captured before log_model/pickling touches
    anything) — the independent reference register.py's serving-parity smoke
    check (both at mint time and at promotion time) verifies against, and
    Phase 9's API parity test reuses.

    Writes reports/calibrate_receipt.json ({"run_id", "logged_model_id",
    "model_uri"}) — never a model_version, since nothing is registered yet
    at this point. This is the pointer threshold.py/evaluate.py/
    error_analysis.py's __main__ blocks read by default when no explicit
    run_id/model_version override is given, and register.py's mint step
    reads by default when no explicit register.run_id override is given
    (see utils/mlflow.py's module docstring for why a local receipt is safe
    here).
    """
    manifest = load_training_manifest(run_id, cfg)
    _check_trial_count_gate(manifest, run_id, cfg)

    loaded = _load_calibration_inputs(manifest, cfg)
    X_dev, y_dev = loaded["X_dev"], loaded["y_dev"]

    selection_result = _select_and_render_calibration(
        loaded["pipeline"], X_dev, y_dev, cfg
    )
    method = selection_result["method"]

    golden = _fit_and_build_golden_fixture(
        loaded["pipeline"], method, cfg, X_dev, y_dev
    )

    slopes_and_summary = _compute_calibration_slopes_and_summary(
        y_dev, golden["dev_customer_ids"], selection_result["selection"], method, cfg
    )
    calibration_summary = slopes_and_summary["calibration_summary"]

    dev_shap = _compute_dev_shap(golden["fitted"], X_dev)

    model_info = _log_calibration_run(
        run_id,
        cfg,
        calibration_summary,
        slopes_and_summary["slope"],
        slopes_and_summary["uncalibrated_slope"],
        selection_result["figure_path"],
        slopes_and_summary["dev_oof_predictions"],
        golden,
        dev_shap,
    )

    write_calibrate_receipt(run_id, model_info.model_id, model_info.model_uri, cfg)

    logger.info(
        "calibration_logged",
        run_id=run_id,
        method=method,
        logged_model_id=model_info.model_id,
        model_uri=model_info.model_uri,
    )

    return {
        "run_id": run_id,
        "method": method,
        "logged_model_id": model_info.model_id,
        "model_uri": model_info.model_uri,
        "calibration_summary": calibration_summary,
    }


if __name__ == "__main__":
    import sys

    import pandera as pa
    from dotenv import load_dotenv

    from telco_churn.utils.logging import configure_logging
    from telco_churn.utils.paths import activate_config, compose_config

    load_dotenv()
    configure_logging()

    def _resolve_run_id(cfg: DictConfig) -> str:
        """Return calibration.run_id if given, else train.py's receipt's run_id.

        No both-given ambiguity guard applies here (unlike
        threshold.py/evaluate.py/error_analysis.py's model_version/run_id
        pair) — calibrate.py has exactly one upstream identifier axis, since
        nothing is registered yet at this point in the pipeline.
        """
        if cfg.calibration.run_id is not None:
            return str(cfg.calibration.run_id)
        return str(read_train_receipt(cfg)["run_id"])

    try:
        cfg = compose_config(overrides=sys.argv[1:] or None)
        activate_config(cfg)
        cli_run_id = _resolve_run_id(cfg)
        result = run_calibration_step(cli_run_id, cfg)
        logger.info(
            "calibration_step_done",
            run_id=result["run_id"],
            method=result["method"],
            logged_model_id=result["logged_model_id"],
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
    except Exception as e:
        logger.error("calibration_failed", error=str(e), exc_info=True)
        sys.exit(1)

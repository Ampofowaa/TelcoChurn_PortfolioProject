"""Sealed-test evaluation — the one module permitted to touch the test partition.

CLAUDE.md's modelling invariant: X_test/y_test are imported and used in
exactly one place, this module. Every other module reaches test data solely
through this module's persisted artifacts (reports/test_predictions.parquet),
never through telco_churn.data.split directly.

Resolves the model being evaluated by explicit run_id/version, never by
alias: models:/telco-churn-pipeline@challenger is a moving pointer, and a
future full-data retrain path could leave it pointing at a model trained on
the test set — which would then silently produce excellent, meaningless
sealed-test metrics. This module logs the resolved version into
reports/metrics.json so the metrics stay attributable to a specific artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
import warnings
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import mlflow
import mlflow.artifacts
import mlflow.sklearn
import mlflow.tracking
import numpy as np
import pandas as pd
from mlflow.data.pandas_dataset import from_pandas as mlflow_dataset_from_pandas
from mlflow.exceptions import MlflowException
from numpy.typing import NDArray
from omegaconf import DictConfig
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from telco_churn.data.split import partition
from telco_churn.features.accessor import load_features
from telco_churn.features.build import TARGET_COL
from telco_churn.models.calibrate import (
    brier_skill_score,
    calibration_slope,
    committed_features_from_manifest,
    expected_calibration_error,
    load_training_manifest,
    murphy_decomposition,
    pooled_brier,
)
from telco_churn.models.diagnostics import (
    FAIRNESS_AXES,
    ROBUSTNESS_AXES,
    build_segment_lookup,
    demographic_parity_difference_by_axis,
    equal_opportunity_difference_by_axis,
    fixed_recall_profile,
    sliced_calibration,
    sliced_decision_rates,
    sliced_ranking_metrics,
)
from telco_churn.models.economics import (
    break_even_retention_rate,
    campaign_cost,
    ev_by_k,
    expected_value,
    retained_revenue,
    sensitivity_oneway,
    sensitivity_twoway,
    tornado,
)
from telco_churn.models.gate import GateBars, GateInputs, decide_promotion
from telco_churn.models.plots import (
    classification_summary_points,
    decile_lift_table,
    pr_curve_points,
    reliability_diagram_bins,
    roc_curve_points,
)
from telco_churn.models.threshold import (
    CostScenario,
    costs_config_hash,
    load_costs_config,
    load_policy_thresholds,
    resolve_policy_scenarios,
    resolve_policy_thresholds_by_scenario,
)
from telco_churn.utils.logging import get_logger
from telco_churn.utils.mlflow import (
    ensure_experiment_metadata,
    load_model_promotion_bars,
    resolve_logged_model_id,
    resolve_model_run_id,
    resolve_tracking_uri,
    set_run_description,
)
from telco_churn.utils.paths import get_project_root
from telco_churn.utils.stats import (
    bootstrap_metric_ci,
    paired_bootstrap_ci,
    paired_bootstrap_metric_ci,
)

__all__ = [
    "build_gate_inputs",
    "check_threshold_provenance",
    "comparative_deltas",
    "content_hash",
    "demographic_parity_difference_by_axis",
    "equal_opportunity_difference_by_axis",
    "load_fitted_model",
    "load_dev_oof_diagnostics",
    "load_incumbent_proba",
    "load_model_promotion_bars",
    "load_policy_thresholds",
    "load_test_customer_ids",
    "load_test_features",
    "load_test_segment_lookup",
    "load_threshold_validation",
    "resolve_champion_version",
    "resolve_logged_model_id",
    "resolve_model_run_id",
    "resolve_policy_scenarios",
    "resolve_policy_thresholds_by_scenario",
    "run_evaluation_step",
    "sealed_test_business_impact",
    "sealed_test_calibration_report",
    "sealed_test_classification_report",
    "sealed_test_decile_lift",
    "sealed_test_fixed_recall_profile",
    "sealed_test_promotion_decision",
    "sealed_test_ranking_metrics",
    "sealed_test_sensitivity_analysis",
    "sliced_business_impact",
    "sliced_calibration",
    "sliced_decision_rates",
    "sliced_ranking_metrics",
]

# ROBUSTNESS_AXES/FAIRNESS_AXES (ANALYSIS.md §0's V1/V2/V2b surface) live in
# diagnostics.py — shared with threshold.py's dev-OOF screen, which computes
# V1/V2/V2b on the dev-OOF surface; this module uses the same axes for its
# own sealed-test reporting slices.

# Below this many rows, or with only one class present, a segment's own
# calibration slope isn't estimable — skipped rather than reported as noise.
_MIN_SLICE_SIZE = 10

logger = get_logger(__name__)

_RUN_DESCRIPTION = (
    "Sealed-test evaluation for one registered model version - PR-AUC, "
    "recall/precision/F1 and confusion matrix per cost scenario, "
    "calibration transfer (BSS/ECE/slope), business-impact and EV "
    "sensitivity analysis, sliced fairness/robustness metrics. Computes the "
    "promotion gate verdict via gate.py::decide_promotion and persists "
    "promotion_decision.json. The sealed test set is touched here, and only "
    "here."
)

# decide_promotion's `incumbent` parameter exists only to select the regime
# via an `is None` check (gate.py's own docstring: "incumbent's own metrics
# are not read here") — the comparative Δ fields already live on `candidate`,
# computed by comparative_deltas below. This constant's field values are
# therefore never inspected by decide_promotion; it exists only to be a
# non-None GateInputs, so a caller need not fabricate a second real
# computation nothing consumes.
_INCUMBENT_EXISTS_MARKER = GateInputs(
    pr_auc=0.0,
    recall=0.0,
    bss=0.0,
    calibration_slope=0.0,
    calibration_slope_ci_lower=0.0,
    calibration_slope_ci_upper=0.0,
)

# Contacting "everyone"/"no one" is expressed as an extreme threshold on the
# real proba vector rather than a fitted DummyClassifier: since proba is
# always in [0, 1], any cut at or below the minimum score contacts every row
# and any cut above 1.0 contacts none — a function of the labels alone, not
# of proba's actual values, so this gives the identical dollar figure a
# DummyClassifier(strategy='constant', ...) baseline would.
_CONTACT_ALL_THRESHOLD = 0.0
_CONTACT_NONE_THRESHOLD = 1.0 + 1e-9


def load_fitted_model(model_version: str, cfg: DictConfig) -> Pipeline:
    """Load the calibrated pipeline for an explicit registered version — never an alias."""
    registered_model_name = str(cfg.mlflow.registered_model_name)
    model: Pipeline = mlflow.sklearn.load_model(
        f"models:/{registered_model_name}/{model_version}"
    )
    return model


def _load_test_partition() -> pd.DataFrame:
    """Return the full sealed-test-partition rows (customerid included), pre feature-subsetting.

    This function, and load_test_features/load_test_customer_ids below it, are
    the only call sites in src/ that read the test side of
    telco_churn.data.split.partition() — the structural half of "test set
    touched once": no other module imports data.split for the test partition.
    """
    df = load_features()
    _dev_df, test_df = partition(df)
    return test_df


def load_test_features(
    committed_features: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """Load the sealed-test rows, restricted to the frozen committed feature set."""
    test_df = _load_test_partition()
    return test_df[committed_features], test_df[TARGET_COL]


def load_test_customer_ids() -> pd.Series:
    """Return the customerid Series for the sealed test partition.

    Row-order-aligned with load_test_features's (X_test, y_test) — both derive
    from the same _load_test_partition() call, the same recompute-rather-than-
    thread-state idiom calibrate.py's dev-side loaders already use.
    """
    return _load_test_partition()["customerid"].reset_index(drop=True)


def load_test_segment_lookup() -> dict[str, pd.Series]:
    """The sealed-test partition's robustness/fairness segment axes, row-order-aligned
    with load_test_features/load_test_customer_ids (all three derive from the same
    _load_test_partition() call)."""
    return build_segment_lookup(_load_test_partition())


def load_threshold_validation(run_id: str, cfg: DictConfig) -> dict[str, Any]:
    """Load threshold.py's threshold_validation.json artifact — the model-dependent stamp."""
    mlflow.set_tracking_uri(resolve_tracking_uri(str(cfg.mlflow.tracking_uri)))
    validation: dict[str, Any] = mlflow.artifacts.load_dict(
        f"runs:/{run_id}/threshold/threshold_validation.json"
    )
    return validation


def load_dev_oof_diagnostics(run_id: str, cfg: DictConfig) -> dict[str, Any]:
    """Load threshold.py's dev_oof_diagnostics.json artifact (V1/V2/V2b) — resolved by run_id, not a local path.

    A fixed local path (reports/dev_oof_diagnostics.json) is what
    reliability_diagram.png warns against elsewhere in this codebase: it
    reflects whichever run last executed threshold.py on this machine, not
    necessarily run_id. Fetching by explicit run_id, same as
    load_threshold_validation, is what makes this correct under CI or a
    fresh checkout where no prior threshold.py run has touched local disk.
    """
    mlflow.set_tracking_uri(resolve_tracking_uri(str(cfg.mlflow.tracking_uri)))
    diagnostics: dict[str, Any] = mlflow.artifacts.load_dict(
        f"runs:/{run_id}/threshold/dev_oof_diagnostics.json"
    )
    return diagnostics


def check_threshold_provenance(
    validation_payload: dict[str, Any], run_id: str, model_version: str
) -> None:
    """Raise ValueError if the threshold's model stamp doesn't match the model being evaluated.

    threshold.py splits the derived threshold into a model-independent policy
    file (configs/policy/threshold.yaml — a pure function of costs.yaml,
    carrying no model stamp) and this model-dependent validation artifact.
    "A re-calibration invalidates a previously-derived threshold" (threshold.py's
    own docstring) is aspirational until something checks it — this is that
    check. Applying a threshold derived against a different calibration map
    would otherwise produce plausible, wrong numbers with nothing raised.
    """
    stamped_run_id = str(validation_payload["model_run_id"])
    stamped_version = str(validation_payload["model_version"])
    if stamped_run_id != run_id or stamped_version != model_version:
        raise ValueError(
            "threshold_validation.json's model stamp "
            f"(run_id={stamped_run_id!r}, version={stamped_version!r}) does not "
            f"match the model being evaluated (run_id={run_id!r}, "
            f"version={model_version!r}) — the threshold was derived against a "
            "different calibration map. Re-run models.threshold before evaluating."
        )


def sealed_test_business_impact(
    y_test: pd.Series,
    proba: NDArray[np.float64],
    scenarios: dict[str, CostScenario],
    thresholds: dict[str, float],
    n_bootstrap: int,
    random_state: int,
) -> dict[str, Any]:
    """Per-scenario dollar impact at the shipped operating point, plus the two policy baselines.

    Reports, per scenario: net expected value (with a sampling-uncertainty
    bootstrap CI), gross campaign cost, gross retained revenue, the number
    and rate contacted, and the break-even retention rate — the full
    picture, not net EV alone, since "this costs $C to run and returns $R"
    is the sentence Finance asks for and a net figure cannot answer it.

    Also reports each scenario's ev_treat_all (contacting everyone, at
    _CONTACT_ALL_THRESHOLD) and ev_treat_none (contacting no one, at
    _CONTACT_NONE_THRESHOLD — always exactly 0 by construction, reported for
    completeness rather than because it varies). The model is expected to
    beat both.

    ⚠ ev_bracket is the min-max across scenarios' point estimates, not a
    statistical interval. ANALYSIS.md §0 is explicit that the three
    scenarios are not an ordered min/mid/max — their parameters partially
    offset — so this brackets whatever they turn out to be rather than
    assuming "conservative" is the floor. parameter_spread_dominates_sampling
    flags whether that bracket is wider than the widest within-scenario
    bootstrap CI: when False, the scenarios in configs/costs.yaml are too
    narrow to express the real uncertainty in r, and the headline EV bracket
    would be published with its dominant error source invisible. This
    function only computes the diagnostic; the caller decides whether to log
    a warning on it.
    """
    y = y_test.to_numpy(dtype=np.int64)
    p = np.asarray(proba, dtype=float)

    per_scenario: dict[str, dict[str, Any]] = {}
    for name, scenario in scenarios.items():
        t = thresholds[name]

        def _ev_metric(
            y_arr: NDArray[np.int_],
            p_arr: NDArray[np.float64],
            _s: CostScenario = scenario,
            _t: float = t,
        ) -> float:
            return expected_value(p_arr, y_arr, _s, _t)

        ev_ci = bootstrap_metric_ci(y, p, _ev_metric, n_bootstrap, random_state)
        per_scenario[name] = {
            "threshold": t,
            "ev": ev_ci["obs"],
            "ev_ci_lower": ev_ci["ci_lower"],
            "ev_ci_upper": ev_ci["ci_upper"],
            "campaign_cost": campaign_cost(p, scenario, t),
            "retained_revenue": retained_revenue(p, y, scenario, t),
            "n_contacted": int(np.sum(p >= t)),
            "contact_rate": float(np.mean(p >= t)),
            "break_even_retention_rate": break_even_retention_rate(p, y, scenario, t),
            "ev_treat_all": expected_value(p, y, scenario, _CONTACT_ALL_THRESHOLD),
            "ev_treat_none": expected_value(p, y, scenario, _CONTACT_NONE_THRESHOLD),
        }

    ev_points = [cast(float, row["ev"]) for row in per_scenario.values()]
    ci_widths = [
        cast(float, row["ev_ci_upper"]) - cast(float, row["ev_ci_lower"])
        for row in per_scenario.values()
    ]
    ev_spread = max(ev_points) - min(ev_points)
    widest_ci = max(ci_widths)

    return {
        "scenarios": per_scenario,
        "ev_bracket_min": min(ev_points),
        "ev_bracket_max": max(ev_points),
        "ev_spread": ev_spread,
        "widest_within_scenario_ci_width": widest_ci,
        "parameter_spread_dominates_sampling": ev_spread > widest_ci,
    }


def sealed_test_sensitivity_analysis(
    y_test: pd.Series,
    proba: NDArray[np.float64],
    base_scenario: CostScenario,
    base_threshold: float,
    retention_rate_values: list[float],
    cost_values: list[float],
    tornado_pct_perturbation: float,
) -> dict[str, Any]:
    """Sensitivity suite on the sealed test set, base scenario/threshold only.

    Because two of the three cost inputs (r, c) are guesses rather than
    measurements, ANALYSIS.md §0 asserts r is the most consequential number
    in the deployment decision — this is what proves it rather than merely
    asserting it. Every piece reuses economics.py's helpers (which themselves
    reuse threshold.py's expected_value_at_threshold), so none of this
    re-derives p·r·LTV − c a second time.
    """
    y = y_test.to_numpy(dtype=np.int64)
    p = np.asarray(proba, dtype=float)

    return {
        "oneway_retention_rate": sensitivity_oneway(
            p,
            y,
            base_scenario,
            base_threshold,
            "retention_rate",
            retention_rate_values,
        ),
        "oneway_cost": sensitivity_oneway(
            p, y, base_scenario, base_threshold, "cost", cost_values
        ),
        "twoway": sensitivity_twoway(
            p, y, base_scenario, base_threshold, retention_rate_values, cost_values
        ),
        "tornado": tornado(
            p, y, base_scenario, base_threshold, tornado_pct_perturbation
        ),
    }


def sealed_test_ranking_metrics(
    y_test: pd.Series, proba: NDArray[np.float64], n_bootstrap: int, random_state: int
) -> dict[str, Any]:
    """Threshold-free sealed-test ranking metrics: PR-AUC (selection) and ROC-AUC (diagnostic).

    Both carry a percentile bootstrap CI via utils.stats.bootstrap_metric_ci
    (row-resampled — average_precision_score and roc_auc_score are set-level
    metrics with no per-row decomposition). Also reports the
    DummyClassifier(strategy='prior') PR-AUC floor, fit and scored on the
    test partition itself: a constant-prevalence prediction cannot overfit
    its own scoring set, so this is the statistical anchor that makes
    "PR-AUC >= 0.60" legible rather than arbitrary (ANALYSIS.md §0) — the
    same reference BSS already uses for calibration.
    """
    y = y_test.to_numpy(dtype=float)
    p = np.asarray(proba, dtype=float)

    pr_auc = bootstrap_metric_ci(
        y,
        p,
        lambda y_arr, p_arr: float(average_precision_score(y_arr, p_arr)),
        n_bootstrap,
        random_state,
    )
    roc_auc = bootstrap_metric_ci(
        y,
        p,
        lambda y_arr, p_arr: float(roc_auc_score(y_arr, p_arr)),
        n_bootstrap,
        random_state,
    )

    dummy = DummyClassifier(strategy="prior")
    dummy_x = np.zeros((len(y), 1))
    dummy.fit(dummy_x, y)
    dummy_proba = dummy.predict_proba(dummy_x)[:, 1]

    return {
        "pr_auc": pr_auc["obs"],
        "pr_auc_ci_lower": pr_auc["ci_lower"],
        "pr_auc_ci_upper": pr_auc["ci_upper"],
        "roc_auc": roc_auc["obs"],
        "roc_auc_ci_lower": roc_auc["ci_lower"],
        "roc_auc_ci_upper": roc_auc["ci_upper"],
        "dummy_pr_auc_floor": float(average_precision_score(y, dummy_proba)),
    }


def _precision_recall_f1_at(
    y_arr: NDArray[np.float64], p_arr: NDArray[np.float64], threshold: float
) -> tuple[float, float, float]:
    """Precision/recall/F1 at a fixed threshold — the bootstrap resample body.

    Duplicates plots.classification_summary_points' confusion-matrix
    arithmetic rather than calling it: that function builds a full
    per-scenario dict (tp_pct, contact_rate, support, ...) on every call, and
    an n_bootstrap-iteration resample loop only needs these three numbers
    back, n_bootstrap times.
    """
    pred = p_arr >= threshold
    tp = float((pred & (y_arr == 1)).sum())
    fp = float((pred & (y_arr == 0)).sum())
    fn = float((~pred & (y_arr == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else float("nan")
    )
    return precision, recall, f1


def sealed_test_classification_report(
    y_test: pd.Series,
    proba: NDArray[np.float64],
    thresholds: dict[str, float],
    n_bootstrap: int,
    random_state: int,
) -> list[dict[str, Any]]:
    """Per-scenario confusion matrix and positive-class rates, with bootstrap CIs.

    Point estimates (confusion counts, row-normalised percentages, support,
    contact rate) come from plots.classification_summary_points — that
    module computes point estimates only, per its own docstring, so the
    bootstrap CIs on precision/recall/F1 are attached here.
    """
    y = y_test.to_numpy(dtype=float)
    p = np.asarray(proba, dtype=float)
    rows = classification_summary_points(y.tolist(), p.tolist(), thresholds)

    rng = np.random.default_rng(random_state)
    n = len(y)
    for row in rows:
        threshold = cast(float, row["threshold"])
        precisions = np.empty(n_bootstrap, dtype=float)
        recalls = np.empty(n_bootstrap, dtype=float)
        f1s = np.empty(n_bootstrap, dtype=float)
        for i in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            precisions[i], recalls[i], f1s[i] = _precision_recall_f1_at(
                y[idx], p[idx], threshold
            )
        # A resample with zero predicted positives (tp + fp == 0) makes
        # _precision_recall_f1_at return nan by design (division is
        # undefined, not erroneous); at a high enough threshold on a small
        # sample every one of the n_bootstrap draws can land there
        # simultaneously, so nanpercentile's all-NaN warning fires on an
        # already-correct nan-in/nan-out result. Suppressed narrowly, by
        # message, so an unrelated RuntimeWarning inside this block would
        # still surface.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="All-NaN slice encountered", category=RuntimeWarning
            )
            row["precision_ci_lower"] = float(np.nanpercentile(precisions, 2.5))
            row["precision_ci_upper"] = float(np.nanpercentile(precisions, 97.5))
            row["recall_ci_lower"] = float(np.nanpercentile(recalls, 2.5))
            row["recall_ci_upper"] = float(np.nanpercentile(recalls, 97.5))
            row["f1_ci_lower"] = float(np.nanpercentile(f1s, 2.5))
            row["f1_ci_upper"] = float(np.nanpercentile(f1s, 97.5))
    return rows


def sealed_test_fixed_recall_profile(
    y_test: pd.Series, proba: NDArray[np.float64], recall_targets: list[float]
) -> list[dict[str, float]]:
    """Sealed-test fixed-recall profile — threshold-planning tool, reused from diagnostics.py.

    Recall targets are computed against, never hard-coded — a threshold-
    planning read at alternative operating points, reported alongside the
    three shipped cost-scenario thresholds.
    """
    return fixed_recall_profile(y_test.tolist(), proba.tolist(), recall_targets)


def sealed_test_calibration_report(
    y_test: pd.Series,
    proba: NDArray[np.float64],
    cfg: DictConfig,
    n_bootstrap: int,
    random_state: int,
) -> dict[str, Any]:
    """Sealed-test calibration report: BSS, ECE, Murphy's decomposition, calibration slope, reliability bins.

    Whether Phase 6's dev-OOF calibration transfers to held-out data is a
    genuinely open question this answers once — Brier alone is not a
    calibration report (ANALYSIS.md §0). The calibration slope computed
    here, not calibrate.py's dev-OOF screen, is the one gate.py reads; ECE,
    Murphy's decomposition, and the reliability diagram are reported
    diagnostics that make a slope veto explicable rather than a bare failed
    number.
    """
    p = np.asarray(proba, dtype=float)
    candidate_brier = pooled_brier(p, y_test)

    dummy = DummyClassifier(strategy="prior")
    dummy_x = np.zeros((len(y_test), 1))
    dummy.fit(dummy_x, y_test)
    dummy_proba = dummy.predict_proba(dummy_x)[:, 1]
    reference_brier = pooled_brier(dummy_proba, y_test)

    return {
        "brier": candidate_brier,
        "dummy_prior_brier": reference_brier,
        "bss": brier_skill_score(candidate_brier, reference_brier),
        "ece": expected_calibration_error(p, y_test, cfg),
        "murphy_decomposition": murphy_decomposition(p, y_test, cfg),
        "calibration_slope": calibration_slope(y_test, p, n_bootstrap, random_state),
        "reliability_bins": reliability_diagram_bins(
            p.tolist(),
            y_test.tolist(),
            n_bins=int(cfg.calibration.ece_n_bins),
            strategy=str(cfg.calibration.ece_strategy),
        ),
    }


def sealed_test_decile_lift(
    y_test: pd.Series, proba: NDArray[np.float64]
) -> list[dict[str, float]]:
    """Sealed-test decile lift/gains table — thin wrapper around plots.decile_lift_table."""
    return decile_lift_table(y_test.tolist(), proba.tolist())


# sliced_ranking_metrics/flag_segment_collapse (V1), sliced_decision_rates +
# equal_opportunity_difference_by_axis/demographic_parity_difference_by_axis (V2),
# and sliced_calibration/flag_calibration_collapse (V2b) now live in diagnostics.py —
# threshold.py's dev-OOF screen owns computing them on the dev-OOF surface; this
# module only re-imports the five still needed for its own sealed-test slices below
# (sliced_ranking_metrics, sliced_decision_rates, sliced_calibration,
# equal_opportunity_difference_by_axis, demographic_parity_difference_by_axis) —
# flag_segment_collapse/flag_calibration_collapse have no sealed-test use.


def sliced_business_impact(
    y_true: pd.Series,
    proba: NDArray[np.float64],
    segment_lookup: dict[str, pd.Series],
    axes: tuple[str, ...],
    scenario: CostScenario,
    threshold: float,
) -> list[dict[str, object]]:
    """Per-segment dollar impact at the shipped base-scenario threshold, across every named axis.

    Extends sliced_decision_rates' FNR/selection-rate gaps into the dollars
    they imply: a segment with a higher FNR has more of its own churners
    going uncontacted, and this reports what that costs in absolute terms —
    campaign_cost, retained_revenue, and ev reuse economics.py's totals
    unchanged, just computed on a segment mask instead of the whole test set.
    missed_revenue (false negatives * retention_rate * ltv) is new: the
    retained revenue this segment's un-contacted churners would have added
    had they been flagged, the number that makes a percentage-point FNR gap
    legible as dollars rather than a rate. No bootstrap CI — these segments
    are already thin, and a second resampling layer would mostly report
    sampling noise on top of it.
    """
    y = y_true.to_numpy(dtype=np.int64)
    p = np.asarray(proba, dtype=float)
    rows: list[dict[str, object]] = []
    for axis in axes:
        group = segment_lookup[axis]
        for value in sorted(group.unique(), key=str):
            mask = (group == value).to_numpy()
            y_seg, p_seg = y[mask], p[mask]
            n_churners = int(y_seg.sum())
            tp = int(np.sum((p_seg >= threshold) & (y_seg == 1)))
            fn = n_churners - tp
            rows.append(
                {
                    "axis": axis,
                    "value": value,
                    "n": int(mask.sum()),
                    "n_churners": n_churners,
                    "campaign_cost": campaign_cost(p_seg, scenario, threshold),
                    "retained_revenue": retained_revenue(
                        p_seg, y_seg, scenario, threshold
                    ),
                    "missed_revenue": float(
                        fn * scenario.retention_rate * scenario.ltv
                    ),
                    "ev": expected_value(p_seg, y_seg, scenario, threshold),
                }
            )
    return rows


def resolve_champion_version(cfg: DictConfig) -> str | None:
    """Resolve the `champion` alias to an explicit version number, once — a
    read of "does a champion currently exist" and "which version is it,
    right now", never re-read as a moving pointer afterward. Returns None
    when no champion alias exists yet (cold start): the registered model
    itself may not exist yet either, and both cases mean the same thing here.
    """
    mlflow.set_tracking_uri(resolve_tracking_uri(str(cfg.mlflow.tracking_uri)))
    registered_model_name = str(cfg.mlflow.registered_model_name)
    client = mlflow.tracking.MlflowClient()
    try:
        version = client.get_model_version_by_alias(registered_model_name, "champion")
    except MlflowException:
        return None
    return str(version.version)


def load_incumbent_proba(champion_version: str, cfg: DictConfig) -> NDArray[np.float64]:
    """Score the champion (by its already-resolved explicit version, never
    re-reading the alias) on the identical sealed-test rows the candidate is
    scored on.

    Loads the full test partition rather than a committed-feature-restricted
    subset: the champion may have been frozen against a different feature
    spec than the candidate being evaluated this cycle, and a Pipeline's own
    ColumnTransformer selects the columns it needs by name regardless of
    which superset of columns it is handed — so passing the same full
    dataframe to both candidate and incumbent is what makes the comparison
    robust to that mismatch, rather than assuming the two share one
    committed_features list.
    """
    incumbent_model = load_fitted_model(champion_version, cfg)
    test_df = _load_test_partition()
    proba: NDArray[np.float64] = incumbent_model.predict_proba(test_df)[:, 1]
    return proba


def comparative_deltas(
    y_test: pd.Series,
    candidate_proba: NDArray[np.float64],
    incumbent_proba: NDArray[np.float64],
    n_bootstrap: int,
    random_state: int,
) -> dict[str, float]:
    """Paired-bootstrap Δ = candidate − incumbent on PR-AUC and Brier, over the
    identical sealed-test rows — the two fields gate.py's comparative regime
    needs and cannot compute itself (it never sees the probability vectors;
    see GateInputs' own docstring).

    PR-AUC has no per-row decomposition, so its Δ goes through
    paired_bootstrap_metric_ci (row-resampled, recomputing average_precision_
    score on each resampled set for both models). Brier is a per-row proper
    score, so its Δ goes through the cheaper paired_bootstrap_ci directly on
    each model's own per-row squared error — the same distinction utils.stats'
    two functions exist to make.
    """
    y = y_test.to_numpy(dtype=float)
    pr_auc_delta = paired_bootstrap_metric_ci(
        y,
        candidate_proba,
        incumbent_proba,
        lambda y_arr, p_arr: float(average_precision_score(y_arr, p_arr)),
        n_bootstrap,
        random_state,
    )
    candidate_brier_terms = (y - np.asarray(candidate_proba, dtype=float)) ** 2
    incumbent_brier_terms = (y - np.asarray(incumbent_proba, dtype=float)) ** 2
    brier_delta = paired_bootstrap_ci(
        candidate_brier_terms, incumbent_brier_terms, n_bootstrap, random_state
    )
    return {
        "pr_auc_delta_obs": pr_auc_delta["delta_obs"],
        "pr_auc_delta_ci_lower": pr_auc_delta["delta_ci_lower"],
        "pr_auc_delta_ci_upper": pr_auc_delta["delta_ci_upper"],
        "brier_delta_obs": brier_delta["delta_obs"],
        "brier_delta_ci_lower": brier_delta["delta_ci_lower"],
        "brier_delta_ci_upper": brier_delta["delta_ci_upper"],
    }


def build_gate_inputs(
    ranking_metrics: dict[str, Any],
    classification_rows: list[dict[str, Any]],
    calibration_report: dict[str, Any],
    base_scenario_name: str,
    deltas: dict[str, float] | None,
) -> GateInputs:
    """Assemble the candidate's GateInputs from this module's own sealed-test outputs.

    recall is read from classification_rows at `base_scenario_name` — the
    recall guardrail is checked at the shipped operating point, not an
    arbitrary one. bss/calibration_slope come from
    sealed_test_calibration_report. deltas is None in the cold-start regime
    (nothing to compare against) and comparative_deltas' output otherwise —
    its keys already match GateInputs' *_delta_* field names, so they thread
    through directly.
    """
    recall = next(
        cast(float, row["recall"])
        for row in classification_rows
        if row["scenario"] == base_scenario_name
    )
    slope = calibration_report["calibration_slope"]
    delta_kwargs = deltas or {}
    return GateInputs(
        pr_auc=cast(float, ranking_metrics["pr_auc"]),
        recall=recall,
        bss=cast(float, calibration_report["bss"]),
        calibration_slope=cast(float, slope["slope"]),
        calibration_slope_ci_lower=cast(float, slope["slope_ci_lower"]),
        calibration_slope_ci_upper=cast(float, slope["slope_ci_upper"]),
        **delta_kwargs,
    )


def sealed_test_promotion_decision(
    y_test: pd.Series,
    candidate_proba: NDArray[np.float64],
    incumbent_proba: NDArray[np.float64] | None,
    ranking_metrics: dict[str, Any],
    classification_rows: list[dict[str, Any]],
    calibration_report: dict[str, Any],
    base_scenario_name: str,
    bars: GateBars,
    n_bootstrap: int,
    random_state: int,
) -> dict[str, Any]:
    """Call gate.py::decide_promotion the moment the sealed-test metrics exist —
    the point ANALYSIS.md §0 and CLAUDE.md fix as where the decision is made.

    incumbent_proba is None for a cold start (no champion yet, resolved by
    resolve_champion_version returning None); otherwise the champion's own
    probabilities on these exact sealed-test rows (load_incumbent_proba),
    scored once by the caller and passed in — never re-derived here — so the
    paired deltas this function computes are over the identical evaluation
    set the veto-only guardrails were also measured on. The caller persists
    the returned verdict to reports/promotion_decision.json; this function
    does not write anything.
    """
    deltas = (
        None
        if incumbent_proba is None
        else comparative_deltas(
            y_test, candidate_proba, incumbent_proba, n_bootstrap, random_state
        )
    )
    candidate_inputs = build_gate_inputs(
        ranking_metrics,
        classification_rows,
        calibration_report,
        base_scenario_name,
        deltas,
    )
    incumbent = None if incumbent_proba is None else _INCUMBENT_EXISTS_MARKER
    return decide_promotion(candidate_inputs, incumbent, bars)


# ---------------------------------------------------------------------------
# Step 6: MLflow orchestration
# ---------------------------------------------------------------------------


def content_hash(payload: dict[str, Any]) -> str:
    """sha256 of `payload`'s JSON-sorted-keys encoding.

    Embedded in promotion_decision.json so register.py can detect a decision
    belonging to a different metrics.json than the one it was computed from —
    same idiom as threshold.py's costs_config_hash, applied to a metrics
    payload instead of a config file. Public: register.py imports this same
    function to recompute the hash it verifies against, rather than
    reimplementing the encoding.
    """
    encoded = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _save_pr_curve_plot(
    pr_points: list[dict[str, float]],
    prevalence: float,
    thresholds: dict[str, float],
    path: Path,
) -> None:
    """PR curve with the prevalence baseline and all three scenario thresholds marked."""
    fig, ax = plt.subplots(figsize=(6, 5))
    recalls = [p["recall"] for p in pr_points]
    precisions = [p["precision"] for p in pr_points]
    ax.plot(recalls, precisions, label="model", color="tab:blue")
    ax.axhline(
        prevalence,
        color="gray",
        linestyle="--",
        label=f"prevalence = {prevalence:.3f}",
    )
    for name, t in thresholds.items():
        idx = min(
            range(len(pr_points)), key=lambda i: abs(pr_points[i]["threshold"] - t)
        )
        ax.scatter(
            [pr_points[idx]["recall"]],
            [pr_points[idx]["precision"]],
            zorder=3,
            label=f"{name} t*={t:.3f}",
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve — sealed test")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _save_roc_curve_plot(roc_points: list[dict[str, float]], path: Path) -> None:
    """ROC curve — a labelled diagnostic, never the headline (ANALYSIS.md §0:
    ROC-AUC is optimistic under this project's class imbalance)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(
        [p["fpr"] for p in roc_points],
        [p["tpr"] for p in roc_points],
        label="model",
        color="tab:blue",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve — sealed test (diagnostic; not gated)")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _save_classification_report_plot(
    classification_rows: list[dict[str, Any]], path: Path
) -> None:
    """One panel per scenario: confusion matrix (counts + row-normalised %) beside
    positive-class precision/recall/F1, support, and contact rate — the honest
    replacement for sklearn.classification_report (ANALYSIS.md §0/plots.py)."""
    n = len(classification_rows)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    axes_list = [axes] if n == 1 else list(axes)
    for ax, row in zip(axes_list, classification_rows, strict=True):
        matrix = np.array([[row["tp"], row["fn"]], [row["fp"], row["tn"]]])
        ax.imshow(matrix, cmap="Blues")
        labels = [
            [
                f"TP={row['tp']:.0f}\n({row['tp_pct']:.1%})",
                f"FN={row['fn']:.0f}\n({row['fn_pct']:.1%})",
            ],
            [
                f"FP={row['fp']:.0f}\n({row['fp_pct']:.1%})",
                f"TN={row['tn']:.0f}\n({row['tn_pct']:.1%})",
            ],
        ]
        for i in range(2):
            for j in range(2):
                ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pred. churn", "Pred. stay"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Actual churn", "Actual stay"])
        ax.set_title(
            f"{row['scenario']} (t={row['threshold']:.3f})\n"
            f"recall={row['recall']:.2f} precision={row['precision']:.2f} "
            f"f1={row['f1']:.2f}\ncontact_rate={row['contact_rate']:.1%}",
            fontsize=9,
        )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _save_reliability_plot_test(bins: list[dict[str, float]], path: Path) -> None:
    """Sealed-test reliability diagram — a distinct filename from Phase 6's
    reliability_diagram.png (the dev-OOF one), so the before/after comparison
    is never destroyed by an overwrite."""
    fig, ax = plt.subplots(figsize=(5, 5))
    means = [b["mean_predicted"] for b in bins]
    obs = [b["observed_frequency"] for b in bins]
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    ax.plot(means, obs, marker="o", label="sealed test", color="tab:blue")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed churn frequency")
    ax.set_title("Reliability diagram — sealed test")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _save_ev_by_budget_plot(
    ev_curves: dict[str, list[dict[str, float]]],
    thresholds: dict[str, float],
    contact_capacity: float,
    path: Path,
) -> None:
    """EV-vs-K budget curve, one line per scenario, with t*'s K, the
    EV-maximising K, and the capacity limit marked — "the single most
    business-legible artifact this project can produce" (PROJECT_PLAN.md):
    at any budget, the business reads off its own number, no new model or
    threshold required.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, rows in ev_curves.items():
        ks = [row["k"] for row in rows]
        evs = [row["ev_cumulative"] for row in rows]
        (line,) = ax.plot(ks, evs, label=name)
        t = thresholds[name]
        # rows are k-ascending / threshold-descending; the first row whose
        # threshold has dropped to (or below) t* is where contacting "iff
        # proba >= t*" lands on this curve.
        t_star_row = next((row for row in rows if row["threshold"] <= t), rows[-1])
        ax.axvline(t_star_row["k"], color=line.get_color(), linestyle=":", alpha=0.6)
        best = max(rows, key=lambda row: row["ev_cumulative"])
        ax.scatter(
            [best["k"]],
            [best["ev_cumulative"]],
            marker="*",
            s=90,
            color=line.get_color(),
            zorder=3,
        )
    ax.axvline(
        contact_capacity,
        color="black",
        linestyle="--",
        label=f"capacity = {contact_capacity:.0f}",
    )
    ax.set_xlabel("Number contacted (K)")
    ax.set_ylabel("Cumulative expected value ($)")
    ax.set_title(
        "EV vs. number contacted — sealed test (dotted = t*, star = EV-maximising K)"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _save_breakeven_heatmap_plot(
    twoway_rows: list[dict[str, float]],
    retention_rate_values: list[float],
    cost_values: list[float],
    path: Path,
) -> None:
    """The r x c break-even grid with the EV = 0 contour — the chart that tells
    a stakeholder "at a $20 contact cost, this campaign is profitable as long
    as we retain at least X% of the churners we call" (PROJECT_PLAN.md),
    read straight off the picture rather than computed by hand.
    """
    grid = np.array(
        [
            [
                next(
                    row["ev"]
                    for row in twoway_rows
                    if row["retention_rate"] == r and row["cost"] == c
                )
                for c in cost_values
            ]
            for r in retention_rate_values
        ]
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    mesh = ax.pcolormesh(
        cost_values, retention_rate_values, grid, shading="auto", cmap="RdYlGn"
    )
    ax.contour(
        cost_values,
        retention_rate_values,
        grid,
        levels=[0.0],
        colors="black",
        linewidths=2,
    )
    fig.colorbar(mesh, ax=ax, label="Expected value per customer ($)")
    ax.set_xlabel(
        "Cost per customer contacted (c) — outreach + retention-offer cost, $"
    )
    ax.set_ylabel("Retention rate (r)")
    ax.set_title("Break-even frontier — sealed test (base scenario)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _save_tornado_plot(tornado_rows: list[dict[str, Any]], path: Path) -> None:
    """Inputs ranked by |ev_high - ev_low| under a symmetric perturbation —
    makes ANALYSIS.md §0's claim that r dominates visible at a glance instead
    of asking the reader to take it on trust.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    params = [cast(str, row["param"]) for row in tornado_rows]
    lows = [cast(float, row["ev_low"]) for row in tornado_rows]
    highs = [cast(float, row["ev_high"]) for row in tornado_rows]
    base_ev = cast(float, tornado_rows[0]["ev_base"])
    y_pos = np.arange(len(params))
    for i, (lo, hi) in enumerate(zip(lows, highs, strict=True)):
        ax.barh(i, hi - lo, left=min(lo, hi), color="tab:blue", alpha=0.7)
    ax.axvline(
        base_ev,
        color="gray",
        linestyle="--",
        linewidth=0.8,
        label=f"base EV/customer = {base_ev:.0f}",
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(params)
    ax.invert_yaxis()
    ax.set_xlabel("Expected value per customer ($)")
    ax.set_title("Sensitivity tornado — sealed test (base scenario)")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _save_gains_lift_plot(decile_rows: list[dict[str, float]], path: Path) -> None:
    """Cumulative gains curve and per-decile lift bars — "how much churn does
    the top-k% capture?", threshold-free and complementary to PR-AUC.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    deciles = [row["decile"] for row in decile_rows]
    capture = [row["cumulative_capture_rate"] for row in decile_rows]
    perfect = [row["perfect_capture_rate"] for row in decile_rows]
    ax1.plot(
        [0, *deciles], [0, *perfect], linestyle="--", color="green", label="perfect"
    )
    ax1.plot([0, *deciles], [0, *capture], marker="o", label="model")
    ax1.plot([0, 10], [0, 1], linestyle="--", color="gray", label="random")
    ax1.set_xlabel("Decile (top-K x 10%)")
    ax1.set_ylabel("Cumulative share of churners captured")
    ax1.set_title("Cumulative gains — sealed test")
    ax1.legend()

    lifts = [row["lift"] for row in decile_rows]
    ax2.bar(deciles, lifts, color="tab:blue")
    ax2.axhline(1.0, color="gray", linestyle="--", label="random")
    ax2.set_xlabel("Decile")
    ax2.set_ylabel("Lift over base rate")
    ax2.set_title("Decile lift — sealed test")
    ax2.legend()

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def run_evaluation_step(model_version: str, cfg: DictConfig) -> dict[str, Any]:
    """Sealed-test evaluation, gate decision, and MLflow logging — the single
    pass CLAUDE.md's "test set touched once" invariant permits.

    Resolves `model_version` explicitly (never an alias), checks threshold
    provenance, scores the sealed test set exactly once, and from that one
    probability vector computes every block this module exposes: ranking,
    per-scenario classification, the fixed-recall profile, the calibration
    report, the decile lift table, the business-impact block, the
    sensitivity suite, the sealed-test disaggregated slices, and — from the
    dev-OOF probability vector calibrate.py already persisted, joined to the
    dev partition's segment columns — the V1/V2/V2b diagnostics (reported,
    non-gating). Resolves
    the incumbent champion (if any) once, scores it on the identical test
    rows, and calls gate.py::decide_promotion.

    Logs a dedicated `evaluation` run (a sibling of Phase 5/6's runs, never
    appended to the dev model's own run — CLAUDE.md), writes
    reports/metrics.json, reports/economics.json,
    reports/promotion_decision.json, and reports/test_predictions.parquet
    (mirrored onto the run as artifacts), tags the dev model version with the
    four gate criteria (alongside its existing training_data_scope: dev), and eight
    figures into the run's figures/ artifacts: the PR curve (operating points
    and prevalence baseline marked), the ROC curve, the classification-report
    panels, the sealed-test reliability diagram, the EV-vs-budget curve (t*
    and the EV-maximising K marked, one line per scenario), the r×c
    break-even heatmap, the sensitivity tornado diagram, and the cumulative
    gains/lift chart. Notebooks in this project never render their own
    matplotlib figures — 04-calibration-and-threshold.ipynb only
    mlflow.artifacts.download_artifacts + IPython.display.Image's what a
    pipeline module already produced — so this module is where every one of
    these figures is built; the eventual 05-evaluation-and-error-analysis.ipynb
    will only display them, following that same convention.
    """
    mlflow.set_tracking_uri(resolve_tracking_uri(str(cfg.mlflow.tracking_uri)))
    registered_model_name = str(cfg.mlflow.registered_model_name)

    run_id = resolve_model_run_id(model_version, cfg)
    validation_payload = load_threshold_validation(run_id, cfg)
    check_threshold_provenance(validation_payload, run_id, model_version)

    manifest = load_training_manifest(run_id, cfg)
    committed_features = committed_features_from_manifest(manifest)
    model = load_fitted_model(model_version, cfg)

    X_test, y_test = load_test_features(committed_features)
    proba: NDArray[np.float64] = model.predict_proba(X_test)[:, 1]
    customer_ids = load_test_customer_ids()

    policy = load_policy_thresholds(cfg)
    scenarios = resolve_policy_scenarios(policy)
    thresholds = resolve_policy_thresholds_by_scenario(policy)
    base_scenario = scenarios["base"]
    base_threshold = thresholds["base"]

    n_bootstrap = int(cfg.evaluate.n_bootstrap)
    random_state = int(cfg.evaluate.random_state)

    ranking_metrics = sealed_test_ranking_metrics(
        y_test, proba, n_bootstrap, random_state
    )
    classification_rows = sealed_test_classification_report(
        y_test, proba, thresholds, n_bootstrap, random_state
    )
    fixed_recall_rows = sealed_test_fixed_recall_profile(
        y_test, proba, [float(r) for r in cfg.training_setup.fixed_recall_thresholds]
    )
    calibration_report = sealed_test_calibration_report(
        y_test, proba, cfg, n_bootstrap, random_state
    )
    decile_rows = sealed_test_decile_lift(y_test, proba)
    business_impact = sealed_test_business_impact(
        y_test, proba, scenarios, thresholds, n_bootstrap, random_state
    )

    costs_cfg = load_costs_config(get_project_root() / str(cfg.paths.costs_config))
    retention_rate_values = [float(r) for r in costs_cfg.retention_rate_sweep]
    cost_values = [base_scenario.cost * m for m in (0.5, 1.0, 1.5, 2.0)]
    sensitivity = sealed_test_sensitivity_analysis(
        y_test,
        proba,
        base_scenario,
        base_threshold,
        retention_rate_values,
        cost_values,
        float(cfg.evaluate.tornado_pct_perturbation),
    )

    all_axes = ROBUSTNESS_AXES + FAIRNESS_AXES
    test_segment_lookup = load_test_segment_lookup()
    test_ranking_slices = sliced_ranking_metrics(
        y_test, proba, test_segment_lookup, all_axes, n_bootstrap, random_state
    )
    test_decision_slices = sliced_decision_rates(
        y_test, proba, test_segment_lookup, all_axes, base_threshold
    )
    test_calibration_slices = sliced_calibration(
        y_test, proba, test_segment_lookup, all_axes, cfg, n_bootstrap, random_state
    )
    test_business_impact_slices = sliced_business_impact(
        y_test,
        proba,
        test_segment_lookup,
        FAIRNESS_AXES,
        base_scenario,
        base_threshold,
    )
    test_fairness_decision_rows = [
        row for row in test_decision_slices if row["axis"] in FAIRNESS_AXES
    ]
    test_equal_opportunity_by_axis = equal_opportunity_difference_by_axis(
        test_fairness_decision_rows
    )
    test_demographic_parity_by_axis = demographic_parity_difference_by_axis(
        test_fairness_decision_rows
    )
    test_equal_opportunity_diff = max(
        (v for v in test_equal_opportunity_by_axis.values() if not math.isnan(v)),
        default=float("nan"),
    )
    test_demographic_parity_diff = max(
        (v for v in test_demographic_parity_by_axis.values() if not math.isnan(v)),
        default=float("nan"),
    )

    bars = load_model_promotion_bars(cfg)

    # V1/V2/V2b are computed by threshold.py's dev-OOF screen, Phase 6's last
    # step — fetched here by run_id from its logged MLflow artifact, unchanged,
    # never recomputed, so the dev-OOF surface has exactly one owner.
    dev_oof_diagnostics = load_dev_oof_diagnostics(run_id, cfg)

    champion_version = resolve_champion_version(cfg)
    incumbent_proba = (
        None
        if champion_version is None
        else load_incumbent_proba(champion_version, cfg)
    )
    decision = sealed_test_promotion_decision(
        y_test,
        proba,
        incumbent_proba,
        ranking_metrics,
        classification_rows,
        calibration_report,
        "base",
        bars,
        n_bootstrap,
        random_state,
    )

    metrics_payload: dict[str, Any] = {
        "model_version": model_version,
        "run_id": run_id,
        "champion_version": champion_version,
        "ranking": ranking_metrics,
        "classification": classification_rows,
        "fixed_recall_profile": fixed_recall_rows,
        "calibration": calibration_report,
        "decile_lift": decile_rows,
        "business_impact": business_impact,
        "sliced": {
            "test": {
                "ranking": test_ranking_slices,
                "decision_rates": test_decision_slices,
                "calibration": test_calibration_slices,
                "business_impact": test_business_impact_slices,
                "equal_opportunity_difference_by_axis": test_equal_opportunity_by_axis,
                "demographic_parity_difference_by_axis": test_demographic_parity_by_axis,
                "equal_opportunity_diff": test_equal_opportunity_diff,
                "demographic_parity_diff": test_demographic_parity_diff,
            },
            "dev_oof_diagnostics": dev_oof_diagnostics,
        },
    }
    y_test_int = y_test.to_numpy(dtype=np.int64)
    ev_curves = {
        name: ev_by_k(proba, y_test_int, scenario)
        for name, scenario in scenarios.items()
    }

    economics_payload: dict[str, Any] = {
        "sensitivity": sensitivity,
        "ev_by_k": ev_curves,
        "ev_treat_all_by_scenario": {
            name: row["ev_treat_all"]
            for name, row in business_impact["scenarios"].items()
        },
        "retention_rate_values_swept": retention_rate_values,
        "cost_values_swept": cost_values,
    }
    promotion_decision_payload: dict[str, Any] = {
        **decision,
        "model_version": model_version,
        "metrics_content_hash": content_hash(metrics_payload),
    }

    figures_dir = get_project_root() / str(cfg.paths.figures)
    pr_curve_path = figures_dir / "pr_curve_test.png"
    roc_curve_path = figures_dir / "roc_curve_test.png"
    classification_report_path = figures_dir / "classification_report_test.png"
    reliability_path = figures_dir / "reliability_diagram_test.png"
    ev_by_budget_path = figures_dir / "ev_by_budget.png"
    breakeven_heatmap_path = figures_dir / "breakeven_heatmap.png"
    sensitivity_tornado_path = figures_dir / "sensitivity_tornado.png"
    gains_lift_path = figures_dir / "gains_lift_test.png"

    _save_pr_curve_plot(
        pr_curve_points(y_test.tolist(), proba.tolist()),
        float(y_test.mean()),
        thresholds,
        pr_curve_path,
    )
    _save_roc_curve_plot(
        roc_curve_points(y_test.tolist(), proba.tolist()), roc_curve_path
    )
    _save_classification_report_plot(classification_rows, classification_report_path)
    _save_reliability_plot_test(
        cast(list[dict[str, float]], calibration_report["reliability_bins"]),
        reliability_path,
    )
    _save_ev_by_budget_plot(
        ev_curves, thresholds, float(costs_cfg.contact_capacity), ev_by_budget_path
    )
    _save_breakeven_heatmap_plot(
        cast(list[dict[str, float]], sensitivity["twoway"]),
        retention_rate_values,
        cost_values,
        breakeven_heatmap_path,
    )
    _save_tornado_plot(
        cast(list[dict[str, Any]], sensitivity["tornado"]), sensitivity_tornado_path
    )
    _save_gains_lift_plot(decile_rows, gains_lift_path)

    ensure_experiment_metadata(cfg)
    model_id = resolve_logged_model_id(model_version, cfg)
    test_dataset = mlflow_dataset_from_pandas(
        pd.concat(
            [X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1
        ),
        name="sealed_test",
        targets=TARGET_COL,
    )

    with mlflow.start_run(run_name="evaluation") as run:
        set_run_description(_RUN_DESCRIPTION)
        eval_run_id = run.info.run_id
        # Needed so a later re-log (e.g. the review notebook stamping `review`
        # onto this same payload) targets this run, not `run_id` above (the
        # evaluated model's own training run) — the two are never the same run.
        promotion_decision_payload["eval_run_id"] = eval_run_id
        mlflow.log_input(test_dataset, context="evaluation")

        scalar_metrics: dict[str, float] = {
            "test_pr_auc": cast(float, ranking_metrics["pr_auc"]),
            "test_pr_auc_ci_lower": cast(float, ranking_metrics["pr_auc_ci_lower"]),
            "test_pr_auc_ci_upper": cast(float, ranking_metrics["pr_auc_ci_upper"]),
            "test_roc_auc": cast(float, ranking_metrics["roc_auc"]),
            "test_dummy_pr_auc_floor": cast(
                float, ranking_metrics["dummy_pr_auc_floor"]
            ),
            "test_brier": cast(float, calibration_report["brier"]),
            "test_bss": cast(float, calibration_report["bss"]),
            "test_ece": cast(float, calibration_report["ece"]),
            "test_calibration_slope": cast(
                float, calibration_report["calibration_slope"]["slope"]
            ),
            "test_calibration_slope_ci_lower": cast(
                float, calibration_report["calibration_slope"]["slope_ci_lower"]
            ),
            "test_calibration_slope_ci_upper": cast(
                float, calibration_report["calibration_slope"]["slope_ci_upper"]
            ),
            "test_equal_opportunity_diff": test_equal_opportunity_diff,
            "test_demographic_parity_diff": test_demographic_parity_diff,
        }
        for row in classification_rows:
            scenario_name = cast(str, row["scenario"])
            for metric_key in ("recall", "precision", "f1", "contact_rate"):
                scalar_metrics[f"test_{metric_key}_{scenario_name}"] = cast(
                    float, row[metric_key]
                )
            for metric_key in ("recall", "precision", "f1"):
                for bound in ("ci_lower", "ci_upper"):
                    scalar_metrics[f"test_{metric_key}_{scenario_name}_{bound}"] = cast(
                        float, row[f"{metric_key}_{bound}"]
                    )
        for scenario_name, row in business_impact["scenarios"].items():
            scalar_metrics[f"test_ev_{scenario_name}"] = cast(float, row["ev"])
            scalar_metrics[f"test_ev_{scenario_name}_ci_lower"] = cast(
                float, row["ev_ci_lower"]
            )
            scalar_metrics[f"test_ev_{scenario_name}_ci_upper"] = cast(
                float, row["ev_ci_upper"]
            )
            scalar_metrics[f"test_campaign_cost_{scenario_name}"] = cast(
                float, row["campaign_cost"]
            )
            scalar_metrics[f"test_retained_revenue_{scenario_name}"] = cast(
                float, row["retained_revenue"]
            )
            scalar_metrics[f"test_n_contacted_{scenario_name}"] = cast(
                float, row["n_contacted"]
            )
            scalar_metrics[f"test_break_even_retention_rate_{scenario_name}"] = cast(
                float, row["break_even_retention_rate"]
            )
            scalar_metrics[f"test_ev_treat_all_{scenario_name}"] = cast(
                float, row["ev_treat_all"]
            )
            scalar_metrics[f"test_ev_treat_none_{scenario_name}"] = cast(
                float, row["ev_treat_none"]
            )
        mlflow.log_metrics(scalar_metrics, model_id=model_id, dataset=test_dataset)

        for scenario_name, scenario in scenarios.items():
            mlflow.log_params(
                {
                    f"cost_{scenario_name}_c": scenario.cost,
                    f"cost_{scenario_name}_r": scenario.retention_rate,
                    f"cost_{scenario_name}_ltv": scenario.ltv,
                    f"cost_{scenario_name}_arpu": scenario.arpu,
                }
            )
        mlflow.log_param("gross_margin", float(costs_cfg.gross_margin))
        mlflow.log_param("contact_capacity", int(costs_cfg.contact_capacity))
        mlflow.set_tag(
            "costs_config_hash",
            costs_config_hash(get_project_root() / str(cfg.paths.costs_config)),
        )

        mlflow.set_tag("gate_regime", decision["regime"])
        mlflow.set_tag("gate_result", decision["gate"])
        for criterion_name, criterion in decision["criteria"].items():
            mlflow.set_tag(
                f"gate_criterion_{criterion_name}",
                "pass" if criterion["passed"] else "fail",
            )

        mlflow.log_dict(metrics_payload, "metrics.json")
        mlflow.log_dict(economics_payload, "economics.json")
        mlflow.log_dict(promotion_decision_payload, "promotion_decision.json")

        for path in (
            pr_curve_path,
            roc_curve_path,
            classification_report_path,
            reliability_path,
            ev_by_budget_path,
            breakeven_heatmap_path,
            sensitivity_tornado_path,
            gains_lift_path,
        ):
            mlflow.log_artifact(str(path), artifact_path="figures")

        with tempfile.TemporaryDirectory() as tmp_dir:
            predictions_path = Path(tmp_dir) / "test_predictions.parquet"
            test_predictions = pd.DataFrame(
                {
                    "customerid": customer_ids,
                    "y_true": y_test.reset_index(drop=True),
                    "p_hat": proba,
                }
            )
            test_predictions.to_parquet(predictions_path, index=False)
            mlflow.log_artifact(str(predictions_path))

    client = mlflow.tracking.MlflowClient()
    client.set_model_version_tag(
        registered_model_name,
        model_version,
        "test_pr_auc",
        str(ranking_metrics["pr_auc"]),
    )
    client.set_model_version_tag(
        registered_model_name,
        model_version,
        "test_recall",
        str(
            next(
                row["recall"]
                for row in classification_rows
                if row["scenario"] == "base"
            )
        ),
    )
    client.set_model_version_tag(
        registered_model_name,
        model_version,
        "test_brier",
        str(calibration_report["brier"]),
    )
    client.set_model_version_tag(
        registered_model_name,
        model_version,
        "test_calibration_slope",
        str(calibration_report["calibration_slope"]["slope"]),
    )
    # register.py's only supported path from "the model version being
    # registered" to this cycle's promotion_decision.json/metrics.json/
    # economics.json/test_predictions.parquet — ModelVersion.model_id doesn't
    # auto-populate in OSS MLflow 3.14, and the same is true of any other
    # run-to-version link, so it must be persisted deliberately (mirrors
    # calibrate.py's logged_model_id tag). A local reports/ path is not a
    # substitute: it only reflects whichever run last executed evaluate.py on
    # this machine, not necessarily this model_version's own cycle.
    client.set_model_version_tag(
        registered_model_name, model_version, "eval_run_id", eval_run_id
    )

    reports_dir = get_project_root() / str(cfg.paths.reports)
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, default=str)
    with open(reports_dir / "economics.json", "w", encoding="utf-8") as f:
        json.dump(economics_payload, f, indent=2, default=str)
    with open(reports_dir / "promotion_decision.json", "w", encoding="utf-8") as f:
        json.dump(promotion_decision_payload, f, indent=2, default=str)
    test_predictions.to_parquet(reports_dir / "test_predictions.parquet", index=False)
    # reports/dev_oof_predictions.parquet and dev_oof_diagnostics.json are not
    # written here — threshold.py's dev-OOF screen (Phase 6's last step)
    # already wrote both; this module only ever fetches the latter (by run_id,
    # via load_dev_oof_diagnostics), never produces either.

    logger.info(
        "evaluation_step_done",
        run_id=run_id,
        eval_run_id=eval_run_id,
        model_version=model_version,
        champion_version=champion_version,
        gate_regime=decision["regime"],
        gate_result=decision["gate"],
    )

    return {
        "eval_run_id": eval_run_id,
        "model_version": model_version,
        "champion_version": champion_version,
        "metrics": metrics_payload,
        "economics": economics_payload,
        "promotion_decision": promotion_decision_payload,
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
        cli_model_version = cfg.evaluate.model_version
        if cli_model_version is None:
            raise ValueError(
                "evaluate.model_version is required, e.g. `python -m "
                "telco_churn.models.evaluate evaluate.model_version=1`"
            )
        result = run_evaluation_step(str(cli_model_version), cfg)
        logger.info(
            "evaluation_step_done",
            model_version=result["model_version"],
            eval_run_id=result["eval_run_id"],
            gate_result=result["promotion_decision"]["gate"],
        )
    except FileNotFoundError as e:
        logger.error("evaluation_data_not_found", error=str(e), exc_info=True)
        sys.exit(1)
    except pa.errors.SchemaError as e:
        logger.error("evaluation_data_schema_invalid", error=str(e), exc_info=True)
        sys.exit(1)
    except ValueError as e:
        logger.error("evaluation_invalid", error=str(e), exc_info=True)
        sys.exit(1)
    except AssertionError as e:
        logger.error("evaluation_assertion_failed", error=str(e), exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error("evaluation_failed", error=str(e), exc_info=True)
        sys.exit(1)

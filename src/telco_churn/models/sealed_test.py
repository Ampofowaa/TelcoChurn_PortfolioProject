"""Dataset-agnostic sealed-test scoring and comparative-gate primitives.

Extracted out of evaluate.py (Phase 10a-ii) so pipelines/performance_check.py
(Phase 10b) can share them without
importing from a __main__-bearing module (CLAUDE.md's
test_no_module_imports_from_a_dunder_main_bearing_module rule) — evaluate.py
still owns the __main__ CLI for the rare/cold-start path, now a thin wrapper
calling into this module rather than defining these functions itself.

CLAUDE.md's modelling invariant: X_test/y_test are imported and used in
exactly two places — this module (the sole importer of the test side of
telco_churn.data.split.partition()) and evaluate.py's own thin __main__
wrapper over it. Every other module reaches test data solely through
persisted artifacts (reports/test_predictions.parquet), never through
telco_churn.data.split directly.

load_incumbent_proba/resolve_incumbent_summary/resolve_evaluation_champion
stay in evaluate.py itself — they power only the rare comparative regime's
historical-alignment machinery (gates on an exact customerid-set match
against the champion's historical test_predictions.parquet), which
performance_check.py never needs, since its comparison cohort is a fresh
reserve cohort every cycle, not the sealed test set.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from omegaconf import DictConfig
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from telco_churn.data.split import partition, sealed_test_ids
from telco_churn.features.accessor import load_features
from telco_churn.features.build import TARGET_COL
from telco_churn.models.calibration_metrics import (
    brier_skill_score,
    calibration_slope,
    expected_calibration_error,
    murphy_decomposition,
    pooled_brier,
)
from telco_churn.models.diagnostics import build_segment_lookup, fixed_recall_profile
from telco_churn.models.economics import (
    break_even_retention_rate,
    campaign_cost,
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
    reliability_diagram_bins,
)
from telco_churn.models.policy_config import CostScenario
from telco_churn.utils.stats import (
    bootstrap_metric_ci,
    paired_bootstrap_ci,
    paired_bootstrap_metric_ci,
)

__all__ = [
    "build_gate_inputs",
    "comparative_deltas",
    "load_test_customer_ids",
    "load_test_features",
    "load_test_segment_lookup",
    "sealed_test_business_impact",
    "sealed_test_calibration_report",
    "sealed_test_classification_report",
    "sealed_test_decile_lift",
    "sealed_test_fixed_recall_profile",
    "sealed_test_promotion_decision",
    "sealed_test_ranking_metrics",
    "sealed_test_sensitivity_analysis",
    "sliced_business_impact",
]

# Contacting "everyone"/"no one" is expressed as an extreme threshold on the
# real proba vector rather than a fitted DummyClassifier: since proba is
# always in [0, 1], any cut at or below the minimum score contacts every row
# and any cut above 1.0 contacts none — a function of the labels alone, not
# of proba's actual values, so this gives the identical dollar figure a
# DummyClassifier(strategy='constant', ...) baseline would.
_CONTACT_ALL_THRESHOLD = 0.0
_CONTACT_NONE_THRESHOLD = 1.0 + 1e-9


def _load_sealed_test_partition() -> pd.DataFrame:
    """Return the sealed-test-partition rows (customerid included), pre feature-subsetting.

    This function, and load_test_features/load_test_customer_ids below it, are
    the only call sites in src/ that read the test side of
    telco_churn.data.split.partition() — the structural half of "test set
    touched once": no other module imports data.split for the test partition.

    Returns test() minus the reserve months (data.split::sealed_test_ids())
    — strictly fewer rows than the full historical ~1,409-row test partition
    once any reserve cohort exists. The
    reserve exclusion lives in sealed_test_ids() itself, not here — this
    function only joins the validated test_df down to that id set, so "who's
    in the sealed test set" stays defined in exactly one place (data/split.py,
    next to dev_ids/test_ids/reserve_ids), never re-derived here.
    """
    df = load_features()
    _dev_df, test_df = partition(df)
    sealed_ids = set(sealed_test_ids())
    return test_df[test_df["customerid"].isin(sealed_ids)].reset_index(drop=True)


def load_test_features(
    committed_features: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """Load the sealed-test rows, restricted to the frozen committed feature set."""
    test_df = _load_sealed_test_partition()
    return test_df[committed_features], test_df[TARGET_COL]


def load_test_customer_ids() -> pd.Series:
    """Return the customerid Series for the sealed test partition.

    Row-order-aligned with load_test_features's (X_test, y_test) — both derive
    from the same _load_sealed_test_partition() call, the same
    recompute-rather-than-thread-state idiom calibrate.py's dev-side loaders
    already use.
    """
    return _load_sealed_test_partition()["customerid"].reset_index(drop=True)


def load_test_segment_lookup() -> dict[str, pd.Series]:
    """Return the sealed-test robustness/fairness segment axes.

    Row-order-aligned with load_test_features/load_test_customer_ids — all
    three derive from the same _load_sealed_test_partition() call.
    """
    return build_segment_lookup(_load_sealed_test_partition())


def sealed_test_business_impact(
    y_test: pd.Series,
    proba: NDArray[np.float64],
    scenarios: dict[str, CostScenario],
    thresholds: dict[str, float],
    n_bootstrap: int,
    random_state: int,
) -> dict[str, Any]:
    """Per-scenario dollar impact at the shipped operating point, plus the two policy baselines.

    Reports net EV (with a bootstrap CI), gross campaign cost, gross
    retained revenue, contacted count/rate, and break-even retention rate
    per scenario — the full picture, since "this costs $C and returns $R"
    is what Finance asks for and net EV alone can't answer it. Also reports
    each scenario's ev_treat_all/ev_treat_none baselines (contact
    everyone/no one); the model is expected to beat both.

    ⚠ ev_bracket is the min-max across scenarios' point estimates, not a
    statistical interval — the three cost scenarios are not an ordered
    min/mid/max (their parameters partially offset; ANALYSIS.md §0), so this
    brackets whatever they turn out to be. parameter_spread_dominates_sampling
    flags whether that bracket exceeds the widest within-scenario bootstrap
    CI: False means configs/costs.yaml's scenarios are too narrow to express
    the real uncertainty in r, hiding the EV bracket's dominant error source.
    This function only computes the diagnostic; the caller decides whether
    to warn on it.
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
    ece_n_bins = int(cfg.calibration.ece_n_bins)
    ece_strategy = str(cfg.calibration.ece_strategy)

    dummy = DummyClassifier(strategy="prior")
    dummy_x = np.zeros((len(y_test), 1))
    dummy.fit(dummy_x, y_test)
    dummy_proba = dummy.predict_proba(dummy_x)[:, 1]
    reference_brier = pooled_brier(dummy_proba, y_test)

    return {
        "brier": candidate_brier,
        "dummy_prior_brier": reference_brier,
        "bss": brier_skill_score(candidate_brier, reference_brier),
        "ece": expected_calibration_error(
            p, y_test, ece_n_bins, ece_strategy, "test_candidate"
        ),
        "murphy_decomposition": murphy_decomposition(
            p, y_test, ece_n_bins, ece_strategy
        ),
        "calibration_slope": calibration_slope(y_test, p, n_bootstrap, random_state),
        "reliability_bins": reliability_diagram_bins(
            p.tolist(),
            y_test.tolist(),
            n_bins=ece_n_bins,
            strategy=ece_strategy,
        ),
    }


def sealed_test_decile_lift(
    y_test: pd.Series, proba: NDArray[np.float64]
) -> list[dict[str, float]]:
    """Sealed-test decile lift/gains table — thin wrapper around plots.decile_lift_table."""
    return decile_lift_table(y_test.tolist(), proba.tolist())


# The V1/V2/V2b slicing helpers (and their collapse flags) live in
# diagnostics.py, owned by threshold.py's dev-OOF screen. This module only
# re-imports build_segment_lookup for sealed-test segment axes and
# fixed_recall_profile for the threshold-planning tool above; evaluate.py's
# own stays-in-place orchestration imports sliced_calibration/
# sliced_decision_rates/sliced_ranking_metrics directly from diagnostics.py
# for its own sliced-diagnostics step.


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


def comparative_deltas(
    y_test: pd.Series,
    candidate_proba: NDArray[np.float64],
    incumbent_proba: NDArray[np.float64],
    base_threshold: float,
    n_bootstrap: int,
    random_state: int,
) -> dict[str, float]:
    """Paired-bootstrap Δ = candidate − incumbent on PR-AUC, Brier, and recall
    at the shipped operating threshold, over the identical sealed-test rows —
    the three fields gate.py's comparative regime needs and cannot compute
    itself (it never sees the probability vectors; see GateInputs' own
    docstring).

    PR-AUC has no per-row decomposition, so its Δ goes through
    paired_bootstrap_metric_ci (row-resampled, recomputing average_precision_
    score on each resampled set for both models). Brier is a per-row proper
    score, so its Δ goes through the cheaper paired_bootstrap_ci directly on
    each model's own per-row squared error — the same distinction utils.stats'
    two functions exist to make. Recall, like PR-AUC, is a ratio (TP / (TP +
    FN)) with no per-row decomposition, so it also goes through
    paired_bootstrap_metric_ci.

    base_threshold is the "base" scenario's t* — the same shipped operating
    point build_gate_inputs reads candidate recall from — applied to *both*
    models, not each one's own threshold: t* = c/(r×LTV) (threshold.py) is a
    pure function of configs/costs.yaml, not of any model's own probability
    distribution, so a single shared threshold is the theoretically correct
    comparison, not an approximation. This also means the comparison is
    against the incumbent's recall *today*, under this cycle's cost
    assumptions, rather than a stale number from whichever cycle it was last
    evaluated in — consistent with costs_config_hash tracking assumption
    changes separately from model changes elsewhere in this module.
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
    recall_delta = paired_bootstrap_metric_ci(
        y,
        candidate_proba,
        incumbent_proba,
        lambda y_arr, p_arr: _precision_recall_f1_at(y_arr, p_arr, base_threshold)[1],
        n_bootstrap,
        random_state,
    )
    return {
        "pr_auc_delta_obs": pr_auc_delta["delta_obs"],
        "pr_auc_delta_ci_lower": pr_auc_delta["delta_ci_lower"],
        "pr_auc_delta_ci_upper": pr_auc_delta["delta_ci_upper"],
        "brier_delta_obs": brier_delta["delta_obs"],
        "brier_delta_ci_lower": brier_delta["delta_ci_lower"],
        "brier_delta_ci_upper": brier_delta["delta_ci_upper"],
        "recall_delta_obs": recall_delta["delta_obs"],
        "recall_delta_ci_lower": recall_delta["delta_ci_lower"],
        "recall_delta_ci_upper": recall_delta["delta_ci_upper"],
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
    resolve_evaluation_champion returning None); otherwise the champion's own
    historical sealed-test probabilities, aligned onto these exact rows
    (load_incumbent_proba, which reads them off the champion's own eval run
    rather than re-scoring it), read once by the caller and passed in — never
    re-derived here — so the paired deltas this function computes are over
    the identical evaluation set the veto-only guardrails were also measured
    on. The caller persists the returned verdict to
    reports/promotion_decision.json; this function does not write anything.
    """
    deltas = None
    if incumbent_proba is not None:
        base_threshold = next(
            cast(float, row["threshold"])
            for row in classification_rows
            if row["scenario"] == base_scenario_name
        )
        deltas = comparative_deltas(
            y_test,
            candidate_proba,
            incumbent_proba,
            base_threshold,
            n_bootstrap,
            random_state,
        )
    candidate_inputs = build_gate_inputs(
        ranking_metrics,
        classification_rows,
        calibration_report,
        base_scenario_name,
        deltas,
    )
    regime: Literal["cold_start", "comparative"] = (
        "cold_start" if incumbent_proba is None else "comparative"
    )
    return decide_promotion(candidate_inputs, regime, bars)

"""Derive and validate the cost-sensitive decision threshold t* = c / (r × LTV).

Contact a customer iff q·r·LTV > c (q = calibrated churn probability, r =
retention success rate, LTV = discounted lifetime value, c = intervention
cost); t* is the break-even q. This supersedes the classical
C_FP/(C_FP + C_FN) rule, which assumes correct decisions are free — here
cost attaches to the action of contacting, not to being wrong.

Leak-free by construction: no unconditional `.fit(` call and no sklearn
estimator import. `run_threshold_step` only reads models.calibrate's
already-computed OOF arrays and telco_churn.data.split's segment/protected
columns — never the test partition.

Also runs Phase 7's pre-seal dev-OOF screen as its last step. Three
*binding* checks, all computed here, before the sealed test set is ever
touched: re-screens calibrate.py's logged aggregate calibration slope
against ANALYSIS.md §0's band; direction_sanity (formerly `v3_direction_sanity`,
still cross-referenced as V3 in ANALYSIS.md's guardrail taxonomy) — ranks
calibrate.py's logged dev-SHAP summary (dev_shap_summary.json), cuts at
`direction_sanity_top_k_features`, and checks each surviving feature's
direction against explain.EXPECTED_EDA_DIRECTIONS (no shap import here;
calibrate.py already computed the SHAP values); and within_ci — whether the
closed-form t* falls inside the base scenario's empirical argmax-EV
bootstrap CI, added to catch a retrain cycle where the inherited operating
point no longer agrees with
where expected value is empirically maximised. Also computes V1 (segment
collapse), V2 (fairness disparity), and V2b (per-group calibration collapse)
on the same dev-OOF surface — reported-only, never gating (CLAUDE.md's
three-guardrail rule) — writing reports/dev_oof_predictions.parquet and
reports/dev_oof_diagnostics.json for evaluate.py, error_analysis.py,
register.py, and drift_reference.py to read back. Raises RuntimeError, after
logging, if any of the three binding checks fails.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import mlflow
import mlflow.artifacts
import mlflow.tracking
import numpy as np
import pandas as pd
from mlflow.data.pandas_dataset import from_pandas as mlflow_dataset_from_pandas
from mlflow.exceptions import MlflowException
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf

from telco_churn.models.artifacts import (
    committed_features_from_manifest,
    load_dev_oof_predictions,
    load_training_manifest,
)
from telco_churn.models.dev_features import (
    load_dev_customer_ids,
    load_dev_features,
    load_dev_partition,
)
from telco_churn.models.diagnostics import (
    FAIRNESS_AXES,
    ROBUSTNESS_AXES,
    build_segment_lookup,
    demographic_parity_difference_by_axis,
    equal_opportunity_difference_by_axis,
    flag_calibration_collapse,
    flag_segment_collapse,
    sliced_calibration,
    sliced_decision_rates,
    sliced_ranking_metrics,
)
from telco_churn.models.explain import (
    EXPECTED_EDA_DIRECTIONS,
    check_top_k_elbow,
    direction_sanity_check,
)
from telco_churn.models.gate import slope_passes
from telco_churn.models.policy_config import (
    CostScenario,
    costs_config_hash,
    expected_value_at_threshold,
    load_costs_config,
    load_model_promotion_bars,
)
from telco_churn.utils.logging import get_logger
from telco_churn.utils.mlflow import (
    TRAINING_CYCLE_RUN_DESCRIPTION,
    ensure_experiment_metadata,
    resolve_logged_model_id,
    resolve_model_identifier,
    resolve_model_run_id,
    resolve_tracking_uri,
    set_run_description,
    write_threshold_rerun_receipt,
)
from telco_churn.utils.paths import get_project_root

__all__ = [
    "arpu_by_scenario",
    "resolve_scenario",
    "resolve_all_scenarios",
    "closed_form_threshold",
    "expected_value_curve",
    "empirical_argmax_threshold",
    "argmax_bootstrap_ci",
    "implied_contact_rate",
    "r_sensitivity_sweep",
    "derive_threshold",
    "load_calibration_summary",
    "load_dev_shap_summary",
    "build_dev_oof_screen_frame",
    "compute_dev_oof_diagnostics",
    "run_threshold_step",
    "run_threshold_rerun_step",
]

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Cost/ARPU resolution
# ---------------------------------------------------------------------------


def arpu_by_scenario(
    monthlycharges_dev: pd.Series, y_dev: pd.Series, costs_cfg: DictConfig
) -> dict[str, float]:
    """Return {scenario_name: ARPU}, each the churner MonthlyCharges quantile costs_cfg.arpu_quantile names.

    Computed on development-set churners only — never the sealed test set.
    """
    churner_charges = monthlycharges_dev[y_dev == 1]
    return {
        str(name): float(churner_charges.quantile(float(q)))
        for name, q in costs_cfg.arpu_quantile.items()
    }


def resolve_scenario(name: str, arpu: float, costs_cfg: DictConfig) -> CostScenario:
    """Resolve one named scenario's LTV and intervention cost from its ARPU and configs/costs.yaml."""
    scenario_cfg = costs_cfg.scenarios[name]
    ltv = arpu * float(costs_cfg.gross_margin) * float(costs_cfg.horizon_months)
    offer_value = (
        arpu * float(scenario_cfg.discount_rate) * float(costs_cfg.discount_months)
    )
    cost = float(scenario_cfg.outreach_cost) + offer_value
    return CostScenario(
        name=name,
        arpu=arpu,
        ltv=ltv,
        cost=cost,
        retention_rate=float(scenario_cfg.retention_rate),
    )


def resolve_all_scenarios(
    monthlycharges_dev: pd.Series, y_dev: pd.Series, costs_cfg: DictConfig
) -> dict[str, CostScenario]:
    """Resolve every scenario named in configs/costs.yaml (conservative/base/optimistic)."""
    arpu = arpu_by_scenario(monthlycharges_dev, y_dev, costs_cfg)
    return {
        name: resolve_scenario(name, arpu[name], costs_cfg)
        for name in costs_cfg.scenarios
    }


# ---------------------------------------------------------------------------
# Threshold math — pure: arrays and parameters in, no estimator, no data.split
# ---------------------------------------------------------------------------


def closed_form_threshold(scenario: CostScenario) -> float:
    """t* = cost / (retention_rate × LTV).

    Raises ValueError rather than emitting inf/nan or a nonsensical t*: a
    zero or negative retention_rate is a costs.yaml typo, not a code bug, and
    c >= retention_rate*LTV (t* >= 1, "never contact anyone") is an equally
    plausible typo that would otherwise ship a model that never fires.
    """
    if scenario.retention_rate <= 0:
        raise ValueError(
            f"retention_rate must be > 0 for scenario {scenario.name!r}; "
            f"got {scenario.retention_rate}. Check configs/costs.yaml."
        )
    t_star = scenario.cost / (scenario.retention_rate * scenario.ltv)
    if not (0.0 < t_star < 1.0):
        raise ValueError(
            f"Closed-form threshold for scenario {scenario.name!r} is "
            f"{t_star:.4f}, outside (0, 1) — check configs/costs.yaml for a "
            "cost/LTV/retention_rate typo (t* >= 1 means 'never contact "
            "anyone')."
        )
    return float(t_star)


def expected_value_curve(
    oof_proba: NDArray[np.float64], y_dev: NDArray[np.int_], scenario: CostScenario
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return (thresholds, ev) — realized per-customer expected value of the "contact if and only if proba >= t" rule.

    ev(t) = [TP(t)·(r·LTV − c) − FP(t)·c] / n. Computed via a single sorted
    cumulative-sum pass (O(n log n)), not a per-threshold mask loop, so a
    1,000-resample bootstrap over ~5,600 rows stays fast. One point per
    distinct observed probability — a tied group can only be contacted
    all-or-nothing, so a threshold strictly between tied values would be
    fiction.
    """
    order = np.argsort(-oof_proba, kind="stable")
    proba_sorted = oof_proba[order]
    y_sorted = y_dev[order]
    n = len(y_sorted)

    tp_cum = np.cumsum(y_sorted)
    n_contacted = np.arange(1, n + 1)
    fp_cum = n_contacted - tp_cum
    ev = (
        tp_cum * (scenario.retention_rate * scenario.ltv - scenario.cost)
        - fp_cum * scenario.cost
    ) / n

    keep = np.r_[np.diff(proba_sorted) != 0, True]
    thresholds = proba_sorted[keep]
    ev = ev[keep]

    # Prepend the "contact no one" reference point (ev = 0 by definition).
    thresholds = np.concatenate([[1.0], thresholds])
    ev = np.concatenate([[0.0], ev])
    return thresholds, ev


def empirical_argmax_threshold(
    oof_proba: NDArray[np.float64], y_dev: NDArray[np.int_], scenario: CostScenario
) -> float:
    """The threshold maximizing realized expected value over expected_value_curve.

    A diagnostic, not the selector: under correct calibration this should
    agree with closed_form_threshold(scenario); a material gap is evidence
    the probabilities are not calibrated near the operating point.
    """
    thresholds, ev = expected_value_curve(oof_proba, y_dev, scenario)
    return float(thresholds[int(np.argmax(ev))])


def argmax_bootstrap_ci(
    oof_proba: NDArray[np.float64],
    y_dev: NDArray[np.int_],
    scenario: CostScenario,
    n_bootstrap: int,
    random_state: int,
) -> tuple[float, float]:
    """95% percentile bootstrap CI of the empirical argmax, resampling OOF rows with replacement.

    t* has zero sampling variance (it is a closed-form function of the cost
    parameters alone) — there is nothing to stabilise there. What this CI
    quantifies is how noisy the *empirical* argmax is on this sample, so the
    agreement check (t* falls inside this CI) has the right null: a
    point-vs-point comparison would fail spuriously on the curve's flat top.
    """
    rng = np.random.default_rng(random_state)
    n = len(y_dev)
    draws = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        draws[i] = empirical_argmax_threshold(oof_proba[idx], y_dev[idx], scenario)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def implied_contact_rate(oof_proba: NDArray[np.float64], t_star: float) -> float:
    """Fraction of development rows contacted at threshold t_star — the retention team's implied workload."""
    return float(np.mean(oof_proba >= t_star))


def r_sensitivity_sweep(
    scenario: CostScenario, retention_rates: list[float]
) -> dict[float, float]:
    """t* at each retention rate in retention_rates, holding scenario's cost/LTV fixed.

    r is the one cost-model parameter not estimable from this dataset — it
    requires intervening on customers and observing outcomes — so this sweep
    is a headline result, not a footnote: it shows how much the operating
    point moves on a benchmark guess versus anything the model does.
    """
    return {
        r: closed_form_threshold(
            CostScenario(
                name=scenario.name,
                arpu=scenario.arpu,
                ltv=scenario.ltv,
                cost=scenario.cost,
                retention_rate=r,
            )
        )
        for r in retention_rates
    }


def derive_threshold(
    oof_proba: NDArray[np.float64],
    y_dev: NDArray[np.int_],
    scenario: CostScenario,
    n_bootstrap: int,
    random_state: int,
) -> dict[str, Any]:
    """The pure decision surface for one cost scenario: closed-form t* plus its agreement diagnostics.

    No conflation: t* is *derived* from cost parameters and *validated* here
    against development OOF probabilities; final cost/recall on the sealed
    test is evaluate.py's job, not this module's. dev_ev_at_t_star is a
    diagnostic, not the headline figure — economics.py computes the sealed-
    test EV at t* separately, which is what the README quotes.
    """
    t_star = closed_form_threshold(scenario)
    argmax_t = empirical_argmax_threshold(oof_proba, y_dev, scenario)
    ci_lower, ci_upper = argmax_bootstrap_ci(
        oof_proba, y_dev, scenario, n_bootstrap, random_state
    )
    return {
        "scenario": scenario.name,
        "threshold": t_star,
        "argmax_ev_threshold": argmax_t,
        "argmax_ev_bootstrap_ci": [ci_lower, ci_upper],
        "within_ci": ci_lower <= t_star <= ci_upper,
        "implied_contact_rate": implied_contact_rate(oof_proba, t_star),
        "dev_ev_at_t_star": expected_value_at_threshold(
            oof_proba, y_dev, scenario, t_star
        ),
        "costs": {
            "c": scenario.cost,
            "r": scenario.retention_rate,
            "ltv": scenario.ltv,
            "arpu": scenario.arpu,
        },
    }


# ---------------------------------------------------------------------------
# Orchestration — the only part of this module that touches MLflow or an
# estimator, and only ever via models.calibrate's already leak-tested OOF
# machinery, never directly.
# ---------------------------------------------------------------------------


def load_calibration_summary(run_id: str, cfg: DictConfig) -> dict[str, Any]:
    """Load calibrate.py's calibration_summary.json artifact — the source of the chosen calibration method."""
    mlflow.set_tracking_uri(resolve_tracking_uri(str(cfg.mlflow.tracking_uri)))
    summary: dict[str, Any] = mlflow.artifacts.load_dict(
        f"runs:/{run_id}/calibration/calibration_summary.json"
    )
    return summary


def load_dev_shap_summary(run_id: str, cfg: DictConfig) -> list[dict[str, Any]]:
    """Load calibrate.py's dev_shap_summary.json — the ranking the V3 pre-seal veto binds on.

    Sorted mean_abs_shap descending (calibrate.py's own explain.global_importance
    sort), each row {feature, mean_abs_shap, direction}. Raises loudly, naming
    the run, if absent — e.g. a hand-run older model version calibrated before
    this project added dev-SHAP logging — rather than silently skipping V3,
    which would let an unvetted model reach the sealed test set.
    """
    mlflow.set_tracking_uri(resolve_tracking_uri(str(cfg.mlflow.tracking_uri)))
    try:
        summary = mlflow.artifacts.load_dict(
            f"runs:/{run_id}/calibration/dev_shap_summary.json"
        )
    except MlflowException as exc:
        raise RuntimeError(
            f"Run {run_id!r} has no calibration/dev_shap_summary.json — this "
            "looks like a hand-run older model version calibrated before "
            "dev-SHAP logging existed. V3 cannot be silently skipped; "
            "re-run models.calibrate for this cycle first."
        ) from exc
    return cast("list[dict[str, Any]]", summary)


def build_dev_oof_screen_frame(
    customerid: pd.Series,
    y_true: NDArray[np.int_],
    p_hat: NDArray[np.float64],
    raw_partition: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join the already-aligned dev-OOF vector to the fitting population's segment/protected columns.

    Takes `run_threshold_step`'s own aligned (customerid, y_true, p_hat)
    rather than re-fetching dev_oof_predictions.parquet from MLflow a second
    time — this runs in the same process, right after that alignment.

    raw_partition, given, overrides the dev-only default with the caller-
    resolved fitting-population raw frame — the retrain-cycle bundle's own
    fourth field (models/train/common.py::load_training_pool_bundle).
    `None` (train/__main__.py's cold-start path) loads dev-only via
    load_dev_partition(), unchanged from today.
    """
    source = raw_partition if raw_partition is not None else load_dev_partition()
    dev_df = source.set_index("customerid").reindex(customerid)
    segment_lookup = build_segment_lookup(dev_df)
    frame = pd.DataFrame(
        {"customerid": customerid.to_numpy(), "y_true": y_true, "p_hat": p_hat}
    )
    for axis, series in segment_lookup.items():
        frame[axis] = series.to_numpy()
    return frame


def compute_dev_oof_diagnostics(
    frame: pd.DataFrame,
    base_threshold: float,
    calibration_slope_band: tuple[float, float],
    cfg: DictConfig,
    n_bootstrap: int,
    random_state: int,
) -> dict[str, Any]:
    """V1 (segment collapse), V2 (fairness disparity), V2b (per-group calibration
    collapse) — ANALYSIS.md §0's reported, non-gating dev-OOF diagnostics.

    Computed once, here, on the dev-OOF surface build_dev_oof_screen_frame
    already assembled; evaluate.py threads the returned dict unchanged into
    metrics.json rather than recomputing any of it.
    """
    y_true = frame["y_true"]
    proba = frame["p_hat"].to_numpy(dtype=float)
    segment_lookup = {axis: frame[axis] for axis in ROBUSTNESS_AXES + FAIRNESS_AXES}

    ranking_slices = sliced_ranking_metrics(
        y_true, proba, segment_lookup, ROBUSTNESS_AXES, n_bootstrap, random_state
    )
    v1_flagged = flag_segment_collapse(ranking_slices)

    decision_slices = sliced_decision_rates(
        y_true, proba, segment_lookup, FAIRNESS_AXES, base_threshold
    )
    equal_opportunity_by_axis = equal_opportunity_difference_by_axis(decision_slices)
    demographic_parity_by_axis = demographic_parity_difference_by_axis(decision_slices)
    v2_equal_opportunity_flagged = {
        axis: diff for axis, diff in equal_opportunity_by_axis.items() if diff > 0.10
    }
    v2_demographic_parity_flagged = {
        axis: diff for axis, diff in demographic_parity_by_axis.items() if diff > 0.10
    }

    calibration_slices = sliced_calibration(
        y_true, proba, segment_lookup, FAIRNESS_AXES, cfg, n_bootstrap, random_state
    )
    v2b_flagged = flag_calibration_collapse(calibration_slices, calibration_slope_band)

    return {
        "segment_pr_auc": ranking_slices,
        "segment_collapse_flagged": v1_flagged,
        "segment_decision_rates": decision_slices,
        "equal_opportunity_gap_by_axis": equal_opportunity_by_axis,
        "demographic_parity_gap_by_axis": demographic_parity_by_axis,
        "equal_opportunity_gap_flagged": v2_equal_opportunity_flagged,
        "demographic_parity_gap_flagged": v2_demographic_parity_flagged,
        "segment_calibration": calibration_slices,
        "calibration_collapse_flagged": v2b_flagged,
    }


def _save_sensitivity_plot(
    sweep: dict[float, float], shipped_threshold: float, path: Path
) -> None:
    """Render and save the r-sensitivity curve, marking the shipped threshold."""
    r_values = sorted(sweep)
    t_values = [sweep[r] for r in r_values]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(r_values, t_values, marker="o")
    ax.axhline(
        shipped_threshold,
        color="gray",
        linestyle="--",
        label=f"shipped t* = {shipped_threshold:.4f}",
    )
    ax.set_xlabel("Retention rate (r)")
    ax.set_ylabel("Operating threshold (t*)")
    ax.set_title("Threshold sensitivity to retention rate")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _save_scenario_threshold_plot(
    results: dict[str, dict[str, Any]], path: Path
) -> None:
    """Render and save the per-scenario t* comparison.

    Closed-form t* has zero sampling variance (derive_threshold's own
    reasoning) — plotted as a point marker with no error bars. The empirical
    argmax is what the bootstrap CI actually describes, so the CI attaches to
    that point instead: conflating the two would put error bars on a number
    that has none.
    """
    names = list(results)
    x = np.arange(len(names))

    closed_form = [results[name]["threshold"] for name in names]
    argmax = [results[name]["argmax_ev_threshold"] for name in names]
    ci_lower = [results[name]["argmax_ev_bootstrap_ci"][0] for name in names]
    ci_upper = [results[name]["argmax_ev_bootstrap_ci"][1] for name in names]
    lower_err = [a - lo for a, lo in zip(argmax, ci_lower, strict=True)]
    upper_err = [hi - a for a, hi in zip(argmax, ci_upper, strict=True)]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(
        x, closed_form, marker="D", color="black", zorder=3, label="closed-form t*"
    )
    ax.errorbar(
        x,
        argmax,
        yerr=[lower_err, upper_err],
        fmt="o",
        capsize=4,
        label="empirical argmax-EV (95% bootstrap CI)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Operating threshold (t*)")
    ax.set_title("Threshold by scenario: closed-form vs. empirical argmax")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _save_scenario_ev_curve_plot(
    ev_curves: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]],
    scenarios: dict[str, CostScenario],
    path: Path,
) -> None:
    """Render and save each scenario's EV curve overlaid, with its closed-form t* marked.

    ev_curves is precomputed once by the caller and reused for both this plot
    and the persisted ev_curve.parquet artifact — not recomputed here, so the
    picture can never silently diverge from the logged evidence.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (thresholds, ev) in ev_curves.items():
        (line,) = ax.plot(thresholds, ev, label=name)
        t_star = closed_form_threshold(scenarios[name])
        ax.axvline(t_star, color=line.get_color(), linestyle=":", alpha=0.6)
    ax.set_xlabel("Threshold (t)")
    ax.set_ylabel("Expected value per customer ($)")
    ax.set_title("Expected-value curve by scenario")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _validate_bundle_override(
    X_train: pd.DataFrame | None,
    y_train: pd.Series | None,
    customer_ids: pd.Series | None,
    raw_partition: pd.DataFrame | None,
) -> None:
    """Raise ValueError if the four-field training-pool bundle override is partial.

    All four or none — a partial override risks silently pairing an
    overridden X_train/y_train with the dev-only customer_ids/raw_partition
    the dev-only default would otherwise supply, misaligning the two by row
    order.
    """
    given = [
        X_train is not None,
        y_train is not None,
        customer_ids is not None,
        raw_partition is not None,
    ]
    if any(given) and not all(given):
        raise ValueError(
            "X_train, y_train, customer_ids, and raw_partition must be given "
            "together or not at all — a partial override risks silently "
            "mixing a dev-only population with an overridden one."
        )


def _load_and_align_dev_oof(
    run_id: str,
    cfg: DictConfig,
    X_train: pd.DataFrame | None = None,
    y_train: pd.Series | None = None,
    customer_ids: pd.Series | None = None,
) -> dict[str, Any]:
    """Load the model's OOF probabilities and align them to the fitting population.

    Loads calibrate.py's persisted dev_oof_predictions.parquet (for the
    method recorded in calibration_summary.json) rather than recomputing —
    this module never sees a fitted estimator or the raw data split. Aligned
    by customerid rather than trusted positionally: dev_oof and
    X_train/y_train are two independent loads, and a silent misalignment
    between them would otherwise never surface.

    X_train/y_train/customer_ids, given together, override the dev-only
    default with the caller-resolved fitting population — the retrain-cycle
    bundle's first three fields (models/train/common.py::
    load_training_pool_bundle). An overridden X_train is restricted to
    committed_features here, the same
    way calibrate.py's own override is. `None` (train/__main__.py's
    cold-start path) loads dev-only, unchanged from today — and this also
    fixes a real bug the override closes: without it, `.reindex(customerid)`
    silently discards every reserve-cohort row calibrate.py's
    dev_oof_predictions.parquet logged on a retrain cycle, since reindexing
    to a subset always finds matches and the `notna().all()` assertion below
    can't catch it.
    """
    mlflow.set_tracking_uri(resolve_tracking_uri(str(cfg.mlflow.tracking_uri)))

    manifest = load_training_manifest(run_id, cfg)
    calibration_summary = load_calibration_summary(run_id, cfg)
    method = str(calibration_summary["method"])

    committed_features = committed_features_from_manifest(manifest)
    if X_train is None:
        X_train, y_train = load_dev_features(committed_features)
        customer_ids = load_dev_customer_ids()
    else:
        assert y_train is not None  # _validate_bundle_override already enforced this
        X_train = X_train[committed_features]

    dev_oof = load_dev_oof_predictions(run_id, cfg).set_index("customerid")
    aligned = dev_oof.reindex(customer_ids)
    assert aligned["p_hat"].notna().all(), (
        "dev_oof_predictions.parquet is missing rows for one or more "
        "fitting-population customerids — it no longer matches the current "
        "features (was it logged against a different fitting population?)."
    )
    oof_proba = aligned["p_hat"].to_numpy()
    y_train_arr = y_train.to_numpy()
    assert np.array_equal(aligned["y_true"].to_numpy(dtype=int), y_train_arr), (
        "dev_oof_predictions.parquet's y_true does not match the current "
        "y_train — the persisted OOF artifact is stale relative to the "
        "current fitting population."
    )

    return {
        "run_id": run_id,
        "calibration_summary": calibration_summary,
        "method": method,
        "X_train": X_train,
        "y_train": y_train,
        "customer_ids": customer_ids,
        "oof_proba": oof_proba,
        "y_train_arr": y_train_arr,
    }


def _derive_scenario_thresholds(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    oof_proba: NDArray[np.float64],
    y_train_arr: NDArray[np.int_],
    cfg: DictConfig,
) -> dict[str, Any]:
    """Derive every scenario's threshold and the base scenario's retention-rate sensitivity sweep."""
    costs_cfg = load_costs_config(get_project_root() / str(cfg.paths.costs_config))
    scenarios = resolve_all_scenarios(X_train["monthlycharges"], y_train, costs_cfg)

    n_bootstrap = int(costs_cfg.argmax_ev_bootstrap_n_samples)
    random_state = int(cfg.threshold.random_state)

    results = {
        name: derive_threshold(
            oof_proba, y_train_arr, scenario, n_bootstrap, random_state
        )
        for name, scenario in scenarios.items()
    }
    base = results["base"]
    for name, result in results.items():
        if not result["within_ci"]:
            logger.warning(
                "threshold_argmax_disagreement",
                scenario=name,
                closed_form=result["threshold"],
                argmax_ev_bootstrap_ci=result["argmax_ev_bootstrap_ci"],
                hint=(
                    "t* falling outside the empirical argmax-EV CI suggests "
                    "miscalibration near the operating point."
                ),
            )

    sweep = r_sensitivity_sweep(
        scenarios["base"], [float(r) for r in costs_cfg.retention_rate_sweep]
    )
    base_r = scenarios["base"].retention_rate
    matching = [r for r in sweep if abs(r - base_r) < 1e-9]
    if matching:
        assert abs(sweep[matching[0]] - base["threshold"]) < 1e-9, (
            "r-sensitivity sweep's base-retention-rate threshold does not "
            "match the shipped base threshold — retention_rate_sweep and "
            "scenarios.base.retention_rate have drifted apart in "
            "configs/costs.yaml."
        )

    return {
        "costs_cfg": costs_cfg,
        "scenarios": scenarios,
        "results": results,
        "base": base,
        "sweep": sweep,
    }


def _render_threshold_figures(
    derived: dict[str, Any],
    oof_proba: NDArray[np.float64],
    y_train_arr: NDArray[np.int_],
    cfg: DictConfig,
) -> dict[str, Any]:
    """Render the three threshold figures and the EV-curve series they (and the persisted artifact) share.

    Logs threshold_sensitivity.png (base's t* vs. retention rate),
    threshold_by_scenario.png (closed-form t* vs. empirical argmax + its
    bootstrap CI, all scenarios), and expected_value_by_scenario.png (each
    scenario's EV curve, its t* marked) onto the run's figures/ artifacts and
    mirrors them to reports/figures/. ev_curves is computed once here and
    shared with the persisted ev_curve.parquet artifact so the picture can
    never diverge from the logged evidence.
    """
    results, base, scenarios = (
        derived["results"],
        derived["base"],
        derived["scenarios"],
    )

    figure_path = (
        get_project_root() / str(cfg.paths.figures) / "threshold_sensitivity.png"
    )
    _save_sensitivity_plot(derived["sweep"], float(base["threshold"]), figure_path)

    scenario_threshold_path = (
        get_project_root() / str(cfg.paths.figures) / "threshold_by_scenario.png"
    )
    _save_scenario_threshold_plot(results, scenario_threshold_path)

    ev_curves = {
        name: expected_value_curve(oof_proba, y_train_arr, scenario)
        for name, scenario in scenarios.items()
    }

    ev_curve_path = (
        get_project_root() / str(cfg.paths.figures) / "expected_value_by_scenario.png"
    )
    _save_scenario_ev_curve_plot(ev_curves, scenarios, ev_curve_path)

    return {
        "figure_path": figure_path,
        "scenario_threshold_path": scenario_threshold_path,
        "ev_curve_path": ev_curve_path,
        "ev_curves": ev_curves,
    }


def _run_direction_sanity_check(run_id: str, cfg: DictConfig) -> dict[str, Any]:
    """Pre-seal direction-sanity veto (ANALYSIS.md's V3): rank calibrate.py's
    dev-SHAP summary, cut at `direction_sanity_top_k_features`, and check
    each surviving feature's direction against `explain.EXPECTED_EDA_DIRECTIONS`.

    No shap import here — calibrate.py already computed the SHAP values and
    logged the ranking (`dev_shap_summary.json`); this module only ranks,
    cuts, and checks the already-persisted summary. A top-k feature whose
    `|direction|` falls below `direction_sanity_min_magnitude` is excluded
    from the checked set (`weak_count`) rather than risking a veto on a sign
    too unstable/noise-prone to trust.
    """
    summary = load_dev_shap_summary(run_id, cfg)
    direction_sanity_top_k = int(cfg.threshold.direction_sanity_top_k_features)
    min_direction_magnitude = float(cfg.threshold.direction_sanity_min_magnitude)

    elbow = check_top_k_elbow(
        [float(row["mean_abs_shap"]) for row in summary], direction_sanity_top_k
    )
    if not elbow["valid"]:
        logger.warning("direction_sanity_top_k_features_elbow_drifted", **elbow)

    top_rows = summary[:direction_sanity_top_k]
    directions = {str(row["feature"]): float(row["direction"]) for row in top_rows}
    strong_features = [
        str(row["feature"])
        for row in top_rows
        if abs(float(row["direction"])) >= min_direction_magnitude
    ]
    weak_count = len(top_rows) - len(strong_features)

    direction_sanity_result = direction_sanity_check(
        strong_features, directions, EXPECTED_EDA_DIRECTIONS
    )
    checked_count = sum(
        1 for row in direction_sanity_result["checked_features"] if row["checked"]
    )

    return {
        "direction_sanity_result": direction_sanity_result,
        "direction_sanity_top_k_features": direction_sanity_top_k,
        "top_k_feature_names": [str(row["feature"]) for row in top_rows],
        "checked_count": checked_count,
        "weak_count": weak_count,
        "elbow": elbow,
    }


def _run_dev_oof_screen(
    loaded: dict[str, Any],
    base: dict[str, Any],
    cfg: DictConfig,
    random_state: int,
    raw_partition: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Screen calibrate.py's logged calibration slope, this module's own
    direction-sanity veto, and the base scenario's within_ci agreement
    against ANALYSIS.md §0's bars.

    Reuses the aligned dev-OOF vector `_load_and_align_dev_oof` already
    built, never re-fetching dev_oof_predictions.parquet. `base` is
    `_derive_scenario_thresholds`'s already-computed base-scenario result
    (threshold, within_ci, argmax_ev_bootstrap_ci, ...) — reused, not
    re-derived, per the "no new number invented" discipline. `raw_partition`,
    given, overrides the dev-only default `build_dev_oof_screen_frame` would
    otherwise load — see that function's own docstring.

    Returns `failures` (a list of `{criterion, detail, remediation}` dicts,
    empty iff the screen passes) rather than raising: the caller logs and
    mirrors the audit trail first, and raises only afterward, so a failing
    screen is recorded rather than silently vanishing. Exactly three
    possible entries — `calibration_slope`, `direction_sanity`, and
    `within_ci` — never V1/V2/V2b, which stay reported-only per CLAUDE.md's
    three-guardrail rule. A direction-sanity veto with zero checked features
    (nothing in the top-k both matched an EDA key and cleared the magnitude
    floor) is itself a failure: a veto that never fired validated nothing.
    The direction-sanity top-k elbow-validity check
    (`explain.check_top_k_elbow`) is recorded into `dev_oof_diagnostics`
    under `direction_sanity_elbow_check` — reported-only like V1/V2/V2b, not
    a fourth failure entry, since it only flags that
    `direction_sanity_top_k_features` may need re-deriving by hand, never
    that the veto itself is untrustworthy. `within_ci` is a veto only —
    never auto-re-derives t*, which stays the inherited, already-shipped
    operating point regardless of this screen's verdict (CLAUDE.md's
    "guardrails may veto; they may never promote" rule).
    """
    base_threshold = float(base["threshold"])
    dev_oof_screen_frame = build_dev_oof_screen_frame(
        loaded["customer_ids"],
        loaded["y_train_arr"],
        loaded["oof_proba"],
        raw_partition,
    )
    slope = loaded["calibration_summary"]["calibration_slope"]
    calibration_slope_band = load_model_promotion_bars(cfg).calibration_slope_band
    slope_ok = slope_passes(
        slope["slope_ci_lower"], slope["slope_ci_upper"], calibration_slope_band
    )

    direction_sanity = _run_direction_sanity_check(loaded["run_id"], cfg)

    failures: list[dict[str, Any]] = []
    if not slope_ok:
        failures.append(
            {
                "criterion": "calibration_slope",
                "detail": (
                    f"dev-OOF slope 95% CI [{slope['slope_ci_lower']:.4f}, "
                    f"{slope['slope_ci_upper']:.4f}] lies entirely outside "
                    f"band [{calibration_slope_band[0]}, "
                    f"{calibration_slope_band[1]}]"
                ),
                "remediation": (
                    "Re-calibrate before evaluating on the sealed test set."
                ),
            }
        )
    direction_sanity_ok = (
        direction_sanity["direction_sanity_result"]["passed"]
        and direction_sanity["checked_count"] > 0
    )
    if not direction_sanity_ok:
        top_k = direction_sanity["direction_sanity_top_k_features"]
        if direction_sanity["checked_count"] == 0:
            detail = (
                f"zero of the dev-SHAP top-{top_k} features could be "
                "checked against an established EDA direction — the veto "
                "cannot validate anything this cycle."
            )
        else:
            violation_names = [
                row["feature"]
                for row in direction_sanity["direction_sanity_result"]["violations"]
            ]
            detail = (
                f"{len(direction_sanity['direction_sanity_result']['violations'])} "
                f"of {direction_sanity['checked_count']} checked dev-SHAP "
                f"top-{top_k} features contradict their established EDA "
                f"direction: {violation_names}"
            )
        failures.append(
            {
                "criterion": "direction_sanity",
                "detail": detail,
                "remediation": (
                    "Investigate whether the model is fitting a "
                    "training-data artifact before evaluating on the "
                    "sealed test set."
                ),
            }
        )
    if not bool(base["within_ci"]):
        ci_lower, ci_upper = base["argmax_ev_bootstrap_ci"]
        failures.append(
            {
                "criterion": "within_ci",
                "detail": (
                    f"inherited closed-form t*={base_threshold:.4f} falls "
                    f"outside the base scenario's empirical argmax-EV "
                    f"bootstrap CI [{ci_lower:.4f}, {ci_upper:.4f}] — the "
                    "inherited operating point no longer agrees with where "
                    "expected value is empirically maximised on this "
                    "cycle's dev-OOF surface."
                ),
                "remediation": (
                    "Check whether this is sampling noise on a small "
                    "dev-OOF sample first; then re-examine the other "
                    "calibration method's OOF vector for a local "
                    "disagreement near t*; if the disagreement is real and "
                    "persistent across cycles, check the logged "
                    "costs.arpu in threshold_payload across recent cycles "
                    "before revisiting configs/costs.yaml's r/LTV/cost "
                    "inputs deliberately (a reviewed edit, never "
                    "automatic) — this screen never auto-re-derives t*."
                ),
            }
        )

    dev_oof_diagnostics = compute_dev_oof_diagnostics(
        dev_oof_screen_frame,
        base_threshold,
        calibration_slope_band,
        cfg,
        int(cfg.threshold.n_bootstrap),
        random_state,
    )
    dev_oof_diagnostics["direction_sanity_result"] = direction_sanity[
        "direction_sanity_result"
    ]
    dev_oof_diagnostics["direction_check_feature_names"] = direction_sanity[
        "top_k_feature_names"
    ]
    dev_oof_diagnostics["direction_checked_count"] = direction_sanity["checked_count"]
    dev_oof_diagnostics["direction_weak_signal_count"] = direction_sanity["weak_count"]
    dev_oof_diagnostics["direction_sanity_elbow_check"] = direction_sanity["elbow"]

    return {
        "dev_oof_screen_frame": dev_oof_screen_frame,
        "slope": slope,
        "calibration_slope_band": calibration_slope_band,
        "failures": failures,
        "dev_oof_diagnostics": dev_oof_diagnostics,
    }


def _assemble_threshold_payloads(
    loaded: dict[str, Any],
    derived: dict[str, Any],
    model_version: str,
    failures: list[dict[str, Any]],
    cfg: DictConfig,
) -> dict[str, Any]:
    """Assemble threshold.json, threshold.yaml (policy), and threshold_validation.json.

    The two files on disk split along model-independence: reports/policy/
    threshold.yaml carries only the pure functions of configs/costs.yaml
    (threshold/costs/retention_rate_sensitivity), pinned by a
    costs_config_hash and carrying no model_run_id/logged_model_id — a
    model-independent file must not carry a model stamp, or a rollback to a
    different champion leaves it describing the wrong model.
    threshold_validation.json carries the model-dependent half (argmax-EV
    threshold, its bootstrap CI, calibration_method, failures, ...) as
    an artifact on this model's own run, so it travels with that version
    rather than sitting at a fixed path a rollback could leave stale.

    Stamped with logged_model_id, not model_version: a LoggedModel is the
    actual scored artifact evaluate.py/error_analysis.py compare against
    (check_threshold_provenance) — model_run_id is kept as a locator only,
    no longer load-bearing.

    Every scenario's full diagnostic bundle ships in both payloads, not just
    its t* — conservative and optimistic are equally auditable. `base` alone
    drives the top-level threshold/within_ci fields: it is the shipped
    operating point, the others are reference alternatives.
    """
    results, base, sweep = derived["results"], derived["base"], derived["sweep"]
    run_id, method = loaded["run_id"], loaded["method"]
    logged_model_id = resolve_logged_model_id(model_version, cfg)

    threshold_payload: dict[str, Any] = {
        "threshold": base["threshold"],
        "scenario": "base",
        "rule": "closed_form_c_over_r_ltv",
        "costs": base["costs"],
        "calibration_method": method,
        "argmax_ev_bootstrap_ci": base["argmax_ev_bootstrap_ci"],
        "implied_contact_rate": base["implied_contact_rate"],
        "scenarios": results,
        "retention_rate_sensitivity": sweep,
        "model_run_id": run_id,
        "logged_model_id": logged_model_id,
    }

    costs_hash = costs_config_hash(get_project_root() / str(cfg.paths.costs_config))
    policy_payload: dict[str, Any] = {
        "threshold": base["threshold"],
        "scenario": "base",
        "rule": "closed_form_c_over_r_ltv",
        "costs": base["costs"],
        "scenarios": {
            name: {
                "scenario": name,
                "threshold": result["threshold"],
                "costs": result["costs"],
            }
            for name, result in results.items()
        },
        "retention_rate_sensitivity": sweep,
        "costs_config_hash": costs_hash,
    }
    validation_payload: dict[str, Any] = {
        "model_run_id": run_id,
        "logged_model_id": logged_model_id,
        "calibration_method": method,
        "argmax_ev_threshold": base["argmax_ev_threshold"],
        "argmax_ev_bootstrap_ci": base["argmax_ev_bootstrap_ci"],
        "within_ci": base["within_ci"],
        "implied_contact_rate": base["implied_contact_rate"],
        # The model-dependent half of the dev-OOF pre-seal screen (calibration
        # slope + V3 direction sanity) — evaluate.py/error_analysis.py/
        # register.py re-check this independently via
        # check_threshold_screen_passed, on top of the RuntimeError
        # run_threshold_step itself already raises when this is non-empty.
        "failures": failures,
        "scenarios": {
            name: {
                "scenario": name,
                "argmax_ev_threshold": result["argmax_ev_threshold"],
                "argmax_ev_bootstrap_ci": result["argmax_ev_bootstrap_ci"],
                "within_ci": result["within_ci"],
                "implied_contact_rate": result["implied_contact_rate"],
                "dev_ev_at_t_star": result["dev_ev_at_t_star"],
            }
            for name, result in results.items()
        },
    }

    return {
        "threshold_payload": threshold_payload,
        "policy_payload": policy_payload,
        "validation_payload": validation_payload,
    }


def _log_threshold_run(
    loaded: dict[str, Any],
    derived: dict[str, Any],
    figures: dict[str, Any],
    payloads: dict[str, Any],
    screen: dict[str, Any],
    model_version: str,
    cfg: DictConfig,
) -> None:
    """Log every threshold artifact and metric onto the model's own MLflow run.

    Logs t_star_{scenario}, implied_contact_rate_{scenario}, and
    dev_ev_at_t_star_{scenario} as MLflow metrics for every scenario, and
    persists the EV-curve points themselves as ev_curve.parquet — the curve
    exists today only as pixels in expected_value_by_scenario.png.

    Also tags this run with costs_config_hash — same tag name evaluate.py
    sets on its own run (CLAUDE.md's business-metrics rule) — so the
    provenance stamp payloads["policy_payload"] pins into
    reports/policy/threshold.yaml has an MLflow-side copy too, not just the
    local file. threshold.py's hash is computed once, at derivation time;
    evaluate.py's is a separately-computed value at evaluate time, so the two
    are not guaranteed identical across a `costs.yaml` edit between the two
    steps — a mismatch here is itself a diagnostic worth surfacing, not
    reconciled away by sharing one value.
    """
    results = derived["results"]
    ev_curves = figures["ev_curves"]

    with mlflow.start_run(run_id=loaded["run_id"]):
        set_run_description(TRAINING_CYCLE_RUN_DESCRIPTION)
        mlflow.log_dict(payloads["threshold_payload"], "threshold/threshold.json")
        mlflow.log_dict(
            payloads["validation_payload"], "threshold/threshold_validation.json"
        )
        mlflow.log_artifact(
            str(figures["figure_path"]), artifact_path="threshold/figures"
        )
        mlflow.log_artifact(
            str(figures["scenario_threshold_path"]), artifact_path="threshold/figures"
        )
        mlflow.log_artifact(
            str(figures["ev_curve_path"]), artifact_path="threshold/figures"
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            ev_curve_data_path = Path(tmp_dir) / "ev_curve.parquet"
            ev_curve_df = pd.concat(
                [
                    pd.DataFrame({"scenario": name, "threshold": thresholds, "ev": ev})
                    for name, (thresholds, ev) in ev_curves.items()
                ],
                ignore_index=True,
            )
            ev_curve_df.to_parquet(ev_curve_data_path, index=False)
            mlflow.log_artifact(str(ev_curve_data_path), artifact_path="threshold")

        scenario_metrics: dict[str, float] = {}
        for name, result in results.items():
            scenario_metrics[f"t_star_{name}"] = result["threshold"]
            scenario_metrics[f"implied_contact_rate_{name}"] = result[
                "implied_contact_rate"
            ]
            scenario_metrics[f"dev_ev_at_t_star_{name}"] = result["dev_ev_at_t_star"]
        mlflow.log_metrics(scenario_metrics)

        model_id = resolve_logged_model_id(model_version, cfg)
        dev_oof_dataset = mlflow_dataset_from_pandas(
            screen["dev_oof_screen_frame"], name="dev_oof_screen", targets="y_true"
        )
        mlflow.log_input(dev_oof_dataset, context="dev_oof_screen")
        slope = screen["slope"]
        mlflow.log_metrics(
            {
                "dev_oof_calibration_slope": slope["slope"],
                "dev_oof_calibration_slope_ci_lower": slope["slope_ci_lower"],
                "dev_oof_calibration_slope_ci_upper": slope["slope_ci_upper"],
            },
            model_id=model_id,
            dataset=dev_oof_dataset,
        )
        mlflow.log_dict(
            screen["dev_oof_diagnostics"], "threshold/dev_oof_diagnostics.json"
        )
        mlflow.set_tag(
            "dev_oof_screen_result", "pass" if not screen["failures"] else "fail"
        )
        mlflow.set_tag(
            "costs_config_hash", payloads["policy_payload"]["costs_config_hash"]
        )


def _write_policy_yaml(policy_payload: dict[str, Any], cfg: DictConfig) -> None:
    """Write reports/policy/threshold.yaml — the model-independent shipped-policy mirror.

    Like reports/figures/reliability_diagram.png, overwritten on every call —
    the copy on disk reflects the most recent local run, not a specific
    run_id. Shared by run_threshold_step and run_threshold_rerun_step: a
    costs.yaml-only re-run genuinely changes the shipped policy this file
    describes, even though it mints no new model version.
    """
    policy_path = get_project_root() / str(cfg.paths.policy) / "threshold.yaml"
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text(
        OmegaConf.to_yaml(OmegaConf.create(policy_payload)), newline="\n"
    )


def _write_threshold_reports(
    loaded: dict[str, Any],
    policy_payload: dict[str, Any],
    dev_oof_diagnostics: dict[str, Any],
    cfg: DictConfig,
) -> None:
    """Write reports/policy/threshold.yaml and the reports/ dev-OOF mirror.

    threshold_validation.json is self-describing (model_run_id,
    model_version), so that is the right place to confirm provenance, not
    these files' mtimes.
    """
    _write_policy_yaml(policy_payload, cfg)

    reports_dir = get_project_root() / str(cfg.paths.reports)
    reports_dir.mkdir(parents=True, exist_ok=True)
    dev_oof_predictions = pd.DataFrame(
        {
            "customerid": loaded["customer_ids"].to_numpy(),
            "y_true": loaded["y_train_arr"],
            "p_hat": loaded["oof_proba"],
        }
    )
    dev_oof_predictions.to_parquet(
        reports_dir / "dev_oof_predictions.parquet", index=False
    )
    with open(
        reports_dir / "dev_oof_diagnostics.json", "w", encoding="utf-8", newline="\n"
    ) as f:
        json.dump(dev_oof_diagnostics, f, indent=2, default=str)
        f.write("\n")


def run_threshold_step(
    run_id: str,
    model_version: str,
    cfg: DictConfig,
    X_train: pd.DataFrame | None = None,
    y_train: pd.Series | None = None,
    customer_ids: pd.Series | None = None,
    raw_partition: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Derive and ship the cost-sensitive threshold for a registered calibrated model.

    Takes `run_id`/`model_version` already resolved by the caller
    (utils.mlflow.resolve_model_identifier — an explicit override, never the
    `challenger` alias, or calibrate.py's receipt): a re-calibration
    invalidates a previously-derived threshold, and an alias is a moving
    pointer that could later point at a different version.

    X_train/y_train/customer_ids/raw_partition, given together, override the
    dev-only default with the retrain-cycle bundle (models/train/common.py::
    load_training_pool_bundle, dev ∪ matured reserve). All four or none;
    train/__main__.py's cold-start path never passes them, so `None` still derives everything
    dev-only, unchanged from today. Named to match the bundle's own
    output-field names — the same naming discipline calibrate.py's own
    override already follows — so nothing is mislabeled on a retrain cycle.

    Finally runs the dev-OOF screen: screens calibrate.py's logged
    calibration slope and this module's own direction-sanity veto against
    ANALYSIS.md §0's bars, screens the base scenario's within_ci agreement
    (§D9 — never auto-re-derives t*, only vetoes), and computes V1/V2/V2b on
    the same aligned dev-OOF vector already built for the threshold
    derivation above (no second fetch), writing
    reports/dev_oof_predictions.parquet and reports/dev_oof_diagnostics.json.
    Raises RuntimeError, after logging, if any of the three binding checks
    fails.
    """
    _validate_bundle_override(X_train, y_train, customer_ids, raw_partition)
    loaded = _load_and_align_dev_oof(run_id, cfg, X_train, y_train, customer_ids)
    derived = _derive_scenario_thresholds(
        loaded["X_train"],
        loaded["y_train"],
        loaded["oof_proba"],
        loaded["y_train_arr"],
        cfg,
    )
    figures = _render_threshold_figures(
        derived, loaded["oof_proba"], loaded["y_train_arr"], cfg
    )
    random_state = int(cfg.threshold.random_state)
    screen = _run_dev_oof_screen(
        loaded, derived["base"], cfg, random_state, raw_partition
    )
    payloads = _assemble_threshold_payloads(
        loaded, derived, model_version, screen["failures"], cfg
    )
    _log_threshold_run(loaded, derived, figures, payloads, screen, model_version, cfg)
    _write_threshold_reports(
        loaded, payloads["policy_payload"], screen["dev_oof_diagnostics"], cfg
    )

    base = derived["base"]
    slope = screen["slope"]
    failures = screen["failures"]
    logger.info(
        "threshold_derived",
        run_id=loaded["run_id"],
        model_version=model_version,
        method=loaded["method"],
        threshold=base["threshold"],
        within_ci=base["within_ci"],
        implied_contact_rate=base["implied_contact_rate"],
        dev_oof_calibration_slope=slope["slope"],
        dev_oof_screen_passed=not failures,
    )

    result = {
        "threshold_payload": payloads["threshold_payload"],
        "policy_payload": payloads["policy_payload"],
        "validation_payload": payloads["validation_payload"],
        "dev_oof_diagnostics": screen["dev_oof_diagnostics"],
        "dev_oof_screen_passed": not failures,
    }

    if failures:
        clauses = "; ".join(f"{f['criterion']}: {f['detail']}" for f in failures)
        raise RuntimeError(
            f"Dev-OOF pre-seal screen failed ({len(failures)} criterion/"
            f"criteria): {clauses} — do not evaluate on the sealed test set "
            "until resolved."
        )

    return result


def _recheck_recall_guardrail(
    oof_proba: NDArray[np.float64],
    y_train_arr: NDArray[np.int_],
    t_star: float,
    recall_bar: float,
) -> dict[str, Any]:
    """Recall of "contact iff proba >= t_star" on the dev-OOF vector, checked against ANALYSIS.md §0's recall bar.

    Dev-OOF, never the sealed test set — this module is leak-free by
    construction and never binds the test partition (CLAUDE.md's
    test-set-touched-once rule reserves that to evaluate.py). A costs.yaml-
    only change mints no new model version, so there is nothing for
    evaluate.py's own sealed-test recall check to run against; this is the
    leak-free dev-OOF proxy run_threshold_rerun_step checks instead.
    """
    predicted = oof_proba >= t_star
    actual_positive = y_train_arr == 1
    tp = int(np.sum(predicted & actual_positive))
    fn = int(np.sum(~predicted & actual_positive))
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {"recall": recall, "recall_bar": recall_bar, "passed": recall >= recall_bar}


def run_threshold_rerun_step(model_version: str, cfg: DictConfig) -> dict[str, Any]:
    """Re-derive t* against a costs.yaml-only change, without minting a new model version.

    A costs.yaml edit changes t* = c/(r x LTV) but not the calibrated
    pipeline itself, so this never fits, calibrates, mints, promotes, or
    rejects anything — it only re-derives every scenario's threshold from
    an already-promoted version's own dev-OOF probabilities (the same
    _load_and_align_dev_oof/_derive_scenario_thresholds machinery
    run_threshold_step uses, called here with no bundle override — the
    population is whatever the champion was originally trained on — against
    whatever configs/costs.yaml currently holds on disk) and rechecks the
    recall guardrail at the new base-scenario t*.

    Logs to a fresh, small MLflow run rather than model_version's own
    training run: that run stays a description of the training cycle that
    produced it, not of every policy edit made afterward. Writes
    reports/threshold_rerun_receipt.json (this cycle's bootstrap pointer,
    rewritten fresh on every re-run) so register.py's tag_threshold_rerun
    can point a threshold_run_id tag at this run without an explicit
    run_id argument; serving/policy_config.py::load_threshold_payload then
    resolves through that tag when present, falling back to the training
    run when it isn't.

    No figures are rendered and reports/dev_oof_predictions.parquet/
    reports/dev_oof_diagnostics.json are not rewritten — the dev-OOF screen
    itself did not re-run, only the threshold derived from its
    already-persisted output. reports/policy/threshold.yaml (the
    model-independent shipped-policy mirror) is refreshed, since the
    shipped policy genuinely changed.

    If the recall guardrail fails at the new t*, this does not raise and
    does not silently ship it either — there is no promotion happening for
    CLAUDE.md's "guardrails may veto; they may never promote" rule to apply
    to. Failure is instead a stop-and-look alarm: logged at error level and
    recorded in both the returned result and the fresh run's validation
    payload for a human to act on deliberately — the same alarm-on-ambiguity
    pattern the dummy-floor and within_ci guardrails already use elsewhere
    in this project, never an automatic retrain trigger by itself.
    """
    run_id = resolve_model_run_id(model_version, cfg)
    loaded = _load_and_align_dev_oof(run_id, cfg)
    derived = _derive_scenario_thresholds(
        loaded["X_train"],
        loaded["y_train"],
        loaded["oof_proba"],
        loaded["y_train_arr"],
        cfg,
    )
    base = derived["base"]
    recall_bar = load_model_promotion_bars(cfg).recall_bar
    guardrail = _recheck_recall_guardrail(
        loaded["oof_proba"], loaded["y_train_arr"], float(base["threshold"]), recall_bar
    )

    payloads = _assemble_threshold_payloads(loaded, derived, model_version, [], cfg)
    payloads["validation_payload"]["recall_guardrail"] = guardrail

    ensure_experiment_metadata(cfg)
    with mlflow.start_run(run_name="threshold_rerun") as run:
        set_run_description(
            "costs.yaml-only threshold re-run — recomputes t* for an "
            "already-promoted model version against the current "
            "configs/costs.yaml, with no new training and no new registry "
            "version. See threshold/threshold_validation.json's "
            "recall_guardrail field for whether the new operating point "
            "still clears ANALYSIS.md §0's recall bar."
        )
        mlflow.set_tag("source_run_id", run_id)
        mlflow.set_tag("source_model_version", model_version)
        mlflow.log_dict(payloads["threshold_payload"], "threshold/threshold.json")
        mlflow.log_dict(
            payloads["validation_payload"], "threshold/threshold_validation.json"
        )
        mlflow.set_tag(
            "costs_config_hash", payloads["policy_payload"]["costs_config_hash"]
        )
        mlflow.set_tag(
            "recall_guardrail_result", "pass" if guardrail["passed"] else "fail"
        )
        threshold_run_id = run.info.run_id

    _write_policy_yaml(payloads["policy_payload"], cfg)
    write_threshold_rerun_receipt(model_version, threshold_run_id, cfg)

    log_fn = logger.info if guardrail["passed"] else logger.error
    log_fn(
        "threshold_rerun_done",
        model_version=model_version,
        threshold_run_id=threshold_run_id,
        threshold=base["threshold"],
        recall_guardrail_passed=guardrail["passed"],
        recall=guardrail["recall"],
        recall_bar=guardrail["recall_bar"],
    )

    return {
        "model_version": model_version,
        "threshold_run_id": threshold_run_id,
        "threshold_payload": payloads["threshold_payload"],
        "validation_payload": payloads["validation_payload"],
        "recall_guardrail_passed": guardrail["passed"],
    }


if __name__ == "__main__":
    import sys

    import pandera as pa
    from dotenv import load_dotenv

    from telco_churn.utils.logging import configure_logging
    from telco_churn.utils.paths import activate_config, compose_config

    load_dotenv()
    configure_logging()

    try:
        cfg = compose_config(overrides=sys.argv[1:] or None)
        activate_config(cfg)
        if cfg.threshold.rerun_model_version is not None:
            rerun_result = run_threshold_rerun_step(
                str(cfg.threshold.rerun_model_version), cfg
            )
            logger.info(
                "threshold_rerun_step_done",
                threshold_run_id=rerun_result["threshold_run_id"],
                model_version=rerun_result["model_version"],
                recall_guardrail_passed=rerun_result["recall_guardrail_passed"],
            )
        else:
            cli_run_id, cli_model_version, _cli_model_uri = resolve_model_identifier(
                cfg.threshold.run_id, cfg.threshold.model_version, cfg
            )
            result = run_threshold_step(cli_run_id, cli_model_version, cfg)
            logger.info(
                "threshold_step_done",
                threshold=result["threshold_payload"]["threshold"],
                model_version=cli_model_version,
            )
    except FileNotFoundError as e:
        logger.error("threshold_data_not_found", error=str(e), exc_info=True)
        sys.exit(1)
    except pa.errors.SchemaError as e:
        logger.error("threshold_data_schema_invalid", error=str(e), exc_info=True)
        sys.exit(1)
    except ValueError as e:
        logger.error("threshold_invalid", error=str(e), exc_info=True)
        sys.exit(1)
    except RuntimeError as e:
        logger.error("threshold_dev_oof_screen_failed", error=str(e), exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error("threshold_failed", error=str(e), exc_info=True)
        sys.exit(1)

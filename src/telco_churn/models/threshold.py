"""Derive the cost-sensitive operating threshold from a calibrated pipeline.

Contact a customer if and only if q·r·LTV > c (q = calibrated churn
probability, r = retention success rate, LTV = discounted lifetime value,
c = intervention cost) — solving for the break-even q gives the closed form
this module ships: t* = c / (r × LTV).

This supersedes the classical C_FP/(C_FP + C_FN) rule, which assumes correct
decisions are free; here contacting costs c regardless of whether the
customer was going to churn, so cost attaches to the action, not the error.

Leak-free by construction: no `.fit(` call, no sklearn import, no
telco_churn.data.split import. The pure functions below take already-computed
arrays and a CostScenario, never an estimator — only run_threshold_step
reaches into models.calibrate for those arrays.
"""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import mlflow.artifacts
import mlflow.tracking
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf

from telco_churn.models.calibrate import (
    committed_features_from_manifest,
    load_dev_features,
    load_training_manifest,
    oof_calibrated_proba,
    unfitted_pipeline_from_manifest,
)
from telco_churn.utils.logging import get_logger
from telco_churn.utils.mlflow import resolve_tracking_uri
from telco_churn.utils.paths import get_project_root

__all__ = [
    "CostScenario",
    "load_costs_config",
    "costs_config_hash",
    "arpu_by_scenario",
    "resolve_scenario",
    "resolve_all_scenarios",
    "closed_form_threshold",
    "expected_value_curve",
    "expected_value_at_threshold",
    "empirical_argmax_threshold",
    "argmax_bootstrap_ci",
    "implied_contact_rate",
    "r_sensitivity_sweep",
    "derive_threshold",
    "load_calibration_summary",
    "run_threshold_step",
]

logger = get_logger(__name__)


@dataclass(frozen=True)
class CostScenario:
    """One resolved cost scenario — the ARPU/LTV/cost/retention-rate values a threshold is derived from."""

    name: str
    arpu: float
    ltv: float
    cost: float
    retention_rate: float


# ---------------------------------------------------------------------------
# Cost/ARPU resolution
# ---------------------------------------------------------------------------


def load_costs_config(path: Path | None = None) -> DictConfig:
    """Load configs/costs.yaml. path defaults to the canonical project location; pass an explicit path in tests."""
    resolved = (
        path if path is not None else get_project_root() / "configs" / "costs.yaml"
    )
    cfg = OmegaConf.load(resolved)
    assert isinstance(cfg, DictConfig)
    return cfg


def costs_config_hash(path: Path | None = None) -> str:
    """Return the sha256 content hash of configs/costs.yaml.

    configs/policy/threshold.yaml carries no model stamp — t* = c / (r × LTV)
    is a pure function of costs.yaml, model-independent by construction — so
    its provenance is pinned by this hash instead: same hash means the model
    changed, a different hash means the cost assumptions did. path defaults
    to the canonical project location; pass an explicit path in tests.
    """
    resolved = (
        path if path is not None else get_project_root() / "configs" / "costs.yaml"
    )
    return hashlib.sha256(resolved.read_bytes()).hexdigest()


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


def expected_value_at_threshold(
    proba: NDArray[np.float64], y: NDArray[np.int_], scenario: CostScenario, t: float
) -> float:
    """Realized per-customer expected value of "contact iff proba >= t", at one threshold t.

    Same ev(t) = [TP(t)·(r·LTV − c) − FP(t)·c] / n formula
    expected_value_curve sweeps over every distinct threshold — this
    evaluates it directly at a single t instead of reading it off the curve.
    The one implementation both call: Phase 7's economics.py imports this
    function rather than redefining p·r·LTV − c a second time, since two
    implementations would agree only until one of them changed.
    """
    contacted = proba >= t
    n = len(y)
    tp = int(np.sum(contacted & (y == 1)))
    fp = int(np.sum(contacted & (y == 0)))
    return float(
        (
            tp * (scenario.retention_rate * scenario.ltv - scenario.cost)
            - fp * scenario.cost
        )
        / n
    )


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
        f"runs:/{run_id}/calibration_summary.json"
    )
    return summary


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
    """Render and save each scenario's expected-value curve overlaid, with its
    closed-form t* marked — where EV actually peaks vs. where the cost model
    says to cut, for all three scenarios at once.

    ev_curves is precomputed once by the caller and reused for both this plot
    and the persisted ev_curve.parquet artifact — not recomputed here, so the
    picture can never silently diverge from the logged evidence (CLAUDE.md §
    Persist the evidence, not just the conclusion).
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


def run_threshold_step(model_version: str, cfg: DictConfig) -> dict[str, Any]:
    """Derive and ship the cost-sensitive threshold for a registered calibrated model.

    Resolved by explicit model_version, never the challenger alias: a
    re-calibration invalidates a previously-derived threshold, and an alias
    is a moving pointer that could later point at a different version.

    Recomputes the same leak-free OOF probabilities calibrate.py's method
    selection used (oof_calibrated_proba, over the method recorded in
    calibration_summary.json) — this module never sees a fitted estimator or
    the raw data split directly.

    threshold.json (the full, unsplit bundle — every scenario's complete
    diagnostic set, model stamp included) is logged onto the model's run as
    the audit-trail copy. What gets mirrored to disk splits along the line
    the closed form already draws, because the two halves have different
    lifetimes:
      - configs/policy/threshold.yaml — the policy: threshold/costs/
        retention_rate_sensitivity, pure functions of configs/costs.yaml
        (t* = c / (r × LTV) is model-independent by construction). Pinned by
        a costs_config_hash, carries no model_run_id/model_version — a file
        whose content is model-independent must not carry a model stamp, or
        an emergency rollback to a different champion leaves it silently
        describing the wrong model.
      - threshold_validation.json — the model-dependent half (argmax-EV
        threshold, its bootstrap CI, within_ci, implied contact rate,
        dev_ev_at_t_star, calibration_method), logged only as an artifact on
        this model's run — same rule as drift_reference.json: an artifact
        describing a specific version travels with that version, never sits
        at a fixed path a rollback could leave stale.

    Every scenario's full diagnostic bundle ships in both threshold_payload
    (unsplit) and validation_payload (model-dependent half only) — not just
    its t* — conservative and optimistic are equally auditable, not merely
    stored numbers nobody can validate later. base is still the only
    scenario that drives the top-level threshold/within_ci warning: it is
    the shipped operating point, the others are reference alternatives.

    Logs t_star_{scenario}, implied_contact_rate_{scenario}, and
    dev_ev_at_t_star_{scenario} as MLflow metrics for every scenario, and
    persists the EV-curve points themselves as ev_curve.parquet — the curve
    exists today only as pixels in expected_value_by_scenario.png.

    Three figures are logged onto the run's figures/ artifacts and mirrored
    to reports/figures/: threshold_sensitivity.png (base's t* vs. retention
    rate), threshold_by_scenario.png (closed-form t* vs. empirical argmax +
    its bootstrap CI, all three scenarios), and
    expected_value_by_scenario.png (each scenario's EV curve overlaid, with
    its t* marked).

    Like reports/figures/reliability_diagram.png, these three are overwritten
    on every call — the copy on disk reflects the most recent local run, not
    a specific run_id. threshold_validation.json is self-describing
    (model_run_id, model_version), so that is the right place to confirm
    provenance, not the PNGs' mtimes.
    """
    mlflow.set_tracking_uri(resolve_tracking_uri(str(cfg.mlflow.tracking_uri)))
    registered_model_name = str(cfg.mlflow.registered_model_name)
    client = mlflow.tracking.MlflowClient()
    run_id = str(client.get_model_version(registered_model_name, model_version).run_id)

    manifest = load_training_manifest(run_id, cfg)
    calibration_summary = load_calibration_summary(run_id, cfg)
    method = str(calibration_summary["method"])

    committed_features = committed_features_from_manifest(manifest)
    pipeline = unfitted_pipeline_from_manifest(manifest)
    X_dev, y_dev = load_dev_features(committed_features)

    oof_proba = oof_calibrated_proba(pipeline, method, X_dev, y_dev, cfg)
    y_dev_arr = y_dev.to_numpy()

    costs_cfg = load_costs_config(get_project_root() / str(cfg.paths.costs_config))
    scenarios = resolve_all_scenarios(X_dev["monthlycharges"], y_dev, costs_cfg)

    n_bootstrap = int(costs_cfg.argmax_ev_bootstrap_n_samples)
    random_state = int(cfg.threshold.random_state)

    results = {
        name: derive_threshold(
            oof_proba, y_dev_arr, scenario, n_bootstrap, random_state
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

    figure_path = (
        get_project_root() / str(cfg.paths.figures) / "threshold_sensitivity.png"
    )
    _save_sensitivity_plot(sweep, float(base["threshold"]), figure_path)

    scenario_threshold_path = (
        get_project_root() / str(cfg.paths.figures) / "threshold_by_scenario.png"
    )
    _save_scenario_threshold_plot(results, scenario_threshold_path)

    # Computed once, shared by the plot and the persisted artifact — a second
    # recomputation is exactly the risk CLAUDE.md's "persist the evidence"
    # rule exists to close off.
    ev_curves = {
        name: expected_value_curve(oof_proba, y_dev_arr, scenario)
        for name, scenario in scenarios.items()
    }

    ev_curve_path = (
        get_project_root() / str(cfg.paths.figures) / "expected_value_by_scenario.png"
    )
    _save_scenario_ev_curve_plot(ev_curves, scenarios, ev_curve_path)

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
        "model_version": str(model_version),
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
        "model_version": str(model_version),
        "calibration_method": method,
        "argmax_ev_threshold": base["argmax_ev_threshold"],
        "argmax_ev_bootstrap_ci": base["argmax_ev_bootstrap_ci"],
        "within_ci": base["within_ci"],
        "implied_contact_rate": base["implied_contact_rate"],
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

    with mlflow.start_run(run_id=run_id):
        mlflow.log_dict(threshold_payload, "threshold.json")
        mlflow.log_dict(validation_payload, "threshold_validation.json")
        mlflow.log_artifact(str(figure_path), artifact_path="figures")
        mlflow.log_artifact(str(scenario_threshold_path), artifact_path="figures")
        mlflow.log_artifact(str(ev_curve_path), artifact_path="figures")

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
            mlflow.log_artifact(str(ev_curve_data_path))

        scenario_metrics: dict[str, float] = {}
        for name, result in results.items():
            scenario_metrics[f"t_star_{name}"] = result["threshold"]
            scenario_metrics[f"implied_contact_rate_{name}"] = result[
                "implied_contact_rate"
            ]
            scenario_metrics[f"dev_ev_at_t_star_{name}"] = result["dev_ev_at_t_star"]
        mlflow.log_metrics(scenario_metrics)

    policy_path = get_project_root() / str(cfg.paths.policy) / "threshold.yaml"
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(policy_payload), policy_path)

    logger.info(
        "threshold_derived",
        run_id=run_id,
        model_version=model_version,
        method=method,
        threshold=base["threshold"],
        within_ci=base["within_ci"],
        implied_contact_rate=base["implied_contact_rate"],
    )

    return {
        "threshold_payload": threshold_payload,
        "policy_payload": policy_payload,
        "validation_payload": validation_payload,
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
        cli_model_version = cfg.threshold.model_version
        if cli_model_version is None:
            raise ValueError(
                "threshold.model_version is required, e.g. `python -m "
                "telco_churn.models.threshold threshold.model_version=1`"
            )
        result = run_threshold_step(str(cli_model_version), cfg)
        logger.info(
            "threshold_step_done",
            threshold=result["threshold_payload"]["threshold"],
            model_version=result["threshold_payload"]["model_version"],
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
    except Exception as e:
        logger.error("threshold_failed", error=str(e), exc_info=True)
        sys.exit(1)

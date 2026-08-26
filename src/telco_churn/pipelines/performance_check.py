"""Binding promotion gate for the two routine, reserve-driven retrain cycles
(Phase 10a-ii).

`evaluate.py`'s sealed-test comparative regime stays the binding gate for a
rare/cold-start cycle (a genuinely new model generation), but never runs at
all on a routine cycle — this module replaces it there. Per the two-gate
split, exactly one gate module runs per cycle, never both, and neither ever
produces a discarded byproduct decision:

1. **Step 0 (non-binding).** Score the candidate against the sealed test
   partition via `models/sealed_test.py`'s dataset-agnostic functions,
   writing the same `test_predictions.parquet`/`metrics.json`/
   `economics.json` shape `evaluate.py` would have produced — this is what
   satisfies `error_analysis.py`/`register.py`'s per-cycle requirement for
   those artifacts now that `evaluate.py`'s CLI never runs this cycle. A
   *minimal-viable* pass, deliberately: `evaluate.py`'s remaining figure/
   sensitivity/sliced-diagnostics assembly lives in private helpers inside a
   `__main__`-bearing module and cannot be imported here (CLAUDE.md's
   no-cross-`__main__`-import rule) — full parity would require extracting
   that assembly too, out of scope for this pass.
2. **Steps 1-8 (binding).** Score the candidate against the reserve
   comparison window instead: the incumbent's own live score comes from
   `prediction_log` (never re-scored — it already scored these customers for
   real, at serving time), the label comes from `prediction_outcomes` once
   matured, and the candidate is backtested on `training_pool`'s frozen
   feature snapshot for that reserve month (never re-queried live from
   `customers_crm`). `models/sealed_test.py::comparative_deltas` /
   `build_gate_inputs` / `models/gate.py::decide_promotion` are the same
   functions `evaluate.py`'s rare-cycle path uses — the decision rule is not
   re-derived, only re-fed from three tables instead of one parquet.

This module holds no cross-cycle state and assumes serial invocation — the
"queue, don't parallelize" rule for a rare trigger arriving mid-comparison-
window (§D2) is `training_cycle.py`'s (Phase 10b) dispatch responsibility,
not this module's; nothing here tracks whether a comparison window is
already in flight. Nothing in this repo calls `run_performance_check` yet,
the same "plumbing ahead of its orchestrator" state
`models/train/common.py::load_training_pool_bundle` and
`data/training_pool.py`'s write-path-2 functions are already in.

`register.py` needs zero changes to consume this module's output (§D6): it
reads `promotion_decision.json`/`metrics.json` by `eval_run_id`, resolved
from `reports/eval_receipt.json` on first pass, exactly as it already does
for `evaluate.py`'s run — it is never told which module produced the run it
is given.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from omegaconf import DictConfig
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine

from telco_churn.data import reserve_ids
from telco_churn.features import SQL_FEATURE_COLS, build_feature_df
from telco_churn.models.artifacts import (
    committed_features_from_manifest,
    load_fitted_model,
    load_threshold_validation,
    load_training_manifest,
    resolve_champion_version,
)
from telco_churn.models.economics import capacity_budget_check, ev_by_k
from telco_churn.models.gate import (
    GateBars,
    check_threshold_provenance,
    check_threshold_screen_passed,
    decide_promotion,
)
from telco_churn.models.policy_config import (
    costs_config_hash,
    load_costs_config,
    load_model_promotion_bars,
    load_policy_thresholds,
    resolve_policy_scenarios,
    resolve_policy_thresholds_by_scenario,
)
from telco_churn.models.sealed_test import (
    build_gate_inputs,
    comparative_deltas,
    load_test_customer_ids,
    load_test_features,
    sealed_test_business_impact,
    sealed_test_calibration_report,
    sealed_test_classification_report,
    sealed_test_decile_lift,
    sealed_test_fixed_recall_profile,
    sealed_test_ranking_metrics,
)
from telco_churn.utils.hashing import content_hash
from telco_churn.utils.logging import get_logger
from telco_churn.utils.mlflow import (
    ensure_experiment_metadata,
    resolve_logged_model_id,
    resolve_tracking_uri,
    set_run_description,
    write_eval_receipt,
)
from telco_churn.utils.paths import get_project_root

__all__ = ["run_performance_check"]

logger = get_logger(__name__)

_RUN_DESCRIPTION = (
    "performance_check.py — routine reserve-driven retrain cycle. Binding "
    "verdict is the comparative decision against the reserve comparison "
    "window (see promotion_decision.json's reserve_month/comparison_cohort "
    "fields); the sealed-test metrics/economics on this run are a "
    "non-binding diagnostic pass."
)

_COMPARISON_COHORT_QUERY = text("""
    SELECT po.customerid, po.churned,
           COALESCE(pl.champion_probability, pl.probability) AS incumbent_probability
    FROM prediction_outcomes po
    JOIN LATERAL (
        SELECT pl2.probability, pl2.champion_probability
        FROM prediction_log pl2
        WHERE pl2.customerid = po.customerid
          AND pl2.predicted_at <= po.observed_at
        ORDER BY pl2.predicted_at DESC
        LIMIT 1
    ) pl ON true
    WHERE po.customerid IN :customerids
    ORDER BY po.customerid
    """).bindparams(bindparam("customerids", expanding=True))


def _load_candidate(
    run_id: str, model_version: str, model_uri: str, cfg: DictConfig
) -> dict[str, Any]:
    """Resolve the candidate's committed features/fitted model and re-check the
    dev-OOF pre-seal screen — the same independent re-check `evaluate.py`
    performs, per `gate.py::check_threshold_screen_passed`'s own docstring."""
    mlflow.set_tracking_uri(resolve_tracking_uri(str(cfg.mlflow.tracking_uri)))
    validation_payload = load_threshold_validation(run_id, cfg)
    logged_model_id = resolve_logged_model_id(model_version, cfg)
    check_threshold_provenance(validation_payload, logged_model_id)
    check_threshold_screen_passed(validation_payload)

    manifest = load_training_manifest(run_id, cfg)
    committed_features = committed_features_from_manifest(manifest)
    model = load_fitted_model(model_uri, cfg)
    return {
        "logged_model_id": logged_model_id,
        "committed_features": committed_features,
        "model": model,
    }


def _load_policy_context(cfg: DictConfig) -> dict[str, Any]:
    """Load the shipped policy thresholds/scenarios and bootstrap settings — same
    resolution `evaluate.py` uses, reused verbatim since it lives in the freely
    importable `policy_config.py`, not behind a `__main__` block."""
    policy = load_policy_thresholds(cfg)
    scenarios = resolve_policy_scenarios(policy)
    thresholds = resolve_policy_thresholds_by_scenario(policy)
    return {
        "scenarios": scenarios,
        "thresholds": thresholds,
        "base_scenario": scenarios["base"],
        "base_threshold": thresholds["base"],
        "n_bootstrap": int(cfg.evaluate.n_bootstrap),
        "random_state": int(cfg.evaluate.random_state),
    }


def _score_sealed_test(model: Any, committed_features: list[str]) -> dict[str, Any]:
    """Step 0 — score the candidate against the sealed test partition once."""
    X_test, y_test = load_test_features(committed_features)
    proba: NDArray[np.float64] = model.predict_proba(X_test)[:, 1]
    customer_ids = load_test_customer_ids()
    return {"y_test": y_test, "proba": proba, "customer_ids": customer_ids}


def _compute_sealed_test_metrics(
    y_test: pd.Series,
    proba: NDArray[np.float64],
    policy_ctx: dict[str, Any],
    cfg: DictConfig,
) -> dict[str, Any]:
    """Minimal-viable sealed-test metrics block — ranking, per-scenario
    classification, fixed-recall, calibration, decile lift, business impact.
    Deliberately excludes evaluate.py's sensitivity/sliced-diagnostics/figure
    assembly (module docstring)."""
    thresholds, scenarios = policy_ctx["thresholds"], policy_ctx["scenarios"]
    n_bootstrap, random_state = policy_ctx["n_bootstrap"], policy_ctx["random_state"]

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
    return {
        "ranking_metrics": ranking_metrics,
        "classification_rows": classification_rows,
        "fixed_recall_rows": fixed_recall_rows,
        "calibration_report": calibration_report,
        "decile_rows": decile_rows,
        "business_impact": business_impact,
    }


def _load_comparison_cohort(
    engine: Engine,
    reserve_month: int,
    reserve_manifest: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return this cycle's reserve comparison window: customerid, churned (the
    matured label), incumbent_probability (§D2 step 2's champion_probability-
    with-probability-fallback rule, applied in SQL via COALESCE — the two are
    never both non-null on the same row, per `prediction_log.py::
    build_log_rows`'s own "all four null together, or all four populated
    together" invariant on the dual-score fields).

    reserve_manifest defaults to reserve_ids() (disk read); pass an in-memory
    manifest in tests — the same "manifest defaults to disk, an in-memory
    frame in tests" pattern data/split.py's own dev_ids/test_ids/partition
    already use.

    Raises ValueError if any reserve-month customerid has no matured
    prediction_outcomes row yet — the release schedule promising a matured
    cohort is a precondition of this function running at all, not something
    it can silently tolerate a partial miss on.
    """
    manifest = reserve_manifest if reserve_manifest is not None else reserve_ids()
    month_ids = manifest.loc[
        manifest["reserve_month"] == reserve_month, "customerid"
    ].tolist()
    if not month_ids:
        raise ValueError(
            f"No customerids found for reserve_month={reserve_month!r} in the "
            "reserve manifest."
        )
    joined = pd.read_sql_query(
        _COMPARISON_COHORT_QUERY, engine, params={"customerids": month_ids}
    )
    missing = set(month_ids) - set(joined["customerid"])
    if missing:
        raise ValueError(
            f"_load_comparison_cohort: {len(missing)} customerid(s) in "
            f"reserve_month={reserve_month!r} have no matured "
            f"prediction_outcomes row yet: {sorted(missing)[:10]}"
        )
    return joined.sort_values("customerid").reset_index(drop=True)


def _score_candidate_on_reserve(
    model: Any,
    committed_features: list[str],
    engine: Engine,
    reserve_month: int,
    expected_ids: pd.Series,
) -> NDArray[np.float64]:
    """Step 4 — backtest the candidate on the reserve cohort's frozen
    `training_pool` feature snapshot (never re-queried live from
    `customers_crm` — the same rule §D4 establishes for training data).

    Reads `customer_features` scoped to exactly this `reserve_month`, not via
    `features/build.py::build_feature_query` (which always folds in the
    `reserve_month IS NULL` seed population alongside any requested months —
    the fold-forward *training* query shape, not what a comparison-only read
    needs).
    """
    cols = ", ".join(SQL_FEATURE_COLS)
    query = text(
        f"SELECT {cols} FROM customer_features WHERE reserve_month = :reserve_month"
    )
    raw = pd.read_sql_query(query, engine, params={"reserve_month": int(reserve_month)})
    if raw.empty:
        raise ValueError(
            f"No customer_features rows found for reserve_month={reserve_month!r} "
            "— has training_pool's write-path-2 reshape run for this cohort yet?"
        )
    engineered = build_feature_df(raw).set_index("customerid")

    missing = set(expected_ids) - set(engineered.index)
    if missing:
        raise ValueError(
            f"_score_candidate_on_reserve: {len(missing)} customerid(s) expected "
            f"from the comparison cohort are missing from customer_features for "
            f"reserve_month={reserve_month!r}: {sorted(missing)[:10]}"
        )
    aligned = engineered.loc[list(expected_ids), committed_features]
    proba: NDArray[np.float64] = model.predict_proba(aligned)[:, 1]
    return proba


def _compute_reserve_decision(
    cohort: pd.DataFrame,
    candidate_proba: NDArray[np.float64],
    policy_ctx: dict[str, Any],
    cfg: DictConfig,
    bars: GateBars,
) -> dict[str, Any]:
    """Steps 5-8 — the binding comparative decision against the reserve cohort.

    Reuses the identical `comparative_deltas`/`build_gate_inputs`/
    `decide_promotion` call sequence `evaluate.py`'s rare-cycle path uses
    (via `models/sealed_test.py::sealed_test_promotion_decision`'s own
    composition) — not re-derived here, since this cohort's `incumbent_proba`
    is already resolved (from `prediction_log`, never re-scored), unlike the
    rare path's `load_incumbent_proba` re-alignment machinery.
    """
    y_reserve = cohort["churned"].astype(int)
    incumbent_proba = cohort["incumbent_probability"].to_numpy(dtype=np.float64)

    ranking_metrics = sealed_test_ranking_metrics(
        y_reserve,
        candidate_proba,
        policy_ctx["n_bootstrap"],
        policy_ctx["random_state"],
    )
    classification_rows = sealed_test_classification_report(
        y_reserve,
        candidate_proba,
        policy_ctx["thresholds"],
        policy_ctx["n_bootstrap"],
        policy_ctx["random_state"],
    )
    calibration_report = sealed_test_calibration_report(
        y_reserve,
        candidate_proba,
        cfg,
        policy_ctx["n_bootstrap"],
        policy_ctx["random_state"],
    )
    deltas = comparative_deltas(
        y_reserve,
        candidate_proba,
        incumbent_proba,
        policy_ctx["base_threshold"],
        policy_ctx["n_bootstrap"],
        policy_ctx["random_state"],
    )
    gate_inputs = build_gate_inputs(
        ranking_metrics, classification_rows, calibration_report, "base", deltas
    )
    decision = decide_promotion(gate_inputs, "comparative", bars)
    return {
        "decision": decision,
        "ranking_metrics": ranking_metrics,
        "classification_rows": classification_rows,
        "calibration_report": calibration_report,
        "deltas": deltas,
        "cohort_size": len(cohort),
    }


def _assemble_payloads(
    model_version: str,
    run_id: str,
    sealed: dict[str, Any],
    sealed_metrics: dict[str, Any],
    reserve_result: dict[str, Any],
    reserve_month: int,
    policy_ctx: dict[str, Any],
    cfg: DictConfig,
    champion_version: str | None,
) -> dict[str, Any]:
    """Assemble metrics.json (sealed-test, diagnostic), economics.json
    (sealed-test), and promotion_decision.json (reserve cohort, binding).

    metrics_content_hash is stamped over metrics_payload as computed here —
    register.py's verification only checks metrics.json hasn't been silently
    swapped/regenerated since the decision was logged, it does not require
    the decision to be about the same population metrics.json describes (the
    routine cycle's whole point is that these are two different cohorts).
    """
    business_impact = sealed_metrics["business_impact"]
    metrics_payload: dict[str, Any] = {
        "model_version": model_version,
        "run_id": run_id,
        "champion_version": champion_version,
        "incumbent_summary": {
            "source": "prediction_log",
            "reserve_month": reserve_month,
            "cohort_size": reserve_result["cohort_size"],
        },
        "ranking": sealed_metrics["ranking_metrics"],
        "classification": sealed_metrics["classification_rows"],
        "fixed_recall_profile": sealed_metrics["fixed_recall_rows"],
        "calibration": sealed_metrics["calibration_report"],
        "decile_lift": sealed_metrics["decile_rows"],
        "business_impact": business_impact,
    }

    y_test_int = sealed["y_test"].to_numpy(dtype=np.int64)
    ev_curves = {
        name: ev_by_k(sealed["proba"], y_test_int, scenario)
        for name, scenario in policy_ctx["scenarios"].items()
    }
    costs_cfg = load_costs_config(get_project_root() / str(cfg.paths.costs_config))
    capacity_flags = capacity_budget_check(
        business_impact["scenarios"],
        float(costs_cfg.contact_capacity),
        float(costs_cfg.campaign_budget),
    )
    economics_payload: dict[str, Any] = {
        "ev_by_k": ev_curves,
        "ev_treat_all_by_scenario": {
            name: row["ev_treat_all"]
            for name, row in business_impact["scenarios"].items()
        },
        "capacity_budget_check": capacity_flags,
    }

    decision = reserve_result["decision"]
    promotion_decision_payload: dict[str, Any] = {
        **decision,
        "model_version": model_version,
        "eval_run_id": run_id,
        "comparison_cohort": "reserve",
        "reserve_month": reserve_month,
        "reserve_cohort_size": reserve_result["cohort_size"],
        "metrics_content_hash": content_hash(metrics_payload),
    }
    return {
        "metrics_payload": metrics_payload,
        "economics_payload": economics_payload,
        "promotion_decision_payload": promotion_decision_payload,
    }


def _log_performance_check_run(
    model_version: str,
    sealed: dict[str, Any],
    payloads: dict[str, Any],
    logged_model_id: str,
    policy_ctx: dict[str, Any],
    cfg: DictConfig,
) -> tuple[str, pd.DataFrame]:
    """Log every artifact onto a dedicated `performance_check` run — a sibling
    of `evaluation`, never appended to the training run (same "one run per
    stage" convention CLAUDE.md's MLflow section establishes)."""
    ensure_experiment_metadata(cfg)
    decision = payloads["promotion_decision_payload"]
    metrics_payload = payloads["metrics_payload"]
    economics_payload = payloads["economics_payload"]

    with mlflow.start_run(run_name="performance_check") as run:
        set_run_description(_RUN_DESCRIPTION)
        run_id = run.info.run_id
        decision["eval_run_id"] = run_id

        for scenario_name, scenario in policy_ctx["scenarios"].items():
            mlflow.log_params(
                {
                    f"cost_{scenario_name}_c": scenario.cost,
                    f"cost_{scenario_name}_r": scenario.retention_rate,
                    f"cost_{scenario_name}_ltv": scenario.ltv,
                    f"cost_{scenario_name}_arpu": scenario.arpu,
                }
            )
        mlflow.set_tag(
            "costs_config_hash",
            costs_config_hash(get_project_root() / str(cfg.paths.costs_config)),
        )
        mlflow.set_tag("reserve_month", str(decision["reserve_month"]))
        mlflow.set_tag("gate_regime", decision["regime"])
        mlflow.set_tag("gate_result", decision["gate"])
        for criterion_name, criterion in decision["criteria"].items():
            mlflow.set_tag(
                f"gate_criterion_{criterion_name}",
                "pass" if criterion["passed"] else "fail",
            )

        mlflow.log_dict(metrics_payload, "metrics.json")
        mlflow.log_dict(economics_payload, "economics.json")
        mlflow.log_dict(decision, "promotion_decision.json")

        test_predictions = pd.DataFrame(
            {
                "customerid": sealed["customer_ids"],
                "y_true": sealed["y_test"].reset_index(drop=True),
                "p_hat": sealed["proba"],
                "logged_model_id": logged_model_id,
            }
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            predictions_path = Path(tmp_dir) / "test_predictions.parquet"
            test_predictions.to_parquet(predictions_path, index=False)
            mlflow.log_artifact(str(predictions_path))

    return run_id, test_predictions


def _write_reports_mirror(
    payloads: dict[str, Any], test_predictions: pd.DataFrame, cfg: DictConfig
) -> None:
    """Mirror metrics.json/economics.json/promotion_decision.json/
    test_predictions.parquet to reports/ — the same local-disk shape
    `evaluate.py` writes, since `error_analysis.py` reads
    `reports/test_predictions.parquet` from a fixed local path, not the run."""
    reports_dir = get_project_root() / str(cfg.paths.reports)
    reports_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in (
        ("metrics.json", payloads["metrics_payload"]),
        ("economics.json", payloads["economics_payload"]),
        ("promotion_decision.json", payloads["promotion_decision_payload"]),
    ):
        with open(reports_dir / name, "w", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f, indent=2, default=str)
            f.write("\n")
    test_predictions.to_parquet(reports_dir / "test_predictions.parquet", index=False)


def run_performance_check(
    run_id: str,
    model_version: str,
    model_uri: str,
    reserve_month: int,
    engine: Engine,
    cfg: DictConfig,
    reserve_manifest: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Run the routine-cycle binding gate: a non-binding sealed-test scoring
    pass (Step 0) plus the binding comparative decision against the reserve
    comparison window (Steps 1-8).

    `run_id`/`model_version`/`model_uri` are the candidate's own training-run
    identity, already resolved by the caller (`training_cycle.py`, Phase
    10b) — same convention `evaluate.py::run_evaluation_step` takes them
    under. `reserve_month` is which matured reserve cohort is this cycle's
    comparison window; `engine` is a live Postgres engine for the
    `prediction_log`/`prediction_outcomes`/`training_pool` reads this module
    needs that `evaluate.py`'s rare-cycle path never does. `reserve_manifest`
    defaults to `reserve_ids()` (disk read); pass an in-memory manifest in
    tests, same as `_load_comparison_cohort`.

    Returns the same shape `run_evaluation_step` returns
    (`eval_run_id`/`model_version`/`champion_version`/`metrics`/`economics`/
    `promotion_decision`), plus `reserve_month` — so `register.py` can be
    handed this dict's `eval_run_id` exactly as it already is for a rare
    cycle's `evaluate.py` run (§D6: register.py needs zero changes to
    consume either gate's output).
    """
    candidate = _load_candidate(run_id, model_version, model_uri, cfg)
    model, committed_features = candidate["model"], candidate["committed_features"]

    policy_ctx = _load_policy_context(cfg)
    bars = load_model_promotion_bars(cfg)
    champion_version = resolve_champion_version(cfg)

    sealed = _score_sealed_test(model, committed_features)
    sealed_metrics = _compute_sealed_test_metrics(
        sealed["y_test"], sealed["proba"], policy_ctx, cfg
    )

    cohort = _load_comparison_cohort(engine, reserve_month, reserve_manifest)
    candidate_proba_reserve = _score_candidate_on_reserve(
        model, committed_features, engine, reserve_month, cohort["customerid"]
    )
    reserve_result = _compute_reserve_decision(
        cohort, candidate_proba_reserve, policy_ctx, cfg, bars
    )

    payloads = _assemble_payloads(
        model_version,
        run_id,
        sealed,
        sealed_metrics,
        reserve_result,
        reserve_month,
        policy_ctx,
        cfg,
        champion_version,
    )
    eval_run_id, test_predictions = _log_performance_check_run(
        model_version, sealed, payloads, candidate["logged_model_id"], policy_ctx, cfg
    )

    write_eval_receipt(model_version, eval_run_id, cfg)
    _write_reports_mirror(payloads, test_predictions, cfg)

    decision = payloads["promotion_decision_payload"]
    logger.info(
        "performance_check_done",
        run_id=run_id,
        eval_run_id=eval_run_id,
        model_version=model_version,
        champion_version=champion_version,
        reserve_month=reserve_month,
        gate_regime=decision["regime"],
        gate_result=decision["gate"],
    )

    return {
        "eval_run_id": eval_run_id,
        "model_version": model_version,
        "champion_version": champion_version,
        "reserve_month": reserve_month,
        "metrics": payloads["metrics_payload"],
        "economics": payloads["economics_payload"],
        "promotion_decision": payloads["promotion_decision_payload"],
    }

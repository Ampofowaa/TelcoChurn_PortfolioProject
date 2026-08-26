"""Cost-scenario resolution and the model-promotion gate bars — both pure
functions of on-disk YAML config, no MLflow, no estimator.

Shared by threshold.py (derives t*), evaluate.py (sealed-test business
impact, gate regime), error_analysis.py, economics.py, and register.py's
tag-writing — none of which own the other's need for these, so they live
here rather than in whichever module happened to define them first.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import mlflow
import mlflow.artifacts
import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf

from telco_churn.models.gate import GateBars
from telco_churn.utils.mlflow import resolve_tracking_uri
from telco_churn.utils.paths import get_project_root

__all__ = [
    "CostScenario",
    "load_costs_config",
    "costs_config_hash",
    "expected_value_at_threshold",
    "expected_value_per_customer",
    "load_policy_thresholds",
    "load_threshold_payload",
    "resolve_policy_scenarios",
    "resolve_policy_thresholds_by_scenario",
    "load_model_promotion_bars",
]


@dataclass(frozen=True)
class CostScenario:
    """One resolved cost scenario — the ARPU/LTV/cost/retention-rate values a threshold is derived from."""

    name: str
    arpu: float
    ltv: float
    cost: float
    retention_rate: float


def load_costs_config(path: Path | None = None) -> DictConfig:
    """Load configs/costs.yaml. path defaults to the canonical project location; pass an explicit path in tests."""
    resolved = (
        path if path is not None else get_project_root() / "configs" / "costs.yaml"
    )
    cfg = OmegaConf.load(resolved)
    assert isinstance(cfg, DictConfig)
    return cfg


def costs_config_hash(path: Path | None = None) -> str:
    """Return the sha256 content hash of configs/costs.yaml's resolved values.

    Pins provenance for reports/policy/threshold.yaml, which is model-
    independent by construction (t* = c/(r × LTV) is a pure function of
    costs.yaml) and so carries no model stamp: same hash means the model
    changed, a different hash means the cost assumptions did. Hashes the
    *resolved* values (OmegaConf, JSON-encoded with sorted keys), not the
    file's raw bytes, so a cosmetic edit (comment, line-ending change) can't
    register as a false cost-assumption change. Same idiom as tuning.py's
    `_study_name` content-addressing.
    """
    content = OmegaConf.to_container(load_costs_config(path), resolve=True)
    encoded = json.dumps(content, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def expected_value_at_threshold(
    proba: NDArray[np.float64], y: NDArray[np.int_], scenario: CostScenario, t: float
) -> float:
    """Realized per-customer expected value of "contact iff proba >= t", at one threshold t.

    Same ev(t) = [TP(t)·(r·LTV − c) − FP(t)·c] / n formula
    threshold.py's expected_value_curve sweeps over every distinct
    threshold — this evaluates it directly at a single t instead of reading
    it off the curve. The one implementation both threshold.py's
    derive_threshold and economics.py call, rather than redefining
    p·r·LTV − c a second time, since two implementations would agree only
    until one of them changed.
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


def expected_value_per_customer(
    proba: NDArray[np.float64], scenario: CostScenario
) -> NDArray[np.float64]:
    """Prospective per-customer expected value, elementwise: p_i * (r * LTV) - c.

    Distinct from expected_value_at_threshold, which needs ground-truth y to
    report a *realized* aggregate EV for evaluation. Ranking candidates at
    serving time has no y to consult, only the model's own probability, so
    this is the formula serving/contact_policy.py::select_contacts ranks by.
    """
    return proba * (scenario.retention_rate * scenario.ltv) - scenario.cost


def load_policy_thresholds(cfg: DictConfig) -> DictConfig:
    """Load reports/policy/threshold.yaml — the model-independent scenario thresholds.

    Carries no model stamp by construction (t* = c/(r × LTV) is a pure
    function of cost parameters, never of the model). Lives here rather than
    in evaluate.py because threshold.py is the module that writes the file
    (`run_threshold_step`) — reading it back stays writer/reader local.
    """
    path = get_project_root() / str(cfg.paths.policy) / "threshold.yaml"
    loaded = OmegaConf.load(path)
    assert isinstance(loaded, DictConfig)
    return loaded


def load_threshold_payload(
    run_id: str, cfg: DictConfig, threshold_run_id: str | None = None
) -> DictConfig:
    """Load threshold/threshold.json off the model's own MLflow run, or a costs.yaml-only re-run's fresh one.

    The model-run-scoped counterpart to load_policy_thresholds' local-file
    read: reports/policy/threshold.yaml is model-independent by construction
    (t* is a pure function of costs.yaml, never of which model is deployed)
    and lives on local disk, which a served container has no guarantee even
    exists. serving/predict.py resolves the champion's (or challenger's) own
    run_id from its ModelVersion and reads this artifact instead, so the
    threshold travels with the model version the same way drift_reference.json
    does — a rollback to a different champion must not leave the API serving
    a threshold derived for a version it no longer runs.

    threshold_run_id, given, is that model version's own threshold_run_id
    tag (register.py's tag_threshold_rerun) — the fresh, small run a
    costs.yaml-only change re-derived t* onto, superseding the training
    run's original threshold.json without minting a new model version. None
    (the common case: no costs.yaml change since promotion) falls back to
    run_id, the training run threshold.py originally logged
    threshold/threshold.json onto.

    threshold.py's threshold_payload["scenarios"] is the raw per-scenario
    results dict — each entry already carries .threshold and
    .costs.{arpu,ltv,c,r}, the exact shape resolve_policy_scenarios/
    resolve_policy_thresholds_by_scenario read off load_policy_thresholds'
    own return value — so this function's DictConfig return is a drop-in for
    both of those, no separate reconstruction needed.
    """
    mlflow.set_tracking_uri(resolve_tracking_uri(str(cfg.mlflow.tracking_uri)))
    effective_run_id = threshold_run_id if threshold_run_id else run_id
    payload = mlflow.artifacts.load_dict(
        f"runs:/{effective_run_id}/threshold/threshold.json"
    )
    loaded = OmegaConf.create(payload)
    assert isinstance(loaded, DictConfig)
    return loaded


def resolve_policy_scenarios(policy: DictConfig) -> dict[str, CostScenario]:
    """Reconstruct each shipped scenario's CostScenario from reports/policy/threshold.yaml.

    Reads the already-resolved cost/LTV/ARPU values `run_threshold_step`
    persisted at derivation time, rather than recomputing ARPU quantiles from
    dev-set MonthlyCharges again — the shipped threshold and the sealed-test
    business-impact figures must rest on identical cost parameters.
    """
    return {
        str(name): CostScenario(
            name=str(name),
            arpu=float(entry.costs.arpu),
            ltv=float(entry.costs.ltv),
            cost=float(entry.costs.c),
            retention_rate=float(entry.costs.r),
        )
        for name, entry in policy.scenarios.items()
    }


def resolve_policy_thresholds_by_scenario(policy: DictConfig) -> dict[str, float]:
    """{scenario_name: t*} from reports/policy/threshold.yaml."""
    return {
        str(name): float(entry.threshold) for name, entry in policy.scenarios.items()
    }


def load_model_promotion_bars(cfg: DictConfig) -> GateBars:
    """Load configs/model_promotion.yaml and build the GateBars it defines.

    Used by decide_promotion and by threshold.py's dev-OOF calibration-slope
    screen. Loaded by path (OmegaConf.load), never through Hydra's
    defaults/CLI-override composition: a bar that decides whether a model
    ships must not be movable by a command-line override with no diff and no
    review — the same reason load_policy_thresholds bypasses composition for
    costs.yaml's derivative.
    """
    path = get_project_root() / str(cfg.paths.model_promotion_config)
    loaded = OmegaConf.load(path)
    assert isinstance(loaded, DictConfig)
    return GateBars(
        pr_auc_bar=float(loaded.pr_auc_bar),
        recall_bar=float(loaded.recall_bar),
        calibration_slope_band=(
            float(loaded.calibration_slope_band[0]),
            float(loaded.calibration_slope_band[1]),
        ),
        pr_auc_materiality_threshold=float(loaded.pr_auc_materiality_threshold),
        brier_non_inferiority_margin=float(loaded.brier_non_inferiority_margin),
        recall_non_inferiority_margin=float(loaded.recall_non_inferiority_margin),
    )

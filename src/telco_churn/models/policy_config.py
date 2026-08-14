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

import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf

from telco_churn.models.gate import GateBars
from telco_churn.utils.paths import get_project_root

__all__ = [
    "CostScenario",
    "load_costs_config",
    "costs_config_hash",
    "expected_value_at_threshold",
    "load_policy_thresholds",
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

    Pins provenance for configs/policy/threshold.yaml, which is model-
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


def load_policy_thresholds(cfg: DictConfig) -> DictConfig:
    """Load configs/policy/threshold.yaml — the model-independent scenario thresholds.

    Carries no model stamp by construction (t* = c/(r × LTV) is a pure
    function of cost parameters, never of the model). Lives here rather than
    in evaluate.py because threshold.py is the module that writes the file
    (`run_threshold_step`) — reading it back stays writer/reader local.
    """
    path = get_project_root() / str(cfg.paths.policy) / "threshold.yaml"
    loaded = OmegaConf.load(path)
    assert isinstance(loaded, DictConfig)
    return loaded


def resolve_policy_scenarios(policy: DictConfig) -> dict[str, CostScenario]:
    """Reconstruct each shipped scenario's CostScenario from configs/policy/threshold.yaml.

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
    """{scenario_name: t*} from configs/policy/threshold.yaml."""
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

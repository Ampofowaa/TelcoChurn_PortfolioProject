"""Unit tests for telco_churn.pipelines.performance_check's pure wiring —
Phase 10a-ii.

The Postgres-touching reads (_load_comparison_cohort's champion_probability-
vs-probability fallback, _score_candidate_on_reserve's reserve_month scoping)
are covered by tests/integration/test_performance_check_postgres.py instead —
this file covers _compute_reserve_decision/_assemble_payloads, which need no
database. The underlying gate primitives
(comparative_deltas/build_gate_inputs/decide_promotion) are already covered
by test_sealed_test.py/test_gate.py; these tests only check this module wires
them together correctly on a reserve-shaped cohort frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf

from telco_churn.models.gate import GateBars
from telco_churn.models.policy_config import (
    resolve_policy_scenarios,
    resolve_policy_thresholds_by_scenario,
)
from telco_churn.pipelines.performance_check import (
    _assemble_payloads,
    _compute_reserve_decision,
    _compute_sealed_test_metrics,
)
from telco_churn.utils.hashing import content_hash

_N_BOOTSTRAP = 200
_RANDOM_STATE = 42

_BARS = GateBars(
    pr_auc_bar=0.60,
    recall_bar=0.65,
    calibration_slope_band=(0.80, 1.25),
    pr_auc_materiality_threshold=0.005,
    brier_non_inferiority_margin=0.005,
    recall_non_inferiority_margin=0.03,
)

# A deliberately wide calibration-slope band, used only by the "gate passes"
# test below. This module's own probabilities are a linear-plus-Gaussian-
# noise synthetic construction (never passed through a sigmoid), so they are
# never genuinely calibrated the way a real trained model's are — a narrow
# band would fail on calibration_slope specifically, which is not what that
# test is checking (PR-AUC/recall/Brier admission, the wiring this module
# owns; calibration_slope's own pass/fail behavior is test_gate.py's job).
_WIDE_CALIBRATION_BARS = GateBars(
    pr_auc_bar=0.60,
    recall_bar=0.65,
    calibration_slope_band=(0.0, 100.0),
    pr_auc_materiality_threshold=0.005,
    brier_non_inferiority_margin=0.005,
    recall_non_inferiority_margin=0.03,
)


@pytest.fixture
def eval_cfg() -> DictConfig:
    return OmegaConf.create(
        {
            "calibration": {"ece_n_bins": 5, "ece_strategy": "uniform"},
            "training_setup": {"fixed_recall_thresholds": [0.7, 0.8, 0.9]},
        }
    )


@pytest.fixture
def policy_fixture() -> DictConfig:
    return OmegaConf.create(
        {
            "scenarios": {
                "base": {
                    "threshold": 0.39,
                    "costs": {"c": 67.76, "r": 0.30, "ltv": 573.12, "arpu": 79.6},
                },
            }
        }
    )


@pytest.fixture
def policy_ctx(policy_fixture: DictConfig) -> dict[str, object]:
    scenarios = resolve_policy_scenarios(policy_fixture)
    thresholds = resolve_policy_thresholds_by_scenario(policy_fixture)
    return {
        "scenarios": scenarios,
        "thresholds": thresholds,
        "base_scenario": scenarios["base"],
        "base_threshold": thresholds["base"],
        "n_bootstrap": _N_BOOTSTRAP,
        "random_state": _RANDOM_STATE,
    }


@pytest.fixture
def reserve_cohort() -> pd.DataFrame:
    """A moderately-separable synthetic reserve comparison cohort — 200 rows,
    ~27% prevalence. incumbent_probability stands in for whatever
    _load_comparison_cohort's COALESCE resolved (already tested against a
    real Postgres in the integration suite)."""
    rng = np.random.default_rng(11)
    n = 300
    churned = (rng.random(n) < 0.27).astype(int)
    incumbent = np.clip(churned * 0.30 + rng.normal(0.25, 0.18, size=n), 0.001, 0.999)
    return pd.DataFrame(
        {
            "customerid": [f"cust-{i}" for i in range(n)],
            "churned": churned.astype(bool),
            "incumbent_probability": incumbent,
        }
    )


def _candidate_proba_better_than(cohort: pd.DataFrame) -> np.ndarray:
    """A candidate probability vector that ranks meaningfully better than the
    incumbent on the same rows — its own independent linear-plus-noise draw
    with a stronger signal and less noise, not a deterministic push off the
    incumbent's own values (which produces implausibly perfect separation)."""
    rng = np.random.default_rng(23)
    y = cohort["churned"].to_numpy(dtype=float)
    return np.clip(y * 0.40 + rng.normal(0.25, 0.14, size=len(cohort)), 0.001, 0.999)


# ---------------------------------------------------------------------------
# _compute_reserve_decision
# ---------------------------------------------------------------------------


def test_compute_reserve_decision_returns_expected_keys(
    reserve_cohort: pd.DataFrame, policy_ctx: dict[str, object], eval_cfg: DictConfig
) -> None:
    candidate_proba = _candidate_proba_better_than(reserve_cohort)
    result = _compute_reserve_decision(
        reserve_cohort, candidate_proba, policy_ctx, eval_cfg, _BARS
    )

    assert set(result) == {
        "decision",
        "ranking_metrics",
        "classification_rows",
        "calibration_report",
        "deltas",
        "cohort_size",
    }
    assert result["decision"]["regime"] == "comparative"
    assert result["cohort_size"] == len(reserve_cohort)


def test_compute_reserve_decision_gate_passes_when_candidate_clearly_better(
    reserve_cohort: pd.DataFrame, policy_ctx: dict[str, object], eval_cfg: DictConfig
) -> None:
    """A candidate that ranks better than the incumbent on the reserve cohort
    should pass the PR-AUC/recall/Brier admission — the same admitting shape
    evaluate.py's rare-cycle path already exercises, now fed from a
    reserve-shaped cohort frame instead of the sealed test set. Uses
    _WIDE_CALIBRATION_BARS (see its own comment) to isolate that wiring from
    this fixture's synthetic, never-actually-calibrated probabilities."""
    candidate_proba = _candidate_proba_better_than(reserve_cohort)
    result = _compute_reserve_decision(
        reserve_cohort, candidate_proba, policy_ctx, eval_cfg, _WIDE_CALIBRATION_BARS
    )

    assert result["decision"]["gate"] == "pass"
    assert result["deltas"]["pr_auc_delta_obs"] > 0


def test_compute_reserve_decision_gate_fails_when_candidate_identical_to_incumbent(
    reserve_cohort: pd.DataFrame, policy_ctx: dict[str, object], eval_cfg: DictConfig
) -> None:
    """An identical candidate has a zero paired-bootstrap delta — the
    comparative regime's materiality/CI-lower-bound-above-zero requirement
    must reject it, never treat a zero delta as passing."""
    candidate_proba = reserve_cohort["incumbent_probability"].to_numpy()
    result = _compute_reserve_decision(
        reserve_cohort, candidate_proba, policy_ctx, eval_cfg, _BARS
    )

    assert result["decision"]["gate"] == "fail"
    assert result["decision"]["criteria"]["pr_auc"]["delta_obs"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _assemble_payloads
# ---------------------------------------------------------------------------


def test_assemble_payloads_stamps_metrics_content_hash_matching_metrics_json(
    reserve_cohort: pd.DataFrame, policy_ctx: dict[str, object], eval_cfg: DictConfig
) -> None:
    """register.py's _load_and_verify_evaluation_artifacts re-hashes
    metrics.json and compares it against promotion_decision.json's
    metrics_content_hash — this must always match, by construction."""
    n = 150
    rng = np.random.default_rng(3)
    y_test = pd.Series((rng.random(n) < 0.27).astype(int), name="churn")
    proba = np.clip(
        y_test.to_numpy() * 0.4 + rng.normal(0.25, 0.15, size=n), 0.001, 0.999
    )
    sealed = {
        "y_test": y_test,
        "proba": proba,
        "customer_ids": pd.Series([f"sealed-{i}" for i in range(n)]),
    }
    sealed_metrics = _compute_sealed_test_metrics(y_test, proba, policy_ctx, eval_cfg)

    candidate_proba = _candidate_proba_better_than(reserve_cohort)
    reserve_result = _compute_reserve_decision(
        reserve_cohort, candidate_proba, policy_ctx, eval_cfg, _BARS
    )

    cfg = OmegaConf.create({"paths": {"costs_config": "configs/costs.yaml"}})
    payloads = _assemble_payloads(
        "7",
        "run-abc",
        sealed,
        sealed_metrics,
        reserve_result,
        reserve_month=1,
        policy_ctx=policy_ctx,
        cfg=cfg,
        champion_version="6",
    )

    stamped_hash = payloads["promotion_decision_payload"]["metrics_content_hash"]
    assert stamped_hash == content_hash(payloads["metrics_payload"])
    assert payloads["promotion_decision_payload"]["reserve_month"] == 1
    assert payloads["promotion_decision_payload"]["comparison_cohort"] == "reserve"
    assert payloads["metrics_payload"]["champion_version"] == "6"
    assert payloads["metrics_payload"]["incumbent_summary"]["reserve_month"] == 1

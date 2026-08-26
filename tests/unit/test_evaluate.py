"""Unit tests for telco_churn.models.evaluate — sealed-test evaluation (Phase 7)."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import mlflow.artifacts
import mlflow.sklearn
import mlflow.tracking
import numpy as np
import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import LogisticRegression

import telco_churn.models.evaluate as evaluate
from telco_churn.models.economics import capacity_budget_check
from telco_churn.models.evaluate import (
    flatten_metrics_summary,
    resolve_incumbent_summary,
)
from telco_churn.models.gate import (
    GateBars,
    decide_promotion,
)
from telco_churn.models.policy_config import (
    CostScenario,
    load_model_promotion_bars,
    resolve_policy_scenarios,
    resolve_policy_thresholds_by_scenario,
)
from telco_churn.models.sealed_test import (
    build_gate_inputs,
    sealed_test_business_impact,
    sealed_test_calibration_report,
    sealed_test_classification_report,
    sealed_test_ranking_metrics,
)
from telco_churn.utils.hashing import content_hash

_N_BOOTSTRAP = 200
_RANDOM_STATE = 42

# Local test fixture values — not tied to configs/model_promotion.yaml's real
# policy, same convention as test_gate.py: this file's tests must hold for
# whatever bars they are handed, not merely today's policy numbers.
_BARS = GateBars(
    pr_auc_bar=0.60,
    recall_bar=0.65,
    calibration_slope_band=(0.80, 1.25),
    pr_auc_materiality_threshold=0.005,
    brier_non_inferiority_margin=0.005,
    recall_non_inferiority_margin=0.03,
)


_FAKE_DATA_CONTENT_HASH = "deadbeef" * 8


@pytest.fixture(scope="module", autouse=True)
def _stub_data_content_hash() -> Iterator[None]:
    """_log_evaluation_run stamps data_content_hash via features_sha256() with
    no path override, resolving to the real, gitignored
    datasets/processed/telco_churn_features.parquet — absent on a fresh
    checkout. Every test below scores synthetic fixtures, never real
    processed data, so stub the hash.
    """
    mp = pytest.MonkeyPatch()
    mp.setattr(evaluate, "features_sha256", lambda path=None: _FAKE_DATA_CONTENT_HASH)
    yield
    mp.undo()


@pytest.fixture
def eval_cfg() -> DictConfig:
    """Small bin count for speed — same shape as production config, not its values."""
    return OmegaConf.create(
        {"calibration": {"ece_n_bins": 5, "ece_strategy": "uniform"}}
    )


@pytest.fixture
def y_proba_fixture() -> tuple[pd.Series, np.ndarray]:
    """A moderately-separable synthetic sealed-test fixture: 300 rows, ~27% prevalence."""
    rng = np.random.default_rng(7)
    n = 300
    y = pd.Series((rng.random(n) < 0.27).astype(int), name="churn")
    proba = np.clip(y.to_numpy() * 0.4 + rng.normal(0.25, 0.15, size=n), 0.001, 0.999)
    return y, proba


@pytest.fixture
def policy_fixture() -> DictConfig:
    """Same shape as reports/policy/threshold.yaml's `scenarios` block."""
    return OmegaConf.create(
        {
            "scenarios": {
                "conservative": {
                    "threshold": 0.27,
                    "costs": {"c": 22.0, "r": 0.20, "ltv": 408.0, "arpu": 56.7},
                },
                "base": {
                    "threshold": 0.39,
                    "costs": {"c": 67.76, "r": 0.30, "ltv": 573.12, "arpu": 79.6},
                },
                "optimistic": {
                    "threshold": 0.50,
                    "costs": {"c": 134.78, "r": 0.40, "ltv": 678.24, "arpu": 94.2},
                },
            }
        }
    )


@pytest.fixture
def scenarios_fixture(policy_fixture: DictConfig) -> dict[str, CostScenario]:
    return resolve_policy_scenarios(policy_fixture)


@pytest.fixture
def thresholds_fixture(policy_fixture: DictConfig) -> dict[str, float]:
    return resolve_policy_thresholds_by_scenario(policy_fixture)


# ---------------------------------------------------------------------------
# resolve_policy_scenarios / resolve_policy_thresholds_by_scenario
# ---------------------------------------------------------------------------


def test_resolve_policy_scenarios_one_per_scenario(policy_fixture: DictConfig) -> None:
    """One CostScenario per named scenario, with fields read from `costs`."""
    scenarios = resolve_policy_scenarios(policy_fixture)
    assert set(scenarios) == {"conservative", "base", "optimistic"}
    base = scenarios["base"]
    assert base.name == "base"
    assert base.cost == pytest.approx(67.76)
    assert base.retention_rate == pytest.approx(0.30)
    assert base.ltv == pytest.approx(573.12)
    assert base.arpu == pytest.approx(79.6)


def test_resolve_policy_thresholds_by_scenario(policy_fixture: DictConfig) -> None:
    """One threshold per named scenario, matching the policy's own `threshold` field."""
    thresholds = resolve_policy_thresholds_by_scenario(policy_fixture)
    assert thresholds == {"conservative": 0.27, "base": 0.39, "optimistic": 0.50}


# ---------------------------------------------------------------------------
# _assemble_metrics_and_economics_payloads — capacity/budget check (QA #4)
# ---------------------------------------------------------------------------


def _minimal_assemble_inputs(
    business_impact: dict[str, Any],
    scenarios: dict[str, CostScenario],
    contact_capacity: float,
    campaign_budget: float,
) -> dict[str, Any]:
    """Stub every _assemble_metrics_and_economics_payloads input this QA
    check doesn't exercise, so each test only varies costs_cfg's limits."""
    return {
        "core_metrics": {
            "ranking_metrics": {},
            "classification_rows": [],
            "fixed_recall_rows": [],
            "calibration_report": {},
            "decile_rows": [],
            "business_impact": business_impact,
        },
        "sliced": {
            "test_ranking_slices": [],
            "test_decision_slices": [],
            "test_calibration_slices": [],
            "test_business_impact_slices": [],
            "test_equal_opportunity_by_axis": {},
            "test_demographic_parity_by_axis": {},
            "test_equal_opportunity_diff": float("nan"),
            "test_demographic_parity_diff": float("nan"),
        },
        "sensitivity_block": {
            "sensitivity": {},
            "retention_rate_values": [],
            "cost_values": [],
            "costs_cfg": OmegaConf.create(
                {
                    "contact_capacity": contact_capacity,
                    "campaign_budget": campaign_budget,
                }
            ),
        },
        "decision_result": {
            "decision": {"gate": "reject", "regime": "cold_start"},
            "champion_version": None,
            "incumbent_summary": None,
        },
        "policy_ctx": {"scenarios": scenarios},
    }


def test_assemble_payloads_flags_and_warns_when_capacity_and_budget_exceeded(
    monkeypatch: pytest.MonkeyPatch,
    y_proba_fixture: tuple[pd.Series, np.ndarray],
    scenarios_fixture: dict[str, CostScenario],
    thresholds_fixture: dict[str, float],
) -> None:
    """economics.json's capacity_budget_check reflects a real breach and logs
    a warning — diagnostic only (ANALYSIS.md §0: EV/economics never gate)."""
    y, proba = y_proba_fixture
    business_impact = sealed_test_business_impact(
        y, proba, scenarios_fixture, thresholds_fixture, _N_BOOTSTRAP, _RANDOM_STATE
    )
    inputs = _minimal_assemble_inputs(
        business_impact, scenarios_fixture, contact_capacity=1, campaign_budget=1.0
    )
    warning_mock = Mock()
    monkeypatch.setattr(evaluate.logger, "warning", warning_mock)

    payloads = evaluate._assemble_metrics_and_economics_payloads(
        "1", "run-id", y, proba, **inputs
    )

    capacity_flags = payloads["economics_payload"]["capacity_budget_check"]
    assert all(
        flags["over_capacity"] and flags["over_budget"]
        for flags in capacity_flags.values()
    )
    assert warning_mock.call_count == len(capacity_flags)
    assert all(
        c.args[0] == "capacity_or_budget_exceeded" for c in warning_mock.call_args_list
    )


def test_assemble_payloads_no_warning_within_limits(
    monkeypatch: pytest.MonkeyPatch,
    y_proba_fixture: tuple[pd.Series, np.ndarray],
    scenarios_fixture: dict[str, CostScenario],
    thresholds_fixture: dict[str, float],
) -> None:
    """No warning and every flag False when limits comfortably cover the shipped policy."""
    y, proba = y_proba_fixture
    business_impact = sealed_test_business_impact(
        y, proba, scenarios_fixture, thresholds_fixture, _N_BOOTSTRAP, _RANDOM_STATE
    )
    inputs = _minimal_assemble_inputs(
        business_impact,
        scenarios_fixture,
        contact_capacity=10_000,
        campaign_budget=10_000_000.0,
    )
    warning_mock = Mock()
    monkeypatch.setattr(evaluate.logger, "warning", warning_mock)

    payloads = evaluate._assemble_metrics_and_economics_payloads(
        "1", "run-id", y, proba, **inputs
    )

    capacity_flags = payloads["economics_payload"]["capacity_budget_check"]
    assert all(
        not flags["over_capacity"] and not flags["over_budget"]
        for flags in capacity_flags.values()
    )
    warning_mock.assert_not_called()


# ---------------------------------------------------------------------------
# load_model_promotion_bars
# ---------------------------------------------------------------------------


def test_load_model_promotion_bars_reads_real_config() -> None:
    """Reads the actual configs/model_promotion.yaml shipped with the project."""
    from telco_churn.utils.paths import compose_config

    cfg = compose_config()
    bars = load_model_promotion_bars(cfg)
    assert bars.pr_auc_bar == pytest.approx(0.60)
    assert bars.recall_bar == pytest.approx(0.65)
    assert bars.calibration_slope_band[0] == pytest.approx(0.80)
    assert bars.calibration_slope_band[1] == pytest.approx(1.25)
    assert bars.pr_auc_materiality_threshold == pytest.approx(0.005)
    assert bars.brier_non_inferiority_margin == pytest.approx(0.005)
    assert bars.recall_non_inferiority_margin == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# resolve_evaluation_champion
# ---------------------------------------------------------------------------


def test_resolve_evaluation_champion_explicit_override_skips_alias_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit evaluate.champion_version is returned verbatim, and
    resolve_champion_version (the live alias read) is never called."""

    def _fail_if_called(_cfg: DictConfig) -> str | None:
        raise AssertionError("resolve_champion_version must not be called")

    monkeypatch.setattr(evaluate, "resolve_champion_version", _fail_if_called)
    cfg = OmegaConf.create({"evaluate": {"champion_version": "3"}})
    assert evaluate.resolve_evaluation_champion(cfg) == "3"


def test_resolve_evaluation_champion_explicit_none_pins_cold_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The literal "none" override pins the cold-start regime explicitly,
    without touching the live alias — even if a champion happens to exist."""

    def _fail_if_called(_cfg: DictConfig) -> str | None:
        raise AssertionError("resolve_champion_version must not be called")

    monkeypatch.setattr(evaluate, "resolve_champion_version", _fail_if_called)
    cfg = OmegaConf.create({"evaluate": {"champion_version": "none"}})
    assert evaluate.resolve_evaluation_champion(cfg) is None


def test_resolve_evaluation_champion_omitted_falls_back_to_live_alias_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omitting the override (the config default, null) falls back to a live
    resolve_champion_version(cfg) read — the interactive/notebook default."""
    monkeypatch.setattr(evaluate, "resolve_champion_version", lambda _cfg: "7")
    cfg = OmegaConf.create({"evaluate": {"champion_version": None}})
    assert evaluate.resolve_evaluation_champion(cfg) == "7"


# ---------------------------------------------------------------------------
# resolve_incumbent_summary
# ---------------------------------------------------------------------------


def _tag_incumbent_version(
    registered_model_name: str,
    version: str,
    eval_run_id: str,
    *,
    include_costs_hash: bool = True,
    include_data_hash: bool = True,
) -> None:
    """Mint the tag set _tag_evaluated_model_version + _log_evaluation_run
    would leave on a real champion: the four gate criteria and eval_run_id
    on the version, costs_config_hash/data_content_hash on the run itself.
    """
    client = mlflow.tracking.MlflowClient()
    client.set_model_version_tag(registered_model_name, version, "test_pr_auc", "0.7")
    client.set_model_version_tag(registered_model_name, version, "test_recall", "0.8")
    client.set_model_version_tag(registered_model_name, version, "test_brier", "0.15")
    client.set_model_version_tag(
        registered_model_name, version, "test_calibration_slope", "1.02"
    )
    client.set_model_version_tag(
        registered_model_name, version, "eval_run_id", eval_run_id
    )
    if include_costs_hash:
        client.set_tag(eval_run_id, "costs_config_hash", "deadbeef")
    if include_data_hash:
        client.set_tag(eval_run_id, "data_content_hash", "feedface")


def test_resolve_incumbent_summary_returns_expected_fields(
    mlflow_test_experiment: Callable[[str], str],
) -> None:
    """Reads the four gate-criteria tags off the version plus
    costs_config_hash off its eval_run_id's own run — the field a reviewer
    needs to tell "recall regressed" apart from "recall was never measured
    under today's cost assumptions."
    """
    tracking_uri = mlflow_test_experiment("test_resolve_incumbent_summary")
    registered_model_name = "incumbent-summary-model"
    version, _model_id = _register_trivial_model(registered_model_name)
    with mlflow.start_run() as run:
        eval_run_id = run.info.run_id
    _tag_incumbent_version(registered_model_name, version, eval_run_id)

    cfg = OmegaConf.create(
        {
            "mlflow": {
                "tracking_uri": tracking_uri,
                "registered_model_name": registered_model_name,
            }
        }
    )
    summary = resolve_incumbent_summary(version, cfg)

    assert summary == {
        "version": version,
        "pr_auc": 0.7,
        "recall": 0.8,
        "brier": 0.15,
        "calibration_slope": 1.02,
        "costs_config_hash": "deadbeef",
        "data_content_hash": "feedface",
        "eval_run_id": eval_run_id,
    }


def test_resolve_incumbent_summary_raises_on_missing_gate_tags(
    mlflow_test_experiment: Callable[[str], str],
) -> None:
    """A version that never went through evaluate.py's tagging step (or was
    tagged before eval_run_id existed) must fail loudly, not silently report
    a partial/None summary a caller could mistake for a real one.
    """
    tracking_uri = mlflow_test_experiment("test_resolve_incumbent_summary")
    registered_model_name = "incumbent-summary-untagged"
    version, _model_id = _register_trivial_model(registered_model_name)

    cfg = OmegaConf.create(
        {
            "mlflow": {
                "tracking_uri": tracking_uri,
                "registered_model_name": registered_model_name,
            }
        }
    )
    with pytest.raises(RuntimeError, match="missing gate-criteria tags"):
        resolve_incumbent_summary(version, cfg)


def test_resolve_incumbent_summary_raises_on_missing_costs_config_hash(
    mlflow_test_experiment: Callable[[str], str],
) -> None:
    """A pre-existing champion whose eval run predates the costs_config_hash
    tag must fail loudly rather than silently omitting the field.
    """
    tracking_uri = mlflow_test_experiment("test_resolve_incumbent_summary")
    registered_model_name = "incumbent-summary-no-cost-hash"
    version, _model_id = _register_trivial_model(registered_model_name)
    with mlflow.start_run() as run:
        eval_run_id = run.info.run_id
    _tag_incumbent_version(
        registered_model_name, version, eval_run_id, include_costs_hash=False
    )

    cfg = OmegaConf.create(
        {
            "mlflow": {
                "tracking_uri": tracking_uri,
                "registered_model_name": registered_model_name,
            }
        }
    )
    with pytest.raises(RuntimeError, match="costs_config_hash"):
        resolve_incumbent_summary(version, cfg)


def test_resolve_incumbent_summary_raises_on_missing_data_content_hash(
    mlflow_test_experiment: Callable[[str], str],
) -> None:
    """A pre-existing champion whose eval run predates the data_content_hash
    tag must fail loudly rather than silently omitting the field.
    """
    tracking_uri = mlflow_test_experiment("test_resolve_incumbent_summary")
    registered_model_name = "incumbent-summary-no-data-hash"
    version, _model_id = _register_trivial_model(registered_model_name)
    with mlflow.start_run() as run:
        eval_run_id = run.info.run_id
    _tag_incumbent_version(
        registered_model_name, version, eval_run_id, include_data_hash=False
    )

    cfg = OmegaConf.create(
        {
            "mlflow": {
                "tracking_uri": tracking_uri,
                "registered_model_name": registered_model_name,
            }
        }
    )
    with pytest.raises(RuntimeError, match="data_content_hash"):
        resolve_incumbent_summary(version, cfg)


# ---------------------------------------------------------------------------
# load_incumbent_proba
# ---------------------------------------------------------------------------


def _log_champion_predictions(tmp_path: Path, df: pd.DataFrame) -> str:
    """Start a throwaway run and log df as its test_predictions.parquet — the
    shape _log_evaluation_run leaves on a real champion's own eval run."""
    predictions_path = tmp_path / "test_predictions.parquet"
    df.to_parquet(predictions_path, index=False)
    with mlflow.start_run() as run:
        mlflow.log_artifact(str(predictions_path))
        return str(run.info.run_id)


def test_load_incumbent_proba_aligns_by_customerid_regardless_of_row_order(
    mlflow_test_experiment: Callable[[str], str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Returned proba follows candidate_customer_ids's order, not whatever
    order the champion's historical parquet happened to store rows in."""
    mlflow_test_experiment("test_load_incumbent_proba")
    monkeypatch.setattr(evaluate, "features_sha256", lambda: "feedface")
    champion_df = pd.DataFrame(
        {
            "customerid": ["c-002", "c-000", "c-001"],
            "y_true": [1, 0, 0],
            "p_hat": [0.9, 0.1, 0.2],
            "logged_model_id": ["m-1", "m-1", "m-1"],
        }
    )
    eval_run_id = _log_champion_predictions(tmp_path, champion_df)

    candidate_customer_ids = pd.Series(["c-000", "c-001", "c-002"])
    candidate_y_test = pd.Series([0, 0, 1])
    cfg = OmegaConf.create({"mlflow": {"tracking_uri": mlflow.get_tracking_uri()}})

    proba = evaluate.load_incumbent_proba(
        "3", eval_run_id, "feedface", candidate_customer_ids, candidate_y_test, cfg
    )
    np.testing.assert_allclose(proba, [0.1, 0.2, 0.9])


def test_load_incumbent_proba_raises_on_data_content_hash_mismatch(
    mlflow_test_experiment: Callable[[str], str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A champion whose historical predictions were computed against a
    different processed-features file must fail loudly, even when the
    customerid set and labels would otherwise line up — a feature-pipeline
    change (e.g. a new engineered column) doesn't necessarily change which
    customers land in the test partition."""
    mlflow_test_experiment("test_load_incumbent_proba")
    monkeypatch.setattr(evaluate, "features_sha256", lambda: "current-hash")
    champion_df = pd.DataFrame(
        {
            "customerid": ["c-000", "c-001"],
            "y_true": [0, 1],
            "p_hat": [0.1, 0.8],
            "logged_model_id": ["m-1", "m-1"],
        }
    )
    eval_run_id = _log_champion_predictions(tmp_path, champion_df)

    candidate_customer_ids = pd.Series(["c-000", "c-001"])
    candidate_y_test = pd.Series([0, 1])
    cfg = OmegaConf.create({"mlflow": {"tracking_uri": mlflow.get_tracking_uri()}})

    with pytest.raises(RuntimeError, match="different processed-features file"):
        evaluate.load_incumbent_proba(
            "3",
            eval_run_id,
            "stale-hash",
            candidate_customer_ids,
            candidate_y_test,
            cfg,
        )


def test_load_incumbent_proba_raises_on_customer_set_mismatch(
    mlflow_test_experiment: Callable[[str], str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A champion whose historical predictions cover a different customer set
    (the split moved since it was last evaluated) must fail loudly."""
    mlflow_test_experiment("test_load_incumbent_proba")
    monkeypatch.setattr(evaluate, "features_sha256", lambda: "feedface")
    champion_df = pd.DataFrame(
        {
            "customerid": ["c-000", "c-999"],
            "y_true": [0, 1],
            "p_hat": [0.1, 0.8],
            "logged_model_id": ["m-1", "m-1"],
        }
    )
    eval_run_id = _log_champion_predictions(tmp_path, champion_df)

    candidate_customer_ids = pd.Series(["c-000", "c-001"])
    candidate_y_test = pd.Series([0, 0])
    cfg = OmegaConf.create({"mlflow": {"tracking_uri": mlflow.get_tracking_uri()}})

    with pytest.raises(RuntimeError, match="different customer set"):
        evaluate.load_incumbent_proba(
            "3", eval_run_id, "feedface", candidate_customer_ids, candidate_y_test, cfg
        )


def test_load_incumbent_proba_raises_on_label_mismatch(
    mlflow_test_experiment: Callable[[str], str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same customer set, but a shared customerid's recorded label disagrees
    between the champion's historical run and the candidate's own — a
    data-integrity signal, never silently resolved either way."""
    mlflow_test_experiment("test_load_incumbent_proba")
    monkeypatch.setattr(evaluate, "features_sha256", lambda: "feedface")
    champion_df = pd.DataFrame(
        {
            "customerid": ["c-000", "c-001"],
            "y_true": [0, 1],
            "p_hat": [0.1, 0.8],
            "logged_model_id": ["m-1", "m-1"],
        }
    )
    eval_run_id = _log_champion_predictions(tmp_path, champion_df)

    candidate_customer_ids = pd.Series(["c-000", "c-001"])
    candidate_y_test = pd.Series([0, 0])  # c-001 disagrees: 1 vs. 0
    cfg = OmegaConf.create({"mlflow": {"tracking_uri": mlflow.get_tracking_uri()}})

    with pytest.raises(RuntimeError, match="labels"):
        evaluate.load_incumbent_proba(
            "3", eval_run_id, "feedface", candidate_customer_ids, candidate_y_test, cfg
        )


# ---------------------------------------------------------------------------
# flatten_metrics_summary
# ---------------------------------------------------------------------------


def _metrics_payload_fixture() -> dict[str, Any]:
    return {
        "model_version": "3",
        "run_id": "run-abc",
        "champion_version": "2",
        "ranking": {"pr_auc": 0.65, "pr_auc_ci_lower": 0.60, "pr_auc_ci_upper": 0.70},
        "classification": [
            {
                "scenario": "base",
                "recall": 0.70,
                "recall_ci_lower": 0.64,
                "recall_ci_upper": 0.76,
            },
        ],
        "calibration": {
            "brier": 0.15,
            "bss": 0.30,
            "calibration_slope": {
                "slope": 1.02,
                "slope_ci_lower": 0.90,
                "slope_ci_upper": 1.15,
            },
        },
        "business_impact": {"scenarios": {"base": {"ev": 12.5}}},
        "sliced": {
            "test": {
                "equal_opportunity_diff": 0.04,
                "demographic_parity_diff": 0.06,
            },
        },
    }


def _decision_payload_fixture(regime: str) -> dict[str, Any]:
    criteria: dict[str, Any] = {
        "pr_auc": {"passed": True},
        "recall": {"passed": True},
        "brier": {"passed": True},
        "calibration_slope": {"passed": True},
    }
    if regime == "comparative":
        criteria["pr_auc"].update(delta_obs=0.02, delta_ci=[0.01, 0.03])
        criteria["recall"].update(delta_obs=0.01, delta_ci=[-0.01, 0.03])
        criteria["brier"].update(delta_obs=-0.01, delta_ci=[-0.02, 0.00])
    return {
        "regime": regime,
        "gate": "pass",
        "criteria": criteria,
        "eval_run_id": "eval-run-xyz",
    }


def test_flatten_metrics_summary_cold_start_omits_delta_keys() -> None:
    """Cold start has no incumbent to delta against — the comparative-only
    keys must be entirely absent, not present-and-null."""
    summary = flatten_metrics_summary(
        _metrics_payload_fixture(), _decision_payload_fixture("cold_start"), "hash-a"
    )
    for key in (
        "pr_auc_delta_obs",
        "recall_delta_obs",
        "brier_delta_obs",
    ):
        assert key not in summary
    assert summary["model_version"] == "3"
    assert summary["eval_run_id"] == "eval-run-xyz"
    assert summary["champion_version"] == "2"
    assert summary["regime"] == "cold_start"
    assert summary["gate"] == "pass"
    assert summary["costs_config_hash"] == "hash-a"
    assert summary["test_pr_auc"] == 0.65
    assert summary["test_pr_auc_ci_lower"] == 0.60
    assert summary["test_pr_auc_ci_upper"] == 0.70
    assert summary["test_recall"] == 0.70
    assert summary["test_recall_ci_lower"] == 0.64
    assert summary["test_recall_ci_upper"] == 0.76
    assert summary["test_brier"] == 0.15
    assert summary["test_bss"] == 0.30
    assert summary["test_calibration_slope"] == 1.02
    assert summary["test_calibration_slope_ci_lower"] == 0.90
    assert summary["test_calibration_slope_ci_upper"] == 1.15
    assert summary["test_ev_base"] == 12.5
    assert summary["test_equal_opportunity_diff"] == 0.04
    assert summary["test_demographic_parity_diff"] == 0.06


def test_flatten_metrics_summary_comparative_includes_deltas() -> None:
    """Comparative regime threads gate.py's own paired-bootstrap deltas through verbatim."""
    decision_payload = _decision_payload_fixture("comparative")
    summary = flatten_metrics_summary(
        _metrics_payload_fixture(), decision_payload, "hash-b"
    )
    assert summary["regime"] == "comparative"
    for criterion in ("pr_auc", "recall", "brier"):
        entry = decision_payload["criteria"][criterion]
        assert summary[f"{criterion}_delta_obs"] == entry["delta_obs"]
        assert summary[f"{criterion}_delta_ci_lower"] == entry["delta_ci"][0]
        assert summary[f"{criterion}_delta_ci_upper"] == entry["delta_ci"][1]


# ---------------------------------------------------------------------------
# MLflow orchestration: _log_evaluation_run / _tag_evaluated_model_version
#
# Real tmp-scoped MLflow experiment (conftest.py::mlflow_test_experiment),
# not a Mock client — same convention test_calibrate.py's run_calibration_step
# tests and test_mlflow.py's registry_cfg tests already use, since the
# (model, dataset) metric-attachment contract and the tag-based registry
# rollback rule can only be verified against a real store.
# ---------------------------------------------------------------------------


def _register_trivial_model(registered_model_name: str) -> tuple[str, str]:
    """Fit a trivial LogisticRegression, log + register it, return (version, model_id).

    Same shape as test_mlflow.py's _log_and_register — resolve_logged_model_id
    (called inside _log_evaluation_run) reads a real tag off a real registered
    version, a contract a mocked client can't verify.
    """
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    model = LogisticRegression().fit(X, [0, 1, 0, 1])
    with mlflow.start_run():
        model_info = mlflow.sklearn.log_model(
            sk_model=model, name="model", registered_model_name=registered_model_name
        )
    return str(model_info.registered_model_version), str(model_info.model_id)


def _build_evaluation_orchestration_inputs(
    tracking_uri: str,
    experiment_name: str,
    registered_model_name: str,
    version: str,
    y_proba_fixture: tuple[pd.Series, np.ndarray],
    policy_fixture: DictConfig,
    eval_cfg: DictConfig,
    tmp_path: Path,
) -> dict[str, Any]:
    """Build _log_evaluation_run/_tag_evaluated_model_version's inputs from the
    same pure functions run_evaluation_step itself calls, so their shapes are
    exactly production shapes rather than a hand-typed guess that could drift
    from what _build_scalar_metrics/_tag_evaluated_model_version actually read.
    """
    y, proba = y_proba_fixture
    scenarios = resolve_policy_scenarios(policy_fixture)
    thresholds = resolve_policy_thresholds_by_scenario(policy_fixture)
    policy_ctx = {
        "scenarios": scenarios,
        "thresholds": thresholds,
        "base_scenario": scenarios["base"],
        "base_threshold": thresholds["base"],
        "n_bootstrap": _N_BOOTSTRAP,
        "random_state": _RANDOM_STATE,
    }

    ranking_metrics = sealed_test_ranking_metrics(y, proba, _N_BOOTSTRAP, _RANDOM_STATE)
    classification_rows = sealed_test_classification_report(
        y, proba, thresholds, _N_BOOTSTRAP, _RANDOM_STATE
    )
    calibration_report = sealed_test_calibration_report(
        y, proba, eval_cfg, _N_BOOTSTRAP, _RANDOM_STATE
    )
    business_impact = sealed_test_business_impact(
        y, proba, scenarios, thresholds, _N_BOOTSTRAP, _RANDOM_STATE
    )
    core_metrics = {
        "ranking_metrics": ranking_metrics,
        "classification_rows": classification_rows,
        "fixed_recall_rows": [],
        "calibration_report": calibration_report,
        "decile_rows": [],
        "business_impact": business_impact,
    }
    # Only the two keys _build_scalar_metrics reads off `sliced` — the full
    # fairness-slice shape belongs to _compute_sliced_diagnostics, which this
    # test deliberately doesn't exercise (it reads real feature data on disk;
    # the orchestration seam under test here doesn't care about its content).
    sliced = {
        "test_equal_opportunity_diff": 0.04,
        "test_demographic_parity_diff": 0.02,
    }

    gate_inputs = build_gate_inputs(
        ranking_metrics, classification_rows, calibration_report, "base", None
    )
    decision = decide_promotion(gate_inputs, "cold_start", _BARS)

    metrics_payload: dict[str, Any] = {
        "model_version": version,
        "champion_version": None,
    }
    economics_payload: dict[str, Any] = {
        "note": "test economics payload",
        "capacity_budget_check": capacity_budget_check(
            business_impact["scenarios"], contact_capacity=500, campaign_budget=15_000
        ),
    }
    promotion_decision_payload = {
        **decision,
        "model_version": version,
        "metrics_content_hash": content_hash(metrics_payload),
    }
    payloads = {
        "metrics_payload": metrics_payload,
        "economics_payload": economics_payload,
        "promotion_decision_payload": promotion_decision_payload,
    }

    X_test = pd.DataFrame({"feature_a": np.arange(len(y), dtype=float)})
    loaded = {
        "X_test": X_test,
        "y_test": y,
        "proba": proba,
        "customer_ids": pd.Series([f"cust-{i:04d}" for i in range(len(y))]),
    }

    sensitivity_block = {
        "costs_cfg": OmegaConf.create(
            {
                "gross_margin": 0.6,
                "contact_capacity": 500,
                "campaign_budget": 15_000,
            }
        )
    }

    figure_keys = (
        "pr_curve_path",
        "roc_curve_path",
        "classification_report_path",
        "reliability_path",
        "ev_by_budget_path",
        "breakeven_heatmap_path",
        "sensitivity_tornado_path",
        "gains_lift_path",
    )
    figures: dict[str, Path] = {}
    for key in figure_keys:
        path = tmp_path / f"{key}.png"
        path.write_bytes(b"fake-figure-bytes")
        figures[key] = path

    cfg = OmegaConf.create(
        {
            "mlflow": {
                "tracking_uri": tracking_uri,
                "experiment_name": experiment_name,
                "registered_model_name": registered_model_name,
            },
            "paths": {"costs_config": "configs/costs.yaml"},
        }
    )

    return {
        "loaded": loaded,
        "core_metrics": core_metrics,
        "sliced": sliced,
        "sensitivity_block": sensitivity_block,
        "payloads": payloads,
        "figures": figures,
        "policy_ctx": policy_ctx,
        "cfg": cfg,
    }


def test_log_evaluation_run_wires_model_id_dataset_and_tags_correctly(
    mlflow_test_experiment: Callable[[str], str],
    y_proba_fixture: tuple[pd.Series, np.ndarray],
    policy_fixture: DictConfig,
    eval_cfg: DictConfig,
    tmp_path: Path,
) -> None:
    """_log_evaluation_run must attach metrics to the resolved model_id/dataset
    pair (not just the run), stamp eval_run_id into the persisted decision
    before logging it, and tag gate_regime/gate_result from the actual
    decision — the wiring a mocked client can't verify, since MLflow raises
    on none of these getting mixed up; only the end state tells you.
    """
    experiment_name = "test_log_evaluation_run"
    tracking_uri = mlflow_test_experiment(experiment_name)
    registered_model_name = "test-eval-orchestration"
    version, model_id = _register_trivial_model(registered_model_name)
    mlflow.tracking.MlflowClient().set_model_version_tag(
        registered_model_name, version, "logged_model_id", model_id
    )

    inputs = _build_evaluation_orchestration_inputs(
        tracking_uri,
        experiment_name,
        registered_model_name,
        version,
        y_proba_fixture,
        policy_fixture,
        eval_cfg,
        tmp_path,
    )

    eval_run_id, test_predictions = evaluate._log_evaluation_run(
        version,
        inputs["loaded"],
        inputs["core_metrics"],
        inputs["sliced"],
        inputs["sensitivity_block"],
        inputs["payloads"],
        inputs["figures"],
        inputs["policy_ctx"],
        inputs["cfg"],
    )
    decision = inputs["payloads"]["promotion_decision_payload"]

    # eval_run_id was stamped into the decision before it was persisted, not
    # left at whatever value the caller happened to pass in.
    persisted_decision = mlflow.artifacts.load_dict(
        f"runs:/{eval_run_id}/promotion_decision.json"
    )
    assert persisted_decision["eval_run_id"] == eval_run_id

    # gate_regime/gate_result tags reflect the actual decision, not a
    # default/stale value.
    run = mlflow.tracking.MlflowClient().get_run(eval_run_id)
    assert run.data.tags["gate_regime"] == decision["regime"]
    assert run.data.tags["gate_result"] == decision["gate"]

    # Metrics attached to the resolved model_id, on the "sealed_test" dataset
    # — the exact (model, dataset) pairing CLAUDE.md requires so metrics are
    # attributable to the LoggedModel rather than only to the run that
    # happened to compute them.
    logged_model = mlflow.get_logged_model(model_id)
    metric_by_name = {
        m.key: m for m in (logged_model.metrics or []) if m.run_id == eval_run_id
    }
    assert metric_by_name["test_pr_auc"].value == pytest.approx(
        inputs["core_metrics"]["ranking_metrics"]["pr_auc"]
    )
    assert all(m.dataset_name == "sealed_test" for m in metric_by_name.values())

    # test_predictions round-trips the exact proba/customer_ids handed in.
    assert list(test_predictions["customerid"]) == list(
        inputs["loaded"]["customer_ids"]
    )
    np.testing.assert_allclose(
        test_predictions["p_hat"].to_numpy(), inputs["loaded"]["proba"]
    )


def test_log_evaluation_run_survives_a_downstream_tagging_failure(
    mlflow_test_experiment: Callable[[str], str],
    y_proba_fixture: tuple[pd.Series, np.ndarray],
    policy_fixture: DictConfig,
    eval_cfg: DictConfig,
    tmp_path: Path,
) -> None:
    """_log_evaluation_run's own artifacts are a complete, valid record
    independent of whatever register.py does with them afterward (B1: model-
    version tagging moved out of evaluate.py entirely, into register.py's
    _tag_gate_criteria/_resolve_and_tag_eval_run_id — see test_register.py
    for that failure mode). Recovering from a downstream tagging failure
    only requires re-running register.py, not re-evaluating.
    """
    experiment_name = "test_log_evaluation_run_survives_downstream_failure"
    tracking_uri = mlflow_test_experiment(experiment_name)
    registered_model_name = "test-eval-orchestration-failure"
    version, model_id = _register_trivial_model(registered_model_name)
    mlflow.tracking.MlflowClient().set_model_version_tag(
        registered_model_name, version, "logged_model_id", model_id
    )

    inputs = _build_evaluation_orchestration_inputs(
        tracking_uri,
        experiment_name,
        registered_model_name,
        version,
        y_proba_fixture,
        policy_fixture,
        eval_cfg,
        tmp_path,
    )
    eval_run_id, _test_predictions = evaluate._log_evaluation_run(
        version,
        inputs["loaded"],
        inputs["core_metrics"],
        inputs["sliced"],
        inputs["sensitivity_block"],
        inputs["payloads"],
        inputs["figures"],
        inputs["policy_ctx"],
        inputs["cfg"],
    )

    persisted_decision = mlflow.artifacts.load_dict(
        f"runs:/{eval_run_id}/promotion_decision.json"
    )
    assert persisted_decision["eval_run_id"] == eval_run_id
    version_info = mlflow.tracking.MlflowClient().get_model_version(
        registered_model_name, version
    )
    assert "eval_run_id" not in version_info.tags
    assert "test_pr_auc" not in version_info.tags

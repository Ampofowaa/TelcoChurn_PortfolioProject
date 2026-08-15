"""Unit tests for telco_churn.models.evaluate — sealed-test evaluation (Phase 7)."""

from __future__ import annotations

import math
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
from telco_churn.models.diagnostics import build_segment_lookup
from telco_churn.models.economics import capacity_budget_check
from telco_churn.models.evaluate import (
    build_gate_inputs,
    comparative_deltas,
    flatten_metrics_summary,
    resolve_incumbent_summary,
    sealed_test_business_impact,
    sealed_test_calibration_report,
    sealed_test_classification_report,
    sealed_test_decile_lift,
    sealed_test_fixed_recall_profile,
    sealed_test_promotion_decision,
    sealed_test_ranking_metrics,
    sealed_test_sensitivity_analysis,
    sliced_business_impact,
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


@pytest.fixture
def segment_fixture() -> tuple[pd.Series, np.ndarray, dict[str, pd.Series]]:
    """400 rows across two contract_type groups with a planted PR-AUC/FNR gap
    between them — enough support in each group for a bootstrap CI."""
    rng = np.random.default_rng(11)
    n = 200
    y_a = (rng.random(n) < 0.30).astype(int)
    proba_a = np.clip(y_a * 0.6 + rng.normal(0.2, 0.1, size=n), 0.001, 0.999)
    y_b = (rng.random(n) < 0.30).astype(int)
    proba_b = np.clip(rng.random(n), 0.001, 0.999)  # no signal in group b

    y = pd.Series(np.concatenate([y_a, y_b]), name="churn")
    proba = np.concatenate([proba_a, proba_b])
    df = pd.DataFrame(
        {
            "tenure": rng.integers(0, 72, size=2 * n),
            "contract_type": ["month-to-month"] * n + ["two-year"] * n,
            "internetservice": ["fiber optic"] * (2 * n),
            "gender": (["male"] * n) + (["female"] * n),
            "seniorcitizen": [0] * (2 * n),
            "has_partner": [1, 0] * n,
            "dependents": [0, 1] * n,
        }
    )
    return y, proba, build_segment_lookup(df)


# ---------------------------------------------------------------------------
# sealed_test_ranking_metrics
# ---------------------------------------------------------------------------


def test_ranking_metrics_returns_expected_keys(
    y_proba_fixture: tuple[pd.Series, np.ndarray],
) -> None:
    """Every expected scalar key is present."""
    y, proba = y_proba_fixture
    result = sealed_test_ranking_metrics(y, proba, _N_BOOTSTRAP, _RANDOM_STATE)
    assert {
        "pr_auc",
        "pr_auc_ci_lower",
        "pr_auc_ci_upper",
        "roc_auc",
        "roc_auc_ci_lower",
        "roc_auc_ci_upper",
        "dummy_pr_auc_floor",
    } <= set(result)


def test_ranking_metrics_pr_auc_beats_dummy_floor_on_separable_fixture(
    y_proba_fixture: tuple[pd.Series, np.ndarray],
) -> None:
    """A model with real signal clears the DummyClassifier(strategy='prior') floor."""
    y, proba = y_proba_fixture
    result = sealed_test_ranking_metrics(y, proba, _N_BOOTSTRAP, _RANDOM_STATE)
    assert result["pr_auc"] > result["dummy_pr_auc_floor"]


def test_ranking_metrics_dummy_floor_matches_prevalence(
    y_proba_fixture: tuple[pd.Series, np.ndarray],
) -> None:
    """The dummy floor is approximately the test set's own churn prevalence."""
    y, proba = y_proba_fixture
    result = sealed_test_ranking_metrics(y, proba, _N_BOOTSTRAP, _RANDOM_STATE)
    assert result["dummy_pr_auc_floor"] == pytest.approx(float(y.mean()), abs=1e-9)


def test_ranking_metrics_ci_bounds_contain_point_estimate(
    y_proba_fixture: tuple[pd.Series, np.ndarray],
) -> None:
    """Both CIs bracket their own point estimate."""
    y, proba = y_proba_fixture
    result = sealed_test_ranking_metrics(y, proba, _N_BOOTSTRAP, _RANDOM_STATE)
    assert result["pr_auc_ci_lower"] <= result["pr_auc"] <= result["pr_auc_ci_upper"]
    assert result["roc_auc_ci_lower"] <= result["roc_auc"] <= result["roc_auc_ci_upper"]


# ---------------------------------------------------------------------------
# sealed_test_classification_report
# ---------------------------------------------------------------------------


def test_classification_report_one_row_per_scenario_with_ci_keys(
    y_proba_fixture: tuple[pd.Series, np.ndarray],
) -> None:
    """One row per scenario, each carrying the six new CI keys alongside the
    point-estimate keys plots.classification_summary_points already returns."""
    y, proba = y_proba_fixture
    thresholds = {"conservative": 0.2, "base": 0.3, "optimistic": 0.45}
    rows = sealed_test_classification_report(
        y, proba, thresholds, _N_BOOTSTRAP, _RANDOM_STATE
    )
    assert len(rows) == 3
    assert {row["scenario"] for row in rows} == set(thresholds)
    ci_keys = {
        "precision_ci_lower",
        "precision_ci_upper",
        "recall_ci_lower",
        "recall_ci_upper",
        "f1_ci_lower",
        "f1_ci_upper",
    }
    for row in rows:
        assert ci_keys <= set(row)


def test_classification_report_ci_bounds_contain_point_estimate(
    y_proba_fixture: tuple[pd.Series, np.ndarray],
) -> None:
    """Each metric's CI brackets its own point estimate."""
    y, proba = y_proba_fixture
    rows = sealed_test_classification_report(
        y, proba, {"base": 0.3}, _N_BOOTSTRAP, _RANDOM_STATE
    )
    row = rows[0]
    assert row["precision_ci_lower"] <= row["precision"] <= row["precision_ci_upper"]
    assert row["recall_ci_lower"] <= row["recall"] <= row["recall_ci_upper"]
    assert row["f1_ci_lower"] <= row["f1"] <= row["f1_ci_upper"]


def test_classification_report_empty_thresholds_returns_empty_list(
    y_proba_fixture: tuple[pd.Series, np.ndarray],
) -> None:
    """No scenarios requested returns an empty list, not an error."""
    y, proba = y_proba_fixture
    assert (
        sealed_test_classification_report(y, proba, {}, _N_BOOTSTRAP, _RANDOM_STATE)
        == []
    )


# ---------------------------------------------------------------------------
# sealed_test_fixed_recall_profile
# ---------------------------------------------------------------------------


def test_fixed_recall_profile_one_row_per_target(
    y_proba_fixture: tuple[pd.Series, np.ndarray],
) -> None:
    """One row per requested recall target, in the same order."""
    y, proba = y_proba_fixture
    targets = [0.70, 0.80, 0.90]
    rows = sealed_test_fixed_recall_profile(y, proba, targets)
    assert [row["recall_target"] for row in rows] == targets


def test_fixed_recall_profile_achieved_recall_meets_target(
    y_proba_fixture: tuple[pd.Series, np.ndarray],
) -> None:
    """Where a qualifying point exists, the achieved recall clears its target."""
    y, proba = y_proba_fixture
    rows = sealed_test_fixed_recall_profile(y, proba, [0.70])
    row = rows[0]
    if not math.isnan(row["recall_achieved"]):
        assert row["recall_achieved"] >= row["recall_target"]


# ---------------------------------------------------------------------------
# sealed_test_calibration_report
# ---------------------------------------------------------------------------


def test_calibration_report_returns_expected_keys(
    y_proba_fixture: tuple[pd.Series, np.ndarray], eval_cfg: DictConfig
) -> None:
    """Every expected block is present."""
    y, proba = y_proba_fixture
    result = sealed_test_calibration_report(
        y, proba, eval_cfg, _N_BOOTSTRAP, _RANDOM_STATE
    )
    assert {
        "brier",
        "dummy_prior_brier",
        "bss",
        "ece",
        "murphy_decomposition",
        "calibration_slope",
        "reliability_bins",
    } <= set(result)


def test_calibration_report_murphy_reconstructs_brier_to_tolerance(
    y_proba_fixture: tuple[pd.Series, np.ndarray], eval_cfg: DictConfig
) -> None:
    """Murphy's decomposition's reconstructed Brier is close to the directly-computed one."""
    y, proba = y_proba_fixture
    result = sealed_test_calibration_report(
        y, proba, eval_cfg, _N_BOOTSTRAP, _RANDOM_STATE
    )
    assert result["murphy_decomposition"]["brier_reconstructed"] == pytest.approx(
        result["brier"], abs=0.05
    )


def test_calibration_report_bss_positive_when_candidate_beats_dummy(
    y_proba_fixture: tuple[pd.Series, np.ndarray], eval_cfg: DictConfig
) -> None:
    """A candidate with real skill scores a positive BSS against the dummy floor."""
    y, proba = y_proba_fixture
    result = sealed_test_calibration_report(
        y, proba, eval_cfg, _N_BOOTSTRAP, _RANDOM_STATE
    )
    assert result["brier"] < result["dummy_prior_brier"]
    assert result["bss"] > 0.0


def test_calibration_report_calibration_slope_has_ci(
    y_proba_fixture: tuple[pd.Series, np.ndarray], eval_cfg: DictConfig
) -> None:
    """The calibration-slope block carries its bootstrap CI, the one gate.py reads."""
    y, proba = y_proba_fixture
    result = sealed_test_calibration_report(
        y, proba, eval_cfg, _N_BOOTSTRAP, _RANDOM_STATE
    )
    slope = result["calibration_slope"]
    assert {"slope", "slope_ci_lower", "slope_ci_upper"} <= set(slope)


# ---------------------------------------------------------------------------
# sealed_test_decile_lift
# ---------------------------------------------------------------------------


def test_decile_lift_returns_ten_deciles(
    y_proba_fixture: tuple[pd.Series, np.ndarray],
) -> None:
    """Ten decile rows, matching plots.decile_lift_table's contract."""
    y, proba = y_proba_fixture
    rows = sealed_test_decile_lift(y, proba)
    assert len(rows) == 10


def test_decile_lift_top_decile_has_highest_mean_predicted(
    y_proba_fixture: tuple[pd.Series, np.ndarray],
) -> None:
    """Decile 1 is the highest-scored group."""
    y, proba = y_proba_fixture
    rows = sealed_test_decile_lift(y, proba)
    means = [row["mean_predicted"] for row in rows]
    assert means == sorted(means, reverse=True)


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
# sealed_test_business_impact
# ---------------------------------------------------------------------------


def test_business_impact_one_entry_per_scenario(
    y_proba_fixture: tuple[pd.Series, np.ndarray],
    scenarios_fixture: dict[str, CostScenario],
    thresholds_fixture: dict[str, float],
) -> None:
    """One scenarios entry per named scenario, each carrying the full field set."""
    y, proba = y_proba_fixture
    result = sealed_test_business_impact(
        y, proba, scenarios_fixture, thresholds_fixture, _N_BOOTSTRAP, _RANDOM_STATE
    )
    assert set(result["scenarios"]) == {"conservative", "base", "optimistic"}
    expected_keys = {
        "threshold",
        "ev",
        "ev_ci_lower",
        "ev_ci_upper",
        "campaign_cost",
        "retained_revenue",
        "n_contacted",
        "contact_rate",
        "break_even_retention_rate",
        "ev_treat_all",
        "ev_treat_none",
    }
    for row in result["scenarios"].values():
        assert expected_keys <= set(row)


def test_business_impact_ev_treat_none_is_always_zero(
    y_proba_fixture: tuple[pd.Series, np.ndarray],
    scenarios_fixture: dict[str, CostScenario],
    thresholds_fixture: dict[str, float],
) -> None:
    """Contacting no one nets exactly zero, by construction, for every scenario."""
    y, proba = y_proba_fixture
    result = sealed_test_business_impact(
        y, proba, scenarios_fixture, thresholds_fixture, _N_BOOTSTRAP, _RANDOM_STATE
    )
    for row in result["scenarios"].values():
        assert row["ev_treat_none"] == 0.0


def test_business_impact_net_ev_equals_revenue_minus_cost(
    y_proba_fixture: tuple[pd.Series, np.ndarray],
    scenarios_fixture: dict[str, CostScenario],
    thresholds_fixture: dict[str, float],
) -> None:
    """Each scenario's point-estimate EV equals retained_revenue minus campaign_cost."""
    y, proba = y_proba_fixture
    result = sealed_test_business_impact(
        y, proba, scenarios_fixture, thresholds_fixture, _N_BOOTSTRAP, _RANDOM_STATE
    )
    for row in result["scenarios"].values():
        assert row["ev"] == pytest.approx(
            row["retained_revenue"] - row["campaign_cost"]
        )


def test_business_impact_ci_bounds_contain_point_estimate(
    y_proba_fixture: tuple[pd.Series, np.ndarray],
    scenarios_fixture: dict[str, CostScenario],
    thresholds_fixture: dict[str, float],
) -> None:
    """Each scenario's EV CI brackets its own point estimate."""
    y, proba = y_proba_fixture
    result = sealed_test_business_impact(
        y, proba, scenarios_fixture, thresholds_fixture, _N_BOOTSTRAP, _RANDOM_STATE
    )
    for row in result["scenarios"].values():
        assert row["ev_ci_lower"] <= row["ev"] <= row["ev_ci_upper"]


def test_business_impact_ev_bracket_matches_min_max_of_scenarios(
    y_proba_fixture: tuple[pd.Series, np.ndarray],
    scenarios_fixture: dict[str, CostScenario],
    thresholds_fixture: dict[str, float],
) -> None:
    """The bracket is exactly the min/max of the three scenarios' point EVs."""
    y, proba = y_proba_fixture
    result = sealed_test_business_impact(
        y, proba, scenarios_fixture, thresholds_fixture, _N_BOOTSTRAP, _RANDOM_STATE
    )
    ev_values = [row["ev"] for row in result["scenarios"].values()]
    assert result["ev_bracket_min"] == pytest.approx(min(ev_values))
    assert result["ev_bracket_max"] == pytest.approx(max(ev_values))


def test_business_impact_parameter_spread_flag_is_boolean(
    y_proba_fixture: tuple[pd.Series, np.ndarray],
    scenarios_fixture: dict[str, CostScenario],
    thresholds_fixture: dict[str, float],
) -> None:
    """parameter_spread_dominates_sampling is exactly ev_spread > widest_ci."""
    y, proba = y_proba_fixture
    result = sealed_test_business_impact(
        y, proba, scenarios_fixture, thresholds_fixture, _N_BOOTSTRAP, _RANDOM_STATE
    )
    expected = result["ev_spread"] > result["widest_within_scenario_ci_width"]
    assert result["parameter_spread_dominates_sampling"] == expected


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
# sealed_test_sensitivity_analysis
# ---------------------------------------------------------------------------


def test_sensitivity_analysis_returns_expected_blocks(
    y_proba_fixture: tuple[pd.Series, np.ndarray],
    scenarios_fixture: dict[str, CostScenario],
    thresholds_fixture: dict[str, float],
) -> None:
    """All four sensitivity blocks are present."""
    y, proba = y_proba_fixture
    base_scenario = scenarios_fixture["base"]
    base_threshold = thresholds_fixture["base"]
    result = sealed_test_sensitivity_analysis(
        y,
        proba,
        base_scenario,
        base_threshold,
        retention_rate_values=[0.15, 0.20, 0.30, 0.40, 0.45],
        cost_values=[33.88, 67.76, 101.64, 135.52],
        tornado_pct_perturbation=0.2,
    )
    assert {"oneway_retention_rate", "oneway_cost", "twoway", "tornado"} <= set(result)
    assert len(result["oneway_retention_rate"]) == 5
    assert len(result["oneway_cost"]) == 4
    assert len(result["twoway"]) == 5 * 4
    assert {row["param"] for row in result["tornado"]} == {
        "retention_rate",
        "cost",
        "ltv",
    }


def test_sensitivity_analysis_ev_increases_with_retention_rate(
    y_proba_fixture: tuple[pd.Series, np.ndarray],
    scenarios_fixture: dict[str, CostScenario],
    thresholds_fixture: dict[str, float],
) -> None:
    """Holding cost/LTV fixed, EV is non-decreasing as retention_rate rises."""
    y, proba = y_proba_fixture
    base_scenario = scenarios_fixture["base"]
    base_threshold = thresholds_fixture["base"]
    result = sealed_test_sensitivity_analysis(
        y,
        proba,
        base_scenario,
        base_threshold,
        retention_rate_values=[0.15, 0.30, 0.45],
        cost_values=[67.76],
        tornado_pct_perturbation=0.2,
    )
    evs = [row["ev"] for row in result["oneway_retention_rate"]]
    assert evs == sorted(evs)


# ---------------------------------------------------------------------------
# sliced_business_impact
# ---------------------------------------------------------------------------


def test_sliced_business_impact_one_row_per_axis_value(
    segment_fixture: tuple[pd.Series, np.ndarray, dict[str, pd.Series]],
    scenarios_fixture: dict[str, CostScenario],
    thresholds_fixture: dict[str, float],
) -> None:
    """One row per (axis, value) pair, carrying dollar totals and n_churners."""
    y, proba, segment_lookup = segment_fixture
    rows = sliced_business_impact(
        y,
        proba,
        segment_lookup,
        ("gender",),
        scenarios_fixture["base"],
        thresholds_fixture["base"],
    )
    assert len(rows) == 2
    assert {row["axis"] for row in rows} == {"gender"}
    for row in rows:
        assert {
            "n",
            "n_churners",
            "campaign_cost",
            "retained_revenue",
            "missed_revenue",
            "ev",
        } <= set(row)


def test_sliced_business_impact_missed_revenue_zero_when_no_false_negatives() -> None:
    """A segment where every churner is contacted (recall 1.0) has missed_revenue == 0."""
    y = pd.Series([1, 1, 0, 0])
    proba = np.array([0.9, 0.9, 0.1, 0.1])
    group = pd.Series(["a"] * 4)
    scenario = CostScenario(
        name="base", arpu=80.0, ltv=500.0, cost=20.0, retention_rate=0.3
    )
    rows = sliced_business_impact(y, proba, {"axis": group}, ("axis",), scenario, 0.5)
    assert rows[0]["missed_revenue"] == pytest.approx(0.0)


def test_sliced_business_impact_missed_revenue_scales_with_false_negatives() -> None:
    """missed_revenue equals false_negatives * retention_rate * ltv exactly."""
    y = pd.Series([1, 1, 0, 0])
    proba = np.array([0.9, 0.1, 0.1, 0.1])  # one of two churners missed
    group = pd.Series(["a"] * 4)
    scenario = CostScenario(
        name="base", arpu=80.0, ltv=500.0, cost=20.0, retention_rate=0.3
    )
    rows = sliced_business_impact(y, proba, {"axis": group}, ("axis",), scenario, 0.5)
    assert rows[0]["missed_revenue"] == pytest.approx(1 * 0.3 * 500.0)
    assert rows[0]["n_churners"] == 2


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
# comparative_deltas
# ---------------------------------------------------------------------------


_BASE_THRESHOLD = 0.3


def test_comparative_deltas_returns_expected_keys(
    y_proba_fixture: tuple[pd.Series, np.ndarray],
) -> None:
    """All nine delta fields are present."""
    y, candidate_proba = y_proba_fixture
    rng = np.random.default_rng(20)
    incumbent_proba = np.clip(
        candidate_proba + rng.normal(0, 0.05, len(y)), 0.001, 0.999
    )
    deltas = comparative_deltas(
        y,
        candidate_proba,
        incumbent_proba,
        _BASE_THRESHOLD,
        _N_BOOTSTRAP,
        _RANDOM_STATE,
    )
    assert {
        "pr_auc_delta_obs",
        "pr_auc_delta_ci_lower",
        "pr_auc_delta_ci_upper",
        "brier_delta_obs",
        "brier_delta_ci_lower",
        "brier_delta_ci_upper",
        "recall_delta_obs",
        "recall_delta_ci_lower",
        "recall_delta_ci_upper",
    } <= set(deltas)


def test_comparative_deltas_zero_when_models_identical(
    y_proba_fixture: tuple[pd.Series, np.ndarray],
) -> None:
    """Comparing a model against itself gives a zero observed delta on every metric."""
    y, proba = y_proba_fixture
    deltas = comparative_deltas(
        y, proba, proba, _BASE_THRESHOLD, _N_BOOTSTRAP, _RANDOM_STATE
    )
    assert deltas["pr_auc_delta_obs"] == pytest.approx(0.0)
    assert deltas["brier_delta_obs"] == pytest.approx(0.0)
    assert deltas["recall_delta_obs"] == pytest.approx(0.0)


def test_comparative_deltas_pr_auc_positive_when_candidate_better(
    y_proba_fixture: tuple[pd.Series, np.ndarray],
) -> None:
    """A candidate with real separation scores a positive PR-AUC delta against
    a no-signal incumbent."""
    y, _ = y_proba_fixture
    rng = np.random.default_rng(21)
    n = len(y)
    y_arr = y.to_numpy()
    candidate_proba = np.clip(y_arr * 0.6 + rng.normal(0.2, 0.1, n), 0.001, 0.999)
    incumbent_proba = np.clip(rng.random(n), 0.001, 0.999)
    deltas = comparative_deltas(
        y,
        candidate_proba,
        incumbent_proba,
        _BASE_THRESHOLD,
        _N_BOOTSTRAP,
        _RANDOM_STATE,
    )
    assert deltas["pr_auc_delta_obs"] > 0.0
    assert deltas["pr_auc_delta_ci_lower"] > 0.0


def test_comparative_deltas_recall_positive_when_candidate_better(
    y_proba_fixture: tuple[pd.Series, np.ndarray],
) -> None:
    """A candidate that ranks true churners above the threshold more often
    than a no-signal incumbent scores a positive recall delta at that
    threshold."""
    y, _ = y_proba_fixture
    rng = np.random.default_rng(21)
    n = len(y)
    y_arr = y.to_numpy()
    candidate_proba = np.clip(y_arr * 0.6 + rng.normal(0.2, 0.1, n), 0.001, 0.999)
    incumbent_proba = np.clip(rng.random(n), 0.001, 0.999)
    deltas = comparative_deltas(
        y,
        candidate_proba,
        incumbent_proba,
        _BASE_THRESHOLD,
        _N_BOOTSTRAP,
        _RANDOM_STATE,
    )
    assert deltas["recall_delta_obs"] > 0.0


# ---------------------------------------------------------------------------
# build_gate_inputs
# ---------------------------------------------------------------------------


def _gate_fixture_blocks() -> (
    tuple[dict[str, float], list[dict[str, object]], dict[str, object]]
):
    ranking_metrics = {"pr_auc": 0.65}
    classification_rows: list[dict[str, object]] = [
        {"scenario": "conservative", "recall": 0.80, "threshold": 0.20},
        {"scenario": "base", "recall": 0.70, "threshold": 0.30},
        {"scenario": "optimistic", "recall": 0.55, "threshold": 0.45},
    ]
    calibration_report: dict[str, object] = {
        "bss": 0.30,
        "calibration_slope": {
            "slope": 1.02,
            "slope_ci_lower": 0.90,
            "slope_ci_upper": 1.15,
        },
    }
    return ranking_metrics, classification_rows, calibration_report


def test_build_gate_inputs_cold_start_leaves_delta_fields_none() -> None:
    """With deltas=None, the resulting GateInputs carries no delta fields."""
    ranking_metrics, classification_rows, calibration_report = _gate_fixture_blocks()
    inputs = build_gate_inputs(
        ranking_metrics, classification_rows, calibration_report, "base", None
    )
    assert inputs.pr_auc == pytest.approx(0.65)
    assert inputs.recall == pytest.approx(0.70)
    assert inputs.bss == pytest.approx(0.30)
    assert inputs.calibration_slope == pytest.approx(1.02)
    assert inputs.pr_auc_delta_obs is None
    assert inputs.brier_delta_obs is None


def test_build_gate_inputs_reads_recall_at_named_scenario() -> None:
    """recall is picked from the row whose scenario matches base_scenario_name."""
    ranking_metrics, classification_rows, calibration_report = _gate_fixture_blocks()
    inputs = build_gate_inputs(
        ranking_metrics, classification_rows, calibration_report, "optimistic", None
    )
    assert inputs.recall == pytest.approx(0.55)


def test_build_gate_inputs_comparative_threads_deltas_through() -> None:
    """Deltas' keys map directly onto GateInputs' *_delta_* fields."""
    ranking_metrics, classification_rows, calibration_report = _gate_fixture_blocks()
    deltas = {
        "pr_auc_delta_obs": 0.02,
        "pr_auc_delta_ci_lower": 0.01,
        "pr_auc_delta_ci_upper": 0.03,
        "brier_delta_obs": -0.01,
        "brier_delta_ci_lower": -0.02,
        "brier_delta_ci_upper": 0.00,
    }
    inputs = build_gate_inputs(
        ranking_metrics, classification_rows, calibration_report, "base", deltas
    )
    assert inputs.pr_auc_delta_obs == pytest.approx(0.02)
    assert inputs.brier_delta_ci_upper == pytest.approx(0.00)


# ---------------------------------------------------------------------------
# sealed_test_promotion_decision
# ---------------------------------------------------------------------------


def test_promotion_decision_cold_start_when_incumbent_proba_is_none(
    y_proba_fixture: tuple[pd.Series, np.ndarray],
) -> None:
    """No incumbent probabilities -> the cold-start regime, regardless of gate outcome."""
    y, proba = y_proba_fixture
    ranking_metrics, classification_rows, calibration_report = _gate_fixture_blocks()
    result = sealed_test_promotion_decision(
        y,
        proba,
        None,
        ranking_metrics,
        classification_rows,
        calibration_report,
        "base",
        _BARS,
        _N_BOOTSTRAP,
        _RANDOM_STATE,
    )
    assert result["regime"] == "cold_start"


def test_promotion_decision_comparative_when_incumbent_proba_given(
    y_proba_fixture: tuple[pd.Series, np.ndarray],
) -> None:
    """Incumbent probabilities present -> the comparative regime."""
    y, proba = y_proba_fixture
    ranking_metrics, classification_rows, calibration_report = _gate_fixture_blocks()
    rng = np.random.default_rng(22)
    incumbent_proba = np.clip(rng.random(len(y)), 0.001, 0.999)
    result = sealed_test_promotion_decision(
        y,
        proba,
        incumbent_proba,
        ranking_metrics,
        classification_rows,
        calibration_report,
        "base",
        _BARS,
        _N_BOOTSTRAP,
        _RANDOM_STATE,
    )
    assert result["regime"] == "comparative"


def test_promotion_decision_cold_start_gate_passes_on_admissible_candidate(
    y_proba_fixture: tuple[pd.Series, np.ndarray],
) -> None:
    """A candidate clearing every bar passes the cold-start gate."""
    y, proba = y_proba_fixture
    ranking_metrics, classification_rows, calibration_report = _gate_fixture_blocks()
    result = sealed_test_promotion_decision(
        y,
        proba,
        None,
        ranking_metrics,
        classification_rows,
        calibration_report,
        "base",
        _BARS,
        _N_BOOTSTRAP,
        _RANDOM_STATE,
    )
    assert result["gate"] == "pass"
    assert "review" not in result


def test_promotion_decision_cold_start_gate_fails_below_pr_auc_bar(
    y_proba_fixture: tuple[pd.Series, np.ndarray],
) -> None:
    """A candidate below the PR-AUC bar fails the cold-start gate even though
    every guardrail passes."""
    y, proba = y_proba_fixture
    ranking_metrics = {"pr_auc": 0.40}  # below _BARS.pr_auc_bar = 0.60
    _rm, classification_rows, calibration_report = _gate_fixture_blocks()
    result = sealed_test_promotion_decision(
        y,
        proba,
        None,
        ranking_metrics,
        classification_rows,
        calibration_report,
        "base",
        _BARS,
        _N_BOOTSTRAP,
        _RANDOM_STATE,
    )
    assert result["gate"] == "fail"
    assert result["criteria"]["pr_auc"]["passed"] is False


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

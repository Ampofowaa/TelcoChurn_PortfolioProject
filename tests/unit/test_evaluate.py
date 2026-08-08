"""Unit tests for telco_churn.models.evaluate — sealed-test evaluation (Phase 7)."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mlflow.artifacts
import mlflow.sklearn
import mlflow.tracking
import numpy as np
import pandas as pd
import pytest
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import LogisticRegression

import telco_churn.models.evaluate as evaluate
from telco_churn.models.diagnostics import build_segment_lookup
from telco_churn.models.evaluate import (
    build_gate_inputs,
    check_threshold_provenance,
    check_threshold_screen_passed,
    comparative_deltas,
    content_hash,
    load_model_promotion_bars,
    resolve_champion_version,
    resolve_incumbent_summary,
    resolve_policy_scenarios,
    resolve_policy_thresholds_by_scenario,
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
from telco_churn.models.gate import GateBars, decide_promotion
from telco_churn.models.threshold import CostScenario

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
    """Same shape as configs/policy/threshold.yaml's `scenarios` block."""
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
# check_threshold_provenance
# ---------------------------------------------------------------------------


def test_check_threshold_provenance_matching_stamp_does_not_raise() -> None:
    """A validation payload whose stamp matches the model being evaluated passes silently."""
    payload = {"model_run_id": "abc123", "logged_model_id": "m-1"}
    check_threshold_provenance(payload, logged_model_id="m-1")


def test_check_threshold_provenance_mismatched_logged_model_id_raises() -> None:
    """A stamp naming a different logged_model_id raises — the threshold was
    derived against a different calibration map. model_run_id is a locator
    only and plays no part in the comparison."""
    payload = {"model_run_id": "old_run", "logged_model_id": "m-1"}
    with pytest.raises(ValueError, match="does not match the model being evaluated"):
        check_threshold_provenance(payload, logged_model_id="m-2")


def test_check_threshold_provenance_compares_as_strings() -> None:
    """An integer logged_model_id in the payload still matches a string
    logged_model_id argument — the comparison is coerced to strings, not
    type-sensitive."""
    payload = {"model_run_id": "abc123", "logged_model_id": 1}
    check_threshold_provenance(payload, logged_model_id="1")


def test_check_threshold_provenance_error_message_includes_both_stamps() -> None:
    """The raised error names both the stamped and the actual logged_model_id,
    so the mismatch is diagnosable from the message alone."""
    payload = {"model_run_id": "old_run", "logged_model_id": "m-1"}
    with pytest.raises(ValueError, match="does not match") as exc_info:
        check_threshold_provenance(payload, logged_model_id="m-2")
    message = str(exc_info.value)
    assert "m-1" in message
    assert "m-2" in message


# ---------------------------------------------------------------------------
# check_threshold_screen_passed
# ---------------------------------------------------------------------------


def test_check_threshold_screen_passed_true_does_not_raise() -> None:
    check_threshold_screen_passed({"screen_passed": True})


def test_check_threshold_screen_passed_false_raises() -> None:
    """No override flag — a failed dev-OOF calibration screen must always
    block downstream evaluation/error analysis."""
    with pytest.raises(RuntimeError, match="screen_passed is False"):
        check_threshold_screen_passed({"screen_passed": False})


def test_check_threshold_screen_passed_missing_key_raises_key_error() -> None:
    """Direct indexing, not .get(...): an older threshold_validation.json
    artifact predating this field is a genuine incompatibility, not an
    implicit pass."""
    with pytest.raises(KeyError):
        check_threshold_screen_passed({})


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
# resolve_champion_version
# ---------------------------------------------------------------------------


def test_resolve_champion_version_returns_none_when_model_never_registered(
    mlflow_test_experiment: Callable[[str], str],
) -> None:
    """A genuinely never-registered model raises RESOURCE_DOES_NOT_EXIST — the
    first of the two real cold-start shapes MLflow's SqlAlchemy-backed
    registry reports — and must resolve to None.
    """
    tracking_uri = mlflow_test_experiment("test_resolve_champion_version")
    cfg = OmegaConf.create(
        {
            "mlflow": {
                "tracking_uri": tracking_uri,
                "registered_model_name": "never-registered-model",
            }
        }
    )
    assert resolve_champion_version(cfg) is None


def test_resolve_champion_version_returns_none_when_alias_never_set(
    mlflow_test_experiment: Callable[[str], str],
) -> None:
    """A registered model with no champion alias ever set raises
    INVALID_PARAMETER_VALUE, not RESOURCE_DOES_NOT_EXIST — the second real
    cold-start shape — and must also resolve to None.
    """
    tracking_uri = mlflow_test_experiment("test_resolve_champion_version")
    registered_model_name = "registered-no-alias"
    mlflow.tracking.MlflowClient().create_registered_model(registered_model_name)
    cfg = OmegaConf.create(
        {
            "mlflow": {
                "tracking_uri": tracking_uri,
                "registered_model_name": registered_model_name,
            }
        }
    )
    assert resolve_champion_version(cfg) is None


def test_resolve_champion_version_reraises_non_not_found_mlflow_errors(
    monkeypatch: pytest.MonkeyPatch, mlflow_test_experiment: Callable[[str], str]
) -> None:
    """A transient/auth/server MLflow failure must propagate, not be read as
    'no champion' — silently swallowing it would flip evaluate.py from the
    comparative gate regime to cold-start against a perfectly healthy
    incumbent.
    """

    def _raise_transient(
        self: mlflow.tracking.MlflowClient, name: str, alias: str
    ) -> None:
        raise MlflowException("temporarily unavailable", error_code=INTERNAL_ERROR)

    monkeypatch.setattr(
        mlflow.tracking.MlflowClient, "get_model_version_by_alias", _raise_transient
    )
    tracking_uri = mlflow_test_experiment("test_resolve_champion_version")
    cfg = OmegaConf.create(
        {
            "mlflow": {
                "tracking_uri": tracking_uri,
                "registered_model_name": "telco-churn-pipeline",
            }
        }
    )
    with pytest.raises(MlflowException, match="temporarily unavailable"):
        resolve_champion_version(cfg)


# ---------------------------------------------------------------------------
# resolve_incumbent_summary
# ---------------------------------------------------------------------------


def _tag_incumbent_version(
    registered_model_name: str,
    version: str,
    eval_run_id: str,
    *,
    include_costs_hash: bool = True,
) -> None:
    """Mint the tag set _tag_evaluated_model_version + _log_evaluation_run
    would leave on a real champion: the four gate criteria and eval_run_id
    on the version, costs_config_hash on the run itself.
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
    with pytest.raises(RuntimeError, match="missing the costs_config_hash tag"):
        resolve_incumbent_summary(version, cfg)


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
    assert result["review"] == "pending"


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
    economics_payload: dict[str, Any] = {"note": "test economics payload"}
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
        "costs_cfg": OmegaConf.create({"gross_margin": 0.6, "contact_capacity": 500})
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


def test_tag_evaluated_model_version_failure_leaves_no_partial_tags(
    mlflow_test_experiment: Callable[[str], str],
    y_proba_fixture: tuple[pd.Series, np.ndarray],
    policy_fixture: DictConfig,
    eval_cfg: DictConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A registry failure while tagging the evaluated model version must not
    leave it half-tagged, and must not corrupt the evaluation run's own
    already-logged artifacts.

    Mirrors test_calibrate.py's
    test_run_calibration_step_parity_failure_leaves_tagged_pending_orphan:
    force the downstream step to fail and assert the resulting state is
    safe, not merely that nothing crashed. Here, the evaluation run
    (_log_evaluation_run) is a complete, valid record even though the
    version-to-run link failed to attach — recovering only requires
    re-running _tag_evaluated_model_version, not re-evaluating.
    """
    experiment_name = "test_tag_evaluated_model_version_failure"
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

    def _raise_transient(*args: object, **kwargs: object) -> None:
        raise MlflowException(
            "registry temporarily unavailable", error_code=INTERNAL_ERROR
        )

    monkeypatch.setattr(
        mlflow.tracking.MlflowClient, "set_model_version_tag", _raise_transient
    )

    with pytest.raises(MlflowException, match="registry temporarily unavailable"):
        evaluate._tag_evaluated_model_version(
            version, eval_run_id, registered_model_name, inputs["core_metrics"]
        )

    monkeypatch.undo()
    client = mlflow.tracking.MlflowClient()
    version_info = client.get_model_version(registered_model_name, version)
    assert "eval_run_id" not in version_info.tags
    assert "test_pr_auc" not in version_info.tags

    # The run _log_evaluation_run already produced is untouched by the
    # tagging failure that happened after it.
    persisted_decision = mlflow.artifacts.load_dict(
        f"runs:/{eval_run_id}/promotion_decision.json"
    )
    assert persisted_decision["eval_run_id"] == eval_run_id

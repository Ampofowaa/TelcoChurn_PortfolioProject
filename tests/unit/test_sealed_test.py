"""Unit tests for telco_churn.models.sealed_test — dataset-agnostic sealed-test
scoring and comparative-gate primitives (Phase 10a-ii — extracted out of
tests/unit/test_evaluate.py alongside the source module's own extraction
from evaluate.py).

load_test_features/load_test_customer_ids/load_test_segment_lookup/
_load_sealed_test_partition are not covered by dedicated unit tests here —
they were untested in evaluate.py before this extraction too (no
test_evaluate.py section existed for them), since they're thin wrappers over
telco_churn.data.split.partition()/sealed_test_ids() already covered by
test_split.py, exercised indirectly by evaluate.py's own subprocess
integration test.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf

from telco_churn.models.diagnostics import build_segment_lookup
from telco_churn.models.gate import GateBars
from telco_churn.models.policy_config import (
    CostScenario,
    resolve_policy_scenarios,
    resolve_policy_thresholds_by_scenario,
)
from telco_churn.models.sealed_test import (
    build_gate_inputs,
    comparative_deltas,
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

_BASE_THRESHOLD = 0.3


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
# comparative_deltas
# ---------------------------------------------------------------------------


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

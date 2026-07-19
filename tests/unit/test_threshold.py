"""Unit tests for telco_churn.models.threshold.

Pure-function tests (cost resolution, closed-form math, EV curve, agreement
diagnostics) use a small hand-built costs_cfg with exact expected values.
The leak-free proof is two tests: a static AST scan, and an inherited-
contamination canary using a real overfit classifier's in-sample vs OOF
probabilities. Orchestration tests (run_threshold_step) go through a real
tmp-scoped MLflow experiment, reusing calibrate.run_calibration_step to
produce a real registered model version — the exact chain this module
consumes in production — with load_costs_config monkeypatched to a small
test fixture so scenario numbers stay exact and independent of the real
configs/costs.yaml.
"""

from __future__ import annotations

import ast
import inspect
from collections.abc import Callable
from pathlib import Path

import mlflow
import mlflow.artifacts
import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

import telco_churn.models.calibrate as calibrate
import telco_churn.models.threshold as threshold
import telco_churn.models.train.log_model as log_model
from telco_churn.features.build import FEATURE_SCHEMA
from telco_churn.features.preprocessing import build_preprocessor
from telco_churn.models.threshold import CostScenario

_OUTER_FOLDS = 3
_INNER_FOLDS = 3

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def costs_cfg() -> OmegaConf:
    """Small hand-built costs config — round numbers, exact hand-computable t*."""
    return OmegaConf.create(
        {
            "gross_margin": 0.60,
            "horizon_months": 12,
            "discount_months": 3,
            "arpu_quantile": {"conservative": 0.25, "base": 0.50, "optimistic": 0.75},
            "scenarios": {
                "conservative": {
                    "outreach_cost": 5.0,
                    "discount_rate": 0.10,
                    "retention_rate": 0.20,
                },
                "base": {
                    "outreach_cost": 20.0,
                    "discount_rate": 0.20,
                    "retention_rate": 0.30,
                },
                "optimistic": {
                    "outreach_cost": 50.0,
                    "discount_rate": 0.30,
                    "retention_rate": 0.40,
                },
            },
            "retention_rate_sweep": [0.15, 0.20, 0.30, 0.40, 0.45],
            "argmax_ev_bootstrap_n_samples": 1000,
        }
    )


@pytest.fixture
def base_scenario() -> CostScenario:
    """A simple, hand-computable scenario: t* = 100 / (0.5 * 500) = 0.4."""
    return CostScenario(
        name="test", arpu=100.0, ltv=500.0, cost=100.0, retention_rate=0.5
    )


# ---------------------------------------------------------------------------
# Cost/ARPU resolution
# ---------------------------------------------------------------------------


def test_costs_config_hash_changes_iff_costs_yaml_changes(tmp_path: Path) -> None:
    """Same resolved content -> same hash; a resolved-value difference -> a
    different hash.

    Pins configs/policy/threshold.yaml's provenance without a model stamp:
    same hash means the model changed, a different hash means the cost
    assumptions did.
    """
    path_a = tmp_path / "costs_a.yaml"
    path_a.write_text("gross_margin: 0.60\n", encoding="utf-8")
    path_b = tmp_path / "costs_b.yaml"
    path_b.write_text("gross_margin: 0.60\n", encoding="utf-8")
    path_c = tmp_path / "costs_c.yaml"
    path_c.write_text("gross_margin: 0.65\n", encoding="utf-8")

    assert threshold.costs_config_hash(path_a) == threshold.costs_config_hash(path_b)
    assert threshold.costs_config_hash(path_a) != threshold.costs_config_hash(path_c)


def test_costs_config_hash_ignores_cosmetic_formatting(tmp_path: Path) -> None:
    """A comment, re-indentation, or a CRLF/LF line-ending swap must not change
    the hash — only a resolved-value change may, or checking the same
    costs.yaml out on a different OS becomes a false provenance mismatch
    against the hash already pinned in configs/policy/threshold.yaml.
    """
    path_plain = tmp_path / "costs_plain.yaml"
    path_plain.write_text("gross_margin: 0.60\nhorizon_months: 12\n", encoding="utf-8")
    path_commented = tmp_path / "costs_commented.yaml"
    path_commented.write_text(
        "# gross margin assumption\n"
        "gross_margin: 0.60\n"
        "\n"
        "horizon_months: 12  # months\n",
        encoding="utf-8",
    )
    path_crlf = tmp_path / "costs_crlf.yaml"
    path_crlf.write_bytes(b"gross_margin: 0.60\r\nhorizon_months: 12\r\n")

    plain_hash = threshold.costs_config_hash(path_plain)
    assert threshold.costs_config_hash(path_commented) == plain_hash
    assert threshold.costs_config_hash(path_crlf) == plain_hash


def test_arpu_by_scenario_uses_churners_only(costs_cfg: OmegaConf) -> None:
    """ARPU is the churner-population quantile — non-churner MonthlyCharges never enter it."""
    monthlycharges = pd.Series([10.0, 20.0, 30.0, 40.0, 1000.0])
    y_dev = pd.Series([1, 1, 1, 1, 0])  # the $1000 outlier is a non-churner

    arpu = threshold.arpu_by_scenario(monthlycharges, y_dev, costs_cfg)

    assert arpu["base"] == pytest.approx(np.quantile([10.0, 20.0, 30.0, 40.0], 0.50))
    assert arpu["base"] < 1000.0


def test_resolve_scenario_ltv_and_cost_formula(costs_cfg: OmegaConf) -> None:
    """LTV = ARPU * gross_margin * horizon_months; cost = outreach_cost + ARPU * discount_rate * discount_months."""
    scenario = threshold.resolve_scenario("base", arpu=100.0, costs_cfg=costs_cfg)

    assert scenario.ltv == pytest.approx(100.0 * 0.60 * 12)  # 720.0
    assert scenario.cost == pytest.approx(20.0 + 100.0 * 0.20 * 3)  # 80.0
    assert scenario.retention_rate == 0.30
    assert scenario.arpu == 100.0


def test_resolve_all_scenarios_returns_all_three(costs_cfg: OmegaConf) -> None:
    """Every scenario named in costs.yaml gets resolved — none silently dropped."""
    monthlycharges = pd.Series(np.linspace(10.0, 100.0, 20))
    y_dev = pd.Series([1] * 20)

    scenarios = threshold.resolve_all_scenarios(monthlycharges, y_dev, costs_cfg)

    assert set(scenarios) == {"conservative", "base", "optimistic"}
    assert all(isinstance(s, CostScenario) for s in scenarios.values())


# ---------------------------------------------------------------------------
# Threshold math — pure
# ---------------------------------------------------------------------------


def test_closed_form_threshold_hand_computed(costs_cfg: OmegaConf) -> None:
    """t* = cost / (retention_rate * LTV), hand-computed for all three scenarios."""
    scenarios = threshold.resolve_all_scenarios(
        pd.Series([100.0] * 10), pd.Series([1] * 10), costs_cfg
    )

    conservative_ltv = 100.0 * 0.60 * 12
    conservative_cost = 5.0 + 100.0 * 0.10 * 3
    assert threshold.closed_form_threshold(scenarios["conservative"]) == pytest.approx(
        conservative_cost / (0.20 * conservative_ltv)
    )

    base_ltv = 100.0 * 0.60 * 12
    base_cost = 20.0 + 100.0 * 0.20 * 3
    assert threshold.closed_form_threshold(scenarios["base"]) == pytest.approx(
        base_cost / (0.30 * base_ltv)
    )

    optimistic_ltv = 100.0 * 0.60 * 12
    optimistic_cost = 50.0 + 100.0 * 0.30 * 3
    assert threshold.closed_form_threshold(scenarios["optimistic"]) == pytest.approx(
        optimistic_cost / (0.40 * optimistic_ltv)
    )


def test_closed_form_threshold_zero_retention_rate_raises() -> None:
    """r = 0 is a costs.yaml typo, not a valid input — must fail loudly, not emit inf."""
    scenario = CostScenario(
        name="degenerate", arpu=100.0, ltv=500.0, cost=100.0, retention_rate=0.0
    )
    with pytest.raises(ValueError, match="retention_rate must be > 0"):
        threshold.closed_form_threshold(scenario)


def test_closed_form_threshold_negative_retention_rate_raises() -> None:
    """A negative r is equally invalid — same guard, different sign."""
    scenario = CostScenario(
        name="degenerate", arpu=100.0, ltv=500.0, cost=100.0, retention_rate=-0.1
    )
    with pytest.raises(ValueError, match="retention_rate must be > 0"):
        threshold.closed_form_threshold(scenario)


def test_closed_form_threshold_t_star_at_or_above_one_raises() -> None:
    """c >= r*LTV means t* >= 1 — 'never contact anyone', an equally plausible typo."""
    scenario = CostScenario(
        name="degenerate", arpu=100.0, ltv=500.0, cost=1000.0, retention_rate=0.3
    )
    with pytest.raises(ValueError, match="outside \\(0, 1\\)"):
        threshold.closed_form_threshold(scenario)


def test_closed_form_threshold_zero_cost_raises() -> None:
    """c = 0 gives t* = 0 exactly — also outside the open interval (0, 1), also a typo signal."""
    scenario = CostScenario(
        name="degenerate", arpu=100.0, ltv=500.0, cost=0.0, retention_rate=0.3
    )
    with pytest.raises(ValueError, match="outside \\(0, 1\\)"):
        threshold.closed_form_threshold(scenario)


def test_expected_value_curve_hand_computed(base_scenario: CostScenario) -> None:
    """ev(t) = [TP*(r*LTV-c) - FP*c] / n, hand-computed on a 4-row example.

    r*LTV - c = 0.5*500 - 100 = 150 (per-TP benefit); c = 100 (per-FP cost).
    Sorted by proba desc: [0.9, 0.7, 0.4, 0.2], y = [1, 0, 1, 0].
    Cumulative EV/4 at each cutoff: 150/4=37.5, (150-100)/4=12.5,
    (300-100)/4=50, (300-200)/4=25 — argmax at threshold 0.4.
    """
    proba = np.array([0.9, 0.7, 0.4, 0.2])
    y = np.array([1, 0, 1, 0])

    thresholds, ev = threshold.expected_value_curve(proba, y, base_scenario)

    np.testing.assert_allclose(thresholds, [1.0, 0.9, 0.7, 0.4, 0.2])
    np.testing.assert_allclose(ev, [0.0, 37.5, 12.5, 50.0, 25.0])
    assert threshold.empirical_argmax_threshold(
        proba, y, base_scenario
    ) == pytest.approx(0.4)


def test_expected_value_curve_ties_collapse_to_one_point(
    base_scenario: CostScenario,
) -> None:
    """A tied probability group can only be contacted all-or-nothing — one EV point per distinct proba."""
    proba = np.array([0.5, 0.5, 0.5, 0.1])
    y = np.array([1, 0, 1, 0])

    thresholds, ev = threshold.expected_value_curve(proba, y, base_scenario)

    # 2 distinct proba values + the prepended "contact no one" point = 3
    assert len(thresholds) == 3
    assert len(ev) == 3


def test_expected_value_curve_single_class_argmax_is_contact_no_one(
    base_scenario: CostScenario,
) -> None:
    """No churners at all: every contact is pure cost, so the best policy is to contact no one."""
    proba = np.array([0.9, 0.7, 0.4, 0.2])
    y = np.array([0, 0, 0, 0])

    assert threshold.empirical_argmax_threshold(
        proba, y, base_scenario
    ) == pytest.approx(1.0)


def test_expected_value_at_threshold_agrees_with_curve(
    base_scenario: CostScenario,
) -> None:
    """expected_value_at_threshold must agree with expected_value_curve at
    each of the curve's own threshold points — the invariant that keeps the
    scalar (used for dev_ev_at_t_star and by Phase 7's economics.py) from
    drifting apart from the curve it is a single point on.
    """
    proba = np.array([0.9, 0.7, 0.4, 0.2])
    y = np.array([1, 0, 1, 0])

    thresholds, ev = threshold.expected_value_curve(proba, y, base_scenario)

    for t, expected_ev in zip(thresholds, ev, strict=True):
        assert threshold.expected_value_at_threshold(
            proba, y, base_scenario, float(t)
        ) == pytest.approx(expected_ev)


def test_expected_value_at_threshold_hand_computed(
    base_scenario: CostScenario,
) -> None:
    """Same 4-row example as test_expected_value_curve_hand_computed, evaluated
    directly at t=0.4 rather than read off the curve: contacts rows with
    proba >= 0.4 (0.9, 0.7, 0.4) -> y = [1, 0, 1], 2 TP + 1 FP ->
    (2*150 - 1*100) / 4 = 50.0.
    """
    proba = np.array([0.9, 0.7, 0.4, 0.2])
    y = np.array([1, 0, 1, 0])

    assert threshold.expected_value_at_threshold(
        proba, y, base_scenario, 0.4
    ) == pytest.approx(50.0)


def test_implied_contact_rate(base_scenario: CostScenario) -> None:
    """Fraction of rows at or above t_star."""
    proba = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
    assert threshold.implied_contact_rate(proba, 0.5) == pytest.approx(3 / 5)
    assert threshold.implied_contact_rate(proba, 1.0) == pytest.approx(0.0)
    assert threshold.implied_contact_rate(proba, 0.0) == pytest.approx(1.0)


def test_r_sensitivity_sweep_matches_closed_form(base_scenario: CostScenario) -> None:
    """Each r in the sweep must independently satisfy t* = c / (r * LTV) — cost/LTV held fixed.

    r values here must keep t* in (0, 1) for base_scenario (cost=100, ltv=500):
    t* < 1 requires r > cost/ltv = 0.2.
    """
    sweep = threshold.r_sensitivity_sweep(base_scenario, [0.25, 0.50, 0.75])

    for r, t_star in sweep.items():
        assert t_star == pytest.approx(base_scenario.cost / (r * base_scenario.ltv))


# ---------------------------------------------------------------------------
# Agreement-check diagnostics (derive_threshold) — hand-crafted synthetic data
# ---------------------------------------------------------------------------


def test_derive_threshold_agrees_on_calibrated_synthetic_data(
    base_scenario: CostScenario,
) -> None:
    """t* falls inside the bootstrap CI of the empirical argmax when probabilities are honestly calibrated."""
    rng = np.random.default_rng(42)
    n = 20_000
    proba = rng.uniform(0.01, 0.99, n)
    y = (rng.uniform(0, 1, n) < proba).astype(int)

    result = threshold.derive_threshold(
        proba, y, base_scenario, n_bootstrap=1000, random_state=42
    )

    assert result["within_ci"] is True
    assert result["threshold"] == pytest.approx(0.4)


def test_derive_threshold_disagrees_on_miscalibration_near_t_star(
    base_scenario: CostScenario,
) -> None:
    """t* falls outside the CI when miscalibration is local to t* specifically.

    A distortion confined to the tails would leave the argmax near t* and this
    test would pass for the wrong reason — see PROJECT context. The distorted
    band here is +-0.05 around t*=0.4, where the true churn rate is forced far
    below what proba implies.
    """
    rng = np.random.default_rng(42)
    n = 20_000
    proba = rng.uniform(0.01, 0.99, n)
    y = (rng.uniform(0, 1, n) < proba).astype(int)

    t_star = threshold.closed_form_threshold(base_scenario)
    band = (proba >= t_star - 0.05) & (proba <= t_star + 0.05)
    rng2 = np.random.default_rng(1)
    y[band] = (rng2.uniform(0, 1, band.sum()) < 0.05).astype(int)

    result = threshold.derive_threshold(
        proba, y, base_scenario, n_bootstrap=1000, random_state=42
    )

    assert result["within_ci"] is False


# ---------------------------------------------------------------------------
# Leak-free proof
# ---------------------------------------------------------------------------


def test_threshold_module_is_leak_free_by_construction() -> None:
    """AST scan: no .fit( call, no sklearn import, no telco_churn.data.split import.

    A module that cannot fit an estimator cannot fit on the wrong rows — this
    is the discipline the touched-once invariant gets from evaluate.py, made
    executable here rather than trusted.
    """
    source = inspect.getsource(threshold)
    tree = ast.parse(source)

    fit_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "fit"
    ]
    assert fit_calls == [], "models/threshold.py must never call .fit("

    sklearn_imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.startswith("sklearn")
    ]
    assert (
        sklearn_imports == []
    ), "models/threshold.py must not import sklearn estimators"

    split_imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and "data.split" in node.module
    ]
    assert (
        split_imports == []
    ), "models/threshold.py must not import telco_churn.data.split"


def test_inherited_contamination_canary(base_scenario: CostScenario) -> None:
    """The agreement check is itself a leak detector, not merely a calibration check.

    A deliberately overfit RandomForest's in-sample predict_proba is
    overconfident — hollowed out near t* — and drags the empirical argmax
    away from it, so the agreement check fails on in-sample probabilities
    and passes on genuine OOF ones. This is the guard that would catch
    someone passing a fitted model's in-sample predictions in a future
    refactor, the single most plausible way this module's contract breaks.
    """
    X, y = make_classification(
        n_samples=1500,
        n_features=15,
        n_informative=6,
        class_sep=0.7,
        flip_y=0.05,
        random_state=42,
    )
    clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)
    clf.fit(X, y)
    in_sample_proba = clf.predict_proba(X)[:, 1]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba", n_jobs=1)[
        :, 1
    ]

    in_sample_result = threshold.derive_threshold(
        in_sample_proba, y, base_scenario, n_bootstrap=1000, random_state=42
    )
    oof_result = threshold.derive_threshold(
        oof_proba, y, base_scenario, n_bootstrap=1000, random_state=42
    )

    assert in_sample_result["within_ci"] is False
    assert oof_result["within_ci"] is True


# ---------------------------------------------------------------------------
# Orchestration — run_threshold_step
# ---------------------------------------------------------------------------


@pytest.fixture
def threshold_mlflow_uri(mlflow_test_experiment: Callable[[str], str]) -> str:
    """Point MLflow at the shared tmp-scoped experiment (conftest.py :: mlflow_test_experiment)."""
    return mlflow_test_experiment("test_run_threshold_step")


@pytest.fixture
def unfitted_pipeline() -> Pipeline:
    """A real ColumnTransformer paired with a fast linear classifier, for calibrate.py's setup."""
    preprocessor = build_preprocessor(
        binary=list(FEATURE_SCHEMA.binary),
        multi_cat=list(FEATURE_SCHEMA.multi_cat),
        numeric=list(FEATURE_SCHEMA.numeric),
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clf", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    )


@pytest.fixture
def full_cfg(threshold_mlflow_uri: str, tmp_path: Path) -> OmegaConf:
    """Cfg covering everything run_calibration_step + run_threshold_step need.

    paths.figures/paths.policy point at an absolute tmp_path — get_project_root()
    joined with an absolute path resolves to that absolute path, sandboxing
    every write away from the real reports/figures/ and configs/policy/.

    paths.costs_config points at a real (placeholder) file rather than the
    real configs/costs.yaml: load_costs_config() is monkeypatched per-test to
    return the small costs_cfg fixture, but costs_config_hash() reads actual
    bytes off disk and is not monkeypatched, so a real file must exist here.
    """
    costs_config_path = tmp_path / "costs.yaml"
    costs_config_path.write_text("gross_margin: 0.60\n", encoding="utf-8")
    return OmegaConf.create(
        {
            "random_seed": 42,
            "training_setup": {"class_weight": "balanced", "delta_threshold": 0.005},
            "training": {
                "fixed": {
                    "subsample_freq": 1,
                    "deterministic": True,
                    "force_row_wise": True,
                    "n_jobs": 1,
                    "verbose": -1,
                },
            },
            "tuning": {
                "cv_folds": 5,
                "es_validation_size": 0.2,
                "random_state": 42,
            },
            "calibration": {
                "method": "sigmoid",
                "outer_cv_folds": _OUTER_FOLDS,
                "inner_cv_folds": _INNER_FOLDS,
                "shuffle": True,
                "random_state": 42,
                "brier_bootstrap_n_samples": 200,
                "slope_bootstrap_n_samples": 50,
                "ece_n_bins": 5,
                "ece_strategy": "uniform",
                "run_id": None,
                "override_trial_count_gate": False,
            },
            "threshold": {"model_version": None, "random_state": 42},
            "mlflow": {
                "tracking_uri": threshold_mlflow_uri,
                "experiment_name": "test_run_threshold_step",
                "registered_model_name": "test-telco-churn-pipeline",
            },
            "paths": {
                "figures": str(tmp_path / "figures"),
                "policy": str(tmp_path / "policy"),
                "costs_config": str(costs_config_path),
            },
        }
    )


@pytest.fixture
def tuning_result(dev_split: tuple[pd.DataFrame, pd.Series]) -> dict:
    """A plausible Step 4 tuning output — small n_estimators for test speed."""
    X_dev, _ = dev_split
    return {
        "best_params": {
            "num_leaves": 8,
            "learning_rate": 0.1,
            "min_child_samples": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "max_depth": 5,
        },
        "best_n_estimators_median": 10,
        "best_cv_pr_auc_mean": 0.6,
        "committed_features": list(X_dev.columns),
        "tuning_summary": {
            "n_trials_requested": 50,
            "n_completed_trials": 16,
            "n_pruned_trials": 34,
            "n_failed_trials": 0,
            "min_completed_trials": 10,
            "trial_count_below_threshold": False,
            "selection_rule": "1se",
            "selected_trial_number": 9,
            "selected_cv_pr_auc": 0.6,
            "raw_best_trial_number": 36,
            "raw_best_cv_pr_auc": 0.6664,
            "se": 0.0139,
            "band_floor": 0.6525,
            "boundary_hits": {"num_leaves": False},
        },
    }


@pytest.fixture
def comparison_result() -> dict:
    """A plausible Step 2 output — the fields run_model_logging_step reads."""
    return {
        "delta_obs": 0.01,
        "delta_ci_lower": -0.01,
        "delta_ci_upper": 0.03,
        "decision": "lgbm",
        "decision_rule": "tie",
        "diagnostics": {"fixed_recall": [], "fairness": [], "robustness": []},
    }


@pytest.fixture
def registered_model_version(
    full_cfg: OmegaConf,
    dev_split: tuple[pd.DataFrame, pd.Series],
    comparison_result: dict,
    tuning_result: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> str:
    """Register a real calibrated model version via the actual production chain
    (log_model.run_model_logging_step -> calibrate.run_calibration_step), the
    exact setup run_threshold_step consumes.

    Also redirects load_dev_features() to the synthetic dev_split fixture, on
    both its calibrate.py definition and threshold.py's separate `from ...
    import load_dev_features` binding — unpatched, both run_calibration_step
    (called here) and run_threshold_step (called directly by every test using
    this fixture) fall through to the real, gitignored datasets/processed/
    directory instead of this fixture's in-memory frame. The patch stays
    active for the rest of the test: monkeypatch's teardown only runs after
    the whole test function completes, not when this fixture returns.

    load_dev_customer_ids() is sandboxed the same way, on both bindings too —
    unpatched, run_calibration_step's dev_oof_predictions.parquet build zips
    this fixture's small y_dev against the real, full-length customerid
    column (a pandas length-mismatch before any calibration logic runs), and
    run_threshold_step's own load_dev_customer_ids() call (used to align the
    persisted dev_oof_predictions.parquet back onto X_dev/y_dev's row order)
    would do the same against the fake X_dev/y_dev built here.
    """
    X_dev, y_dev = dev_split

    def _fake_load_dev_features(
        committed_features: list[str],
    ) -> tuple[pd.DataFrame, pd.Series]:
        return X_dev[committed_features], y_dev

    def _fake_load_dev_customer_ids() -> pd.Series:
        return pd.Series(
            [f"cust-{i:04d}" for i in range(len(y_dev))], name="customerid"
        )

    monkeypatch.setattr(calibrate, "load_dev_features", _fake_load_dev_features)
    monkeypatch.setattr(threshold, "load_dev_features", _fake_load_dev_features)
    monkeypatch.setattr(calibrate, "load_dev_customer_ids", _fake_load_dev_customer_ids)
    monkeypatch.setattr(threshold, "load_dev_customer_ids", _fake_load_dev_customer_ids)

    with mlflow.start_run(run_name="tuning_study") as run:
        tuning_result = {**tuning_result, "parent_run_id": run.info.run_id}
    log_result = log_model.run_model_logging_step(
        X_dev, y_dev, comparison_result, tuning_result, full_cfg
    )
    cal_result = calibrate.run_calibration_step(log_result["run_id"], full_cfg)
    return str(cal_result["model_version"])


def test_run_threshold_step_ships_all_three_scenarios(
    full_cfg: OmegaConf,
    registered_model_version: str,
    costs_cfg: OmegaConf,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """threshold_payload["scenarios"] carries every scenario's full diagnostic
    bundle, not just its threshold — conservative/optimistic are equally
    auditable, not just base.
    """
    monkeypatch.setattr(threshold, "load_costs_config", lambda path=None: costs_cfg)

    result = threshold.run_threshold_step(registered_model_version, full_cfg)
    payload = result["threshold_payload"]

    assert set(payload["scenarios"]) == {"conservative", "base", "optimistic"}
    for scenario_result in payload["scenarios"].values():
        assert set(scenario_result) == {
            "scenario",
            "threshold",
            "argmax_ev_threshold",
            "argmax_ev_bootstrap_ci",
            "within_ci",
            "implied_contact_rate",
            "dev_ev_at_t_star",
            "costs",
        }
    assert payload["scenario"] == "base"
    assert payload["threshold"] == payload["scenarios"]["base"]["threshold"]
    assert payload["model_version"] == registered_model_version


def test_run_threshold_step_logs_three_figures(
    full_cfg: OmegaConf,
    registered_model_version: str,
    costs_cfg: OmegaConf,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """threshold_sensitivity.png, threshold_by_scenario.png, and
    expected_value_by_scenario.png are all logged onto the run — the shipped
    threshold's audit trail, not just the number.
    """
    monkeypatch.setattr(threshold, "load_costs_config", lambda path=None: costs_cfg)

    result = threshold.run_threshold_step(registered_model_version, full_cfg)
    run_id = result["threshold_payload"]["model_run_id"]

    client = mlflow.tracking.MlflowClient()
    artifact_paths = {a.path for a in client.list_artifacts(run_id, "figures")}
    assert "figures/threshold_sensitivity.png" in artifact_paths
    assert "figures/threshold_by_scenario.png" in artifact_paths
    assert "figures/expected_value_by_scenario.png" in artifact_paths

    artifact_paths_root = {a.path for a in client.list_artifacts(run_id)}
    assert "threshold.json" in artifact_paths_root


def test_run_threshold_step_logs_validation_json_and_ev_curve(
    full_cfg: OmegaConf,
    registered_model_version: str,
    costs_cfg: OmegaConf,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """threshold_validation.json (the model-dependent half) and ev_curve.parquet
    (the EV curve's actual points, not just its rendered pixels) are both
    logged onto the model's run.
    """
    monkeypatch.setattr(threshold, "load_costs_config", lambda path=None: costs_cfg)

    result = threshold.run_threshold_step(registered_model_version, full_cfg)
    run_id = result["threshold_payload"]["model_run_id"]

    client = mlflow.tracking.MlflowClient()
    artifact_paths_root = {a.path for a in client.list_artifacts(run_id)}
    assert "threshold_validation.json" in artifact_paths_root
    assert "ev_curve.parquet" in artifact_paths_root

    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="ev_curve.parquet"
    )
    ev_curve = pd.read_parquet(local_path)
    assert set(ev_curve.columns) == {"scenario", "threshold", "ev"}
    assert set(ev_curve["scenario"]) == {"conservative", "base", "optimistic"}

    validation = result["validation_payload"]
    assert validation["model_run_id"] == run_id
    assert validation["model_version"] == registered_model_version
    assert set(validation["scenarios"]) == {"conservative", "base", "optimistic"}
    for scenario_result in validation["scenarios"].values():
        assert set(scenario_result) == {
            "scenario",
            "argmax_ev_threshold",
            "argmax_ev_bootstrap_ci",
            "within_ci",
            "implied_contact_rate",
            "dev_ev_at_t_star",
        }


def test_run_threshold_step_logs_scenario_metrics(
    full_cfg: OmegaConf,
    registered_model_version: str,
    costs_cfg: OmegaConf,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """t_star_{scenario}, implied_contact_rate_{scenario}, and
    dev_ev_at_t_star_{scenario} land in the metrics panel for every scenario —
    not only inside threshold.json — so they are plottable across cycles.
    """
    monkeypatch.setattr(threshold, "load_costs_config", lambda path=None: costs_cfg)

    result = threshold.run_threshold_step(registered_model_version, full_cfg)
    run_id = result["threshold_payload"]["model_run_id"]

    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)

    for name in ("conservative", "base", "optimistic"):
        assert f"t_star_{name}" in run.data.metrics
        assert f"implied_contact_rate_{name}" in run.data.metrics
        assert f"dev_ev_at_t_star_{name}" in run.data.metrics
        assert run.data.metrics[f"t_star_{name}"] == pytest.approx(
            result["threshold_payload"]["scenarios"][name]["threshold"]
        )


def test_run_threshold_step_writes_policy_file(
    full_cfg: OmegaConf,
    registered_model_version: str,
    costs_cfg: OmegaConf,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """configs/policy/threshold.yaml (sandboxed to tmp_path here) mirrors the
    policy-only half — no model stamp, since t* is a pure function of
    configs/costs.yaml and must not go stale across a rollback.
    """
    monkeypatch.setattr(threshold, "load_costs_config", lambda path=None: costs_cfg)

    result = threshold.run_threshold_step(registered_model_version, full_cfg)

    policy_path = Path(str(full_cfg.paths.policy)) / "threshold.yaml"
    assert policy_path.exists()
    written = OmegaConf.load(policy_path)
    assert written.threshold == pytest.approx(result["threshold_payload"]["threshold"])


def test_run_threshold_step_policy_file_has_no_model_dependent_fields(
    full_cfg: OmegaConf,
    registered_model_version: str,
    costs_cfg: OmegaConf,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The policy file must carry no model_run_id, no model_version, and no
    other model-dependent field — asserted as an absence, since the failure
    mode is a field creeping back in that a presence-only test would never
    notice. costs_config_hash must be present and change iff costs.yaml does.
    """
    monkeypatch.setattr(threshold, "load_costs_config", lambda path=None: costs_cfg)

    threshold.run_threshold_step(registered_model_version, full_cfg)

    policy_path = Path(str(full_cfg.paths.policy)) / "threshold.yaml"
    written = OmegaConf.load(policy_path)

    assert "model_run_id" not in written
    assert "model_version" not in written
    assert "calibration_method" not in written
    assert "argmax_ev_threshold" not in written
    assert "argmax_ev_bootstrap_ci" not in written
    assert "within_ci" not in written
    assert "implied_contact_rate" not in written

    assert written.costs_config_hash == threshold.costs_config_hash(
        Path(str(full_cfg.paths.costs_config))
    )
    for scenario_name in ("conservative", "base", "optimistic"):
        scenario_fields = set(written.scenarios[scenario_name])
        assert scenario_fields == {"scenario", "threshold", "costs"}


def test_run_threshold_step_resolves_by_explicit_version_not_alias(
    full_cfg: OmegaConf,
    registered_model_version: str,
    costs_cfg: OmegaConf,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit model_version resolves correctly even after challenger moves to
    a different version — the whole reason this module takes a version, not
    an alias. A re-calibration invalidates a previously-derived threshold, and
    an alias is a moving pointer that could later point at a different version.
    """
    monkeypatch.setattr(threshold, "load_costs_config", lambda path=None: costs_cfg)

    client = mlflow.tracking.MlflowClient()
    registered_name = str(full_cfg.mlflow.registered_model_name)
    expected_run_id = client.get_model_version(
        registered_name, registered_model_version
    ).run_id

    # Register a second, unrelated version and move challenger onto it —
    # simulating a later re-calibration cycle.
    with mlflow.start_run():
        mlflow.sklearn.log_model(
            sk_model=LogisticRegression().fit([[0], [1]], [0, 1]),
            name="model",
            registered_model_name=registered_name,
        )
    newest_version = max(
        int(v.version)
        for v in client.search_model_versions(f"name='{registered_name}'")
    )
    client.set_registered_model_alias(
        registered_name, "challenger", str(newest_version)
    )

    result = threshold.run_threshold_step(registered_model_version, full_cfg)

    assert result["threshold_payload"]["model_run_id"] == expected_run_id

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
import json
from collections.abc import Iterator
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
import telco_churn.models.register as register
import telco_churn.models.threshold as threshold
import telco_churn.models.train.log_model as log_model
from telco_churn.features.build import FEATURE_SCHEMA, TARGET_COL
from telco_churn.features.preprocessing import build_preprocessor
from telco_churn.models.policy_config import CostScenario
from telco_churn.models.train.common import _FEATURE_COLS

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
    """AST scan: no .fit( call, no sklearn import.

    A module that cannot fit an estimator cannot fit on the wrong rows — this
    is the discipline the touched-once invariant gets from evaluate.py, made
    executable here rather than trusted.

    This module does import telco_churn.data.split (for the dev-OOF screen's
    segment columns via load_dev_partition) — but only ever takes the dev
    half; test_threshold_never_touches_test_partition asserts that
    structurally, the same guard calibration_screen.py used to carry before
    this module absorbed it.
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


def test_threshold_never_touches_test_partition() -> None:
    """Structural guard: threshold.py must never import the test split — it
    runs before evaluate.py and never spends the seal, even though it now
    imports telco_churn.data.split.partition for the dev-OOF screen's segment
    columns (load_dev_partition takes only the dev half)."""
    source = inspect.getsource(threshold)
    assert "test_ids" not in source
    assert "load_test_features" not in source
    assert "load_test_partition" not in source


def test_threshold_dev_oof_screen_has_no_refit_or_reslope_machinery() -> None:
    """Structural guard: the dev-OOF screen folded into this module never
    recomputes the aggregate calibration slope (calibrate.calibration_slope is
    not called at all) and imports no CV-refitting machinery — its only path
    to a probability vector is calibrate.py's already-fitted dev-OOF artifact,
    and its only path to the aggregate slope is reading
    calibration_summary.json."""
    source = inspect.getsource(threshold)
    assert "cross_val_predict" not in source
    assert "CalibratedClassifierCV" not in source
    assert "calibration_slope(" not in source


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


@pytest.fixture(scope="module")
def threshold_mlflow_uri(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Point MLflow at a module-scoped tmp SQLite store with an explicit
    artifact_location — inlines conftest.py::mlflow_test_experiment's logic
    rather than requesting it, since that fixture depends on the
    function-scoped tmp_path and a module-scoped fixture can't depend on a
    narrower-scoped one. See registered_model_version's docstring for why
    this whole chain is module-scoped."""
    tmp_path = tmp_path_factory.mktemp("threshold_mlflow")
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    artifact_location = (tmp_path / "artifacts").as_uri()
    experiment_id = mlflow.create_experiment(
        "test_run_threshold_step", artifact_location=artifact_location
    )
    mlflow.set_experiment(experiment_id=experiment_id)
    return tracking_uri


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


@pytest.fixture(scope="module")
def model_promotion_config_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("model_promotion") / "model_promotion.yaml"
    OmegaConf.save(
        OmegaConf.create(
            {
                "pr_auc_bar": 0.60,
                "recall_bar": 0.65,
                "calibration_slope_band": [0.80, 1.25],
                "pr_auc_materiality_threshold": 0.005,
                "brier_non_inferiority_margin": 0.005,
                "recall_non_inferiority_margin": 0.01,
            }
        ),
        path,
    )
    return path


@pytest.fixture(scope="module")
def full_cfg(
    threshold_mlflow_uri: str,
    tmp_path_factory: pytest.TempPathFactory,
    model_promotion_config_path: Path,
) -> OmegaConf:
    """Cfg covering everything run_calibration_step + run_threshold_step need.

    paths.figures/paths.policy point at an absolute tmp_path — get_project_root()
    joined with an absolute path resolves to that absolute path, sandboxing
    every write away from the real reports/figures/ and configs/policy/.

    paths.costs_config points at a real (placeholder) file rather than the
    real configs/costs.yaml: load_costs_config() is monkeypatched per-test to
    return the small costs_cfg fixture, but costs_config_hash() reads actual
    bytes off disk and is not monkeypatched, so a real file must exist here.

    paths.reports/paths.model_promotion_config and threshold.n_bootstrap are
    needed by the dev-OOF screen folded into run_threshold_step: the screen
    writes reports/dev_oof_predictions.parquet + reports/dev_oof_diagnostics.json
    and checks the aggregate slope against configs/model_promotion.yaml's band.
    threshold.v3_top_k_features=2 (not the production 8) keeps the V3
    direction-sanity veto small enough to reason about against this file's
    synthetic feature set.

    Module-scoped, sharing one tmp_path across every test in this file that
    consumes full_cfg/registered_model_version — see registered_model_version's
    docstring for the full rationale and the one caveat it introduces.
    """
    tmp_path = tmp_path_factory.mktemp("full_cfg")
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
                "golden_n_rows": 5,
            },
            "threshold": {
                "model_version": None,
                "random_state": 42,
                "n_bootstrap": 200,
                "v3_top_k_features": 2,
                "v3_min_direction_magnitude": 0.3,
            },
            "register": {"golden_atol": 1.0e-9},
            "mlflow": {
                "tracking_uri": threshold_mlflow_uri,
                "experiment_name": "test_run_threshold_step",
                "registered_model_name": "test-telco-churn-pipeline",
            },
            "paths": {
                "figures": str(tmp_path / "figures"),
                "policy": str(tmp_path / "policy"),
                "reports": str(tmp_path / "reports"),
                "costs_config": str(costs_config_path),
                "model_promotion_config": str(model_promotion_config_path),
            },
        }
    )


@pytest.fixture(scope="module")
def _module_feature_df() -> pd.DataFrame:
    """Module-scoped mirror of conftest.py::feature_df — identical body.
    registered_model_version needs a module-scoped source frame and can't
    depend on the shared conftest fixture (function-scoped, would raise a
    pytest ScopeMismatch). Pure and deterministic (seeded rng), so a second
    copy is behaviourally identical to the original."""
    rng = np.random.default_rng(0)
    n = 120  # matches conftest.py::feature_df's _TRAIN_N
    return pd.DataFrame(
        {
            "customerid": [f"cust-{i:04d}" for i in range(n)],
            "gender": rng.choice(["Male", "Female"], size=n),
            "has_partner": rng.choice(["Yes", "No"], size=n),
            "dependents": rng.choice(["Yes", "No"], size=n),
            "phoneservice": rng.choice(["Yes", "No"], size=n),
            "paperlessbilling": rng.choice(["Yes", "No"], size=n),
            "seniorcitizen": rng.integers(0, 2, size=n).tolist(),
            "multiplelines": rng.choice(["Yes", "No", "No phone service"], size=n),
            "internetservice": rng.choice(["DSL", "Fiber optic", "No"], size=n),
            "onlinesecurity": rng.choice(["Yes", "No", "No internet service"], size=n),
            "onlinebackup": rng.choice(["Yes", "No", "No internet service"], size=n),
            "deviceprotection": rng.choice(
                ["Yes", "No", "No internet service"], size=n
            ),
            "techsupport": rng.choice(["Yes", "No", "No internet service"], size=n),
            "streamingtv": rng.choice(["Yes", "No", "No internet service"], size=n),
            "streamingmovies": rng.choice(["Yes", "No", "No internet service"], size=n),
            "contract_type": rng.choice(
                ["Month-to-month", "One year", "Two year"], size=n
            ),
            "paymentmethod": rng.choice(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                size=n,
            ),
            "tenure": rng.integers(0, 73, size=n).tolist(),
            "monthlycharges": rng.uniform(18.25, 118.75, size=n).tolist(),
            "totalcharges": rng.uniform(18.25, 8684.8, size=n).tolist(),
            "charge_per_service": rng.uniform(0.5, 50.0, size=n).tolist(),
            "churn": rng.integers(0, 2, size=n).tolist(),
        }
    )


@pytest.fixture(scope="module")
def _module_dev_split(
    _module_feature_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Module-scoped mirror of conftest.py::dev_split — see _module_feature_df."""
    return _module_feature_df[_FEATURE_COLS], _module_feature_df[TARGET_COL]


@pytest.fixture(scope="module")
def tuning_result(_module_dev_split: tuple[pd.DataFrame, pd.Series]) -> dict:
    """A plausible Step 4 tuning output — small n_estimators for test speed."""
    X_dev, _ = _module_dev_split
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


@pytest.fixture(scope="module")
def registered_model_version(
    full_cfg: OmegaConf,
    _module_dev_split: tuple[pd.DataFrame, pd.Series],
    _module_feature_df: pd.DataFrame,
    tuning_result: dict,
) -> Iterator[str]:
    """Register a real calibrated model version via the actual production chain
    (log_model.run_model_logging_step -> calibrate.run_calibration_step ->
    register.register_challenger, B1's mint step, decoupled from
    calibrate.py — called directly here, mirroring what register.py's own
    mint-mode CLI does), the exact setup run_threshold_step consumes.

    Also redirects load_dev_features() to the synthetic dev_split fixture, on
    both its calibrate.py definition and threshold.py's separate `from ...
    import load_dev_features` binding — unpatched, both run_calibration_step
    (called here) and run_threshold_step (called directly by every test using
    this fixture) fall through to the real, gitignored datasets/processed/
    directory instead of this fixture's in-memory frame. The patch stays
    active for the rest of the module: torn down (mp.undo()) only once every
    test in this file has run.

    load_dev_customer_ids() is sandboxed the same way, on both bindings too —
    unpatched, run_calibration_step's dev_oof_predictions.parquet build zips
    this fixture's small y_dev against the real, full-length customerid
    column (a pandas length-mismatch before any calibration logic runs), and
    run_threshold_step's own load_dev_customer_ids() call (used to align the
    persisted dev_oof_predictions.parquet back onto X_dev/y_dev's row order)
    would do the same against the fake X_dev/y_dev built here.

    threshold.load_dev_partition (the dev-OOF screen's segment-column source,
    folded in from calibration_screen.py) is sandboxed to feature_df directly
    — its customerid column already uses the same "cust-{i:04d}" ordering as
    _fake_load_dev_customer_ids below, so the screen's by-customerid join
    lines up rather than reindexing to all-NaN.

    threshold.load_dev_shap_summary is sandboxed to a fixed, EDA-consistent
    two-feature summary (matching full_cfg's threshold.v3_top_k_features=2) —
    _module_feature_df's `churn` column is uniform random noise
    (rng.integers(0, 2)), so the real fitted model's dev-SHAP directions on it
    are themselves noise with no reason to agree with
    explain.EXPECTED_EDA_DIRECTIONS; without this, the V3 pre-seal veto this
    module now binds on would non-deterministically fail every consumer of
    this fixture on a signal this fixture was never designed to carry. V3's
    own pass/fail mechanics are covered directly, against controlled inputs,
    in test_run_v3_direction_sanity_* below — not by this shared fixture.

    Module-scoped: every consuming test calls run_threshold_step itself (this
    fixture only builds the registered version it's called against), so the
    ~9-10s train->calibrate cost is paid once per module instead of once per
    test. Verified safe against all 9 current consumers — 8 only read the
    registered version and its run; the ninth
    (test_run_threshold_step_resolves_by_explicit_version_not_alias)
    registers an *additional* model version under the same registered_model_
    name and re-points the challenger alias, which permanently mutates
    shared state, but no other test in this file resolves by the challenger
    alias or assumes a specific version count, so it doesn't affect them.
    Caveat for future maintainers: a new test added to this file that relies
    on the registry being "clean" (single version, challenger unset) would
    silently break depending on where it's added relative to that test —
    this coupling didn't exist under function scoping.
    """
    mp = pytest.MonkeyPatch()
    X_dev, y_dev = _module_dev_split

    def _fake_load_dev_features(
        committed_features: list[str],
    ) -> tuple[pd.DataFrame, pd.Series]:
        return X_dev[committed_features], y_dev

    def _fake_load_dev_customer_ids() -> pd.Series:
        return pd.Series(
            [f"cust-{i:04d}" for i in range(len(y_dev))], name="customerid"
        )

    def _fake_load_dev_shap_summary(run_id: str, cfg: object) -> list[dict]:
        # >= v3_top_k_features + 1 rows: explain.check_top_k_elbow indexes
        # deltas[configured_k - 1], which needs len(values) >= configured_k + 1.
        return [
            {"feature": "numeric__tenure", "mean_abs_shap": 0.5, "direction": -0.9},
            {
                "feature": "numeric__monthlycharges",
                "mean_abs_shap": 0.3,
                "direction": 0.9,
            },
            {
                "feature": "numeric__charge_per_service",
                "mean_abs_shap": 0.1,
                "direction": 0.2,
            },
        ]

    mp.setattr(calibrate, "load_dev_features", _fake_load_dev_features)
    mp.setattr(threshold, "load_dev_features", _fake_load_dev_features)
    mp.setattr(calibrate, "load_dev_customer_ids", _fake_load_dev_customer_ids)
    mp.setattr(threshold, "load_dev_customer_ids", _fake_load_dev_customer_ids)
    mp.setattr(threshold, "load_dev_partition", lambda: _module_feature_df)
    mp.setattr(threshold, "load_dev_shap_summary", _fake_load_dev_shap_summary)

    with mlflow.start_run(run_name="tuning_study") as run:
        tuning_result = {**tuning_result, "parent_run_id": run.info.run_id}
    log_result = log_model.run_model_logging_step(X_dev, y_dev, tuning_result, full_cfg)
    cal_result = calibrate.run_calibration_step(log_result["run_id"], full_cfg)
    # B1's call-site decoupling: calibrate.py no longer registers anything
    # itself, so this fixture mints the challenger the same way register.py's
    # own mint-mode CLI would, in-process.
    model_version = register.register_challenger(
        full_cfg,
        str(cal_result["run_id"]),
        str(cal_result["model_uri"]),
        str(cal_result["logged_model_id"]),
    )
    try:
        yield model_version
    finally:
        mp.undo()


@pytest.fixture(scope="module")
def registered_model_run_id(full_cfg: OmegaConf, registered_model_version: str) -> str:
    """The tuning_study run_id registered_model_version was minted from.

    A cheap MLflow lookup on the already-built module-scoped fixture above —
    not a second train->calibrate cycle — the same pattern two tests in this
    file already used ad hoc before run_threshold_step took run_id as an
    explicit argument.
    """
    client = mlflow.tracking.MlflowClient()
    registered_name = str(full_cfg.mlflow.registered_model_name)
    return str(
        client.get_model_version(registered_name, registered_model_version).run_id
    )


def test_run_threshold_step_ships_all_three_scenarios(
    full_cfg: OmegaConf,
    registered_model_run_id: str,
    registered_model_version: str,
    costs_cfg: OmegaConf,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """threshold_payload["scenarios"] carries every scenario's full diagnostic
    bundle, not just its threshold — conservative/optimistic are equally
    auditable, not just base.
    """
    monkeypatch.setattr(threshold, "load_costs_config", lambda path=None: costs_cfg)

    result = threshold.run_threshold_step(
        registered_model_run_id, registered_model_version, full_cfg
    )
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
    assert payload["model_run_id"] == registered_model_run_id
    assert payload["logged_model_id"]


def test_run_threshold_step_logs_three_figures(
    full_cfg: OmegaConf,
    registered_model_run_id: str,
    registered_model_version: str,
    costs_cfg: OmegaConf,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """threshold_sensitivity.png, threshold_by_scenario.png, and
    expected_value_by_scenario.png are all logged onto the run — the shipped
    threshold's audit trail, not just the number.
    """
    monkeypatch.setattr(threshold, "load_costs_config", lambda path=None: costs_cfg)

    result = threshold.run_threshold_step(
        registered_model_run_id, registered_model_version, full_cfg
    )
    run_id = result["threshold_payload"]["model_run_id"]

    client = mlflow.tracking.MlflowClient()
    artifact_paths = {
        a.path for a in client.list_artifacts(run_id, "threshold/figures")
    }
    assert "threshold/figures/threshold_sensitivity.png" in artifact_paths
    assert "threshold/figures/threshold_by_scenario.png" in artifact_paths
    assert "threshold/figures/expected_value_by_scenario.png" in artifact_paths

    artifact_paths_threshold = {
        a.path for a in client.list_artifacts(run_id, "threshold")
    }
    assert "threshold/threshold.json" in artifact_paths_threshold


def test_run_threshold_step_logs_validation_json_and_ev_curve(
    full_cfg: OmegaConf,
    registered_model_run_id: str,
    registered_model_version: str,
    costs_cfg: OmegaConf,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """threshold_validation.json (the model-dependent half) and ev_curve.parquet
    (the EV curve's actual points, not just its rendered pixels) are both
    logged onto the model's run.
    """
    monkeypatch.setattr(threshold, "load_costs_config", lambda path=None: costs_cfg)

    result = threshold.run_threshold_step(
        registered_model_run_id, registered_model_version, full_cfg
    )
    run_id = result["threshold_payload"]["model_run_id"]

    client = mlflow.tracking.MlflowClient()
    artifact_paths_threshold = {
        a.path for a in client.list_artifacts(run_id, "threshold")
    }
    assert "threshold/threshold_validation.json" in artifact_paths_threshold
    assert "threshold/ev_curve.parquet" in artifact_paths_threshold

    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="threshold/ev_curve.parquet"
    )
    ev_curve = pd.read_parquet(local_path)
    assert set(ev_curve.columns) == {"scenario", "threshold", "ev"}
    assert set(ev_curve["scenario"]) == {"conservative", "base", "optimistic"}

    validation = result["validation_payload"]
    assert validation["model_run_id"] == run_id
    assert validation["logged_model_id"]
    assert isinstance(validation["failures"], list)
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
    registered_model_run_id: str,
    registered_model_version: str,
    costs_cfg: OmegaConf,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """t_star_{scenario}, implied_contact_rate_{scenario}, and
    dev_ev_at_t_star_{scenario} land in the metrics panel for every scenario —
    not only inside threshold.json — so they are plottable across cycles.
    """
    monkeypatch.setattr(threshold, "load_costs_config", lambda path=None: costs_cfg)

    result = threshold.run_threshold_step(
        registered_model_run_id, registered_model_version, full_cfg
    )
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
    registered_model_run_id: str,
    registered_model_version: str,
    costs_cfg: OmegaConf,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """configs/policy/threshold.yaml (sandboxed to tmp_path here) mirrors the
    policy-only half — no model stamp, since t* is a pure function of
    configs/costs.yaml and must not go stale across a rollback.
    """
    monkeypatch.setattr(threshold, "load_costs_config", lambda path=None: costs_cfg)

    result = threshold.run_threshold_step(
        registered_model_run_id, registered_model_version, full_cfg
    )

    policy_path = Path(str(full_cfg.paths.policy)) / "threshold.yaml"
    assert policy_path.exists()
    written = OmegaConf.load(policy_path)
    assert written.threshold == pytest.approx(result["threshold_payload"]["threshold"])


def test_run_threshold_step_policy_file_has_no_model_dependent_fields(
    full_cfg: OmegaConf,
    registered_model_run_id: str,
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

    threshold.run_threshold_step(
        registered_model_run_id, registered_model_version, full_cfg
    )

    policy_path = Path(str(full_cfg.paths.policy)) / "threshold.yaml"
    written = OmegaConf.load(policy_path)

    assert "model_run_id" not in written
    assert "model_version" not in written
    assert "logged_model_id" not in written
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
    registered_model_run_id: str,
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

    result = threshold.run_threshold_step(
        registered_model_run_id, registered_model_version, full_cfg
    )

    assert result["threshold_payload"]["model_run_id"] == expected_run_id


# ---------------------------------------------------------------------------
# Dev-OOF screen (folded in from calibration_screen.py) — slope check + V1/V2/V2b
# ---------------------------------------------------------------------------


def _assert_dev_oof_report_shapes(
    frame: pd.DataFrame,
    diagnostics: dict,
    dev_split: tuple[pd.DataFrame, pd.Series],
) -> None:
    assert set(frame.columns) == {"customerid", "y_true", "p_hat"}
    assert len(frame) == len(dev_split[1])
    assert frame["p_hat"].between(0, 1).all()
    assert set(diagnostics) == {
        "segment_pr_auc",
        "segment_collapse_flagged",
        "segment_decision_rates",
        "equal_opportunity_gap_by_axis",
        "demographic_parity_gap_by_axis",
        "equal_opportunity_gap_flagged",
        "demographic_parity_gap_flagged",
        "segment_calibration",
        "calibration_collapse_flagged",
        "direction_sanity_result",
        "direction_check_feature_names",
        "direction_checked_count",
        "direction_weak_signal_count",
        "direction_sanity_elbow_check",
    }


def test_run_threshold_step_screens_dev_oof_slope_and_writes_reports(
    full_cfg: OmegaConf,
    registered_model_run_id: str,
    registered_model_version: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    costs_cfg: OmegaConf,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dev-OOF screen folded into run_threshold_step reads calibrate.py's
    already-logged calibration_slope rather than recomputing it; writes
    reports/dev_oof_predictions.parquet (3 columns — no re-logged segment
    join) and reports/dev_oof_diagnostics.json (V1/V2/V2b); logs
    dev_oof_*-namespaced metrics attached to the dev model's model_id, onto
    the same run threshold.json/threshold_validation.json already log to.
    """
    monkeypatch.setattr(threshold, "load_costs_config", lambda path=None: costs_cfg)

    client = mlflow.tracking.MlflowClient()
    registered_name = str(full_cfg.mlflow.registered_model_name)
    run_id = client.get_model_version(registered_name, registered_model_version).run_id

    try:
        result = threshold.run_threshold_step(
            registered_model_run_id, registered_model_version, full_cfg
        )
    except RuntimeError:
        # The real sigmoid-calibrated fit on this synthetic fixture may or
        # may not clear the band — either outcome is a valid resting state
        # for this test; the artifact/logging contract (asserted below)
        # holds regardless of which branch fired.
        reports_dir = Path(str(full_cfg.paths.reports))
        frame = pd.read_parquet(reports_dir / "dev_oof_predictions.parquet")
        diagnostics = json.loads(
            (reports_dir / "dev_oof_diagnostics.json").read_text(encoding="utf-8")
        )
        _assert_dev_oof_report_shapes(frame, diagnostics, dev_split)
        run = client.get_run(run_id)
        assert run.data.tags["dev_oof_screen_result"] == "fail"
        return

    reports_dir = Path(str(full_cfg.paths.reports))
    frame = pd.read_parquet(reports_dir / "dev_oof_predictions.parquet")
    diagnostics = json.loads(
        (reports_dir / "dev_oof_diagnostics.json").read_text(encoding="utf-8")
    )
    _assert_dev_oof_report_shapes(frame, diagnostics, dev_split)
    # NaN-safe: equal_opportunity/demographic_parity_difference_by_axis can
    # legitimately return float("nan") for a thin axis, and NaN != NaN under
    # Python's own semantics — comparing the JSON-serialized form (where both
    # sides render the same NaN literal identically) instead of the parsed
    # dicts directly is what actually verifies no silent default=str
    # corruption crept into the round trip.
    assert json.dumps(diagnostics, sort_keys=True) == json.dumps(
        result["dev_oof_diagnostics"], sort_keys=True, default=str
    )

    run = client.get_run(run_id)
    assert "dev_oof_calibration_slope" in run.data.metrics
    assert "dev_oof_calibration_slope_ci_lower" in run.data.metrics
    assert "dev_oof_calibration_slope_ci_upper" in run.data.metrics
    assert run.data.tags["dev_oof_screen_result"] in {"pass", "fail"}
    assert (run.data.tags["dev_oof_screen_result"] == "pass") == result[
        "dev_oof_screen_passed"
    ]

    artifact_paths = {a.path for a in client.list_artifacts(run_id, "threshold")}
    assert "threshold/dev_oof_diagnostics.json" in artifact_paths


def test_run_threshold_step_raises_on_bad_slope_read_from_calibration_summary(
    full_cfg: OmegaConf,
    registered_model_run_id: str,
    registered_model_version: str,
    costs_cfg: OmegaConf,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A calibration_summary.json slope deliberately outside the band raises —
    after logging, so the audit trail records the failing attempt. Mocking
    load_calibration_summary (not calibrate.calibration_slope, which the
    folded-in screen never calls) proves it reads the number rather than
    recomputing it.
    """
    monkeypatch.setattr(threshold, "load_costs_config", lambda path=None: costs_cfg)

    def _bad_summary(run_id: str, cfg: object) -> dict:
        return {
            "method": "sigmoid",
            "calibration_slope": {
                "slope": 0.2,
                "intercept": 0.0,
                "slope_ci_lower": 0.1,
                "slope_ci_upper": 0.3,
            },
        }

    monkeypatch.setattr(threshold, "load_calibration_summary", _bad_summary)

    with pytest.raises(RuntimeError, match="Dev-OOF pre-seal screen failed"):
        threshold.run_threshold_step(
            registered_model_run_id, registered_model_version, full_cfg
        )

    reports_dir = Path(str(full_cfg.paths.reports))
    assert (reports_dir / "dev_oof_predictions.parquet").exists()
    assert (reports_dir / "dev_oof_diagnostics.json").exists()

    client = mlflow.tracking.MlflowClient()
    registered_name = str(full_cfg.mlflow.registered_model_name)
    run_id = client.get_model_version(registered_name, registered_model_version).run_id
    run = client.get_run(run_id)
    assert run.data.tags["dev_oof_screen_result"] == "fail"
    assert "dev_oof_calibration_slope" in run.data.metrics


# ---------------------------------------------------------------------------
# _run_v3_direction_sanity — the V3 pre-seal veto's own mechanics, against
# controlled inputs rather than a real (signal-free) fit.
# ---------------------------------------------------------------------------


def _v3_cfg(top_k: int, min_direction_magnitude: float = 0.3) -> OmegaConf:
    return OmegaConf.create(
        {
            "threshold": {
                "v3_top_k_features": top_k,
                "v3_min_direction_magnitude": min_direction_magnitude,
            }
        }
    )


def test_run_v3_direction_sanity_passes_on_correctly_signed_top_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every top-k feature matches an EDA key with the expected sign -> passes."""

    def _summary(run_id: str, cfg: object) -> list[dict]:
        return [
            {"feature": "numeric__tenure", "mean_abs_shap": 0.5, "direction": -0.9},
            {
                "feature": "numeric__monthlycharges",
                "mean_abs_shap": 0.3,
                "direction": 0.9,
            },
            {
                "feature": "numeric__totalcharges",
                "mean_abs_shap": 0.1,
                "direction": -0.6,
            },
        ]

    monkeypatch.setattr(threshold, "load_dev_shap_summary", _summary)
    result = threshold._run_v3_direction_sanity("fake-run", _v3_cfg(top_k=2))
    assert result["v3_result"]["passed"] is True
    assert result["checked_count"] == 2
    assert result["weak_count"] == 0


def test_run_v3_direction_sanity_fails_on_contradicting_direction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A top-k feature whose direction contradicts EXPECTED_EDA_DIRECTIONS fails."""

    def _summary(run_id: str, cfg: object) -> list[dict]:
        return [
            # tenure's established direction is -1; planted here as +0.9.
            {"feature": "numeric__tenure", "mean_abs_shap": 0.5, "direction": 0.9},
            {"feature": "numeric__noise", "mean_abs_shap": 0.1, "direction": 0.1},
        ]

    monkeypatch.setattr(threshold, "load_dev_shap_summary", _summary)
    result = threshold._run_v3_direction_sanity("fake-run", _v3_cfg(top_k=1))
    assert result["v3_result"]["passed"] is False
    assert result["checked_count"] == 1
    assert result["v3_result"]["violations"][0]["feature"] == "numeric__tenure"


def test_run_v3_direction_sanity_excludes_weak_direction_from_checked_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A top-k feature whose |direction| falls below v3_min_direction_magnitude
    is excluded from the checked set — even though its (unstable) sign here
    would otherwise contradict the established EDA direction."""

    def _summary(run_id: str, cfg: object) -> list[dict]:
        return [
            # Would contradict tenure's -1 if checked, but |0.1| < 0.3 floor.
            {"feature": "numeric__tenure", "mean_abs_shap": 0.5, "direction": 0.1},
            {"feature": "numeric__noise", "mean_abs_shap": 0.1, "direction": 0.05},
        ]

    monkeypatch.setattr(threshold, "load_dev_shap_summary", _summary)
    result = threshold._run_v3_direction_sanity("fake-run", _v3_cfg(top_k=1))
    assert result["checked_count"] == 0
    assert result["weak_count"] == 1
    assert result["v3_result"]["passed"] is True


def test_run_v3_direction_sanity_zero_checked_when_no_eda_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A top-k feature matching no EXPECTED_EDA_DIRECTIONS key is unchecked —
    checked_count is 0, distinct from a passing verdict with real evidence."""

    def _summary(run_id: str, cfg: object) -> list[dict]:
        return [
            {
                "feature": "numeric__charge_per_service",
                "mean_abs_shap": 0.5,
                "direction": 0.9,
            },
            {"feature": "numeric__noise", "mean_abs_shap": 0.1, "direction": 0.05},
        ]

    monkeypatch.setattr(threshold, "load_dev_shap_summary", _summary)
    result = threshold._run_v3_direction_sanity("fake-run", _v3_cfg(top_k=1))
    assert result["checked_count"] == 0
    assert result["v3_result"]["passed"] is True


# ---------------------------------------------------------------------------
# _run_dev_oof_screen — pinning the veto set (calibration_slope + V3 only)
# ---------------------------------------------------------------------------


def test_run_dev_oof_screen_ignores_v1_v2_v2b_flags_in_failures(
    model_promotion_config_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """V1/V2/V2b are reported-only (CLAUDE.md's three-guardrail rule): even
    with compute_dev_oof_diagnostics reporting every one of them flagged,
    `failures` stays empty as long as the calibration slope and V3 both
    pass — exactly two possible criteria ever enter `failures`, and V1/V2/V2b
    are not among them.
    """
    n = 10
    customerid = pd.Series([f"cust-{i:04d}" for i in range(n)], name="customerid")
    y_dev_arr = np.array([1, 0] * (n // 2))
    oof_proba = np.linspace(0.1, 0.9, n)
    # build_dev_oof_screen_frame's build_segment_lookup call needs these
    # columns regardless of compute_dev_oof_diagnostics being mocked below.
    feature_df = pd.DataFrame(
        {
            "customerid": customerid,
            "tenure": list(range(n)),
            "contract_type": ["Month-to-month"] * n,
            "internetservice": ["Fiber optic"] * n,
            "gender": ["Male"] * n,
            "seniorcitizen": [0] * n,
            "has_partner": ["No"] * n,
            "dependents": ["No"] * n,
        }
    )
    monkeypatch.setattr(threshold, "load_dev_partition", lambda: feature_df)

    def _fake_dev_oof_diagnostics(*args: object, **kwargs: object) -> dict:
        return {
            "segment_pr_auc": {},
            "segment_collapse_flagged": ["contract_type"],
            "segment_decision_rates": {},
            "equal_opportunity_gap_by_axis": {"gender": 0.5},
            "demographic_parity_gap_by_axis": {"gender": 0.5},
            "equal_opportunity_gap_flagged": {"gender": 0.5},
            "demographic_parity_gap_flagged": {"gender": 0.5},
            "segment_calibration": {},
            "calibration_collapse_flagged": ["gender"],
        }

    monkeypatch.setattr(
        threshold, "compute_dev_oof_diagnostics", _fake_dev_oof_diagnostics
    )

    def _passing_shap_summary(run_id: str, cfg: object) -> list[dict]:
        return [
            {"feature": "numeric__tenure", "mean_abs_shap": 0.5, "direction": -0.9},
            {
                "feature": "numeric__monthlycharges",
                "mean_abs_shap": 0.3,
                "direction": 0.9,
            },
            {
                "feature": "numeric__totalcharges",
                "mean_abs_shap": 0.1,
                "direction": -0.6,
            },
        ]

    monkeypatch.setattr(threshold, "load_dev_shap_summary", _passing_shap_summary)

    cfg = OmegaConf.create(
        {
            "threshold": {
                "n_bootstrap": 10,
                "v3_top_k_features": 2,
                "v3_min_direction_magnitude": 0.3,
            },
            "paths": {"model_promotion_config": str(model_promotion_config_path)},
        }
    )
    loaded = {
        "customerid": customerid,
        "y_dev_arr": y_dev_arr,
        "oof_proba": oof_proba,
        "calibration_summary": {
            "calibration_slope": {
                "slope": 1.0,
                "intercept": 0.0,
                "slope_ci_lower": 0.95,
                "slope_ci_upper": 1.05,
            }
        },
        "run_id": "fake-run-id",
    }

    screen = threshold._run_dev_oof_screen(
        loaded, base_threshold=0.5, cfg=cfg, random_state=42
    )

    assert screen["failures"] == []
    assert screen["dev_oof_diagnostics"]["segment_collapse_flagged"] == [
        "contract_type"
    ]
    assert screen["dev_oof_diagnostics"]["direction_checked_count"] == 2

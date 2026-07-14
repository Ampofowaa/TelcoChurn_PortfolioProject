"""Unit tests for telco_churn.models.calibrate.

Step 1 (build/wrap) and Step 2 (select method) tests operate on a lightweight
Pipeline built directly in-process — no MLflow involved, since these functions
never touch the tracking server. Step 3 (run_calibration_step) tests go
through a real tmp-scoped MLflow experiment, reusing
models.train.log_model.run_model_logging_step to produce a realistic parent
run + training_manifest.json, exactly the chain this module consumes in
production.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import mlflow
import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import telco_churn.models.calibrate as calibrate
import telco_churn.models.train.log_model as log_model
from telco_churn.features.build import FEATURE_SCHEMA
from telco_churn.features.preprocessing import build_preprocessor

_OUTER_FOLDS = 3
_INNER_FOLDS = 3

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def calibration_cfg() -> OmegaConf:
    """Small fold counts for speed — same shape as production config, not its values."""
    return OmegaConf.create(
        {
            "calibration": {
                "method": "sigmoid",
                "outer_cv_folds": _OUTER_FOLDS,
                "inner_cv_folds": _INNER_FOLDS,
                "shuffle": True,
                "random_state": 42,
                "brier_bootstrap_n_samples": 200,
                "ece_n_bins": 5,
                "ece_strategy": "uniform",
            },
            "training_setup": {"delta_threshold": 0.005},
        }
    )


@pytest.fixture
def unfitted_pipeline() -> Pipeline:
    """The real tree-family ColumnTransformer, paired with a fast linear classifier.

    LogisticRegression stands in for LightGBM purely for test speed — these
    tests exercise calibrate.py's own CV-wrapping logic, not the model family.
    """
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
def miscalibrated_data() -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic data engineered to make GaussianNB overconfident, not just separable.

    n_redundant features are linear combinations of the informative ones —
    exactly the correlation GaussianNB's independence assumption can't
    represent, which is what pushes its raw probabilities toward 0/1 while
    ranking (AP) stays intact. Empirically verified before writing this test:
    pooled Brier improves by ~0.03 under sigmoid calibration on this exact
    setup, with per-fold mean AP unchanged.
    """
    X, y = make_classification(
        n_samples=400,
        n_features=8,
        n_informative=4,
        n_redundant=4,
        n_clusters_per_class=2,
        class_sep=2.2,
        flip_y=0.03,
        random_state=42,
    )
    X_df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    return X_df, pd.Series(y, name="churn")


@pytest.fixture
def miscalibrated_pipeline() -> Pipeline:
    """GaussianNB is the deliberately-miscalibrated candidate — see miscalibrated_data."""
    return Pipeline(steps=[("scaler", StandardScaler()), ("clf", GaussianNB())])


# ---------------------------------------------------------------------------
# Step 1: build/wrap the calibrated pipeline
# ---------------------------------------------------------------------------


def test_build_calibrated_pipeline_preprocessor_refit_count(
    unfitted_pipeline: Pipeline,
    dev_split: tuple[pd.DataFrame, pd.Series],
    calibration_cfg: OmegaConf,
) -> None:
    """Leak canary: ColumnTransformer.fit_transform fires exactly inner_folds + 1
    times under ensemble=False (one per inner fold, plus the final refit on all
    of development) — the only observable proof the preprocessor refits inside
    every calibration fold instead of leaking held-out statistics across them.
    """
    X_dev, y_dev = dev_split
    calibrated = calibrate.build_calibrated_pipeline(
        unfitted_pipeline, "sigmoid", calibration_cfg
    )

    with patch.object(
        ColumnTransformer,
        "fit_transform",
        autospec=True,
        side_effect=ColumnTransformer.fit_transform,
    ) as mock_fit_transform:
        calibrated.fit(X_dev, y_dev)

    assert mock_fit_transform.call_count == _INNER_FOLDS + 1


def test_build_calibrated_pipeline_single_calibrated_classifier(
    unfitted_pipeline: Pipeline,
    dev_split: tuple[pd.DataFrame, pd.Series],
    calibration_cfg: OmegaConf,
) -> None:
    """ensemble=False collapses calibrated_classifiers_ to length 1, whose
    .estimator is a Pipeline refit on all of development — the SHAP access
    path a later evaluation stage depends on.
    """
    X_dev, y_dev = dev_split
    calibrated = calibrate.build_calibrated_pipeline(
        unfitted_pipeline, "sigmoid", calibration_cfg
    )
    calibrated.fit(X_dev, y_dev)

    assert len(calibrated.calibrated_classifiers_) == 1
    assert isinstance(calibrated.calibrated_classifiers_[0].estimator, Pipeline)


def test_oof_calibrated_proba_run_twice_is_deterministic(
    unfitted_pipeline: Pipeline,
    dev_split: tuple[pd.DataFrame, pd.Series],
    calibration_cfg: OmegaConf,
) -> None:
    """Same explicit, seeded StratifiedKFold on both calls — bit-identical OOF
    vectors, not just close ones.
    """
    X_dev, y_dev = dev_split

    first = calibrate.oof_calibrated_proba(
        unfitted_pipeline, "sigmoid", X_dev, y_dev, calibration_cfg
    )
    second = calibrate.oof_calibrated_proba(
        unfitted_pipeline, "sigmoid", X_dev, y_dev, calibration_cfg
    )

    assert np.array_equal(first, second)


# ---------------------------------------------------------------------------
# Step 2: select the calibration method
# ---------------------------------------------------------------------------


def test_calibration_improves_pooled_brier_on_miscalibrated_classifier(
    miscalibrated_pipeline: Pipeline,
    miscalibrated_data: tuple[pd.DataFrame, pd.Series],
    calibration_cfg: OmegaConf,
) -> None:
    """Calibrated outer-OOF Brier beats uncalibrated on a classifier that is
    genuinely overconfident by construction — both scored on the same pooled
    outer-OOF vector, over the same folds.
    """
    X, y = miscalibrated_data
    uncal_proba = calibrate.oof_uncalibrated_proba(
        miscalibrated_pipeline, X, y, calibration_cfg
    )
    cal_proba = calibrate.oof_calibrated_proba(
        miscalibrated_pipeline, "sigmoid", X, y, calibration_cfg
    )

    assert calibrate.pooled_brier(cal_proba, y) < calibrate.pooled_brier(uncal_proba, y)


def test_calibration_pr_auc_within_delta_of_uncalibrated(
    miscalibrated_pipeline: Pipeline,
    miscalibrated_data: tuple[pd.DataFrame, pd.Series],
    calibration_cfg: OmegaConf,
) -> None:
    """The sigmoid/isotonic gate: calibration must not degrade ranking even
    while it improves Brier — per-fold mean AP, never pooled.
    """
    X, y = miscalibrated_data
    uncal_proba = calibrate.oof_uncalibrated_proba(
        miscalibrated_pipeline, X, y, calibration_cfg
    )
    cal_proba = calibrate.oof_calibrated_proba(
        miscalibrated_pipeline, "sigmoid", X, y, calibration_cfg
    )
    uncal_ap = calibrate.per_fold_average_precision(uncal_proba, X, y, calibration_cfg)
    cal_ap = calibrate.per_fold_average_precision(cal_proba, X, y, calibration_cfg)

    assert calibrate.pr_auc_gate_passes(cal_ap, uncal_ap, calibration_cfg)


@pytest.mark.parametrize(
    ("candidate_ap", "uncalibrated_ap", "expected"),
    [
        pytest.param([0.60] * 5, [0.60] * 5, True, id="equal"),
        pytest.param([0.598] * 5, [0.60] * 5, True, id="just_within_delta"),
        pytest.param([0.585] * 5, [0.60] * 5, False, id="beyond_delta"),
        pytest.param([0.70] * 5, [0.60] * 5, True, id="candidate_better"),
    ],
)
def test_pr_auc_gate_passes_boundary(
    candidate_ap: list[float],
    uncalibrated_ap: list[float],
    expected: bool,
    calibration_cfg: OmegaConf,
) -> None:
    """A candidate within Δ* = 0.005 of uncalibrated still passes; one clearly
    beyond it does not. (The exact bit-boundary at delta == -Δ* is deliberately
    not tested here — float subtraction near an exact threshold is fragile;
    pr_auc_gate_passes's `>=` is a one-line, self-evidently-inclusive check.)
    """
    assert (
        calibrate.pr_auc_gate_passes(candidate_ap, uncalibrated_ap, calibration_cfg)
        is expected
    )


@pytest.mark.parametrize(
    ("sigmoid_brier", "isotonic_brier", "expected_method", "expected_rule"),
    [
        pytest.param(
            [0.30, 0.28, 0.31, 0.29, 0.30, 0.32, 0.29, 0.31],
            [0.18, 0.16, 0.19, 0.17, 0.18, 0.20, 0.17, 0.19],
            "isotonic",
            "isotonic_win",
            id="isotonic_win",
        ),
        pytest.param(
            [0.16, 0.18, 0.17, 0.15, 0.17, 0.19, 0.16, 0.18],
            [0.30, 0.32, 0.29, 0.31, 0.30, 0.33, 0.29, 0.31],
            "sigmoid",
            "sigmoid_win",
            id="sigmoid_win",
        ),
        pytest.param(
            [0.20, 0.21, 0.19, 0.20, 0.22, 0.18, 0.21, 0.20],
            [0.20, 0.21, 0.19, 0.20, 0.22, 0.18, 0.21, 0.20],
            "sigmoid",
            "tie",
            id="tie_identical_folds",
        ),
    ],
)
def test_brier_switch_decision(
    sigmoid_brier: list[float],
    isotonic_brier: list[float],
    expected_method: str,
    expected_rule: str,
    calibration_cfg: OmegaConf,
) -> None:
    """Three outcomes only: isotonic must decisively beat sigmoid to win — a
    tie or a sigmoid win both keep the incumbent.
    """
    result = calibrate.brier_switch_decision(
        sigmoid_brier, isotonic_brier, calibration_cfg
    )
    assert result["method"] == expected_method
    assert result["decision_rule"] == expected_rule


def test_expected_calibration_error_near_zero_when_calibrated(
    calibration_cfg: OmegaConf,
) -> None:
    """A probability vector whose predicted mean matches the observed frequency
    in its single bin has ~zero ECE."""
    proba = np.full(20, 0.10)
    y = pd.Series([1] * 2 + [0] * 18)  # observed frequency 0.10, matches proba

    ece = calibrate.expected_calibration_error(proba, y, calibration_cfg)

    assert ece == pytest.approx(0.0, abs=1e-9)


def test_expected_calibration_error_large_when_miscalibrated(
    calibration_cfg: OmegaConf,
) -> None:
    """A confidently wrong vector (predicted 0.9, observed 0.1) has ECE ~= 0.8."""
    proba = np.full(20, 0.90)
    y = pd.Series([1] * 2 + [0] * 18)  # observed frequency 0.10, predicted 0.90

    ece = calibrate.expected_calibration_error(proba, y, calibration_cfg)

    assert ece == pytest.approx(0.8, abs=1e-9)


def test_select_calibration_method_pinned_returns_proba_arrays(
    unfitted_pipeline: Pipeline,
    dev_split: tuple[pd.DataFrame, pd.Series],
    calibration_cfg: OmegaConf,
) -> None:
    """Pinned mode returns calibrated_proba/uncalibrated_proba aligned to y_dev
    — the arrays the reliability diagram is rendered from — not just diagnostics.
    """
    X_dev, y_dev = dev_split
    calibration_cfg.calibration.method = "sigmoid"

    result = calibrate.select_calibration_method(
        unfitted_pipeline, X_dev, y_dev, calibration_cfg
    )

    assert result["method"] == "sigmoid"
    assert len(result["calibrated_proba"]) == len(y_dev)
    assert len(result["uncalibrated_proba"]) == len(y_dev)
    assert result["switch_decision"] == {
        "method": "sigmoid",
        "decision_rule": "pinned",
        "delta_brier_obs": None,
        "delta_brier_ci_lower": None,
        "delta_brier_ci_upper": None,
        "n_bootstrap": None,
    }


def test_select_calibration_method_pinned_raises_on_gate_failure(
    unfitted_pipeline: Pipeline,
    dev_split: tuple[pd.DataFrame, pd.Series],
    calibration_cfg: OmegaConf,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pinned method that regresses ranking on a later retrain must fail
    loudly, not register silently — pr_auc_gate_passes forced False to
    exercise the raise without needing a real classifier to regress.
    """
    X_dev, y_dev = dev_split
    calibration_cfg.calibration.method = "sigmoid"
    monkeypatch.setattr(calibrate, "pr_auc_gate_passes", lambda *args, **kwargs: False)

    with pytest.raises(ValueError, match="failed the PR-AUC gate"):
        calibrate.select_calibration_method(
            unfitted_pipeline, X_dev, y_dev, calibration_cfg
        )


def test_select_calibration_method_unknown_method_raises(
    unfitted_pipeline: Pipeline,
    dev_split: tuple[pd.DataFrame, pd.Series],
    calibration_cfg: OmegaConf,
) -> None:
    """Anything other than 'sigmoid' / 'isotonic' / 'auto' is a config typo, not
    a silently-ignored default."""
    X_dev, y_dev = dev_split
    calibration_cfg.calibration.method = "platt"

    with pytest.raises(ValueError, match="Unknown calibration.method"):
        calibrate.select_calibration_method(
            unfitted_pipeline, X_dev, y_dev, calibration_cfg
        )


@pytest.mark.parametrize(
    "winning_method", [pytest.param("sigmoid"), pytest.param("isotonic")]
)
def test_select_calibration_method_auto_calibrated_proba_matches_winner(
    unfitted_pipeline: Pipeline,
    dev_split: tuple[pd.DataFrame, pd.Series],
    calibration_cfg: OmegaConf,
    monkeypatch: pytest.MonkeyPatch,
    winning_method: str,
) -> None:
    """calibrated_proba must be the winning method's OOF vector, not whichever
    was computed first — sigmoid and isotonic OOF are mocked to distinct
    sentinel arrays so this can't pass by coincidence.
    """
    X_dev, y_dev = dev_split
    calibration_cfg.calibration.method = "auto"
    n = len(y_dev)
    sentinels = {
        "sigmoid": np.full(n, 0.11),
        "isotonic": np.full(n, 0.22),
    }

    def fake_oof_calibrated_proba(
        pipeline: Pipeline,
        method: str,
        X: pd.DataFrame,
        y: pd.Series,
        cfg: OmegaConf,
    ) -> np.ndarray:
        return sentinels[method]

    monkeypatch.setattr(calibrate, "oof_calibrated_proba", fake_oof_calibrated_proba)
    monkeypatch.setattr(
        calibrate,
        "pr_auc_gate_passes",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        calibrate,
        "brier_switch_decision",
        lambda *args, **kwargs: {
            "method": winning_method,
            "decision_rule": f"{winning_method}_win",
            "delta_brier_obs": 0.0,
            "delta_brier_ci_lower": 0.0,
            "delta_brier_ci_upper": 0.0,
            "n_bootstrap": 1,
        },
    )

    result = calibrate.select_calibration_method(
        unfitted_pipeline, X_dev, y_dev, calibration_cfg
    )

    assert result["method"] == winning_method
    assert np.array_equal(result["calibrated_proba"], sentinels[winning_method])


def test_select_calibration_method_auto_disqualifies_isotonic_on_pr_auc_gate(
    miscalibrated_pipeline: Pipeline,
    miscalibrated_data: tuple[pd.DataFrame, pd.Series],
    calibration_cfg: OmegaConf,
) -> None:
    """Real, unmocked 'auto' run: isotonic overfits GaussianNB's small-sample
    OOF badly enough to fail the PR-AUC gate, so sigmoid ships instead — the
    exact isotonic_disqualified_pr_auc_gate branch that fired in the real
    production calibration run (ANALYSIS.md §5), previously exercised only via
    a fully-mocked pr_auc_gate_passes/brier_switch_decision and never actually
    reached in the test suite (calibrate.py lines 496-504 were uncovered).
    """
    X, y = miscalibrated_data
    calibration_cfg.calibration.method = "auto"

    result = calibrate.select_calibration_method(
        miscalibrated_pipeline, X, y, calibration_cfg
    )

    assert result["method"] == "sigmoid"
    assert result["switch_decision"] == {
        "method": "sigmoid",
        "decision_rule": "isotonic_disqualified_pr_auc_gate",
        "delta_brier_obs": None,
        "delta_brier_ci_lower": None,
        "delta_brier_ci_upper": None,
        "n_bootstrap": None,
    }
    assert "isotonic" in result["diagnostics"]


def test_select_calibration_method_auto_runs_real_brier_bootstrap_when_isotonic_eligible(
    unfitted_pipeline: Pipeline,
    dev_split: tuple[pd.DataFrame, pd.Series],
    calibration_cfg: OmegaConf,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Force isotonic past the PR-AUC gate — every per-fold AP/Brier value and
    the switch decision itself stay genuinely computed — to exercise the real
    Brier-bootstrap code path end-to-end (calibrate.py lines 515-534), only
    reachable before this test via a fully-mocked brier_switch_decision.
    """
    X_dev, y_dev = dev_split
    calibration_cfg.calibration.method = "auto"
    monkeypatch.setattr(calibrate, "pr_auc_gate_passes", lambda *args, **kwargs: True)

    result = calibrate.select_calibration_method(
        unfitted_pipeline, X_dev, y_dev, calibration_cfg
    )

    assert result["method"] in ("sigmoid", "isotonic")
    switch_decision = result["switch_decision"]
    assert switch_decision["decision_rule"] in ("isotonic_win", "sigmoid_win", "tie")
    assert "delta_brier_obs" in switch_decision
    assert "isotonic" in result["diagnostics"]
    assert "sigmoid" in result["diagnostics"]


# ---------------------------------------------------------------------------
# Step 3: register the training cycle's single deployable artifact
# ---------------------------------------------------------------------------


@pytest.fixture
def calibration_mlflow_uri(mlflow_test_experiment: Callable[[str], str]) -> str:
    """Point MLflow at the shared tmp-scoped experiment (conftest.py ::
    mlflow_test_experiment)."""
    return mlflow_test_experiment("test_run_calibration_step")


@pytest.fixture
def registration_cfg(calibration_mlflow_uri: str, tmp_path: Path) -> OmegaConf:
    """Full cfg for run_calibration_step: training + calibration + mlflow + paths.

    paths.figures is an absolute tmp_path — get_project_root() / an absolute
    path resolves to the absolute path, so this sandboxes the reliability plot
    away from the real reports/figures/ directory.
    """
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
            "calibration": {
                "method": "sigmoid",
                "outer_cv_folds": _OUTER_FOLDS,
                "inner_cv_folds": _INNER_FOLDS,
                "shuffle": True,
                "random_state": 42,
                "brier_bootstrap_n_samples": 200,
                "ece_n_bins": 5,
                "ece_strategy": "uniform",
                "run_id": None,
                "override_trial_count_gate": False,
            },
            "mlflow": {
                "tracking_uri": calibration_mlflow_uri,
                "experiment_name": "test_run_calibration_step",
                "registered_model_name": "test-telco-churn-pipeline",
            },
            "paths": {"figures": str(tmp_path / "figures")},
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
def sandboxed_dev_features(
    monkeypatch: pytest.MonkeyPatch, dev_split: tuple[pd.DataFrame, pd.Series]
) -> None:
    """Redirect run_calibration_step's load_dev_features() to the synthetic dev_split fixture.

    Unpatched, load_dev_features() falls through to load_features() ->
    partition() -> load_split(), all called with no path override — reading
    the real, gitignored datasets/processed/ directory. That happens to work
    today only because a real processed CSV + split manifest exist locally;
    on a fresh clone (no pipeline run yet), every Step 3 test would fail with
    FileNotFoundError before touching any calibration logic.
    """
    X_dev, y_dev = dev_split

    def _fake_load_dev_features(
        committed_features: list[str],
    ) -> tuple[pd.DataFrame, pd.Series]:
        return X_dev[committed_features], y_dev

    monkeypatch.setattr(calibrate, "load_dev_features", _fake_load_dev_features)


def _log_parent_run(
    dev_split: tuple[pd.DataFrame, pd.Series],
    comparison_result: dict,
    tuning_result: dict,
    cfg: OmegaConf,
) -> str:
    """Reuse log_model.run_model_logging_step to produce a real tuning_study run
    with a valid training_manifest.json — the exact chain calibrate.py consumes
    in production, not a hand-built substitute.
    """
    X_dev, y_dev = dev_split
    with mlflow.start_run(run_name="tuning_study") as run:
        tuning_result = {**tuning_result, "parent_run_id": run.info.run_id}
    result = log_model.run_model_logging_step(
        X_dev, y_dev, comparison_result, tuning_result, cfg
    )
    return str(result["run_id"])


def test_run_calibration_step_registers_and_tags(
    registration_cfg: OmegaConf,
    dev_split: tuple[pd.DataFrame, pd.Series],
    comparison_result: dict,
    tuning_result: dict,
    sandboxed_dev_features: None,
) -> None:
    """Registers exactly one version, tags refit_scope=dev, and points
    challenger at it — the training cycle's single registration point.
    """
    run_id = _log_parent_run(
        dev_split, comparison_result, tuning_result, registration_cfg
    )

    result = calibrate.run_calibration_step(run_id, registration_cfg)

    assert result["run_id"] == run_id
    assert result["method"] == "sigmoid"
    assert result["parity_ok"] is True

    client = mlflow.tracking.MlflowClient()
    registered_name = str(registration_cfg.mlflow.registered_model_name)
    version = client.get_model_version(registered_name, result["model_version"])
    assert version.tags["refit_scope"] == "dev"

    registered_model = client.get_registered_model(registered_name)
    assert str(registered_model.aliases["challenger"]) == result["model_version"]


def test_run_calibration_step_logs_reliability_diagram(
    registration_cfg: OmegaConf,
    dev_split: tuple[pd.DataFrame, pd.Series],
    comparison_result: dict,
    tuning_result: dict,
    sandboxed_dev_features: None,
) -> None:
    """The pre/post-calibration reliability diagram is logged onto the run's
    figures/ artifacts alongside calibration_summary.json — not left to a
    notebook to remember to produce.
    """
    run_id = _log_parent_run(
        dev_split, comparison_result, tuning_result, registration_cfg
    )

    calibrate.run_calibration_step(run_id, registration_cfg)

    client = mlflow.tracking.MlflowClient()
    artifact_paths = {a.path for a in client.list_artifacts(run_id, "figures")}
    assert "figures/reliability_diagram.png" in artifact_paths

    artifact_paths_root = {a.path for a in client.list_artifacts(run_id)}
    assert "calibration_summary.json" in artifact_paths_root


def test_run_calibration_step_blocks_on_low_trial_count(
    registration_cfg: OmegaConf,
    dev_split: tuple[pd.DataFrame, pd.Series],
    comparison_result: dict,
    tuning_result: dict,
    sandboxed_dev_features: None,
) -> None:
    """A data-quality gate on the tuning result, not a performance comparison
    — too few completed Optuna trials to trust the 1-SE pick blocks
    registration outright.
    """
    tuning_result["tuning_summary"]["trial_count_below_threshold"] = True
    run_id = _log_parent_run(
        dev_split, comparison_result, tuning_result, registration_cfg
    )

    with pytest.raises(RuntimeError, match="trial_count_below_threshold"):
        calibrate.run_calibration_step(run_id, registration_cfg)

    client = mlflow.tracking.MlflowClient()
    assert client.search_registered_models() == []


def test_run_calibration_step_override_trial_count_gate(
    registration_cfg: OmegaConf,
    dev_split: tuple[pd.DataFrame, pd.Series],
    comparison_result: dict,
    tuning_result: dict,
    sandboxed_dev_features: None,
) -> None:
    """override_trial_count_gate=true forces registration despite the low-trial
    warning — an explicit human override, not a silent bypass."""
    tuning_result["tuning_summary"]["trial_count_below_threshold"] = True
    registration_cfg.calibration.override_trial_count_gate = True
    run_id = _log_parent_run(
        dev_split, comparison_result, tuning_result, registration_cfg
    )

    result = calibrate.run_calibration_step(run_id, registration_cfg)

    assert result["model_version"] == "1"

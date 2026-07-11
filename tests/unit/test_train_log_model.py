"""Unit tests for telco_churn.models.train.log_model — Step 5 (C2)."""

from __future__ import annotations

from collections.abc import Callable

import mlflow
import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

import telco_churn.models.train.log_model as log_model

# ---------------------------------------------------------------------------
# run_model_logging_step (C2)
# ---------------------------------------------------------------------------


@pytest.fixture
def tuning_mlflow_uri(mlflow_test_experiment: Callable[[str], str]) -> str:
    """Point MLflow at the shared tmp-scoped experiment (conftest.py ::
    mlflow_test_experiment)."""
    return mlflow_test_experiment("test_run_model_logging_step")


@pytest.fixture
def registration_cfg() -> OmegaConf:
    """Minimal cfg for run_model_logging_step — training.fixed + a registry name."""
    return OmegaConf.create(
        {
            "random_seed": 42,
            "training_setup": {"class_weight": "balanced"},
            "training": {
                "fixed": {
                    "subsample_freq": 1,
                    "deterministic": True,
                    "force_row_wise": True,
                    "n_jobs": 1,
                    "verbose": -1,
                },
            },
            "tuning": {"selection_rule": "1se"},
            "mlflow": {
                "tracking_uri": "placeholder",
                "experiment_name": "test_run_model_logging_step",
                "registered_model_name": "test-telco-churn-pipeline",
            },
        }
    )


def _start_parent_run() -> str:
    """Create a real finished run to stand in for the Step 4 tuning_study parent.

    Relies on the tuning_mlflow_uri fixture already having set the tracking URI
    and active experiment — re-setting them here was redundant and, once the
    active experiment carries an explicit artifact_location (see conftest.py ::
    mlflow_test_experiment), re-resolving it by name only accidentally lands on
    the same experiment.
    """
    with mlflow.start_run(run_name="tuning_study") as run:
        return str(run.info.run_id)


@pytest.fixture
def tuning_result(dev_split: tuple[pd.DataFrame, pd.Series]) -> dict:
    """A plausible Step 4 output — 8 LightGBM hyperparameters + a committed feature list."""
    X_dev, _ = dev_split
    return {
        "best_params": {
            "num_leaves": 15,
            "learning_rate": 0.1,
            "min_child_samples": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "max_depth": 5,
        },
        "best_n_estimators_median": 20,
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
    """A plausible Step 2 output — the fields run_model_logging_step reads for training_manifest.json."""
    return {
        "delta_obs": 0.01,
        "delta_ci_lower": -0.01,
        "delta_ci_upper": 0.03,
        "decision": "lgbm",
        "decision_rule": "tie",
        "diagnostics": {"fixed_recall": [], "fairness": [], "robustness": []},
    }


def test_run_model_logging_step_returns_expected_keys(
    tuning_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    registration_cfg: OmegaConf,
    tuning_result: dict,
    comparison_result: dict,
) -> None:
    """run_model_logging_step logs and reloads, returning the documented keys —
    no 'version', since this step never registers.
    """
    registration_cfg.mlflow.tracking_uri = tuning_mlflow_uri
    X_dev, y_dev = dev_split
    tuning_result["parent_run_id"] = _start_parent_run()

    result = log_model.run_model_logging_step(
        X_dev, y_dev, comparison_result, tuning_result, registration_cfg
    )

    assert set(result) == {
        "run_id",
        "model_uri",
        "parity_ok",
        "training_manifest",
    }
    assert result["run_id"] == tuning_result["parent_run_id"]
    assert result["parity_ok"] is True


def test_run_model_logging_step_training_manifest_has_expected_fields(
    tuning_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    registration_cfg: OmegaConf,
    tuning_result: dict,
    comparison_result: dict,
) -> None:
    """training_manifest.json is grouped one section per pipeline step —
    model_comparison, feature_selection, training_summary, tuning_summary —
    with a logged_model_uri, but no 'alias', since this step never registers.
    """
    registration_cfg.mlflow.tracking_uri = tuning_mlflow_uri
    X_dev, y_dev = dev_split
    tuning_result["parent_run_id"] = _start_parent_run()

    result = log_model.run_model_logging_step(
        X_dev, y_dev, comparison_result, tuning_result, registration_cfg
    )

    manifest = result["training_manifest"]

    # tuning_summary: everything run_tuning_step returned, passed through
    # unchanged, plus selected_hyperparameters added at manifest-build time.
    for key, value in tuning_result["tuning_summary"].items():
        assert manifest["tuning_summary"][key] == value
    assert manifest["tuning_summary"]["selected_hyperparameters"]["num_leaves"] == 15
    assert manifest["tuning_summary"]["selected_hyperparameters"]["n_estimators"] == 20
    assert manifest["tuning_summary"]["selected_cv_pr_auc"] == 0.6

    # training_summary: the fixed knobs applied to every fit, not tuning output.
    assert (
        manifest["training_summary"]["fixed_hyperparameters"]["class_weight"]
        == "balanced"
    )
    assert "num_leaves" not in manifest["training_summary"]["fixed_hyperparameters"]

    # model_comparison: Step 2's decision, not tuning's.
    assert manifest["model_comparison"]["decision_rule"] == "tie"
    assert manifest["model_comparison"]["delta_ci_lower"] == -0.01

    # feature_selection: Step 3's frozen input space.
    assert (
        manifest["feature_selection"]["model_features"]
        == tuning_result["committed_features"]
    )
    assert set(manifest["feature_selection"]["feature_space"]) >= set(
        tuning_result["committed_features"]
    )

    assert "git_sha" in manifest
    assert "dvc_data_hash" in manifest
    assert manifest["logged_model_uri"] == result["model_uri"]

    # old flat/duplicate top-level keys must be gone, not just supplemented.
    for stale_key in (
        "hyperparameters",
        "cv_pr_auc_mean",
        "tuning_selection_rule",
        "feature_space",
        "feature_columns",
        "paired_delta_vs_logreg",
        "alias",
        "version",
    ):
        assert stale_key not in manifest


def test_run_model_logging_step_does_not_register(
    tuning_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    registration_cfg: OmegaConf,
    tuning_result: dict,
    comparison_result: dict,
) -> None:
    """No registry version is created — the registry stays empty until Phase 6's
    calibrate.py performs the training cycle's single registration.

    CLAUDE.md: an uncalibrated pipeline is a stage of construction, not a valid
    rollback target, and must never occupy a registry version number.
    """
    registration_cfg.mlflow.tracking_uri = tuning_mlflow_uri
    X_dev, y_dev = dev_split
    tuning_result["parent_run_id"] = _start_parent_run()

    log_model.run_model_logging_step(
        X_dev, y_dev, comparison_result, tuning_result, registration_cfg
    )

    client = mlflow.tracking.MlflowClient()
    assert client.search_registered_models() == []


def test_run_model_logging_step_signature_declares_float_output(
    tuning_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    registration_cfg: OmegaConf,
    tuning_result: dict,
    comparison_result: dict,
) -> None:
    """The logged signature declares a float probability output, not int64 —
    proof pyfunc_predict_fn="predict_proba" is actually wired, not just the
    parity check (which calls predict_proba directly on the sklearn flavour and
    would pass even if the pyfunc flavour still defaulted to predict()).
    """
    registration_cfg.mlflow.tracking_uri = tuning_mlflow_uri
    X_dev, y_dev = dev_split
    tuning_result["parent_run_id"] = _start_parent_run()

    result = log_model.run_model_logging_step(
        X_dev, y_dev, comparison_result, tuning_result, registration_cfg
    )

    model_info = mlflow.models.get_model_info(result["model_uri"])
    assert model_info.signature is not None
    output_types = [str(spec.type) for spec in model_info.signature.outputs.inputs]
    assert all("float" in t for t in output_types)
    assert not any("int" in t for t in output_types)


def test_run_model_logging_step_parity_failure_raises(
    tuning_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    registration_cfg: OmegaConf,
    tuning_result: dict,
    comparison_result: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reload that disagrees with the in-memory model aborts logging."""
    registration_cfg.mlflow.tracking_uri = tuning_mlflow_uri
    X_dev, y_dev = dev_split
    tuning_result["parent_run_id"] = _start_parent_run()

    class _WrongPredictions:
        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            return np.zeros((len(X), 2))  # guaranteed mismatch vs. real probabilities

    monkeypatch.setattr(
        log_model.mlflow.sklearn, "load_model", lambda uri: _WrongPredictions()
    )

    with pytest.raises(AssertionError, match="Reload parity check failed"):
        log_model.run_model_logging_step(
            X_dev, y_dev, comparison_result, tuning_result, registration_cfg
        )

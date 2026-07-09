"""Unit tests for telco_churn.models.train.registration — Step 5 (C2)."""

from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

import telco_churn.models.train.registration as registration

# ---------------------------------------------------------------------------
# run_registration_step (C2)
# ---------------------------------------------------------------------------


@pytest.fixture
def tuning_mlflow_uri(tmp_path: Path) -> str:
    """Point MLflow at a throwaway local SQLite store (see test_train_comparison.py)."""
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("test_run_registration_step")
    return uri


@pytest.fixture
def registration_cfg() -> OmegaConf:
    """Minimal cfg for run_registration_step — training.fixed + a registry name."""
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
                "experiment_name": "test_run_registration_step",
                "registered_model_name": "test-telco-churn-pipeline",
            },
        }
    )


def _start_parent_run(tracking_uri: str, experiment_name: str) -> str:
    """Create a real finished run to stand in for the Step 4 tuning_study parent."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
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
    """A plausible Step 2 output — the fields run_registration_step reads for training_manifest.json."""
    return {
        "delta_obs": 0.01,
        "delta_ci_lower": -0.01,
        "delta_ci_upper": 0.03,
        "decision": "lgbm",
        "decision_rule": "tie_immaterial",
        "diagnostics": {"fixed_recall": [], "fairness": [], "robustness": []},
    }


def test_run_registration_step_returns_expected_keys(
    tuning_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    registration_cfg: OmegaConf,
    tuning_result: dict,
    comparison_result: dict,
) -> None:
    """run_registration_step logs, reloads, and registers, returning the documented keys."""
    registration_cfg.mlflow.tracking_uri = tuning_mlflow_uri
    X_dev, y_dev = dev_split
    tuning_result["parent_run_id"] = _start_parent_run(
        tuning_mlflow_uri, str(registration_cfg.mlflow.experiment_name)
    )

    result = registration.run_registration_step(
        X_dev, y_dev, comparison_result, tuning_result, registration_cfg
    )

    assert set(result) == {
        "run_id",
        "model_uri",
        "version",
        "parity_ok",
        "training_manifest",
    }
    assert result["run_id"] == tuning_result["parent_run_id"]
    assert result["parity_ok"] is True


def test_run_registration_step_training_manifest_has_expected_fields(
    tuning_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    registration_cfg: OmegaConf,
    tuning_result: dict,
    comparison_result: dict,
) -> None:
    """training_manifest.json carries hyperparameters, hashes, CV PR-AUC, and paired-delta CI."""
    registration_cfg.mlflow.tracking_uri = tuning_mlflow_uri
    X_dev, y_dev = dev_split
    tuning_result["parent_run_id"] = _start_parent_run(
        tuning_mlflow_uri, str(registration_cfg.mlflow.experiment_name)
    )

    result = registration.run_registration_step(
        X_dev, y_dev, comparison_result, tuning_result, registration_cfg
    )

    manifest = result["training_manifest"]
    assert manifest["hyperparameters"]["num_leaves"] == 15
    assert manifest["hyperparameters"]["n_estimators"] == 20
    assert manifest["cv_pr_auc_mean"] == 0.6
    assert manifest["paired_delta_vs_logreg"]["decision_rule"] == "tie_immaterial"
    assert manifest["paired_delta_vs_logreg"]["delta_ci_lower"] == -0.01
    assert manifest["feature_columns"] == tuning_result["committed_features"]
    assert set(manifest["feature_space"]) >= set(tuning_result["committed_features"])
    assert "git_sha" in manifest
    assert "dvc_data_hash" in manifest
    assert manifest["tuning_summary"] == tuning_result["tuning_summary"]


def test_run_registration_step_sets_challenger_alias(
    tuning_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    registration_cfg: OmegaConf,
    tuning_result: dict,
    comparison_result: dict,
) -> None:
    """The registered model version is reachable via the 'challenger' alias."""
    registration_cfg.mlflow.tracking_uri = tuning_mlflow_uri
    X_dev, y_dev = dev_split
    tuning_result["parent_run_id"] = _start_parent_run(
        tuning_mlflow_uri, str(registration_cfg.mlflow.experiment_name)
    )

    result = registration.run_registration_step(
        X_dev, y_dev, comparison_result, tuning_result, registration_cfg
    )

    client = mlflow.tracking.MlflowClient()
    version = client.get_model_version_by_alias(
        str(registration_cfg.mlflow.registered_model_name), "challenger"
    )
    assert str(version.version) == str(result["version"])


def test_run_registration_step_parity_failure_raises(
    tuning_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    registration_cfg: OmegaConf,
    tuning_result: dict,
    comparison_result: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reload that disagrees with the in-memory model aborts registration."""
    registration_cfg.mlflow.tracking_uri = tuning_mlflow_uri
    X_dev, y_dev = dev_split
    tuning_result["parent_run_id"] = _start_parent_run(
        tuning_mlflow_uri, str(registration_cfg.mlflow.experiment_name)
    )

    class _WrongPredictions:
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return np.full(
                len(X), -1
            )  # -1 is not a valid class label — guaranteed mismatch

    monkeypatch.setattr(
        registration.mlflow.sklearn, "load_model", lambda uri: _WrongPredictions()
    )

    with pytest.raises(AssertionError, match="Reload parity check failed"):
        registration.run_registration_step(
            X_dev, y_dev, comparison_result, tuning_result, registration_cfg
        )

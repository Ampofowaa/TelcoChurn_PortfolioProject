"""Unit tests for telco_churn.models.train.feature_freeze — Step 3 orchestrator (B6).

features/select.py's own units (PermutationImportanceSelector, decide_survivors,
run_selection_cv, ...) are covered in test_select.py. These tests instead cover the
top-level `run_selection_step` wiring: does it call those units with the right
arguments, log the documented MLflow artifacts, and return a result consistent with
its own internal decision — the piece that had no direct unit coverage before.
"""

from __future__ import annotations

from collections.abc import Callable

import mlflow
import pandas as pd
import pytest
from omegaconf import OmegaConf
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import RepeatedStratifiedKFold

import telco_churn.models.train.feature_freeze as feature_freeze
from telco_churn.features.build import FEATURE_SCHEMA
from telco_churn.models.train.common import cv_score_candidate

_CV_FOLDS = 2
_CV_REPEATS = 1


@pytest.fixture
def selection_mlflow_uri(mlflow_test_experiment: Callable[[str], str]) -> str:
    """Point MLflow at the shared tmp-scoped experiment (conftest.py ::
    mlflow_test_experiment)."""
    return mlflow_test_experiment("test_run_selection_step")


@pytest.fixture
def selection_cfg() -> OmegaConf:
    """A tiny selection config — 2-fold CV, few permutation repeats, a small LightGBM."""
    return OmegaConf.create(
        {
            "random_seed": 42,
            "training_setup": {
                "cv_folds": _CV_FOLDS,
                "cv_repeats": _CV_REPEATS,
                "delta_threshold": 0.005,
                "cv_n_jobs": 1,
                "class_weight": "balanced",
            },
            "training": {
                "candidate": {
                    "n_estimators": 20,
                    "num_leaves": 8,
                    "min_child_samples": 5,
                },
                "fixed": {
                    "subsample_freq": 1,
                    "deterministic": True,
                    "force_row_wise": True,
                    "n_jobs": 1,
                    "verbose": -1,
                },
            },
            "selection": {
                "n_repeats": 5,
                "noise_floor_margin": 0.005,
                "inner_val_size": 0.2,
                "random_state": 42,
                "bootstrap_n_samples": 200,
                "correlated_groups": [["tenure", "totalcharges", "monthlycharges"]],
            },
            "mlflow": {
                "tracking_uri": "placeholder",
                "experiment_name": "test_run_selection_step",
            },
        }
    )


@pytest.fixture
def full_candidate_results(
    dev_split: tuple[pd.DataFrame, pd.Series],
) -> dict[str, dict]:
    """A stand-in Step 1 lgbm_default result, CV'd on the exact fold scheme
    run_selection_step builds internally (n_splits=_CV_FOLDS, n_repeats=_CV_REPEATS,
    random_state=selection_cfg.random_seed) — required so the paired bootstrap test
    inside run_selection_step compares fold i of the full model against fold i of the
    reduced model, not two independently-resampled arrays.
    """
    X_dev, y_dev = dev_split
    cv = RepeatedStratifiedKFold(
        n_splits=_CV_FOLDS, n_repeats=_CV_REPEATS, random_state=42
    )
    return {
        "lgbm_default": cv_score_candidate(
            DummyClassifier(strategy="prior", random_state=42), X_dev, y_dev, cv
        )
    }


def test_run_selection_step_returns_expected_keys(
    selection_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    full_candidate_results: dict[str, dict],
    selection_cfg: OmegaConf,
) -> None:
    """run_selection_step returns the documented keys with a valid decision."""
    selection_cfg.mlflow.tracking_uri = selection_mlflow_uri
    X_dev, y_dev = dev_split

    result = feature_freeze.run_selection_step(
        X_dev, y_dev, full_candidate_results, selection_cfg
    )

    assert set(result) == {
        "decision",
        "committed_features",
        "permutation_importance_table",
        "shap_audit",
        "shap_audit_committed",
        "high_shap_dropouts",
        "stability",
        "bootstrap_test",
    }
    assert result["decision"] in {"reduced", "full"}
    assert len(result["committed_features"]) > 0


def test_run_selection_step_committed_features_match_decision_source(
    selection_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    full_candidate_results: dict[str, dict],
    selection_cfg: OmegaConf,
) -> None:
    """committed_features is the permutation selector's survivors on a 'reduced'
    decision, or the full feature space on a 'full' decision — never a stale or
    mismatched list from the wrong branch.
    """
    selection_cfg.mlflow.tracking_uri = selection_mlflow_uri
    X_dev, y_dev = dev_split

    result = feature_freeze.run_selection_step(
        X_dev, y_dev, full_candidate_results, selection_cfg
    )

    if result["decision"] == "reduced":
        survived = {
            row["feature"]
            for row in result["permutation_importance_table"]
            if row["survived"]
        }
        assert set(result["committed_features"]) == survived
    else:
        all_features = set(
            FEATURE_SCHEMA.binary + FEATURE_SCHEMA.multi_cat + FEATURE_SCHEMA.numeric
        )
        assert set(result["committed_features"]) == all_features


def test_run_selection_step_shap_audit_committed_matches_decision(
    selection_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    full_candidate_results: dict[str, dict],
    selection_cfg: OmegaConf,
) -> None:
    """shap_audit_committed (the ship-accurate SHAP view) is populated only when the
    reduced set actually ships — never computed (and never None) in the wrong branch.
    """
    selection_cfg.mlflow.tracking_uri = selection_mlflow_uri
    X_dev, y_dev = dev_split

    result = feature_freeze.run_selection_step(
        X_dev, y_dev, full_candidate_results, selection_cfg
    )

    if result["decision"] == "reduced" and result["committed_features"]:
        assert result["shap_audit_committed"] is not None
        committed_shap_features = {
            row["feature"] for row in result["shap_audit_committed"]
        }
        assert committed_shap_features == set(result["committed_features"])
    else:
        assert result["shap_audit_committed"] is None


def test_run_selection_step_bootstrap_test_decision_matches_top_level_decision(
    selection_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    full_candidate_results: dict[str, dict],
    selection_cfg: OmegaConf,
) -> None:
    """The top-level 'decision' is exactly bootstrap_test's decision — a wiring
    regression test (e.g. reading the wrong dict key) would fail this.
    """
    selection_cfg.mlflow.tracking_uri = selection_mlflow_uri
    X_dev, y_dev = dev_split

    result = feature_freeze.run_selection_step(
        X_dev, y_dev, full_candidate_results, selection_cfg
    )

    assert result["decision"] == result["bootstrap_test"]["decision"]


def test_run_selection_step_stability_covers_full_feature_space(
    selection_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    full_candidate_results: dict[str, dict],
    selection_cfg: OmegaConf,
) -> None:
    """stability (per-fold survival rate) covers every source feature, not just the
    ones that ultimately survived — it's an audit trail over the full candidate space.
    """
    selection_cfg.mlflow.tracking_uri = selection_mlflow_uri
    X_dev, y_dev = dev_split

    result = feature_freeze.run_selection_step(
        X_dev, y_dev, full_candidate_results, selection_cfg
    )

    all_features = set(
        FEATURE_SCHEMA.binary + FEATURE_SCHEMA.multi_cat + FEATURE_SCHEMA.numeric
    )
    assert {row["feature"] for row in result["stability"]} == all_features


def test_run_selection_step_logs_selection_artifacts(
    selection_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    full_candidate_results: dict[str, dict],
    selection_cfg: OmegaConf,
) -> None:
    """The feature_selection run carries the permutation-importance table, per-fold
    stability, SHAP audit, dropouts, committed list, and both figures — the full
    artifact contract documented in run_selection_step's docstring.
    """
    selection_cfg.mlflow.tracking_uri = selection_mlflow_uri
    X_dev, y_dev = dev_split

    result = feature_freeze.run_selection_step(
        X_dev, y_dev, full_candidate_results, selection_cfg
    )

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("test_run_selection_step")
    runs = client.search_runs(
        [experiment.experiment_id], filter_string="tags.stage = 'selection'"
    )
    assert len(runs) == 1
    run = runs[0]

    artifact_paths = {
        f.path for f in client.list_artifacts(run.info.run_id, "selection")
    }
    expected = {
        "selection/permutation_importance_table.csv",
        "selection/per_fold_stability.csv",
        "selection/shap_importance_audit.csv",
        "selection/high_shap_dropouts.txt",
        "selection/committed_features.txt",
        "selection/bootstrap_delta_dist.png",
        "selection/pr_curves.png",
    }
    assert expected <= artifact_paths
    if result["decision"] == "reduced" and result["committed_features"]:
        assert "selection/shap_importance_audit_committed.csv" in artifact_paths
    else:
        assert "selection/shap_importance_audit_committed.csv" not in artifact_paths

    assert run.data.params["decision"] == result["decision"]
    assert int(run.data.params["n_committed_features"]) == len(
        result["committed_features"]
    )
    assert "reduced_cv_pr_auc_mean" in run.data.metrics
    assert "full_cv_pr_auc_mean" in run.data.metrics


def test_run_selection_step_tags_stage_git_sha_and_dvc_hash(
    selection_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    full_candidate_results: dict[str, dict],
    selection_cfg: OmegaConf,
) -> None:
    """The run carries the same stage/git_sha/dvc_data_hash tag contract every other
    Phase 5 step run carries (candidates.py, comparison.py, tuning.py)."""
    selection_cfg.mlflow.tracking_uri = selection_mlflow_uri
    X_dev, y_dev = dev_split

    feature_freeze.run_selection_step(
        X_dev, y_dev, full_candidate_results, selection_cfg
    )

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("test_run_selection_step")
    run = client.search_runs([experiment.experiment_id])[0]

    assert run.data.tags["stage"] == "selection"
    assert "git_sha" in run.data.tags
    assert "dvc_data_hash" in run.data.tags

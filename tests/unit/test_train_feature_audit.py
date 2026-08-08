"""Unit tests for telco_churn.models.train.feature_audit — Step 3 orchestrator (B6).

features/select.py's own units (PermutationImportanceSelector, decide_survivors,
mint_committed_list, ...) are covered in test_select.py. These tests instead cover
the top-level `run_feature_audit_step` wiring: does it call those units with the right
arguments, log the documented MLflow artifacts, and return a result consistent with
the full (always-committed) feature space — the piece that had no direct unit
coverage before.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import mlflow
import pandas as pd
import pytest
from omegaconf import OmegaConf

import telco_churn.models.train.feature_audit as feature_audit
from telco_churn.features.accessor import features_path
from telco_churn.features.build import COMMITTED_FEATURES, FEATURE_SCHEMA


@pytest.fixture
def selection_mlflow_uri(mlflow_test_experiment: Callable[[str], str]) -> str:
    """Point MLflow at the shared tmp-scoped experiment (conftest.py ::
    mlflow_test_experiment)."""
    return mlflow_test_experiment("test_run_feature_audit_step")


@pytest.fixture
def selection_cfg() -> OmegaConf:
    """A tiny selection config — few permutation repeats, a small LightGBM."""
    return OmegaConf.create(
        {
            "random_seed": 42,
            "training_setup": {
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
                "correlated_groups": [["tenure", "totalcharges", "monthlycharges"]],
            },
            "mlflow": {
                "tracking_uri": "placeholder",
                "experiment_name": "test_run_feature_audit_step",
            },
        }
    )


def test_run_feature_audit_step_returns_expected_keys(
    selection_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    selection_cfg: OmegaConf,
) -> None:
    """run_feature_audit_step returns the documented keys."""
    selection_cfg.mlflow.tracking_uri = selection_mlflow_uri
    X_dev, y_dev = dev_split

    result = feature_audit.run_feature_audit_step(X_dev, y_dev, selection_cfg)

    assert set(result) == {
        "committed_features",
        "permutation_importance_table",
        "shap_audit",
        "high_shap_dropouts",
    }
    assert len(result["committed_features"]) > 0


def test_run_feature_audit_step_committed_features_matches_the_schema_constant(
    selection_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    selection_cfg: OmegaConf,
) -> None:
    """committed_features is read from features/schema.py::COMMITTED_FEATURES — a
    hand-maintained constant, not computed by a live CV decision (ANALYSIS.md §4b).
    The keep-vs-reduce ablation that decided its contents still exists
    (features.select.run_selection_cv/reduced_set_bootstrap_test) but is called only
    from notebooks/03b-feature-selection.ipynb's on-demand review, never from here.
    """
    selection_cfg.mlflow.tracking_uri = selection_mlflow_uri
    X_dev, y_dev = dev_split

    result = feature_audit.run_feature_audit_step(X_dev, y_dev, selection_cfg)

    assert result["committed_features"] == list(COMMITTED_FEATURES)


def test_run_feature_audit_step_permutation_importance_table_covers_full_feature_space(
    selection_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    selection_cfg: OmegaConf,
) -> None:
    """permutation_importance_table has one row per source feature — the
    diagnostic audit trail, not a filtered/committed-only view.
    """
    selection_cfg.mlflow.tracking_uri = selection_mlflow_uri
    X_dev, y_dev = dev_split

    result = feature_audit.run_feature_audit_step(X_dev, y_dev, selection_cfg)

    all_features = set(
        FEATURE_SCHEMA.binary + FEATURE_SCHEMA.multi_cat + FEATURE_SCHEMA.numeric
    )
    table_features = {row["feature"] for row in result["permutation_importance_table"]}
    assert table_features == all_features


def test_run_feature_audit_step_shap_audit_covers_full_feature_space(
    selection_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    selection_cfg: OmegaConf,
) -> None:
    """shap_audit ranks every candidate feature, flagging which are committed —
    with committed_features always the full set, every row is flagged committed.
    """
    selection_cfg.mlflow.tracking_uri = selection_mlflow_uri
    X_dev, y_dev = dev_split

    result = feature_audit.run_feature_audit_step(X_dev, y_dev, selection_cfg)

    all_features = set(
        FEATURE_SCHEMA.binary + FEATURE_SCHEMA.multi_cat + FEATURE_SCHEMA.numeric
    )
    shap_features = {row["feature"] for row in result["shap_audit"]}
    assert shap_features == all_features
    assert all(row["committed"] for row in result["shap_audit"])


def test_run_feature_audit_step_logs_selection_artifacts(
    selection_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    selection_cfg: OmegaConf,
) -> None:
    """The feature_audit run carries the permutation-importance table, SHAP
    audit, dropouts, committed list, and group-importance evidence — the full
    artifact contract documented in run_feature_audit_step's docstring.
    """
    selection_cfg.mlflow.tracking_uri = selection_mlflow_uri
    X_dev, y_dev = dev_split

    result = feature_audit.run_feature_audit_step(X_dev, y_dev, selection_cfg)

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("test_run_feature_audit_step")
    runs = client.search_runs(
        [experiment.experiment_id], filter_string="tags.stage = 'selection'"
    )
    assert len(runs) == 1
    run = runs[0]

    artifact_paths = {f.path for f in client.list_artifacts(run.info.run_id)}
    expected = {
        "permutation_importance_table.csv",
        "shap_importance_audit.csv",
        "high_shap_dropouts.txt",
        "committed_features.txt",
        "group_importance.json",
    }
    assert expected <= artifact_paths

    figures_paths = {f.path for f in client.list_artifacts(run.info.run_id, "figures")}
    assert figures_paths == {
        "figures/permutation_importance.png",
        "figures/shap_importance_audit.png",
    }

    assert int(run.data.params["n_committed_features"]) == len(
        result["committed_features"]
    )


def test_run_feature_audit_step_tags_stage_git_sha_and_data_content_hash(
    selection_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    selection_cfg: OmegaConf,
) -> None:
    """The run carries the same stage/git_sha/data_content_hash tag contract every other
    Phase 5 step run carries (candidates.py, comparison.py, tuning.py)."""
    selection_cfg.mlflow.tracking_uri = selection_mlflow_uri
    X_dev, y_dev = dev_split

    feature_audit.run_feature_audit_step(X_dev, y_dev, selection_cfg)

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("test_run_feature_audit_step")
    run = client.search_runs([experiment.experiment_id])[0]

    assert run.data.tags["stage"] == "selection"
    assert "git_sha" in run.data.tags
    assert "data_content_hash" in run.data.tags


def test_run_feature_audit_step_logs_dataset_source_via_accessor(
    selection_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    selection_cfg: OmegaConf,
) -> None:
    """The run's logged dataset input (Fix 6) resolves through
    features/accessor.py::features_path() — the same canonical source
    candidates.py already logs — not a second, hand-assembled copy that could
    silently go stale.
    """
    selection_cfg.mlflow.tracking_uri = selection_mlflow_uri
    X_dev, y_dev = dev_split

    feature_audit.run_feature_audit_step(X_dev, y_dev, selection_cfg)

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("test_run_feature_audit_step")
    run = client.get_run(client.search_runs([experiment.experiment_id])[0].info.run_id)

    dataset_inputs = run.inputs.dataset_inputs
    assert len(dataset_inputs) == 1
    source = json.loads(dataset_inputs[0].dataset.source)
    assert source["uri"] == str(features_path())

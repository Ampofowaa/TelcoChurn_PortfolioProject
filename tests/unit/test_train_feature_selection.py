"""Unit tests for telco_churn.models.train.feature_selection — the notebook-only
full-vs-reduced paired-bootstrap ablation behind COMMITTED_FEATURES."""

from __future__ import annotations

import json
from collections.abc import Callable

import mlflow
import pandas as pd
import pytest
from omegaconf import OmegaConf

import telco_churn.models.train.feature_selection as feature_selection
from telco_churn.features.accessor import features_path
from telco_churn.features.build import FEATURE_SCHEMA

_TEST_EXPERIMENT_NAME = "test_run_feature_selection_step"


@pytest.fixture
def feature_selection_mlflow_uri(
    monkeypatch: pytest.MonkeyPatch, mlflow_test_experiment: Callable[[str], str]
) -> str:
    """Point MLflow at the shared tmp-scoped experiment (conftest.py ::
    mlflow_test_experiment), and patch the module's hardcoded review-experiment
    name to match it — the function deliberately never reads
    cfg.mlflow.experiment_name for this (see module docstring: the review
    experiment is dedicated and separate from the training one), so a test
    must retarget the module constant rather than cfg.
    """
    monkeypatch.setattr(
        feature_selection, "_REVIEW_EXPERIMENT_NAME", _TEST_EXPERIMENT_NAME
    )
    return mlflow_test_experiment(_TEST_EXPERIMENT_NAME)


@pytest.fixture
def feature_selection_cfg() -> OmegaConf:
    """A tiny ablation config — few permutation repeats, a small LightGBM, few folds."""
    return OmegaConf.create(
        {
            "random_seed": 42,
            "training_setup": {
                "delta_threshold": 0.005,
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
                "n_repeats": 3,
                "noise_floor_margin": 0.005,
                "inner_val_size": 0.2,
                "random_state": 42,
                "correlated_groups": [],
            },
            "mlflow": {
                "tracking_uri": "placeholder",
            },
        }
    )


def test_run_feature_selection_step_returns_expected_keys(
    feature_selection_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    feature_selection_cfg: OmegaConf,
) -> None:
    """The returned dict is reduced_set_bootstrap_test's contract plus the
    orchestration-level summary keys."""
    feature_selection_cfg.mlflow.tracking_uri = feature_selection_mlflow_uri
    X_dev, y_dev = dev_split

    result = feature_selection.run_feature_selection_step(
        X_dev, y_dev, feature_selection_cfg, cv_folds=2, cv_repeats=1, n_bootstrap=200
    )

    assert set(result) == {
        "delta_obs",
        "delta_ci_lower",
        "delta_ci_upper",
        "p_value",
        "decision",
        "decision_rule",
        "n_bootstrap",
        "bootstrap_deltas",
        "full_cv_pr_auc_mean",
        "reduced_cv_pr_auc_mean",
        "fold_win_rate",
        "stability",
        "permutation_importance_table",
        "group_importance",
        "shap_audit",
        "high_shap_dropouts",
        "recommended_committed_features",
        "run_id",
    }
    assert result["decision"] in {"full", "reduced"}
    assert result["decision_rule"] in {
        "full_features_win",
        "reduced_features_win",
        "tie",
    }


def test_run_feature_selection_step_fold_win_rate_in_bounds(
    feature_selection_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    feature_selection_cfg: OmegaConf,
) -> None:
    """fold_win_rate is a fraction of folds, so it must land in [0, 1]."""
    feature_selection_cfg.mlflow.tracking_uri = feature_selection_mlflow_uri
    X_dev, y_dev = dev_split

    result = feature_selection.run_feature_selection_step(
        X_dev, y_dev, feature_selection_cfg, cv_folds=2, cv_repeats=1, n_bootstrap=200
    )

    assert 0.0 <= result["fold_win_rate"] <= 1.0


def test_run_feature_selection_step_stability_covers_full_feature_space(
    feature_selection_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    feature_selection_cfg: OmegaConf,
) -> None:
    """stability has one row per source feature — the reduced-set membership
    evidence a human needs if the decision ever flips to 'reduced'."""
    feature_selection_cfg.mlflow.tracking_uri = feature_selection_mlflow_uri
    X_dev, y_dev = dev_split

    result = feature_selection.run_feature_selection_step(
        X_dev, y_dev, feature_selection_cfg, cv_folds=2, cv_repeats=1, n_bootstrap=200
    )

    all_features = set(
        FEATURE_SCHEMA.binary + FEATURE_SCHEMA.multi_cat + FEATURE_SCHEMA.numeric
    )
    stability_features = {row["feature"] for row in result["stability"]}
    assert stability_features == all_features


def test_run_feature_selection_step_permutation_importance_table_covers_full_feature_space(
    feature_selection_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    feature_selection_cfg: OmegaConf,
) -> None:
    """permutation_importance_table (aggregated across run_selection_cv's own
    100 folds, not a second mint_committed_list fit) has one row per source
    feature, the same diagnostic-audit-trail contract feature_audit.py's own
    table has."""
    feature_selection_cfg.mlflow.tracking_uri = feature_selection_mlflow_uri
    X_dev, y_dev = dev_split

    result = feature_selection.run_feature_selection_step(
        X_dev, y_dev, feature_selection_cfg, cv_folds=2, cv_repeats=1, n_bootstrap=200
    )

    all_features = set(
        FEATURE_SCHEMA.binary + FEATURE_SCHEMA.multi_cat + FEATURE_SCHEMA.numeric
    )
    table_features = {row["feature"] for row in result["permutation_importance_table"]}
    assert table_features == all_features


def test_run_feature_selection_step_survived_column_is_stability_derived(
    feature_selection_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    feature_selection_cfg: OmegaConf,
) -> None:
    """permutation_importance_table's survived column is derived from
    run_selection_cv's own 100-fold stability (majority vote) — sourced
    entirely from the same run_selection_cv call that produced the bootstrap
    decision, never from a second, separately-fit mint_committed_list estimate
    (the "minting defect": a single all-dev fit is a different computation
    from 100 refits on different folds/seeds and can disagree with the
    fold-survival picture)."""
    feature_selection_cfg.mlflow.tracking_uri = feature_selection_mlflow_uri
    X_dev, y_dev = dev_split

    result = feature_selection.run_feature_selection_step(
        X_dev, y_dev, feature_selection_cfg, cv_folds=2, cv_repeats=1, n_bootstrap=200
    )

    for row in result["permutation_importance_table"]:
        assert "stability" in row
        assert "real_importance" in row
        assert "decoy_importance" in row
        # The fallback path (stability-survivors empty) forces survived=True
        # on the single most-stable feature even if it's just under the
        # threshold, so this is an implication, not a strict equality: every
        # feature clearing the threshold must be marked survived.
        if row["stability"] >= feature_selection._STABILITY_THRESHOLD:
            assert row["survived"] is True


def test_run_feature_selection_step_shap_audit_flags_recommended_features_as_committed(
    feature_selection_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    feature_selection_cfg: OmegaConf,
) -> None:
    """shap_audit (features.select.compute_shap_audit, reused not reimplemented)
    ranks every candidate feature, with exactly recommended_committed_features
    flagged committed — the SHAP audit is scored against *this review's own*
    recommendation, not whatever COMMITTED_FEATURES happens to be today."""
    feature_selection_cfg.mlflow.tracking_uri = feature_selection_mlflow_uri
    X_dev, y_dev = dev_split

    result = feature_selection.run_feature_selection_step(
        X_dev, y_dev, feature_selection_cfg, cv_folds=2, cv_repeats=1, n_bootstrap=200
    )

    all_features = set(
        FEATURE_SCHEMA.binary + FEATURE_SCHEMA.multi_cat + FEATURE_SCHEMA.numeric
    )
    shap_features = {row["feature"] for row in result["shap_audit"]}
    assert shap_features == all_features

    committed_flagged = {
        row["feature"] for row in result["shap_audit"] if row["committed"]
    }
    assert committed_flagged == set(result["recommended_committed_features"])


def test_run_feature_selection_step_recommended_features_match_decision(
    feature_selection_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    feature_selection_cfg: OmegaConf,
) -> None:
    """recommended_committed_features is the full feature space when the
    ablation decides 'full', and exactly the stability-surviving features
    (run_selection_cv's per-feature stability, via the 'survived' column)
    when it decides 'reduced' — an immediately actionable list either way,
    not just a decision label a human must translate by hand."""
    feature_selection_cfg.mlflow.tracking_uri = feature_selection_mlflow_uri
    X_dev, y_dev = dev_split

    result = feature_selection.run_feature_selection_step(
        X_dev, y_dev, feature_selection_cfg, cv_folds=2, cv_repeats=1, n_bootstrap=200
    )

    all_features = set(
        FEATURE_SCHEMA.binary + FEATURE_SCHEMA.multi_cat + FEATURE_SCHEMA.numeric
    )
    recommended = set(result["recommended_committed_features"])
    assert recommended <= all_features
    assert len(recommended) > 0

    if result["decision"] == "full":
        assert recommended == all_features
    else:
        survived = {
            row["feature"]
            for row in result["permutation_importance_table"]
            if row["survived"]
        }
        assert recommended == survived


def test_run_feature_selection_step_logs_review_artifacts(
    feature_selection_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    feature_selection_cfg: OmegaConf,
) -> None:
    """The feature_selection run carries the per-fold stability CSV, the
    bootstrap-delta figure, and the decision/params/metrics contract."""
    feature_selection_cfg.mlflow.tracking_uri = feature_selection_mlflow_uri
    X_dev, y_dev = dev_split

    result = feature_selection.run_feature_selection_step(
        X_dev, y_dev, feature_selection_cfg, cv_folds=2, cv_repeats=1, n_bootstrap=200
    )

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(_TEST_EXPERIMENT_NAME)
    runs = client.search_runs(
        [experiment.experiment_id], filter_string="tags.stage = 'selection_review'"
    )
    assert len(runs) == 1
    run = runs[0]
    assert run.info.run_id == result["run_id"]

    assert run.data.params["decision"] == result["decision"]
    assert run.data.params["decision_rule"] == result["decision_rule"]
    assert int(run.data.params["n_recommended_features"]) == len(
        result["recommended_committed_features"]
    )
    assert int(run.data.params["n_high_shap_dropouts"]) == len(
        result["high_shap_dropouts"]
    )
    assert "bootstrap_delta_obs" in run.data.metrics
    assert "full_cv_pr_auc_mean" in run.data.metrics
    assert "reduced_cv_pr_auc_mean" in run.data.metrics
    assert "fold_win_rate" in run.data.metrics

    review_artifacts = {f.path for f in client.list_artifacts(run.info.run_id)}
    assert review_artifacts == {
        "per_fold_stability.csv",
        "permutation_importance_table.csv",
        "group_importance.json",
        "shap_importance_audit.csv",
        "high_shap_dropouts.txt",
        "recommended_committed_features.txt",
        "figures",
    }

    figures_paths = {f.path for f in client.list_artifacts(run.info.run_id, "figures")}
    assert figures_paths == {
        "figures/bootstrap_delta_dist.png",
        "figures/pr_curves.png",
        "figures/permutation_importance.png",
        "figures/per_fold_stability.png",
        "figures/shap_importance_audit_survived.png",
        "figures/shap_importance_audit.png",
    }


def test_run_feature_selection_step_tags_stage_git_sha_and_data_content_hash(
    feature_selection_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    feature_selection_cfg: OmegaConf,
) -> None:
    """The run carries the same stage/git_sha/data_content_hash tag contract every
    other training-package step run carries (candidates.py, comparison.py,
    feature_audit.py, tuning.py)."""
    feature_selection_cfg.mlflow.tracking_uri = feature_selection_mlflow_uri
    X_dev, y_dev = dev_split

    feature_selection.run_feature_selection_step(
        X_dev, y_dev, feature_selection_cfg, cv_folds=2, cv_repeats=1, n_bootstrap=200
    )

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(_TEST_EXPERIMENT_NAME)
    run = client.search_runs([experiment.experiment_id])[0]

    assert run.data.tags["stage"] == "selection_review"
    assert run.data.tags["triggered_by"] == "manual"
    assert "git_sha" in run.data.tags
    assert "data_content_hash" in run.data.tags


def test_run_feature_selection_step_logs_dataset_source_via_accessor(
    feature_selection_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    feature_selection_cfg: OmegaConf,
) -> None:
    """The run's logged dataset input resolves through features/accessor.py::
    features_path() — the same canonical source every other step logs — not a
    second, hand-assembled copy that could silently go stale."""
    feature_selection_cfg.mlflow.tracking_uri = feature_selection_mlflow_uri
    X_dev, y_dev = dev_split

    feature_selection.run_feature_selection_step(
        X_dev, y_dev, feature_selection_cfg, cv_folds=2, cv_repeats=1, n_bootstrap=200
    )

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(_TEST_EXPERIMENT_NAME)
    run = client.get_run(client.search_runs([experiment.experiment_id])[0].info.run_id)

    dataset_inputs = run.inputs.dataset_inputs
    assert len(dataset_inputs) == 1
    source = json.loads(dataset_inputs[0].dataset.source)
    assert source["uri"] == str(features_path())


def test_run_feature_selection_step_sets_experiment_description(
    feature_selection_mlflow_uri: str,
    dev_split: tuple[pd.DataFrame, pd.Series],
    feature_selection_cfg: OmegaConf,
) -> None:
    """The review experiment (not just the run) carries an 'mlflow.note.content'
    description, set idempotently every call — the same self-healing pattern
    utils.mlflow.ensure_experiment_metadata uses for telco-churn-training, so a
    human landing on the experiment in the MLflow UI sees what it's for without
    opening a run."""
    feature_selection_cfg.mlflow.tracking_uri = feature_selection_mlflow_uri
    X_dev, y_dev = dev_split

    feature_selection.run_feature_selection_step(
        X_dev, y_dev, feature_selection_cfg, cv_folds=2, cv_repeats=1, n_bootstrap=200
    )

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(_TEST_EXPERIMENT_NAME)

    assert (
        experiment.tags.get("mlflow.note.content")
        == feature_selection._REVIEW_EXPERIMENT_DESCRIPTION
    )


def test_review_experiment_name_is_dedicated_and_separate_from_training() -> None:
    """The (unpatched) module constant is a fixed name distinct from
    cfg.mlflow.experiment_name's usual value — every review run adds a dated
    entry to its own experiment without cluttering the per-cycle training runs."""
    assert (
        feature_selection._REVIEW_EXPERIMENT_NAME
        == "telco-churn-feature-selection-review"
    )
    assert feature_selection._REVIEW_EXPERIMENT_NAME != "telco-churn-training"

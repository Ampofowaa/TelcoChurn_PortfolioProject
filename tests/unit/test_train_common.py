"""Unit tests for telco_churn.models.train.common — shared loaders/helpers."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from unittest.mock import Mock

import mlflow
import pandas as pd
import pandera as pa
import pytest
from lightgbm import LGBMClassifier
from omegaconf import OmegaConf
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline

import telco_churn.models.train.common as common
from telco_churn.features.accessor import features_path
from telco_churn.features.build import FEATURE_SCHEMA, TARGET_COL
from telco_churn.features.preprocessing import build_preprocessor
from telco_churn.models.train.common import cv_score_candidate

# ---------------------------------------------------------------------------
# _git_sha / _dvc_hash — audit-trail metadata resolution (QA finding #4:
# failures are logged, not silently swallowed into an indistinguishable
# 'unknown', since these values feed training_manifest.json's audit trail)
# ---------------------------------------------------------------------------


def test_git_sha_returns_stripped_output_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful git call returns the SHA with trailing whitespace stripped."""
    monkeypatch.setattr(common.subprocess, "check_output", lambda *a, **k: "abc123\n")
    assert common._git_sha() == "abc123"


def test_git_sha_returns_unknown_and_warns_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed git invocation returns 'unknown' and logs a warning — not a
    silent fallback indistinguishable from a working git call.
    """

    def _raise(*args: object, **kwargs: object) -> str:
        raise FileNotFoundError("git executable not found")

    monkeypatch.setattr(common.subprocess, "check_output", _raise)
    warning_mock = Mock()
    monkeypatch.setattr(common.logger, "warning", warning_mock)

    result = common._git_sha()

    assert result == "unknown"
    warning_mock.assert_called_once()
    assert warning_mock.call_args.args[0] == "git_sha_unavailable"
    assert warning_mock.call_args.kwargs["exc_info"] is True


def test_dvc_hash_returns_unknown_and_logs_debug_when_file_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A missing .dvc file (DVC not tracked yet — the routine pre-Phase-8 case)
    returns 'unknown' and logs at debug only, never a warning.
    """
    monkeypatch.setattr(common, "get_project_root", lambda: tmp_path)
    cfg = OmegaConf.create({"paths": {"processed_data": "."}})
    debug_mock = Mock()
    warning_mock = Mock()
    monkeypatch.setattr(common.logger, "debug", debug_mock)
    monkeypatch.setattr(common.logger, "warning", warning_mock)

    result = common._dvc_hash(cfg)

    assert result == "unknown"
    debug_mock.assert_called_once()
    assert debug_mock.call_args.args[0] == "dvc_hash_not_tracked"
    warning_mock.assert_not_called()


def test_dvc_hash_returns_hash_when_file_present(tmp_path: Path) -> None:
    """A well-formed .dvc file returns its recorded md5 hash."""
    dvc_file = tmp_path / "telco_churn_processed.csv.dvc"
    dvc_file.write_text("outs:\n- md5: deadbeef1234\n")
    cfg = OmegaConf.create({"paths": {"processed_data": str(tmp_path)}})

    assert common._dvc_hash(cfg) == "deadbeef1234"  # pragma: allowlist secret


def test_dvc_hash_returns_unknown_and_warns_on_malformed_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A .dvc file that exists but doesn't have the expected structure returns
    'unknown' and logs a warning — distinct from the missing-file debug case,
    since this signals a genuine problem worth a human look.
    """
    dvc_file = tmp_path / "telco_churn_processed.csv.dvc"
    dvc_file.write_text("not_outs: []\n")
    cfg = OmegaConf.create({"paths": {"processed_data": str(tmp_path)}})
    warning_mock = Mock()
    monkeypatch.setattr(common.logger, "warning", warning_mock)

    result = common._dvc_hash(cfg)

    assert result == "unknown"
    warning_mock.assert_called_once()
    assert warning_mock.call_args.args[0] == "dvc_hash_unexpected_error"
    assert warning_mock.call_args.kwargs["exc_info"] is True


# ---------------------------------------------------------------------------
# compose_config — Hydra config loading (D1)
#
# The only place the *real* configs/config.yaml + defaults tree (training,
# logreg, selection, tuning) actually gets composed and read, catching a renamed
# or missing YAML key that a hand-built test cfg would never surface.
# ---------------------------------------------------------------------------


def test_compose_config_loads_expected_structure() -> None:
    """compose_config() resolves the full defaults tree with the values train.py relies on."""
    from telco_churn.utils.paths import compose_config

    cfg = compose_config()

    assert str(cfg.mlflow.registered_model_name) == "telco-churn-pipeline"
    assert str(cfg.mlflow.experiment_name) == "telco-churn-training"
    assert int(cfg.training_setup.cv_folds) == 10
    assert int(cfg.training_setup.cv_repeats) == 10
    assert float(cfg.training_setup.delta_threshold) == pytest.approx(0.005)
    assert int(cfg.training.candidate.n_estimators) == 100
    assert int(cfg.logreg.Cs) == 10
    assert int(cfg.selection.n_repeats) == 15
    assert float(cfg.selection.noise_floor_margin) == pytest.approx(0.005)
    assert int(cfg.tuning.n_trials) == 50
    assert str(cfg.tuning.selection_rule) == "1se"


def test_compose_config_accepts_cli_overrides() -> None:
    """compose_config(overrides=[...]) applies Hydra-style key=value overrides —
    the mechanism the train.py __main__ subprocess integration test relies on to
    run a fast, reduced-scale pipeline without touching any production YAML.
    """
    from telco_churn.utils.paths import compose_config

    cfg = compose_config(overrides=["tuning.n_trials=3"])

    assert int(cfg.tuning.n_trials) == 3


# ---------------------------------------------------------------------------
# _load_processed — schema guard (B3a)
# ---------------------------------------------------------------------------


def test_load_processed_raises_on_schema_violation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, feature_df: pd.DataFrame
) -> None:
    """A processed CSV that violates FeatureOutputSchema aborts the load."""
    bad_df = feature_df.copy()
    bad_df.loc[0, "contract_type"] = "Not-a-real-contract"
    bad_df.to_csv(tmp_path / "telco_churn_processed.csv", index=False)
    monkeypatch.setattr(common, "get_project_root", lambda: tmp_path)
    cfg = OmegaConf.create({"paths": {"processed_data": "."}})

    with pytest.raises(pa.errors.SchemaError):
        common._load_processed(cfg)


def test_load_processed_passes_on_clean_data(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, feature_df: pd.DataFrame
) -> None:
    """A schema-conformant processed CSV loads without error."""
    feature_df.to_csv(tmp_path / "telco_churn_processed.csv", index=False)
    monkeypatch.setattr(common, "get_project_root", lambda: tmp_path)
    cfg = OmegaConf.create({"paths": {"processed_data": "."}})

    df = common._load_processed(cfg)

    assert len(df) == len(feature_df)


# ---------------------------------------------------------------------------
# _load_dev_features
# ---------------------------------------------------------------------------
#
# Split reproducibility/integrity (determinism, disjointness, stratification) is
# owned by test_split.py — this file only asserts train.py imports the canonical
# split correctly and never reads the test partition (CLAUDE.md modelling
# invariants: 'test set touched once').


class _PoisonFrame:
    """Stand-in for the test partition — raises if ever read.

    Passed as the second element of partition()'s return value in tests below, so
    any code path that touches the test frame fails loudly instead of silently
    leaking test rows into training.
    """

    def __getattr__(self, name: str) -> None:
        raise AssertionError(f"test partition was accessed via .{name}")

    def __getitem__(self, key: object) -> None:
        raise AssertionError("test partition was accessed via __getitem__")


def test_load_dev_features_returns_only_dev_rows(
    monkeypatch: pytest.MonkeyPatch, feature_df: pd.DataFrame
) -> None:
    """_load_dev_features returns exactly the dev-partition rows and feature columns."""
    dev_df = feature_df.iloc[:96].reset_index(drop=True)
    test_df = feature_df.iloc[96:].reset_index(drop=True)
    monkeypatch.setattr(common, "_load_processed", lambda cfg: feature_df)
    monkeypatch.setattr(common, "partition", lambda df: (dev_df, test_df))

    X_dev, y_dev = common._load_dev_features(cfg=None)

    assert len(X_dev) == len(dev_df) == len(y_dev)
    assert set(X_dev.columns) == set(common._FEATURE_COLS)
    assert "customerid" not in X_dev.columns


def test_load_dev_features_never_touches_test_partition(
    monkeypatch: pytest.MonkeyPatch, feature_df: pd.DataFrame
) -> None:
    """The test-partition frame returned by partition() must never be read."""
    dev_df = feature_df.iloc[:96].reset_index(drop=True)
    monkeypatch.setattr(common, "_load_processed", lambda cfg: feature_df)
    monkeypatch.setattr(common, "partition", lambda df: (dev_df, _PoisonFrame()))

    common._load_dev_features(cfg=None)  # raises AssertionError if test_df is touched


# ---------------------------------------------------------------------------
# _log_dev_input (Fix 6: dataset lineage shared by comparison.py,
# feature_freeze.py, and tuning.py — candidates.py builds its own copy since
# each of its three candidate runs needs its own log_input call)
# ---------------------------------------------------------------------------


@pytest.fixture
def dev_input_mlflow_uri(mlflow_test_experiment: Callable[[str], str]) -> str:
    """Point MLflow at the shared tmp-scoped experiment (conftest.py ::
    mlflow_test_experiment)."""
    return mlflow_test_experiment("test_log_dev_input")


def test_log_dev_input_source_resolves_to_accessor_canonical_path(
    dev_input_mlflow_uri: str, dev_split: tuple[pd.DataFrame, pd.Series]
) -> None:
    """The logged dataset's source resolves to features_path() — the
    accessor's single canonical location — not a hand-assembled copy that
    could silently go stale at Phase 8's CSV -> parquet rename. This is the
    one test of the shared implementation; comparison.py/feature_freeze.py/
    tuning.py's own tests only need to confirm they call this, not re-prove
    the source resolution themselves.
    """
    mlflow.set_tracking_uri(dev_input_mlflow_uri)
    X_dev, y_dev = dev_split

    with mlflow.start_run() as run:
        common._log_dev_input(X_dev, y_dev, context="training")

    full_run = mlflow.get_run(run.info.run_id)
    dataset_inputs = full_run.inputs.dataset_inputs
    assert len(dataset_inputs) == 1
    assert dataset_inputs[0].dataset.name == "telco_churn_dev"
    source = json.loads(dataset_inputs[0].dataset.source)
    assert source["uri"] == str(features_path())

    input_tags = {t.key: t.value for t in dataset_inputs[0].tags}
    assert input_tags.get("mlflow.data.context") == "training"


def test_log_dev_input_dataset_includes_target_column(
    dev_input_mlflow_uri: str, dev_split: tuple[pd.DataFrame, pd.Series]
) -> None:
    """The logged dataset's schema carries the target column, not features only —
    matching candidates.py's own dev_df[TARGET_COL] = y_dev.values construction.
    """
    mlflow.set_tracking_uri(dev_input_mlflow_uri)
    X_dev, y_dev = dev_split

    with mlflow.start_run() as run:
        common._log_dev_input(X_dev, y_dev, context="training")

    full_run = mlflow.get_run(run.info.run_id)
    schema = full_run.inputs.dataset_inputs[0].dataset.schema
    column_names = {col["name"] for col in json.loads(schema)["mlflow_colspec"]}
    assert TARGET_COL in column_names
    assert set(X_dev.columns) <= column_names


# ---------------------------------------------------------------------------
# cv_score_candidate
# ---------------------------------------------------------------------------


def test_cv_score_candidate_return_keys(
    dev_split: tuple[pd.DataFrame, pd.Series],
) -> None:
    """All expected keys are present in the result dict."""
    X_dev, y_dev = dev_split
    cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=2, random_state=42)
    result = cv_score_candidate(
        DummyClassifier(strategy="prior", random_state=42), X_dev, y_dev, cv
    )
    assert set(result) == {
        "pr_auc_mean",
        "pr_auc_std",
        "pr_auc_scores",
        "oof_proba",
        "oof_true",
        "train_time_s",
        "predict_time_s",
    }


def test_cv_score_candidate_fold_count(
    dev_split: tuple[pd.DataFrame, pd.Series],
) -> None:
    """pr_auc_scores length equals n_splits × n_repeats."""
    X_dev, y_dev = dev_split
    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=42)
    result = cv_score_candidate(
        DummyClassifier(strategy="prior", random_state=42), X_dev, y_dev, cv
    )
    assert len(result["pr_auc_scores"]) == 6


def test_cv_score_candidate_oof_length(
    dev_split: tuple[pd.DataFrame, pd.Series],
) -> None:
    """OOF arrays are aligned to the full dev set (one entry per row)."""
    X_dev, y_dev = dev_split
    cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=2, random_state=42)
    result = cv_score_candidate(
        DummyClassifier(strategy="prior", random_state=42), X_dev, y_dev, cv
    )
    assert len(result["oof_proba"]) == len(X_dev)
    assert len(result["oof_true"]) == len(X_dev)


def test_cv_score_candidate_oof_proba_range(
    dev_split: tuple[pd.DataFrame, pd.Series],
) -> None:
    """OOF probabilities must be in [0, 1]."""
    X_dev, y_dev = dev_split
    cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=2, random_state=42)
    result = cv_score_candidate(
        DummyClassifier(strategy="prior", random_state=42), X_dev, y_dev, cv
    )
    proba = result["oof_proba"]
    assert all(0.0 <= p <= 1.0 for p in proba)


# ---------------------------------------------------------------------------
# Run-twice determinism (D1) — verifies the determinism block (deterministic=True,
# force_row_wise=True, pinned n_jobs=1, random_state=42), distinct from the C2
# serialization round-trip.
# ---------------------------------------------------------------------------


def _build_lgbm_candidate_pipeline(
    binary: list[str], multi_cat: list[str], numeric: list[str]
) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(binary, multi_cat, numeric)),
            (
                "model",
                LGBMClassifier(
                    n_estimators=20,
                    num_leaves=10,
                    min_child_samples=5,
                    class_weight="balanced",
                    subsample_freq=1,
                    deterministic=True,
                    force_row_wise=True,
                    n_jobs=1,
                    verbose=-1,
                    random_state=42,
                ),
            ),
        ]
    )


def test_lgbm_candidate_run_twice_is_bit_identical(
    dev_split: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Two independent LightGBM candidate fits under the determinism block produce
    bit-identical per-fold PR-AUC scores and OOF probabilities — not just
    statistically close ones.
    """
    X_dev, y_dev = dev_split
    binary = list(FEATURE_SCHEMA.binary)
    multi_cat = list(FEATURE_SCHEMA.multi_cat)
    numeric = list(FEATURE_SCHEMA.numeric)

    cv1 = RepeatedStratifiedKFold(n_splits=2, n_repeats=2, random_state=42)
    result1 = cv_score_candidate(
        _build_lgbm_candidate_pipeline(binary, multi_cat, numeric), X_dev, y_dev, cv1
    )

    cv2 = RepeatedStratifiedKFold(n_splits=2, n_repeats=2, random_state=42)
    result2 = cv_score_candidate(
        _build_lgbm_candidate_pipeline(binary, multi_cat, numeric), X_dev, y_dev, cv2
    )

    assert result1["pr_auc_scores"] == result2["pr_auc_scores"]
    assert result1["oof_proba"] == result2["oof_proba"]


def test_cv_score_candidate_parallel_matches_sequential(
    dev_split: tuple[pd.DataFrame, pd.Series],
) -> None:
    """n_jobs=2 returns bit-identical results to n_jobs=1 — folds are independent,
    so running them concurrently changes wall-clock time only, never the scores.
    """
    X_dev, y_dev = dev_split
    binary = list(FEATURE_SCHEMA.binary)
    multi_cat = list(FEATURE_SCHEMA.multi_cat)
    numeric = list(FEATURE_SCHEMA.numeric)

    cv1 = RepeatedStratifiedKFold(n_splits=2, n_repeats=2, random_state=42)
    sequential = cv_score_candidate(
        _build_lgbm_candidate_pipeline(binary, multi_cat, numeric),
        X_dev,
        y_dev,
        cv1,
        n_jobs=1,
    )

    cv2 = RepeatedStratifiedKFold(n_splits=2, n_repeats=2, random_state=42)
    parallel = cv_score_candidate(
        _build_lgbm_candidate_pipeline(binary, multi_cat, numeric),
        X_dev,
        y_dev,
        cv2,
        n_jobs=2,
    )

    assert sequential["pr_auc_scores"] == parallel["pr_auc_scores"]
    assert sequential["oof_proba"] == parallel["oof_proba"]

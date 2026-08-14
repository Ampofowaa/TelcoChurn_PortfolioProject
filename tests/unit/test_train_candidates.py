"""Unit tests for telco_churn.models.train.candidates — Step 1 (B2, B3a/b)."""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock

import pandas as pd
import pytest
from omegaconf import OmegaConf

import telco_churn.models.train.candidates as candidates
import telco_churn.models.train.common as common
from telco_churn.features.accessor import features_path

_FAKE_DATA_CONTENT_HASH = "deadbeef" * 8


@pytest.fixture(scope="module", autouse=True)
def _stub_data_content_hash() -> Iterator[None]:
    """run_candidate_step stamps data_content_hash via features_sha256() with
    no path override, resolving to the real, gitignored
    datasets/processed/telco_churn_features.parquet — absent on a fresh
    checkout. Every test below trains on synthetic fixtures, never real
    processed data, so stub the hash.
    """
    mp = pytest.MonkeyPatch()
    mp.setattr(candidates, "features_sha256", lambda path=None: _FAKE_DATA_CONTENT_HASH)
    yield
    mp.undo()


# ---------------------------------------------------------------------------
# _assert_dummy_canary (B3b)
# ---------------------------------------------------------------------------


def test_assert_dummy_canary_passes_on_chance_level() -> None:
    """A dummy result at chance level does not raise."""
    y_dev = pd.Series([0, 1] * 50)
    dummy_result = {
        "oof_true": y_dev.tolist(),
        "oof_proba": [0.5] * 100,
        "pr_auc_mean": float(y_dev.mean()),
    }
    candidates._assert_dummy_canary(dummy_result, y_dev)  # must not raise


def test_assert_dummy_canary_raises_on_above_chance_roc_auc() -> None:
    """An above-chance dummy ROC-AUC trips the leakage canary."""
    y_dev = pd.Series([0] * 50 + [1] * 50)
    dummy_result = {
        "oof_true": y_dev.tolist(),
        "oof_proba": [0.1] * 50 + [0.9] * 50,  # perfectly separates — not chance
        "pr_auc_mean": float(y_dev.mean()),
    }
    with pytest.raises(AssertionError, match="ROC-AUC"):
        candidates._assert_dummy_canary(dummy_result, y_dev)


def test_assert_dummy_canary_raises_on_pr_auc_prevalence_mismatch() -> None:
    """A dummy PR-AUC far from prevalence trips the leakage canary."""
    y_dev = pd.Series([0] * 90 + [1] * 10)  # prevalence 0.10
    dummy_result = {
        "oof_true": y_dev.tolist(),
        "oof_proba": [0.5] * 100,  # chance-level ROC-AUC
        "pr_auc_mean": 0.50,  # far from prevalence 0.10
    }
    with pytest.raises(AssertionError, match="PR-AUC"):
        candidates._assert_dummy_canary(dummy_result, y_dev)


# ---------------------------------------------------------------------------
# run_candidate_step — metric-logging contract (D1, mocked MLflow client)
# ---------------------------------------------------------------------------


class _FakeRunInfo:
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id


class _FakeRun:
    """Minimal context-manager stand-in for mlflow.start_run()."""

    def __init__(self, run_id: str) -> None:
        self.info = _FakeRunInfo(run_id)

    def __enter__(self) -> _FakeRun:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def test_run_candidate_step_logs_metric_contract(
    monkeypatch: pytest.MonkeyPatch,
    dev_split: tuple[pd.DataFrame, pd.Series],
) -> None:
    """run_candidate_step logs the documented param/metric keys for every candidate
    — the metric-logging contract — against a mocked MLflow client, so this needs
    no tracking backend at all.
    """
    X_dev, y_dev = dev_split
    cfg = OmegaConf.create(
        {
            "random_seed": 42,
            "training_setup": {
                "cv_folds": 2,
                "cv_repeats": 1,
                "class_weight": "balanced",
            },
            "logreg": {
                "Cs": 2,
                "cv_folds": 2,
                "max_iter": 200,
                "solver": "lbfgs",
                "l1_ratio": 0.0,
            },
            "training": {
                "candidate": {
                    "n_estimators": 10,
                    "num_leaves": 5,
                    "min_child_samples": 3,
                },
                "fixed": {
                    "subsample_freq": 1,
                    "deterministic": True,
                    "force_row_wise": True,
                    "n_jobs": 1,
                    "verbose": -1,
                },
            },
            "mlflow": {"tracking_uri": "placeholder", "experiment_name": "test"},
            "paths": {"processed_data": "."},
        }
    )

    logged_params: list[dict] = []
    logged_metrics: list[dict] = []
    run_counter = iter(range(100))

    monkeypatch.setattr(candidates.mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(
        candidates.mlflow, "set_experiment", lambda name: MagicMock(experiment_id="0")
    )
    monkeypatch.setattr(
        candidates.mlflow.tracking,
        "MlflowClient",
        lambda: MagicMock(set_experiment_tag=lambda *a, **k: None),
    )
    monkeypatch.setattr(
        candidates.mlflow,
        "start_run",
        lambda run_name: _FakeRun(f"run-{next(run_counter)}"),
    )
    monkeypatch.setattr(candidates.mlflow, "set_tags", lambda tags: None)
    monkeypatch.setattr(candidates.mlflow, "log_input", lambda *a, **k: None)
    monkeypatch.setattr(
        candidates.mlflow, "log_params", lambda params: logged_params.append(params)
    )
    monkeypatch.setattr(
        candidates.mlflow, "log_metrics", lambda metrics: logged_metrics.append(metrics)
    )
    monkeypatch.setattr(candidates.mlflow, "log_metric", lambda *a, **k: None)
    monkeypatch.setattr(common, "mlflow_from_pandas", lambda *a, **k: MagicMock())
    monkeypatch.setattr(candidates, "_git_sha", lambda: "deadbeef")

    results = candidates.run_candidate_step(
        X_dev,
        y_dev,
        cfg,
        cv_folds=int(cfg.training_setup.cv_folds),
        cv_repeats=int(cfg.training_setup.cv_repeats),
    )

    assert set(results) == {"dummy_prior", "logreg_cv", "lgbm_default"}
    logged_candidates = {p["candidate"] for p in logged_params if "candidate" in p}
    assert logged_candidates == {"dummy_prior", "logreg_cv", "lgbm_default"}

    # logreg_cv/lgbm_default each get a second log_params call for their
    # model-construction kwargs (configs/training/logreg.yaml,lightgbm.yaml) —
    # class_weight is deliberately excluded there, already logged generically above.
    model_param_calls = [p for p in logged_params if "candidate" not in p]
    assert len(model_param_calls) == 2
    for params in model_param_calls:
        assert "class_weight" not in params
    logreg_model_params = next(p for p in model_param_calls if "solver" in p)
    assert logreg_model_params["Cs"] == 2
    lgbm_model_params = next(p for p in model_param_calls if "num_leaves" in p)
    assert lgbm_model_params["n_estimators"] == 10
    for metrics in logged_metrics:
        assert {
            "cv_pr_auc_mean",
            "cv_pr_auc_std",
            "cv_train_time_s",
            "cv_predict_time_s",
        } <= set(metrics)


def test_run_candidate_step_dataset_source_uses_accessor_canonical_path(
    monkeypatch: pytest.MonkeyPatch,
    dev_split: tuple[pd.DataFrame, pd.Series],
) -> None:
    """The mlflow.data dataset's source resolves to features_path() — the
    accessor's single canonical location — not a second, hand-assembled copy
    of it. A hardcoded inline path silently goes stale at any rename of the
    processed-artifact location (e.g. Phase 8's CSV -> parquet swap).
    """
    X_dev, y_dev = dev_split
    cfg = OmegaConf.create(
        {
            "random_seed": 42,
            "training_setup": {
                "cv_folds": 2,
                "cv_repeats": 1,
                "class_weight": "balanced",
            },
            "logreg": {
                "Cs": 2,
                "cv_folds": 2,
                "max_iter": 200,
                "solver": "lbfgs",
                "l1_ratio": 0.0,
            },
            "training": {
                "candidate": {
                    "n_estimators": 10,
                    "num_leaves": 5,
                    "min_child_samples": 3,
                },
                "fixed": {
                    "subsample_freq": 1,
                    "deterministic": True,
                    "force_row_wise": True,
                    "n_jobs": 1,
                    "verbose": -1,
                },
            },
            "mlflow": {"tracking_uri": "placeholder", "experiment_name": "test"},
            "paths": {"processed_data": "."},
        }
    )
    run_counter = iter(range(100))
    captured_sources: list[str] = []

    def _fake_from_pandas(df: pd.DataFrame, **kwargs: object) -> MagicMock:
        captured_sources.append(str(kwargs["source"]))
        return MagicMock()

    monkeypatch.setattr(candidates.mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(
        candidates.mlflow, "set_experiment", lambda name: MagicMock(experiment_id="0")
    )
    monkeypatch.setattr(
        candidates.mlflow.tracking,
        "MlflowClient",
        lambda: MagicMock(set_experiment_tag=lambda *a, **k: None),
    )
    monkeypatch.setattr(
        candidates.mlflow,
        "start_run",
        lambda run_name: _FakeRun(f"run-{next(run_counter)}"),
    )
    monkeypatch.setattr(candidates.mlflow, "set_tags", lambda tags: None)
    monkeypatch.setattr(candidates.mlflow, "log_input", lambda *a, **k: None)
    monkeypatch.setattr(candidates.mlflow, "log_params", lambda params: None)
    monkeypatch.setattr(candidates.mlflow, "log_metrics", lambda metrics: None)
    monkeypatch.setattr(candidates.mlflow, "log_metric", lambda *a, **k: None)
    monkeypatch.setattr(common, "mlflow_from_pandas", _fake_from_pandas)
    monkeypatch.setattr(candidates, "_git_sha", lambda: "deadbeef")

    candidates.run_candidate_step(
        X_dev,
        y_dev,
        cfg,
        cv_folds=int(cfg.training_setup.cv_folds),
        cv_repeats=int(cfg.training_setup.cv_repeats),
    )

    assert captured_sources == [str(features_path())]

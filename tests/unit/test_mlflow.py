"""Unit tests for telco_churn.utils.mlflow."""

from __future__ import annotations

import re
from collections.abc import Callable
from urllib.parse import urlparse

import mlflow
import mlflow.sklearn
import mlflow.tracking
import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import LogisticRegression

import telco_churn.utils.mlflow as tc_mlflow
from telco_churn.utils.mlflow import (
    ensure_experiment_metadata,
    load_model_promotion_bars,
    resolve_logged_model_id,
    resolve_model_run_id,
    resolve_tracking_uri,
    set_logged_model_description,
    set_registered_model_description,
    set_run_description,
)
from telco_churn.utils.paths import get_project_root


def test_resolve_tracking_uri_passes_through_http() -> None:
    """An http(s) URI (Docker / remote MLflow server) is returned unchanged."""
    assert resolve_tracking_uri("http://localhost:5000") == "http://localhost:5000"


def test_resolve_tracking_uri_passes_through_sqlite() -> None:
    """A sqlite:// URI is a real backend scheme, not a bare relative path — must
    not be mangled by anchoring it under the project root (regression: this used
    to break every non-HTTP MLflow backend, including the sqlite fixture used
    throughout the models/train/* test suite).
    """
    uri = "sqlite:///C:/tmp/mlflow.db"
    assert resolve_tracking_uri(uri) == uri


def test_resolve_tracking_uri_anchors_bare_relative_path() -> None:
    """A bare relative path like 'mlruns' is anchored to the project root and
    returned as a file:// URI — not a bare OS path, which on Windows is
    misread as scheme 'c' (the drive letter) by MLflow's store registry.
    """
    resolved = resolve_tracking_uri("mlruns")
    assert urlparse(resolved).scheme == "file"
    assert resolved == (get_project_root() / "mlruns").as_uri()


def test_resolve_tracking_uri_anchors_relative_sqlite_path() -> None:
    """A relative sqlite:/// path contains '://' like any other scheme, but the
    path after the prefix is CWD-relative, not scheme-anchored — the same
    failure mode this function exists to prevent for a bare 'mlruns' path,
    reintroduced through a different scheme. It must be anchored to the
    project root, not passed through.
    """
    resolved = resolve_tracking_uri("sqlite:///mlflow.db")
    expected = "sqlite:///" + (get_project_root() / "mlflow.db").as_posix()
    assert resolved == expected


def test_resolve_tracking_uri_passes_through_unix_absolute_sqlite() -> None:
    """A Unix-style absolute sqlite path (sqlite:////abs/path) is already
    anchored and must not be re-rooted under the project.
    """
    uri = "sqlite:////tmp/mlflow.db"
    assert resolve_tracking_uri(uri) == uri


def test_config_default_tracking_uri_never_resolves_to_a_file_store() -> None:
    """Regression guard for the failure mode nobody sees locally: without
    .env or the infra profile up, configs/config.yaml's fallback must resolve
    to a database backend (sqlite), never MLflow's file store — which is in
    maintenance mode as of MLflow 3.14 and raises on use.
    """
    default_uri = "sqlite:///mlflow.db"
    resolved = resolve_tracking_uri(default_uri)
    assert urlparse(resolved).scheme == "sqlite"


# ---------------------------------------------------------------------------
# Shared registry fixture (real tmp-scoped SQLite store, not a mock)
# ---------------------------------------------------------------------------


@pytest.fixture
def registry_cfg(
    mlflow_test_experiment: Callable[[str], str], request: pytest.FixtureRequest
) -> DictConfig:
    """cfg carrying the mlflow.* keys resolve_model_run_id/resolve_logged_model_id/
    ensure_experiment_metadata all read, against a tmp-scoped SQLite MLflow store
    (conftest.py::mlflow_test_experiment) — a real registry, not a mocked client,
    since these functions are thin wrappers around real MlflowClient calls whose
    contract only a real store can verify.
    """
    tracking_uri = mlflow_test_experiment("test_utils_mlflow")
    registered_model_name = "test-mlflow-utils-" + re.sub(
        r"[^A-Za-z0-9_-]", "_", request.node.name
    )
    return OmegaConf.create(
        {
            "mlflow": {
                "tracking_uri": tracking_uri,
                "experiment_name": "test_utils_mlflow",
                "registered_model_name": registered_model_name,
            }
        }
    )


def _log_and_register(registry_cfg: DictConfig) -> tuple[str, str]:
    """Fit a trivial LogisticRegression, log it, and register it under
    registry_cfg's model name. Returns (version, model_id)."""
    mlflow.set_tracking_uri(str(registry_cfg.mlflow.tracking_uri))
    registered_name = str(registry_cfg.mlflow.registered_model_name)
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    model = LogisticRegression().fit(X, [0, 1, 0, 1])
    with mlflow.start_run():
        model_info = mlflow.sklearn.log_model(
            sk_model=model, name="model", registered_model_name=registered_name
        )
    return str(model_info.registered_model_version), str(model_info.model_id)


# ---------------------------------------------------------------------------
# resolve_model_run_id
# ---------------------------------------------------------------------------


def test_resolve_model_run_id_returns_registering_run(
    registry_cfg: DictConfig,
) -> None:
    mlflow.set_tracking_uri(str(registry_cfg.mlflow.tracking_uri))
    registered_name = str(registry_cfg.mlflow.registered_model_name)
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    model = LogisticRegression().fit(X, [0, 1, 0, 1])
    with mlflow.start_run() as run:
        expected_run_id = run.info.run_id
        model_info = mlflow.sklearn.log_model(
            sk_model=model, name="model", registered_model_name=registered_name
        )
    version = str(model_info.registered_model_version)

    resolved_run_id = resolve_model_run_id(version, registry_cfg)

    assert resolved_run_id == expected_run_id


# ---------------------------------------------------------------------------
# resolve_logged_model_id
# ---------------------------------------------------------------------------


def test_resolve_logged_model_id_returns_tagged_id(registry_cfg: DictConfig) -> None:
    version, model_id = _log_and_register(registry_cfg)
    registered_name = str(registry_cfg.mlflow.registered_model_name)
    mlflow.tracking.MlflowClient().set_model_version_tag(
        registered_name, version, "logged_model_id", model_id
    )

    resolved = resolve_logged_model_id(version, registry_cfg)

    assert resolved == model_id


def test_resolve_logged_model_id_raises_when_tag_missing(
    registry_cfg: DictConfig,
) -> None:
    """calibrate.py's registration step is the only writer of this tag — a
    version registered without it means calibrate.py never ran against this
    registry, and the caller (evaluate.py/threshold.py) needs a clear signal
    to re-run it rather than a silent KeyError two calls downstream."""
    version, _model_id = _log_and_register(registry_cfg)

    with pytest.raises(ValueError, match="no logged_model_id tag"):
        resolve_logged_model_id(version, registry_cfg)


# ---------------------------------------------------------------------------
# load_model_promotion_bars
# ---------------------------------------------------------------------------


def test_load_model_promotion_bars_matches_yaml_file() -> None:
    """Cross-check against the real configs/model_promotion.yaml — the
    ANALYSIS.md §0 pre-registered gate policy — rather than hardcoding the
    bar values, so this test doesn't go stale (or silently start asserting
    the wrong thing) the moment the policy is revised.
    """
    cfg = OmegaConf.create(
        {"paths": {"model_promotion_config": "configs/model_promotion.yaml"}}
    )
    raw = OmegaConf.load(get_project_root() / "configs" / "model_promotion.yaml")

    bars = load_model_promotion_bars(cfg)

    assert bars.pr_auc_bar == pytest.approx(float(raw.pr_auc_bar))
    assert bars.recall_bar == pytest.approx(float(raw.recall_bar))
    lo, hi = bars.calibration_slope_band
    assert lo == pytest.approx(float(raw.calibration_slope_band[0]))
    assert hi == pytest.approx(float(raw.calibration_slope_band[1]))
    assert bars.pr_auc_materiality_threshold == pytest.approx(
        float(raw.pr_auc_materiality_threshold)
    )
    assert bars.brier_non_inferiority_margin == pytest.approx(
        float(raw.brier_non_inferiority_margin)
    )


# ---------------------------------------------------------------------------
# ensure_experiment_metadata
# ---------------------------------------------------------------------------


def test_ensure_experiment_metadata_sets_description_and_tags(
    registry_cfg: DictConfig,
) -> None:
    exp = ensure_experiment_metadata(registry_cfg)

    client = mlflow.tracking.MlflowClient()
    refreshed = client.get_experiment(exp.experiment_id)
    assert refreshed.tags["mlflow.note.content"] == tc_mlflow._EXPERIMENT_DESCRIPTION
    for tag_key, tag_val in tc_mlflow._EXPERIMENT_TAGS.items():
        assert refreshed.tags[tag_key] == tag_val


def test_ensure_experiment_metadata_is_idempotent(registry_cfg: DictConfig) -> None:
    """Calling twice (e.g. once each from candidates.py/calibrate.py/evaluate.py/
    error_analysis.py in the same cycle) must not raise and must leave the same
    description/tags in place, per the self-healing pattern its docstring claims."""
    ensure_experiment_metadata(registry_cfg)
    exp = ensure_experiment_metadata(registry_cfg)

    client = mlflow.tracking.MlflowClient()
    refreshed = client.get_experiment(exp.experiment_id)
    assert refreshed.tags["mlflow.note.content"] == tc_mlflow._EXPERIMENT_DESCRIPTION


# ---------------------------------------------------------------------------
# set_run_description
# ---------------------------------------------------------------------------


def test_set_run_description_sets_note_content_tag(registry_cfg: DictConfig) -> None:
    mlflow.set_tracking_uri(str(registry_cfg.mlflow.tracking_uri))
    with mlflow.start_run() as run:
        set_run_description("a training-cycle run")
        run_id = run.info.run_id

    full_run = mlflow.get_run(run_id)
    assert full_run.data.tags["mlflow.note.content"] == "a training-cycle run"


# ---------------------------------------------------------------------------
# set_registered_model_description
# ---------------------------------------------------------------------------


def test_set_registered_model_description_sets_description(
    registry_cfg: DictConfig,
) -> None:
    mlflow.set_tracking_uri(str(registry_cfg.mlflow.tracking_uri))
    name = str(registry_cfg.mlflow.registered_model_name)
    mlflow.tracking.MlflowClient().create_registered_model(name)

    set_registered_model_description(name, "a churn model")

    refreshed = mlflow.tracking.MlflowClient().get_registered_model(name)
    assert refreshed.description == "a churn model"


# ---------------------------------------------------------------------------
# set_logged_model_description
# ---------------------------------------------------------------------------


def test_set_logged_model_description_sets_note_content_tag(
    registry_cfg: DictConfig,
) -> None:
    _version, model_id = _log_and_register(registry_cfg)

    set_logged_model_description(model_id, "calibrated churn model")

    logged_model = mlflow.get_logged_model(model_id)
    assert logged_model.tags["mlflow.note.content"] == "calibrated churn model"

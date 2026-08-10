"""Unit tests for telco_churn.models.artifacts."""

from __future__ import annotations

from collections.abc import Callable

import mlflow.tracking
import pytest
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR
from omegaconf import OmegaConf

from telco_churn.models.artifacts import resolve_champion_version

# ---------------------------------------------------------------------------
# resolve_champion_version
# ---------------------------------------------------------------------------


def test_resolve_champion_version_returns_none_when_model_never_registered(
    mlflow_test_experiment: Callable[[str], str],
) -> None:
    """A genuinely never-registered model raises RESOURCE_DOES_NOT_EXIST — the
    first of the two real cold-start shapes MLflow's SqlAlchemy-backed
    registry reports — and must resolve to None.
    """
    tracking_uri = mlflow_test_experiment("test_resolve_champion_version")
    cfg = OmegaConf.create(
        {
            "mlflow": {
                "tracking_uri": tracking_uri,
                "registered_model_name": "never-registered-model",
            }
        }
    )
    assert resolve_champion_version(cfg) is None


def test_resolve_champion_version_returns_none_when_alias_never_set(
    mlflow_test_experiment: Callable[[str], str],
) -> None:
    """A registered model with no champion alias ever set raises
    INVALID_PARAMETER_VALUE, not RESOURCE_DOES_NOT_EXIST — the second real
    cold-start shape — and must also resolve to None.
    """
    tracking_uri = mlflow_test_experiment("test_resolve_champion_version")
    registered_model_name = "registered-no-alias"
    mlflow.tracking.MlflowClient().create_registered_model(registered_model_name)
    cfg = OmegaConf.create(
        {
            "mlflow": {
                "tracking_uri": tracking_uri,
                "registered_model_name": registered_model_name,
            }
        }
    )
    assert resolve_champion_version(cfg) is None


def test_resolve_champion_version_reraises_non_not_found_mlflow_errors(
    monkeypatch: pytest.MonkeyPatch, mlflow_test_experiment: Callable[[str], str]
) -> None:
    """A transient/auth/server MLflow failure must propagate, not be read as
    'no champion' — silently swallowing it would flip evaluate.py from the
    comparative gate regime to cold-start against a perfectly healthy
    incumbent.
    """

    def _raise_transient(
        self: mlflow.tracking.MlflowClient, name: str, alias: str
    ) -> None:
        raise MlflowException("temporarily unavailable", error_code=INTERNAL_ERROR)

    monkeypatch.setattr(
        mlflow.tracking.MlflowClient, "get_model_version_by_alias", _raise_transient
    )
    tracking_uri = mlflow_test_experiment("test_resolve_champion_version")
    cfg = OmegaConf.create(
        {
            "mlflow": {
                "tracking_uri": tracking_uri,
                "registered_model_name": "telco-churn-pipeline",
            }
        }
    )
    with pytest.raises(MlflowException, match="temporarily unavailable"):
        resolve_champion_version(cfg)

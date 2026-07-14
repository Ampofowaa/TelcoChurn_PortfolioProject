"""Unit tests for telco_churn.utils.mlflow — tracking-URI resolution."""

from __future__ import annotations

from urllib.parse import urlparse

from telco_churn.utils.mlflow import resolve_tracking_uri
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

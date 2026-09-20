"""Unit tests for telco_churn.utils.db engine factory."""

import pytest

import telco_churn.utils.db as db_module
from telco_churn.utils.db import get_engine, host_from_url


@pytest.fixture(autouse=True)
def reset_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset the module-level _engine singleton before each test."""
    monkeypatch.setattr(db_module, "_engine", None)


def test_get_engine_raises_when_url_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_engine raises OSError when POSTGRES_URL is not set."""
    monkeypatch.delenv("POSTGRES_URL", raising=False)
    with pytest.raises(OSError, match="POSTGRES_URL"):
        get_engine()


def test_get_engine_returns_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated calls return the same engine object without creating a new connection."""
    monkeypatch.setenv(
        "POSTGRES_URL",
        "postgresql://user:pass@localhost:5432/test",  # pragma: allowlist secret
    )
    engine_one = get_engine()
    engine_two = get_engine()
    assert engine_one is engine_two


def test_host_from_url_extracts_rds_hostname() -> None:
    """A real RDS endpoint's hostname is returned, password stripped."""
    url = "postgresql://telco_admin:s3cr3t@telco-churn-db.cct88is8iujn.us-east-1.rds.amazonaws.com:5432/mlflow"  # pragma: allowlist secret
    assert (
        host_from_url(url) == "telco-churn-db.cct88is8iujn.us-east-1.rds.amazonaws.com"
    )


def test_host_from_url_extracts_localhost() -> None:
    """A local Postgres URL's host is returned as-is."""
    url = (
        "postgresql://user:pass@localhost:5432/telco_churn"  # pragma: allowlist secret
    )
    assert host_from_url(url) == "localhost"


def test_host_from_url_sqlite_returns_local() -> None:
    """A file-based backend (sqlite, MLFLOW_TRACKING_URI's fallback) has no
    host — resolved as 'local' rather than None, so callers never have to
    special-case a missing value."""
    assert host_from_url("sqlite:///mlflow.db") == "local"

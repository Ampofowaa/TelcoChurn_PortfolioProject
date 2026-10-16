"""Unit tests for telco_churn.utils.db engine factory."""

from typing import Any

import pytest

import telco_churn.utils.db as db_module
from telco_churn.utils.db import get_async_engine, get_engine, host_from_url


@pytest.fixture(autouse=True)
def reset_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset the module-level _engine singleton before each test."""
    monkeypatch.setattr(db_module, "_engine", None)
    monkeypatch.setattr(db_module, "_async_engine", None)


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


def test_get_async_engine_recycles_and_bounds_every_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The async pool recycles idle connections by age and bounds connect/query
    time, so a silently dropped connection can never hang a request."""
    monkeypatch.setenv(
        "POSTGRES_URL",
        "postgresql://user:pass@localhost:5432/test",  # pragma: allowlist secret
    )
    captured: dict[str, Any] = {}

    def _fake_create_async_engine(url: str, **kwargs: Any) -> object:
        captured["url"] = url
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(db_module, "create_async_engine", _fake_create_async_engine)

    get_async_engine()

    assert captured["url"].startswith("postgresql+asyncpg://")
    assert captured["pool_pre_ping"] is True
    assert 0 < captured["pool_recycle"] <= 3600
    assert captured["connect_args"]["timeout"] > 0
    assert captured["connect_args"]["command_timeout"] > 0


def test_get_async_engine_returns_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated calls share one async engine (and so one connection pool)."""
    monkeypatch.setenv(
        "POSTGRES_URL",
        "postgresql://user:pass@localhost:5432/test",  # pragma: allowlist secret
    )
    assert get_async_engine() is get_async_engine()

"""Unit tests for telco_churn.utils.db engine factory."""

import pytest

import telco_churn.utils.db as db_module
from telco_churn.utils.db import get_engine


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

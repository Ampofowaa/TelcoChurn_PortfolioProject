"""Unit tests for src/telco_churn/features/sql_features.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from telco_churn.features import build_sql_features

# Private constant imported directly — single source of truth; update import if renamed.
from telco_churn.features.sql_features import _SQL_FILES as _EXPECTED_SQL_FILES


@pytest.fixture(scope="module")
def sql_dir() -> Path:
    """Resolve the SQL features directory from config at execution time, not collection time."""
    return Path(OmegaConf.load("configs/config.yaml").paths.sql_features)


@pytest.fixture
def mock_engine() -> tuple[MagicMock, MagicMock]:
    """Return (engine, conn) mocks wired for the `with engine.begin() as conn` pattern."""
    conn = MagicMock()
    engine = MagicMock()
    engine.begin.return_value.__enter__.return_value = conn
    return engine, conn


# ---------------------------------------------------------------------------
# Execution count and ordering
# ---------------------------------------------------------------------------


def test_build_sql_features_executes_all_three_files(
    mock_engine: tuple[MagicMock, MagicMock],
    sql_dir: Path,
) -> None:
    """build_sql_features calls conn.execute once per SQL file (3 total)."""
    engine, conn = mock_engine
    build_sql_features(engine, sql_dir)
    assert conn.execute.call_count == len(_EXPECTED_SQL_FILES)


def test_build_sql_features_execution_order(
    mock_engine: tuple[MagicMock, MagicMock],
    sql_dir: Path,
) -> None:
    """SQL files are executed in dependency order: tenure_buckets → charge_per_service → customer_features."""
    engine, conn = mock_engine
    build_sql_features(engine, sql_dir)

    executed_sql = [str(call.args[0]) for call in conn.execute.call_args_list]
    expected_order = ["tenure_buckets", "charge_per_service", "customer_features"]
    for i, view_name in enumerate(expected_order):
        assert (
            view_name in executed_sql[i]
        ), f"Expected call {i} to contain '{view_name}', got: {executed_sql[i][:80]}"


def test_build_sql_features_uses_transaction(
    mock_engine: tuple[MagicMock, MagicMock],
    sql_dir: Path,
) -> None:
    """build_sql_features wraps all executes in a single engine.begin() transaction."""
    engine, conn = mock_engine
    build_sql_features(engine, sql_dir)
    engine.begin.assert_called_once()


def test_build_sql_features_propagates_filename_on_error(
    mock_engine: tuple[MagicMock, MagicMock],
    sql_dir: Path,
) -> None:
    """RuntimeError message includes the failing SQL filename."""
    engine, conn = mock_engine
    conn.execute.side_effect = Exception("syntax error at or near SELECT")
    with pytest.raises(RuntimeError, match="tenure_buckets.sql"):
        build_sql_features(engine, sql_dir)


# ---------------------------------------------------------------------------
# SQL file content — idempotency contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("filename", _EXPECTED_SQL_FILES)
def test_sql_files_are_idempotent(filename: str, sql_dir: Path) -> None:
    """Each SQL file uses CREATE OR REPLACE VIEW so the operation is idempotent."""
    sql = (sql_dir / filename).read_text(encoding="utf-8")
    assert "CREATE OR REPLACE VIEW" in sql.upper()


@pytest.mark.parametrize("filename", _EXPECTED_SQL_FILES)
def test_sql_files_exist(filename: str, sql_dir: Path) -> None:
    """All three SQL files are present on disk at the expected path."""
    assert (sql_dir / filename).exists()

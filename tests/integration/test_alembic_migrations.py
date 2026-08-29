"""Integration test: the Alembic migration chain's up/down/up roundtrip
against a real testcontainers Postgres — the actual proof the five
migrations (customers_raw, customers_crm, prediction_log,
prediction_outcomes, training_pool) are reversible, not just that
`alembic upgrade head` produces the right schema (PROJECT_PLAN.md's Phase
10a-i Verification, extended to training_pool by Phase 10a-ii §E1).
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Engine
from testcontainers.postgres import PostgresContainer

from telco_churn.utils.db import apply_migrations
from telco_churn.utils.paths import get_project_root

pytestmark = pytest.mark.integration

_PROJECT_ROOT = get_project_root()

_MIGRATED_TABLES = {
    "customers_raw",
    "customers_crm",
    "prediction_log",
    "prediction_outcomes",
    "training_pool",
}


def _alembic_config(database_url: str):  # type: ignore[no-untyped-def]
    from alembic.config import Config

    alembic_cfg = Config(str(_PROJECT_ROOT / "alembic.ini"))
    alembic_cfg.set_main_option("sqlalchemy.url", database_url)
    return alembic_cfg


@pytest.fixture
def pg_url() -> Iterator[str]:
    with PostgresContainer("postgres:16") as pg:
        yield pg.get_connection_url(driver=None)


def test_upgrade_head_creates_all_five_tables_with_expected_shape(
    pg_url: str,
) -> None:
    apply_migrations(pg_url)

    engine: Engine = create_engine(pg_url)
    try:
        inspector = inspect(engine)
        assert _MIGRATED_TABLES <= set(inspector.get_table_names())

        prediction_log_index_columns = {
            column
            for idx in inspector.get_indexes("prediction_log")
            for column in idx["column_names"]
        }
        assert {"predicted_at", "customerid"} <= prediction_log_index_columns

        outcome_unique_constraints = inspector.get_unique_constraints(
            "prediction_outcomes"
        )
        assert any(
            set(uc["column_names"]) == {"customerid", "observed_at", "source"}
            for uc in outcome_unique_constraints
        )

        training_pool_columns = {
            col["name"]: col for col in inspector.get_columns("training_pool")
        }
        assert training_pool_columns["reserve_month"]["nullable"] is True
        assert training_pool_columns["churn"]["nullable"] is False
        training_pool_index_columns = {
            column
            for idx in inspector.get_indexes("training_pool")
            for column in idx["column_names"]
        }
        assert "reserve_month" in training_pool_index_columns
    finally:
        engine.dispose()


def test_downgrade_base_then_upgrade_head_reproduces_the_same_schema(
    pg_url: str,
) -> None:
    from alembic import command

    apply_migrations(pg_url)
    alembic_cfg = _alembic_config(pg_url)

    command.downgrade(alembic_cfg, "base")
    engine: Engine = create_engine(pg_url)
    try:
        remaining = set(inspect(engine).get_table_names()) & _MIGRATED_TABLES
        assert remaining == set(), f"downgrade base left tables behind: {remaining}"
    finally:
        engine.dispose()

    command.upgrade(alembic_cfg, "head")
    engine2: Engine = create_engine(pg_url)
    try:
        assert _MIGRATED_TABLES <= set(inspect(engine2).get_table_names())
    finally:
        engine2.dispose()

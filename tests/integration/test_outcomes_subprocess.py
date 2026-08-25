"""Integration test: serving/outcomes.py's __main__ CLI, via subprocess.run —
the required subprocess test for a __main__-bearing module (CLAUDE.md).
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from testcontainers.postgres import PostgresContainer

from telco_churn.utils.db import apply_migrations
from telco_churn.utils.paths import get_project_root

_PROJECT_ROOT = get_project_root()

pytestmark = pytest.mark.integration

_CUSTOMERID = "outcome-cust-0001"
_OBSERVED_AT = "2026-08-22T00:00:00+00:00"


@pytest.fixture
def pg_engine() -> Iterator[Engine]:
    """Ephemeral Postgres, migrated to head (all four tables) — prediction_outcomes
    has no dependency on customers_raw/customers_crm being seeded."""
    with PostgresContainer("postgres:16") as pg:
        url = pg.get_connection_url(driver=None)
        apply_migrations(url)
        engine = create_engine(pg.get_connection_url())
        yield engine
        engine.dispose()


def _run_outcomes_cli(pg_url: str, *overrides: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "POSTGRES_URL": pg_url}
    return subprocess.run(
        [sys.executable, "-m", "telco_churn.serving.outcomes", *overrides],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(_PROJECT_ROOT),
    )


def test_outcomes_main_cli_exits_zero_and_records_outcome(pg_engine: Engine) -> None:
    """CLAUDE.md: every __main__ entry point requires a subprocess integration
    test covering the full composition path — load_dotenv -> compose_config ->
    activate_config -> apply_migrations() -> get_engine() -> write_outcomes()
    -> sys.exit.
    """
    pg_url = pg_engine.url.render_as_string(hide_password=False)
    result = _run_outcomes_cli(
        pg_url,
        f"outcomes.customerid={_CUSTOMERID}",
        "outcomes.churned=true",
        f"outcomes.observed_at={_OBSERVED_AT}",
        "outcomes.source=synthetic_seed",
    )
    assert (
        result.returncode == 0
    ), f"outcomes CLI exited non-zero:\nstdout: {result.stdout}\nstderr: {result.stderr}"

    with pg_engine.connect() as conn:
        rows = (
            conn.execute(
                text(
                    "SELECT customerid, churned, source FROM prediction_outcomes "
                    "WHERE customerid = :customerid"
                ),
                {"customerid": _CUSTOMERID},
            )
            .mappings()
            .all()
        )

    assert len(rows) == 1
    assert rows[0]["churned"] is True
    assert rows[0]["source"] == "synthetic_seed"


def test_outcomes_main_cli_second_invocation_is_deduped(pg_engine: Engine) -> None:
    """A second invocation with the same (customerid, observed_at, source)
    stays at row count 1 — proof the ON CONFLICT DO NOTHING dedup holds,
    since this CLI has no live trigger to guard against accidental re-runs.
    """
    pg_url = pg_engine.url.render_as_string(hide_password=False)
    overrides = (
        f"outcomes.customerid={_CUSTOMERID}",
        "outcomes.churned=false",
        f"outcomes.observed_at={_OBSERVED_AT}",
        "outcomes.source=manual",
    )

    first = _run_outcomes_cli(pg_url, *overrides)
    assert first.returncode == 0, first.stderr
    second = _run_outcomes_cli(pg_url, *overrides)
    assert second.returncode == 0, second.stderr

    with pg_engine.connect() as conn:
        count = conn.execute(
            text(
                "SELECT COUNT(*) FROM prediction_outcomes WHERE customerid = :customerid"
            ),
            {"customerid": _CUSTOMERID},
        ).scalar()

    assert count == 1


def test_outcomes_main_cli_exits_one_when_customerid_missing(pg_engine: Engine) -> None:
    pg_url = pg_engine.url.render_as_string(hide_password=False)
    result = _run_outcomes_cli(
        pg_url,
        "outcomes.churned=true",
        f"outcomes.observed_at={_OBSERVED_AT}",
    )
    assert result.returncode == 1, (
        f"outcomes CLI should exit 1 with no customerid supplied:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )

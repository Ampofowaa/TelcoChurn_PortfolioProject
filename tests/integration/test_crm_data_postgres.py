"""Integration tests: customers_crm generation/load via testcontainers Postgres,
including the required subprocess test for crm_data.py's __main__ entry point.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from testcontainers.postgres import PostgresContainer

from telco_churn.data.ingest import ingest
from telco_churn.serving.crm_data import (
    CrmGenerationParams,
    generate_crm_rows,
    load_crm,
)
from telco_churn.utils.db import apply_migrations
from telco_churn.utils.paths import get_project_root

_PROJECT_ROOT = get_project_root()

pytestmark = pytest.mark.integration

_SAMPLE_CSV = """\
customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges,Churn
7590-VHVEG,Female,0,Yes,No,1,No,No phone service,DSL,No,Yes,No,No,No,No,Month-to-month,Yes,Electronic check,29.85,29.85,No
5575-GNVDE,Male,0,No,No,34,Yes,No,DSL,Yes,No,Yes,No,No,No,One year,No,Mailed check,56.95,1889.50,No
3668-QPYBK,Male,0,No,No,2,Yes,No,DSL,Yes,Yes,No,No,No,No,Month-to-month,Yes,Mailed check,53.85,108.15,Yes
7795-CFOCW,Male,0,No,No,45,No,No phone service,DSL,Yes,No,Yes,Yes,No,No,One year,No,Bank transfer (automatic),42.30,1840.75,No
zero-tenure,Female,0,No,No,0,No,No phone service,DSL,No,No,No,No,No,No,Month-to-month,Yes,Electronic check,20.00, ,No
"""

_PARAMS = CrmGenerationParams(
    random_state=42,
    tenure_advance_min_months=1,
    tenure_advance_max_months=6,
    contract_upgrade_probability=0.08,
    totalcharges_noise_scale=0.02,
)


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    path = tmp_path / "telco_sample.csv"
    path.write_text(_SAMPLE_CSV)
    return path


@pytest.fixture
def pg_engine_with_raw(sample_csv: Path) -> Iterator[Engine]:
    """Ephemeral Postgres, migrated to head (all four tables) and seeded with
    customers_raw; customers_crm exists but is empty."""
    with PostgresContainer("postgres:16") as pg:
        url = pg.get_connection_url(driver=None)
        apply_migrations(url)
        engine = create_engine(pg.get_connection_url())
        ingest(sample_csv, engine)
        yield engine
        engine.dispose()


def test_load_crm_persists_generated_rows(pg_engine_with_raw: Engine) -> None:
    raw_df = pd.read_sql_table("customers_raw", pg_engine_with_raw)
    crm_rows = generate_crm_rows(raw_df, _PARAMS)

    n = load_crm(crm_rows, pg_engine_with_raw)

    assert n == 5
    with pg_engine_with_raw.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM customers_crm")).scalar()
    assert count == 5


def test_load_crm_is_idempotent_under_a_fixed_seed(pg_engine_with_raw: Engine) -> None:
    """Re-running generate+load against the same raw table reproduces the
    same row count with no duplicate-key failure — a truncate-and-reload,
    not an upsert, is safe because the seeded nudges are deterministic."""
    raw_df = pd.read_sql_table("customers_raw", pg_engine_with_raw)

    load_crm(generate_crm_rows(raw_df, _PARAMS), pg_engine_with_raw)
    load_crm(generate_crm_rows(raw_df, _PARAMS), pg_engine_with_raw)

    with pg_engine_with_raw.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM customers_crm")).scalar()
    assert count == 5


# ---------------------------------------------------------------------------
# __main__ CLI composition: load_dotenv -> compose_config -> activate_config ->
#   apply_migrations() -> get_engine() -> pd.read_sql_table ->
#   generate_crm_rows -> load_crm -> sys.exit
# ---------------------------------------------------------------------------


def test_crm_data_main_cli_exits_zero_and_populates_customers_crm(
    pg_engine_with_raw: Engine,
) -> None:
    """crm_data.py __main__ exits 0 and populates customers_crm from an
    already-ingested customers_raw table.

    CLAUDE.md: every __main__ entry point requires a subprocess integration
    test covering the full composition path.
    """
    pg_url = pg_engine_with_raw.url.render_as_string(hide_password=False)
    env = {**os.environ, "POSTGRES_URL": pg_url}
    result = subprocess.run(
        [sys.executable, "-m", "telco_churn.serving.crm_data"],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(_PROJECT_ROOT),
    )
    assert (
        result.returncode == 0
    ), f"crm_data CLI exited non-zero:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    with pg_engine_with_raw.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM customers_crm")).scalar()
    assert count == 5


# Deliberately unreachable: port 1 on loopback refuses the connection almost
# instantly (ECONNREFUSED) rather than timing out, and connect_timeout bounds
# the worst case — same pattern test_sql_features_postgres.py uses for its
# own connection-failure exit-1 tests.
_BAD_POSTGRES_URL = "postgresql://baduser:badpass@127.0.0.1:1/nonexistentdb?connect_timeout=2"  # pragma: allowlist secret


def test_crm_data_main_cli_exits_one_on_connection_failure() -> None:
    """crm_data.py __main__ exits 1 when Postgres is unreachable.

    apply_migrations() now runs first in __main__ (belt-and-braces schema
    creation), so a genuinely missing customers_raw table is no longer
    reachable from a fresh Postgres — the migration creates it (empty) before
    pd.read_sql_table ever runs. An unreachable database is the exit-1 path
    that's actually still reachable: apply_migrations()'s own connection
    attempt fails first.
    """
    env = {**os.environ, "POSTGRES_URL": _BAD_POSTGRES_URL}
    result = subprocess.run(
        [sys.executable, "-m", "telco_churn.serving.crm_data"],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(_PROJECT_ROOT),
    )
    assert result.returncode == 1, (
        f"crm_data CLI should exit 1 on an unreachable DB:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )

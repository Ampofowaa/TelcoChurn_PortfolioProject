"""Integration tests: CSV → Postgres ingest via testcontainers."""

import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from testcontainers.postgres import PostgresContainer

from telco_churn.data.ingest import ingest
from telco_churn.data.validate import ValidationError
from telco_churn.utils.paths import get_project_root

_PROJECT_ROOT = get_project_root()

pytestmark = pytest.mark.integration

_DUPLICATE_ID_CSV = """\
customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges,Churn
7590-VHVEG,Female,0,Yes,No,1,No,No phone service,DSL,No,Yes,No,No,No,No,Month-to-month,Yes,Electronic check,29.85,29.85,No
7590-VHVEG,Female,0,Yes,No,1,No,No phone service,DSL,No,Yes,No,No,No,No,Month-to-month,Yes,Electronic check,29.85,29.85,No
"""

_SAMPLE_CSV = """\
customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges,Churn
7590-VHVEG,Female,0,Yes,No,1,No,No phone service,DSL,No,Yes,No,No,No,No,Month-to-month,Yes,Electronic check,29.85,29.85,No
5575-GNVDE,Male,0,No,No,34,Yes,No,DSL,Yes,No,Yes,No,No,No,One year,No,Mailed check,56.95,1889.50,No
3668-QPYBK,Male,0,No,No,2,Yes,No,DSL,Yes,Yes,No,No,No,No,Month-to-month,Yes,Mailed check,53.85,108.15,Yes
7795-CFOCW,Male,0,No,No,45,No,No phone service,DSL,Yes,No,Yes,Yes,No,No,One year,No,Bank transfer (automatic),42.30,1840.75,No
zero-tenure,Female,0,No,No,0,No,No phone service,DSL,No,No,No,No,No,No,Month-to-month,Yes,Electronic check,20.00, ,No
"""


@pytest.fixture(scope="module")
def pg_engine() -> Iterator[Engine]:
    """Spin up an ephemeral Postgres 16 container for the test module."""
    with PostgresContainer("postgres:16") as pg:
        engine = create_engine(pg.get_connection_url())
        yield engine
        engine.dispose()


@pytest.fixture(scope="module")
def pg_url(pg_engine: Engine) -> str:
    """Full connection URL for the testcontainers Postgres instance."""
    return pg_engine.url.render_as_string(hide_password=False)


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    """Five-row CSV fixture including one zero-tenure whitespace TotalCharges."""
    path = tmp_path / "telco_sample.csv"
    path.write_text(_SAMPLE_CSV)
    return path


def test_ingest_row_count(pg_engine: Engine, sample_csv: Path) -> None:
    """ingest() returns the correct row count."""
    n = ingest(sample_csv, pg_engine)
    assert n == 5


def test_ingest_persists_to_db(pg_engine: Engine, sample_csv: Path) -> None:
    """Rows are queryable from Postgres after ingest."""
    ingest(sample_csv, pg_engine)
    with pg_engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM customers_raw")).scalar()
    assert count == 5


def test_ingest_churn_is_binary(pg_engine: Engine, sample_csv: Path) -> None:
    """The churn column contains only 0 and 1 values."""
    ingest(sample_csv, pg_engine)
    with pg_engine.connect() as conn:
        rows = conn.execute(
            text("SELECT DISTINCT churn FROM customers_raw ORDER BY churn")
        ).fetchall()
    values = {r[0] for r in rows}
    assert values.issubset({0, 1})


def test_ingest_is_idempotent(pg_engine: Engine, sample_csv: Path) -> None:
    """Running ingest twice leaves exactly the original row count in the table."""
    ingest(sample_csv, pg_engine)
    ingest(sample_csv, pg_engine)
    with pg_engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM customers_raw")).scalar()
    assert count == 5


def test_staging_table_dropped_after_ingest(
    pg_engine: Engine, sample_csv: Path
) -> None:
    """The staging table is dropped inside the merge transaction and never left behind.

    In the staging-table pattern the throwaway table must not outlive the ingest
    call — leaving it behind wastes storage and signals an incomplete run.
    """
    ingest(sample_csv, pg_engine)
    with pg_engine.connect() as conn:
        result = conn.execute(
            text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_name = 'customers_raw_staging'"
            )
        ).fetchone()
    assert (
        result is None
    ), "customers_raw_staging must be dropped after a successful ingest"


def test_primary_key_survives_second_ingest(
    pg_engine: Engine, sample_csv: Path
) -> None:
    """The customerid PRIMARY KEY constraint is present even after a second ingest.

    With the old to_sql(if_exists='replace') approach this constraint was silently
    dropped on every call. The upsert path keeps the table intact.
    """
    ingest(sample_csv, pg_engine)
    ingest(sample_csv, pg_engine)
    with pg_engine.connect() as conn:
        result = conn.execute(
            text(
                """
                SELECT constraint_type
                FROM information_schema.table_constraints
                WHERE table_name = 'customers_raw'
                  AND constraint_type = 'PRIMARY KEY'
            """
            )
        ).fetchone()
    assert (
        result is not None
    ), "customers_raw must have a PRIMARY KEY constraint after ingest"


def test_ingest_validates_before_writing(pg_engine: Engine, tmp_path: Path) -> None:
    """ingest() raises ValidationError and leaves the DB unchanged when the CSV fails gate checks.

    Uses duplicate customerIDs to trigger Gate 2 (ERROR severity). The DB row
    count must be identical before and after the failed attempt — nothing is
    written when validation fails.
    """
    bad_csv = tmp_path / "dupes.csv"
    bad_csv.write_text(_DUPLICATE_ID_CSV)

    with pg_engine.connect() as conn:
        count_before = conn.execute(text("SELECT COUNT(*) FROM customers_raw")).scalar()

    with pytest.raises(ValidationError):
        ingest(bad_csv, pg_engine)

    with pg_engine.connect() as conn:
        count_after = conn.execute(text("SELECT COUNT(*) FROM customers_raw")).scalar()

    assert (
        count_after == count_before
    ), "customers_raw must not be modified when pre-load validation fails"


# ---------------------------------------------------------------------------
# __main__ CLI composition: load_dotenv → OmegaConf.load → argparse →
#   ingest(path, engine=None) → get_engine() → sys.exit
# ---------------------------------------------------------------------------


def test_ingest_main_cli_exits_zero(
    pg_engine: Engine, pg_url: str, sample_csv: Path
) -> None:
    """ingest.py __main__ exits 0 and populates customers_raw for a valid CSV path.

    CLAUDE.md: every __main__ entry point requires a subprocess integration test
    covering the full composition path. The engine=None branch in ingest() is only
    exercised here — existing tests always pass an explicit engine, bypassing
    get_engine() and the POSTGRES_URL join.
    """
    env = {**os.environ, "POSTGRES_URL": pg_url}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "telco_churn.data.ingest",
            "--csv-path",
            str(sample_csv),
        ],
        env=env,
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
    )
    assert (
        result.returncode == 0
    ), f"ingest CLI exited non-zero:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    with pg_engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM customers_raw")).scalar()
    assert count == 5


def test_ingest_main_cli_exits_one_on_missing_csv(pg_url: str, tmp_path: Path) -> None:
    """ingest.py __main__ exits 1 when the CSV path does not exist.

    FileNotFoundError is raised inside load_raw_csv() before any DB connection
    is attempted, so the error path does not require a populated table.
    """
    missing = tmp_path / "no_such_file.csv"
    env = {**os.environ, "POSTGRES_URL": pg_url}
    result = subprocess.run(
        [sys.executable, "-m", "telco_churn.data.ingest", "--csv-path", str(missing)],
        env=env,
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
    )
    assert (
        result.returncode == 1
    ), f"ingest CLI should exit 1 on missing CSV:\nstdout: {result.stdout}\nstderr: {result.stderr}"

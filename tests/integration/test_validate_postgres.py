"""Integration tests: validate.py __main__ CLI composition via testcontainers Postgres."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from testcontainers.postgres import PostgresContainer

from telco_churn.data.ingest import ingest
from telco_churn.utils.paths import get_project_root

pytestmark = pytest.mark.integration

_PROJECT_ROOT = get_project_root()

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
    """Ephemeral Postgres 16 container for the test module."""
    with PostgresContainer("postgres:16") as pg:
        engine = create_engine(pg.get_connection_url())
        yield engine
        engine.dispose()


@pytest.fixture(scope="module")
def pg_url(pg_engine: Engine) -> str:
    """Full connection URL for the testcontainers Postgres instance."""
    return pg_engine.url.render_as_string(hide_password=False)


@pytest.fixture(scope="module")
def seeded_engine(
    pg_engine: Engine, tmp_path_factory: pytest.TempPathFactory
) -> Engine:
    """Ingest sample CSV into customers_raw once for the module."""
    csv_path = tmp_path_factory.mktemp("validate_csv") / "sample.csv"
    csv_path.write_text(_SAMPLE_CSV)
    ingest(csv_path, pg_engine)
    return pg_engine


# ---------------------------------------------------------------------------
# __main__ CLI composition: load_dotenv → OmegaConf.load → get_engine →
#   read_sql → validate_raw → sys.exit
# ---------------------------------------------------------------------------


def test_validate_main_exits_zero_on_valid_data(
    seeded_engine: Engine,
    pg_url: str,
) -> None:
    """validate.py __main__ exits 0 when customers_raw passes all ERROR-severity gates.

    CLAUDE.md: every __main__ entry point requires an integration test exercising
    the full composition path (DB read → validate → exit code). seeded_engine
    ensures customers_raw is populated before the subprocess connects.

    Row count < 1000 triggers a WARNING (not ERROR), so can_proceed remains True
    and the process exits 0.
    """
    env = {**os.environ, "POSTGRES_URL": pg_url}
    result = subprocess.run(
        [sys.executable, "-m", "telco_churn.data.validate"],
        env=env,
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
    )
    assert (
        result.returncode == 0
    ), f"validate CLI exited non-zero:\nstdout: {result.stdout}\nstderr: {result.stderr}"


def test_validate_main_exits_one_on_blocking_error(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """validate.py __main__ exits 1 when customers_raw has ERROR-severity gate failures.

    Uses an isolated container so the corrupt churn value (9) does not affect
    the shared seeded_engine used by the happy-path test. The out-of-range churn
    value triggers both the schema gate (ERROR) and the churn_labels gate (ERROR),
    making can_proceed False and causing sys.exit(1).
    """
    csv_path = tmp_path_factory.mktemp("bad_validate_csv") / "sample.csv"
    csv_path.write_text(_SAMPLE_CSV)

    with PostgresContainer("postgres:16") as pg:
        engine = create_engine(pg.get_connection_url())
        ingest(csv_path, engine)
        with engine.connect() as conn:
            conn.execute(
                text(
                    "UPDATE customers_raw SET churn = 9 "
                    "WHERE customerid = '7590-VHVEG'"
                )
            )
            conn.commit()

        bad_url = engine.url.render_as_string(hide_password=False)
        env = {**os.environ, "POSTGRES_URL": bad_url}
        proc = subprocess.run(
            [sys.executable, "-m", "telco_churn.data.validate"],
            env=env,
            capture_output=True,
            text=True,
            cwd=str(_PROJECT_ROOT),
        )
        engine.dispose()

    assert (
        proc.returncode == 1
    ), f"validate CLI should exit 1 on errors:\nstdout: {proc.stdout}\nstderr: {proc.stderr}"

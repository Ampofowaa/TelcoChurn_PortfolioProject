"""Integration tests: CSV → Postgres ingest via testcontainers."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from testcontainers.postgres import PostgresContainer

from telco_churn.data.ingest import ingest

pytestmark = pytest.mark.integration

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

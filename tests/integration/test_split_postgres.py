"""Integration tests: split.py __main__ reading customers_raw via testcontainers Postgres."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from testcontainers.postgres import PostgresContainer

from telco_churn.data.ingest import ingest
from telco_churn.data.split import DEV, SPLIT_COL, TEST
from telco_churn.utils.paths import get_project_root

pytestmark = pytest.mark.integration

_PROJECT_ROOT = get_project_root()

_HEADER = (
    "customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,"
    "MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,"
    "TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,"
    "PaymentMethod,MonthlyCharges,TotalCharges,Churn"
)


def _row(i: int, churn: str) -> str:
    return (
        f"cust-{i:04d},Female,0,No,No,{10 + i},Yes,No,DSL,No,No,No,No,No,No,"
        f"Month-to-month,Yes,Mailed check,50.00,{500 + i}.00,{churn}"
    )


# 20 non-churners + 10 churners — enough of both classes that a stratified
# 80/20 split (configs/config.yaml training_setup.test_size) never collapses
# a class to zero rows in either partition.
_SEED_CSV = (
    "\n".join(
        [
            _HEADER,
            *(_row(i, "No") for i in range(20)),
            *(_row(i, "Yes") for i in range(20, 30)),
        ]
    )
    + "\n"
)


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
    """Ingest a 30-row synthetic CSV into customers_raw once for the module."""
    csv_path = tmp_path_factory.mktemp("split_seed") / "seed.csv"
    csv_path.write_text(_SEED_CSV)
    ingest(csv_path, pg_engine)
    return pg_engine


def test_split_main_cli_exits_zero(
    seeded_engine: Engine, pg_url: str, tmp_path: Path
) -> None:
    """split.py __main__ reads customers_raw, writes a valid manifest, and exits 0.

    CLAUDE.md: every __main__ entry point requires a subprocess integration test
    covering the full composition path — here that's get_engine() -> POSTGRES_URL
    -> pd.read_sql_table(customers_raw) -> make_split -> write_split.
    """
    out_dir = tmp_path
    env = {**os.environ, "POSTGRES_URL": pg_url, "PROCESSED_DATA_DIR": str(out_dir)}
    result = subprocess.run(
        [sys.executable, "-m", "telco_churn.data.split"],
        env=env,
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
    )
    assert (
        result.returncode == 0
    ), f"split CLI exited non-zero:\nstdout: {result.stdout}\nstderr: {result.stderr}"

    out_path = out_dir / "split_manifest.parquet"
    assert out_path.exists()
    manifest = pd.read_parquet(out_path)
    assert len(manifest) == 30
    assert set(manifest[SPLIT_COL].unique()) == {DEV, TEST}
    assert manifest["customerid"].is_unique
    n_test = int((manifest[SPLIT_COL] == TEST).sum())
    assert n_test == 6  # 30 rows * 0.2 test_size


def test_split_main_cli_exits_one_when_customers_raw_missing(tmp_path: Path) -> None:
    """split.py __main__ exits 1 when customers_raw does not exist yet.

    Uses a fresh, unseeded container (ingest never ran) — the realistic failure
    mode of running `make split` before `make ingest`. pd.read_sql_table raises
    ValueError when the table is absent, caught by split.py's ValueError branch,
    so no manifest is written.
    """
    with PostgresContainer("postgres:16") as pg:
        engine = create_engine(pg.get_connection_url())
        empty_url = engine.url.render_as_string(hide_password=False)
        engine.dispose()

        env = {
            **os.environ,
            "POSTGRES_URL": empty_url,
            "PROCESSED_DATA_DIR": str(tmp_path),
        }
        result = subprocess.run(
            [sys.executable, "-m", "telco_churn.data.split"],
            env=env,
            capture_output=True,
            text=True,
            cwd=str(_PROJECT_ROOT),
        )

    assert result.returncode == 1, (
        f"split CLI should exit 1 when customers_raw is missing:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert not (tmp_path / "split_manifest.parquet").exists()

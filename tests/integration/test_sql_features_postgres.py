"""Integration tests: SQL feature views via testcontainers Postgres."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from testcontainers.postgres import PostgresContainer

from telco_churn.data.ingest import ingest
from telco_churn.features import (
    FEATURE_SCHEMA,
    SQL_FEATURE_COLS,
    build_feature_df,
    build_sql_features,
)
from telco_churn.utils.paths import get_project_root

pytestmark = pytest.mark.integration

_FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
_RAW_COUNT = 13

# Deliberately unreachable: port 1 on loopback refuses the connection almost
# instantly (ECONNREFUSED) rather than timing out, and connect_timeout bounds
# the worst case. Used to trigger a real, deterministic connection failure
# without depending on whether another test in this module has already
# seeded customers_raw.
_BAD_POSTGRES_URL = "postgresql://baduser:badpass@127.0.0.1:1/nonexistentdb?connect_timeout=2"  # pragma: allowlist secret


@pytest.fixture(scope="module")
def sql_dir() -> Path:
    """Resolve the SQL features directory from config at execution time, not collection time."""
    return Path(OmegaConf.load("configs/config.yaml").paths.sql_features)


@pytest.fixture(scope="module")
def pg_engine() -> Iterator[Engine]:
    """Ephemeral Postgres 16 container seeded with sample data and feature views."""
    with PostgresContainer("postgres:16") as pg:
        engine = create_engine(pg.get_connection_url())
        yield engine
        engine.dispose()


@pytest.fixture(scope="module")
def pg_url(pg_engine: Engine) -> str:
    """Full connection URL for the testcontainers Postgres instance."""
    return pg_engine.url.render_as_string(hide_password=False)


@pytest.fixture(scope="module")
def seeded_engine(pg_engine: Engine, sql_dir: Path) -> Engine:
    """Ingest sample CSV and build SQL feature views once for the module."""
    ingest(_FIXTURES_DIR / "sample_features.csv", pg_engine)
    build_sql_features(pg_engine, sql_dir)
    return pg_engine


# ---------------------------------------------------------------------------
# charge_per_service view
# ---------------------------------------------------------------------------


def test_charge_per_service_row_count_matches_raw(seeded_engine: Engine) -> None:
    """charge_per_service view has the same row count as customers_raw."""
    with seeded_engine.connect() as conn:
        raw = conn.execute(text("SELECT COUNT(*) FROM customers_raw")).scalar()
        view = conn.execute(text("SELECT COUNT(*) FROM charge_per_service")).scalar()
    assert view == raw == _RAW_COUNT


def test_charge_per_service_no_null(seeded_engine: Engine) -> None:
    """charge_per_service is never NULL (GREATEST prevents divide-by-zero)."""
    with seeded_engine.connect() as conn:
        nulls = conn.execute(
            text(
                "SELECT COUNT(*) FROM charge_per_service "
                "WHERE charge_per_service IS NULL"
            )
        ).scalar()
    assert nulls == 0


def test_charge_per_service_positive(seeded_engine: Engine) -> None:
    """charge_per_service is positive for all rows."""
    with seeded_engine.connect() as conn:
        non_pos = conn.execute(
            text(
                "SELECT COUNT(*) FROM charge_per_service "
                "WHERE charge_per_service <= 0"
            )
        ).scalar()
    # No zero-monthlycharges row in the sample; 0.0 / N = 0.0 would be flagged by this assertion.
    assert non_pos == 0


def test_charge_per_service_lte_monthly_charges(seeded_engine: Engine) -> None:
    """charge_per_service <= monthlycharges for all rows (at least one service)."""
    with seeded_engine.connect() as conn:
        violators = conn.execute(
            text(
                "SELECT COUNT(*) FROM charge_per_service cps "
                "JOIN customers_raw r ON cps.customerid = r.customerid "
                "WHERE cps.charge_per_service > r.monthlycharges + 0.001"
            )
        ).scalar()
    assert violators == 0


def test_charge_per_service_fiber_optic_counts_as_internet(
    seeded_engine: Engine,
) -> None:
    """Fiber optic triggers internetservice <> 'No': phone(1)+multilines(1)+internet(1)=3; 90/3=30."""
    with seeded_engine.connect() as conn:
        val = conn.execute(
            text(
                "SELECT charge_per_service FROM charge_per_service "
                "WHERE customerid = :id"
            ).bindparams(id="fiber-optic-1")
        ).scalar()
    assert val is not None
    assert float(val) == pytest.approx(30.00)


def test_charge_per_service_no_internet_excluded_from_flag(
    seeded_engine: Engine,
) -> None:
    """InternetService='No' contributes 0 to service_count: phone(1) only; 19.90/1=19.90."""
    with seeded_engine.connect() as conn:
        val = conn.execute(
            text(
                "SELECT charge_per_service FROM charge_per_service "
                "WHERE customerid = :id"
            ).bindparams(id="no-internet-1")
        ).scalar()
    assert val is not None
    assert float(val) == pytest.approx(19.90)


@pytest.mark.parametrize(
    ("customerid", "expected"),
    [
        # phone=No(0), multilines='No phone service'(0), internet=DSL(1),
        # onlinebackup=Yes(1) → service_count=2; 29.85/2=14.925
        ("7590-VHVEG", 14.925),
    ],
)
def test_charge_per_service_value_correctness(
    seeded_engine: Engine, customerid: str, expected: float
) -> None:
    """charge_per_service equals monthlycharges / service_count for a known row."""
    with seeded_engine.connect() as conn:
        val = conn.execute(
            text(
                "SELECT charge_per_service FROM charge_per_service "
                "WHERE customerid = :id"
            ).bindparams(id=customerid)
        ).scalar()
    assert val is not None
    assert float(val) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# customer_features view
# ---------------------------------------------------------------------------


def test_customer_features_row_count_matches_raw(seeded_engine: Engine) -> None:
    """customer_features view preserves all rows from customers_raw."""
    with seeded_engine.connect() as conn:
        raw = conn.execute(text("SELECT COUNT(*) FROM customers_raw")).scalar()
        view = conn.execute(text("SELECT COUNT(*) FROM customer_features")).scalar()
    assert view == raw == _RAW_COUNT


def test_customer_features_charge_per_service_no_null(seeded_engine: Engine) -> None:
    """charge_per_service is non-NULL for every row in customer_features."""
    with seeded_engine.connect() as conn:
        nulls = conn.execute(
            text(
                "SELECT COUNT(*) FROM customer_features "
                "WHERE charge_per_service IS NULL"
            )
        ).scalar()
    assert nulls == 0


def test_customer_features_contains_expected_columns(seeded_engine: Engine) -> None:
    """customer_features exposes charge_per_service and churn alongside the raw cols."""
    with seeded_engine.connect() as conn:
        row = (
            conn.execute(text("SELECT * FROM customer_features LIMIT 1"))
            .mappings()
            .fetchone()
        )
    assert row is not None
    cols = set(row.keys())
    assert "charge_per_service" in cols
    assert "churn" in cols


# ---------------------------------------------------------------------------
# Full pipeline: customer_features view → build_feature_df
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def feature_df(seeded_engine: Engine) -> pd.DataFrame:
    """DataFrame produced by the full SQL → build_feature_df pipeline."""
    df_raw = pd.read_sql_table(
        "customer_features", seeded_engine, columns=SQL_FEATURE_COLS
    )
    return build_feature_df(df_raw)


def test_pipeline_row_count(feature_df: pd.DataFrame) -> None:
    """Pipeline preserves all rows from customers_raw."""
    assert feature_df.shape[0] == _RAW_COUNT


def test_pipeline_all_feature_columns_present(feature_df: pd.DataFrame) -> None:
    """All declared feature columns are present after the SQL → Python pipeline."""
    for col in list(
        FEATURE_SCHEMA.binary + FEATURE_SCHEMA.multi_cat + FEATURE_SCHEMA.numeric
    ):
        assert col in feature_df.columns


def test_pipeline_no_unexpected_nulls(feature_df: pd.DataFrame) -> None:
    """Only totalcharges may be null in pipeline output (11 zero-tenure rows)."""
    non_nullable = [
        c
        for c in list(
            FEATURE_SCHEMA.binary + FEATURE_SCHEMA.multi_cat + FEATURE_SCHEMA.numeric
        )
        if c != "totalcharges"
    ]
    assert not feature_df[non_nullable].isnull().any().any()


def test_pipeline_csv_write_shape(
    seeded_engine: Engine, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Pipeline output written to CSV has the expected shape."""
    df_raw = pd.read_sql_table(
        "customer_features", seeded_engine, columns=SQL_FEATURE_COLS
    )
    df_out = build_feature_df(df_raw)
    out_path = tmp_path_factory.mktemp("out") / "processed.csv"
    df_out.to_csv(out_path, index=False)

    df_loaded = pd.read_csv(out_path)
    expected_cols = (
        len(
            list(
                FEATURE_SCHEMA.binary
                + FEATURE_SCHEMA.multi_cat
                + FEATURE_SCHEMA.numeric
            )
        )
        + 2
    )  # + customerid + churn
    assert df_loaded.shape == (_RAW_COUNT, expected_cols)


# ---------------------------------------------------------------------------
# __main__ CLI composition: load_dotenv → OmegaConf.load → get_engine →
#   read_sql_table → build_feature_df → CSV write
# ---------------------------------------------------------------------------


def test_build_main_cli_composition(
    seeded_engine: Engine,
    pg_url: str,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Full build.py __main__ composition runs end-to-end and produces the expected CSV.

    CLAUDE.md: every __main__ entry point requires an integration test exercising the
    full composition path (DB read → transform → file write). seeded_engine ensures
    customers_raw is populated before the subprocess connects.
    """
    out_dir = tmp_path_factory.mktemp("build_cli")

    env = {**os.environ, "POSTGRES_URL": pg_url, "PROCESSED_DATA_DIR": str(out_dir)}
    result = subprocess.run(
        [sys.executable, "-m", "telco_churn.features.build"],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(get_project_root()),
    )
    assert (
        result.returncode == 0
    ), f"build CLI exited non-zero:\nstdout: {result.stdout}\nstderr: {result.stderr}"

    out_path = out_dir / "telco_churn_processed.csv"
    assert out_path.exists(), "processed CSV was not written by the CLI"
    df = pd.read_csv(out_path)
    expected_cols = (
        len(
            list(
                FEATURE_SCHEMA.binary
                + FEATURE_SCHEMA.multi_cat
                + FEATURE_SCHEMA.numeric
            )
        )
        + 2
    )  # + customerid + churn
    assert df.shape == (_RAW_COUNT, expected_cols)


def test_build_main_cli_exits_one_on_connection_failure() -> None:
    """build.py __main__ exits 1 when the database is unreachable.

    CLAUDE.md requires both the exit-0 and exit-1 paths covered per CLI entry
    point; test_build_main_cli_composition above only ever exercised exit-0.
    An unreachable POSTGRES_URL fails inside build_sql_features's
    engine.begin(), which build.py's __main__ catches as SQLAlchemyError.
    """
    env = {**os.environ, "POSTGRES_URL": _BAD_POSTGRES_URL}
    result = subprocess.run(
        [sys.executable, "-m", "telco_churn.features.build"],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(get_project_root()),
    )
    assert result.returncode == 1, (
        f"build CLI should exit 1 on an unreachable DB:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


# ---------------------------------------------------------------------------
# sql_features.py __main__ CLI composition: load_dotenv → OmegaConf.load →
#   get_engine → build_sql_features → sys.exit
#
# Not exercised by test_build_main_cli_composition above — that test calls
# build_sql_features() as a shared function from within build.py's own
# __main__, never sql_features.py's own CLI body (its own cfg load, its own
# get_engine() call, its own try/except sys.exit(1)).
# ---------------------------------------------------------------------------


def test_sql_features_main_cli_exits_zero(seeded_engine: Engine, pg_url: str) -> None:
    """sql_features.py __main__ exits 0 and (re)creates both feature views.

    Runs against the already-seeded module container; CREATE OR REPLACE VIEW
    makes rerunning the CLI idempotent, so this validates the CLI's own
    composition path end-to-end without needing a second container.
    """
    env = {**os.environ, "POSTGRES_URL": pg_url}
    result = subprocess.run(
        [sys.executable, "-m", "telco_churn.features.sql_features"],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(get_project_root()),
    )
    assert result.returncode == 0, (
        f"sql_features CLI exited non-zero:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    with seeded_engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM customer_features")).scalar()
    assert count == _RAW_COUNT


def test_sql_features_main_cli_exits_one_on_connection_failure() -> None:
    """sql_features.py __main__ exits 1 when the database is unreachable.

    Fails inside build_sql_features's engine.begin(), before any SQL file is
    read — deterministic and independent of ingestion order in this module.
    """
    env = {**os.environ, "POSTGRES_URL": _BAD_POSTGRES_URL}
    result = subprocess.run(
        [sys.executable, "-m", "telco_churn.features.sql_features"],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(get_project_root()),
    )
    assert result.returncode == 1, (
        f"sql_features CLI should exit 1 on an unreachable DB:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )

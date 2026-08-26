"""Integration tests: training_pool's cyclical reshape (write path 2) against
a real testcontainers Postgres.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from testcontainers.postgres import PostgresContainer

from telco_churn.data.training_pool import (
    RAW_FEATURE_COLS,
    RESERVE_COL,
    build_training_pool_cohort,
    write_training_pool_cohort,
)
from telco_churn.utils.db import apply_migrations

pytestmark = pytest.mark.integration

_BASE_SNAPSHOT: dict[str, object] = {
    "gender": "Female",
    "seniorcitizen": 0,
    "has_partner": "Yes",
    "dependents": "No",
    "tenure": 12,
    "phoneservice": "Yes",
    "multiplelines": "No",
    "internetservice": "DSL",
    "onlinesecurity": "No",
    "onlinebackup": "Yes",
    "deviceprotection": "No",
    "techsupport": "No",
    "streamingtv": "No",
    "streamingmovies": "No",
    "contract_type": "Month-to-month",
    "paperlessbilling": "Yes",
    "paymentmethod": "Electronic check",
    "monthlycharges": 55.5,
    "totalcharges": 666.0,
}


@pytest.fixture
def pg_engine() -> Iterator[Engine]:
    """Ephemeral, migrated-to-head Postgres per test — this module writes
    prediction_log/prediction_outcomes/training_pool rows directly and needs
    a clean slate each time."""
    with PostgresContainer("postgres:16") as pg:
        apply_migrations(pg.get_connection_url(driver=None))
        engine = create_engine(pg.get_connection_url())
        yield engine
        engine.dispose()


def _insert_prediction_log(
    engine: Engine,
    customerid: str,
    predicted_at: datetime,
    snapshot: dict[str, object],
    request_id: str = "req-1",
) -> None:
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO prediction_log
                    (request_id, customerid, predicted_at, route,
                     feature_snapshot, model_version, run_id, probability,
                     threshold, decision)
                VALUES
                    (:request_id, :customerid, :predicted_at, 'single',
                     CAST(:feature_snapshot AS JSONB), '1', 'run-1', 0.5,
                     0.3, true)
                """),
            {
                "request_id": request_id,
                "customerid": customerid,
                "predicted_at": predicted_at,
                "feature_snapshot": json.dumps(snapshot),
            },
        )


def _insert_prediction_outcome(
    engine: Engine, customerid: str, churned: bool, observed_at: datetime
) -> None:
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO prediction_outcomes
                    (customerid, churned, observed_at, source)
                VALUES (:customerid, :churned, :observed_at, 'synthetic_seed')
                """),
            {"customerid": customerid, "churned": churned, "observed_at": observed_at},
        )


_T0 = datetime(2026, 1, 1, tzinfo=UTC)


def test_build_training_pool_cohort_picks_most_recent_prediction_before_observed_at(
    pg_engine: Engine,
) -> None:
    """Two prediction_log rows exist before observed_at; a third exists after
    it. The reshape must pick the most recent of the two *before* — never the
    oldest, and never the one after maturation."""
    older_snapshot = {**_BASE_SNAPSHOT, "tenure": 10}
    newer_snapshot = {**_BASE_SNAPSHOT, "tenure": 16}
    after_maturity_snapshot = {**_BASE_SNAPSHOT, "tenure": 99}

    _insert_prediction_log(pg_engine, "cust-A", _T0, older_snapshot, "req-1")
    _insert_prediction_log(
        pg_engine, "cust-A", _T0 + timedelta(days=10), newer_snapshot, "req-2"
    )
    _insert_prediction_log(
        pg_engine,
        "cust-A",
        _T0 + timedelta(days=40),
        after_maturity_snapshot,
        "req-3",
    )
    _insert_prediction_outcome(
        pg_engine, "cust-A", churned=True, observed_at=_T0 + timedelta(days=30)
    )

    reserve_manifest = pd.DataFrame({"customerid": ["cust-A"], RESERVE_COL: [1]})
    result = build_training_pool_cohort(
        pg_engine, pd.Series(["cust-A"]), reserve_manifest
    )

    assert len(result) == 1
    row = result.iloc[0]
    assert row["tenure"] == 16
    assert row["churn"] == 1
    assert row[RESERVE_COL] == 1


def test_build_training_pool_cohort_flattens_all_raw_fields(pg_engine: Engine) -> None:
    """Every RAW_FEATURE_COLS entry lands correctly from feature_snapshot's JSON."""
    _insert_prediction_log(pg_engine, "cust-B", _T0, _BASE_SNAPSHOT)
    _insert_prediction_outcome(
        pg_engine, "cust-B", churned=False, observed_at=_T0 + timedelta(days=5)
    )
    reserve_manifest = pd.DataFrame({"customerid": ["cust-B"], RESERVE_COL: [2]})

    result = build_training_pool_cohort(
        pg_engine, pd.Series(["cust-B"]), reserve_manifest
    )

    row = result.iloc[0]
    for col, expected in _BASE_SNAPSHOT.items():
        assert row[col] == expected, f"{col} mismatch"
    assert row["churn"] == 0
    assert row["customerid"] == "cust-B"


def test_build_training_pool_cohort_returns_raw_fields_only(pg_engine: Engine) -> None:
    """The result carries exactly customerid + RAW_FEATURE_COLS + churn +
    reserve_month — no prediction_log-specific columns (probability,
    model_version, dual_score_mode, ...) leak through."""
    _insert_prediction_log(pg_engine, "cust-C", _T0, _BASE_SNAPSHOT)
    _insert_prediction_outcome(
        pg_engine, "cust-C", churned=True, observed_at=_T0 + timedelta(days=5)
    )
    reserve_manifest = pd.DataFrame({"customerid": ["cust-C"], RESERVE_COL: [3]})

    result = build_training_pool_cohort(
        pg_engine, pd.Series(["cust-C"]), reserve_manifest
    )

    expected_cols = {"customerid", "churn", RESERVE_COL, *RAW_FEATURE_COLS}
    assert set(result.columns) == expected_cols


def test_build_training_pool_cohort_missing_outcome_raises(pg_engine: Engine) -> None:
    """A requested customerid with no matured prediction_outcomes row fails
    loudly rather than silently omitting it."""
    _insert_prediction_log(pg_engine, "cust-D", _T0, _BASE_SNAPSHOT)
    # No prediction_outcomes row for cust-D.
    reserve_manifest = pd.DataFrame({"customerid": ["cust-D"], RESERVE_COL: [1]})

    with pytest.raises(ValueError, match="matured prediction_outcomes"):
        build_training_pool_cohort(pg_engine, pd.Series(["cust-D"]), reserve_manifest)


def test_build_training_pool_cohort_missing_reserve_manifest_entry_raises(
    pg_engine: Engine,
) -> None:
    """A matured customerid absent from reserve_manifest fails loudly — it
    must never be silently inserted with a NULL/guessed reserve_month."""
    _insert_prediction_log(pg_engine, "cust-E", _T0, _BASE_SNAPSHOT)
    _insert_prediction_outcome(
        pg_engine, "cust-E", churned=True, observed_at=_T0 + timedelta(days=5)
    )
    reserve_manifest = pd.DataFrame({"customerid": ["someone-else"], RESERVE_COL: [1]})

    with pytest.raises(ValueError, match="absent from reserve_manifest"):
        build_training_pool_cohort(pg_engine, pd.Series(["cust-E"]), reserve_manifest)


def test_build_training_pool_cohort_empty_customerids_raises(pg_engine: Engine) -> None:
    """An empty customerid input fails loudly rather than silently producing
    an empty result."""
    with pytest.raises(ValueError, match="at least one customerid"):
        build_training_pool_cohort(
            pg_engine,
            pd.Series([], dtype=str),
            pd.DataFrame({"customerid": [], RESERVE_COL: []}),
        )


def test_write_training_pool_cohort_appends_without_touching_seed_rows(
    pg_engine: Engine,
) -> None:
    """write_training_pool_cohort is a pure append — it must never disturb
    the reserve_month IS NULL seed rows write path 1 owns."""
    with pg_engine.begin() as conn:
        conn.execute(text("""
                INSERT INTO training_pool
                    (customerid, gender, seniorcitizen, has_partner, dependents,
                     tenure, phoneservice, multiplelines, internetservice,
                     onlinesecurity, onlinebackup, deviceprotection, techsupport,
                     streamingtv, streamingmovies, contract_type, paperlessbilling,
                     paymentmethod, monthlycharges, totalcharges, churn, reserve_month)
                VALUES
                    ('seed-cust', 'Female', 0, 'Yes', 'No', 5, 'Yes', 'No', 'DSL',
                     'No', 'No', 'No', 'No', 'No', 'No', 'Month-to-month', 'Yes',
                     'Mailed check', 40.0, 200.0, 0, NULL)
                """))

    _insert_prediction_log(pg_engine, "cust-F", _T0, _BASE_SNAPSHOT)
    _insert_prediction_outcome(
        pg_engine, "cust-F", churned=True, observed_at=_T0 + timedelta(days=5)
    )
    reserve_manifest = pd.DataFrame({"customerid": ["cust-F"], RESERVE_COL: [1]})
    cohort = build_training_pool_cohort(
        pg_engine, pd.Series(["cust-F"]), reserve_manifest
    )

    n_written = write_training_pool_cohort(cohort, pg_engine)
    assert n_written == 1

    with pg_engine.connect() as conn:
        null_count = conn.execute(
            text("SELECT COUNT(*) FROM training_pool WHERE reserve_month IS NULL")
        ).scalar()
        reserved_count = conn.execute(
            text("SELECT COUNT(*) FROM training_pool WHERE reserve_month = 1")
        ).scalar()
    assert null_count == 1
    assert reserved_count == 1

"""Integration tests: pipelines.performance_check's Postgres-touching reads
against a real testcontainers Postgres.

Scoped to the two functions with genuinely new, DB-dependent logic —
`_load_comparison_cohort`'s champion_probability-vs-probability fallback (the
correctness-critical point) and
`_score_candidate_on_reserve`'s reserve_month-only feature read. Everything
else this module does (comparative_deltas/build_gate_inputs/decide_promotion
wiring) is already covered by test_sealed_test.py/test_gate.py's own unit
suites and re-exercised here only at the wiring level
(tests/unit/test_performance_check.py).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from omegaconf import OmegaConf
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from testcontainers.postgres import PostgresContainer

from telco_churn.data.ingest import ingest
from telco_churn.features import build_sql_features
from telco_churn.pipelines.performance_check import (
    _load_comparison_cohort,
    _score_candidate_on_reserve,
)
from telco_churn.utils.db import apply_migrations

pytestmark = pytest.mark.integration

_FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
_T0 = datetime(2026, 1, 1, tzinfo=UTC)

_RESERVE_ROW: dict[str, object] = {
    "customerid": "RESERVE-0001",
    "gender": "Female",
    "seniorcitizen": 0,
    "has_partner": "Yes",
    "dependents": "No",
    "tenure": 20,
    "phoneservice": "Yes",
    "multiplelines": "No",
    "internetservice": "DSL",
    "onlinesecurity": "No",
    "onlinebackup": "Yes",
    "deviceprotection": "No",
    "techsupport": "No",
    "streamingtv": "No",
    "streamingmovies": "No",
    "contract_type": "One year",
    "paperlessbilling": "Yes",
    "paymentmethod": "Electronic check",
    "monthlycharges": 75.0,
    "totalcharges": 1500.0,
    "churn": 0,
}


@pytest.fixture(scope="module")
def sql_dir() -> Path:
    return Path(OmegaConf.load("configs/config.yaml").paths.sql_features)


@pytest.fixture
def pg_engine(sql_dir: Path) -> Iterator[Engine]:
    """Ephemeral Postgres per test — this module writes prediction_log/
    prediction_outcomes/training_pool rows directly and needs a clean slate
    each time (unlike test_sql_features_postgres.py's module-scoped fixture,
    which only ever reads)."""
    with PostgresContainer("postgres:16") as pg:
        apply_migrations(pg.get_connection_url(driver=None))
        engine = create_engine(pg.get_connection_url())
        ingest(_FIXTURES_DIR / "sample_features.csv", engine)
        build_sql_features(engine, sql_dir)
        yield engine
        engine.dispose()


def _insert_reserve_training_pool_row(
    engine: Engine, row: dict[str, object], reserve_month: int
) -> None:
    cols = [*row.keys(), "reserve_month"]
    placeholders = ", ".join(f":{c}" for c in cols)
    with engine.begin() as conn:
        conn.execute(
            text(
                f"INSERT INTO training_pool ({', '.join(cols)}) VALUES ({placeholders})"
            ),
            {**row, "reserve_month": reserve_month},
        )


def _insert_prediction_log(
    engine: Engine,
    customerid: str,
    predicted_at: datetime,
    probability: float,
    champion_probability: float | None,
    dual_score_mode: str | None,
    request_id: str,
) -> None:
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO prediction_log
                    (request_id, customerid, predicted_at, route,
                     feature_snapshot, model_version, run_id, probability,
                     threshold, decision, dual_score_mode, champion_probability)
                VALUES
                    (:request_id, :customerid, :predicted_at, 'single',
                     CAST(:feature_snapshot AS JSONB), '1', 'run-1',
                     :probability, 0.3, true, :dual_score_mode,
                     :champion_probability)
                """),
            {
                "request_id": request_id,
                "customerid": customerid,
                "predicted_at": predicted_at,
                "feature_snapshot": json.dumps({"tenure": 5}),
                "probability": probability,
                "dual_score_mode": dual_score_mode,
                "champion_probability": champion_probability,
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


# ---------------------------------------------------------------------------
# _load_comparison_cohort
# ---------------------------------------------------------------------------


def test_load_comparison_cohort_uses_champion_probability_when_dual_scored(
    pg_engine: Engine,
) -> None:
    """A row with dual_score_mode set (shadow/canary) must read
    champion_probability, never the served `probability` field — §D2's
    explicit rule, since a canary-routed row's `probability` can be the
    challenger's, not the champion's."""
    _insert_prediction_log(
        pg_engine,
        "cust-A",
        _T0,
        probability=0.91,
        champion_probability=0.42,
        dual_score_mode="canary",
        request_id="req-A",
    )
    _insert_prediction_outcome(pg_engine, "cust-A", churned=True, observed_at=_T0)
    manifest = pd.DataFrame({"customerid": ["cust-A"], "reserve_month": [1]})

    cohort = _load_comparison_cohort(pg_engine, 1, reserve_manifest=manifest)

    assert len(cohort) == 1
    assert cohort.iloc[0]["incumbent_probability"] == pytest.approx(0.42)


def test_load_comparison_cohort_falls_back_to_probability_when_not_dual_scored(
    pg_engine: Engine,
) -> None:
    """A row with no dual-scoring active (dual_score_mode/champion_probability
    both NULL) falls back to the served `probability` — unambiguously the
    champion's own score in that case."""
    _insert_prediction_log(
        pg_engine,
        "cust-B",
        _T0,
        probability=0.63,
        champion_probability=None,
        dual_score_mode=None,
        request_id="req-B",
    )
    _insert_prediction_outcome(pg_engine, "cust-B", churned=False, observed_at=_T0)
    manifest = pd.DataFrame({"customerid": ["cust-B"], "reserve_month": [1]})

    cohort = _load_comparison_cohort(pg_engine, 1, reserve_manifest=manifest)

    assert len(cohort) == 1
    assert cohort.iloc[0]["incumbent_probability"] == pytest.approx(0.63)
    assert bool(cohort.iloc[0]["churned"]) is False


def test_load_comparison_cohort_picks_most_recent_prediction_before_observed_at(
    pg_engine: Engine,
) -> None:
    """Same join contract as training_pool's write-path-2 (§B8): most recent
    prediction before observed_at, never a later one."""
    _insert_prediction_log(
        pg_engine,
        "cust-C",
        _T0,
        probability=0.10,
        champion_probability=None,
        dual_score_mode=None,
        request_id="req-C1",
    )
    _insert_prediction_log(
        pg_engine,
        "cust-C",
        _T0 + timedelta(days=5),
        probability=0.80,
        champion_probability=None,
        dual_score_mode=None,
        request_id="req-C2",
    )
    _insert_prediction_outcome(
        pg_engine, "cust-C", churned=True, observed_at=_T0 + timedelta(days=10)
    )
    manifest = pd.DataFrame({"customerid": ["cust-C"], "reserve_month": [1]})

    cohort = _load_comparison_cohort(pg_engine, 1, reserve_manifest=manifest)

    assert cohort.iloc[0]["incumbent_probability"] == pytest.approx(0.80)


def test_load_comparison_cohort_missing_outcome_raises(pg_engine: Engine) -> None:
    """A reserve-month customerid with no matured outcome yet fails loudly."""
    manifest = pd.DataFrame({"customerid": ["cust-D"], "reserve_month": [1]})

    with pytest.raises(ValueError, match="matured"):
        _load_comparison_cohort(pg_engine, 1, reserve_manifest=manifest)


def test_load_comparison_cohort_empty_month_raises(pg_engine: Engine) -> None:
    """A reserve_month absent from the manifest entirely fails loudly rather
    than silently returning an empty cohort."""
    manifest = pd.DataFrame({"customerid": ["cust-E"], "reserve_month": [1]})

    with pytest.raises(ValueError, match="No customerids found"):
        _load_comparison_cohort(pg_engine, 99, reserve_manifest=manifest)


# ---------------------------------------------------------------------------
# _score_candidate_on_reserve
# ---------------------------------------------------------------------------


class _FakeModel:
    """predict_proba stands in for a fitted sklearn Pipeline — this suite only
    verifies the right rows/columns reach it, never real prediction values."""

    def predict_proba(self, X: pd.DataFrame) -> NDArray[np.float64]:
        n = len(X)
        p1 = np.linspace(0.1, 0.9, n)
        return np.column_stack([1 - p1, p1])


def test_score_candidate_on_reserve_scopes_to_exact_reserve_month(
    pg_engine: Engine,
) -> None:
    """Reads only the requested reserve_month's rows — never dev (reserve_month
    IS NULL) alongside it, unlike features/build.py::build_feature_query's
    always-folds-in-dev training-query shape."""
    _insert_reserve_training_pool_row(pg_engine, _RESERVE_ROW, reserve_month=1)
    other_row = {**_RESERVE_ROW, "customerid": "RESERVE-0002"}
    _insert_reserve_training_pool_row(pg_engine, other_row, reserve_month=2)

    proba = _score_candidate_on_reserve(
        _FakeModel(),
        ["tenure", "monthlycharges"],
        pg_engine,
        1,
        pd.Series(["RESERVE-0001"]),
    )

    assert len(proba) == 1


def test_score_candidate_on_reserve_missing_customerid_raises(
    pg_engine: Engine,
) -> None:
    """A customerid the comparison cohort expects but customer_features
    doesn't have for this reserve_month fails loudly."""
    _insert_reserve_training_pool_row(pg_engine, _RESERVE_ROW, reserve_month=1)

    with pytest.raises(ValueError, match="missing from customer_features"):
        _score_candidate_on_reserve(
            _FakeModel(),
            ["tenure", "monthlycharges"],
            pg_engine,
            1,
            pd.Series(["RESERVE-0001", "NOT-PRESENT"]),
        )


def test_score_candidate_on_reserve_empty_month_raises(pg_engine: Engine) -> None:
    """No customer_features rows at all for the requested reserve_month fails
    loudly rather than scoring zero rows silently."""
    with pytest.raises(ValueError, match="No customer_features rows found"):
        _score_candidate_on_reserve(
            _FakeModel(), ["tenure", "monthlycharges"], pg_engine, 7, pd.Series(["x"])
        )

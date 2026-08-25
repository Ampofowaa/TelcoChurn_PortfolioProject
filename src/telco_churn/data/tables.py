"""SQLAlchemy Core table definitions for this project's Postgres schema.

The only consumer today is alembic/env.py's target_metadata — every other
module in this project reads/writes these tables via raw text() SQL or
pandas to_sql/read_sql_table (data/ingest.py, serving/crm_data.py,
serving/customer_lookup.py), and continues to do so. This module exists
solely so Alembic has a MetaData object to diff/autogenerate against; it is
Core Table objects, not a declarative ORM layer, to match that existing style
rather than introduce a second data-access pattern nothing else uses.

customers_raw/customers_crm mirror sql/schema/001_create_raw.sql and
sql/schema/003_create_customers_crm.sql exactly — those two files are
deleted once the Alembic migrations that reproduce them land (see
PROJECT_PLAN.md's Phase 10a-i). prediction_log/prediction_outcomes have no
prior CREATE TABLE IF NOT EXISTS file; their column shapes come from
prediction_logging_plan.md §B1 and PROJECT_PLAN.md's Phase 10a-i deliverable
list.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    Identity,
    Index,
    MetaData,
    Numeric,
    SmallInteger,
    String,
    Table,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB

__all__ = [
    "metadata",
    "customers_raw",
    "customers_crm",
    "prediction_log",
    "prediction_outcomes",
]

metadata = MetaData()


def _customer_columns() -> tuple[Column[Any], ...]:
    """Column shape shared by customers_raw and customers_crm — both are
    LOOKUP_COLUMNS-shaped snapshots of the same 19 source fields, differing
    only in their key column (churn vs. crm_snapshot_at).

    A Column instance may only ever belong to one Table, so this returns a
    fresh tuple per call rather than a shared module-level constant — reusing
    the same Column objects across two Table(...) calls raises
    ArgumentError("... already assigned to Table ...").
    """
    return (
        Column("gender", String(10), nullable=False),
        Column(
            "seniorcitizen",
            SmallInteger,
            CheckConstraint("seniorcitizen IN (0, 1)", name="ck_seniorcitizen_binary"),
            nullable=False,
        ),
        Column("has_partner", String(3), nullable=False),
        Column("dependents", String(3), nullable=False),
        Column(
            "tenure",
            SmallInteger,
            CheckConstraint("tenure >= 0", name="ck_tenure_nonnegative"),
            nullable=False,
        ),
        Column("phoneservice", String(3), nullable=False),
        Column("multiplelines", String(25), nullable=False),
        Column("internetservice", String(25), nullable=False),
        Column("onlinesecurity", String(25), nullable=False),
        Column("onlinebackup", String(25), nullable=False),
        Column("deviceprotection", String(25), nullable=False),
        Column("techsupport", String(25), nullable=False),
        Column("streamingtv", String(25), nullable=False),
        Column("streamingmovies", String(25), nullable=False),
        Column("contract_type", String(20), nullable=False),
        Column("paperlessbilling", String(3), nullable=False),
        Column("paymentmethod", String(45), nullable=False),
        Column(
            "monthlycharges",
            Numeric(8, 2),
            CheckConstraint(
                "monthlycharges >= 0", name="ck_monthlycharges_nonnegative"
            ),
            nullable=False,
        ),
        Column("totalcharges", Numeric(10, 2), nullable=True),
    )


customers_raw = Table(
    "customers_raw",
    metadata,
    Column("customerid", String(20), primary_key=True),
    *_customer_columns(),
    Column(
        "churn",
        SmallInteger,
        CheckConstraint("churn IN (0, 1)", name="ck_customers_raw_churn_binary"),
        nullable=False,
    ),
)

customers_crm = Table(
    "customers_crm",
    metadata,
    Column("customerid", String(20), primary_key=True),
    *_customer_columns(),
    Column("crm_snapshot_at", DateTime(timezone=True), nullable=False),
)

# request_id deliberately carries no uniqueness constraint — prediction_log
# is an append-only event log, and a client retry producing a second row
# with the same request_id is expected and harmless (PROJECT_PLAN.md's
# Phase 10a-i note).
prediction_log = Table(
    "prediction_log",
    metadata,
    Column("prediction_id", BigInteger, Identity(), primary_key=True),
    Column("request_id", String(36), nullable=False),
    Column("customerid", String(20), nullable=True),
    Column(
        "predicted_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    ),
    Column("route", String(20), nullable=False),
    Column("feature_snapshot", JSONB, nullable=False),
    Column("model_version", String(20), nullable=False),
    Column("run_id", String(64), nullable=False),
    Column("probability", Float, nullable=False),
    Column("threshold", Float, nullable=False),
    Column("decision", Boolean, nullable=False),
    Column("contact", Boolean, nullable=True),
    Column("dual_score_mode", String(10), nullable=True),
    Column("challenger_version", String(20), nullable=True),
    Column("challenger_probability", Float, nullable=True),
    Column("champion_probability", Float, nullable=True),
    Column("resolution_kind", String(20), nullable=True),
    Index("ix_prediction_log_predicted_at", "predicted_at"),
    Index("ix_prediction_log_customerid", "customerid"),
)

# Unique on (customerid, observed_at, source) is what makes serving/outcomes.py's
# INSERT ... ON CONFLICT DO NOTHING a safe re-run rather than a silent duplicate.
prediction_outcomes = Table(
    "prediction_outcomes",
    metadata,
    Column("outcome_id", BigInteger, Identity(), primary_key=True),
    Column("customerid", String(20), nullable=False),
    Column("churned", Boolean, nullable=False),
    Column("observed_at", DateTime(timezone=True), nullable=False),
    Column(
        "recorded_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    ),
    Column("source", String(20), nullable=False),
    UniqueConstraint(
        "customerid",
        "observed_at",
        "source",
        name="uq_prediction_outcomes_customer_observed_source",
    ),
)

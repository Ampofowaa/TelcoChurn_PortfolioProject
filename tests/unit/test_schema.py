"""Unit tests for schema consistency across the two authoritative representations.

The column set is defined in two places that serve different purposes:
  - data/tables.py customers_raw  — DB-level types and constraints (PRIMARY KEY, NOT NULL),
    the SQLAlchemy Core Table alembic/versions/2a3418a6f529_create_customers_raw.py
    reproduces as real DDL
  - data/schema.py RawSchema      — application-level value rules (isin, ge, nullable)

Neither can replace the other, but they must agree on which columns exist.
This test catches divergence at pytest time (no Docker needed) rather than at
ingest runtime.
"""

from __future__ import annotations

import pandas as pd
import pandera as pa
import pytest
from helpers import make_row

from telco_churn.data.schema import RawSchema
from telco_churn.data.tables import customers_raw


def test_ddl_and_rawschema_define_same_columns() -> None:
    """DDL column set must equal RawSchema field set.

    Catches the case where a column is added to data/tables.py's customers_raw
    Table but missed in RawSchema (or vice versa). Without this test the gap
    is only caught at ingest runtime — after Docker is running and a CSV is
    present.
    """
    ddl_cols = frozenset(customers_raw.columns.keys())
    schema_cols = frozenset(RawSchema.to_schema().columns.keys())

    only_in_ddl = ddl_cols - schema_cols
    only_in_schema = schema_cols - ddl_cols

    assert not only_in_ddl, f"Columns in DDL but missing from RawSchema: {only_in_ddl}"
    assert (
        not only_in_schema
    ), f"Columns in RawSchema but missing from DDL: {only_in_schema}"


def test_raw_schema_rejects_billed_customer_with_totalcharges_below_monthlycharges() -> (
    None
):
    """RawSchema must reject totalcharges < monthlycharges for billed (tenure>=1) customers.

    make_row() defaults to tenure=12 and monthlycharges=29.85; passing totalcharges=10.0
    creates a row where the invariant is violated.
    """
    df = pd.DataFrame([make_row(totalcharges=10.0)])
    with pytest.raises(pa.errors.SchemaError):
        RawSchema.validate(df)

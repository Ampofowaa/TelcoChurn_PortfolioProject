"""Unit tests for schema consistency across the two authoritative representations.

The column set is defined in two places that serve different purposes:
  - sql/schema/001_create_raw.sql  — DB-level types and constraints (PRIMARY KEY, NOT NULL)
  - data/schema.py RawSchema       — application-level value rules (isin, ge, nullable)

Neither can replace the other, but they must agree on which columns exist.
This test catches divergence at pytest time (pure file read, no Docker needed)
rather than at ingest runtime.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandera as pa
import pytest
from helpers import make_row

from telco_churn.data.schema import RawSchema

_DDL_PATH = (
    Path(__file__).resolve().parents[2] / "sql" / "schema" / "001_create_raw.sql"
)


def _parse_ddl_columns(ddl: str) -> frozenset[str]:
    """Extract column names from a CREATE TABLE statement.

    Skips comment lines, the CREATE TABLE header, and the closing paren.
    Takes the first token from every remaining non-empty line as the column name.
    """
    cols = set()
    for line in ddl.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("CREATE", "--", ")")):
            continue
        cols.add(stripped.split()[0].lower())
    return frozenset(cols)


def test_ddl_and_rawschema_define_same_columns() -> None:
    """DDL column set must equal RawSchema field set.

    Catches the case where a column is added to 001_create_raw.sql but missed
    in RawSchema (or vice versa). Without this test the gap is only caught at
    ingest runtime — after Docker is running and a CSV is present.
    """
    ddl = _DDL_PATH.read_text()
    ddl_cols = _parse_ddl_columns(ddl)
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

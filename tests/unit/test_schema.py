"""Unit tests for schema consistency across the two authoritative representations.

The column set is defined in two places that serve different purposes:
  - sql/schema/001_create_raw.sql  — DB-level types and constraints (PRIMARY KEY, NOT NULL)
  - data/schema.py RawSchema       — application-level value rules (isin, ge, nullable)

Neither can replace the other, but they must agree on which columns exist.
This test catches divergence at pytest time (pure file read, no Docker needed)
rather than at ingest runtime.

Cross-schema invariant tests enforce that CleanedSchema's
totalcharges_gte_monthlycharges_for_billed_customers check and
FeatureOutputSchema's le=1.0 constraint on monthly_to_total_ratio stay
consistent — a change to either without updating the other breaks a test
rather than silently diverging.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandera as pa
import pytest
from helpers import make_row

from telco_churn.data.schema import CleanedSchema, RawSchema
from telco_churn.features.schema import FeatureOutputSchema

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


# ---------------------------------------------------------------------------
# Cross-schema invariant: CleanedSchema ↔ FeatureOutputSchema.monthly_to_total_ratio
# ---------------------------------------------------------------------------


def _make_feature_row(monthly_to_total_ratio: float = 0.083) -> dict[str, object]:
    """Return a minimal dict valid against FeatureOutputSchema."""
    return {
        "gender": "Male",
        "seniorcitizen": 0,
        "has_partner": "Yes",
        "dependents": "No",
        "phoneservice": "Yes",
        "paperlessbilling": "Yes",
        "multiplelines": "No",
        "internetservice": "DSL",
        "onlinesecurity": "Yes",
        "onlinebackup": "No",
        "deviceprotection": "No",
        "techsupport": "No",
        "streamingtv": "No",
        "streamingmovies": "No",
        "contract_type": "Month-to-month",
        "paymentmethod": "Electronic check",
        "tenure_cohort": "0–12 mo",
        "tenure": 12,
        "monthlycharges": 29.85,
        "totalcharges": 358.20,
        "charge_per_service": 5.0,
        "is_long_month_to_month": 0,
        "fiber_contract": "Not Fiber optic",
        "dsl_contract": "Month-to-month_DSL",
        "monthly_to_total_ratio": monthly_to_total_ratio,
    }


def test_cleaned_schema_rejects_billed_customer_with_totalcharges_below_monthlycharges() -> (
    None
):
    """CleanedSchema must reject totalcharges < monthlycharges for billed (tenure>=1) customers.

    make_row() defaults to tenure=12 and monthlycharges=29.85; passing totalcharges=10.0
    creates a row where the invariant is violated.
    """
    df = pd.DataFrame([make_row(totalcharges=10.0)])
    with pytest.raises(pa.errors.SchemaError):
        CleanedSchema.validate(df)


def test_feature_output_schema_rejects_monthly_to_total_ratio_above_one() -> None:
    """FeatureOutputSchema must reject monthly_to_total_ratio > 1.0 (le=1.0 constraint)."""
    df = pd.DataFrame([_make_feature_row(monthly_to_total_ratio=1.5)])
    with pytest.raises(pa.errors.SchemaError):
        FeatureOutputSchema.validate(df)


@pytest.mark.parametrize(
    ("monthlycharges", "totalcharges"),
    [
        (29.85, 358.20),  # typical 12-month customer
        (100.0, 100.0),  # equal charges: ratio exactly 1.0
        (0.01, 100.0),  # very small monthly relative to total
    ],
)
def test_cleaned_schema_invariant_implies_ratio_le_one(
    monthlycharges: float, totalcharges: float
) -> None:
    """Mathematical consistency: totalcharges >= monthlycharges always yields ratio <= 1.0.

    CleanedSchema guarantees totalcharges >= monthlycharges for billed customers.
    FeatureOutputSchema enforces monthly_to_total_ratio <= 1.0.
    These constraints are consistent iff monthlycharges / totalcharges <= 1.0
    whenever totalcharges >= monthlycharges > 0.
    """
    assert (
        monthlycharges <= totalcharges
    ), "precondition: CleanedSchema invariant must hold"
    assert monthlycharges / totalcharges <= 1.0

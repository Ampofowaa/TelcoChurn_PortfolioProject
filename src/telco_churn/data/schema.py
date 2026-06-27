"""Pandera schema definitions for raw and cleaned Telco customer data."""

from __future__ import annotations

from typing import Final

import pandas as pd
import pandera as pa
from pandera.pandas import DataFrameModel, Field
from pandera.typing import Series

__all__ = [
    "YES_NO",
    "YES_NO_NO_PHONE",
    "YES_NO_NO_INTERNET",
    "RawSchema",
    "CleanedSchema",
]

# ---------------------------------------------------------------------------
# Allowed categorical values (source of truth: notebooks/_archive/EDA-original.ipynb)
# ---------------------------------------------------------------------------

YES_NO: Final[frozenset[str]] = frozenset({"Yes", "No"})
YES_NO_NO_PHONE: Final[frozenset[str]] = frozenset({"Yes", "No", "No phone service"})
YES_NO_NO_INTERNET: Final[frozenset[str]] = frozenset(
    {"Yes", "No", "No internet service"}
)

# ---------------------------------------------------------------------------
# Pandera SchemaModel definitions
# ---------------------------------------------------------------------------


class RawSchema(DataFrameModel):  # type: ignore[misc]
    """Schema for the raw customers_raw table (sql/schema/001_create_raw.sql).

    totalcharges is nullable — the 11 zero-tenure customers in the IBM Telco
    dataset have whitespace in the source CSV, coerced to NaN by ingest.py.
    """

    customerid: Series[str] = Field(nullable=False)
    gender: Series[str] = Field(isin={"Male", "Female"}, nullable=False)
    seniorcitizen: Series[int] = Field(isin=[0, 1], nullable=False)
    has_partner: Series[str] = Field(isin=YES_NO, nullable=False)
    dependents: Series[str] = Field(isin=YES_NO, nullable=False)
    tenure: Series[int] = Field(ge=0, nullable=False)
    phoneservice: Series[str] = Field(isin=YES_NO, nullable=False)
    multiplelines: Series[str] = Field(isin=YES_NO_NO_PHONE, nullable=False)
    internetservice: Series[str] = Field(
        isin={"DSL", "Fiber optic", "No"}, nullable=False
    )
    onlinesecurity: Series[str] = Field(isin=YES_NO_NO_INTERNET, nullable=False)
    onlinebackup: Series[str] = Field(isin=YES_NO_NO_INTERNET, nullable=False)
    deviceprotection: Series[str] = Field(isin=YES_NO_NO_INTERNET, nullable=False)
    techsupport: Series[str] = Field(isin=YES_NO_NO_INTERNET, nullable=False)
    streamingtv: Series[str] = Field(isin=YES_NO_NO_INTERNET, nullable=False)
    streamingmovies: Series[str] = Field(isin=YES_NO_NO_INTERNET, nullable=False)
    contract_type: Series[str] = Field(
        isin={"Month-to-month", "One year", "Two year"}, nullable=False
    )
    paperlessbilling: Series[str] = Field(isin=YES_NO, nullable=False)
    paymentmethod: Series[str] = Field(
        isin={
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        },
        nullable=False,
    )
    monthlycharges: Series[float] = Field(ge=0.0, nullable=False)
    totalcharges: Series[float] = Field(ge=0.0, nullable=True)
    churn: Series[int] = Field(isin=[0, 1], nullable=False)

    class Config:
        strict = True  # unexpected columns fail loudly — schema drift must be a conscious decision
        coerce = True  # handles Postgres NUMERIC→float and SMALLINT→int coercion

    @pa.dataframe_check  # type: ignore[untyped-decorator]
    @classmethod
    def totalcharges_zero_implies_no_monthly_charge(cls, df: pd.DataFrame) -> pd.Series:
        """totalcharges=0.0 with monthlycharges>0 is a pipeline data integrity violation.

        The only legitimate zero-totalcharges case is NULL (11 zero-tenure customers
        whose first bill has not yet been issued). A non-null zero with a positive
        monthly charge cannot occur in valid data and indicates an upstream bug.
        NaN rows pass this check — NaN == 0.0 is False in pandas.
        """
        return ~((df["totalcharges"] == 0.0) & (df["monthlycharges"] > 0))


class CleanedSchema(RawSchema):
    """RawSchema with totalcharges required to be non-null.

    The model training sklearn ColumnTransformer (SimpleImputer) is a prerequisite —
    validate_clean() should be called after imputation, not before.
    """

    totalcharges: Series[float] = Field(ge=0.0, nullable=False)

    @pa.dataframe_check  # type: ignore[untyped-decorator]
    @classmethod
    def totalcharges_gte_monthlycharges_for_billed_customers(
        cls, df: pd.DataFrame
    ) -> pd.Series:
        """totalcharges >= monthlycharges for customers with tenure >= 1.

        A customer billed for at least one month must have accrued at least one
        month's charges — totalcharges below monthlycharges indicates a data
        corruption or ingestion error. Zero-tenure customers (tenure=0) are
        exempt: their totalcharges is the imputed median, not an accrued billing
        total. NaN tenure rows return False from >= 1 in pandas and pass the check.
        """
        billed_mask = df["tenure"] >= 1
        return ~(billed_mask & (df["totalcharges"] < df["monthlycharges"]))

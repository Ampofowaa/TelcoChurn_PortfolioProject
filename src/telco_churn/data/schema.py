"""Pandera schema definitions for raw and cleaned Telco customer data."""

from __future__ import annotations

from typing import Final

from pandera.pandas import DataFrameModel, Field
from pandera.typing import Series

# ---------------------------------------------------------------------------
# Allowed categorical values (source of truth: notebooks/_archive/EDA-original.ipynb)
# ---------------------------------------------------------------------------

_YES_NO: Final[frozenset[str]] = frozenset({"Yes", "No"})
_YES_NO_NO_PHONE: Final[frozenset[str]] = frozenset({"Yes", "No", "No phone service"})
_YES_NO_NO_INTERNET: Final[frozenset[str]] = frozenset(
    {"Yes", "No", "No internet service"}
)

REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "customerid",
        "gender",
        "seniorcitizen",
        "has_partner",
        "dependents",
        "tenure",
        "phoneservice",
        "multiplelines",
        "internetservice",
        "onlinesecurity",
        "onlinebackup",
        "deviceprotection",
        "techsupport",
        "streamingtv",
        "streamingmovies",
        "contract_type",
        "paperlessbilling",
        "paymentmethod",
        "monthlycharges",
        "totalcharges",
        "churn",
    }
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
    gender: Series[str] = Field(isin={"Male", "Female"}, nullable=True)
    seniorcitizen: Series[int] = Field(isin=[0, 1], nullable=True)
    has_partner: Series[str] = Field(isin=_YES_NO, nullable=True)
    dependents: Series[str] = Field(isin=_YES_NO, nullable=True)
    tenure: Series[int] = Field(ge=0, nullable=True)
    phoneservice: Series[str] = Field(isin=_YES_NO, nullable=True)
    multiplelines: Series[str] = Field(isin=_YES_NO_NO_PHONE, nullable=True)
    internetservice: Series[str] = Field(
        isin={"DSL", "Fiber optic", "No"}, nullable=True
    )
    onlinesecurity: Series[str] = Field(isin=_YES_NO_NO_INTERNET, nullable=True)
    onlinebackup: Series[str] = Field(isin=_YES_NO_NO_INTERNET, nullable=True)
    deviceprotection: Series[str] = Field(isin=_YES_NO_NO_INTERNET, nullable=True)
    techsupport: Series[str] = Field(isin=_YES_NO_NO_INTERNET, nullable=True)
    streamingtv: Series[str] = Field(isin=_YES_NO_NO_INTERNET, nullable=True)
    streamingmovies: Series[str] = Field(isin=_YES_NO_NO_INTERNET, nullable=True)
    contract_type: Series[str] = Field(
        isin={"Month-to-month", "One year", "Two year"}, nullable=True
    )
    paperlessbilling: Series[str] = Field(isin=_YES_NO, nullable=True)
    paymentmethod: Series[str] = Field(
        isin={
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        },
        nullable=True,
    )
    monthlycharges: Series[float] = Field(ge=0.0, nullable=True)
    totalcharges: Series[float] = Field(ge=0.0, nullable=True)
    churn: Series[int] = Field(isin=[0, 1], nullable=False)

    class Config:
        strict = True  # unexpected columns fail loudly — schema drift must be a conscious decision
        coerce = True  # handles Postgres NUMERIC→float and SMALLINT→int coercion


class CleanedSchema(RawSchema):
    """RawSchema with totalcharges required to be non-null.

    Imputation via clean_dataframe() is a prerequisite before validating
    against this schema.
    """

    totalcharges: Series[float] = Field(ge=0.0, nullable=False)

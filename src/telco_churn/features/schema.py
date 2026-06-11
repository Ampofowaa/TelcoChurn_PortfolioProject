"""Pandera schemas for the feature engineering layer.

CustomerFeaturesSchema: validates the SQL-view DataFrame passed into build_feature_df.
FeatureOutputSchema: validates the augmented DataFrame returned by build_feature_df.
FeatureOutputSchema inherits from CustomerFeaturesSchema and adds the four Python-engineered
columns, keeping column constraints in one place.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from pandera.pandas import DataFrameModel, Field
from pandera.typing import Series

from telco_churn.data.schema import YES_NO, YES_NO_NO_INTERNET, YES_NO_NO_PHONE

_TENURE_COHORTS: Final[frozenset[str]] = frozenset(
    {"0–12 mo", "13–24 mo", "25–48 mo", "49+ mo"}
)
_FIBER_CONTRACT_VALUES: Final[frozenset[str]] = frozenset(
    {
        "Month-to-month_Fiber optic",
        "One year_Fiber optic",
        "Two year_Fiber optic",
        "Not Fiber optic",
    }
)
_DSL_CONTRACT_VALUES: Final[frozenset[str]] = frozenset(
    {
        "Month-to-month_DSL",
        "One year_DSL",
        "Two year_DSL",
        "Not DSL",
    }
)


class CustomerFeaturesSchema(DataFrameModel):  # type: ignore[misc]
    """Input schema for build_feature_df — validates the customer_features SQL view output.

    Declares the 21 SQL-sourced feature columns. customerid and churn are expected
    pass-throughs: validated by the data layer (Phase 2) and preserved here for
    downstream traceability. strict=False allows them without triggering a schema error.
    coerce=True handles Postgres NUMERIC → float and SMALLINT → int coercion on read.
    """

    # --- BINARY_COLS (raw pass-through) ---
    gender: Series[str] = Field(isin={"Male", "Female"}, nullable=False)
    seniorcitizen: Series[int] = Field(isin=[0, 1], nullable=False)
    has_partner: Series[str] = Field(isin=YES_NO, nullable=False)
    dependents: Series[str] = Field(isin=YES_NO, nullable=False)
    phoneservice: Series[str] = Field(isin=YES_NO, nullable=False)
    paperlessbilling: Series[str] = Field(isin=YES_NO, nullable=False)

    # --- MULTI_CAT_COLS (raw pass-through) ---
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
    paymentmethod: Series[str] = Field(
        isin={
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        },
        nullable=False,
    )
    tenure_cohort: Series[str] = Field(isin=_TENURE_COHORTS, nullable=False)

    # --- NUMERIC_COLS (raw + SQL-engineered pass-through) ---
    tenure: Series[int] = Field(ge=0, nullable=False)
    monthlycharges: Series[float] = Field(ge=0.0, lt=np.inf, nullable=False)
    totalcharges: Series[float] = Field(
        ge=0.0, nullable=True
    )  # NaN for 11 zero-tenure rows
    charge_per_service: Series[float] = Field(ge=0.0, nullable=False)

    class Config:
        strict = False  # customerid and churn may be present; not validated here
        coerce = True


class FeatureOutputSchema(CustomerFeaturesSchema):
    """Output schema for the DataFrame returned by build_feature_df.

    Inherits the 21 SQL-sourced column constraints from CustomerFeaturesSchema and
    adds the 4 Python-engineered columns produced by _add_python_features (H1–H3).
    strict=False: customerid and churn pass through unchanged — y extraction and
    feature selection are the caller's responsibility (Phase 5 train.py).
    coerce=False: feature column types must be correct by construction; a mismatch
    is a bug, not a cast.
    """

    # --- Python-engineered BINARY ---
    is_long_month_to_month: Series[int] = Field(isin=[0, 1], nullable=False)

    # --- Python-engineered MULTI_CAT ---
    fiber_contract: Series[str] = Field(isin=_FIBER_CONTRACT_VALUES, nullable=False)
    dsl_contract: Series[str] = Field(isin=_DSL_CONTRACT_VALUES, nullable=False)

    # --- Python-engineered NUMERIC ---
    # le=1.0: totalcharges >= monthlycharges for all tenure>=1 customers by construction.
    monthly_to_total_ratio: Series[float] = Field(ge=0.0, le=1.0, nullable=True)

    class Config(CustomerFeaturesSchema.Config):
        coerce = False  # types must already be correct — mismatch is a bug, not a cast

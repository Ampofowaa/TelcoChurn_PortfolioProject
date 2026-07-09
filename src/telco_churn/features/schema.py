"""Pandera schemas and column-group registry for the feature engineering layer.

CustomerFeaturesSchema: validates the SQL-view DataFrame passed into build_feature_df.
FeatureOutputSchema: identical to CustomerFeaturesSchema — the adopted feature
(charge_per_service) is SQL-engineered; no additional columns are computed in Python.
Kept as a distinct exported name so model training imports remain stable when the output
contract is extended.
FeatureSchema: frozen dataclass owning the three column groups that feed the
ColumnTransformer. FEATURE_SCHEMA is the module-level singleton — import it directly
rather than constructing a new instance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pandera.pandas import DataFrameModel, Field
from pandera.typing import Series

from telco_churn.data.schema import YES_NO, YES_NO_NO_INTERNET, YES_NO_NO_PHONE

__all__ = [
    "CustomerFeaturesSchema",
    "FeatureOutputSchema",
    "FeatureSchema",
    "FEATURE_SCHEMA",
]


class CustomerFeaturesSchema(DataFrameModel):  # type: ignore[misc]
    """Input schema for build_feature_df — validates the customer_features SQL view output.

    Declares the 20 SQL-sourced feature columns. customerid and churn are expected
    pass-throughs: validated by the data validation layer and preserved here for
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

    Identical to CustomerFeaturesSchema — the sole adopted feature (charge_per_service)
    is computed in SQL; no additional columns are added in Python.
    Kept as a distinct exported name so model training imports remain stable when the output
    contract is extended.
    """

    class Config(CustomerFeaturesSchema.Config):
        coerce = False  # types must be correct by construction — mismatch is a bug, not a cast


@dataclass(frozen=True)
class FeatureSchema:
    """Immutable column-group registry for the feature engineering pipeline.

    Owns the three groups fed to the ColumnTransformer at training and inference:
    binary (str + int combined), multi_cat, numeric. The Pandera schemas above
    validate row *values*; this class owns column *grouping* — complementary roles
    in one module.

    Use the module-level FEATURE_SCHEMA singleton rather than constructing a new instance.
    frozen=True prevents runtime mutation; evolving the feature set requires explicit
    code changes to FEATURE_SCHEMA and build_feature_df, never accidental side-effects.
    """

    binary: tuple[str, ...]
    multi_cat: tuple[str, ...]
    numeric: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate non-empty and no intra-group duplicates."""
        for field_name, cols in (
            ("binary", self.binary),
            ("multi_cat", self.multi_cat),
            ("numeric", self.numeric),
        ):
            if not cols:
                raise ValueError(f"FeatureSchema.{field_name} must not be empty")
            if len(cols) != len(set(cols)):
                raise ValueError(
                    f"FeatureSchema.{field_name} contains duplicate column names"
                )


FEATURE_SCHEMA = FeatureSchema(
    binary=(
        "gender",
        "has_partner",
        "dependents",
        "phoneservice",
        "paperlessbilling",
        "seniorcitizen",
    ),
    multi_cat=(
        "multiplelines",
        "internetservice",
        "onlinesecurity",
        "onlinebackup",
        "deviceprotection",
        "techsupport",
        "streamingtv",
        "streamingmovies",
        "contract_type",
        "paymentmethod",
    ),
    numeric=(
        "tenure",
        "monthlycharges",
        "totalcharges",
        "charge_per_service",
    ),
)

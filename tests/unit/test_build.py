"""Unit tests for src/telco_churn/features/build.py."""

from __future__ import annotations

import json
import math
import re

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from pandera.errors import SchemaError, SchemaErrors

from telco_churn.features import (
    FEATURE_SCHEMA,
    SQL_FEATURE_COLS,
    TARGET_COL,
    FeatureOutputSchema,
    build_feature_df,
)
from telco_churn.features.build import _reject_if_empty
from telco_churn.utils.paths import get_project_root

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_feature_row(
    tenure: int = 12,
    monthlycharges: float = 29.85,
    totalcharges: float = 358.20,
    churn: int = 0,
) -> dict[str, object]:
    """Minimal row matching the customer_features SQL view schema."""
    phoneservice = "Yes"
    multiplelines = "No"
    internetservice = "DSL"
    onlinesecurity = "Yes"
    onlinebackup = "No"
    deviceprotection = "No"
    techsupport = "No"
    streamingtv = "No"
    streamingmovies = "No"
    service_count = max(
        1,
        int(phoneservice == "Yes")
        + int(multiplelines == "Yes")
        + int(internetservice != "No")
        + int(onlinesecurity == "Yes")
        + int(onlinebackup == "Yes")
        + int(deviceprotection == "Yes")
        + int(techsupport == "Yes")
        + int(streamingtv == "Yes")
        + int(streamingmovies == "Yes"),
    )
    return {
        "customerid": "1111-AAAAA",
        "gender": "Male",
        "seniorcitizen": 0,
        "has_partner": "Yes",
        "dependents": "No",
        "tenure": tenure,
        "phoneservice": phoneservice,
        "multiplelines": multiplelines,
        "internetservice": internetservice,
        "onlinesecurity": onlinesecurity,
        "onlinebackup": onlinebackup,
        "deviceprotection": deviceprotection,
        "techsupport": techsupport,
        "streamingtv": streamingtv,
        "streamingmovies": streamingmovies,
        "contract_type": "Month-to-month",
        "paperlessbilling": "Yes",
        "paymentmethod": "Electronic check",
        "monthlycharges": monthlycharges,
        "totalcharges": totalcharges,
        "churn": churn,
        "charge_per_service": monthlycharges / service_count,
    }


@pytest.fixture
def minimal_df() -> pd.DataFrame:
    """Two-row DataFrame covering both churn classes."""
    return pd.DataFrame(
        [
            _make_feature_row(tenure=12, churn=0),
            _make_feature_row(tenure=36, churn=1),
        ]
    )


# ---------------------------------------------------------------------------
# build_feature_df — shape and content
# ---------------------------------------------------------------------------


def test_build_feature_df_row_count(minimal_df: pd.DataFrame) -> None:
    """Output row count equals input row count."""
    assert build_feature_df(minimal_df).shape[0] == len(minimal_df)


def test_build_feature_df_feature_columns_present(minimal_df: pd.DataFrame) -> None:
    """Output contains all declared feature columns."""
    feat_df = build_feature_df(minimal_df)
    for col in list(
        FEATURE_SCHEMA.binary + FEATURE_SCHEMA.multi_cat + FEATURE_SCHEMA.numeric
    ):
        assert col in feat_df.columns


def test_build_feature_df_no_nan_in_features_for_well_formed_input(
    minimal_df: pd.DataFrame,
) -> None:
    """No NaN values in feature columns when totalcharges is non-null."""
    feat_df = build_feature_df(minimal_df)
    feature_cols = list(
        FEATURE_SCHEMA.binary + FEATURE_SCHEMA.multi_cat + FEATURE_SCHEMA.numeric
    )
    assert not feat_df[feature_cols].isnull().any().any()


def test_build_feature_df_null_totalcharges_is_valid() -> None:
    """NaN in totalcharges is accepted — schema marks it nullable for 11 zero-tenure rows."""
    df = pd.DataFrame(
        [_make_feature_row(tenure=0, totalcharges=float("nan")), _make_feature_row()]
    )
    feat_df = build_feature_df(df)
    assert feat_df["totalcharges"].isna().any()


def test_build_feature_df_preserves_churn(minimal_df: pd.DataFrame) -> None:
    """churn is preserved in the output — y extraction is the caller's responsibility."""
    feat_df = build_feature_df(minimal_df)
    assert TARGET_COL in feat_df.columns
    np.testing.assert_array_equal(
        feat_df[TARGET_COL].to_numpy(), minimal_df[TARGET_COL].to_numpy()
    )


def test_build_feature_df_preserves_customerid(minimal_df: pd.DataFrame) -> None:
    """customerid is preserved in the output for downstream traceability."""
    assert "customerid" in build_feature_df(minimal_df).columns


def test_build_feature_df_does_not_mutate_input(minimal_df: pd.DataFrame) -> None:
    """build_feature_df returns a new DataFrame and leaves the input unchanged."""
    original_cols = set(minimal_df.columns)
    build_feature_df(minimal_df)
    assert set(minimal_df.columns) == original_cols


def test_build_feature_df_empty_dataframe_raises() -> None:
    """build_feature_df raises SchemaError when the input has no columns."""
    with pytest.raises(SchemaError):
        build_feature_df(pd.DataFrame())


# ---------------------------------------------------------------------------
# _reject_if_empty — zero-row guard (schema checks pass vacuously on 0 rows)
# ---------------------------------------------------------------------------


def test_reject_if_empty_raises_on_zero_rows() -> None:
    """_reject_if_empty raises ValueError on a well-formed but zero-row DataFrame."""
    empty_df = pd.DataFrame(columns=list(_make_feature_row().keys()))
    with pytest.raises(ValueError, match="zero rows"):
        _reject_if_empty(empty_df)


def test_reject_if_empty_passes_on_non_empty_df(minimal_df: pd.DataFrame) -> None:
    """_reject_if_empty is a no-op when the DataFrame has rows."""
    _reject_if_empty(minimal_df)


# ---------------------------------------------------------------------------
# build_feature_df — input schema guard (CustomerFeaturesSchema)
# ---------------------------------------------------------------------------


def test_build_feature_df_missing_charge_per_service_raises(
    minimal_df: pd.DataFrame,
) -> None:
    """build_feature_df raises SchemaError when charge_per_service is absent (SQL step skipped)."""
    with pytest.raises((SchemaError, SchemaErrors)):
        build_feature_df(minimal_df.drop(columns=["charge_per_service"]))


def test_build_feature_df_wrong_dtype_on_tenure_raises(
    minimal_df: pd.DataFrame,
) -> None:
    """build_feature_df raises SchemaErrors when tenure cannot be coerced to int."""
    bad_df = minimal_df.copy()
    bad_df["tenure"] = "twelve"
    with pytest.raises(SchemaErrors):
        build_feature_df(bad_df)


# ---------------------------------------------------------------------------
# FeatureOutputSchema — direct schema contract tests
# ---------------------------------------------------------------------------


def test_feature_output_schema_accepts_output_with_churn_and_customerid(
    minimal_df: pd.DataFrame,
) -> None:
    """FeatureOutputSchema validates successfully when churn and customerid are present."""
    FeatureOutputSchema.validate(build_feature_df(minimal_df))


# ---------------------------------------------------------------------------
# Provenance cross-check
# ---------------------------------------------------------------------------


def test_adopted_features_present_in_column_groups() -> None:
    """Every feature in adopted_features.json appears in the column groups.

    Ties the implementation to the Phase 4a adoption decision — fails fast if the
    adoption list changes and build.py is not updated to match.

    tests/fixtures/adopted_features.json is the committed contract; it must be
    updated in the same commit as any change to the adoption list. The full
    provenance artifact (reports/feature_discovery/adopted_features.json) is
    gitignored pipeline output and is intentionally separate from this fixture.
    """
    provenance_path = (
        get_project_root() / "tests" / "fixtures" / "adopted_features.json"
    )
    adopted = json.loads(provenance_path.read_text())["adopted_features"]
    all_feature_cols = set(
        list(FEATURE_SCHEMA.binary + FEATURE_SCHEMA.multi_cat + FEATURE_SCHEMA.numeric)
    )
    for feature in adopted:
        assert (
            feature in all_feature_cols
        ), f"Adopted feature '{feature}' missing from column groups in build.py"


def test_sql_feature_cols_match_customer_features_view() -> None:
    """Every column in SQL_FEATURE_COLS must appear in the customer_features SQL SELECT list.

    Catches the case where a column is added to the typed lists in build.py but the
    matching SQL view column is not added — a mismatch that otherwise only surfaces
    as a SQLAlchemyError at dvc repro runtime.
    """
    sql_path = get_project_root() / "sql" / "features" / "customer_features.sql"
    sql = sql_path.read_text(encoding="utf-8")
    select_block = re.search(r"SELECT\s+(.*?)\s+FROM", sql, re.DOTALL | re.IGNORECASE)
    assert select_block, "Could not find SELECT...FROM in customer_features.sql"
    sql_cols = {m.group(1) for m in re.finditer(r"\b\w+\.(\w+)", select_block.group(1))}
    for col in SQL_FEATURE_COLS:
        assert (
            col in sql_cols
        ), f"'{col}' in SQL_FEATURE_COLS not found in customer_features.sql SELECT list"


# ---------------------------------------------------------------------------
# Hypothesis property tests
# ---------------------------------------------------------------------------

_TENURE_STRATEGY = st.integers(min_value=0, max_value=72)
_MONTHLY_STRATEGY = st.floats(min_value=0.01, max_value=120.0, allow_nan=False)
_TOTAL_STRATEGY = st.one_of(
    st.floats(min_value=0.01, max_value=10_000.0, allow_nan=False),
    st.just(float("nan")),
)


def _make_property_df(tenure: int, monthly: float, total: float) -> pd.DataFrame:
    rows = [
        _make_feature_row(tenure=tenure, monthlycharges=monthly, totalcharges=total),
        _make_feature_row(tenure=24, monthlycharges=50.0, totalcharges=1200.0),
    ]
    return pd.DataFrame(rows)


@given(tenure=_TENURE_STRATEGY, monthly=_MONTHLY_STRATEGY, total=_TOTAL_STRATEGY)
@settings(max_examples=25, deadline=None)
def test_column_count_invariant(tenure: int, monthly: float, total: float) -> None:
    """Column count is identical regardless of input values."""
    assume(math.isnan(total) or total >= monthly)
    baseline = build_feature_df(_make_property_df(12, 29.85, 358.20))
    result = build_feature_df(_make_property_df(tenure, monthly, total))
    assert result.shape[1] == baseline.shape[1]


@given(tenure=_TENURE_STRATEGY, monthly=_MONTHLY_STRATEGY, total=_TOTAL_STRATEGY)
@settings(max_examples=25, deadline=None)
def test_build_feature_df_is_deterministic(
    tenure: int, monthly: float, total: float
) -> None:
    """build_feature_df produces identical output on repeated calls with the same input."""
    assume(math.isnan(total) or total >= monthly)
    df = _make_property_df(tenure, monthly, total)
    pd.testing.assert_frame_equal(build_feature_df(df), build_feature_df(df))

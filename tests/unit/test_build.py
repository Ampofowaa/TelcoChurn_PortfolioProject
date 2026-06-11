"""Unit tests for src/telco_churn/features/build.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pandera.errors import SchemaError, SchemaErrors

from telco_churn.features import (
    BINARY_INT_COLS,
    BINARY_STR_COLS,
    MULTI_CAT_COLS,
    NUMERIC_COLS,
    TARGET_COL,
    FeatureOutputSchema,
    build_feature_df,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _tenure_cohort(tenure: int) -> str:
    """Mirror the SQL CASE boundaries in tenure_buckets.sql."""
    if tenure <= 12:
        return "0–12 mo"
    if tenure <= 24:
        return "13–24 mo"
    if tenure <= 48:
        return "25–48 mo"
    return "49+ mo"


def _make_feature_row(
    tenure: int = 12,
    monthlycharges: float = 29.85,
    totalcharges: float = 358.20,
    contract_type: str = "Month-to-month",
    internetservice: str = "DSL",
    churn: int = 0,
) -> dict[str, object]:
    """Minimal row matching the customer_features view schema."""
    phoneservice = "Yes"
    multiplelines = "No"
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
        "contract_type": contract_type,
        "paperlessbilling": "Yes",
        "paymentmethod": "Electronic check",
        "monthlycharges": monthlycharges,
        "totalcharges": totalcharges,
        "churn": churn,
        "tenure_cohort": _tenure_cohort(tenure),
        "charge_per_service": monthlycharges / service_count,
    }


@pytest.fixture
def minimal_df() -> pd.DataFrame:
    """Two-row DataFrame covering both churn classes."""
    return pd.DataFrame(
        [
            _make_feature_row(tenure=12, churn=0),
            _make_feature_row(tenure=36, contract_type="Two year", churn=1),
        ]
    )


@pytest.fixture
def null_totalcharges_df() -> pd.DataFrame:
    """Row with NULL totalcharges (zero-tenure gotcha)."""
    row = _make_feature_row(tenure=0, totalcharges=float("nan"))
    return pd.DataFrame([row, _make_feature_row()])


# ---------------------------------------------------------------------------
# _add_python_features
# ---------------------------------------------------------------------------


def test_add_python_features_h1_flag_set_for_long_mtm() -> None:
    """is_long_month_to_month is 1 when tenure > 24 and contract is Month-to-month."""
    row = pd.DataFrame([_make_feature_row(tenure=36, contract_type="Month-to-month")])
    out = build_feature_df(row)
    assert out["is_long_month_to_month"].iloc[0] == 1


def test_add_python_features_h1_flag_unset_for_short_tenure() -> None:
    """is_long_month_to_month is 0 when tenure <= 24 even on Month-to-month."""
    row = pd.DataFrame([_make_feature_row(tenure=12, contract_type="Month-to-month")])
    out = build_feature_df(row)
    assert out["is_long_month_to_month"].iloc[0] == 0


def test_add_python_features_h1_flag_unset_for_non_mtm() -> None:
    """is_long_month_to_month is 0 for long-tenure customers not on Month-to-month."""
    row = pd.DataFrame([_make_feature_row(tenure=36, contract_type="Two year")])
    out = build_feature_df(row)
    assert out["is_long_month_to_month"].iloc[0] == 0


def test_add_python_features_h1_flag_unset_for_tenure_24() -> None:
    """is_long_month_to_month is 0 at exactly tenure=24 (boundary: condition is > 24)."""
    row = pd.DataFrame([_make_feature_row(tenure=24, contract_type="Month-to-month")])
    out = build_feature_df(row)
    assert out["is_long_month_to_month"].iloc[0] == 0


def test_add_python_features_h1_flag_set_for_tenure_25() -> None:
    """is_long_month_to_month is 1 at tenure=25 (first value strictly above the boundary)."""
    row = pd.DataFrame([_make_feature_row(tenure=25, contract_type="Month-to-month")])
    out = build_feature_df(row)
    assert out["is_long_month_to_month"].iloc[0] == 1


def test_add_python_features_h2_ratio_computed() -> None:
    """monthly_to_total_ratio equals monthlycharges / totalcharges."""
    row = pd.DataFrame([_make_feature_row(monthlycharges=50.0, totalcharges=200.0)])
    out = build_feature_df(row)
    assert out["monthly_to_total_ratio"].iloc[0] == pytest.approx(0.25)


def test_add_python_features_h2_nan_for_zero_tenure() -> None:
    """monthly_to_total_ratio is NaN when totalcharges is NaN (zero-tenure rows)."""
    row = pd.DataFrame([_make_feature_row(tenure=0, totalcharges=float("nan"))])
    out = build_feature_df(row)
    assert pd.isna(out["monthly_to_total_ratio"].iloc[0])


def test_add_python_features_h2_nan_not_inf_for_zero_totalcharges() -> None:
    """monthly_to_total_ratio is NaN (not inf) when totalcharges is 0.0.

    0.0 / monthlycharges would produce inf without the replace guard. The guard
    coerces 0.0 to NaN so the same SimpleImputer path handles it as the NaN rows.
    """
    row = pd.DataFrame([_make_feature_row(tenure=1, totalcharges=0.0)])
    out = build_feature_df(row)
    assert pd.isna(out["monthly_to_total_ratio"].iloc[0])
    assert not np.isinf(out["monthly_to_total_ratio"].iloc[0])


def test_add_python_features_h3a_fiber_contract_label() -> None:
    """fiber_contract encodes contract_type + '_Fiber optic' for fiber customers."""
    for contract in ["Month-to-month", "One year", "Two year"]:
        row = pd.DataFrame(
            [_make_feature_row(contract_type=contract, internetservice="Fiber optic")]
        )
        out = build_feature_df(row)
        assert out["fiber_contract"].iloc[0] == f"{contract}_Fiber optic"


def test_add_python_features_h3a_fiber_contract_not_fiber() -> None:
    """fiber_contract is 'Not Fiber optic' for non-fiber customers."""
    for internet in ["DSL", "No"]:
        row = pd.DataFrame([_make_feature_row(internetservice=internet)])
        out = build_feature_df(row)
        assert out["fiber_contract"].iloc[0] == "Not Fiber optic"


def test_add_python_features_h3b_dsl_contract_label() -> None:
    """dsl_contract encodes contract_type + '_DSL' for DSL customers."""
    for contract in ["Month-to-month", "One year", "Two year"]:
        row = pd.DataFrame(
            [_make_feature_row(contract_type=contract, internetservice="DSL")]
        )
        out = build_feature_df(row)
        assert out["dsl_contract"].iloc[0] == f"{contract}_DSL"


def test_add_python_features_h3b_dsl_contract_not_dsl() -> None:
    """dsl_contract is 'Not DSL' for non-DSL customers."""
    for internet in ["Fiber optic", "No"]:
        row = pd.DataFrame([_make_feature_row(internetservice=internet)])
        out = build_feature_df(row)
        assert out["dsl_contract"].iloc[0] == "Not DSL"


def test_add_python_features_does_not_mutate_input(minimal_df: pd.DataFrame) -> None:
    """build_feature_df returns a new DataFrame and leaves the input unchanged."""
    original_cols = set(minimal_df.columns)
    build_feature_df(minimal_df)
    assert set(minimal_df.columns) == original_cols


# ---------------------------------------------------------------------------
# build_feature_df — shape and content
# ---------------------------------------------------------------------------


def test_build_feature_df_row_count(minimal_df: pd.DataFrame) -> None:
    """Output row count equals input row count."""
    feat_df = build_feature_df(minimal_df)
    assert feat_df.shape[0] == len(minimal_df)


def test_build_feature_df_feature_columns_present(minimal_df: pd.DataFrame) -> None:
    """Output contains all declared feature columns."""
    feat_df = build_feature_df(minimal_df)
    for col in BINARY_STR_COLS + BINARY_INT_COLS + MULTI_CAT_COLS + NUMERIC_COLS:
        assert col in feat_df.columns


def test_build_feature_df_no_nan_in_features_for_well_formed_input(
    minimal_df: pd.DataFrame,
) -> None:
    """No NaN values in feature columns when totalcharges is non-null."""
    feat_df = build_feature_df(minimal_df)
    feature_cols = BINARY_STR_COLS + BINARY_INT_COLS + MULTI_CAT_COLS + NUMERIC_COLS
    assert not feat_df[feature_cols].isnull().any().any()


def test_build_feature_df_preserves_null_monthly_to_total_ratio(
    null_totalcharges_df: pd.DataFrame,
) -> None:
    """NaN in totalcharges propagates to totalcharges and monthly_to_total_ratio only."""
    feat_df = build_feature_df(null_totalcharges_df)
    assert feat_df["monthly_to_total_ratio"].isna().any()
    assert feat_df["totalcharges"].isna().any()
    null_passthrough = {"monthly_to_total_ratio", "totalcharges"}
    feature_cols = BINARY_STR_COLS + BINARY_INT_COLS + MULTI_CAT_COLS + NUMERIC_COLS
    other_cols = [c for c in feature_cols if c not in null_passthrough]
    assert not feat_df[other_cols].isnull().any().any()


def test_build_feature_df_preserves_churn(minimal_df: pd.DataFrame) -> None:
    """churn is preserved in the output — y extraction is the caller's responsibility."""
    feat_df = build_feature_df(minimal_df)
    assert TARGET_COL in feat_df.columns
    np.testing.assert_array_equal(
        feat_df[TARGET_COL].to_numpy(), minimal_df[TARGET_COL].to_numpy()
    )


def test_build_feature_df_preserves_customerid(minimal_df: pd.DataFrame) -> None:
    """customerid is preserved in the output for downstream traceability."""
    feat_df = build_feature_df(minimal_df)
    assert "customerid" in feat_df.columns


def test_build_feature_df_empty_dataframe_raises() -> None:
    """build_feature_df raises SchemaError when the input has no columns."""
    with pytest.raises(SchemaError):
        build_feature_df(pd.DataFrame())


# ---------------------------------------------------------------------------
# build_feature_df — input schema guard (CustomerFeaturesSchema)
# ---------------------------------------------------------------------------


def test_build_feature_df_missing_tenure_cohort_raises(
    minimal_df: pd.DataFrame,
) -> None:
    """build_feature_df raises SchemaError when tenure_cohort is absent (SQL step skipped)."""
    with pytest.raises((SchemaError, SchemaErrors)):
        build_feature_df(minimal_df.drop(columns=["tenure_cohort"]))


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


def test_build_feature_df_invalid_tenure_cohort_raises(
    minimal_df: pd.DataFrame,
) -> None:
    """build_feature_df raises SchemaError when tenure_cohort holds an invalid category."""
    bad_df = minimal_df.copy()
    bad_df["tenure_cohort"] = "invalid-bucket"
    with pytest.raises((SchemaError, SchemaErrors)):
        build_feature_df(bad_df)


# ---------------------------------------------------------------------------
# FeatureOutputSchema — direct schema contract tests
# ---------------------------------------------------------------------------


def test_feature_output_schema_rejects_missing_feature_column(
    minimal_df: pd.DataFrame,
) -> None:
    """FeatureOutputSchema raises SchemaError when a required feature column is missing."""
    feat_df = build_feature_df(minimal_df)
    with pytest.raises((SchemaError, SchemaErrors)):
        FeatureOutputSchema.validate(feat_df.drop(columns=["is_long_month_to_month"]))


def test_feature_output_schema_accepts_output_with_churn_and_customerid(
    minimal_df: pd.DataFrame,
) -> None:
    """FeatureOutputSchema validates successfully when churn and customerid are present."""
    feat_df = build_feature_df(minimal_df)
    FeatureOutputSchema.validate(feat_df)


def test_feature_output_schema_rejects_ratio_above_one(
    minimal_df: pd.DataFrame,
) -> None:
    """FeatureOutputSchema raises SchemaError when monthly_to_total_ratio > 1.0 (le=1.0 constraint)."""
    feat_df = build_feature_df(minimal_df).copy()
    feat_df["monthly_to_total_ratio"] = 1.5
    with pytest.raises((SchemaError, SchemaErrors)):
        FeatureOutputSchema.validate(feat_df)


# ---------------------------------------------------------------------------
# Hypothesis property tests
# ---------------------------------------------------------------------------

_TENURE_STRATEGY = st.integers(min_value=0, max_value=72)


@st.composite
def _monthly_total_strategy(draw):
    """Draw (monthly, total) enforcing totalcharges >= monthlycharges for non-null, non-zero totals.

    NaN and 0.0 are kept as explicit cases to exercise the zero-tenure and replace(0.0, NaN)
    guard paths in _add_python_features. All other values satisfy the real-data invariant
    that totalcharges accumulates over time and can never be less than the current monthly rate.
    """
    monthly = draw(st.floats(min_value=0.01, max_value=120.0, allow_nan=False))
    total = draw(
        st.one_of(
            st.floats(min_value=monthly, max_value=10_000.0, allow_nan=False),
            st.just(float("nan")),
            st.just(
                0.0
            ),  # exercises the replace(0.0, NaN) guard in _add_python_features
        )
    )
    return monthly, total


def _make_property_df(tenure: int, monthly: float, total: float) -> pd.DataFrame:
    rows = [
        _make_feature_row(tenure=tenure, monthlycharges=monthly, totalcharges=total),
        _make_feature_row(tenure=24, monthlycharges=50.0, totalcharges=1200.0),
    ]
    return pd.DataFrame(rows)


@given(tenure=_TENURE_STRATEGY, monthly_total=_monthly_total_strategy())
@settings(max_examples=100, deadline=1000)
def test_nan_confined_to_monthly_to_total_ratio(
    tenure: int, monthly_total: tuple[float, float]
) -> None:
    """NaN in totalcharges propagates only to totalcharges and monthly_to_total_ratio."""
    monthly, total = monthly_total
    df = _make_property_df(tenure, monthly, total)
    feat_df = build_feature_df(df)
    null_passthrough = {"monthly_to_total_ratio", "totalcharges"}
    feature_cols = BINARY_STR_COLS + BINARY_INT_COLS + MULTI_CAT_COLS + NUMERIC_COLS
    other_cols = [c for c in feature_cols if c not in null_passthrough]
    assert not feat_df[other_cols].isnull().any().any()


@given(tenure=_TENURE_STRATEGY, monthly_total=_monthly_total_strategy())
@settings(max_examples=100, deadline=1000)
def test_column_count_invariant(
    tenure: int, monthly_total: tuple[float, float]
) -> None:
    """Column count is identical regardless of input values."""
    monthly, total = monthly_total
    baseline_df = _make_property_df(12, 29.85, 358.20)
    feat_base = build_feature_df(baseline_df)
    df = _make_property_df(tenure, monthly, total)
    feat = build_feature_df(df)
    assert feat.shape[1] == feat_base.shape[1]


@given(tenure=_TENURE_STRATEGY, monthly_total=_monthly_total_strategy())
@settings(max_examples=100, deadline=1000)
def test_build_feature_df_is_deterministic(
    tenure: int, monthly_total: tuple[float, float]
) -> None:
    """build_feature_df produces identical output on repeated calls with the same input."""
    monthly, total = monthly_total
    df = _make_property_df(tenure, monthly, total)
    out1 = build_feature_df(df)
    out2 = build_feature_df(df)
    pd.testing.assert_frame_equal(out1, out2)

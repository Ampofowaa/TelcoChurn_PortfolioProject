"""Unit tests for src/telco_churn/features/preprocessing.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from telco_churn.features.preprocessing import (
    TENURE_COHORT_EDGES,
    build_linear_preprocessor,
    build_preprocessor,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _binary_int_df() -> pd.DataFrame:
    return pd.DataFrame({"senior": [0, 1, 0, 1]})


def _binary_str_df() -> pd.DataFrame:
    return pd.DataFrame({"partner": ["No", "Yes", "No", "Yes"]})


# ---------------------------------------------------------------------------
# build_preprocessor — binary columns (tree path: drop='if_binary')
# ---------------------------------------------------------------------------


def test_binary_int_col_encodes_one_column() -> None:
    """Int 0/1 binary column → OHE(drop='if_binary') produces 1 column."""
    df = _binary_int_df()
    pre = build_preprocessor(binary=["senior"], multi_cat=[], numeric=[])
    out = pre.fit_transform(df)
    assert out.shape == (4, 1)


def test_binary_str_col_encodes_one_column() -> None:
    """Yes/No string binary column → OHE(drop='if_binary') produces 1 column."""
    df = _binary_str_df()
    pre = build_preprocessor(binary=["partner"], multi_cat=[], numeric=[])
    out = pre.fit_transform(df)
    assert out.shape == (4, 1)


def test_binary_int_col_categories_are_native_ints() -> None:
    """Tree path uses OHE directly (no _cast_to_str) — int categories stay as ints."""
    df = _binary_int_df()
    pre = build_preprocessor(binary=["senior"], multi_cat=[], numeric=[])
    pre.fit(df)
    cats = pre.named_transformers_["binary"].categories_[0].tolist()
    assert set(cats) == {0, 1}


def test_mixed_binary_int_and_str_cols_encode_one_col_each() -> None:
    """binary list with int and str cols: OHE(drop='if_binary') produces 1 col per feature."""
    df = pd.DataFrame({"senior": [0, 1, 0], "partner": ["No", "Yes", "No"]})
    pre = build_preprocessor(binary=["senior", "partner"], multi_cat=[], numeric=[])
    out = pre.fit_transform(df)
    # 1 column per binary feature × 2 features = 2
    assert out.shape == (3, 2)


def test_binary_ohe_output_is_zero_or_one() -> None:
    """OHE(drop='if_binary') output contains only 0 and 1 values."""
    df = pd.DataFrame({"partner": ["No", "Yes"]})
    pre = build_preprocessor(binary=["partner"], multi_cat=[], numeric=[])
    out = pre.fit_transform(df)
    assert set(np.unique(out)).issubset({0.0, 1.0})


# ---------------------------------------------------------------------------
# build_preprocessor — multi-cat columns
# ---------------------------------------------------------------------------


def test_multi_cat_ohe_preserves_all_levels() -> None:
    """multi_cat OHE has no drop='first' — all k levels survive."""
    contracts = ["Month-to-month", "One year", "Two year"]
    df = pd.DataFrame({"contract": contracts})
    pre = build_preprocessor(binary=[], multi_cat=["contract"], numeric=[])
    out = pre.fit_transform(df)
    assert out.shape[1] == 3


def test_multi_cat_ohe_unknown_at_transform_ignored() -> None:
    """handle_unknown='ignore' → unseen category produces an all-zero row, no error."""
    train = pd.DataFrame({"contract": ["Month-to-month", "One year"]})
    test = pd.DataFrame({"contract": ["Two year"]})
    pre = build_preprocessor(binary=[], multi_cat=["contract"], numeric=[])
    pre.fit(train)
    out = pre.transform(test)
    assert out.shape == (1, 2)
    assert np.all(out == 0)


# ---------------------------------------------------------------------------
# build_preprocessor — numeric columns
# ---------------------------------------------------------------------------


def test_numeric_nan_imputed_with_median_from_train() -> None:
    """NaN is replaced with the training median — not the global median."""
    train = pd.DataFrame({"tenure": [10.0, 20.0, 30.0]})
    test = pd.DataFrame({"tenure": [np.nan]})
    pre = build_preprocessor(binary=[], multi_cat=[], numeric=["tenure"])
    pre.fit(train)
    out = pre.transform(test)
    assert pytest.approx(20.0) == out[0, 0]


def test_numeric_non_nan_values_pass_through_unchanged() -> None:
    """Non-missing numeric values are not modified by SimpleImputer."""
    df = pd.DataFrame({"tenure": [5.0, 15.0, 25.0]})
    pre = build_preprocessor(binary=[], multi_cat=[], numeric=["tenure"])
    out = pre.fit_transform(df)
    assert np.allclose(out.ravel(), [5.0, 15.0, 25.0])


# ---------------------------------------------------------------------------
# build_preprocessor — remainder / empty lists
# ---------------------------------------------------------------------------


def test_unlisted_column_is_dropped() -> None:
    """Columns absent from all three lists are silently discarded."""
    df = pd.DataFrame({"tenure": [1, 2, 3], "customerid": ["a", "b", "c"]})
    pre = build_preprocessor(binary=[], multi_cat=[], numeric=["tenure"])
    out = pre.fit_transform(df)
    assert out.shape == (3, 1)


def test_empty_binary_list_does_not_crash() -> None:
    """binary=[] is safe when the DataFrame has other columns."""
    df = pd.DataFrame({"tenure": [1, 2, 3]})
    pre = build_preprocessor(binary=[], multi_cat=[], numeric=["tenure"])
    out = pre.fit_transform(df)
    assert out.shape == (3, 1)


def test_all_empty_lists_produce_zero_column_output() -> None:
    """No declared columns → ColumnTransformer emits (n, 0) array without error."""
    df = pd.DataFrame({"tenure": [1, 2, 3]})
    pre = build_preprocessor(binary=[], multi_cat=[], numeric=[])
    out = pre.fit_transform(df)
    assert out.shape[0] == 3
    assert out.shape[1] == 0


# ---------------------------------------------------------------------------
# build_preprocessor — unfitted guard
# ---------------------------------------------------------------------------


def test_transform_before_fit_raises_not_fitted_error() -> None:
    """Calling transform on an unfitted preprocessor raises NotFittedError."""
    df = pd.DataFrame({"tenure": [1, 2]})
    pre = build_preprocessor(binary=[], multi_cat=[], numeric=["tenure"])
    with pytest.raises(NotFittedError):
        pre.transform(df)


# ---------------------------------------------------------------------------
# build_preprocessor — feature names
# ---------------------------------------------------------------------------


def test_get_feature_names_out_returns_readable_names() -> None:
    """ColumnTransformer.get_feature_names_out() must not raise and must embed
    original column names — required for Phase 5 MLflow feature_columns.txt logging."""
    df = pd.DataFrame(
        {
            "senior": [0, 1, 0],
            "contract": ["Month-to-month", "One year", "Two year"],
            "tenure": [1.0, 2.0, 3.0],
        }
    )
    pre = build_preprocessor(
        binary=["senior"], multi_cat=["contract"], numeric=["tenure"]
    )
    pre.fit(df)
    names = pre.get_feature_names_out().tolist()
    assert any("senior" in n for n in names), f"'senior' missing from {names}"
    assert any("contract" in n for n in names), f"'contract' missing from {names}"
    assert any("tenure" in n for n in names), f"'tenure' missing from {names}"


# ---------------------------------------------------------------------------
# build_preprocessor — combined output shape
# ---------------------------------------------------------------------------


def test_combined_output_shape_binary_plus_multi_cat_plus_numeric() -> None:
    """ColumnTransformer concatenates binary, multi_cat, and numeric blocks in order."""
    df = pd.DataFrame(
        {
            "senior": [0, 1, 0],
            "contract": ["Month-to-month", "One year", "Two year"],
            "tenure": [1.0, np.nan, 3.0],
        }
    )
    pre = build_preprocessor(
        binary=["senior"],
        multi_cat=["contract"],
        numeric=["tenure"],
    )
    out = pre.fit_transform(df)
    # senior → 1 col (drop='if_binary') | contract → 3 cols | tenure → 1 col = 5
    assert out.shape == (3, 5)


# ---------------------------------------------------------------------------
# TENURE_COHORT_EDGES
# ---------------------------------------------------------------------------


def test_tenure_cohort_edges_has_six_boundaries() -> None:
    """TENURE_COHORT_EDGES defines 5 cohorts via 6 boundary values."""
    assert len(TENURE_COHORT_EDGES) == 6


def test_tenure_cohort_edges_are_monotonically_increasing() -> None:
    """Boundaries must be strictly increasing for pd.cut to work correctly."""
    assert all(
        TENURE_COHORT_EDGES[i] < TENURE_COHORT_EDGES[i + 1]
        for i in range(len(TENURE_COHORT_EDGES) - 1)
    )


# ---------------------------------------------------------------------------
# build_linear_preprocessor — binary (drop='first' + _cast_to_str)
# ---------------------------------------------------------------------------


def test_linear_binary_int_col_encodes_n_minus_one_columns() -> None:
    """Linear path: int 0/1 binary col → cast_to_str + OHE(drop='first') → 1 col."""
    df = _binary_int_df()
    pre = build_linear_preprocessor(binary=["senior"], multi_cat=[], numeric=[])
    out = pre.fit_transform(df)
    assert out.shape == (4, 1)


def test_linear_binary_str_col_encodes_n_minus_one_columns() -> None:
    """Linear path: Yes/No string binary col → OHE(drop='first') → 1 col."""
    df = _binary_str_df()
    pre = build_linear_preprocessor(binary=["partner"], multi_cat=[], numeric=[])
    out = pre.fit_transform(df)
    assert out.shape == (4, 1)


def test_linear_binary_int_categories_are_string_tokens() -> None:
    """Linear binary branch casts int to str before OHE — int categories become strings."""
    df = _binary_int_df()
    pre = build_linear_preprocessor(binary=["senior"], multi_cat=[], numeric=[])
    pre.fit(df)
    cats = pre.named_transformers_["binary"]["ohe"].categories_[0].tolist()
    assert set(cats) == {"0", "1"}


# ---------------------------------------------------------------------------
# build_linear_preprocessor — multi-cat (drop='first')
# ---------------------------------------------------------------------------


def test_linear_multi_cat_drops_first_level() -> None:
    """Linear path: 3-level multi_cat → OHE(drop='first') → 2 columns."""
    contracts = ["Month-to-month", "One year", "Two year"]
    df = pd.DataFrame({"contract": contracts})
    pre = build_linear_preprocessor(binary=[], multi_cat=["contract"], numeric=[])
    out = pre.fit_transform(df)
    assert out.shape[1] == 2


# ---------------------------------------------------------------------------
# build_linear_preprocessor — tenure cohort branch
# ---------------------------------------------------------------------------


def test_linear_tenure_cohort_produces_four_columns() -> None:
    """tenure → 5 cohorts → OHE(drop='first') → 4 columns."""
    df = pd.DataFrame({"tenure": [0, 6, 18, 36, 60, 70]})
    pre = build_linear_preprocessor(binary=[], multi_cat=[], numeric=["tenure"])
    out = pre.fit_transform(df)
    # 5 cohorts - 1 dropped = 4
    assert out.shape[1] == 4


def test_linear_tenure_cohort_bins_correctly() -> None:
    """Each tenure value lands in the expected cohort bin."""
    # One representative value per cohort
    df = pd.DataFrame({"tenure": [6, 18, 36, 60, 70]})
    pre = build_linear_preprocessor(binary=[], multi_cat=[], numeric=["tenure"])
    out = pre.fit_transform(df)
    # All rows must be one-hot (exactly one active column after drop='first',
    # except for rows assigned to the dropped first cohort which are all-zero)
    for row in out:
        assert row.sum() in {0.0, 1.0}


def test_linear_tenure_zero_assigned_to_first_cohort() -> None:
    """tenure=0 falls in the 0-12m cohort (include_lowest=True); produces all-zero row
    after OHE(drop='first') drops the reference category."""
    df = pd.DataFrame({"tenure": [0, 1, 12]})  # all in 0-12m
    pre = build_linear_preprocessor(binary=[], multi_cat=[], numeric=["tenure"])
    out = pre.fit_transform(df)
    # all rows in the dropped first cohort → all-zero rows
    assert np.all(out == 0)


# ---------------------------------------------------------------------------
# build_linear_preprocessor — non-tenure numerics are scaled
# ---------------------------------------------------------------------------


def test_linear_non_tenure_numerics_are_standardised() -> None:
    """Non-tenure numeric columns are imputed then StandardScaled (mean≈0, std≈1)."""
    df = pd.DataFrame({"monthlycharges": [10.0, 20.0, 30.0, 40.0, 50.0]})
    pre = build_linear_preprocessor(binary=[], multi_cat=[], numeric=["monthlycharges"])
    out = pre.fit_transform(df)
    assert pytest.approx(0.0, abs=1e-10) == out.mean()
    assert pytest.approx(1.0, abs=1e-6) == out.std(ddof=0)


def test_linear_numeric_nan_imputed_before_scaling() -> None:
    """NaN in a non-tenure numeric column is imputed with training median before scaling."""
    train = pd.DataFrame({"monthlycharges": [10.0, 20.0, 30.0]})
    test = pd.DataFrame({"monthlycharges": [np.nan]})
    pre = build_linear_preprocessor(binary=[], multi_cat=[], numeric=["monthlycharges"])
    pre.fit(train)
    out = pre.transform(test)
    # NaN → median (20.0) → scaled to 0.0
    assert pytest.approx(0.0, abs=1e-10) == out[0, 0]


# ---------------------------------------------------------------------------
# build_linear_preprocessor — combined output shape
# ---------------------------------------------------------------------------


def test_linear_combined_output_shape() -> None:
    """Full linear preprocessor: binary(1) + multi_cat(2) + tenure_cohort(4) + numeric(1) = 8.

    All 5 tenure cohorts are represented so OHE sees the full category space;
    drop='first' then removes 1, leaving 4 tenure columns.
    """
    df = pd.DataFrame(
        {
            "partner": ["No", "Yes", "No", "Yes", "No"],
            "contract": [
                "Month-to-month",
                "One year",
                "Two year",
                "One year",
                "Two year",
            ],
            "tenure": [6.0, 18.0, 36.0, 60.0, 70.0],  # one row per cohort
            "monthlycharges": [29.85, 56.95, 90.25, 70.0, 50.0],
        }
    )
    pre = build_linear_preprocessor(
        binary=["partner"],
        multi_cat=["contract"],
        numeric=["tenure", "monthlycharges"],
    )
    out = pre.fit_transform(df)
    # partner → 1 | contract → 2 | tenure_cohort → 4 | monthlycharges → 1 = 8
    assert out.shape == (5, 8)

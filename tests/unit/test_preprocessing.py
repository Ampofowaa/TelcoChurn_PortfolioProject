"""Unit tests for src/telco_churn/features/preprocessing.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from telco_churn.features.preprocessing import build_preprocessor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _binary_int_df() -> pd.DataFrame:
    return pd.DataFrame({"senior": [0, 1, 0, 1]})


def _binary_str_df() -> pd.DataFrame:
    return pd.DataFrame({"partner": ["No", "Yes", "No", "Yes"]})


# ---------------------------------------------------------------------------
# Binary columns
# ---------------------------------------------------------------------------


def test_binary_int_col_encodes_two_categories() -> None:
    """Int 0/1 column goes through _cast_to_str → OHE produces 2 columns."""
    df = _binary_int_df()
    pre = build_preprocessor(binary=["senior"], multi_cat=[], numeric=[])
    out = pre.fit_transform(df)
    assert out.shape == (4, 2)


def test_binary_str_col_encodes_two_categories() -> None:
    """Yes/No string column → OHE produces 2 columns."""
    df = _binary_str_df()
    pre = build_preprocessor(binary=["partner"], multi_cat=[], numeric=[])
    out = pre.fit_transform(df)
    assert out.shape == (4, 2)


def test_binary_int_categories_are_string_tokens() -> None:
    """_cast_to_str converts int 0/1 to str '0'/'1' before OHE fits categories."""
    df = _binary_int_df()
    pre = build_preprocessor(binary=["senior"], multi_cat=[], numeric=[])
    pre.fit(df)
    cats = pre.named_transformers_["binary"]["ohe"].categories_[0].tolist()
    assert set(cats) == {"0", "1"}


def test_mixed_binary_int_and_str_cols_encode_without_error() -> None:
    """binary list containing both int and str cols passes through _cast_to_str cleanly."""
    df = pd.DataFrame({"senior": [0, 1, 0], "partner": ["No", "Yes", "No"]})
    pre = build_preprocessor(binary=["senior", "partner"], multi_cat=[], numeric=[])
    out = pre.fit_transform(df)
    # 2 categories per column × 2 columns = 4
    assert out.shape == (3, 4)


def test_binary_ohe_encoding_values_are_correct() -> None:
    """OHE output for a binary str column is a valid one-hot matrix."""
    df = pd.DataFrame({"partner": ["No", "Yes"]})
    pre = build_preprocessor(binary=["partner"], multi_cat=[], numeric=[])
    out = pre.fit_transform(df)
    # each row must sum to 1 (exactly one active level)
    assert np.allclose(out.sum(axis=1), 1.0)


# ---------------------------------------------------------------------------
# Multi-cat columns
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
# Numeric columns
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
# remainder="drop"
# ---------------------------------------------------------------------------


def test_unlisted_column_is_dropped() -> None:
    """Columns absent from all three lists are silently discarded."""
    df = pd.DataFrame({"tenure": [1, 2, 3], "customerid": ["a", "b", "c"]})
    pre = build_preprocessor(binary=[], multi_cat=[], numeric=["tenure"])
    out = pre.fit_transform(df)
    # customerid dropped; only tenure column survives
    assert out.shape == (3, 1)


# ---------------------------------------------------------------------------
# Empty lists
# ---------------------------------------------------------------------------


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
# Unfitted guard
# ---------------------------------------------------------------------------


def test_transform_before_fit_raises_not_fitted_error() -> None:
    """Calling transform on an unfitted preprocessor raises NotFittedError."""
    df = pd.DataFrame({"tenure": [1, 2]})
    pre = build_preprocessor(binary=[], multi_cat=[], numeric=["tenure"])
    with pytest.raises(NotFittedError):
        pre.transform(df)


# ---------------------------------------------------------------------------
# get_feature_names_out
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
    assert any(
        "senior" in n for n in names
    ), f"column name 'senior' missing from {names}"
    assert any(
        "contract" in n for n in names
    ), f"column name 'contract' missing from {names}"
    assert any(
        "tenure" in n for n in names
    ), f"column name 'tenure' missing from {names}"


# ---------------------------------------------------------------------------
# Combined output shape
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
    # senior → 2 OHE cols | contract → 3 OHE cols | tenure → 1 col = 6 total
    assert out.shape == (3, 6)

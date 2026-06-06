"""Unit tests for src/telco_churn/data/eda.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from telco_churn.data.eda import (
    BINARY_INT_FEATURES,
    CAT_FEATURES,
    NUM_FEATURES,
    churn_rate_by_group,
    compute_chi2_tests,
    compute_mann_whitney,
    compute_vif,
    correlation_with_target,
    detect_outliers,
    encoded_correlation_matrix,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def eda_df() -> pd.DataFrame:
    """200-row DataFrame with normalized Telco column names and both churn classes."""
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame(
        {
            "customerid": [f"C{i:04d}" for i in range(n)],
            "gender": rng.choice(["Male", "Female"], n).tolist(),
            "seniorcitizen": rng.choice([0, 1], n).tolist(),
            "has_partner": rng.choice(["Yes", "No"], n).tolist(),
            "dependents": rng.choice(["Yes", "No"], n).tolist(),
            "tenure": rng.integers(0, 72, n).tolist(),
            "phoneservice": rng.choice(["Yes", "No"], n).tolist(),
            "multiplelines": rng.choice(["Yes", "No", "No phone service"], n).tolist(),
            "internetservice": rng.choice(["DSL", "Fiber optic", "No"], n).tolist(),
            "onlinesecurity": rng.choice(
                ["Yes", "No", "No internet service"], n
            ).tolist(),
            "onlinebackup": rng.choice(
                ["Yes", "No", "No internet service"], n
            ).tolist(),
            "deviceprotection": rng.choice(
                ["Yes", "No", "No internet service"], n
            ).tolist(),
            "techsupport": rng.choice(["Yes", "No", "No internet service"], n).tolist(),
            "streamingtv": rng.choice(["Yes", "No", "No internet service"], n).tolist(),
            "streamingmovies": rng.choice(
                ["Yes", "No", "No internet service"], n
            ).tolist(),
            "contract_type": rng.choice(
                ["Month-to-month", "One year", "Two year"], n
            ).tolist(),
            "paperlessbilling": rng.choice(["Yes", "No"], n).tolist(),
            "paymentmethod": rng.choice(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                n,
            ).tolist(),
            "monthlycharges": rng.uniform(20, 100, n).tolist(),
            "totalcharges": rng.uniform(0, 7000, n).tolist(),
            "churn": rng.choice([0, 1], n, p=[0.73, 0.27]).tolist(),
        }
    )


# ---------------------------------------------------------------------------
# detect_outliers
# ---------------------------------------------------------------------------


def test_detect_outliers_returns_expected_columns(eda_df: pd.DataFrame) -> None:
    result = detect_outliers(eda_df)
    assert set(result.columns) == {
        "Q1",
        "Q3",
        "IQR",
        "lower_bound",
        "upper_bound",
        "n_outliers",
        "pct_outliers",
    }


def test_detect_outliers_indexed_by_feature(eda_df: pd.DataFrame) -> None:
    result = detect_outliers(eda_df)
    assert set(result.index) == set(NUM_FEATURES)


def test_detect_outliers_counts_non_negative(eda_df: pd.DataFrame) -> None:
    result = detect_outliers(eda_df)
    assert (result["n_outliers"] >= 0).all()


def test_detect_outliers_pct_bounded(eda_df: pd.DataFrame) -> None:
    result = detect_outliers(eda_df)
    assert result["pct_outliers"].between(0, 100).all()


def test_detect_outliers_custom_features(eda_df: pd.DataFrame) -> None:
    result = detect_outliers(eda_df, num_features=["tenure"])
    assert list(result.index) == ["tenure"]


def test_detect_outliers_empty_dataframe(eda_df: pd.DataFrame) -> None:
    empty_df = eda_df.iloc[0:0].copy()
    with pytest.warns(UserWarning, match="no observations"):
        result = detect_outliers(empty_df)
    assert set(result.index) == set(NUM_FEATURES)
    assert (result["n_outliers"] == 0).all()
    assert (result["pct_outliers"] == 0.0).all()
    assert result["Q1"].isna().all()


def test_detect_outliers_handles_nulls(eda_df: pd.DataFrame) -> None:
    df_null = eda_df.copy()
    df_null.loc[0, "tenure"] = float("nan")
    result_with_null = detect_outliers(df_null, num_features=["tenure"])
    result_without_null = detect_outliers(eda_df, num_features=["tenure"])
    # Removing an observation via NaN can only decrease or maintain the outlier count —
    # it can never introduce a new one.
    assert (
        result_with_null.loc["tenure", "n_outliers"]
        <= result_without_null.loc["tenure", "n_outliers"]
    )


def test_detect_outliers_wrong_dtype_raises(eda_df: pd.DataFrame) -> None:
    with pytest.raises(TypeError, match="unsupported operand type"):
        detect_outliers(eda_df, num_features=["gender"])


# ---------------------------------------------------------------------------
# churn_rate_by_group
# ---------------------------------------------------------------------------


def test_churn_rate_by_group_returns_series(eda_df: pd.DataFrame) -> None:
    result = churn_rate_by_group(eda_df, "contract_type")
    assert isinstance(result, pd.Series)


def test_churn_rate_by_group_values_in_0_1(eda_df: pd.DataFrame) -> None:
    result = churn_rate_by_group(eda_df, "contract_type")
    assert result.between(0, 1).all()


def test_churn_rate_by_group_sorted_descending(eda_df: pd.DataFrame) -> None:
    result = churn_rate_by_group(eda_df, "contract_type")
    assert list(result.values) == sorted(result.values, reverse=True)


def test_churn_rate_by_group_index_matches_categories(eda_df: pd.DataFrame) -> None:
    result = churn_rate_by_group(eda_df, "internetservice")
    assert set(result.index) == {"DSL", "Fiber optic", "No"}


def test_churn_rate_by_group_excludes_null_groups(eda_df: pd.DataFrame) -> None:
    df_null = eda_df.copy()
    df_null.loc[0, "contract_type"] = float("nan")
    with pytest.warns(UserWarning, match="NaN"):
        result = churn_rate_by_group(df_null, "contract_type")
    assert not any(pd.isna(k) for k in result.index)


def test_churn_rate_by_group_no_warning_when_no_nulls(
    eda_df: pd.DataFrame, recwarn: pytest.WarningsChecker
) -> None:
    churn_rate_by_group(eda_df, "contract_type")
    assert len(recwarn) == 0


def test_churn_rate_by_group_empty_dataframe_returns_empty_series(
    eda_df: pd.DataFrame,
) -> None:
    result = churn_rate_by_group(eda_df.iloc[0:0], "contract_type")
    assert isinstance(result, pd.Series)
    assert len(result) == 0


def test_churn_rate_by_group_wrong_target_dtype_raises(eda_df: pd.DataFrame) -> None:
    df_bad = eda_df.copy()
    df_bad["churn"] = df_bad["churn"].map({0: "No", 1: "Yes"})
    with pytest.raises(TypeError):
        churn_rate_by_group(df_bad, "contract_type")


# ---------------------------------------------------------------------------
# compute_chi2_tests
# ---------------------------------------------------------------------------


def test_compute_chi2_tests_default_features(eda_df: pd.DataFrame) -> None:
    result = compute_chi2_tests(eda_df)
    assert set(result.columns) == {"feature", "cramers_v", "chi2", "p_value", "dof"}
    assert len(result) == len(CAT_FEATURES) + len(BINARY_INT_FEATURES)


def test_compute_chi2_tests_cramers_v_bounded(eda_df: pd.DataFrame) -> None:
    result = compute_chi2_tests(eda_df)
    assert result["cramers_v"].between(0, 1).all()


def test_compute_chi2_tests_sorted_descending(eda_df: pd.DataFrame) -> None:
    result = compute_chi2_tests(eda_df)
    vals = result["cramers_v"].tolist()
    assert vals == sorted(vals, reverse=True)


def test_compute_chi2_tests_custom_features(eda_df: pd.DataFrame) -> None:
    result = compute_chi2_tests(eda_df, cat_features=["contract_type", "gender"])
    assert len(result) == 2
    assert set(result["feature"]) == {"contract_type", "gender"}


def test_compute_chi2_tests_p_values_bounded(eda_df: pd.DataFrame) -> None:
    result = compute_chi2_tests(eda_df)
    assert result["p_value"].between(0, 1).all()


def test_compute_chi2_tests_handles_nulls(eda_df: pd.DataFrame) -> None:
    df_null = eda_df.copy()
    df_null.loc[0, "contract_type"] = float("nan")
    with pytest.warns(UserWarning, match="NaN"):
        result = compute_chi2_tests(df_null, cat_features=["contract_type"])
    assert len(result) == 1
    assert result.iloc[0]["feature"] == "contract_type"


def test_compute_chi2_tests_empty_dataframe(eda_df: pd.DataFrame) -> None:
    empty_df = eda_df.iloc[0:0].copy()
    with pytest.warns(UserWarning, match="no observations"):
        result = compute_chi2_tests(empty_df, cat_features=["contract_type", "gender"])
    assert len(result) == 0
    assert set(result.columns) == {"feature", "cramers_v", "chi2", "p_value", "dof"}


def test_compute_chi2_tests_skips_constant_column(eda_df: pd.DataFrame) -> None:
    df_const = eda_df.copy()
    df_const["contract_type"] = "Month-to-month"
    with pytest.warns(UserWarning, match="constant"):
        result = compute_chi2_tests(df_const, cat_features=["contract_type", "gender"])
    assert "contract_type" not in result["feature"].values
    assert "gender" in result["feature"].values


def test_compute_chi2_tests_warns_on_sparse_table() -> None:
    rng = np.random.default_rng(0)
    # 20 rows × 4-level feature × 2-level target → expected ~2.5 per cell, reliably < 5
    df_sparse = pd.DataFrame(
        {
            "paymentmethod": rng.choice(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                20,
            ).tolist(),
            "churn": rng.choice([0, 1], 20).tolist(),
        }
    )
    with pytest.warns(UserWarning, match="expected count < 5"):
        result = compute_chi2_tests(df_sparse, cat_features=["paymentmethod"])
    assert len(result) == 1


def test_compute_chi2_tests_all_constant_returns_empty(eda_df: pd.DataFrame) -> None:
    df_const = eda_df.copy()
    df_const["gender"] = "Male"
    df_const["contract_type"] = "Month-to-month"
    with pytest.warns(UserWarning, match="constant"):
        result = compute_chi2_tests(df_const, cat_features=["gender", "contract_type"])
    assert len(result) == 0
    assert set(result.columns) == {"feature", "cramers_v", "chi2", "p_value", "dof"}


def test_compute_chi2_tests_incompatible_column_raises(eda_df: pd.DataFrame) -> None:
    df_bad = eda_df.copy()
    df_bad["contract_type"] = [[1, 2]] * len(eda_df)
    with pytest.raises(ValueError, match="operands could not be broadcast"):
        compute_chi2_tests(df_bad, cat_features=["contract_type"])


# ---------------------------------------------------------------------------
# compute_mann_whitney
# ---------------------------------------------------------------------------


def test_compute_mann_whitney_default_features(eda_df: pd.DataFrame) -> None:
    result = compute_mann_whitney(eda_df)
    expected_cols = {
        "feature",
        "rank_biserial_r",
        "U_stat",
        "p_value",
        "mean_churners",
        "mean_non_churners",
    }
    assert set(result.columns) == expected_cols
    assert len(result) == len(NUM_FEATURES)


def test_compute_mann_whitney_r_bounded(eda_df: pd.DataFrame) -> None:
    result = compute_mann_whitney(eda_df)
    assert result["rank_biserial_r"].between(-1, 1).all()


def test_compute_mann_whitney_sorted_by_abs_r(eda_df: pd.DataFrame) -> None:
    result = compute_mann_whitney(eda_df)
    abs_vals = result["rank_biserial_r"].abs().tolist()
    assert abs_vals == sorted(abs_vals, reverse=True)


def test_compute_mann_whitney_custom_features(eda_df: pd.DataFrame) -> None:
    result = compute_mann_whitney(eda_df, num_features=["tenure"])
    assert len(result) == 1
    assert result.iloc[0]["feature"] == "tenure"


def test_compute_mann_whitney_handles_nulls(eda_df: pd.DataFrame) -> None:
    df_with_nulls = eda_df.copy()
    df_with_nulls.loc[0, "totalcharges"] = float("nan")
    result = compute_mann_whitney(df_with_nulls, num_features=["totalcharges"])
    assert len(result) == 1


def test_compute_mann_whitney_skips_empty_group(eda_df: pd.DataFrame) -> None:
    df_null = eda_df.copy()
    df_null.loc[df_null["churn"] == 1, "totalcharges"] = float("nan")
    with pytest.warns(UserWarning, match="empty group"):
        result = compute_mann_whitney(df_null, num_features=["totalcharges"])
    assert "totalcharges" not in result["feature"].values


def test_compute_mann_whitney_wrong_dtype_raises(eda_df: pd.DataFrame) -> None:
    with pytest.raises(TypeError, match="ufunc.*not supported"):
        compute_mann_whitney(eda_df, num_features=["contract_type"])


# ---------------------------------------------------------------------------
# compute_vif
# ---------------------------------------------------------------------------


def test_compute_vif_empty_dataframe(eda_df: pd.DataFrame) -> None:
    empty_df = eda_df.iloc[0:0].drop(columns=["churn"]).copy()
    with pytest.warns(UserWarning, match="no observations"):
        result = compute_vif(empty_df)
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {"feature", "VIF"}
    assert len(result) == 0


def test_compute_vif_returns_dataframe(eda_df: pd.DataFrame) -> None:
    df_pred = eda_df.drop(columns=["churn"])
    result = compute_vif(df_pred)
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {"feature", "VIF"}


def test_compute_vif_positive_values(eda_df: pd.DataFrame) -> None:
    df_pred = eda_df.drop(columns=["churn"])
    result = compute_vif(df_pred)
    assert result["VIF"].notna().all()
    # VIF = 1/(1-R²) is always ≥ 1 for finite values; inf is valid for perfect collinearity
    assert (result["VIF"] >= 1.0).all()


def test_compute_vif_sorted_descending(eda_df: pd.DataFrame) -> None:
    df_pred = eda_df.drop(columns=["churn"])
    result = compute_vif(df_pred)
    vals = result["VIF"].tolist()
    assert vals == sorted(vals, reverse=True)


def test_compute_vif_custom_numeric_cols(eda_df: pd.DataFrame) -> None:
    df_pred = eda_df.drop(columns=["churn"])
    result = compute_vif(df_pred, num_cols=["tenure", "monthlycharges"])
    features = set(result["feature"])
    assert {"tenure", "monthlycharges"}.issubset(features)
    # seniorcitizen is in the default num_cols but was excluded here
    assert "seniorcitizen" not in features


def test_compute_vif_perfect_collinearity_returns_inf(eda_df: pd.DataFrame) -> None:
    df_collinear = eda_df.drop(columns=["churn"]).copy()
    df_collinear["tenure_double"] = df_collinear["tenure"] * 2
    result = compute_vif(
        df_collinear, num_cols=["tenure", "tenure_double"], cat_cols=[]
    )
    collinear_vifs = result.loc[
        result["feature"].isin(["tenure", "tenure_double"]), "VIF"
    ]
    assert len(collinear_vifs) == 2
    assert (collinear_vifs == float("inf")).all()


def test_compute_vif_warns_on_dropped_nulls(eda_df: pd.DataFrame) -> None:
    df_null = eda_df.copy()
    df_null.loc[0, "totalcharges"] = float("nan")
    with pytest.warns(UserWarning, match="NaN dropped"):
        result = compute_vif(df_null.drop(columns=["churn"]))
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0
    assert set(result.columns) == {"feature", "VIF"}


def test_compute_vif_high_cardinality_cat_col_silently_returns_inf(
    eda_df: pd.DataFrame,
) -> None:
    # customerid has n unique values — one-hot expansion produces more columns than
    # rows, making every regression R²=1 and every VIF=inf with no error raised.
    # The caller is responsible for not passing high-cardinality columns in cat_cols.
    df_pred = eda_df.drop(columns=["churn"])
    result = compute_vif(df_pred, cat_cols=["customerid"])
    assert (result["VIF"] == float("inf")).all()


# ---------------------------------------------------------------------------
# encoded_correlation_matrix
# ---------------------------------------------------------------------------


def test_encoded_correlation_matrix_is_square(eda_df: pd.DataFrame) -> None:
    corr = encoded_correlation_matrix(eda_df)
    assert corr.shape[0] == corr.shape[1]


def test_encoded_correlation_matrix_diagonal_is_one(eda_df: pd.DataFrame) -> None:
    corr = encoded_correlation_matrix(eda_df)
    np.testing.assert_allclose(np.diag(corr.values), 1.0, atol=1e-10)


def test_encoded_correlation_matrix_symmetric(eda_df: pd.DataFrame) -> None:
    corr = encoded_correlation_matrix(eda_df)
    np.testing.assert_allclose(corr.values, corr.T.values, atol=1e-10)


def test_encoded_correlation_matrix_no_customerid(eda_df: pd.DataFrame) -> None:
    corr = encoded_correlation_matrix(eda_df)
    assert "customerid" not in corr.columns


def test_encoded_correlation_matrix_contains_churn(eda_df: pd.DataFrame) -> None:
    corr = encoded_correlation_matrix(eda_df)
    assert "churn" in corr.columns
    assert "churn" in corr.index


def test_encoded_correlation_matrix_empty_dataframe(eda_df: pd.DataFrame) -> None:
    empty_df = eda_df.iloc[0:0].copy()
    corr = encoded_correlation_matrix(empty_df)
    assert corr.shape[0] == corr.shape[1]
    assert corr.isna().all().all()


def test_encoded_correlation_matrix_handles_nulls(eda_df: pd.DataFrame) -> None:
    df_null = eda_df.copy()
    df_null.loc[0, "monthlycharges"] = float("nan")
    corr = encoded_correlation_matrix(df_null)
    assert corr.shape[0] == corr.shape[1]
    np.testing.assert_allclose(np.diag(corr.values), 1.0, atol=1e-10)


def test_encoded_correlation_matrix_wrong_dtype(eda_df: pd.DataFrame) -> None:
    df_bad = eda_df.copy()
    df_bad["tenure"] = "bad"
    result = encoded_correlation_matrix(df_bad)
    assert result.shape[0] == result.shape[1]
    assert np.isnan(result.loc["tenure", "tenure"])


# ---------------------------------------------------------------------------
# correlation_with_target
# ---------------------------------------------------------------------------


@pytest.fixture
def corr_matrix(eda_df: pd.DataFrame) -> pd.DataFrame:
    return encoded_correlation_matrix(eda_df)


def test_correlation_with_target_length_capped(corr_matrix: pd.DataFrame) -> None:
    result = correlation_with_target(corr_matrix, top_n=10)
    assert len(result) == 10


def test_correlation_with_target_sorted_by_abs(corr_matrix: pd.DataFrame) -> None:
    result = correlation_with_target(corr_matrix, top_n=20)
    abs_vals = result.abs().tolist()
    assert abs_vals == sorted(abs_vals, reverse=True)


def test_correlation_with_target_values_bounded(corr_matrix: pd.DataFrame) -> None:
    result = correlation_with_target(corr_matrix, top_n=20)
    assert result.between(-1, 1).all()


def test_correlation_with_target_missing_target_returns_empty(
    eda_df: pd.DataFrame,
) -> None:
    corr_no_churn = encoded_correlation_matrix(eda_df.drop(columns=["churn"]))
    result = correlation_with_target(corr_no_churn)
    assert len(result) == 0


def test_correlation_with_target_top_n_exceeds_features(
    corr_matrix: pd.DataFrame,
) -> None:
    result = correlation_with_target(corr_matrix, top_n=10_000)
    assert len(result) > 0
    assert len(result) < 10_000


def test_correlation_with_target_handles_nulls(eda_df: pd.DataFrame) -> None:
    df_null = eda_df.copy()
    df_null.loc[0, "monthlycharges"] = float("nan")
    corr = encoded_correlation_matrix(df_null)
    result = correlation_with_target(corr, top_n=20)
    assert isinstance(result, pd.Series)
    assert result.notna().all()
    assert result.between(-1, 1).all()


def test_correlation_with_target_wrong_dtype(eda_df: pd.DataFrame) -> None:
    df_bad = eda_df.copy()
    df_bad["tenure"] = "bad"
    corr = encoded_correlation_matrix(df_bad)
    result = correlation_with_target(corr, top_n=20)
    assert isinstance(result, pd.Series)


def test_correlation_with_target_empty_dataframe(eda_df: pd.DataFrame) -> None:
    empty_corr = encoded_correlation_matrix(eda_df.iloc[0:0])
    result = correlation_with_target(empty_corr, top_n=20)
    assert isinstance(result, pd.Series)
    assert len(result) == 0


def test_correlation_with_target_excludes_churn(corr_matrix: pd.DataFrame) -> None:
    result = correlation_with_target(corr_matrix, top_n=20)
    assert "churn" not in result.index

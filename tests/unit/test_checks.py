"""Unit tests for telco_churn.data.checks gate functions and result types."""

from __future__ import annotations

import pandas as pd

from telco_churn.data.checks import (
    Severity,
    check_churn_labels,
    check_distribution_sanity,
    check_duplicate_ids,
    check_schema,
    check_totalcharges_nulls,
)

# ---------------------------------------------------------------------------
# Gate 1 — schema
# ---------------------------------------------------------------------------


def test_valid_dataframe_passes_schema_check(valid_raw_df: pd.DataFrame) -> None:
    """A well-formed DataFrame passes Gate 1 without errors."""
    result = check_schema(valid_raw_df)
    assert result.passed is True
    assert result.failure_severity == Severity.ERROR


def test_missing_column_fails_schema_check(valid_raw_df: pd.DataFrame) -> None:
    """Dropping a required column triggers a blocking schema error."""
    df = valid_raw_df.drop(columns=["tenure"])
    result = check_schema(df)
    assert result.passed is False
    assert result.failure_severity == Severity.ERROR


def test_invalid_gender_value_fails_schema_check(valid_raw_df: pd.DataFrame) -> None:
    """An unrecognised gender value triggers a blocking schema error."""
    df = valid_raw_df.copy()
    df.loc[0, "gender"] = "Other"
    result = check_schema(df)
    assert result.passed is False


def test_negative_tenure_fails_schema_check(valid_raw_df: pd.DataFrame) -> None:
    """Negative tenure violates the pandera range check and fails Gate 1."""
    df = valid_raw_df.copy()
    df.loc[0, "tenure"] = -1
    result = check_schema(df)
    assert result.passed is False


def test_cleaned_schema_rejects_null_totalcharges(valid_raw_df: pd.DataFrame) -> None:
    """CleanedSchema rejects a DataFrame that still has NULL totalcharges."""
    df = valid_raw_df.copy()
    df.loc[0, "totalcharges"] = float("nan")
    result = check_schema(df, cleaned=True)
    assert result.passed is False


def test_raw_schema_accepts_null_totalcharges(
    zero_tenure_df: pd.DataFrame,
) -> None:
    """RawSchema accepts NULL totalcharges (expected for zero-tenure customers)."""
    result = check_schema(zero_tenure_df, cleaned=False)
    assert result.passed is True


def test_schema_check_failure_populates_detail(valid_raw_df: pd.DataFrame) -> None:
    """check_schema attaches the full pandera failure_cases DataFrame on failure."""
    df = valid_raw_df.copy()
    df.loc[0, "gender"] = "Unknown"
    result = check_schema(df)
    assert result.passed is False
    assert result.detail is not None
    assert len(result.detail) > 0


def test_schema_check_pass_has_no_detail(valid_raw_df: pd.DataFrame) -> None:
    """check_schema leaves detail as None when validation passes."""
    result = check_schema(valid_raw_df)
    assert result.passed is True
    assert result.detail is None


# ---------------------------------------------------------------------------
# Gate 2 — duplicate IDs
# ---------------------------------------------------------------------------


def test_unique_ids_pass_duplicate_check(valid_raw_df: pd.DataFrame) -> None:
    """Unique customerid values pass Gate 2."""
    result = check_duplicate_ids(valid_raw_df)
    assert result.passed is True


def test_duplicate_customerid_is_blocking_error(valid_raw_df: pd.DataFrame) -> None:
    """Duplicate customerid triggers a blocking ERROR with detail DataFrame."""
    df = pd.concat([valid_raw_df, valid_raw_df.iloc[[0]]], ignore_index=True)
    result = check_duplicate_ids(df)
    assert result.passed is False
    assert result.failure_severity == Severity.ERROR
    assert result.affected_rows == 1
    assert result.detail is not None
    # keep=False includes both the original and the duplicate row
    assert len(result.detail) == 2
    assert result.detail["customerid"].nunique() == 1


def test_missing_customerid_column_is_error() -> None:
    """Gate 2 reports an ERROR when customerid column is absent."""
    df = pd.DataFrame({"churn": [0, 1]})
    result = check_duplicate_ids(df)
    assert result.passed is False
    assert result.failure_severity == Severity.ERROR


# ---------------------------------------------------------------------------
# Gate 3 — churn labels
# ---------------------------------------------------------------------------


def test_valid_churn_labels_pass(valid_raw_df: pd.DataFrame) -> None:
    """Binary churn labels with no missing values pass Gate 3."""
    result = check_churn_labels(valid_raw_df)
    assert result.passed is True


def test_churn_value_2_is_blocking_error(valid_raw_df: pd.DataFrame) -> None:
    """A churn value outside {0, 1} triggers a blocking ERROR with detail."""
    df = valid_raw_df.copy()
    df.loc[0, "churn"] = 2
    result = check_churn_labels(df)
    assert result.passed is False
    assert result.failure_severity == Severity.ERROR
    assert result.affected_rows >= 1
    assert result.detail is not None
    assert len(result.detail) == 1


def test_missing_churn_value_is_blocking_error(valid_raw_df: pd.DataFrame) -> None:
    """A NULL churn value triggers a blocking ERROR with detail."""
    df = valid_raw_df.copy()
    df["churn"] = df["churn"].astype(float)
    df.loc[0, "churn"] = float("nan")
    result = check_churn_labels(df)
    assert result.passed is False
    assert result.failure_severity == Severity.ERROR
    assert result.detail is not None
    assert len(result.detail) == 1


def test_missing_churn_column_is_error() -> None:
    """Gate 3 reports an ERROR when the churn column is absent."""
    df = pd.DataFrame({"customerid": ["x"]})
    result = check_churn_labels(df)
    assert result.passed is False
    assert result.failure_severity == Severity.ERROR


# ---------------------------------------------------------------------------
# Gate 4 — totalcharges unexpected nulls
# ---------------------------------------------------------------------------


def test_valid_totalcharges_pass_null_check(valid_raw_df: pd.DataFrame) -> None:
    """No unexpected NULL totalcharges passes Gate 4."""
    result = check_totalcharges_nulls(valid_raw_df)
    assert result.passed is True


def test_null_totalcharges_for_zero_tenure_passes(
    zero_tenure_df: pd.DataFrame,
) -> None:
    """NULL totalcharges on a zero-tenure row does not trigger Gate 4."""
    result = check_totalcharges_nulls(zero_tenure_df)
    assert result.passed is True


def test_null_totalcharges_for_nonzero_tenure_is_warning(
    valid_raw_df: pd.DataFrame,
) -> None:
    """NULL totalcharges for a non-zero-tenure row triggers a WARNING with detail."""
    df = valid_raw_df.copy()
    df.loc[0, "totalcharges"] = float("nan")  # tenure is 12 on this row
    result = check_totalcharges_nulls(df)
    assert result.passed is False
    assert result.failure_severity == Severity.WARNING
    assert result.detail is not None
    assert len(result.detail) == 1


# ---------------------------------------------------------------------------
# Gate 5 — distribution sanity
# ---------------------------------------------------------------------------


def test_large_valid_df_passes_distribution_check(
    large_valid_df: pd.DataFrame,
) -> None:
    """A 1001-row valid DataFrame clears the row-count threshold."""
    results = check_distribution_sanity(large_valid_df)
    row_check = next(r for r in results if r.name == "row_count")
    assert row_check.passed is True


def test_low_row_count_is_warning(valid_raw_df: pd.DataFrame) -> None:
    """Fewer than min_rows rows triggers a WARNING, not an ERROR."""
    results = check_distribution_sanity(valid_raw_df, min_rows=1_000)
    row_check = next(r for r in results if r.name == "row_count")
    assert row_check.passed is False
    assert row_check.failure_severity == Severity.WARNING


def test_zero_rows_is_blocking_error(empty_telco_df: pd.DataFrame) -> None:
    """An empty DataFrame triggers a blocking ERROR in Gate 5."""
    results = check_distribution_sanity(empty_telco_df)
    assert len(results) == 1
    assert results[0].name == "row_count"
    assert results[0].failure_severity == Severity.ERROR
    assert results[0].passed is False


def test_high_null_rate_in_critical_column_is_warning(
    valid_raw_df: pd.DataFrame,
) -> None:
    """Null rate above threshold in a critical column triggers a WARNING with detail."""
    df = valid_raw_df.copy()
    df["tenure"] = float("nan")
    results = check_distribution_sanity(df, min_rows=1, max_null_rate=0.05)
    tenure_check = next((r for r in results if r.name == "null_rate_tenure"), None)
    assert tenure_check is not None
    assert tenure_check.passed is False
    assert tenure_check.failure_severity == Severity.WARNING
    assert tenure_check.detail is not None
    assert len(tenure_check.detail) == len(df)

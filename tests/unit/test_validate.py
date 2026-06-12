"""Unit tests for telco_churn.data.validate wrappers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from telco_churn.data.checks import CheckResult, Severity
from telco_churn.data.validate import (
    ValidationError,
    ValidationResult,
    save_validation_report,
    validate_clean,
    validate_raw,
)

# ---------------------------------------------------------------------------
# ValidationResult properties
# ---------------------------------------------------------------------------


def test_validation_result_passed_when_no_errors() -> None:
    """ValidationResult.can_proceed is True when all checks pass or only warnings fail."""
    result = ValidationResult(
        checks=[
            CheckResult("a", Severity.ERROR, passed=True, message="ok"),
            CheckResult("b", Severity.WARNING, passed=False, message="warn"),
        ]
    )
    assert result.can_proceed is True


def test_validation_result_not_passed_when_error_present() -> None:
    """ValidationResult.can_proceed is False when at least one ERROR check fails."""
    result = ValidationResult(
        checks=[
            CheckResult("a", Severity.ERROR, passed=False, message="bad"),
        ]
    )
    assert result.can_proceed is False


def test_validation_result_errors_filters_correctly() -> None:
    """ValidationResult.errors returns only failed ERROR-severity checks."""
    err = CheckResult("e", Severity.ERROR, passed=False, message="x")
    warn = CheckResult("w", Severity.WARNING, passed=False, message="y")
    ok = CheckResult("o", Severity.ERROR, passed=True, message="z")
    result = ValidationResult(checks=[err, warn, ok])
    assert result.errors == [err]


def test_validation_result_warnings_filters_correctly() -> None:
    """ValidationResult.warnings returns only failed WARNING-severity checks."""
    err = CheckResult("e", Severity.ERROR, passed=False, message="x")
    warn = CheckResult("w", Severity.WARNING, passed=False, message="y")
    result = ValidationResult(checks=[err, warn])
    assert result.warnings == [warn]


# ---------------------------------------------------------------------------
# validate_raw
# ---------------------------------------------------------------------------


def test_valid_df_passes_validate_raw(
    valid_raw_df: pd.DataFrame, tmp_path: Path
) -> None:
    """A well-formed DataFrame passes validate_raw (errors=0, strict or not)."""
    result = validate_raw(valid_raw_df, strict=False, reports_dir=tmp_path)
    assert result.can_proceed is True
    assert len(result.errors) == 0


def test_strict_mode_raises_on_blocking_error(
    valid_raw_df: pd.DataFrame, tmp_path: Path
) -> None:
    """validate_raw with strict=True raises ValidationError when errors exist."""
    df = valid_raw_df.copy()
    df.loc[0, "churn"] = 9
    with pytest.raises(ValidationError) as exc_info:
        validate_raw(df, strict=True, reports_dir=tmp_path)
    assert len(exc_info.value.result.errors) >= 1


def test_non_strict_mode_returns_result_with_errors(
    valid_raw_df: pd.DataFrame, tmp_path: Path
) -> None:
    """validate_raw with strict=False returns a result even when errors exist."""
    df = valid_raw_df.copy()
    df.loc[0, "churn"] = 9
    result = validate_raw(df, strict=False, reports_dir=tmp_path)
    assert result.can_proceed is False
    assert len(result.errors) >= 1


def test_empty_df_fails_validate_raw(
    empty_telco_df: pd.DataFrame, tmp_path: Path
) -> None:
    """An empty DataFrame fails validate_raw with a blocking ERROR."""
    result = validate_raw(empty_telco_df, strict=False, reports_dir=tmp_path)
    assert result.can_proceed is False
    error_names = {e.name for e in result.errors}
    assert "row_count" in error_names


# ---------------------------------------------------------------------------
# validate_clean
# ---------------------------------------------------------------------------


def test_validate_clean_fails_when_nulls_remain(
    valid_raw_df: pd.DataFrame, tmp_path: Path
) -> None:
    """validate_clean returns an error when totalcharges still contains NULLs."""
    df = valid_raw_df.copy()
    df.loc[0, "totalcharges"] = float("nan")
    result = validate_clean(df, strict=False, reports_dir=tmp_path)
    assert result.can_proceed is False


def test_validate_clean_strict_raises_on_blocking_error(
    valid_raw_df: pd.DataFrame, tmp_path: Path
) -> None:
    """validate_clean with strict=True raises ValidationError when errors exist."""
    df = valid_raw_df.copy()
    df.loc[0, "totalcharges"] = float("nan")
    with pytest.raises(ValidationError) as exc_info:
        validate_clean(df, strict=True, reports_dir=tmp_path)
    assert len(exc_info.value.result.errors) >= 1


# ---------------------------------------------------------------------------
# save_validation_report
# ---------------------------------------------------------------------------


def test_save_validation_report_creates_summary_on_failure(
    valid_raw_df: pd.DataFrame, tmp_path: Path
) -> None:
    """save_validation_report writes summary.csv when checks fail."""
    df = valid_raw_df.copy()
    df.loc[0, "churn"] = 9
    result = validate_raw(df, strict=False, reports_dir=tmp_path)
    report_dir = save_validation_report(result, base_dir=tmp_path)
    assert report_dir is not None
    assert (report_dir / "summary.csv").exists()
    summary = pd.read_csv(report_dir / "summary.csv")
    assert len(summary) >= 1
    assert "check" in summary.columns
    assert "failure_severity" in summary.columns
    assert "message" in summary.columns


def test_save_validation_report_writes_schema_detail_csv(
    valid_raw_df: pd.DataFrame, tmp_path: Path
) -> None:
    """save_validation_report writes schema_failures.csv when schema check fails."""
    df = valid_raw_df.copy()
    df.loc[0, "gender"] = "Unknown"
    result = validate_raw(df, strict=False, reports_dir=tmp_path)
    report_dir = save_validation_report(result, base_dir=tmp_path)
    assert report_dir is not None
    assert (report_dir / "schema_failures.csv").exists()
    detail = pd.read_csv(report_dir / "schema_failures.csv")
    assert len(detail) >= 1


def test_save_validation_report_returns_none_when_all_pass(
    large_valid_df: pd.DataFrame, tmp_path: Path
) -> None:
    """save_validation_report returns None and creates no files when all checks pass."""
    result = validate_raw(large_valid_df, strict=False, reports_dir=tmp_path)
    report_dir = save_validation_report(result, base_dir=tmp_path)
    assert report_dir is None

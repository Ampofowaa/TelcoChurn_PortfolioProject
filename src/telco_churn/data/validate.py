"""Two-tier validation: blocking ERRORs stop the pipeline; WARNINGs are logged and continue."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from telco_churn.data.checks import (
    CheckResult,
    Severity,
    check_churn_labels,
    check_distribution_sanity,
    check_duplicate_ids,
    check_schema,
    check_totalcharges_nulls,
)
from telco_churn.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Aggregated outcome of all gates for one validation call."""

    checks: list[CheckResult] = field(default_factory=list)

    @property
    def errors(self) -> list[CheckResult]:
        """Blocking (ERROR-severity) checks that failed."""
        return [
            c
            for c in self.checks
            if c.failure_severity == Severity.ERROR and not c.passed
        ]

    @property
    def warnings(self) -> list[CheckResult]:
        """Non-blocking (WARNING-severity) checks that failed."""
        return [
            c
            for c in self.checks
            if c.failure_severity == Severity.WARNING and not c.passed
        ]

    @property
    def can_proceed(self) -> bool:
        """True when no ERROR-severity checks failed — pipeline may continue."""
        return len(self.errors) == 0


class ValidationError(Exception):
    """Raised by strict validation when blocking (ERROR-severity) gate(s) fail."""

    def __init__(self, result: ValidationResult) -> None:
        self.result = result
        msgs = "; ".join(e.message for e in result.errors)
        super().__init__(f"{len(result.errors)} blocking validation error(s): {msgs}")


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Impute NULL totalcharges with the column median.

    The 11 zero-tenure customers in the IBM Telco dataset have whitespace
    TotalCharges in the source CSV, which ingest.py coerces to NaN. Imputing
    with the median preserves these rows rather than dropping them.

    Phase 2 placeholder — used to make validate_clean() testable before the
    Phase 4 feature engineering stage exists. In the DVC pipeline, imputation
    is performed by a fitted sklearn SimpleImputer in the features stage, which
    stores the training-set median and applies it consistently to future batches.
    """
    out = df.copy()
    if out["totalcharges"].notna().any():
        median_tc = float(out["totalcharges"].median())
        out["totalcharges"] = out["totalcharges"].fillna(median_tc)
    return out


_REPORTS_DIR = Path("reports/validation")


def save_validation_report(
    result: ValidationResult,
    base_dir: Path = _REPORTS_DIR,
) -> Path | None:
    """Save a timestamped validation report whenever any check fails.

    Creates a directory under base_dir named by the current timestamp and writes:
    - summary.csv: every failing check with its severity, message, and affected_rows
    - <check_name>_failures.csv: row-level detail for any check that carries one
      (currently only the schema check, which attaches pandera's failure_cases)

    Returns the report directory path, or None if all checks passed.
    """
    failing = [c for c in result.checks if not c.passed]
    if not failing:
        return None

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    report_dir: Path = base_dir / timestamp
    report_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame(
        [
            {
                "check": c.name,
                "failure_severity": str(c.failure_severity),
                "message": c.message,
                "affected_rows": c.affected_rows,
            }
            for c in failing
        ]
    )
    summary.to_csv(report_dir / "summary.csv", index=False)

    for check in failing:
        if check.detail is not None:
            check.detail.to_csv(report_dir / f"{check.name}_failures.csv", index=False)

    logger.info(
        "validation_report_saved",
        path=str(report_dir),
        failing_checks=len(failing),
    )
    return report_dir


def _log_result(result: ValidationResult) -> None:
    """Emit structured log events for all failed checks."""
    for check in result.errors:
        logger.error(
            "validation_error",
            check=check.name,
            message=check.message,
            affected_rows=check.affected_rows,
        )
    for check in result.warnings:
        logger.warning(
            "validation_warning",
            check=check.name,
            message=check.message,
            affected_rows=check.affected_rows,
        )


def validate_raw(
    df: pd.DataFrame,
    strict: bool = True,
    reports_dir: Path = _REPORTS_DIR,
) -> ValidationResult:
    """Run the five EDA data-quality gates against raw telco data.

    Args:
        df: Raw DataFrame loaded from the database via
            pd.read_sql("SELECT * FROM customers_raw", engine).
        strict: Raise ValidationError when any ERROR-severity gate fails.
        reports_dir: Directory for validation report artifacts. Override in
            tests to avoid writing to the real filesystem (pass tmp_path).

    Returns:
        ValidationResult with all gate outcomes categorised by severity.

    Raises:
        ValidationError: When strict=True and at least one ERROR gate fails.
    """
    checks: list[CheckResult] = [
        check_schema(df, cleaned=False),
        check_duplicate_ids(df),
        check_churn_labels(df),
        check_totalcharges_nulls(df),
        *check_distribution_sanity(df),
    ]
    result = ValidationResult(checks=checks)
    _log_result(result)
    save_validation_report(result, base_dir=reports_dir)
    if strict and not result.can_proceed:
        raise ValidationError(result)
    return result


def validate_clean(
    df: pd.DataFrame,
    strict: bool = True,
    reports_dir: Path = _REPORTS_DIR,
) -> ValidationResult:
    """Run the five EDA data-quality gates after totalcharges imputation.

    Called after the Phase 4 features stage has imputed totalcharges with
    the training-set median. Validates that no NULL totalcharges remain
    before the rest of feature engineering proceeds.

    Args:
        df: DataFrame with totalcharges imputed — all other columns remain
            in their raw form as loaded from customers_raw.
        strict: Raise ValidationError when any ERROR-severity gate fails.
        reports_dir: Directory for validation report artifacts. Override in
            tests to avoid writing to the real filesystem (pass tmp_path).

    Returns:
        ValidationResult with all gate outcomes categorised by severity.

    Raises:
        ValidationError: When strict=True and at least one ERROR gate fails.
    """
    checks: list[CheckResult] = [
        check_schema(df, cleaned=True),
        check_duplicate_ids(df),
        check_churn_labels(df),
        check_totalcharges_nulls(df),
        *check_distribution_sanity(df),
    ]
    result = ValidationResult(checks=checks)
    _log_result(result)
    save_validation_report(result, base_dir=reports_dir)
    if strict and not result.can_proceed:
        raise ValidationError(result)
    return result


if __name__ == "__main__":
    import sys

    from dotenv import load_dotenv

    from telco_churn.utils.db import get_engine
    from telco_churn.utils.logging import configure_logging

    load_dotenv()
    configure_logging()

    engine = get_engine()
    df = pd.read_sql("SELECT * FROM customers_raw", engine)
    result = validate_raw(df, strict=False)

    if result.can_proceed:
        logger.info("validation_passed", warnings=len(result.warnings))
        sys.exit(0)
    else:
        logger.error(
            "validation_failed",
            errors=len(result.errors),
            warnings=len(result.warnings),
        )
        sys.exit(1)

"""Two-tier validation gate definitions and result types for raw and cleaned Telco data."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Final

import pandas as pd
from pandera.errors import SchemaErrors

from telco_churn.data.schema import CleanedSchema, RawSchema

__all__ = [
    "Severity",
    "CheckResult",
    "MIN_ROWS",
    "MAX_NULL_RATE",
    "check_schema",
    "check_duplicate_ids",
    "check_churn_labels",
    "check_totalcharges_nulls",
    "check_distribution_sanity",
]

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class Severity(enum.StrEnum):
    """Severity attached to each validation gate."""

    ERROR = "ERROR"  # Blocking — stop the pipeline on failure.
    WARNING = "WARNING"  # Non-blocking — log and continue on failure.


@dataclass(frozen=True)
class CheckResult:
    """Outcome of a single validation gate."""

    name: str
    failure_severity: Severity
    passed: bool
    message: str
    affected_rows: int = 0
    detail: pd.DataFrame | None = field(
        default=None, hash=False, compare=False, repr=False
    )


# ---------------------------------------------------------------------------
# Gate constants
# ---------------------------------------------------------------------------

# Columns checked for high null rates in Gate 5.
# Derived from RawSchema so the set stays in sync with schema changes automatically.
# totalcharges excluded: nullable=True in RawSchema (11 zero-tenure NULLs expected).
# customerid excluded: Gate 1 (schema) and Gate 2 (duplicate IDs) already cover it;
# including it here would triple-report the same problem in the validation report.
_NULL_CHECKED_COLS: Final[frozenset[str]] = frozenset(
    col_name
    for col_name, col_schema in RawSchema.to_schema().columns.items()
    if not col_schema.nullable and col_name != "customerid"
)

MIN_ROWS: Final[int] = 1_000
MAX_NULL_RATE: Final[float] = 0.05


# ---------------------------------------------------------------------------
# Gate functions
# ---------------------------------------------------------------------------


def check_schema(df: pd.DataFrame, cleaned: bool = False) -> CheckResult:
    """Gate 1 — column presence, types, value ranges, and categorical values.

    Uses RawSchema when cleaned=False, CleanedSchema when cleaned=True.
    Severity: ERROR — a schema failure means downstream stages cannot safely run.
    """
    try:
        if cleaned:
            CleanedSchema.validate(df, lazy=True)
        else:
            RawSchema.validate(df, lazy=True)
        return CheckResult(
            name="schema",
            failure_severity=Severity.ERROR,
            passed=True,
            message="Schema validation passed.",
        )
    except SchemaErrors as exc:
        failure_df: pd.DataFrame = exc.failure_cases
        n_failures = len(failure_df)
        affected = (
            int(failure_df["index"].dropna().nunique())
            if "index" in failure_df.columns
            else 0
        )
        first = failure_df.iloc[0].to_dict() if n_failures > 0 else {}
        return CheckResult(
            name="schema",
            failure_severity=Severity.ERROR,
            passed=False,
            message=(
                f"Schema validation failed: {n_failures} violation(s) across "
                f"{affected} row(s). First failure: {first}"
            ),
            affected_rows=affected,
            detail=failure_df,
        )


def check_duplicate_ids(df: pd.DataFrame) -> CheckResult:
    """Gate 2 — customerid uniqueness.

    Severity: ERROR — duplicate IDs corrupt feature joins and prediction logging.
    """
    if "customerid" not in df.columns:
        return CheckResult(
            name="duplicate_ids",
            failure_severity=Severity.ERROR,
            passed=False,
            message="Column 'customerid' is missing — cannot check for duplicates.",
        )
    n_dupes = int(df["customerid"].duplicated().sum())
    if n_dupes == 0:
        return CheckResult(
            name="duplicate_ids",
            failure_severity=Severity.ERROR,
            passed=True,
            message="No duplicate customerid values.",
        )
    detail = df[df["customerid"].duplicated(keep=False)].copy()
    return CheckResult(
        name="duplicate_ids",
        failure_severity=Severity.ERROR,
        passed=False,
        message=f"Found {n_dupes} duplicate customerid row(s).",
        affected_rows=n_dupes,
        detail=detail,
    )


def check_churn_labels(df: pd.DataFrame) -> CheckResult:
    """Gate 3 — churn is binary (0/1) with no missing values.

    Severity: ERROR — target corruption makes training and evaluation meaningless.
    """
    if "churn" not in df.columns:
        return CheckResult(
            name="churn_labels",
            failure_severity=Severity.ERROR,
            passed=False,
            message="Column 'churn' is missing — cannot validate labels.",
        )
    missing = int(df["churn"].isna().sum())
    bad = int((~df["churn"].isin([0, 1])).sum())
    total = missing + bad
    if total == 0:
        return CheckResult(
            name="churn_labels",
            failure_severity=Severity.ERROR,
            passed=True,
            message="Churn labels are valid (binary, no missing values).",
        )
    parts: list[str] = []
    if missing:
        parts.append(f"{missing} missing")
    if bad:
        parts.append(f"{bad} out-of-range")
    detail = df[df["churn"].isna() | ~df["churn"].isin([0, 1])].copy()
    return CheckResult(
        name="churn_labels",
        failure_severity=Severity.ERROR,
        passed=False,
        message=f"Churn label issues: {', '.join(parts)} value(s).",
        affected_rows=total,
        detail=detail,
    )


def check_totalcharges_nulls(df: pd.DataFrame) -> CheckResult:
    """Gate 4 — unexpected NULL patterns in totalcharges.

    The 11 zero-tenure customers in the IBM Telco dataset have whitespace
    TotalCharges in the source CSV, coerced to NaN by ingest.py. NULLs are
    therefore expected only where tenure == 0. Any NULL with tenure > 0 is
    unexpected and worth flagging.

    Severity: WARNING — the model training ColumnTransformer will impute these
    with the training-set median, so the pipeline can continue, but the anomaly
    should be investigated.

    Non-negativity for tenure, monthlycharges, and totalcharges is enforced
    by Gate 1 (RawSchema Field(ge=0)) and is not duplicated here.
    """
    tc_cols = {"totalcharges", "tenure"}
    if not tc_cols.issubset(df.columns):
        return CheckResult(
            name="totalcharges_unexpected_nulls",
            failure_severity=Severity.WARNING,
            passed=True,
            message="Skipped totalcharges NULL check — required column(s) missing.",
        )

    unexpected = int((df["totalcharges"].isna() & (df["tenure"] > 0)).sum())
    if unexpected == 0:
        return CheckResult(
            name="totalcharges_unexpected_nulls",
            failure_severity=Severity.WARNING,
            passed=True,
            message="No unexpected NULL totalcharges values.",
        )

    detail = df[df["totalcharges"].isna() & (df["tenure"] > 0)].copy()
    return CheckResult(
        name="totalcharges_unexpected_nulls",
        failure_severity=Severity.WARNING,
        passed=False,
        message=(
            f"{unexpected} row(s) with NULL totalcharges but tenure > 0. "
            "Expected NULLs only where tenure == 0."
        ),
        affected_rows=unexpected,
        detail=detail,
    )


def check_distribution_sanity(
    df: pd.DataFrame,
    min_rows: int = MIN_ROWS,
    max_null_rate: float = MAX_NULL_RATE,
) -> list[CheckResult]:
    """Gate 5 — row count and per-column null-rate bounds.

    Sub-checks:
    - Zero rows → ERROR: pipeline cannot proceed with an empty DataFrame.
    - Row count below min_rows → WARNING: stale or partial data pull.
    - Null rate above max_null_rate in any critical column → WARNING.

    All WARNING-severity checks allow the pipeline to continue while surfacing
    potential operational issues for investigation.
    """
    results: list[CheckResult] = []
    n_rows = len(df)

    if n_rows == 0:
        results.append(
            CheckResult(
                name="row_count",
                failure_severity=Severity.ERROR,
                passed=False,
                message="DataFrame has zero rows — pipeline cannot proceed.",
            )
        )
        return results  # remaining checks are meaningless on an empty DataFrame

    if n_rows < min_rows:
        results.append(
            CheckResult(
                name="row_count",
                failure_severity=Severity.WARNING,
                passed=False,
                message=(
                    f"Only {n_rows} row(s) present; expected ≥ {min_rows}. "
                    "May indicate a stale or partial data pull."
                ),
            )
        )
    else:
        results.append(
            CheckResult(
                name="row_count",
                failure_severity=Severity.WARNING,
                passed=True,
                message=f"Row count {n_rows} meets the minimum threshold of {min_rows}.",
            )
        )

    for col in sorted(_NULL_CHECKED_COLS):
        if col not in df.columns:
            continue
        null_rate = float(df[col].isna().mean())
        if null_rate > max_null_rate:
            results.append(
                CheckResult(
                    name=f"null_rate_{col}",
                    failure_severity=Severity.WARNING,
                    passed=False,
                    message=(
                        f"'{col}' null rate {null_rate:.1%} exceeds "
                        f"threshold {max_null_rate:.1%}."
                    ),
                    affected_rows=int(df[col].isna().sum()),
                    detail=df[df[col].isna()].copy(),
                )
            )
        else:
            results.append(
                CheckResult(
                    name=f"null_rate_{col}",
                    failure_severity=Severity.WARNING,
                    passed=True,
                    message=f"'{col}' null rate {null_rate:.1%} is within threshold.",
                )
            )

    return results

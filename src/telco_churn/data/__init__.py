"""Public API for the telco_churn.data package."""

from telco_churn.data.checks import CheckResult, Severity
from telco_churn.data.ingest import ingest
from telco_churn.data.schema import CleanedSchema, RawSchema
from telco_churn.data.split import (
    DEV,
    SPLIT_COL,
    TEST,
    dev_ids,
    load_split,
    make_split,
    partition,
    test_ids,
    write_split,
)
from telco_churn.data.validate import (
    ValidationError,
    ValidationResult,
    save_validation_report,
    validate_clean,
    validate_raw,
)

__all__ = [
    # result types — needed to annotate / inspect validate_raw output
    "CheckResult",
    "Severity",
    "ValidationError",
    "ValidationResult",
    # schemas — callers that validate DataFrames directly
    "CleanedSchema",
    "RawSchema",
    # canonical split — constants + helpers
    "DEV",
    "SPLIT_COL",
    "TEST",
    "dev_ids",
    "load_split",
    "make_split",
    "partition",
    "test_ids",
    "write_split",
    # pipeline entry points
    "ingest",
    "save_validation_report",
    "validate_clean",
    "validate_raw",
]

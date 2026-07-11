"""Public API for the telco_churn.data package."""

import importlib
from typing import Any

from telco_churn.data.checks import CheckResult, Severity
from telco_churn.data.schema import CleanedSchema, RawSchema

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

_LAZY_SOURCES: dict[str, str] = {
    "ingest": "telco_churn.data.ingest",
    "DEV": "telco_churn.data.split",
    "SPLIT_COL": "telco_churn.data.split",
    "TEST": "telco_churn.data.split",
    "dev_ids": "telco_churn.data.split",
    "load_split": "telco_churn.data.split",
    "make_split": "telco_churn.data.split",
    "partition": "telco_churn.data.split",
    "test_ids": "telco_churn.data.split",
    "write_split": "telco_churn.data.split",
    "ValidationError": "telco_churn.data.validate",
    "ValidationResult": "telco_churn.data.validate",
    "save_validation_report": "telco_churn.data.validate",
    "validate_clean": "telco_churn.data.validate",
    "validate_raw": "telco_churn.data.validate",
}


def __getattr__(name: str) -> Any:
    """Lazily resolve the ingest/split/validate re-exports.

    ingest.py, split.py, and validate.py each carry a `__main__` CLI block.
    Eagerly importing them here, at telco_churn.data package-import time, would
    register them in sys.modules under their real dotted name before `python -m
    telco_churn.data.<module>` gets a chance to load them fresh as __main__ —
    runpy then re-executes the module's top-level code a second time under a
    second identity and warns about the collision. Deferring the import until
    the symbol is actually accessed avoids it.
    """
    module_path = _LAZY_SOURCES.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_path)
    value = getattr(module, name)
    globals()[name] = value
    return value

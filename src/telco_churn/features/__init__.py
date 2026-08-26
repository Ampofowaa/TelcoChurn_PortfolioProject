"""Public API for the telco_churn.features package."""

import importlib
from typing import Any

from telco_churn.features.accessor import (
    FEATURES_FILENAME,
    features_path,
    features_sha256,
    load_features,
)
from telco_churn.features.generate import (
    CORR_THRESHOLD,
    CRAMERS_V_THRESHOLD,
    IMPORTANCE_NOISE_FLOOR_MARGIN,
    MIN_PR_AUC_DELTA,
    AdoptionDecision,
    BackwardEliminationResult,
    ImportanceResult,
    LapRecord,
    RedundancyResult,
    adoption_gate,
    backward_elimination,
    bootstrap_pr_auc_ci,
    candidate_importance,
    oof_predictions,
    profile_false_negatives,
    redundancy_screen,
    serving_available,
    subgroup_recall,
    write_provenance,
)
from telco_churn.features.preprocessing import (
    TENURE_COHORT_EDGES,
    build_linear_preprocessor,
    build_preprocessor,
)
from telco_churn.features.schema import CustomerFeaturesSchema, FeatureOutputSchema

_LAZY_SOURCES: dict[str, str] = {
    "FEATURE_SCHEMA": "telco_churn.features.build",
    "SQL_FEATURE_COLS": "telco_churn.features.build",
    "TARGET_COL": "telco_churn.features.build",
    "FeatureSchema": "telco_churn.features.build",
    "build_feature_df": "telco_churn.features.build",
    "build_feature_query": "telco_churn.features.build",
    "load_customer_features": "telco_churn.features.build",
    "build_sql_features": "telco_churn.features.sql_features",
}


def __getattr__(name: str) -> Any:
    """Lazily resolve re-exports that would otherwise pay an eager import cost.

    Two distinct reasons live behind this one hook:
    - SERVING_COLS calls generate._get_serving_cols() at package-import time
      otherwise, paying Pandera's import/validation cost even for callers who
      never touch serving columns.
    - build.py and sql_features.py each carry a `__main__` CLI block. Eagerly
      importing them here would register them in sys.modules under their real
      dotted name before `python -m telco_churn.features.<module>` gets a
      chance to load them fresh as __main__ — runpy then re-executes the
      module's top-level code a second time under a second identity and warns
      about the collision. Deferring the import until the symbol is actually
      accessed avoids it.
    """
    if name == "SERVING_COLS":
        from telco_churn.features.generate import _get_serving_cols

        val = _get_serving_cols()
        globals()["SERVING_COLS"] = val
        return val
    module_path = _LAZY_SOURCES.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_path)
    value = getattr(module, name)
    globals()[name] = value
    return value


__all__ = [
    "FEATURE_SCHEMA",
    "FeatureSchema",
    "SQL_FEATURE_COLS",
    "TARGET_COL",
    "CustomerFeaturesSchema",
    "FeatureOutputSchema",
    "FEATURES_FILENAME",
    "features_path",
    "features_sha256",
    "load_features",
    "CORR_THRESHOLD",
    "CRAMERS_V_THRESHOLD",
    "IMPORTANCE_NOISE_FLOOR_MARGIN",
    "MIN_PR_AUC_DELTA",
    "SERVING_COLS",
    "AdoptionDecision",
    "BackwardEliminationResult",
    "ImportanceResult",
    "LapRecord",
    "RedundancyResult",
    "adoption_gate",
    "backward_elimination",
    "bootstrap_pr_auc_ci",
    "candidate_importance",
    "oof_predictions",
    "profile_false_negatives",
    "redundancy_screen",
    "serving_available",
    "subgroup_recall",
    "write_provenance",
    "build_feature_df",
    "build_feature_query",
    "load_customer_features",
    "TENURE_COHORT_EDGES",
    "build_linear_preprocessor",
    "build_preprocessor",
    "build_sql_features",
]

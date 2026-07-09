"""Public API for the telco_churn.features package."""

from telco_churn.features.build import (
    FEATURE_SCHEMA,
    SQL_FEATURE_COLS,
    TARGET_COL,
    FeatureSchema,
    build_feature_df,
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
from telco_churn.features.sql_features import build_sql_features


def __getattr__(name: str) -> object:
    """Lazily resolve SERVING_COLS so package import does not pay Pandera cost eagerly."""
    if name == "SERVING_COLS":
        from telco_churn.features.generate import _get_serving_cols

        val = _get_serving_cols()
        globals()["SERVING_COLS"] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FEATURE_SCHEMA",
    "FeatureSchema",
    "SQL_FEATURE_COLS",
    "TARGET_COL",
    "CustomerFeaturesSchema",
    "FeatureOutputSchema",
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
    "TENURE_COHORT_EDGES",
    "build_linear_preprocessor",
    "build_preprocessor",
    "build_sql_features",
]

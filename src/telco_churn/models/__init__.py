"""Public API for the telco_churn.models package.

Package layout follows one rule: a module lives under `train/` iff its reason
to run is "the step before it just produced my input" — candidates.py ->
comparison.py (notebook-only) and feature_audit.py -> tuning.py ->
log_model.py each take the prior step's return value as an argument, share
one dev-data snapshot, and (tuning.py/log_model.py) even share one MLflow
run; there is no standalone use for a hyperparameter search that never gets
fit, so they run behind one `__main__.py` and one `make train`. Every other
module here (calibrate.py, threshold.py, evaluate.py, error_analysis.py,
register.py) is independently invocable because its reason to run is
external to the step before it — a costs.yaml edit (threshold.py), a
recalibration with no retrain (calibrate.py), a human approval on human
time (register.py) — so each gets its own CLI entry point, `make` target,
and (Phase 8) DVC stage, addressed by an explicit run_id/model_version
rather than "whatever train just produced."
"""

from telco_churn.models.train import (
    bootstrap_comparison,
    boundary_hit_check,
    cv_score_candidate,
    run_candidate_step,
    run_comparison_step,
    run_diagnostics_step,
    run_feature_audit_step,
    run_feature_selection_step,
    run_model_logging_step,
    run_tuning_step,
    select_best_trial,
)

__all__ = [
    "boundary_hit_check",
    "bootstrap_comparison",
    "cv_score_candidate",
    "run_candidate_step",
    "run_comparison_step",
    "run_diagnostics_step",
    "run_feature_audit_step",
    "run_feature_selection_step",
    "run_model_logging_step",
    "run_tuning_step",
    "select_best_trial",
]

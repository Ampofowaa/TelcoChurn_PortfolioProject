"""Public API for the telco_churn.models package.

Package layout follows one rule: a module lives under `train/` iff its reason
to run is "the step before it just produced my input" — candidates.py ->
comparison.py (notebook-only) and feature_audit.py -> tuning.py ->
log_model.py each take the prior step's return value as an argument, share
one dev-data snapshot, and (tuning.py/log_model.py) even share one MLflow
run; there is no standalone use for a hyperparameter search that never gets
fit, so they run behind one `__main__.py` and one `dvc repro train`. Every
other module here (calibrate.py, threshold.py, evaluate.py, error_analysis.py,
review.py, register.py) is independently invocable because its reason to
run is external to the step before it — a costs.yaml edit (threshold.py), a
recalibration with no retrain (calibrate.py), a human review on human time
(review.py), a human approval decision on human time (register.py) — so
each gets its own CLI entry point and DVC stage where applicable, plus a
`make` target for calibrate.py/threshold.py/evaluate.py/error_analysis.py's
RUN_ID/MODEL_VERSION overrides and for review.py/register.py, which have no
DVC stage at all (review.py and register.py mutate the registry and are
excluded from the DVC DAG — see register.py's own docstring), addressed by
an explicit run_id/model_version rather than "whatever train just
produced."

The re-export below is lazy (PEP 562 module `__getattr__`), not a plain
`from telco_churn.models.train import (...)`. `train/__init__.py` pulls in
optuna, matplotlib, and (via feature_audit.py -> features/select.py) shap —
real weight that only training/notebook code needs. serving/app.py and
ui/streamlit_app.py both import plain sibling modules here (policy_config,
artifacts, explain, shap_values, environment_parity), and every one of those
imports first runs this file's module body — an eager `from
telco_churn.models.train import ...` would drag the training package into
both Docker images regardless of what either actually calls. Nothing in
src/ or notebooks/ currently imports these names via `telco_churn.models`
directly (train/__main__.py imports `telco_churn.models.train` itself), so
laziness costs nothing today; it only stops paying for train/'s import
weight on every other submodule's behalf.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
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


def __getattr__(name: str) -> object:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from telco_churn.models import train

    return getattr(train, name)

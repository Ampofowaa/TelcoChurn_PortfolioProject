"""Model training pipeline: five steps, each in its own module.

1. candidates.py — CV-score Dummy / LogReg / LightGBM on the dev set.
2. comparison.py — paired-bootstrap family decision, plus non-gating diagnostics.
3. feature_freeze.py — permutation-importance/SHAP audit of the already-frozen
   input space (committed via features/schema.py::COMMITTED_FEATURES, edited
   only by a human after an on-demand notebook review — never recomputed here).
4. tuning.py — Optuna hyperparameter search on the frozen feature set.
5. log_model.py — log the tuned pipeline as an MLflow run artifact, unregistered
   (Phase 6's calibrate.py performs the cycle's single registration).

Steps 1-2 are notebook-only (notebooks/03a-model-selection.ipynb) — the family
decision they produce is frozen into common.py::COMMITTED_MODEL_FAMILY by a
human, via a reviewed code change, never recomputed live. feature_selection.py
plays the identical role for the feature axis: its full-vs-reduced paired-
bootstrap ablation decides features/schema.py::COMMITTED_FEATURES, called only
from notebooks/03b-feature-selection.ipynb on a real trigger, never every
retrain — the same relationship candidates.py/comparison.py have to
COMMITTED_MODEL_FAMILY, just one module instead of two since there's no
per-candidate step to split out. `python -m telco_churn.models.train`
(see __main__.py) runs Steps 3-5 against both frozen constants.
"""

from __future__ import annotations

import sys

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")  # non-interactive backend for CLI/DVC — skip in notebooks

from telco_churn.models.train.candidates import run_candidate_step  # noqa: E402
from telco_churn.models.train.common import (  # noqa: E402
    cv_score_candidate,
    lgbm_default_params,
)
from telco_churn.models.train.comparison import (  # noqa: E402
    bootstrap_comparison,
    run_comparison_step,
    run_diagnostics_step,
)
from telco_churn.models.train.feature_freeze import run_feature_audit_step  # noqa: E402
from telco_churn.models.train.feature_selection import (  # noqa: E402
    run_feature_selection_step,
)
from telco_churn.models.train.log_model import run_model_logging_step  # noqa: E402
from telco_churn.models.train.tuning import (  # noqa: E402
    boundary_hit_check,
    run_tuning_step,
    select_best_trial,
)

__all__ = [
    "bootstrap_comparison",
    "boundary_hit_check",
    "cv_score_candidate",
    "lgbm_default_params",
    "run_candidate_step",
    "run_comparison_step",
    "run_diagnostics_step",
    "run_feature_audit_step",
    "run_feature_selection_step",
    "run_model_logging_step",
    "run_tuning_step",
    "select_best_trial",
]

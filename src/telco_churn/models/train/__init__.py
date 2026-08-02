"""Model training pipeline: five sequential steps, each in its own module.

1. candidates.py — CV-score Dummy / LogReg / LightGBM on the dev set.
2. comparison.py — paired-bootstrap family decision, plus non-gating diagnostics.
3. feature_freeze.py — permutation-importance selection; freezes the input space.
4. tuning.py — Optuna hyperparameter search on the frozen feature set.
5. log_model.py — log the tuned pipeline as an MLflow run artifact, unregistered
   (Phase 6's calibrate.py performs the cycle's single registration).

Run as `python -m telco_churn.models.train` (see __main__.py) to execute all five
in sequence.
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
from telco_churn.models.train.feature_freeze import run_selection_step  # noqa: E402
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
    "run_model_logging_step",
    "run_selection_step",
    "run_tuning_step",
    "select_best_trial",
]

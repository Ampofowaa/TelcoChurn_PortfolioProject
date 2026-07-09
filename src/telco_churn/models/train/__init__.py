"""Model training pipeline — five steps, run in sequence.

Step 1: candidate training — CV-score Dummy / LogReg / LightGBM on the dev set.
    See candidates.py.
Step 2: paired-bootstrap comparison + decision rule, plus non-gating diagnostics
    (fixed-recall profile, fairness/robustness flags via diagnostics.py).
    See comparison.py.
Step 3: feature selection via features/select.py — freezes the input space.
    See feature_freeze.py.
Step 4: Optuna hyperparameter tuning on the frozen feature set.
    See tuning.py.
Step 5: register the tuned pipeline as MLflow challenger.
    See registration.py.

Run as `python -m telco_churn.models.train` (see __main__.py) to execute all five
steps in sequence.
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
from telco_churn.models.train.registration import run_registration_step  # noqa: E402
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
    "run_registration_step",
    "run_selection_step",
    "run_tuning_step",
    "select_best_trial",
]

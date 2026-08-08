"""Step 1: candidate training — CV-score Dummy / LogReg / LightGBM on the dev set.

Not wired into the automated pipeline (models/train/__main__.py) — the family
decision it feeds is frozen into common.py::COMMITTED_MODEL_FAMILY by a human,
via a reviewed code change, never recomputed live. This is a design-time /
periodic-review tool, called only from notebooks/03a-model-selection.ipynb, on
the same footing as features/select.py's retired keep-vs-reduce ablation. See
common.py::COMMITTED_MODEL_FAMILY and ANALYSIS.md §4a for the frozen decision
and its run id. Re-run the comparison (03a) only on a real trigger — a new
candidate family, a drift signal, or a scheduled periodic review — not on
every retrain.
"""

from __future__ import annotations

from typing import Any

import mlflow
import pandas as pd
from lightgbm import LGBMClassifier
from omegaconf import DictConfig
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline

from telco_churn.features.accessor import features_sha256
from telco_churn.features.build import FEATURE_SCHEMA
from telco_churn.features.preprocessing import (
    build_linear_preprocessor,
    build_preprocessor,
)
from telco_churn.models.train.common import (
    _build_dev_dataset,
    _git_sha,
    cv_score_candidate,
    lgbm_default_params,
    logreg_default_params,
)
from telco_churn.utils.logging import get_logger
from telco_churn.utils.mlflow import ensure_experiment_metadata

__all__ = ["run_candidate_step"]

logger = get_logger(__name__)

_DUMMY_ROC_AUC_TARGET = 0.5
_DUMMY_ROC_AUC_TOL = 0.05
_DUMMY_PR_AUC_TOL = 0.03

# Family-review-only (notebooks/03a-model-selection.ipynb), not read from Hydra
# config: nothing in the automated pipeline calls run_candidate_step, so there
# is no CLI-override use case — same module-constant pattern as
# features/select.py::ABLATION_N_BOOTSTRAP.
_FAMILY_REVIEW_CV_FOLDS: int = 10
_FAMILY_REVIEW_CV_REPEATS: int = 10

# n_jobs=-1 (process-based parallelism) was tried for logreg_cv's folds — its
# internal Cs-search makes each fit expensive enough that this looked
# worthwhile on paper. Re-measured on real dev data at the production fold
# count (10x10=100 folds): only a ~1.26x wall-clock gain (105.0s -> 83.4s),
# not the ~2.5x an earlier measurement had suggested, while per-fold
# train_time_s became contention-noisy (1.0s -> 4-8s, varying run to run) and
# no longer comparable to the other two candidates' sequential timings. Not
# worth it — all three candidates stay sequential.


def _assert_dummy_canary(dummy_result: dict[str, Any], y_dev: pd.Series) -> None:
    """Abort the run if the feature-blind dummy_prior candidate scores off chance.

    Raises AssertionError if ROC-AUC or PR-AUC falls outside tolerance — signals a
    broken eval harness (e.g. leakage), not a modelling result.
    """
    dummy_roc_auc = float(
        roc_auc_score(dummy_result["oof_true"], dummy_result["oof_proba"])
    )
    prevalence = float(y_dev.mean())
    assert abs(dummy_roc_auc - _DUMMY_ROC_AUC_TARGET) <= _DUMMY_ROC_AUC_TOL, (
        f"Leakage canary failed: dummy_prior ROC-AUC={dummy_roc_auc:.3f} deviates from "
        f"{_DUMMY_ROC_AUC_TARGET} by more than {_DUMMY_ROC_AUC_TOL} — a feature-blind "
        "classifier should score at chance; investigate for label leakage or a "
        "misaligned eval harness."
    )
    assert abs(dummy_result["pr_auc_mean"] - prevalence) <= _DUMMY_PR_AUC_TOL, (
        f"Leakage canary failed: dummy_prior PR-AUC={dummy_result['pr_auc_mean']:.3f} "
        f"deviates from prevalence={prevalence:.3f} by more than {_DUMMY_PR_AUC_TOL} — "
        "investigate for a broken eval harness."
    )


def run_candidate_step(
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    cfg: DictConfig,
    cv_folds: int = _FAMILY_REVIEW_CV_FOLDS,
    cv_repeats: int = _FAMILY_REVIEW_CV_REPEATS,
) -> dict[str, dict[str, Any]]:
    """Train and CV-score the dummy_prior / logreg_cv / lgbm_default candidates.

    Shares one RepeatedStratifiedKFold instance across all three so every candidate
    trains and validates on identical folds. Each candidate is logged as its own
    MLflow run under cfg.mlflow.experiment_name, with the model-construction kwargs
    it was actually fit with (configs/training/logreg.yaml, configs/training/
    lightgbm.yaml) logged as params alongside the generic CV-setup ones — so a
    config edit between cycles is legible on the run, not just inferable from
    the resulting PR-AUC.

    cv_folds/cv_repeats default to the family-review constants above (notebook
    call sites take the defaults); overridable for fast test fixtures.

    Returns per-candidate result dicts keyed by run name, for Step 2 comparison.
    Each dict carries cv_score_candidate's keys plus "run_id" (the MLflow run
    just logged) — the caller resolves its own runs directly rather than
    re-querying MLflow by name/recency afterward.
    """
    random_state = int(cfg.random_seed)
    cc = cfg.training_setup
    cv = RepeatedStratifiedKFold(
        n_splits=cv_folds,
        n_repeats=cv_repeats,
        random_state=random_state,
    )

    binary = list(FEATURE_SCHEMA.binary)
    multi_cat = list(FEATURE_SCHEMA.multi_cat)
    numeric = list(FEATURE_SCHEMA.numeric)

    logreg_params = logreg_default_params(cfg, random_state)
    lgbm_params = lgbm_default_params(cfg, random_state)

    candidates: dict[str, Any] = {
        "dummy_prior": DummyClassifier(strategy="prior", random_state=random_state),
        "logreg_cv": Pipeline(
            [
                ("preprocessor", build_linear_preprocessor(binary, multi_cat, numeric)),
                ("model", LogisticRegressionCV(**logreg_params)),
            ]
        ),
        "lgbm_default": Pipeline(
            [
                ("preprocessor", build_preprocessor(binary, multi_cat, numeric)),
                ("model", LGBMClassifier(**lgbm_params)),
            ]
        ),
    }

    # Model-construction kwargs, logged alongside the generic CV-setup params below
    # so a config edit (configs/training/logreg.yaml, configs/training/lightgbm.yaml)
    # is legible on the run itself, not just inferable from the estimator it produced.
    # class_weight is dropped here — already logged once, generically, below.
    _model_construction_params: dict[str, dict[str, Any]] = {
        "dummy_prior": {},
        "logreg_cv": {k: v for k, v in logreg_params.items() if k != "class_weight"},
        "lgbm_default": {k: v for k, v in lgbm_params.items() if k != "class_weight"},
    }

    ensure_experiment_metadata(cfg)

    git_sha = _git_sha()
    data_content_hash = features_sha256()
    _dev_dataset = _build_dev_dataset(X_dev, y_dev)

    _model_family = {
        "dummy_prior": "dummy",
        "logreg_cv": "logreg",
        "lgbm_default": "lightgbm",
    }
    _stage = {
        "dummy_prior": "baseline",
        "logreg_cv": "comparison",
        "lgbm_default": "comparison",
    }
    results: dict[str, dict[str, Any]] = {}

    for name, estimator in candidates.items():
        with mlflow.start_run(run_name=name) as run:
            mlflow.set_tags(
                {
                    "stage": _stage[name],
                    "model_family": _model_family[name],
                    "git_sha": git_sha,
                    "data_content_hash": data_content_hash,
                }
            )
            mlflow.log_input(_dev_dataset, context="training")
            logger.info("candidate_cv_start", candidate=name)

            result = cv_score_candidate(estimator, X_dev, y_dev, cv)
            result["run_id"] = run.info.run_id
            results[name] = result

            if name == "dummy_prior":
                _assert_dummy_canary(result, y_dev)

            mlflow.log_params(
                {
                    "candidate": name,
                    "cv_n_splits": cv_folds,
                    "cv_n_repeats": cv_repeats,
                    "cv_random_state": random_state,
                    "class_weight": str(cc.class_weight),
                }
            )
            if _model_construction_params[name]:
                mlflow.log_params(_model_construction_params[name])
            mlflow.log_metrics(
                {
                    "cv_pr_auc_mean": round(result["pr_auc_mean"], 3),
                    "cv_pr_auc_std": round(result["pr_auc_std"], 3),
                    "cv_train_time_s": result["train_time_s"],
                    "cv_predict_time_s": result["predict_time_s"],
                }
            )
            for step, fold_score in enumerate(result["pr_auc_scores"]):
                mlflow.log_metric("cv_pr_auc_fold", round(fold_score, 3), step=step)

            logger.info(
                "candidate_cv_done",
                candidate=name,
                pr_auc_mean=round(result["pr_auc_mean"], 3),
                pr_auc_std=round(result["pr_auc_std"], 3),
                run_id=run.info.run_id,
            )

    return results

"""Step 1: candidate training — CV-score Dummy / LogReg / LightGBM on the dev set."""

from __future__ import annotations

from typing import Any

import mlflow
import pandas as pd
from lightgbm import LGBMClassifier
from mlflow.data.pandas_dataset import from_pandas as mlflow_from_pandas
from omegaconf import DictConfig
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline

from telco_churn.features.build import FEATURE_SCHEMA, TARGET_COL
from telco_churn.features.preprocessing import (
    build_linear_preprocessor,
    build_preprocessor,
)
from telco_churn.models.train.common import (
    _dvc_hash,
    _git_sha,
    cv_score_candidate,
    lgbm_default_params,
    logreg_default_params,
)
from telco_churn.utils.logging import get_logger
from telco_churn.utils.mlflow import resolve_tracking_uri
from telco_churn.utils.paths import get_project_root

__all__ = ["run_candidate_step"]

logger = get_logger(__name__)

_DUMMY_ROC_AUC_TARGET = 0.5
_DUMMY_ROC_AUC_TOL = 0.05
_DUMMY_PR_AUC_TOL = 0.03


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
) -> dict[str, dict[str, Any]]:
    """Train and CV-score the dummy_prior / logreg_cv / lgbm_default candidates.

    Shares one RepeatedStratifiedKFold instance across all three so every candidate
    trains and validates on identical folds. Each candidate is logged as its own
    MLflow run under cfg.mlflow.experiment_name.

    Returns per-candidate result dicts keyed by run name, for Step 2 comparison.
    """
    random_state = int(cfg.random_seed)
    cc = cfg.training_setup
    cv = RepeatedStratifiedKFold(
        n_splits=int(cc.cv_folds),
        n_repeats=int(cc.cv_repeats),
        random_state=random_state,
    )

    binary = list(FEATURE_SCHEMA.binary)
    multi_cat = list(FEATURE_SCHEMA.multi_cat)
    numeric = list(FEATURE_SCHEMA.numeric)

    candidates: dict[str, Any] = {
        "dummy_prior": DummyClassifier(strategy="prior", random_state=random_state),
        "logreg_cv": Pipeline(
            [
                ("preprocessor", build_linear_preprocessor(binary, multi_cat, numeric)),
                (
                    "model",
                    LogisticRegressionCV(**logreg_default_params(cfg, random_state)),
                ),
            ]
        ),
        "lgbm_default": Pipeline(
            [
                ("preprocessor", build_preprocessor(binary, multi_cat, numeric)),
                ("model", LGBMClassifier(**lgbm_default_params(cfg, random_state))),
            ]
        ),
    }

    mlflow.set_tracking_uri(resolve_tracking_uri(str(cfg.mlflow.tracking_uri)))
    exp = mlflow.set_experiment(cfg.mlflow.experiment_name)
    _client = mlflow.tracking.MlflowClient()
    _client.set_experiment_tag(
        exp.experiment_id,
        "mlflow.note.content",
        (
            "Model selection — candidate comparison (Dummy / Logistic Regression / LightGBM) "
            "and paired-bootstrap decision on the IBM Telco Customer Churn dev set. "
            "Selection criterion: PR-AUC (average precision). "
            "Winning family proceeds to feature selection and Optuna tuning."
        ),
    )
    for _tag_key, _tag_val in {
        "project": "telco-churn",
        "phase": "model-selection",
        "dataset": "ibm-telco-churn",
        "env": "development",
    }.items():
        _client.set_experiment_tag(exp.experiment_id, _tag_key, _tag_val)

    git_sha = _git_sha()
    dvc_hash = _dvc_hash(cfg)
    dev_df = X_dev.copy()
    dev_df[TARGET_COL] = y_dev.values
    _dev_dataset = mlflow_from_pandas(
        dev_df,
        source=str(
            get_project_root() / cfg.paths.processed_data / "telco_churn_processed.csv"
        ),
        name="telco_churn_dev",
        targets=TARGET_COL,
    )

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
                    "dvc_data_hash": dvc_hash,
                }
            )
            mlflow.log_input(_dev_dataset, context="training")
            logger.info("candidate_cv_start", candidate=name)

            # n_jobs left at cv_score_candidate's sequential default: each fold here is
            # a single cheap fit (~0.25s) — too light for joblib's process-based
            # parallelism to pay off on Windows (spawn-based worker startup + per-task
            # dispatch overhead measured ~2x slower than sequential for this workload).
            # Feature selection's much heavier per-fold cost (feature_freeze.py) is
            # where cv_n_jobs is actually applied.
            result = cv_score_candidate(estimator, X_dev, y_dev, cv)
            results[name] = result

            if name == "dummy_prior":
                _assert_dummy_canary(result, y_dev)

            mlflow.log_params(
                {
                    "candidate": name,
                    "cv_n_splits": int(cc.cv_folds),
                    "cv_n_repeats": int(cc.cv_repeats),
                    "cv_random_state": random_state,
                    "class_weight": str(cc.class_weight),
                }
            )
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

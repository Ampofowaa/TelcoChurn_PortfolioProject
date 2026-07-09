"""Step 5: register the tuned pipeline as MLflow challenger."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from omegaconf import DictConfig
from sklearn.pipeline import Pipeline

from telco_churn.features.build import FEATURE_SCHEMA
from telco_churn.features.preprocessing import build_preprocessor
from telco_churn.models.train.common import (
    _dvc_hash,
    _git_sha,
    _lgbm_fixed_knobs,
    _resolve_tracking_uri,
)
from telco_churn.utils.logging import get_logger

__all__ = ["run_registration_step"]

logger = get_logger(__name__)


def run_registration_step(
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    comparison: dict[str, Any],
    tuning_result: dict[str, Any],
    cfg: DictConfig,
) -> dict[str, Any]:
    """Register the tuned pipeline as MLflow challenger.

    Refits [tree_preprocessor -> LightGBM] on all of development with the Step 4
    hyperparameters (n_estimators fixed at the selected trial's median
    early-stopped tree count — no further early stopping here). Logged onto the
    same MLflow run as the Step 4 tuning_study parent, reopened via its run_id.

    Logs the full Pipeline (not the bare estimator) with a signature + input
    example, runs a log -> reload -> predict parity check (hard assertion), and
    logs feature_space.txt / feature_columns.txt / preprocessing.pkl plus
    training_manifest.json — the engineering audit trail (git SHA, DVC data hash,
    hyperparameters, feature space/columns, CV PR-AUC, paired-Δ vs LogReg with
    full bootstrap CI, and tuning_summary — the trial counts and 1-SE band
    diagnostics behind the selection_rule pick, passed through from
    run_tuning_step unchanged). The stakeholder-facing model_card.json is a
    Phase 7 deliverable, written once at champion promotion when calibration,
    threshold, and sealed-test results are real — not here.

    Registers the model as cfg.mlflow.registered_model_name and points the
    'challenger' alias at it. This challenger is uncalibrated, un-thresholded, and
    not evaluated on the sealed test set — not serving-ready.

    Returns {"run_id", "model_uri", "version", "parity_ok", "training_manifest"}.
    """
    random_state = int(cfg.random_seed)
    committed_features = list(tuning_result["committed_features"])

    binary = [c for c in FEATURE_SCHEMA.binary if c in committed_features]
    multi_cat = [c for c in FEATURE_SCHEMA.multi_cat if c in committed_features]
    numeric = [c for c in FEATURE_SCHEMA.numeric if c in committed_features]

    model_params = {
        "n_estimators": int(tuning_result["best_n_estimators_median"]),
        **dict(tuning_result["best_params"]),
        **_lgbm_fixed_knobs(cfg, random_state),
    }

    X_committed = X_dev[committed_features]
    preprocessor = build_preprocessor(binary, multi_cat, numeric)
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", LGBMClassifier(**model_params)),
        ]
    )
    pipeline.fit(X_committed, y_dev)

    input_example = X_committed.head(5)
    in_memory_preds = pipeline.predict(input_example)
    signature = infer_signature(X_committed, pipeline.predict(X_committed))

    full_feature_space = (
        list(FEATURE_SCHEMA.binary)
        + list(FEATURE_SCHEMA.multi_cat)
        + list(FEATURE_SCHEMA.numeric)
    )

    training_manifest: dict[str, Any] = {
        "model_name": str(cfg.mlflow.registered_model_name),
        "alias": "challenger",
        "git_sha": _git_sha(),
        "dvc_data_hash": _dvc_hash(cfg),
        "model_family": "lightgbm",
        "hyperparameters": model_params,
        "feature_space": full_feature_space,
        "feature_columns": committed_features,
        "cv_pr_auc_mean": tuning_result["best_cv_pr_auc_mean"],
        "tuning_selection_rule": str(cfg.tuning.selection_rule),
        "tuning_summary": tuning_result["tuning_summary"],
        "paired_delta_vs_logreg": {
            "delta_obs": comparison["delta_obs"],
            "delta_ci_lower": comparison["delta_ci_lower"],
            "delta_ci_upper": comparison["delta_ci_upper"],
            "decision": comparison["decision"],
            "decision_rule": comparison["decision_rule"],
        },
    }

    mlflow.set_tracking_uri(_resolve_tracking_uri(str(cfg.mlflow.tracking_uri)))
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    run_id = str(tuning_result["parent_run_id"])
    with mlflow.start_run(run_id=run_id):
        mlflow.log_text("\n".join(full_feature_space), "feature_space.txt")
        mlflow.log_text("\n".join(committed_features), "feature_columns.txt")
        mlflow.log_dict(training_manifest, "training_manifest.json")

        with tempfile.TemporaryDirectory() as tmp_dir:
            preprocessing_path = Path(tmp_dir) / "preprocessing.pkl"
            joblib.dump(preprocessor, preprocessing_path)
            mlflow.log_artifact(str(preprocessing_path))

        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=str(cfg.mlflow.registered_model_name),
            # mlflow>=3's skops default rejects LightGBM's Booster/OrderedDict
            # internals (untrusted types by skops' allowlist) — cloudpickle
            # handles the full Pipeline's arbitrary object graph without a
            # per-type trust list to maintain.
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

    reloaded = mlflow.sklearn.load_model(model_info.model_uri)
    reload_preds = reloaded.predict(input_example)
    parity_ok = bool(np.array_equal(in_memory_preds, reload_preds))
    assert parity_ok, (
        "Reload parity check failed: predictions from the reloaded model differ "
        "from the in-memory pipeline on the same input sample — the serialized "
        "model is not safe to register."
    )

    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(
        name=str(cfg.mlflow.registered_model_name),
        alias="challenger",
        version=str(model_info.registered_model_version),
    )

    logger.info(
        "registration_step_done",
        run_id=run_id,
        model_version=model_info.registered_model_version,
        parity_ok=parity_ok,
        n_committed_features=len(committed_features),
    )

    return {
        "run_id": run_id,
        "model_uri": model_info.model_uri,
        "version": model_info.registered_model_version,
        "parity_ok": parity_ok,
        "training_manifest": training_manifest,
    }

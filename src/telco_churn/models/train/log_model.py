"""Step 5: log the tuned pipeline as an MLflow run artifact — no registration."""

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
)
from telco_churn.utils.logging import get_logger
from telco_churn.utils.mlflow import resolve_tracking_uri

__all__ = ["run_model_logging_step"]

logger = get_logger(__name__)


def run_model_logging_step(
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    comparison: dict[str, Any],
    tuning_result: dict[str, Any],
    cfg: DictConfig,
) -> dict[str, Any]:
    """Log the tuned pipeline to the tuning_study MLflow run — do not register it.

    Refits [tree_preprocessor -> LightGBM] on all of development with the Step 4
    hyperparameters (n_estimators fixed at the selected trial's median
    early-stopped tree count — no further early stopping here). Logged onto the
    same MLflow run as the Step 4 tuning_study parent, reopened via its run_id.

    Logs the full Pipeline (not the bare estimator) with a probability signature +
    input example, runs a log -> reload -> predict_proba parity check (hard
    assertion), and logs feature_space.txt / feature_columns.txt /
    preprocessing.pkl plus training_manifest.json — the engineering audit trail,
    one section per pipeline step: model_comparison (Step 2's family decision),
    feature_selection (Step 3's frozen input space), training_summary (the
    fixed LightGBM knobs applied to every fit, tuned or not), and tuning_summary
    (Step 4's trial counts, 1-SE band diagnostics, and the selected
    hyperparameters, passed through from run_tuning_step plus
    selected_hyperparameters added here). The stakeholder-facing
    model_card.json is a Phase 7 deliverable, written once at champion promotion
    when calibration, threshold, and sealed-test results are real — not here.

    This artifact is uncalibrated, un-thresholded, and not evaluated on the
    sealed test set — not a valid rollback target, and therefore not registered
    (CLAUDE.md § Run artifacts vs. registry versions). Phase 6's calibrate.py
    performs the training cycle's single registration, on the calibrated
    artifact, resolved via this manifest's logged_model_uri.

    Returns {"run_id", "model_uri", "parity_ok", "training_manifest"}.
    """
    random_state = int(cfg.random_seed)
    committed_features = list(tuning_result["committed_features"])

    binary = [c for c in FEATURE_SCHEMA.binary if c in committed_features]
    multi_cat = [c for c in FEATURE_SCHEMA.multi_cat if c in committed_features]
    numeric = [c for c in FEATURE_SCHEMA.numeric if c in committed_features]

    fixed_hyperparameters = _lgbm_fixed_knobs(cfg, random_state)
    selected_hyperparameters = {
        "n_estimators": int(tuning_result["best_n_estimators_median"]),
        **dict(tuning_result["best_params"]),
    }
    model_params = {**selected_hyperparameters, **fixed_hyperparameters}

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
    in_memory_preds = pipeline.predict_proba(input_example)
    signature = infer_signature(X_committed, pipeline.predict_proba(X_committed))

    full_feature_space = (
        list(FEATURE_SCHEMA.binary)
        + list(FEATURE_SCHEMA.multi_cat)
        + list(FEATURE_SCHEMA.numeric)
    )

    training_manifest: dict[str, Any] = {
        "model_name": str(cfg.mlflow.registered_model_name),
        "model_family": "lightgbm",
        "git_sha": _git_sha(),
        "dvc_data_hash": _dvc_hash(cfg),
        # One section per pipeline step, in dependency order: comparison (Step 2)
        # decides the family, feature_selection (Step 3) freezes the input space,
        # training_summary is the fixed config every trial AND the final fit use
        # (tuning searches on top of it, not after it), tuning_summary is what
        # that search found. Nothing here is recomputed — this is the same data
        # as before, grouped so "is this hyperparameter tuned or fixed?" is
        # answered by which section it's in, not by cross-referencing config.
        "model_comparison": {
            "delta_obs": comparison["delta_obs"],
            "delta_ci_lower": comparison["delta_ci_lower"],
            "delta_ci_upper": comparison["delta_ci_upper"],
            "decision": comparison["decision"],
            "decision_rule": comparison["decision_rule"],
        },
        "feature_selection": {
            "feature_space": full_feature_space,
            "model_features": committed_features,
        },
        "training_summary": {
            "fixed_hyperparameters": fixed_hyperparameters,
        },
        "tuning_summary": {
            **tuning_result["tuning_summary"],
            "selected_hyperparameters": selected_hyperparameters,
        },
    }

    mlflow.set_tracking_uri(resolve_tracking_uri(str(cfg.mlflow.tracking_uri)))
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    run_id = str(tuning_result["parent_run_id"])
    with mlflow.start_run(run_id=run_id):
        mlflow.log_text("\n".join(full_feature_space), "feature_space.txt")
        mlflow.log_text("\n".join(committed_features), "feature_columns.txt")

        with tempfile.TemporaryDirectory() as tmp_dir:
            preprocessing_path = Path(tmp_dir) / "preprocessing.pkl"
            joblib.dump(preprocessor, preprocessing_path)
            mlflow.log_artifact(str(preprocessing_path))

        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            signature=signature,
            input_example=input_example,
            # default pyfunc_predict_fn is "predict" -> 0/1 labels; Phase 9 loads
            # by pyfunc URI and takes what comes out, so the declared and
            # exercised paths must be the same path.
            pyfunc_predict_fn="predict_proba",
            # mlflow>=3's skops default rejects LightGBM's Booster/OrderedDict
            # internals (untrusted types by skops' allowlist) — cloudpickle
            # handles the full Pipeline's arbitrary object graph without a
            # per-type trust list to maintain.
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

        # models:/m-<id> under MLflow 3 — a permanent handle on this artifact.
        # calibrate.py resolves the unfitted pipeline through this field, never
        # through runs:/<run_id>/model, which becomes ambiguous once Phase 6
        # logs a second model onto this same run.
        training_manifest["logged_model_uri"] = model_info.model_uri
        # LoggedModel.model_id — distinct from run_id. Phase 7's evaluate.py
        # attaches sealed-test metrics to this (model, dataset) pair via
        # log_metric(..., model_id=...); ModelVersion.model_id does not
        # auto-populate in OSS MLflow 3.14, so this must be persisted here or
        # the registry has no supported path to the model it points at.
        training_manifest["logged_model_id"] = model_info.model_id
        mlflow.log_dict(training_manifest, "training_manifest.json")

    reloaded = mlflow.sklearn.load_model(model_info.model_uri)
    reload_preds = reloaded.predict_proba(input_example)
    parity_ok = bool(np.allclose(in_memory_preds, reload_preds, rtol=0, atol=1e-12))
    assert parity_ok, (
        "Reload parity check failed: predictions from the reloaded model differ "
        "from the in-memory pipeline on the same input sample — the serialized "
        "model is not safe to log."
    )

    logger.info(
        "model_logged",
        run_id=run_id,
        model_uri=model_info.model_uri,
        parity_ok=parity_ok,
        n_committed_features=len(committed_features),
    )

    return {
        "run_id": run_id,
        "model_uri": model_info.model_uri,
        "parity_ok": parity_ok,
        "training_manifest": training_manifest,
    }

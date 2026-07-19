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
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
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


def _cv_pr_auc_at_n_estimators(
    n_estimators: int,
    X_committed: pd.DataFrame,
    y_dev: pd.Series,
    lgbm_params: dict[str, Any],
    binary: list[str],
    multi_cat: list[str],
    numeric: list[str],
    cv_folds: int,
    random_state: int,
) -> float:
    """Mean CV PR-AUC for the tuned spec at a fixed n_estimators — no early stopping.

    Reconstructs tuning.py's outer StratifiedKFold(cv_folds, shuffle=True,
    random_state=cfg.tuning.random_state) split by construction alone: a
    StratifiedKFold's partition depends only on n_samples, the y labels, and
    (shuffle, random_state) — never on X's column content — so passing the
    same y_dev with the same three parameters reproduces tuning.py's exact
    fold indices without threading them through tuning_result. This is the
    two-count diagnostic (PROJECT_PLAN.md Fix 5): a confirmation that the
    a-priori scaling rule holds on this project's own data, never a
    selection — it fits plain estimators in memory and logs no MLflow model,
    so it cannot mint a second registered candidate.

    Preprocessing regime deliberately differs from a tuning trial's: with no
    early stopping here, there is no es_validation_size slice to carve out of
    X_tr, so the preprocessor is fit on the *full* fold-training rows —
    whereas a trial fits it on the smaller X_fit slice (X_tr minus the ES
    slice; see tuning.py's _tuning_objective). This function's absolute score
    is therefore not expected to reproduce a trial's logged cv_pr_auc_mean at
    the same n_estimators; only the two counts compared *against each other*
    here (same folds, same preprocessing regime for both) are meaningful.
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scores: list[float] = []
    for train_idx, val_idx in skf.split(X_committed, y_dev):
        X_tr, X_val = X_committed.iloc[train_idx], X_committed.iloc[val_idx]
        y_tr, y_val = y_dev.iloc[train_idx], y_dev.iloc[val_idx]

        preprocessor = build_preprocessor(binary, multi_cat, numeric)
        X_tr_t = preprocessor.fit_transform(X_tr)
        X_val_t = preprocessor.transform(X_val)

        model = LGBMClassifier(n_estimators=n_estimators, **lgbm_params)
        model.fit(X_tr_t, y_tr)
        proba = model.predict_proba(X_val_t)[:, 1]
        scores.append(float(average_precision_score(y_val, proba)))
    return float(np.mean(scores))


def run_model_logging_step(
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    comparison: dict[str, Any],
    tuning_result: dict[str, Any],
    cfg: DictConfig,
) -> dict[str, Any]:
    """Log the tuned pipeline to the tuning_study MLflow run — do not register it.

    Refits [tree_preprocessor -> LightGBM] on all of development with the Step 4
    hyperparameters. n_estimators is not used at its raw early-stopped median —
    that count was derived on each fold's *training* partition, smaller than
    this final fit by construction (tuning.py carves an es_validation_size
    slice out of every fold's training rows), so it is scaled by
    n_final_fit / n_fold_fit before shipping (PROJECT_PLAN.md Fix 5). No
    further early stopping happens here. Logged onto the same MLflow run as
    the Step 4 tuning_study parent, reopened via its run_id.

    Logs the full Pipeline (not the bare estimator) with a probability signature +
    input example, runs a log -> reload -> predict_proba parity check (hard
    assertion), and logs feature_space.txt / feature_columns.txt /
    preprocessing.pkl plus training_manifest.json — the engineering audit trail,
    one section per pipeline step: model_comparison (Step 2's family decision),
    feature_selection (Step 3's frozen input space), training_summary (the
    fixed LightGBM knobs applied to every fit, tuned or not), and tuning_summary
    (Step 4's trial counts, 1-SE band diagnostics, and the selected
    hyperparameters, passed through from run_tuning_step plus
    selected_hyperparameters added here — plus the tree-count scaling
    derivation: n_estimators_es_median, n_fold_fit, n_final_fit,
    n_estimators_scale_factor, n_estimators_shipped, and the two-count
    diagnostic cv_pr_auc_at_n_es_median/cv_pr_auc_at_n_scaled, also logged as
    MLflow metrics so they're plottable across cycles). The two-count
    diagnostic never gates or selects — it only confirms the a-priori scaling
    rule against this project's own data; a regression is logged as a warning
    for manual investigation, not enforced here. The stakeholder-facing
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
    best_params = dict(tuning_result["best_params"])
    X_committed = X_dev[committed_features]

    # Tree-count scaling correction (PROJECT_PLAN.md Fix 5): n_estimators is an
    # early-stopping *output*, derived on each fold's training partition after
    # tuning.py carves out an es_validation_size slice — smaller than the final
    # fit's row count by construction, so the raw median under-boosts the final
    # pipeline unless corrected here.
    n_estimators_es_median = int(tuning_result["best_n_estimators_median"])
    cv_folds = int(cfg.tuning.cv_folds)
    es_validation_size = float(cfg.tuning.es_validation_size)
    n_final_fit = len(y_dev)
    n_fold_fit = round(
        n_final_fit * (cv_folds - 1) / cv_folds * (1 - es_validation_size)
    )
    n_estimators_scale_factor = n_final_fit / n_fold_fit
    n_estimators_final = round(n_estimators_es_median * n_estimators_scale_factor)

    # Two-count diagnostic: confirms the a-priori scaling rule on this
    # project's own data rather than by citation alone — a check, never a
    # selection. Reuses tuning.py's exact outer-fold structure
    # (cfg.tuning.random_state, not cfg.random_seed, is the seed that
    # actually produced those folds); fits plain estimators in memory and
    # logs no MLflow model, so neither count can mint a second candidate.
    diagnostic_lgbm_params = {**best_params, **fixed_hyperparameters}
    cv_pr_auc_at_n_es_median = _cv_pr_auc_at_n_estimators(
        n_estimators_es_median,
        X_committed,
        y_dev,
        diagnostic_lgbm_params,
        binary,
        multi_cat,
        numeric,
        cv_folds,
        int(cfg.tuning.random_state),
    )
    cv_pr_auc_at_n_scaled = _cv_pr_auc_at_n_estimators(
        n_estimators_final,
        X_committed,
        y_dev,
        diagnostic_lgbm_params,
        binary,
        multi_cat,
        numeric,
        cv_folds,
        int(cfg.tuning.random_state),
    )
    if cv_pr_auc_at_n_scaled < cv_pr_auc_at_n_es_median:
        logger.warning(
            "n_estimators_scaling_regressed",
            cv_pr_auc_at_n_es_median=cv_pr_auc_at_n_es_median,
            cv_pr_auc_at_n_scaled=cv_pr_auc_at_n_scaled,
            n_estimators_es_median=n_estimators_es_median,
            n_estimators_final=n_estimators_final,
            hint=(
                "the scaled tree count scored worse than the raw early-stopped "
                "median on the same CV folds — investigate the tuned "
                "regularisation before trusting the scaling correction; do not "
                "ship the textbook answer on faith"
            ),
        )

    selected_hyperparameters = {
        "n_estimators": n_estimators_final,
        **best_params,
    }
    model_params = {**selected_hyperparameters, **fixed_hyperparameters}

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
            # Tree-count scaling correction provenance (Fix 5) — the derivation
            # must be legible without re-deriving it: n_estimators is not a
            # tuned hyperparameter, so its shipped value has no other audit trail.
            "n_estimators_es_median": n_estimators_es_median,
            "n_fold_fit": n_fold_fit,
            "n_final_fit": n_final_fit,
            "n_estimators_scale_factor": n_estimators_scale_factor,
            "n_estimators_shipped": n_estimators_final,
            "cv_pr_auc_at_n_es_median": cv_pr_auc_at_n_es_median,
            "cv_pr_auc_at_n_scaled": cv_pr_auc_at_n_scaled,
        },
    }

    mlflow.set_tracking_uri(resolve_tracking_uri(str(cfg.mlflow.tracking_uri)))
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    run_id = str(tuning_result["parent_run_id"])
    with mlflow.start_run(run_id=run_id):
        mlflow.log_text("\n".join(full_feature_space), "feature_space.txt")
        mlflow.log_text("\n".join(committed_features), "feature_columns.txt")
        mlflow.log_metrics(
            {
                "cv_pr_auc_at_n_es_median": cv_pr_auc_at_n_es_median,
                "cv_pr_auc_at_n_scaled": cv_pr_auc_at_n_scaled,
            }
        )

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

"""Step 4: Optuna hyperparameter tuning on the frozen feature set."""

from __future__ import annotations

import hashlib
import json
import math
from functools import partial
from typing import Any

import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sqlalchemy import text

from telco_churn.features.build import FEATURE_SCHEMA
from telco_churn.features.preprocessing import build_preprocessor
from telco_churn.models.train.common import (
    _dvc_hash,
    _git_sha,
    _lgbm_fixed_knobs,
    _log_dev_input,
)
from telco_churn.utils.db import get_engine
from telco_churn.utils.logging import get_logger
from telco_churn.utils.mlflow import resolve_tracking_uri

__all__ = [
    "boundary_hit_check",
    "run_tuning_step",
    "select_best_trial",
]

logger = get_logger(__name__)


def _raw_best_diagnostics(trial_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Raw-best trial identity and its 1-SE band, independent of selection_rule.

    Shared by select_best_trial's 1-SE branch and run_tuning_step's returned
    tuning_summary, so the SE/band-floor formula has exactly one implementation.
    Notebooks and training_manifest.json read the same numbers selection actually
    used, instead of re-deriving them client-side from MLflow's rounded
    fold_pr_auc metric history.
    """
    best = max(trial_summaries, key=lambda t: t["value"])
    n_folds = len(best["fold_scores"])
    se = (
        float(np.std(best["fold_scores"], ddof=1)) / math.sqrt(n_folds)
        if n_folds > 1
        else 0.0
    )
    return {
        "raw_best_trial_number": best["number"],
        "raw_best_cv_pr_auc": best["value"],
        "se": se,
        "band_floor": best["value"] - se,
    }


def select_best_trial(
    trial_summaries: list[dict[str, Any]],
    selection_rule: str,
) -> dict[str, Any]:
    """Pick the best completed Optuna trial by raw argmax or the 1-SE rule.

    trial_summaries — one dict per completed trial: "number", "value" (mean CV
    PR-AUC), "fold_scores" (per-fold PR-AUC list), "params" (hyperparameter dict).

    'argmax' returns the highest-value trial. '1se' picks the most-regularized
    trial within one standard error of the best, breaking ties by fewest
    num_leaves then fewest n_estimators_median — both are structural capacity
    controls in leaf-wise boosting (leaf count, tree count), not penalty-term
    magnitudes, so the tiebreak stays in the same complexity register as the
    primary key. num_leaves itself is LightGBM's documented main lever for
    tree complexity under leaf-wise growth (per its Parameters Tuning guide),
    independent of this study's own fANOVA ranking in
    notebooks/03c-hyperparameter-tuning.ipynb §3.
    """
    if not trial_summaries:
        raise ValueError("trial_summaries must not be empty")

    if selection_rule == "argmax":
        return max(trial_summaries, key=lambda t: t["value"])

    diagnostics = _raw_best_diagnostics(trial_summaries)
    within_band = [
        t for t in trial_summaries if t["value"] >= diagnostics["band_floor"]
    ]

    return min(
        within_band,
        key=lambda t: (t["params"]["num_leaves"], t["n_estimators_median"]),
    )


def boundary_hit_check(
    params: dict[str, Any],
    search_space: dict[str, Any],
    rel_tol: float = 1e-9,
) -> dict[str, bool]:
    """Flag any selected hyperparameter sitting on its searched range's edge.

    Gates nothing — recorded so a narrow search range doesn't silently under-tune
    the model.
    """
    hits: dict[str, bool] = {}
    for name, spec in search_space.items():
        if name not in params:
            continue
        value = float(params[name])
        low, high = float(spec["low"]), float(spec["high"])
        hits[name] = math.isclose(value, low, rel_tol=rel_tol) or math.isclose(
            value, high, rel_tol=rel_tol
        )
    return hits


def _suggest_lgbm_params(
    trial: optuna.Trial, search_space: DictConfig
) -> dict[str, Any]:
    """Sample one LightGBM hyperparameter set from the Hydra-configured search space."""
    params: dict[str, Any] = {}
    for name, spec in search_space.items():
        if str(spec.type) == "int":
            params[str(name)] = trial.suggest_int(
                str(name), int(spec.low), int(spec.high)
            )
        else:
            params[str(name)] = trial.suggest_float(
                str(name),
                float(spec.low),
                float(spec.high),
                log=bool(spec.get("log", False)),
            )
    return params


def _tuning_objective(
    trial: optuna.Trial,
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: list[tuple[Any, Any]],
    binary: list[str],
    multi_cat: list[str],
    numeric: list[str],
    tuning_cfg: DictConfig,
    fixed_params: dict[str, Any],
    random_state: int,
    pruning_enabled: bool,
) -> float:
    """One Optuna trial: sample params, run early-stopped CV, log a nested MLflow run.

    The early-stopping validation slice is carved out of each fold's training
    portion, not its held-out rows — preventing leakage into the CV score.

    A pruned trial's MLflow run closes cleanly (TrialPruned is raised after the
    `with` block exits), so pruning isn't logged as a FAILED run. A mid-fit
    exception instead propagates out of the block, closing the run as FAILED;
    run_tuning_step's `study.optimize(..., catch=...)` then catches it so one
    bad trial doesn't kill the study.
    """
    params = _suggest_lgbm_params(trial, tuning_cfg.search_space)
    es_validation_size = float(tuning_cfg.es_validation_size)

    pruned = False
    cv_mean = float("nan")
    with mlflow.start_run(run_name=f"trial_{trial.number:03d}", nested=True):
        mlflow.log_params(params)
        fold_scores: list[float] = []
        fold_n_estimators: list[int] = []

        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            X_fit, X_es, y_fit, y_es = train_test_split(
                X_tr,
                y_tr,
                test_size=es_validation_size,
                stratify=y_tr,
                random_state=random_state,
            )

            preprocessor = build_preprocessor(binary, multi_cat, numeric)
            X_fit_t = preprocessor.fit_transform(X_fit)
            X_es_t = preprocessor.transform(X_es)
            X_val_t = preprocessor.transform(X_val)

            model = LGBMClassifier(
                n_estimators=int(tuning_cfg.n_estimators_ceiling),
                **fixed_params,
                **params,
            )
            model.fit(
                X_fit_t,
                y_fit,
                eval_set=[(X_es_t, y_es)],
                eval_metric="average_precision",
                callbacks=[
                    lgb.early_stopping(
                        int(tuning_cfg.early_stopping_rounds), verbose=False
                    )
                ],
            )
            proba = model.predict_proba(X_val_t)[:, 1]
            fold_ap = float(average_precision_score(y_val, proba))
            fold_scores.append(fold_ap)
            fold_n_estimators.append(
                int(model.best_iteration_ or tuning_cfg.n_estimators_ceiling)
            )

            trial.report(fold_ap, fold_idx)
            # Logged rounded for readability only — fold_ap itself stays full-precision
            # for trial.report/pruning and the 1-SE selection band below, since
            # quantizing the actual optimization signal to 3dp risks masking the
            # small, close-trial gaps TPE and the pruner are trying to resolve.
            mlflow.log_metric("fold_pr_auc", round(fold_ap, 3), step=fold_idx)

            if pruning_enabled and trial.should_prune():
                mlflow.set_tag("pruned", "true")
                pruned = True
                break

        if not pruned:
            cv_mean = float(np.mean(fold_scores))
            n_estimators_median = int(np.median(fold_n_estimators))
            trial.set_user_attr("fold_scores", fold_scores)
            trial.set_user_attr("n_estimators_median", n_estimators_median)
            # Logged full-precision, unlike other reported metrics elsewhere in this
            # pipeline: 03c-hyperparameter-tuning.ipynb reconstructs a pseudo-study from
            # this exact metric to compute fANOVA hyperparameter importance, so it's an
            # input to a downstream calculation, not a terminal reported value — rounding
            # it can collapse two close-but-distinct trials to an identical value and
            # crash fANOVA on zero variance when few trials complete.
            mlflow.log_metrics(
                {
                    "cv_pr_auc_mean": cv_mean,
                    "cv_pr_auc_std": float(np.std(fold_scores, ddof=1)),
                }
            )
            mlflow.log_param("n_estimators_median", n_estimators_median)

    if pruned:
        raise optuna.TrialPruned()
    return cv_mean


def _build_optuna_storage() -> optuna.storages.RDBStorage:
    """Postgres-backed Optuna storage, isolated in its own 'optuna' schema.

    Reuses POSTGRES_URL (the same server MLflow's backend store runs against)
    rather than a local sqlite file, so a study survives a crashed process and
    can be reloaded directly with optuna.load_study — no more reconstructing it
    from MLflow-logged trial params just to run fANOVA importance. The schema
    is isolated (not the 'public' schema) so Optuna's own tables (studies,
    trials, ...) can't collide with application tables.
    """
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS optuna"))
    url = engine.url.render_as_string(hide_password=False)
    return optuna.storages.RDBStorage(
        url=url,
        engine_kwargs={"connect_args": {"options": "-csearch_path=optuna"}},
    )


def _study_name(cfg: DictConfig, committed_features: list[str]) -> str:
    """Content-addressed study name: same inputs resume the same study.

    Hashes data version, frozen features, and the tuning config (search space,
    CV scheme, early-stopping settings, pruner) — everything that must stay fixed
    across a study's trials for Optuna's per-parameter distributions to stay valid,
    plus pruner (including its n_warmup_steps): a trial pruned under one policy
    and a trial that ran unpruned under another aren't a fair comparison, so
    mixing them into one pool would let 1-SE selection silently compare apples
    to oranges. Any change to these starts a fresh study instead of silently
    mixing incompatible trials into an old one.
    """
    tuning_cfg = cfg.tuning
    key = {
        "dvc_hash": _dvc_hash(cfg),
        "committed_features": sorted(committed_features),
        "search_space": OmegaConf.to_container(tuning_cfg.search_space, resolve=True),
        "cv_folds": int(tuning_cfg.cv_folds),
        "random_state": int(tuning_cfg.random_state),
        "n_estimators_ceiling": int(tuning_cfg.n_estimators_ceiling),
        "early_stopping_rounds": int(tuning_cfg.early_stopping_rounds),
        "es_validation_size": float(tuning_cfg.es_validation_size),
        "pruner": str(tuning_cfg.pruner),
        "pruner_n_warmup_steps": int(tuning_cfg.pruner_n_warmup_steps),
    }
    digest = hashlib.sha256(
        json.dumps(key, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    return f"tuning_{digest}"


def run_tuning_step(
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    committed_features: list[str],
    cfg: DictConfig,
    storage: optuna.storages.BaseStorage | None = None,
) -> dict[str, Any]:
    """Optuna tuning of LightGBM on the frozen feature set.

    Must run after feature selection freezes the input space — rerunning
    selection afterward invalidates the study. Uses single stratified CV
    (cfg.tuning.cv_folds), not the repeated CV used earlier for model and
    feature-set comparison — this step already multiplies cost by n_trials,
    so repeats aren't worth the added compute here. n_estimators is not
    searched; each trial resolves its own ceiling via early stopping on
    average_precision.

    A trial that raises is marked FAILED rather than aborting the study. Too
    few completed trials (cfg.tuning.min_completed_trials) logs a warning and
    persists trial_count_below_threshold into tuning_summary and the
    trial_count_below_threshold MLflow metric — visible after the fact, not
    just in the log stream — but does not block selection: this step only
    flags an untrustworthy result, it does not gate on it. (Enforcement
    belongs at the point a tuned pipeline becomes a registered model — Phase
    6's calibrate.py — not here; see PROJECT_PLAN.md Phase 6.) storage
    defaults to a Postgres-backed study (see _build_optuna_storage); pass an
    InMemoryStorage for tests.

    Idempotent against a study that already reached n_trials: only the trials
    still needed to reach n_trials are run, so re-running against a completed
    study reuses its existing trials instead of piling n_trials more on top.

    Returns {"best_params", "best_n_estimators_median", "best_cv_pr_auc_mean",
    "boundary_hits", "n_completed_trials", "parent_run_id", "committed_features",
    "tuning_summary"}. tuning_summary carries the audit trail behind the
    selection_rule pick — trial counts, selected vs. raw-best trial number/score,
    the 1-SE standard error and band floor — so training_manifest.json and any
    notebook narrating the decision read the same numbers select_best_trial
    actually used, not a client-side reconstruction.
    """
    tuning_cfg = cfg.tuning
    random_state = int(tuning_cfg.random_state)
    raw_timeout = tuning_cfg.get("timeout_seconds", None)
    timeout_seconds = float(raw_timeout) if raw_timeout is not None else None
    raw_min_completed = tuning_cfg.get("min_completed_trials", None)
    min_completed_trials = (
        int(raw_min_completed) if raw_min_completed is not None else None
    )

    binary = [c for c in FEATURE_SCHEMA.binary if c in committed_features]
    multi_cat = [c for c in FEATURE_SCHEMA.multi_cat if c in committed_features]
    numeric = [c for c in FEATURE_SCHEMA.numeric if c in committed_features]

    fixed_params = _lgbm_fixed_knobs(cfg, random_state)

    skf = StratifiedKFold(
        n_splits=int(tuning_cfg.cv_folds), shuffle=True, random_state=random_state
    )
    cv_splits = list(skf.split(X_dev, y_dev))

    pruning_enabled = str(tuning_cfg.pruner) == "median"
    pruner: optuna.pruners.BasePruner = (
        optuna.pruners.MedianPruner(
            n_warmup_steps=int(tuning_cfg.pruner_n_warmup_steps)
        )
        if pruning_enabled
        else optuna.pruners.NopPruner()
    )
    sampler = optuna.samplers.TPESampler(
        seed=int(tuning_cfg.sampler_seed),
        n_startup_trials=int(tuning_cfg.n_startup_trials),
    )
    study_name = _study_name(cfg, committed_features)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage if storage is not None else _build_optuna_storage(),
        load_if_exists=True,
        direction=str(tuning_cfg.direction),
        sampler=sampler,
        pruner=pruner,
    )

    raw_warm_start = tuning_cfg.get("warm_start_params", None)
    if raw_warm_start is not None:
        warm_start_params = {str(k): v for k, v in raw_warm_start.items()}
        study.enqueue_trial(warm_start_params, skip_if_exists=True)

    mlflow.set_tracking_uri(resolve_tracking_uri(str(cfg.mlflow.tracking_uri)))
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run(run_name="tuning_study") as parent_run:
        mlflow.set_tags(
            {
                "stage": "tuning",
                "git_sha": _git_sha(),
                "dvc_data_hash": _dvc_hash(cfg),
            }
        )
        # Also covers log_model.py's "model" and calibrate.py's "calibrated_model"
        # LoggedModels for free — both reuse this run's run_id, and log_input
        # attaches to the run, not to any one LoggedModel logged onto it.
        _log_dev_input(X_dev, y_dev, context="training")
        mlflow.log_params(
            {
                "optuna_study_name": study_name,
                "n_trials": int(tuning_cfg.n_trials),
                "sampler": "tpe",
                "sampler_seed": int(tuning_cfg.sampler_seed),
                "n_startup_trials": int(tuning_cfg.n_startup_trials),
                "pruner": str(tuning_cfg.pruner),
                "pruner_n_warmup_steps": int(tuning_cfg.pruner_n_warmup_steps),
                "selection_rule": str(tuning_cfg.selection_rule),
                "cv_folds": int(tuning_cfg.cv_folds),
                "n_estimators_ceiling": int(tuning_cfg.n_estimators_ceiling),
                "early_stopping_rounds": int(tuning_cfg.early_stopping_rounds),
                "n_committed_features": len(committed_features),
                "timeout_seconds": timeout_seconds,
                "min_completed_trials": min_completed_trials,
            }
        )

        objective = partial(
            _tuning_objective,
            X=X_dev,
            y=y_dev,
            cv_splits=cv_splits,
            binary=binary,
            multi_cat=multi_cat,
            numeric=numeric,
            tuning_cfg=tuning_cfg,
            fixed_params=fixed_params,
            random_state=random_state,
            pruning_enabled=pruning_enabled,
        )
        # load_if_exists=True resumes a study interrupted mid-run (e.g. a crashed
        # process) rather than losing its trials — but without this guard, re-running
        # a study that already reached n_trials would just pile on n_trials more on
        # top of the existing ones every time, silently growing the candidate pool
        # each run instead of reproducing it.
        n_remaining_trials = max(int(tuning_cfg.n_trials) - len(study.trials), 0)
        if n_remaining_trials > 0:
            study.optimize(
                objective,
                n_trials=n_remaining_trials,
                timeout=timeout_seconds,
                catch=(Exception,),
            )
        else:
            logger.info(
                "tuning_study_already_complete",
                study_name=study_name,
                n_trials=len(study.trials),
            )

        n_failed_trials = sum(
            1 for t in study.trials if t.state == optuna.trial.TrialState.FAIL
        )
        n_pruned_trials = sum(
            1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
        )
        if n_failed_trials:
            logger.warning(
                "tuning_trials_failed",
                n_failed_trials=n_failed_trials,
                n_total_trials=len(study.trials),
            )

        trial_summaries = [
            {
                "number": t.number,
                "value": t.value,
                "fold_scores": t.user_attrs["fold_scores"],
                "n_estimators_median": t.user_attrs["n_estimators_median"],
                "params": t.params,
            }
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        trial_count_below_threshold = (
            min_completed_trials is not None
            and len(trial_summaries) < min_completed_trials
        )
        if trial_count_below_threshold:
            logger.warning(
                "tuning_too_few_completed_trials",
                n_completed_trials=len(trial_summaries),
                min_completed_trials=min_completed_trials,
                n_failed_trials=n_failed_trials,
                selection_rule=str(tuning_cfg.selection_rule),
                hint=(
                    "selection_rule pick is drawn from too few completed trials "
                    "to be a meaningful selection — investigate pruning/failures "
                    "or increase n_trials before trusting this run's champion"
                ),
            )
        selected = select_best_trial(trial_summaries, str(tuning_cfg.selection_rule))
        diagnostics = _raw_best_diagnostics(trial_summaries)
        search_space_dict = {
            str(name): {"low": spec.low, "high": spec.high}
            for name, spec in tuning_cfg.search_space.items()
        }
        boundary_hits = boundary_hit_check(selected["params"], search_space_dict)
        hit_params = [name for name, hit in boundary_hits.items() if hit]
        if hit_params:
            logger.warning(
                "tuning_boundary_hit",
                hit_params=hit_params,
                selected_params=selected["params"],
                hint=(
                    "selected trial's params sit on a searched range's edge — "
                    "widen the range in configs/tuning/optuna.yaml search_space "
                    "and re-run"
                ),
            )

        fig_hist, ax_hist = plt.subplots(figsize=(7, 4))
        trial_numbers = [t["number"] for t in trial_summaries]
        values = [t["value"] for t in trial_summaries]
        running_best = np.maximum.accumulate(values)
        ax_hist.scatter(trial_numbers, values, alpha=0.5, label="trial CV PR-AUC")
        ax_hist.plot(trial_numbers, running_best, color="C1", label="running best")
        ax_hist.set_xlabel("Trial")
        ax_hist.set_ylabel("CV PR-AUC")
        ax_hist.set_title("Optuna Optimization History")
        ax_hist.legend()

        mlflow.log_params({f"best_{k}": v for k, v in selected["params"].items()})
        mlflow.log_params(
            {
                "best_trial_number": selected["number"],
                "best_n_estimators_median": selected["n_estimators_median"],
                "raw_best_trial_number": diagnostics["raw_best_trial_number"],
            }
        )
        mlflow.log_metrics(
            {
                # "selected_" (not "best_"): this is the 1-SE-adopted trial's score,
                # not the raw-argmax winner — "best_cv_pr_auc_mean" read as exactly
                # the opposite of what it is on first glance in the MLflow UI.
                "selected_cv_pr_auc_mean": round(selected["value"], 3),
                "raw_best_cv_pr_auc_mean": round(diagnostics["raw_best_cv_pr_auc"], 3),
                # "one_se_band_*" (not "raw_best_*"): these aren't a property of the
                # raw-best trial in isolation, they're the 1-SE selection band derived
                # from it — the mechanism that decides who counts as "close enough" to
                # the winner. "raw_best_se" reads as "SE of the raw-best score" rather
                # than "the band selection is judged against."
                "one_se_band_se": diagnostics["se"],
                "one_se_band_floor": diagnostics["band_floor"],
                "n_completed_trials": len(trial_summaries),
                "n_pruned_trials": n_pruned_trials,
                "n_boundary_hits": sum(boundary_hits.values()),
                "n_failed_trials": n_failed_trials,
                # int(): MLflow metrics must be numeric. Surfaces the too-few-trials
                # warning (logged above, ephemeral) as a persistent, queryable
                # signal in the MLflow UI/API, not just a vanished log line.
                "trial_count_below_threshold": int(trial_count_below_threshold),
            }
        )
        mlflow.log_table(
            pd.DataFrame(trial_summaries).drop(columns=["fold_scores"]),
            "tuning/trials.json",
        )
        mlflow.log_dict(boundary_hits, "tuning/boundary_hits.json")
        mlflow.log_figure(fig_hist, "tuning/optimization_history.png")
        plt.close(fig_hist)

        parent_run_id = parent_run.info.run_id

    logger.info(
        "tuning_step_done",
        best_trial_number=selected["number"],
        best_cv_pr_auc_mean=round(selected["value"], 3),
        selection_rule=str(tuning_cfg.selection_rule),
        boundary_hits=boundary_hits,
    )

    return {
        "best_params": selected["params"],
        "best_n_estimators_median": selected["n_estimators_median"],
        "best_cv_pr_auc_mean": selected["value"],
        "boundary_hits": boundary_hits,
        "n_completed_trials": len(trial_summaries),
        "parent_run_id": parent_run_id,
        "committed_features": committed_features,
        "tuning_summary": {
            "n_trials_requested": int(tuning_cfg.n_trials),
            "n_completed_trials": len(trial_summaries),
            "n_pruned_trials": n_pruned_trials,
            "n_failed_trials": n_failed_trials,
            "min_completed_trials": min_completed_trials,
            "trial_count_below_threshold": trial_count_below_threshold,
            "selection_rule": str(tuning_cfg.selection_rule),
            "selected_trial_number": selected["number"],
            "selected_cv_pr_auc": selected["value"],
            "raw_best_trial_number": diagnostics["raw_best_trial_number"],
            "raw_best_cv_pr_auc": diagnostics["raw_best_cv_pr_auc"],
            "se": diagnostics["se"],
            "band_floor": diagnostics["band_floor"],
            "boundary_hits": boundary_hits,
        },
    }

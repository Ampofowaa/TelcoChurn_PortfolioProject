"""Step 4: Optuna hyperparameter tuning on the frozen feature set."""

from __future__ import annotations

import hashlib
import json
import math
import os
from functools import partial
from typing import Any

import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from matplotlib.figure import Figure
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sqlalchemy import create_engine, text

from telco_churn.features.accessor import features_sha256
from telco_churn.features.build import FEATURE_SCHEMA
from telco_churn.features.preprocessing import build_preprocessor
from telco_churn.models.train.common import (
    _git_sha,
    _lgbm_fixed_knobs,
    _log_dev_input,
)
from telco_churn.utils.logging import get_logger
from telco_churn.utils.mlflow import ensure_experiment_metadata
from telco_churn.utils.paths import get_project_root

__all__ = [
    "boundary_hit_check",
    "run_tuning_step",
    "select_best_trial",
]

logger = get_logger(__name__)

# Path to the authoritative DDL for Optuna's isolated schema — same
# sql/schema/*.sql convention customers_raw's table follows (ingest.py's
# _SQL_SCHEMA), not an inline string here.
_OPTUNA_SCHEMA_SQL = (
    get_project_root() / "sql" / "schema" / "002_create_optuna_schema.sql"
)


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
    """Sample one LightGBM hyperparameter set from the Hydra-configured search space.

    max_depth and num_leaves are independent leaf-wise-growth regularizers —
    LightGBM applies whichever binds first, so a trial where max_depth caps
    the tree before num_leaves is exhausted is a valid shallow-tree config,
    not an error. An earlier version of this function coupled max_depth's
    low to ceil(log2(num_leaves.high)) to prevent that case; on this dataset
    it excluded the shallow-tree region the study actually preferred
    (max_depth=4 paired with num_leaves=6 in the 1-SE-selected trial,
    ANALYSIS.md §4b) and measurably lowered both the raw-best and 1-SE CV
    PR-AUC on a re-run. Removed rather than re-anchored.
    """
    params: dict[str, Any] = {}
    for name, spec in search_space.items():
        name = str(name)
        if str(spec.type) == "int":
            params[name] = trial.suggest_int(name, int(spec.low), int(spec.high))
        else:
            params[name] = trial.suggest_float(
                name,
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
    study_name: str,
) -> float:
    """One Optuna trial: sample params, run early-stopped CV, log a nested MLflow run.

    The early-stopping validation slice is carved out of each fold's training
    portion, not its held-out rows — preventing leakage into the CV score.

    A pruned trial's MLflow run closes cleanly (TrialPruned is raised after the
    `with` block exits), so pruning isn't logged as a FAILED run. A mid-fit
    exception instead propagates out of the block, closing the run as FAILED;
    run_tuning_step's `study.optimize(..., catch=...)` then catches it so one
    bad trial doesn't kill the study.

    Tagged with `optuna_study_name` (not just implicitly parented via MLflow's
    own `mlflow.parentRunId`) because the study — not the parent run — is the
    unit of continuity: `load_if_exists=True` resumes a content-addressed study
    across separate `run_tuning_step` invocations (a crash-resume, or a rerun
    that finds the study already complete), so a trial's earlier siblings can
    live under a different, previous parent run entirely. Querying by this tag
    finds every trial a study has ever run; querying by parentRunId only finds
    the ones logged under one specific invocation.
    """
    params = _suggest_lgbm_params(trial, tuning_cfg.search_space)
    es_validation_size = float(tuning_cfg.es_validation_size)

    pruned = False
    cv_mean = float("nan")
    with mlflow.start_run(run_name=f"trial_{trial.number:03d}", nested=True):
        mlflow.set_tag("optuna_study_name", study_name)
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
    """Optuna storage: Postgres-backed when POSTGRES_URL is set, a local
    SQLite file otherwise — the same zero-infra fallback utils.mlflow's
    tracking_uri already uses, so a fresh clone or CI run doesn't need Docker
    just to run this step.

    Postgres path reuses POSTGRES_URL (the same server MLflow's backend store
    runs against) rather than a local sqlite file, so a study survives a
    crashed process and can be reloaded directly with optuna.load_study — no
    more reconstructing it from MLflow-logged trial params just to run fANOVA
    importance. Isolated in its own 'optuna' schema (not 'public') so Optuna's
    own tables (studies, trials, ...) can't collide with application tables —
    the schema's DDL lives in sql/schema/002_create_optuna_schema.sql, not an
    inline string here; this executes it explicitly (the same belt-and-braces
    role utils.db.py::apply_migrations plays for the Alembic-managed tables)
    since docker-compose.yml's docker-entrypoint-initdb.d mount only fires
    against a fresh Postgres data volume — 002_create_optuna_schema.sql stays
    outside this project's Alembic instance permanently (PROJECT_PLAN.md's
    Phase 10a-i scope-boundary note), so it keeps this direct-execution
    pattern rather than moving into a migration. Builds its own engine rather
    than reusing utils.db.get_engine()'s shared singleton: that one stays strict (raises
    with no POSTGRES_URL) for ingest.py/sql_features.py, whose SQL is
    Postgres-only and must never silently run against a SQLite fallback.

    The SQLite fallback skips schema isolation (SQLite has no schema/
    search_path concept, and nothing else runs against the fallback file to
    collide with) and isn't shared across machines, so a study doesn't
    survive a crashed process the way the Postgres path does — acceptable for
    a fresh clone or a CI run, both single-shot by construction.
    """
    url = os.environ.get("POSTGRES_URL")
    if not url:
        return optuna.storages.RDBStorage(
            url=f"sqlite:///{get_project_root()}/optuna.db"
        )

    engine = create_engine(url, pool_pre_ping=True)
    with engine.begin() as conn:
        conn.execute(text(_OPTUNA_SCHEMA_SQL.read_text()))
    return optuna.storages.RDBStorage(
        url=engine.url.render_as_string(hide_password=False),
        engine_kwargs={"connect_args": {"options": "-csearch_path=optuna"}},
    )


def _study_name(cfg: DictConfig, committed_features: list[str]) -> str:
    """Content-addressed study name: same inputs resume the same study.

    Two independent hashes, not one combined digest, so the name itself is
    legible: a human scanning the `optuna` schema or the MLflow UI can tell
    at a glance whether a new study came from a data change or a config
    change without decoding anything. `data_hash` is the processed-features
    file's own sha256 (features/accessor.py::features_sha256) — real and
    unconditional, unlike the retired `_dvc_hash`, which read a `.dvc`
    sidecar that never existed for a pipeline-stage output and was
    permanently `"unknown"`. `config_digest` hashes frozen features plus the
    tuning config (search space, CV scheme, early-stopping settings, pruner,
    sampler, objective) — everything that must stay fixed across a study's
    trials for Optuna's per-parameter distributions to stay valid and for its
    stored trial values to mean the same thing, plus the pruner's
    n_warmup_steps: a trial pruned under one policy and a trial that ran
    unpruned under another aren't a fair comparison, so mixing them into one
    pool would let 1-SE selection silently compare apples to oranges. `metric`
    and `direction` guard against the same failure at the objective level — a
    metric swap or a maximize/minimize flip changes what a stored trial value
    even means, and a `direction` flip is worse than incomparable: trials good
    under "maximize" would read as bad under "minimize" and vice versa.
    `sampler_seed`/`n_startup_trials` guard the reproducibility claim in this
    docstring's own first line — two runs with a different seed or warm-up
    budget aren't "the same inputs" even though nothing about the search
    space changed. A change to any of these starts a fresh study instead of
    silently mixing incompatible trials into an old one.
    """
    data_hash = features_sha256()
    tuning_cfg = cfg.tuning
    config_key = {
        "committed_features": sorted(committed_features),
        "search_space": OmegaConf.to_container(tuning_cfg.search_space, resolve=True),
        "cv_folds": int(tuning_cfg.cv_folds),
        "random_state": int(tuning_cfg.random_state),
        "n_estimators_ceiling": int(tuning_cfg.n_estimators_ceiling),
        "early_stopping_rounds": int(tuning_cfg.early_stopping_rounds),
        "es_validation_size": float(tuning_cfg.es_validation_size),
        "pruner": str(tuning_cfg.pruner),
        "pruner_n_warmup_steps": int(tuning_cfg.pruner_n_warmup_steps),
        "metric": str(tuning_cfg.metric),
        "direction": str(tuning_cfg.direction),
        "sampler_seed": int(tuning_cfg.sampler_seed),
        "n_startup_trials": int(tuning_cfg.n_startup_trials),
    }
    config_digest = hashlib.sha256(
        json.dumps(config_key, sort_keys=True, default=str).encode()
    ).hexdigest()
    return f"tuning_{data_hash[:8]}_{config_digest[:8]}"


def _discard_incomplete_study_unless_resuming(
    storage: optuna.storages.BaseStorage,
    study_name: str,
    n_trials_requested: int,
    resume: bool,
) -> None:
    """Delete a same-named study left incomplete by a prior invocation, unless
    `tuning.resume` explicitly opts into continuing it.

    A complete study (>= n_trials_requested trials already recorded) is
    always reused regardless of `resume` — that's a cache hit, not a resume
    decision, and is handled by `optuna.create_study`'s own `load_if_exists`
    afterward. An *incomplete* study is ambiguous: it might be a deliberate
    in-progress run worth continuing, or a crashed/interrupted process's
    leftover state. Defaulting to discard-and-restart keeps a stale partial
    study from silently pooling with a later invocation's trials.
    """
    try:
        existing = optuna.load_study(study_name=study_name, storage=storage)
    except KeyError:
        return
    if len(existing.trials) >= n_trials_requested or resume:
        return
    optuna.delete_study(study_name=study_name, storage=storage)


def _build_optuna_study(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    committed_features: list[str],
    cfg: DictConfig,
    storage: optuna.storages.BaseStorage | None,
    warm_start_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build (or resume) the Optuna study, its pruner/sampler, and the CV splits trials will use.

    Study creation and CV-split resolution don't need an active MLflow run —
    only trial execution does, since each trial opens its own nested run.

    A same-named incomplete study from a prior invocation is discarded first
    unless `tuning.resume` is true (see `_discard_incomplete_study_unless_resuming`)
    — `load_if_exists=True` below then either finds nothing (fresh study) or
    finds a study eligible to be reused (complete, or resume explicitly requested).

    warm_start_params, when given, takes precedence over
    `cfg.tuning.warm_start_params` as the trial enqueued before the search
    starts — from Phase 10b on, the caller resolves this dynamically from
    the current champion's own `tuned_hyperparameters` model-version tag
    rather than this project's one-time, hand-set config prior. Falling
    back to the config
    value when the caller passes nothing keeps today's Phase 5 cold-start
    behavior unchanged.
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
    cv_splits = list(skf.split(X_train, y_train))

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
    study_storage = storage if storage is not None else _build_optuna_storage()
    _discard_incomplete_study_unless_resuming(
        study_storage,
        study_name,
        n_trials_requested=int(tuning_cfg.n_trials),
        resume=bool(tuning_cfg.resume),
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=study_storage,
        load_if_exists=True,
        direction=str(tuning_cfg.direction),
        sampler=sampler,
        pruner=pruner,
    )

    resolved_warm_start = (
        warm_start_params
        if warm_start_params is not None
        else tuning_cfg.get("warm_start_params", None)
    )
    if resolved_warm_start is not None:
        enqueue_params = {str(k): v for k, v in resolved_warm_start.items()}
        study.enqueue_trial(enqueue_params, skip_if_exists=True)

    return {
        "tuning_cfg": tuning_cfg,
        "random_state": random_state,
        "timeout_seconds": timeout_seconds,
        "min_completed_trials": min_completed_trials,
        "binary": binary,
        "multi_cat": multi_cat,
        "numeric": numeric,
        "fixed_params": fixed_params,
        "cv_splits": cv_splits,
        "pruning_enabled": pruning_enabled,
        "study_name": study_name,
        "study": study,
    }


def _run_study_trials(
    setup: dict[str, Any], X_train: pd.DataFrame, y_train: pd.Series
) -> dict[str, Any]:
    """Run the study's remaining trials — call only from inside an active MLflow run.

    Each trial opens its own nested MLflow run, which requires a parent run
    already open. load_if_exists=True resumes a study interrupted mid-run
    (e.g. a crashed process) rather than losing its trials — n_remaining_trials
    guards against piling n_trials more on top of a study that already
    reached n_trials on every re-run. n_trials_before (captured ahead of
    optimize) is this invocation's reused-trial count — whatever
    `_discard_incomplete_study_unless_resuming` left the study holding, be
    that zero (fresh study) or a prior invocation's trials (complete, or
    resume explicitly requested).
    """
    study = setup["study"]
    tuning_cfg = setup["tuning_cfg"]
    n_trials_before = len(study.trials)

    objective = partial(
        _tuning_objective,
        X=X_train,
        y=y_train,
        cv_splits=setup["cv_splits"],
        binary=setup["binary"],
        multi_cat=setup["multi_cat"],
        numeric=setup["numeric"],
        tuning_cfg=tuning_cfg,
        fixed_params=setup["fixed_params"],
        random_state=setup["random_state"],
        pruning_enabled=setup["pruning_enabled"],
        study_name=setup["study_name"],
    )
    n_remaining_trials = max(int(tuning_cfg.n_trials) - n_trials_before, 0)
    if n_remaining_trials > 0:
        study.optimize(
            objective,
            n_trials=n_remaining_trials,
            timeout=setup["timeout_seconds"],
            catch=(Exception,),
        )
    else:
        logger.info(
            "tuning_study_already_complete",
            study_name=setup["study_name"],
            n_trials=n_trials_before,
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
    # catch=(Exception,) above lets one bad hyperparameter combination fail
    # without killing the study, but that same breadth would also swallow a
    # systematic bug (bad search-space config, a typo in the objective) as
    # a per-trial FAIL — which n_failed_trials only warns about, and a
    # resumed study with pre-existing completed trials could dodge the
    # min_completed_trials floor entirely. Every submitted trial failing is
    # never a legitimate per-trial fluke, so it raises instead of warning.
    if n_remaining_trials > 0 and n_failed_trials == n_remaining_trials:
        raise RuntimeError(
            f"All {n_remaining_trials} trials submitted this run failed — "
            "this is not an expected per-trial failure rate and indicates a "
            "systematic bug (search-space config, objective code) rather than "
            "an unlucky hyperparameter draw. Inspect the study's failed trial "
            "exceptions before re-running."
        )

    return {
        "n_failed_trials": n_failed_trials,
        "n_pruned_trials": n_pruned_trials,
        "n_trials_reused": n_trials_before,
        "n_trials_run_this_invocation": len(study.trials) - n_trials_before,
    }


def _summarize_completed_trials(
    setup: dict[str, Any], trial_result: dict[str, Any]
) -> dict[str, Any]:
    """Summarize completed trials, apply the selection rule, and check for boundary hits."""
    study = setup["study"]
    tuning_cfg = setup["tuning_cfg"]

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
        setup["min_completed_trials"] is not None
        and len(trial_summaries) < setup["min_completed_trials"]
    )
    if trial_count_below_threshold:
        logger.warning(
            "tuning_too_few_completed_trials",
            n_completed_trials=len(trial_summaries),
            min_completed_trials=setup["min_completed_trials"],
            n_failed_trials=trial_result["n_failed_trials"],
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

    return {
        "trial_summaries": trial_summaries,
        "trial_count_below_threshold": trial_count_below_threshold,
        "selected": selected,
        "diagnostics": diagnostics,
        "boundary_hits": boundary_hits,
    }


def _plot_optimization_history(trial_summaries: list[dict[str, Any]]) -> Figure:
    """Plot each trial's CV PR-AUC alongside the running best."""
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
    return fig_hist


def _compute_hyperparameter_importance(
    study: optuna.Study, random_state: int
) -> pd.Series | None:
    """fANOVA importance per hyperparameter, computed on the live study's completed trials.

    Runs directly against the real Optuna study (this step already holds it in
    memory) rather than a reconstruction from MLflow-logged trial params — the
    workaround a notebook rendering this after the fact has to fall back to.

    Returns None, after logging a warning, rather than raising if fANOVA can't
    be computed — e.g. too few completed trials for its internal random-forest
    surrogate to fit, which legitimately happens on a tiny study (an
    interrupted run, a fast test fixture) and should not fail a training cycle
    over a diagnostic.
    """
    evaluator = optuna.importance.FanovaImportanceEvaluator(seed=random_state)
    try:
        importances = optuna.importance.get_param_importances(
            study, evaluator=evaluator
        )
    except Exception as e:
        logger.warning(
            "hyperparameter_importance_unavailable", error=str(e), exc_info=True
        )
        return None
    return pd.Series(importances, name="importance").sort_values(ascending=True)


def _plot_hyperparameter_importance(importance: pd.Series, n_completed: int) -> Figure:
    """Horizontal bar chart of fANOVA importance, sorted ascending."""
    fig, ax = plt.subplots(figsize=(7, 4))
    importance.plot.barh(ax=ax, color="steelblue")
    ax.set_xlabel("fANOVA importance")
    ax.set_title(f"Hyperparameter importance ({n_completed} completed trials)")
    plt.tight_layout()
    return fig


def _log_tuning_artifacts(
    summary: dict[str, Any],
    trial_result: dict[str, Any],
    fig_hist: Figure,
    importance: pd.Series | None,
    fig_importance: Figure | None,
) -> None:
    """Log the selected trial's params/metrics, the trials table, boundary hits, and the history/importance plots.

    Call from inside the active `tuning_study` MLflow run. importance/
    fig_importance are None together iff fANOVA couldn't be computed
    (_compute_hyperparameter_importance already logged why) — nothing to log
    in that case, not an error.
    """
    selected, diagnostics = summary["selected"], summary["diagnostics"]
    boundary_hits = summary["boundary_hits"]
    trial_summaries = summary["trial_summaries"]

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
            "n_pruned_trials": trial_result["n_pruned_trials"],
            "n_boundary_hits": sum(boundary_hits.values()),
            "n_failed_trials": trial_result["n_failed_trials"],
            "n_trials_reused": trial_result["n_trials_reused"],
            "n_trials_run_this_invocation": trial_result[
                "n_trials_run_this_invocation"
            ],
            # int(): MLflow metrics must be numeric. Surfaces the too-few-trials
            # warning (logged above, ephemeral) as a persistent, queryable
            # signal in the MLflow UI/API, not just a vanished log line.
            "trial_count_below_threshold": int(summary["trial_count_below_threshold"]),
        }
    )
    mlflow.log_table(
        pd.DataFrame(trial_summaries).drop(columns=["fold_scores"]),
        "tuning/trials.json",
    )
    mlflow.log_dict(boundary_hits, "tuning/boundary_hits.json")
    mlflow.log_figure(fig_hist, "tuning/optimization_history.png")
    plt.close(fig_hist)
    if importance is not None and fig_importance is not None:
        mlflow.log_dict(importance.to_dict(), "tuning/hyperparameter_importance.json")
        mlflow.log_figure(fig_importance, "tuning/hyperparameter_importance.png")
        plt.close(fig_importance)


def _run_pinned_tuning_step(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    committed_features: list[str],
    cfg: DictConfig,
    pinned_params: dict[str, Any],
) -> dict[str, Any]:
    """Skip the Optuna search entirely; ship the champion's own tuned hyperparameters unchanged.

    pinned_params is the `tuned_hyperparameters` model-version tag's payload
    (register.py, §E9) — training_manifest.json's own
    `tuning_summary.selected_hyperparameters` from the cycle that minted the
    current champion, so it already carries an "n_estimators" key (the
    champion's own final, already-scaled tree count). That key is popped out
    and returned as `best_n_estimators_median`, treated exactly like a
    search-path trial's early-stopped median: log_model.py's
    `_scale_n_estimators` still rescales it against *this* cycle's (larger)
    X_train/y_train before shipping, since the population genuinely grows
    cycle to cycle even when the hyperparameters themselves don't search.
    `best_params` below therefore excludes "n_estimators" — the same shape a
    search-path trial's own `best_params` always has (Optuna never searches
    tree count; early stopping resolves it) — so nothing here needs a special
    case to avoid the two colliding.

    Returns a result shaped identically to `run_tuning_step`'s search branch
    — same top-level keys, same nested `tuning_summary` shape
    (`mode="pinned"` instead of `"search"`, every trial-count/CV field zeroed
    or null) — so `run_model_logging_step` needs no changes at all: it
    consumes `best_params`/`best_n_estimators_median` either way and cannot
    tell the two paths apart.

    Still opens and tags a `tuning_study` MLflow run — log_model.py reopens
    this run's id to log the model onto it, on either path — but logs no
    nested trial runs and no CV metrics, since none were computed.
    """
    if "n_estimators" not in pinned_params:
        raise ValueError(
            "pinned_params must include 'n_estimators' — it is expected to be "
            "the tuned_hyperparameters model-version tag payload "
            "(training_manifest.json's tuning_summary.selected_hyperparameters), "
            "which always carries the champion's shipped tree count."
        )
    best_n_estimators_median = int(pinned_params["n_estimators"])
    best_params = {k: v for k, v in pinned_params.items() if k != "n_estimators"}

    tuning_cfg = cfg.tuning
    search_space_dict = {
        str(name): {"low": spec.low, "high": spec.high}
        for name, spec in tuning_cfg.search_space.items()
    }
    boundary_hits = boundary_hit_check(best_params, search_space_dict)
    hit_params = [name for name, hit in boundary_hits.items() if hit]
    if hit_params:
        logger.warning(
            "tuning_boundary_hit",
            hit_params=hit_params,
            selected_params=best_params,
            hint=(
                "pinned params (inherited unchanged from the champion) sit on "
                "a searched range's edge — investigate before shipping another "
                "cycle unchanged"
            ),
        )

    ensure_experiment_metadata(cfg)

    with mlflow.start_run(run_name="tuning_study") as parent_run:
        mlflow.set_tags(
            {
                "stage": "tuning",
                "git_sha": _git_sha(),
                "data_content_hash": features_sha256(),
            }
        )
        _log_dev_input(X_train, y_train, context="training")
        mlflow.log_params(
            {
                "mode": "pinned",
                "n_committed_features": len(committed_features),
                **{f"pinned_{k}": v for k, v in pinned_params.items()},
            }
        )
        mlflow.log_dict(boundary_hits, "tuning/boundary_hits.json")
        parent_run_id = parent_run.info.run_id

    logger.info(
        "tuning_step_done",
        mode="pinned",
        best_n_estimators_median=best_n_estimators_median,
        boundary_hits=boundary_hits,
    )

    return {
        "best_params": best_params,
        "best_n_estimators_median": best_n_estimators_median,
        "best_cv_pr_auc_mean": None,
        "boundary_hits": boundary_hits,
        "n_completed_trials": 0,
        "parent_run_id": parent_run_id,
        "committed_features": committed_features,
        "tuning_summary": {
            "mode": "pinned",
            "n_trials_requested": 0,
            "n_completed_trials": 0,
            "n_pruned_trials": 0,
            "n_failed_trials": 0,
            "n_trials_reused": 0,
            "n_trials_run_this_invocation": 0,
            "min_completed_trials": None,
            "trial_count_below_threshold": False,
            "selection_rule": None,
            "selected_trial_number": None,
            "selected_cv_pr_auc": None,
            "raw_best_trial_number": None,
            "raw_best_cv_pr_auc": None,
            "se": None,
            "band_floor": None,
            "boundary_hits": boundary_hits,
        },
    }


def run_tuning_step(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    committed_features: list[str],
    cfg: DictConfig,
    storage: optuna.storages.BaseStorage | None = None,
    pinned_params: dict[str, Any] | None = None,
    warm_start_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Optuna tuning of LightGBM on the frozen feature set.

    pinned_params, when given, skips the search entirely — see
    `_run_pinned_tuning_step`. Reserved for the two reserve-driven routine
    retrain cycles: a scheduled retrain
    is supposed to answer "did new data help?", and a fresh search changes
    the data *and* the hyperparameters at once, making a challenger's
    win/loss unattributable. `train/__main__.py`'s cold-start path never
    passes it, so v1 always takes the search branch below, unchanged.
    `warm_start_params` is ignored when `pinned_params` is given — pinning
    skips study construction altogether, so there is no study to seed.

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
    flags an untrustworthy result, it does not gate on it. Enforcement
    belongs at the point a tuned pipeline becomes a registered model —
    Phase 6's calibrate.py — not here. storage defaults to a Postgres-backed
    study (see _build_optuna_storage); pass an InMemoryStorage for tests.

    Idempotent against a study that already reached n_trials: only the trials
    still needed to reach n_trials are run, so re-running against a completed
    study reuses its existing trials instead of piling n_trials more on top.
    An *incomplete* same-named study is only reused when cfg.tuning.resume is
    true — otherwise it's discarded and a fresh study starts in its place
    (see _discard_incomplete_study_unless_resuming). tuning_summary's
    n_trials_reused/n_trials_run_this_invocation record which case applied.

    Also logs fANOVA hyperparameter importance — tuning/hyperparameter_importance.png
    and the underlying per-parameter values (tuning/hyperparameter_importance.json,
    persisted per CLAUDE.md's "log the array, not only the chart" rule) — computed
    once here, against the live study, rather than reconstructed later by every
    notebook that wants to render it. Never gates anything; a study too small for
    fANOVA to fit logs a warning and skips both artifacts rather than raising.

    Returns {"best_params", "best_n_estimators_median", "best_cv_pr_auc_mean",
    "boundary_hits", "n_completed_trials", "parent_run_id", "committed_features",
    "tuning_summary"}. tuning_summary carries the audit trail behind the
    selection_rule pick — trial counts, selected vs. raw-best trial number/score,
    the 1-SE standard error and band floor — so training_manifest.json and any
    notebook narrating the decision read the same numbers select_best_trial
    actually used, not a client-side reconstruction.
    """
    if pinned_params is not None:
        return _run_pinned_tuning_step(
            X_train, y_train, committed_features, cfg, pinned_params
        )

    setup = _build_optuna_study(
        X_train, y_train, committed_features, cfg, storage, warm_start_params
    )
    tuning_cfg = setup["tuning_cfg"]

    ensure_experiment_metadata(cfg)

    with mlflow.start_run(run_name="tuning_study") as parent_run:
        mlflow.set_tags(
            {
                "stage": "tuning",
                "git_sha": _git_sha(),
                "data_content_hash": features_sha256(),
            }
        )
        # Also covers log_model.py's "model" and calibrate.py's "calibrated_model"
        # LoggedModels for free — both reuse this run's run_id, and log_input
        # attaches to the run, not to any one LoggedModel logged onto it.
        _log_dev_input(X_train, y_train, context="training")
        mlflow.log_params(
            {
                "mode": "search",
                "optuna_study_name": setup["study_name"],
                "n_trials": int(tuning_cfg.n_trials),
                "resume": bool(tuning_cfg.resume),
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
                "timeout_seconds": setup["timeout_seconds"],
                "min_completed_trials": setup["min_completed_trials"],
            }
        )

        trial_result = _run_study_trials(setup, X_train, y_train)
        summary = _summarize_completed_trials(setup, trial_result)
        fig_hist = _plot_optimization_history(summary["trial_summaries"])
        importance = _compute_hyperparameter_importance(
            setup["study"], setup["random_state"]
        )
        fig_importance = (
            _plot_hyperparameter_importance(importance, len(summary["trial_summaries"]))
            if importance is not None
            else None
        )
        _log_tuning_artifacts(
            summary, trial_result, fig_hist, importance, fig_importance
        )

        parent_run_id = parent_run.info.run_id

    selected, diagnostics = summary["selected"], summary["diagnostics"]
    boundary_hits = summary["boundary_hits"]
    trial_summaries = summary["trial_summaries"]

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
            "mode": "search",
            "n_trials_requested": int(tuning_cfg.n_trials),
            "n_completed_trials": len(trial_summaries),
            "n_pruned_trials": trial_result["n_pruned_trials"],
            "n_failed_trials": trial_result["n_failed_trials"],
            "n_trials_reused": trial_result["n_trials_reused"],
            "n_trials_run_this_invocation": trial_result[
                "n_trials_run_this_invocation"
            ],
            "min_completed_trials": setup["min_completed_trials"],
            "trial_count_below_threshold": summary["trial_count_below_threshold"],
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

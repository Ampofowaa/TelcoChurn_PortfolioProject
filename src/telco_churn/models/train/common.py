"""Shared helpers for the model training pipeline.

Data loading, DVC/git metadata resolution, and the LightGBM knob builders
reused across every step (candidates.py, feature_audit.py, tuning.py,
log_model.py) so each step's fit is representative of the one that ships.
MLflow tracking-URI resolution lives in telco_churn.utils.mlflow — shared
outside the training package (models/calibrate.py is the second consumer).
"""

from __future__ import annotations

import subprocess
import time
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from mlflow.data.pandas_dataset import PandasDataset
from mlflow.data.pandas_dataset import from_pandas as mlflow_from_pandas
from numpy.typing import NDArray
from omegaconf import DictConfig
from sklearn.base import clone
from sklearn.metrics import PrecisionRecallDisplay, average_precision_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sqlalchemy.engine import Engine

from telco_churn.data.split import partition
from telco_churn.features.accessor import features_path, load_features
from telco_churn.features.build import (
    FEATURE_SCHEMA,
    TARGET_COL,
    build_feature_df,
    load_customer_features,
)
from telco_churn.utils.logging import get_logger

__all__ = [
    "COMMITTED_MODEL_FAMILY",
    "COMMITTED_MODEL_FAMILY_DECISION_RUN_ID",
    "cv_score_candidate",
    "lgbm_default_params",
    "load_training_pool_bundle",
    "logreg_default_params",
]

logger = get_logger(__name__)

# The model family Steps 3-5 build against. Frozen by the Step 1/2 comparison
# (notebooks/03a-model-selection.ipynb, candidates.py::run_candidate_step +
# comparison.py::run_comparison_step), not recomputed per training cycle — see
# ANALYSIS.md §4a. LightGBM wins on the evidence (paired-bootstrap Δ = 0.007,
# CI [0.002, 0.012], p = 0.0020, run `c405f6fe31454c3a9899423680644411`) and on
# build-specific rationale (SHAP speed, calibration continuity, training
# speed for Optuna/weekly retrain). Edit this constant — never hand-fit a
# LogReg pipeline into Steps 3-5 — if a re-run of 03a's on-demand review
# concludes a different family should ship; same "explicit code changes,
# never accidental side-effects" discipline features/schema.py's
# COMMITTED_FEATURES documents.
COMMITTED_MODEL_FAMILY: str = "lightgbm"

# The model_comparison MLflow run (telco-churn-training experiment) whose
# logged Δ/CI back the decision above — log_model.py stamps this into
# training_manifest.json's model_family_committed section so the frozen
# decision is a resolvable reference, not a number retyped from ANALYSIS.md.
COMMITTED_MODEL_FAMILY_DECISION_RUN_ID: str = "c405f6fe31454c3a9899423680644411"

_FEATURE_COLS: list[str] = (
    list(FEATURE_SCHEMA.binary)
    + list(FEATURE_SCHEMA.multi_cat)
    + list(FEATURE_SCHEMA.numeric)
)


def _git_sha() -> str:
    """Return the current git HEAD SHA, or 'unknown' if git is unavailable.

    Logged as a warning rather than swallowed silently: this value feeds
    training_manifest.json's engineering audit trail, so a genuinely broken git
    invocation (not just running outside a checkout) should be discoverable in
    pipeline logs, not indistinguishable from the routine 'unknown' fallback.
    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception as e:
        logger.warning("git_sha_unavailable", error=str(e), exc_info=True)
        return "unknown"


def _load_dev_features() -> tuple[pd.DataFrame, pd.Series]:
    """Load the processed feature frame and return only its dev-partition rows.

    The v1/cold-start path's only loader — train/__main__.py calls this
    directly and unconditionally. load_training_pool_bundle() (below) is the
    retrain-cycle equivalent, sourced from training_pool rather than the
    static processed-features parquet; it never replaces this function, only
    supplements it for a caller that needs dev ∪ matured reserve instead of
    dev alone.
    """
    df = load_features()
    dev_df, _test_df = partition(df)
    return dev_df[_FEATURE_COLS], dev_df[TARGET_COL]


def load_training_pool_bundle(
    engine: Engine, reserve_months: list[int] | None = None
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Query training_pool for the fold-forward-eligible population; return the
    4-field bundle a retrain cycle's train/calibrate/threshold steps all draw
    from.

    reserve_months is the same fold-forward filter features/build.py::
    build_feature_query already applies — None selects only the original
    CSV-seeded population (reserve_month IS NULL), the same rows
    _load_dev_features() reads for cold start; a non-empty list additionally
    folds in those matured reserve cohorts.

    Intended to be called exactly once per cycle, at flow start, by
    training_cycle.py (Phase 10b) — nothing in this repo calls it yet, the
    same "no live caller until its orchestrator lands" state
    data/training_pool.py's write-path-2 functions are already in. The
    returned (X_train, y_train, customer_ids) feed run_feature_audit_step/
    run_tuning_step/run_model_logging_step directly; all four fields feed
    calibrate.py's/threshold.py's own override parameters — one query, never
    a second independent assembly, so no stage of a cycle can silently see a
    different training_pool snapshot than its siblings.

    raw_partition is the full customer_features frame (customerid + every
    FEATURE_SCHEMA column + churn) — the same shape
    models/dev_features.py::load_dev_partition() returns for the cold-start
    path. threshold.py's V1/V2/V2b fairness diagnostics need it for
    segment/protected columns (gender, seniorcitizen, ...); X_train/y_train
    alone don't carry customerid or churn.
    """
    raw_partition = build_feature_df(load_customer_features(engine, reserve_months))
    return (
        raw_partition[_FEATURE_COLS],
        raw_partition[TARGET_COL],
        raw_partition["customerid"].reset_index(drop=True),
        raw_partition,
    )


def _build_dev_dataset(X_train: pd.DataFrame, y_train: pd.Series) -> PandasDataset:
    """Build the training-population MLflow dataset object, not yet logged to any run.

    X_train/y_train is dev-only at cold start, dev ∪ matured reserve on a
    retrain cycle — named to match
    load_training_pool_bundle()'s own output fields so the value passes
    through unchanged in both name and meaning from assembly to logging, on
    either path. source resolves through features/accessor.py's canonical
    path, the same single-owner rule candidates.py already follows, so this
    can't go stale at Phase 8's CSV->parquet rename. The dataset's digest is
    content-based (from_pandas), so this is purely lineage metadata — it
    records what data a run touched, never something a computed number
    depends on.
    """
    dataset_df = X_train.copy()
    dataset_df[TARGET_COL] = y_train.values
    return mlflow_from_pandas(
        dataset_df,
        source=str(features_path()),
        name="telco_churn_dev",
        targets=TARGET_COL,
    )


def _log_dev_input(X_train: pd.DataFrame, y_train: pd.Series, context: str) -> None:
    """Log the training-population dataset as an MLflow run input — call from inside an active run.

    Shared by comparison.py, feature_audit.py, and tuning.py (candidates.py
    calls _build_dev_dataset directly instead, since each of its three
    candidate runs needs its own log_input call against one shared dataset
    object, not a fresh build-and-log per run).
    """
    mlflow.log_input(_build_dev_dataset(X_train, y_train), context=context)


def _plot_bootstrap_delta(
    bootstrap_deltas: NDArray[np.float64],
    delta_obs: float,
    delta_threshold: float,
    title: str,
    xlabel: str,
) -> Figure:
    """Histogram of the paired-bootstrap Δ distribution against the null and the decision threshold.

    Shared by comparison.py's model-family decision and feature_selection.py's
    full-vs-reduced feature-set decision — both plot the identical shape
    (bootstrap_deltas histogram, Δ_obs/0/Δ* reference lines) against a
    different Δ definition and title.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(bootstrap_deltas, bins=50, edgecolor="none", alpha=0.75)
    ax.axvline(delta_obs, color="C1", linestyle="--", label=f"Δ_obs = {delta_obs:.3f}")
    ax.axvline(0.0, color="black", linestyle=":", label="Δ = 0 (null)")
    ax.axvline(
        delta_threshold,
        color="C2",
        linestyle="--",
        label=f"Δ* = {delta_threshold}",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Bootstrap resamples")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def _plot_pr_curves(
    candidates: dict[str, dict[str, Any]],
    title: str,
) -> Figure:
    """Pooled OOF precision-recall curve per candidate — one line per label.

    Shared by comparison.py's model-family decision and feature_selection.py's
    full-vs-reduced feature-set decision — both score candidates via
    cv_score_candidate/run_selection_cv, which return the identical
    {"oof_true": [...], "oof_proba": [...]} shape, so this plots any number of
    candidates without knowing which produced them.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, result in candidates.items():
        PrecisionRecallDisplay.from_predictions(
            result["oof_true"], result["oof_proba"], name=label, ax=ax
        )
    ax.set_title(title)
    fig.tight_layout()
    return fig


def _plot_shap_audit(
    shap_audit: pd.DataFrame,
    high_shap_dropouts: list[str],
    title: str,
    flag_column: str = "committed",
    legend_labels: tuple[str, str] = ("Committed", "Not committed"),
) -> Figure:
    """Horizontal bar chart of mean(|SHAP|) per feature — colour = flag_column, dropouts flagged.

    Shared by feature_audit.py's per-cycle diagnostic and feature_selection.py's
    ablation review (twice, for the latter — once coloured by committed, once by
    §3's permutation-importance survived, via flag_column) — all call
    features.select.compute_shap_audit / flag_high_shap_dropouts against their
    own committed_features and plot the identical shape. shap_audit is
    compute_shap_audit's output (feature, mean_abs_shap, committed), optionally
    with another boolean column merged in for flag_column to read instead;
    high_shap_dropouts is flag_high_shap_dropouts' output — a dropped feature
    that outranks a committed one gets its own colour rather than blending into
    the "not [flag_labels[0]]" red, so the exact check flag_high_shap_dropouts
    computes is visible on the chart, not just inferable by counting bar
    positions against the top-N cutoff.
    """
    ordered = shap_audit.sort_values("mean_abs_shap", ascending=False)
    dropout_set = set(high_shap_dropouts)
    colors = [
        "goldenrod" if feature in dropout_set else ("seagreen" if flag else "indianred")
        for feature, flag in zip(ordered["feature"], ordered[flag_column], strict=True)
    ]

    fig, ax = plt.subplots(figsize=(8, max(4.0, 0.3 * len(ordered))))
    ax.barh(ordered["feature"], ordered["mean_abs_shap"], color=colors)
    ax.set_xlabel("Mean |SHAP|")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.legend(
        handles=[
            Patch(color="seagreen", label=legend_labels[0]),
            Patch(color="indianred", label=legend_labels[1]),
            Patch(color="goldenrod", label="High-SHAP dropout"),
        ],
        loc="lower right",
    )
    fig.tight_layout()
    return fig


def _plot_permutation_importance(
    importance_table: pd.DataFrame,
    title: str,
) -> Figure:
    """Horizontal bar chart of real permutation importance vs. the decoy floor — colour = survived.

    Shared by feature_audit.py's per-cycle diagnostic (features.select.mint_committed_list's
    single all-dev fit) and feature_selection.py's ablation review
    (features.select.run_selection_cv's cross-fold-averaged fit) — both produce a table
    shaped feature/real_importance/importance_floor/survived and plot the identical shape;
    only the underlying computation (and hence what "survived" means — one fit's floor
    check vs. stability >= threshold across folds) differs between callers.
    """
    ordered = importance_table.sort_values("real_importance", ascending=False)
    colors = ordered["survived"].map({True: "seagreen", False: "indianred"})

    fig, ax = plt.subplots(figsize=(8, max(4.0, 0.3 * len(ordered))))
    ax.barh(ordered["feature"], ordered["real_importance"], color=colors)
    ax.axvline(
        ordered["importance_floor"].iloc[0],
        color="black",
        linestyle="--",
        label="decoy floor",
    )
    ax.set_xlabel("Real permutation importance (mean PR-AUC drop)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def _plot_stability(
    importance_table: pd.DataFrame,
    title: str,
) -> Figure:
    """Horizontal bar chart of per-fold selection stability — colour = survived (stability >= threshold).

    feature_selection.py-only: per-fold survival only exists for the cross-fold ablation
    (features.select.run_selection_cv); feature_audit.py's single all-dev fit
    (mint_committed_list) has no fold dimension to plot here. Answers a different question
    from the permutation-importance chart above: not "how big is this feature's importance
    on one fit," but "does the selector keep choosing this feature fold to fold, or does the
    survivor list bounce around."
    """
    ordered = importance_table.sort_values("stability")
    colors = ordered["survived"].map({True: "seagreen", False: "indianred"})

    fig, ax = plt.subplots(figsize=(8, max(4.0, 0.3 * len(ordered))))
    ax.barh(ordered["feature"], ordered["stability"], color=colors)
    ax.set_xlabel("Fraction of folds survived")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    return fig


def _fit_and_score_fold(
    estimator: Any,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[float, float, float, np.ndarray[Any, Any]]:
    """Clone, fit, and score one CV fold. Returns (pr_auc, train_time_s, predict_time_s, proba).

    Clones estimator rather than mutating the shared instance passed to
    cv_score_candidate — required once folds run concurrently (joblib.Parallel's
    default process-based backend already copies it per worker, but cloning
    explicitly keeps this correct even if the backend ever changes to threads).
    """
    est = clone(estimator)

    t0 = time.perf_counter()
    est.fit(X_tr, y_tr)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    proba = est.predict_proba(X_val)[:, 1]
    predict_time = time.perf_counter() - t0

    return float(average_precision_score(y_val, proba)), train_time, predict_time, proba


def cv_score_candidate(
    estimator: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv: RepeatedStratifiedKFold,
    n_jobs: int = 1,
) -> dict[str, Any]:
    """Run RepeatedStratifiedKFold CV and return per-fold PR-AUC scores and timing.

    estimator — any sklearn-compatible classifier or Pipeline with fit / predict_proba.
    cv        — shared RSKF instance; pass the same object to every candidate so all
                candidates train and validate on identical fold indices.
    n_jobs    — folds run independently (each clones estimator and fits on its own
                data), so joblib.Parallel(n_jobs=...) changes only wall-clock time,
                never the returned scores — folds carry no shared state and
                Parallel preserves input order in its output regardless of which
                worker finishes first. LightGBM's own n_jobs (training.fixed.n_jobs,
                pinned to 1 for determinism) is a separate, inner concern: each
                fold's fit stays single-threaded even when folds themselves run
                concurrently across processes.

    Returns a dict with keys:
        pr_auc_mean    — mean CV PR-AUC across all folds
        pr_auc_std     — sample std (ddof=1) of fold scores
        pr_auc_scores  — raw per-fold list for paired bootstrap (Step 2)
        oof_proba      — OOF positive-class probabilities aligned to X row order,
                         averaged across repeats (each row scored n_repeats times)
        oof_true       — true labels aligned to X row order (same as y.tolist())
        train_time_s   — mean per-fold fit time (seconds)
        predict_time_s — mean per-fold predict_proba time (seconds)
    """
    fold_indices = list(cv.split(X, y))
    fold_results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_and_score_fold)(
            estimator,
            X.iloc[train_idx],
            y.iloc[train_idx],
            X.iloc[val_idx],
            y.iloc[val_idx],
        )
        for train_idx, val_idx in fold_indices
    )

    scores = [r[0] for r in fold_results]
    train_times = [r[1] for r in fold_results]
    predict_times = [r[2] for r in fold_results]

    oof_proba_sum = np.zeros(len(X))
    oof_counts = np.zeros(len(X), dtype=int)
    for (_, val_idx), (_, _, _, proba) in zip(fold_indices, fold_results, strict=True):
        oof_proba_sum[val_idx] += proba
        oof_counts[val_idx] += 1

    return {
        "pr_auc_mean": float(np.mean(scores)),
        "pr_auc_std": float(np.std(scores, ddof=1)),
        "pr_auc_scores": scores,
        "oof_proba": (oof_proba_sum / oof_counts).tolist(),
        "oof_true": y.tolist(),
        "train_time_s": float(np.mean(train_times)),
        "predict_time_s": float(np.mean(predict_times)),
    }


def _lgbm_fixed_knobs(cfg: DictConfig, random_state: int) -> dict[str, Any]:
    """Determinism + imbalance knobs applied to every LightGBM fit (Steps 1, 3, 4, 5).

    Shared unconditionally across every step's fit so each LightGBM model in the
    pipeline is representative of the one that ships.
    """
    cc = cfg.training_setup
    fixed = cfg.training.fixed
    return {
        "class_weight": str(cc.class_weight),
        "subsample_freq": int(fixed.subsample_freq),
        "deterministic": bool(fixed.deterministic),
        "force_row_wise": bool(fixed.force_row_wise),
        "n_jobs": int(fixed.n_jobs),
        "verbose": int(fixed.verbose),
        "random_state": random_state,
    }


def lgbm_default_params(cfg: DictConfig, random_state: int) -> dict[str, Any]:
    """Default-config (untuned) LightGBM constructor kwargs shared by Step 1 and Step 3.

    Public (not underscore-prefixed): notebooks re-fitting the Step 1 default-config
    candidate for a diagnostic (e.g. 03a's 2c bias/variance check, 03c's
    full/reduced/tuned progression) must build the exact same kwargs `lgbm_default`
    trained with, not a hand-copied reconstruction that can silently drift from it.
    """
    return {
        "n_estimators": int(cfg.training.candidate.n_estimators),
        "num_leaves": int(cfg.training.candidate.num_leaves),
        "min_child_samples": int(cfg.training.candidate.min_child_samples),
        **_lgbm_fixed_knobs(cfg, random_state),
    }


def logreg_default_params(cfg: DictConfig, random_state: int) -> dict[str, Any]:
    """LogisticRegressionCV constructor kwargs shared by Step 1's logreg_cv candidate
    and 03a-model-selection.ipynb's odds-ratio exhibit.

    Public (not underscore-prefixed): the notebook refits this exact candidate to
    read off coefficients and must build the same kwargs `logreg_cv` trained with,
    not a hand-copied reconstruction that can silently drift from it.

    l1_ratios (not penalty=): sklearn 1.8 deprecated the penalty string in favor of
    l1_ratios (0=L2, 1=L1), removed entirely in 1.10 — l1_ratios=(0,) is identical to
    the old penalty='l2'. use_legacy_attributes=False opts fitted attributes (coef_,
    C_, ...) into the post-1.10 shape now, since nothing reads the legacy shape yet —
    the odds-ratio exhibit above doesn't exist yet either, so there's nothing to migrate.
    """
    logreg = cfg.logreg
    return {
        "Cs": int(logreg.Cs),
        "cv": int(logreg.cv_folds),
        "scoring": "average_precision",
        "solver": str(logreg.solver),
        "l1_ratios": (float(logreg.l1_ratio),),
        "max_iter": int(logreg.max_iter),
        "class_weight": str(cfg.training_setup.class_weight),
        "random_state": random_state,
        "n_jobs": 1,
        "use_legacy_attributes": False,
    }

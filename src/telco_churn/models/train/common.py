"""Shared helpers for the model training pipeline.

Data loading, MLflow/DVC/git metadata resolution, and the LightGBM knob builders
reused across every step (candidates.py, feature_freeze.py, tuning.py,
log_model.py) so each step's fit is representative of the one that ships.
"""

from __future__ import annotations

import subprocess
import time
from typing import Any

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from omegaconf import DictConfig
from sklearn.base import clone
from sklearn.metrics import average_precision_score
from sklearn.model_selection import RepeatedStratifiedKFold

from telco_churn.data.split import partition
from telco_churn.features.build import FEATURE_SCHEMA, TARGET_COL
from telco_churn.features.schema import FeatureOutputSchema
from telco_churn.utils.logging import get_logger
from telco_churn.utils.paths import get_project_root

__all__ = ["cv_score_candidate", "lgbm_default_params", "logreg_default_params"]

logger = get_logger(__name__)

_FEATURE_COLS: list[str] = (
    list(FEATURE_SCHEMA.binary)
    + list(FEATURE_SCHEMA.multi_cat)
    + list(FEATURE_SCHEMA.numeric)
)


def _load_processed(cfg: DictConfig) -> pd.DataFrame:
    """Load the processed feature CSV produced by features/build.py.

    Validated against FeatureOutputSchema before use: the DVC
    validate->features->train DAG doesn't exist yet, so a standalone
    `python -m telco_churn.models.train` run must not silently fit on a
    stale or schema-drifted processed file.
    """
    path = get_project_root() / cfg.paths.processed_data / "telco_churn_processed.csv"
    df = pd.read_csv(path)
    FeatureOutputSchema.validate(df)
    return df


def _resolve_tracking_uri(uri: str) -> str:
    """Resolve relative MLflow tracking URIs to absolute project-rooted file:// URIs.

    Any URI with an explicit scheme (http(s)://, sqlite://, postgresql://, ...) is
    returned unchanged — only a bare relative path like 'mlruns' needs anchoring to
    get_project_root() so the tracking store is always written to the same
    location regardless of the shell's working directory.

    Returns a file:// URI, not a bare path — on Windows, str(path) yields
    'C:\\...\\mlruns', and MLflow's store registry reads urlparse's scheme off
    that string, which is 'c' for a drive letter, not a recognized backend. This
    is the default branch for any fresh clone or CI runner (no .env, so
    tracking_uri resolves to the bare 'mlruns' default in configs/config.yaml),
    so it isn't a latent edge case.
    """
    if "://" in uri:
        return uri
    return (get_project_root() / uri).as_uri()


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


def _dvc_hash(cfg: DictConfig) -> str:
    """Return the DVC content hash of the processed CSV, or 'unknown' if not tracked.

    A missing .dvc file is the expected, common case before Phase 8 wires up DVC
    tracking — logged at debug only, not a warning. Any other failure (corrupted
    YAML, unexpected file structure, a cfg missing paths.processed_data) is
    logged as a warning so it doesn't masquerade as the same routine
    'not tracked yet' fallback. The path construction stays inside the try along
    with the file read — cfg.paths.processed_data can itself raise (e.g. a
    minimal test cfg with no paths key), and that failure must be caught here
    too, not just the file-open step.
    """
    try:
        dvc_file = (
            get_project_root()
            / cfg.paths.processed_data
            / "telco_churn_processed.csv.dvc"
        )
        with open(dvc_file) as f:
            return str(yaml.safe_load(f)["outs"][0]["md5"])
    except FileNotFoundError as e:
        logger.debug("dvc_hash_not_tracked", error=str(e))
        return "unknown"
    except Exception as e:
        logger.warning("dvc_hash_unexpected_error", error=str(e), exc_info=True)
        return "unknown"


def _load_dev_features(cfg: DictConfig) -> tuple[pd.DataFrame, pd.Series]:
    """Load the processed feature frame and return only its dev-partition rows."""
    df = _load_processed(cfg)
    dev_df, _test_df = partition(df)
    return dev_df[_FEATURE_COLS], dev_df[TARGET_COL]


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

"""Non-gating diagnostic helpers for model selection.

Pure, side-effect-free functions. fixed_recall_profile, segment_oof_errors, and
segment_bootstrap_delta are reused by models/train/comparison.py; generalization_gap
and learning_curve_points are called directly from notebooks/03a-model-selection.ipynb.
None of these decide the model family — that is the paired-bootstrap PR-AUC gate in
models/train/comparison.py's bootstrap_comparison. They flag robustness/fairness
concerns and characterize bias/variance for the analyst.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

__all__ = [
    "fixed_recall_profile",
    "generalization_gap",
    "learning_curve_points",
    "segment_bootstrap_delta",
    "segment_oof_errors",
]

_MIN_SEGMENT_SIZE = 10


def fixed_recall_profile(
    y_true: Sequence[int],
    proba: Sequence[float],
    recall_targets: Sequence[float],
) -> list[dict[str, float]]:
    """Return precision/F1/threshold at each fixed-recall operating point.

    For each target, selects the highest-precision point on the PR curve whose
    recall is >= target — the least-permissive threshold that still clears the
    target. A target above the curve's max achievable recall (e.g. > 1.0) has no
    qualifying point and returns NaN for precision/recall_achieved/f1/threshold.
    """
    precision, recall, thresholds = precision_recall_curve(
        np.asarray(y_true), np.asarray(proba)
    )
    # precision_recall_curve appends a final (precision=1, recall=0) point with no
    # corresponding threshold — drop it so recall/precision/thresholds stay aligned.
    precision, recall = precision[:-1], recall[:-1]

    rows: list[dict[str, float]] = []
    for target in recall_targets:
        qualifying = np.where(recall >= target)[0]
        if qualifying.size == 0:
            rows.append(
                {
                    "recall_target": float(target),
                    "precision": float("nan"),
                    "recall_achieved": float("nan"),
                    "f1": float("nan"),
                    "threshold": float("nan"),
                }
            )
            continue
        best_idx = qualifying[np.argmax(precision[qualifying])]
        p, r = float(precision[best_idx]), float(recall[best_idx])
        f1 = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
        rows.append(
            {
                "recall_target": float(target),
                "precision": p,
                "recall_achieved": r,
                "f1": float(f1),
                "threshold": float(thresholds[best_idx]),
            }
        )
    return rows


def segment_oof_errors(
    y_true: Sequence[int],
    proba: Sequence[float],
    group: pd.Series,
) -> list[dict[str, object]]:
    """Return per-segment n / churn rate / PR-AUC for a disaggregation column.

    group is a categorical Series aligned to y_true/proba (e.g. contract_type,
    tenure_cohort, gender) — one row per segment value. Segments smaller than
    _MIN_SEGMENT_SIZE are dropped: PR-AUC on a handful of rows is noise, not signal.
    A single-class segment falls back to its churn rate (PR-AUC is undefined with
    only one class present) so callers never see a NaN.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    proba_arr = np.asarray(proba, dtype=float)

    rows: list[dict[str, object]] = []
    for value in sorted(group.unique(), key=str):
        mask = (group == value).to_numpy()
        n = int(mask.sum())
        if n < _MIN_SEGMENT_SIZE:
            continue
        y_seg, p_seg = y_true_arr[mask], proba_arr[mask]
        churn_rate = float(y_seg.mean())
        pr_auc = (
            float(average_precision_score(y_seg, p_seg))
            if len(np.unique(y_seg)) > 1
            else churn_rate
        )
        rows.append(
            {
                "segment": group.name,
                "value": value,
                "n": n,
                "churn_rate": churn_rate,
                "pr_auc": pr_auc,
            }
        )
    return rows


def segment_bootstrap_delta(
    y_true: Sequence[int],
    proba_lgbm: Sequence[float],
    proba_logreg: Sequence[float],
    group: pd.Series,
    n_bootstrap: int,
    random_state: int,
) -> list[dict[str, object]]:
    """Paired row-level bootstrap CI on Δ = PR-AUC(LGBM) − PR-AUC(LogReg), per segment.

    Mirrors bootstrap_comparison's pairing logic (models/train/comparison.py) at the
    row level instead of the fold level: each resample draws row indices once per
    segment and scores both candidates on the identical resampled rows, so shared
    segment-level noise cancels in the delta rather than surviving into two separate
    per-candidate CIs compared informally. Segments smaller than _MIN_SEGMENT_SIZE,
    or a resample with only one class present, are skipped — delta is undefined
    either way. Flag-only: never decides the model family (bootstrap_comparison's
    fold-level test alone does that).
    """
    y_arr = np.asarray(y_true, dtype=float)
    lgbm_arr = np.asarray(proba_lgbm, dtype=float)
    logreg_arr = np.asarray(proba_logreg, dtype=float)
    rng = np.random.default_rng(random_state)

    rows: list[dict[str, object]] = []
    for value in sorted(group.unique(), key=str):
        mask = (group == value).to_numpy()
        n = int(mask.sum())
        if n < _MIN_SEGMENT_SIZE:
            continue
        y_seg = y_arr[mask]
        if len(np.unique(y_seg)) < 2:
            continue
        lgbm_seg, logreg_seg = lgbm_arr[mask], logreg_arr[mask]
        delta_obs = float(
            average_precision_score(y_seg, lgbm_seg)
            - average_precision_score(y_seg, logreg_seg)
        )

        deltas: list[float] = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, n)
            y_b = y_seg[idx]
            if len(np.unique(y_b)) < 2:
                continue
            deltas.append(
                float(
                    average_precision_score(y_b, lgbm_seg[idx])
                    - average_precision_score(y_b, logreg_seg[idx])
                )
            )

        rows.append(
            {
                "segment": group.name,
                "value": value,
                "n": n,
                "delta_obs": delta_obs,
                "delta_ci_lower": float(np.percentile(deltas, 2.5)),
                "delta_ci_upper": float(np.percentile(deltas, 97.5)),
            }
        )
    return rows


def generalization_gap(
    train_scores: Sequence[float],
    cv_scores: Sequence[float],
) -> dict[str, float]:
    """Per-fold train-vs-CV PR-AUC gap — the generalization (overfitting) read.

    train_scores/cv_scores are paired per fold: the same fold's in-sample (training
    partition) PR-AUC and held-out (validation partition) PR-AUC. A wide mean gap
    signals overfitting (variance); a narrow gap says the model generalizes about as
    well out-of-fold as in-sample.
    """
    train_arr = np.asarray(train_scores, dtype=float)
    cv_arr = np.asarray(cv_scores, dtype=float)
    gaps = train_arr - cv_arr
    return {
        "train_pr_auc_mean": float(train_arr.mean()),
        "cv_pr_auc_mean": float(cv_arr.mean()),
        "gap_mean": float(gaps.mean()),
        "gap_std": float(gaps.std(ddof=1)) if len(gaps) > 1 else 0.0,
    }


def learning_curve_points(
    train_sizes: Sequence[int],
    train_scores: Sequence[Sequence[float]],
    cv_scores: Sequence[Sequence[float]],
) -> list[dict[str, float]]:
    """Summarize per-training-size train/CV score arrays for a data-sufficiency learning curve.

    train_scores/cv_scores are 2-D — one row of repeat/fold scores per training size,
    the shape sklearn.model_selection.learning_curve returns.
    """
    rows: list[dict[str, float]] = []
    for size, t_scores, c_scores in zip(
        train_sizes, train_scores, cv_scores, strict=True
    ):
        t_arr = np.asarray(t_scores, dtype=float)
        c_arr = np.asarray(c_scores, dtype=float)
        rows.append(
            {
                "train_size": float(size),
                "train_pr_auc_mean": float(t_arr.mean()),
                "train_pr_auc_std": float(t_arr.std(ddof=1)) if len(t_arr) > 1 else 0.0,
                "cv_pr_auc_mean": float(c_arr.mean()),
                "cv_pr_auc_std": float(c_arr.std(ddof=1)) if len(c_arr) > 1 else 0.0,
            }
        )
    return rows

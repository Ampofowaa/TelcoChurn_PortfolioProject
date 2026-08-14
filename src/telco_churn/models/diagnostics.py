"""Non-gating diagnostic helpers for model selection and evaluation.

Pure, side-effect-free — no I/O, no config, no knowledge of which columns are
segment axes. fixed_recall_profile, segment_oof_errors, and
segment_bootstrap_delta are reused by models/train/comparison.py;
generalization_gap and learning_curve_points are called directly from
notebooks/03a-model-selection.ipynb. None of these decide the model family —
that's the paired-bootstrap PR-AUC gate in comparison.py's
bootstrap_comparison. They flag robustness/fairness concerns and characterize
bias/variance for the analyst.

segment_bootstrap_ci and segment_decision_rates are consumed by
models/evaluate.py (sealed-test slices) and models/threshold.py's dev-OOF
screen (dev-OOF slices), both of which use build_segment_lookup below to join
the segment-axis columns onto their own probability vectors before calling
these. Their outputs feed the fairness/robustness veto-only guardrails, but
the veto decision itself lives in models/gate.py, never here.

sliced_ranking_metrics/flag_segment_collapse (segment collapse),
sliced_decision_rates + equal_opportunity_difference_by_axis/
demographic_parity_difference_by_axis (fairness disparity), and
sliced_calibration/flag_calibration_collapse (per-group calibration) are the
per-axis wrappers both callers invoke — threshold.py's dev-OOF screen for its
own diagnostic pass, evaluate.py for the sealed-test reporting slices (which
have no flagging of their own; they are reported, not screened).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import cast

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.metrics import average_precision_score, precision_recall_curve

from telco_churn.features.preprocessing import TENURE_COHORT_EDGES, TENURE_COHORT_LABELS
from telco_churn.models.calibration_metrics import (
    calibration_slope,
    expected_calibration_error,
)

__all__ = [
    "FAIRNESS_AXES",
    "ROBUSTNESS_AXES",
    "build_segment_lookup",
    "demographic_parity_difference_by_axis",
    "equal_opportunity_difference_by_axis",
    "fixed_recall_profile",
    "flag_calibration_collapse",
    "flag_segment_collapse",
    "generalization_gap",
    "learning_curve_points",
    "segment_bootstrap_ci",
    "segment_bootstrap_delta",
    "sliced_calibration",
    "sliced_decision_rates",
    "sliced_ranking_metrics",
    "segment_decision_rates",
    "segment_oof_errors",
]

_MIN_SEGMENT_SIZE = 10

# Fairness/robustness slicing axes feeding the segment-collapse, fairness-
# disparity, and per-group-calibration veto guardrails. Shared by
# threshold.py's dev-OOF screen and evaluate.py's sealed-test reporting
# slices — mirrors models.train.comparison's
# _ROBUSTNESS_SEGMENTS/_FAIRNESS_SEGMENTS (a private pair scoped to model
# family selection, used on data neither of these modules touches),
# duplicated rather than imported for that reason.
ROBUSTNESS_AXES: tuple[str, ...] = (
    "contract_type",
    "tenure_cohort",
    "internetservice",
)
FAIRNESS_AXES: tuple[str, ...] = (
    "gender",
    "seniorcitizen",
    "has_partner",
    "dependents",
)


def build_segment_lookup(df: pd.DataFrame) -> dict[str, pd.Series]:
    """{axis_name: Series} for every robustness/fairness axis, aligned to df's index.

    tenure_cohort is derived via pd.cut over the fixed TENURE_COHORT_EDGES
    (same edges models.train.comparison uses) rather than read as a raw
    column — it doesn't exist as one. The other six axes are raw columns.
    Shared by evaluate.py (test-partition segments) and threshold.py's
    dev-OOF screen (dev-partition segments) so both surfaces are computed by
    identical code.
    """
    tenure_cohort = (
        pd.cut(
            df["tenure"],
            bins=TENURE_COHORT_EDGES,
            labels=TENURE_COHORT_LABELS,
            include_lowest=True,
        )
        .astype(str)
        .rename("tenure_cohort")
    )
    return {
        "contract_type": df["contract_type"],
        "tenure_cohort": tenure_cohort,
        "internetservice": df["internetservice"],
        "gender": df["gender"],
        "seniorcitizen": df["seniorcitizen"],
        "has_partner": df["has_partner"],
        "dependents": df["dependents"],
    }


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


def _segment_bootstrap(
    y_true: np.ndarray,
    group: pd.Series,
    score_fn: Callable[..., float],
    arrays: tuple[np.ndarray, ...],
    n_bootstrap: int,
    random_state: int,
) -> list[dict[str, object]]:
    """Shared per-segment percentile-bootstrap scaffolding.

    `score_fn(y_seg, *arrays_seg) -> float` computes the observed statistic
    from row-aligned, already segment-masked arrays — a bare PR-AUC for
    segment_bootstrap_ci, a Δ between two candidates for
    segment_bootstrap_delta. Each resample draws row indices once per segment
    and applies score_fn to the identical resampled rows across every array,
    so a paired caller's shared segment-level noise cancels in its delta
    instead of surviving into two separately-resampled CIs. Segments smaller
    than _MIN_SEGMENT_SIZE, or a resample with only one class present, are
    skipped — the statistic is undefined either way.
    """
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, object]] = []
    for value in sorted(group.unique(), key=str):
        mask = (group == value).to_numpy()
        n = int(mask.sum())
        if n < _MIN_SEGMENT_SIZE:
            continue
        y_seg = y_true[mask]
        if len(np.unique(y_seg)) < 2:
            continue
        seg_arrays = tuple(a[mask] for a in arrays)
        obs = float(score_fn(y_seg, *seg_arrays))

        stats: list[float] = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, n)
            y_b = y_seg[idx]
            if len(np.unique(y_b)) < 2:
                continue
            stats.append(float(score_fn(y_b, *(a[idx] for a in seg_arrays))))

        rows.append(
            {
                "segment": group.name,
                "value": value,
                "n": n,
                "obs": obs,
                "ci_lower": float(np.percentile(stats, 2.5)),
                "ci_upper": float(np.percentile(stats, 97.5)),
            }
        )
    return rows


def segment_bootstrap_ci(
    y_true: Sequence[int],
    proba: Sequence[float],
    group: pd.Series,
    n_bootstrap: int,
    random_state: int,
) -> list[dict[str, object]]:
    """Single-model per-segment PR-AUC with a percentile bootstrap CI.

    Feeds the segment-collapse veto (a segment's PR-AUC CI lower bound
    falling below that segment's own churn-rate floor) and the sealed-test
    per-slice ranking diagnostic. Unlike segment_bootstrap_delta, there's no
    second candidate to pair against — this scores one model's probabilities
    in isolation and never decides the model family.
    """
    y_arr = np.asarray(y_true, dtype=float)
    proba_arr = np.asarray(proba, dtype=float)

    def _pr_auc(y: np.ndarray, p: np.ndarray) -> float:
        return float(average_precision_score(y, p))

    raw = _segment_bootstrap(
        y_arr, group, _pr_auc, (proba_arr,), n_bootstrap, random_state
    )
    return [
        {
            "segment": row["segment"],
            "value": row["value"],
            "n": row["n"],
            "pr_auc_obs": row["obs"],
            "pr_auc_ci_lower": row["ci_lower"],
            "pr_auc_ci_upper": row["ci_upper"],
        }
        for row in raw
    ]


def segment_bootstrap_delta(
    y_true: Sequence[int],
    proba_lgbm: Sequence[float],
    proba_logreg: Sequence[float],
    group: pd.Series,
    n_bootstrap: int,
    random_state: int,
) -> list[dict[str, object]]:
    """Paired row-level bootstrap CI on Δ = PR-AUC(LGBM) − PR-AUC(LogReg), per segment.

    Mirrors bootstrap_comparison's pairing logic at the row level instead of
    the fold level: each resample draws row indices once per segment and
    scores both candidates on the identical resampled rows, so shared
    segment-level noise cancels in the delta instead of surviving into two
    separately-compared CIs. Flag-only: never decides the model family
    (bootstrap_comparison's fold-level test alone does that).
    """
    y_arr = np.asarray(y_true, dtype=float)
    lgbm_arr = np.asarray(proba_lgbm, dtype=float)
    logreg_arr = np.asarray(proba_logreg, dtype=float)

    def _delta(y: np.ndarray, lgbm: np.ndarray, logreg: np.ndarray) -> float:
        return float(
            average_precision_score(y, lgbm) - average_precision_score(y, logreg)
        )

    raw = _segment_bootstrap(
        y_arr, group, _delta, (lgbm_arr, logreg_arr), n_bootstrap, random_state
    )
    return [
        {
            "segment": row["segment"],
            "value": row["value"],
            "n": row["n"],
            "delta_obs": row["obs"],
            "delta_ci_lower": row["ci_lower"],
            "delta_ci_upper": row["ci_upper"],
        }
        for row in raw
    ]


def segment_decision_rates(
    y_true: Sequence[int],
    proba: Sequence[float],
    group: pd.Series,
    threshold: float,
) -> list[dict[str, object]]:
    """Per-segment decision-level rates at a fixed threshold.

    PR-AUC parity cannot detect allocation harm: it is threshold-free and measures
    ranking within a group, but a customer is affected by the decision p >= threshold,
    not by the ranking. Returns, per segment, the selection rate (fraction contacted),
    FNR, FPR, and precision at `threshold` — the rates the V2 (demographic-parity
    diff = max selection-rate gap) and V2b veto criteria are computed from. FNR/FPR/
    precision fall back to NaN when their denominator (n_pos/n_neg/predicted-positive
    count) is zero in a segment, rather than raising.
    """
    y_arr = np.asarray(y_true, dtype=float)
    pred_arr = (np.asarray(proba, dtype=float) >= threshold).astype(float)

    rows: list[dict[str, object]] = []
    for value in sorted(group.unique(), key=str):
        mask = (group == value).to_numpy()
        n = int(mask.sum())
        if n < _MIN_SEGMENT_SIZE:
            continue
        y_seg, pred_seg = y_arr[mask], pred_arr[mask]
        tp = float(((pred_seg == 1) & (y_seg == 1)).sum())
        fp = float(((pred_seg == 1) & (y_seg == 0)).sum())
        fn = float(((pred_seg == 0) & (y_seg == 1)).sum())
        tn = float(((pred_seg == 0) & (y_seg == 0)).sum())
        n_pos, n_neg, n_pred_pos = tp + fn, fp + tn, tp + fp

        rows.append(
            {
                "segment": group.name,
                "value": value,
                "n": n,
                "selection_rate": float(pred_seg.mean()),
                "fnr": float(fn / n_pos) if n_pos > 0 else float("nan"),
                "fpr": float(fp / n_neg) if n_neg > 0 else float("nan"),
                "precision": float(tp / n_pred_pos) if n_pred_pos > 0 else float("nan"),
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


def sliced_ranking_metrics(
    y_true: pd.Series,
    proba: np.ndarray,
    segment_lookup: dict[str, pd.Series],
    axes: tuple[str, ...],
    n_bootstrap: int,
    random_state: int,
) -> list[dict[str, object]]:
    """Per-segment PR-AUC with a bootstrap CI and its own churn-rate floor, across
    every named axis — the surface the V1 veto flags a segment collapse
    from. Reused for both the dev-OOF diagnostic pass (threshold.py's dev-OOF
    screen, reported, non-gating) and the sealed-test reporting pass (evaluate.py) — the
    caller decides which by what (y_true, proba, segment_lookup) it passes in.

    Attaches each row's own churn_rate_floor (the segment's y.mean()) rather
    than requiring a second join in flag_segment_collapse: V1 fires when
    pr_auc_ci_lower falls below this value — a model no better than the
    segment's own base rate.
    """
    y = y_true.to_numpy(dtype=float)
    p = np.asarray(proba, dtype=float)
    rows: list[dict[str, object]] = []
    for axis in axes:
        group = segment_lookup[axis]
        for row in segment_bootstrap_ci(
            y.tolist(), p.tolist(), group, n_bootstrap, random_state
        ):
            mask = (group == row["value"]).to_numpy()
            rows.append(
                {**row, "axis": axis, "churn_rate_floor": float(y[mask].mean())}
            )
    return rows


def flag_segment_collapse(
    ranking_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """V1: rows whose PR-AUC 95% CI lower bound falls below the segment's own churn-rate floor.

    Computed by threshold.py's dev-OOF screen on the dev-OOF surface,
    non-gating — a model not distinguishable from random inside a real
    cohort is worth flagging even though the sealed test set is too thin to
    decide it there. Returns only the flagged rows; an empty list means no
    segment is flagged.
    """
    return [
        row
        for row in ranking_rows
        if cast(float, row["pr_auc_ci_lower"]) < cast(float, row["churn_rate_floor"])
    ]


def sliced_decision_rates(
    y_true: pd.Series,
    proba: np.ndarray,
    segment_lookup: dict[str, pd.Series],
    axes: tuple[str, ...],
    threshold: float,
) -> list[dict[str, object]]:
    """Per-segment selection rate/FNR/FPR/precision at `threshold`, across every
    named axis — the surface the V2 veto reads. Reused for the dev-OOF
    diagnostic pass (threshold.py's dev-OOF screen) and the sealed-test
    reporting pass (evaluate.py), same convention as sliced_ranking_metrics.
    """
    y = y_true.to_numpy(dtype=float)
    p = np.asarray(proba, dtype=float)
    rows: list[dict[str, object]] = []
    for axis in axes:
        for row in segment_decision_rates(
            y.tolist(), p.tolist(), segment_lookup[axis], threshold
        ):
            rows.append({**row, "axis": axis})
    return rows


def equal_opportunity_difference_by_axis(
    decision_rows: list[dict[str, object]],
) -> dict[str, float]:
    """Max FNR gap within each axis's own groups — the V2(a) equal-opportunity metric, per axis.

    Computed separately per axis, never pooled across axes: equal-opportunity
    difference is conventionally within one sensitive attribute (male vs.
    female; senior vs. non-senior), and pooling unrelated attributes' FNRs
    into one list would mix two different questions into one number. NaN FNR
    rows (no churners in that group) are excluded; an axis left with fewer
    than two comparable groups returns NaN rather than a spurious 0.
    """
    by_axis: dict[str, list[float]] = {}
    for row in decision_rows:
        fnr = cast(float, row["fnr"])
        if math.isnan(fnr):
            continue
        by_axis.setdefault(cast(str, row["axis"]), []).append(fnr)
    return {
        axis: (max(values) - min(values)) if len(values) >= 2 else float("nan")
        for axis, values in by_axis.items()
    }


def demographic_parity_difference_by_axis(
    decision_rows: list[dict[str, object]],
) -> dict[str, float]:
    """Max selection-rate gap within each axis's own groups — the V2(b) demographic-parity metric, per axis.

    Same per-axis convention as equal_opportunity_difference_by_axis.
    """
    by_axis: dict[str, list[float]] = {}
    for row in decision_rows:
        by_axis.setdefault(cast(str, row["axis"]), []).append(
            cast(float, row["selection_rate"])
        )
    return {
        axis: (max(values) - min(values)) if len(values) >= 2 else float("nan")
        for axis, values in by_axis.items()
    }


def sliced_calibration(
    y_true: pd.Series,
    proba: np.ndarray,
    segment_lookup: dict[str, pd.Series],
    axes: tuple[str, ...],
    cfg: DictConfig,
    n_bootstrap: int,
    random_state: int,
) -> list[dict[str, object]]:
    """Per-segment calibration slope (with CI) and ECE, across every named axis —
    the surface the V2b veto reads. A group whose slope CI falls
    entirely outside [0.80, 1.25] is a group the shipped t* is systematically
    wrong for, in a direction no aggregate slope reveals. Segments smaller
    than _MIN_SEGMENT_SIZE, or with only one class present, are skipped —
    neither a slope nor an ECE is estimable there.
    """
    y = y_true.to_numpy(dtype=float)
    p = np.asarray(proba, dtype=float)
    rows: list[dict[str, object]] = []
    for axis in axes:
        group = segment_lookup[axis]
        for value in sorted(group.unique(), key=str):
            mask = (group == value).to_numpy()
            n = int(mask.sum())
            if n < _MIN_SEGMENT_SIZE or len(np.unique(y[mask])) < 2:
                continue
            y_seg, p_seg = y[mask], p[mask]
            slope = calibration_slope(y_seg, p_seg, n_bootstrap, random_state)
            ece = expected_calibration_error(
                p_seg,
                pd.Series(y_seg),
                int(cfg.calibration.ece_n_bins),
                str(cfg.calibration.ece_strategy),
                f"segment:{axis}={value}",
            )
            rows.append(
                {
                    "axis": axis,
                    "value": value,
                    "n": n,
                    "slope": slope["slope"],
                    "slope_ci_lower": slope["slope_ci_lower"],
                    "slope_ci_upper": slope["slope_ci_upper"],
                    "ece": ece,
                }
            )
    return rows


def flag_calibration_collapse(
    calibration_rows: list[dict[str, object]], band: tuple[float, float]
) -> list[dict[str, object]]:
    """V2b: rows whose calibration-slope CI falls entirely outside `band`.

    Same pass-unless-the-CI-clears-it framing as gate.py's aggregate
    calibration guardrail, applied per group instead of in aggregate.
    Returns only the flagged rows; an empty list means V2b passes.
    """
    lower, upper = band
    return [
        row
        for row in calibration_rows
        if cast(float, row["slope_ci_upper"]) < lower
        or cast(float, row["slope_ci_lower"]) > upper
    ]

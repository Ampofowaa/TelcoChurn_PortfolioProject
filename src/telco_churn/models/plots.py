"""Pure plotting-data helpers for evaluate.py's curves and summary charts.

Pure, side-effect-free (no matplotlib import) — like models/diagnostics.py,
returns list[dict] rows a caller renders however it likes.

Cost/EV curves and the r-sensitivity sweep are not duplicated here: they
already exist as pure functions in models.threshold (expected_value_curve,
r_sensitivity_sweep), which needs them internally to derive t*. Callers
wanting those series import them from there directly.

classification_summary_points computes point estimates only — bootstrap CIs
on its precision/recall/F1 are the caller's job (utils.stats), attached
afterward, so this module never needs a random_state parameter.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

__all__ = [
    "classification_summary_points",
    "decile_lift_table",
    "pr_curve_points",
    "reliability_diagram_bins",
    "roc_curve_points",
]


def reliability_diagram_bins(
    proba: Sequence[float],
    y_true: Sequence[int],
    n_bins: int,
    strategy: str,
) -> list[dict[str, float]]:
    """Bin proba against y_true for a reliability diagram — one row per non-empty bin.

    Same edge construction (quantile or uniform, per strategy) as
    calibrate.expected_calibration_error, so a rendered diagram's bins match
    the ECE number computed alongside it. Empty bins are dropped rather than
    returned as NaN — a plotted NaN point is worse than one fewer marker.
    """
    p = np.asarray(proba, dtype=float)
    y = np.asarray(y_true, dtype=float)

    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1)))
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    edges[0], edges[-1] = 0.0, 1.0

    bin_ids = np.clip(np.digitize(p, edges[1:-1], right=True), 0, len(edges) - 2)

    rows: list[dict[str, float]] = []
    for b in range(len(edges) - 1):
        mask = bin_ids == b
        if not mask.any():
            continue
        rows.append(
            {
                "bin_lower": float(edges[b]),
                "bin_upper": float(edges[b + 1]),
                "mean_predicted": float(p[mask].mean()),
                "observed_frequency": float(y[mask].mean()),
                "count": float(mask.sum()),
            }
        )
    return rows


def pr_curve_points(
    y_true: Sequence[int], proba: Sequence[float]
) -> list[dict[str, float]]:
    """Precision-recall curve points, for rendering the PR curve behind PR-AUC.

    Drops the trailing (precision=1, recall=0) point that has no
    corresponding threshold — same convention as
    diagnostics.py::fixed_recall_profile, so the two agree on where the
    curve ends.
    """
    precision, recall, thresholds = precision_recall_curve(
        np.asarray(y_true), np.asarray(proba)
    )
    precision, recall = precision[:-1], recall[:-1]
    return [
        {"precision": float(p), "recall": float(r), "threshold": float(t)}
        for p, r, t in zip(precision, recall, thresholds, strict=True)
    ]


def roc_curve_points(
    y_true: Sequence[int], proba: Sequence[float]
) -> list[dict[str, float]]:
    """ROC curve points, for rendering a labelled diagnostic curve.

    ROC-AUC is optimistic under this project's class imbalance and is not
    used as a gate — this curve is conventional and cheap to produce, but the
    caller should never render it as the headline.
    """
    fpr, tpr, thresholds = roc_curve(np.asarray(y_true), np.asarray(proba))
    return [
        {"fpr": float(f), "tpr": float(t), "threshold": float(th)}
        for f, t, th in zip(fpr, tpr, thresholds, strict=True)
    ]


def decile_lift_table(
    y_true: Sequence[int], proba: Sequence[float]
) -> list[dict[str, float]]:
    """Decile-ranked lift/gains table, highest-predicted-probability decile first.

    Sorts by proba descending (stable sort, so tied rows keep their original
    order) and splits into 10 groups via np.array_split — the "how much of
    the model's value shows up in the top N% contacted" read a retention
    team wants before a PR-AUC number means anything to them. Per decile:
    count, n_churners, churn_rate, lift (churn_rate / base rate), and
    cumulative_capture_rate (share of all churners found by this decile).

    Also carries perfect_capture_rate — the capture rate a flawless ranker
    would achieve at this decile's cumulative population share — and
    headroom, the gap to that ceiling. perfect_capture_rate saturates at 1.0
    once the contacted share passes the base rate, so it hits 1.0 well before
    decile 10.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(proba, dtype=float)
    base_rate = float(y.mean())
    total_churners = float(y.sum())
    total_count = float(len(y))

    order = np.argsort(-p, kind="stable")
    groups = np.array_split(order, 10)

    rows: list[dict[str, float]] = []
    cumulative_churners = 0.0
    cumulative_count = 0.0
    for i, idx in enumerate(groups, start=1):
        y_g = y[idx]
        count = float(len(idx))
        n_churners = float(y_g.sum())
        churn_rate = n_churners / count if count > 0 else float("nan")
        cumulative_churners += n_churners
        cumulative_count += count
        cumulative_capture_rate = (
            cumulative_churners / total_churners if total_churners > 0 else float("nan")
        )
        population_fraction = (
            cumulative_count / total_count if total_count > 0 else float("nan")
        )
        perfect_capture_rate = (
            min(population_fraction / base_rate, 1.0) if base_rate > 0 else float("nan")
        )
        rows.append(
            {
                "decile": float(i),
                "count": count,
                "n_churners": n_churners,
                "churn_rate": churn_rate,
                "mean_predicted": float(p[idx].mean()) if count > 0 else float("nan"),
                "lift": churn_rate / base_rate if base_rate > 0 else float("nan"),
                "cumulative_capture_rate": cumulative_capture_rate,
                "perfect_capture_rate": perfect_capture_rate,
                "headroom": perfect_capture_rate - cumulative_capture_rate,
            }
        )
    return rows


def classification_summary_points(
    y_true: Sequence[int], proba: Sequence[float], thresholds: dict[str, float]
) -> list[dict[str, object]]:
    """Per-scenario confusion matrix, positive-class rates, support, and contact rate.

    One row per named threshold in `thresholds` — a confusion matrix with
    row-normalised percentages (TP/FN as a share of actual churners, FP/TN
    of actual non-churners) beside positive-class precision/recall/F1,
    support, and contact rate.

    Positive-class only, deliberately: sklearn.classification_report's
    negative-class row and macro/weighted averages are dominated by the
    majority class at ~27% prevalence and mislead exactly where nobody acts
    on a prediction (a customer who will stay). Point estimates only —
    bootstrap CIs are the caller's job.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(proba, dtype=float)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())

    rows: list[dict[str, object]] = []
    for scenario, t in thresholds.items():
        pred = (p >= t).astype(float)
        tp = float(((pred == 1) & (y == 1)).sum())
        fp = float(((pred == 1) & (y == 0)).sum())
        fn = float(((pred == 0) & (y == 1)).sum())
        tn = float(((pred == 0) & (y == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else float("nan")
        )
        rows.append(
            {
                "scenario": scenario,
                "threshold": float(t),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "tp_pct": tp / n_pos if n_pos > 0 else float("nan"),
                "fn_pct": fn / n_pos if n_pos > 0 else float("nan"),
                "fp_pct": fp / n_neg if n_neg > 0 else float("nan"),
                "tn_pct": tn / n_neg if n_neg > 0 else float("nan"),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "n_churners": n_pos,
                "n_non_churners": n_neg,
                "contact_rate": float(pred.mean()),
            }
        )
    return rows

"""Pure SHAP helpers for the champion's error analysis.

No I/O, no MLflow, no shap import — the same contract as diagnostics.py,
economics.py, and plots.py. These take already-computed SHAP values (a shap
library explainer call is I/O-adjacent: it resolves a fitted estimator and
runs inference) and return summary statistics. models/error_analysis.py is
the only caller: it resolves the champion pipeline, computes
`shap.TreeExplainer(...)(...)` once on the sealed-test rows, and calls these
functions on the result. Interpretability is its own discipline,
distinct from performance measurement (evaluate.py) and from error diagnosis
proper — it lives here so neither of those modules has to import shap.

Direction, not magnitude, is what most of this module reports. Global mean
|SHAP| (global_importance) says a feature matters; it has no sign by
construction and cannot say *which way* it pushes a prediction. V3's veto
("a top-8 feature whose effect direction contradicts the established EDA
relationship") and the FN-cohort finding ("do the usual churn signals fire as
strongly for the misses as for the catches?") both need the signed value,
which is why dependence_points and cohort_shap report signed contributions
rather than magnitudes.

local_explanations returns everything a waterfall chart needs (base value,
top-k contributions, an "other features" remainder, and the final
prediction) — not because Phase 7 requires a waterfall figure, but because
the illustrative FN/FP case studies the notebook renders are exactly what a
waterfall is for, and Phase 9's serving UI will want the identical shape for
a live customer later.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "binary_feature_effects",
    "cohort_shap",
    "dependence_points",
    "global_importance",
    "local_explanations",
]


def global_importance(
    shap_values: NDArray[np.float64], feature_names: list[str]
) -> list[dict[str, Any]]:
    """Mean |SHAP| per feature, sorted descending.

    Model documentation, not error analysis: it says what the champion keys
    on, which the model card needs, and nothing about where the model is
    wrong. Mirrors features.select.compute_shap_audit's mean(|SHAP|)
    computation against the tuned champion rather than the default-config
    candidate — but takes already-computed SHAP values rather than fitting a
    model itself, since error_analysis.py already holds the champion's
    shap_values in memory by the time this is called.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    rows: list[dict[str, Any]] = [
        {"feature": name, "mean_abs_shap": float(value)}
        for name, value in zip(feature_names, mean_abs, strict=True)
    ]
    rows.sort(key=lambda row: row["mean_abs_shap"], reverse=True)
    return rows


def dependence_points(
    feature_values: NDArray[np.float64], shap_values: NDArray[np.float64]
) -> dict[str, Any]:
    """Paired (feature value, SHAP value) points for one feature, plus a signed direction summary.

    V3's instrument: V3 vetoes on a top-8 feature whose effect direction
    contradicts the established EDA relationship, and direction is unreadable
    from mean |SHAP| (an absolute value, no sign by construction). `direction`
    is the Pearson correlation between the raw feature value and its own SHAP
    value across rows — its *sign* is what a caller checks against the
    expected EDA relationship; the paired points are what a dependence plot
    renders. A feature with zero variance in either array (degenerate input)
    returns direction 0.0 rather than NaN — there is no direction to read.
    """
    if np.std(feature_values) == 0 or np.std(shap_values) == 0:
        direction = 0.0
    else:
        direction = float(np.corrcoef(feature_values, shap_values)[0, 1])
    return {
        "feature_values": [float(v) for v in feature_values],
        "shap_values": [float(v) for v in shap_values],
        "direction": direction,
    }


def binary_feature_effects(
    feature_values: NDArray[np.float64], shap_values: NDArray[np.float64]
) -> dict[str, Any]:
    """Signed SHAP summary for one 0/1-encoded feature, split by level.

    A binary-encoded feature (e.g. gender, has_partner) has exactly one
    column and exactly two possible raw values — dependence_points' full
    point cloud is built for a continuous feature's scatter/dependence plot
    and is the wrong shape for two levels, and mean |SHAP|
    (global_importance) collapses the two levels together and loses the
    sign entirely. This reports the two group means directly: the average
    signed push toward/away from churn for the "1" level (e.g. Male) versus
    the "0" level (e.g. Female), which is what a beeswarm's colour split
    shows visually but a stakeholder-facing table needs as numbers. `n_at_1`
    is the customer count for `feature_values == 1`; a group with zero rows
    (a constant column) returns a NaN-free mean of 0.0 for that side, since
    there is nothing to average.
    """
    at_1 = feature_values == 1
    at_0 = ~at_1
    if np.std(feature_values) == 0 or np.std(shap_values) == 0:
        direction = 0.0
    else:
        direction = float(np.corrcoef(feature_values, shap_values)[0, 1])
    return {
        "direction": direction,
        "mean_shap_at_1": float(shap_values[at_1].mean()) if at_1.any() else 0.0,
        "mean_shap_at_0": float(shap_values[at_0].mean()) if at_0.any() else 0.0,
        "n_at_1": int(at_1.sum()),
        "n_at_0": int(at_0.sum()),
    }


def cohort_shap(
    shap_values: NDArray[np.float64],
    cohort_mask: NDArray[np.bool_],
    feature_names: list[str],
) -> list[dict[str, Any]]:
    """Signed mean SHAP per feature within a single cohort.

    Calling this once for the FN cohort and once for the TP cohort (or any
    other two masks) and diffing the results is what surfaces "what pushed
    this cohort's scores down": sign, not magnitude, answers the question —
    mean |SHAP| (global_importance) would destroy exactly the information
    needed, since magnitude only says a feature mattered, not which way it
    argued.
    """
    cohort_values = shap_values[cohort_mask]
    mean_signed = cohort_values.mean(axis=0)
    return [
        {"feature": name, "mean_signed_shap": float(value)}
        for name, value in zip(feature_names, mean_signed, strict=True)
    ]


def local_explanations(
    shap_values: NDArray[np.float64],
    base_value: float,
    feature_names: list[str],
    indices: list[int],
    top_k: int,
    feature_values: NDArray[np.float64] | None = None,
) -> list[dict[str, Any]]:
    """Per-row waterfall-ready SHAP breakdown, for a handful of representative cases.

    Strictly illustrative — an individual explanation is an anecdote (n=1, no
    support) and must never be used as evidence in its own right; the
    cohort-level functions above are what actually support a claim. Per-
    customer explanation has a real home in Phase 9's serving UI ("top reasons
    this customer was flagged"); here it exists only to make the cohort
    finding concrete with a couple of illustrative FN/FP cases.

    Returns four quantities per row that together draw a complete
    base -> prediction waterfall without loss, even though only the top_k
    features are labelled individually: `base_value` (shared across rows —
    the explainer's expected output before any feature is considered),
    `top_features` (top_k by |SHAP| magnitude, each carrying its signed
    contribution and, when `feature_values` is supplied, the customer's raw
    value for that feature — the "feature = value" label a waterfall
    annotates each bar with), `other_contribution` (the summed SHAP of every
    feature *not* shown individually, so the remaining bars collapse into one
    honest "other features" bar rather than being silently dropped), and
    `prediction` (base_value plus every feature's SHAP value, top_k or not —
    the endpoint the waterfall's bars must sum to).
    """
    rows: list[dict[str, Any]] = []
    for idx in indices:
        row_shap = shap_values[idx]
        order = np.argsort(-np.abs(row_shap))[:top_k]
        remaining = np.setdiff1d(np.arange(len(row_shap)), order)

        top_features: list[dict[str, Any]] = []
        for j in order:
            entry: dict[str, Any] = {
                "feature": feature_names[j],
                "shap_value": float(row_shap[j]),
            }
            if feature_values is not None:
                entry["feature_value"] = float(feature_values[idx, j])
            top_features.append(entry)

        rows.append(
            {
                "row_index": int(idx),
                "base_value": float(base_value),
                "top_features": top_features,
                "other_contribution": float(row_shap[remaining].sum()),
                "prediction": float(base_value + row_shap.sum()),
            }
        )
    return rows

"""Pure SHAP summary helpers for error analysis.

No I/O, no MLflow, no shap import (same contract as diagnostics.py,
economics.py, plots.py) — these take already-computed SHAP values and return
summary statistics. models/error_analysis.py is the sole caller: it runs
`shap.TreeExplainer(...)(...)` once on the sealed-test rows and passes the
result here.

Most functions report signed direction, not magnitude: global mean |SHAP|
(global_importance) says a feature matters but has no sign by construction,
while V3's veto (a top feature's effect direction contradicting the
established EDA relationship) and the FN-cohort finding both need the sign —
hence dependence_points and cohort_shap report signed contributions.
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

    For the model card, not error diagnosis — magnitude says what the
    champion keys on, not where it's wrong. Mirrors
    features.select.compute_shap_audit's mean(|SHAP|) computation, but takes
    already-computed SHAP values rather than fitting a model itself.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    rows: list[dict[str, Any]] = [
        {"feature": name, "mean_abs_shap": float(value)}
        for name, value in zip(feature_names, mean_abs, strict=True)
    ]
    rows.sort(key=lambda row: row["mean_abs_shap"], reverse=True)
    return rows


def _signed_direction(
    feature_values: NDArray[np.float64], shap_values: NDArray[np.float64]
) -> float:
    """Pearson correlation between a feature's raw values and its own SHAP values.

    Zero variance in either array (degenerate input) returns 0.0 rather than
    NaN — there is no direction to read.
    """
    if np.std(feature_values) == 0 or np.std(shap_values) == 0:
        return 0.0
    return float(np.corrcoef(feature_values, shap_values)[0, 1])


def dependence_points(
    feature_values: NDArray[np.float64], shap_values: NDArray[np.float64]
) -> dict[str, Any]:
    """Paired (feature value, SHAP value) points for one feature, plus a signed direction summary.

    Backs V3's veto, which checks a top feature's effect *sign* against the
    expected EDA relationship — unreadable from mean |SHAP| alone. The paired
    points are what a dependence plot renders; `direction` is 0.0 (not NaN)
    when either array has zero variance.
    """
    direction = _signed_direction(feature_values, shap_values)
    return {
        "feature_values": [float(v) for v in feature_values],
        "shap_values": [float(v) for v in shap_values],
        "direction": direction,
    }


def binary_feature_effects(
    feature_values: NDArray[np.float64], shap_values: NDArray[np.float64]
) -> dict[str, Any]:
    """Signed SHAP summary for one 0/1-encoded feature, split by level.

    A binary-encoded feature (e.g. gender) has two possible raw values —
    dependence_points' point cloud is the wrong shape for that, and
    global_importance collapses the levels and loses the sign. This reports
    the mean signed push for the "1" level and the "0" level directly, what a
    beeswarm's colour split shows visually but a stakeholder table needs as
    numbers. A level with zero rows returns mean 0.0, not NaN.
    """
    at_1 = feature_values == 1
    at_0 = ~at_1
    direction = _signed_direction(feature_values, shap_values)
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

    Call once for the FN cohort and once for the TP cohort, then diff the
    results, to see what pushed that cohort's scores down — sign, not
    magnitude (global_importance), answers that question.
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

    Illustrative only (n=1, no statistical support) — real evidence comes
    from the cohort-level functions above; this exists to make a cohort
    finding concrete with a couple of FN/FP case studies.

    Returns, per row, everything a waterfall needs without loss: `base_value`
    (shared, the explainer's expected output before any feature), `top_features`
    (top_k by |SHAP| magnitude, with the customer's raw value attached when
    `feature_values` is given), `other_contribution` (summed SHAP of every
    feature not shown individually, so nothing is silently dropped), and
    `prediction` (base_value plus every feature's SHAP value — the endpoint
    the bars must sum to).
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

"""Pure SHAP summary helpers, shared by calibrate.py, threshold.py, and error_analysis.py.

No I/O, no MLflow, no shap import (same contract as diagnostics.py,
economics.py, plots.py) — these take already-computed SHAP values (computed
by models/shap_values.py, the sole shap-importing module) and return summary
statistics or verdicts.

Most functions report signed direction, not magnitude: global mean |SHAP|
(global_importance) says a feature matters but has no sign by construction,
while direction_sanity_check's veto (a top feature's effect direction
contradicting the established EDA relationship, EXPECTED_EDA_DIRECTIONS) and
the FN-cohort finding both need the sign — hence dependence_points,
feature_directions, and cohort_shap report signed contributions.
direction_sanity_check and check_top_k_elbow are used twice each per training
cycle: threshold.py binds them on the champion's dev-SHAP ranking (the
pre-seal veto), error_analysis.py re-runs them on the sealed-test ranking as
a reported-only audit.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "EXPECTED_EDA_DIRECTIONS",
    "binary_feature_effects",
    "check_top_k_elbow",
    "cohort_shap",
    "dependence_points",
    "direction_sanity_check",
    "feature_directions",
    "global_importance",
    "local_explanations",
]

# EDA's established bivariate relationships, keyed by a lowercase substring
# of the transformed (post-ColumnTransformer) feature name — the longest
# matching key wins, mirroring features.select's column-map resolution. +1
# means "higher/present pushes toward churn"; -1 the reverse.
# A top-k SHAP feature with no key here has no established relationship to
# contradict and is simply not checked (direction_sanity_check is silent,
# not passing-by-default on a claim it never evaluated).
EXPECTED_EDA_DIRECTIONS: dict[str, int] = {
    "tenure": -1,  # longer tenure -> lower churn
    "totalcharges": -1,  # higher lifetime spend -> lower churn (only long-tenure customers accumulate it)
    "monthlycharges": 1,  # higher monthly spend (fiber concentration) -> higher churn
    "month-to-month": 1,  # contract_type dummy: r ~= +0.40
    "two year": -1,  # contract_type dummy: r ~= -0.30
    "onlinesecurity_no": 1,  # no security add-on -> higher churn, Cramer's V 0.35
    "techsupport_no": 1,  # no support add-on -> higher churn, Cramer's V 0.34
    "onlinebackup_no": 1,  # no backup add-on -> higher churn, Cramer's V 0.29
    "deviceprotection_no": 1,  # no device protection -> higher churn, Cramer's V 0.28
    # The matching "..._no internet service" level (all six add-on columns)
    # is deliberately unkeyed: those columns are identical-by-construction for
    # every no-internet customer and score exactly 0.0 mean |SHAP| (see
    # global_importance), so they can never enter a top-k ranking and a key
    # here would never be exercised.
    "fiber optic": 1,  # internetservice dummy: disproportionately high churn, Cramer's V 0.32
    "internetservice_no": -1,  # no internet service -> lowest churn rate of the three tiers, every contract length
    "electronic check": 1,  # paymentmethod dummy: highest-churn payment method, Cramer's V 0.30
    "paperlessbilling": 1,  # paperless billing -> higher churn, Cramer's V 0.19
    "seniorcitizen": 1,  # senior citizen -> higher churn, Cramer's V 0.15
    "has_partner": -1,  # has a partner -> lower churn, Cramer's V 0.15
    "dependents": -1,  # has dependents -> lower churn, Cramer's V 0.16
}


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


def feature_directions(
    Xt: NDArray[np.float64], shap_values: NDArray[np.float64], feature_names: list[str]
) -> dict[str, float]:
    """Signed direction (see _signed_direction) for every transformed feature, not just a top-k slice.

    calibrate.py's dev-SHAP summary needs a direction for every feature in
    the model input space to log alongside global_importance's mean |SHAP| —
    dependence_points would also work per-feature but additionally builds and
    returns the full paired-points lists, wasted work when only the scalar
    direction is wanted for every column.
    """
    return {
        name: _signed_direction(Xt[:, i], shap_values[:, i])
        for i, name in enumerate(feature_names)
    }


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


def _match_expected_direction_key(
    feature_name: str, expected_directions: dict[str, int]
) -> str | None:
    """Longest-matching key in `expected_directions` that is a substring of
    `feature_name` (case-insensitive), or None if nothing matches."""
    name_lower = feature_name.lower()
    candidates = [key for key in expected_directions if key in name_lower]
    if not candidates:
        return None
    return max(candidates, key=len)


def direction_sanity_check(
    top_feature_names: Sequence[str],
    directions: dict[str, float],
    expected_directions: dict[str, int],
) -> dict[str, Any]:
    """Does any of `top_feature_names`' effect direction contradict the
    established EDA relationship?

    `directions` is {feature_name: signed correlation} — feature_directions'
    or dependence_points' `direction` output for each of `top_feature_names`.
    `expected_directions` maps a substring of a feature name to its
    established sign (+1/-1); a feature matching no key has no established
    relationship to contradict and is recorded as checked=False rather than
    silently counted as a pass. threshold.py's pre-seal screen calls this on
    the champion's own dev-SHAP top-k as a binding veto
    (`v3_direction_sanity`); error_analysis.py calls it again on the sealed-
    test SHAP ranking as a reported-only re-audit
    (`direction_sanity_check_test`) — a sign flip on a feature this
    influential means the model is fitting an artifact, and no aggregate
    metric would reveal it.
    """
    rows: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    for name in top_feature_names:
        key = _match_expected_direction_key(name, expected_directions)
        if key is None:
            rows.append(
                {
                    "feature": name,
                    "checked": False,
                    "matched_eda_relationship": None,
                    "expected_sign": None,
                    "observed_direction": directions[name],
                    "contradicts": False,
                }
            )
            continue
        expected_sign = expected_directions[key]
        observed = directions[name]
        contradicts = observed != 0.0 and np.sign(observed) != np.sign(expected_sign)
        row = {
            "feature": name,
            "checked": True,
            "matched_eda_relationship": key,
            "expected_sign": expected_sign,
            "observed_direction": observed,
            "contradicts": bool(contradicts),
        }
        rows.append(row)
        if contradicts:
            violations.append(row)
    return {
        "checked_features": rows,
        "violations": violations,
        "passed": len(violations) == 0,
    }


def check_top_k_elbow(
    values: list[float],
    configured_k: int,
    plateau_window: int = 3,
    min_elbow_ratio: float = 1.5,
) -> dict[str, Any]:
    """Verify the plateau-then-elbow shape that justified `configured_k` still holds.

    `v3_top_k_features` (threshold.py), `top_k_dependence_features`,
    `top_k_cohort_gap_features`, and `top_k_cohort_gap_features_fp_tn`
    (error_analysis.py) were each hand-derived once from the sorted deltas of
    exactly this kind of ranking (see their config comments), and nothing
    re-derives them when the champion changes — Phase 10's retrain calls
    these modules with no human in the loop, so a stale cutoff would
    otherwise drift silently.

    Deliberately not "auto-pick k = argmax(delta)": on this project's own
    global-importance ranking, the single biggest delta sits at rank 1-2
    (0.22), not the plateau's real end (rank 8, delta 0.046) — an argmax
    picker would discard most of the features the human derivation correctly
    kept. Instead this asks a narrower question: is the drop right after
    `configured_k` still clearly bigger than the drops right before it?
    `plateau_window` deltas ending at `configured_k` set the local baseline;
    `min_elbow_ratio` is how many multiples of it the elbow drop must clear
    to pass. A ratio well below that threshold means the plateau has shifted
    and `k` needs re-deriving by hand. This catches gross drift only — it
    doesn't resolve boundary-adjacent ambiguity (an off-by-one `k` may still
    pass) and never raises: too few features before `configured_k` to form a
    baseline reports `valid=True` rather than a spurious failure.
    """
    deltas = [values[i] - values[i + 1] for i in range(len(values) - 1)]
    elbow_delta = deltas[configured_k - 1]
    window_start = max(0, configured_k - 1 - plateau_window)
    plateau = sorted(deltas[window_start : configured_k - 1])
    if not plateau:
        return {
            "configured_k": configured_k,
            "elbow_delta": elbow_delta,
            "plateau_median_delta": None,
            "ratio": None,
            "valid": True,
        }
    mid = len(plateau) // 2
    plateau_median = (
        plateau[mid] if len(plateau) % 2 else (plateau[mid - 1] + plateau[mid]) / 2
    )
    ratio = elbow_delta / plateau_median if plateau_median > 0 else float("inf")
    return {
        "configured_k": configured_k,
        "elbow_delta": elbow_delta,
        "plateau_median_delta": plateau_median,
        "ratio": ratio,
        "valid": ratio >= min_elbow_ratio,
    }

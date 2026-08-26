"""Error-diagnosis step, run after evaluate.py. Owns reports/error_analysis.json.

Answers what aggregate metrics can't: *where* the errors are, *how wrong*
they were, and *what they cost*. Delegates SHAP computation to
models/shap_values.py and SHAP summary statistics/verdicts to explain.py.

Data provenance:
- Test/dev-OOF predictions come from evaluate.py's test_predictions.parquet
  and calibrate.py's dev-OOF vector (via threshold.load_dev_oof_predictions,
  fetched by run_id — never threshold.py's local parquet mirror, which only
  reflects whichever run last executed threshold.py on this machine).
- The champion model is loaded by explicit version, never alias, and never
  via data/split.py, so evaluate.py stays the sole importer of the test split.
- Raw feature values are recovered via features.accessor.load_features(),
  restricted to an already-sealed id set — never a fresh split.

register.py aborts without this module's output: model-card fairness and
robustness sections must be assembled from an artifact, not retyped from a
rendered notebook cell.

Error-concentration scanning (exploratory cohort search) runs on dev OOF
only — the 1,409-row sealed test set is too small for that much slicing to
be sound, and dev has roughly four times the statistical power. Every other
analysis here (SHAP, error confidence, value-weighted errors, illustrative
cases) runs on the sealed test set, mirroring what evaluate.py measured its
headline numbers on.

SHAP values are computed against the champion's base LightGBM decision
function, not the final calibrated probability — TreeExplainer cannot see
through CalibratedClassifierCV's post-hoc sigmoid/isotonic map (the same
constraint features.select.compute_shap_audit and calibrate.py's dev-SHAP
logging work under). That map is monotone, so a feature's *direction* of
effect is provably identical pre- and post-calibration — threshold.py's V3
pre-seal veto relies on exactly this, and only this. Its *magnitude* is not
similarly protected: a non-linear monotone map can stretch or compress SHAP
magnitudes unevenly, so global_importance's mean-|SHAP| ranking (and
therefore which features make a top-k cut) is a pre-calibration ranking, not
guaranteed to match a post-calibration one.

This module's own SHAP pass runs on the *sealed test set*, after evaluate.py.
"""

from __future__ import annotations

import json
import re
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, cast

import matplotlib.pyplot as plt
import mlflow
import mlflow.tracking
import numpy as np
import pandas as pd
import shap
from matplotlib.axes import Axes
from mlflow.data.pandas_dataset import from_pandas as mlflow_dataset_from_pandas
from numpy.typing import NDArray
from omegaconf import DictConfig
from scipy.stats import mannwhitneyu

from telco_churn.features.accessor import load_features
from telco_churn.features.build import TARGET_COL
from telco_churn.features.schema import FEATURE_SCHEMA
from telco_churn.models.artifacts import (
    committed_features_from_manifest,
    load_dev_oof_diagnostics,
    load_dev_oof_predictions,
    load_fitted_model,
    load_threshold_validation,
    load_training_manifest,
)
from telco_churn.models.explain import (
    EXPECTED_EDA_DIRECTIONS,
    binary_feature_effects,
    check_top_k_elbow,
    cohort_shap,
    dependence_points,
    direction_sanity_check,
    global_importance,
    local_explanations,
)
from telco_churn.models.gate import (
    check_threshold_provenance,
    check_threshold_screen_passed,
)
from telco_churn.models.policy_config import (
    load_policy_thresholds,
    resolve_policy_thresholds_by_scenario,
)
from telco_churn.models.shap_values import (
    compute_shap_values,
    unwrap_calibrated_pipeline,
)
from telco_churn.utils.logging import get_logger
from telco_churn.utils.mlflow import (
    ensure_experiment_metadata,
    resolve_logged_model_id,
    resolve_model_identifier,
    resolve_tracking_uri,
    set_run_description,
    write_error_analysis_receipt,
)
from telco_churn.utils.paths import get_project_root

__all__ = [
    "display_feature_name",
    "error_concentration_scan",
    "error_confidence_profile",
    "near_miss_band_from_validation",
    "run_error_analysis_step",
    "value_weighted_error_profile",
]

logger = get_logger(__name__)

_RUN_DESCRIPTION = (
    "Error analysis for one evaluated model version - SHAP global "
    "importance and values (model documentation, not error analysis), "
    "FN/FP case inspection, and a direction-sanity check feeding the human "
    "review stamped onto promotion_decision.json. Writes "
    "error_analysis.json, later read into model_card.json at promotion."
)

# Sentinel for a dev_oof_diagnostics key that postdates an older logged
# threshold/dev_oof_diagnostics.json artifact — never [] or {}, which read as
# "assessed, nothing flagged" and are indistinguishable from an honest empty
# result. None of today's four v1/v2/v2b keys need it (threshold.py's dev-OOF
# screen always produces all four together), but the mechanism exists so a
# future added key follows this rule instead of reintroducing the encoding
# defect this replaces.
_NOT_ASSESSED = "not_assessed"

# MLflow metric names allow only alphanumerics, underscores, dashes, periods,
# spaces, and slashes — one-hot feature names from ColumnTransformer.
# get_feature_names_out() routinely carry other characters (e.g. "...Bank
# transfer (automatic)"), so every character outside that set is replaced
# with an underscore before a feature name becomes part of a metric key.
_METRIC_NAME_DISALLOWED_CHARS_RE = re.compile(r"[^a-zA-Z0-9_\-. /]")


# Human-readable label for every FEATURE_SCHEMA column — a fixed lookup rather
# than a `.replace("_", " ").title()` heuristic, since most raw columns
# (phoneservice, onlinesecurity, monthlycharges, ...) are concatenated with no
# separator at all and would title-case unreadably ("Phoneservice") without
# an explicit mapping. Chart display only; the underlying artifacts keep the
# raw ColumnTransformer name as the audit trail.
_COLUMN_DISPLAY_NAMES: dict[str, str] = {
    "gender": "Gender",
    "has_partner": "Has Partner",
    "dependents": "Dependents",
    "phoneservice": "Phone Service",
    "paperlessbilling": "Paperless Billing",
    "seniorcitizen": "Senior Citizen",
    "multiplelines": "Multiple Lines",
    "internetservice": "Internet Service",
    "onlinesecurity": "Online Security",
    "onlinebackup": "Online Backup",
    "deviceprotection": "Device Protection",
    "techsupport": "Tech Support",
    "streamingtv": "Streaming TV",
    "streamingmovies": "Streaming Movies",
    "contract_type": "Contract Type",
    "paymentmethod": "Payment Method",
    "tenure": "Tenure",
    "monthlycharges": "Monthly Charges",
    "totalcharges": "Total Charges",
    "charge_per_service": "Charge Per Service",
}


def display_feature_name(raw_name: str) -> str:
    """Human-readable label for a raw ColumnTransformer feature name, for
    chart display only.

    "numeric__monthlycharges" -> "Monthly Charges"; "multi_cat__contract_type_
    Month-to-month" -> "Contract Type: Month-to-month". Matches the known
    column against FEATURE_SCHEMA's binary/multi_cat lists (longest match
    wins, mirroring explain.EXPECTED_EDA_DIRECTIONS' convention) rather than
    splitting on the first underscore, since several raw columns
    (contract_type, charge_per_service) already contain underscores that
    would make a naive split land in the wrong place. Falls back to the raw
    name for anything unrecognised — a display helper must never raise on an
    unexpected feature.
    """
    prefix, sep, remainder = raw_name.partition("__")
    if not sep:
        return raw_name

    if prefix == "numeric":
        return _COLUMN_DISPLAY_NAMES.get(remainder, remainder)

    if prefix in ("binary", "multi_cat"):
        columns = (
            FEATURE_SCHEMA.binary if prefix == "binary" else FEATURE_SCHEMA.multi_cat
        )
        matches = [
            c for c in columns if remainder == c or remainder.startswith(c + "_")
        ]
        if not matches:
            return raw_name
        column = max(matches, key=len)
        category = remainder[len(column) :].lstrip("_")
        pretty_column = _COLUMN_DISPLAY_NAMES.get(column, column)
        return f"{pretty_column}: {category}" if category else pretty_column

    return raw_name


def near_miss_band_from_validation(
    validation_payload: dict[str, Any], fallback: float
) -> float:
    """Data-driven near-miss band: half the width of threshold.py's
    argmax-EV bootstrap CI (threshold_validation.json's `argmax_ev_bootstrap_ci`
    for the base scenario), rather than an arbitrary constant.

    That CI already answers the exact question a near-miss band is trying to
    approximate — "how far could the empirically optimal cut point plausibly
    have landed, given sampling noise in the dev-OOF data?" — so a FN/FP
    inside this radius of t* sits in a region where t* itself is not
    statistically distinguishable from another plausible threshold, which is
    a sharper definition of "threshold artifact" than a fixed number picked
    without reference to the data. Falls back to `fallback` (configs/
    error_analysis/default.yaml's near_miss_band) when the CI key is absent,
    e.g. for a run predating threshold.py logging it.
    """
    ci = validation_payload.get("argmax_ev_bootstrap_ci")
    if not ci:
        return fallback
    ci_lower, ci_upper = ci
    return float(ci_upper - ci_lower) / 2.0


def error_confidence_profile(
    y_true: Sequence[int],
    proba: Sequence[float],
    threshold: float,
    near_miss_band: float,
) -> dict[str, Any]:
    """FN/FP score distributions against the shipped threshold, split into
    near-miss (a threshold artifact) vs. confident failure (a model failure).

    A false negative at p just under `threshold` sits on the wrong side of an
    arbitrary cut and would flip under a marginally different t*; one near 0.0
    is a case the model never considered a churn risk at all — those two
    events call for different responses, and nothing else in the phase tells
    them apart. Reports both raw score arrays (for the notebook's histogram)
    and each cohort's near-miss share: the fraction scoring within
    `near_miss_band` of `threshold`. An empty FN or FP cohort reports NaN
    shares rather than raising or dividing by zero — a legitimate, if
    unusual, outcome.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(proba, dtype=float)
    fn_mask = (y == 1) & (p < threshold)
    fp_mask = (y == 0) & (p >= threshold)
    fn_scores = p[fn_mask]
    fp_scores = p[fp_mask]

    def _near_miss_share(scores: NDArray[np.float64]) -> float:
        if scores.size == 0:
            return float("nan")
        return float((np.abs(scores - threshold) <= near_miss_band).mean())

    return {
        "threshold": float(threshold),
        "near_miss_band": float(near_miss_band),
        "n_fn": int(fn_mask.sum()),
        "n_fp": int(fp_mask.sum()),
        "fn_scores": fn_scores.tolist(),
        "fp_scores": fp_scores.tolist(),
        "fn_near_miss_share": _near_miss_share(fn_scores),
        "fp_near_miss_share": _near_miss_share(fp_scores),
    }


def value_weighted_error_profile(
    y_true: Sequence[int],
    proba: Sequence[float],
    value: Sequence[float],
    threshold: float,
    n_deciles: int,
) -> list[dict[str, Any]]:
    """Per-decile FN/FP rate against `value` (MonthlyCharges) — is the model
    systematically missing its highest-value churners?

    Missing a $110/month fiber customer is not the same event as missing a
    $20/month DSL customer, and nothing else in the phase distinguishes them.
    Deciles are quantile bins of `value` (pd.qcut, ascending; duplicates
    collapsed when too few distinct values fill every bin), so decile 1 is
    always the cheapest customers and the highest-numbered decile the most
    valuable, regardless of how many bins survive. FN/FP rate in each decile
    is computed against that decile's own actual positives/negatives — NaN
    when a decile has none of either, since a small, high-value decile can
    legitimately contain zero churners. `n_pos`/`n_neg` are persisted
    alongside the rates (not just re-derivable from `n`) because they are
    the rate's own denominator and its reliability signal: a decile's
    fn_rate resting on 7 churners is not the same evidence as one resting on
    61, and that distinction is invisible from `n` and the rate alone.
    """
    y = np.asarray(y_true, dtype=float)
    pred_pos = np.asarray(proba, dtype=float) >= threshold
    v = np.asarray(value, dtype=float)

    bins = pd.qcut(v, q=n_deciles, duplicates="drop")
    rows: list[dict[str, Any]] = []
    for decile, category in enumerate(bins.categories, start=1):
        mask = np.asarray(bins == category)
        n = int(mask.sum())
        y_seg, pred_seg, v_seg = y[mask], pred_pos[mask], v[mask]
        n_pos = int((y_seg == 1).sum())
        n_neg = int((y_seg == 0).sum())
        fn = int(((y_seg == 1) & ~pred_seg).sum())
        fp = int(((y_seg == 0) & pred_seg).sum())
        rows.append(
            {
                "decile": decile,
                "value_range": [float(category.left), float(category.right)],
                "n": n,
                "mean_value": float(v_seg.mean()) if n > 0 else float("nan"),
                "n_pos": n_pos,
                "n_neg": n_neg,
                "fn_rate": float(fn / n_pos) if n_pos > 0 else float("nan"),
                "fp_rate": float(fp / n_neg) if n_neg > 0 else float("nan"),
            }
        )
    return rows


def error_concentration_scan(
    y_true: Sequence[int],
    proba: Sequence[float],
    threshold: float,
    feature_frame: pd.DataFrame,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    n_quantile_bins: int,
    min_support: int,
    top_k: int,
    metric: Literal["fnr", "fpr"],
) -> list[dict[str, Any]]:
    """Highest-error-rate cohorts across every feature (categorical as-is,
    numeric quantile-binned) — an automated slice search, not a check against
    a fixed axis list.

    `metric` selects which rate cohorts are ranked by: "fnr" (missed
    churners — the primary business risk this project's cost structure is
    built around) or "fpr" (false alarms — cheaper, but still worth
    surfacing, since a cohort with a wildly disproportionate false-alarm rate
    can erode trust in the campaign or blow through `contact_capacity` for no
    retained revenue). Every row carries its own `metric` field, so a caller
    combining both scans' output never has to infer which rate a row reports
    from context.

    V1/V2/V2b (diagnostics.py, called from evaluate.py) check exactly seven
    pre-registered axes; a catastrophic cohort on any other feature — or on a
    feature this project never had a prior hypothesis about — is invisible to
    them by construction, and a veto cannot fire on something nobody computed.
    This scans feature_frame's full column set instead. Confirmatory, not
    generative: a discovered cohort may inform the human review and belongs
    in the model card, but must never send you back to engineer a feature —
    that is training-time feature engineering's job, and this runs on
    already-sealed evaluation data.

    Cohorts below `min_support`, or carrying zero of the relevant actual class
    (no positives for "fnr", no negatives for "fpr" — the rate is undefined),
    are dropped before ranking — a two-row cohort with one error is not a
    signal. Returns the top_k cohorts by rate descending.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(proba, dtype=float)
    is_error = p < threshold if metric == "fnr" else p >= threshold
    target_class = 1 if metric == "fnr" else 0

    rows: list[dict[str, Any]] = []

    def _add_rows(series: pd.Series, feature_name: str, feature_type: str) -> None:
        for value in sorted(series.dropna().unique(), key=str):
            mask = (series == value).to_numpy()
            n = int(mask.sum())
            if n < min_support:
                continue
            y_seg = y[mask]
            n_target = int((y_seg == target_class).sum())
            if n_target == 0:
                continue
            errors = int(((y_seg == target_class) & is_error[mask]).sum())
            rows.append(
                {
                    "feature": feature_name,
                    "value": str(value),
                    "type": feature_type,
                    "n": n,
                    "n_target": n_target,
                    "rate": float(errors / n_target),
                    "metric": metric,
                }
            )

    for col in categorical_features:
        _add_rows(feature_frame[col], col, "categorical")
    for col in numeric_features:
        binned = pd.qcut(feature_frame[col], q=n_quantile_bins, duplicates="drop")
        _add_rows(binned.astype(str), col, "numeric_bin")

    rows.sort(key=lambda row: cast(float, row["rate"]), reverse=True)
    return rows[:top_k]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _sanitize_metric_name(name: str) -> str:
    """Replace every character MLflow's metric-name validator rejects with an
    underscore — see _METRIC_NAME_DISALLOWED_CHARS_RE."""
    return _METRIC_NAME_DISALLOWED_CHARS_RE.sub("_", name)


def _features_by_customer_id(
    customer_ids: pd.Series,
    raw_partition_override: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Full raw feature rows for exactly `customer_ids`, via the unsplit
    feature table — never telco_churn's data/split.py. `customer_ids` already
    carries whatever partition membership an upstream artifact (evaluate.py's
    test_predictions.parquet, calibrate.py's dev-OOF vector) sealed; this only
    joins raw columns onto that already-fixed identity set, it does not
    re-derive a split.

    `raw_partition_override` defaults to `load_features()` (the static,
    customers_raw-only processed-features parquet). Pass
    `models/train/common.py::load_training_pool_bundle`'s own `raw_partition`
    return once the dev-OOF vector includes matured reserve cohorts — a
    reserve customerid is found in
    `load_features()` too (same physical customer), so the missing-id check
    below never fires, but the row returned would be that customer's
    *original, frozen* customers_raw snapshot rather than the nudged values
    (via training_pool) the model actually trained and was evaluated on —
    quietly wrong SHAP feature values next to a customer's explanation, not a
    raised error. Sealed-test lookups never pass this override: test
    customers are never part of the reserve mechanism.

    Checks for a missing lookup by id membership, not by scanning the joined
    frame for any NaN: 11 zero-tenure customers carry a legitimate NaN
    totalcharges (the raw CSV's blank-string TotalCharges for zero-tenure
    rows, imputed downstream by the model's ColumnTransformer, not by
    load_features()), and one of them routinely lands in the sealed test
    set. An any-NaN check would treat that expected value as a broken join.
    """
    source = (
        raw_partition_override
        if raw_partition_override is not None
        else load_features()
    )
    df = source.set_index("customerid")
    missing_ids = set(customer_ids) - set(df.index)
    assert not missing_ids, (
        f"{'raw_partition_override' if raw_partition_override is not None else 'load_features()'} "
        f"is missing {len(missing_ids)} of the requested customerids — it no "
        "longer matches the prediction artifact this customer id set came from."
    )
    return df.reindex(customer_ids).reset_index()


def _illustrative_indices(
    y_true: NDArray[np.float64],
    proba: NDArray[np.float64],
    threshold: float,
    n_examples: int,
) -> dict[str, list[int]]:
    """Row indices for the most confidently-wrong FN and FP cases, plus the
    single most-confident TP and TN — strictly illustrative (explain.py::
    local_explanations' own docstring: an anecdote, never evidence). FN/FP
    each get `n_examples` rows furthest on the wrong side of `threshold`,
    since those are the cases a waterfall chart makes for the most legible
    story. TP/TN each get exactly one — the single most-confident correct
    case per side — not `n_examples`: their purpose is a same-chart-type
    comparison against the FN/FP panels ("does a false positive really look
    like a true positive?"), which one clean reference case per side answers
    as well as several would; multiplying it by n_examples would just be more
    panels making the same point.
    """
    fn_idx = np.where((y_true == 1) & (proba < threshold))[0]
    fp_idx = np.where((y_true == 0) & (proba >= threshold))[0]
    tp_idx = np.where((y_true == 1) & (proba >= threshold))[0]
    tn_idx = np.where((y_true == 0) & (proba < threshold))[0]
    fn_sorted = fn_idx[np.argsort(proba[fn_idx])][:n_examples]
    fp_sorted = fp_idx[np.argsort(-proba[fp_idx])][:n_examples]
    tp_sorted = tp_idx[np.argsort(-proba[tp_idx])][:1]
    tn_sorted = tn_idx[np.argsort(proba[tn_idx])][:1]
    return {
        "fn": fn_sorted.tolist(),
        "fp": fp_sorted.tolist(),
        "tp": tp_sorted.tolist(),
        "tn": tn_sorted.tolist(),
    }


def _save_shap_global_importance_plot(rows: list[dict[str, Any]], path: Path) -> None:
    """Horizontal bar chart of mean |SHAP| for every feature in the model
    input space — model documentation, not error analysis
    (explain.py::global_importance). Unlike the dependence subset
    (top_k_dependence_features), this renders the full ranking: a feature
    ranked 9th or lower is outside that audit set but is exactly what a
    stakeholder skimming the model card asks about next."""
    fig, ax = plt.subplots(figsize=(10, 0.3 * len(rows) + 1.5))
    names = [display_feature_name(cast(str, r["feature"])) for r in rows][::-1]
    values = [cast(float, r["mean_abs_shap"]) for r in rows][::-1]
    ax.barh(names, values, color="tab:blue")
    ax.set_xlabel("mean |SHAP|")
    ax.set_title(f"SHAP global importance — all {len(rows)} features")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _save_shap_beeswarm_plot(
    shap_values: NDArray[np.float64],
    Xt: NDArray[np.float64],
    feature_names: list[str],
    path: Path,
) -> None:
    """SHAP beeswarm (summary) plot — every feature, every sealed-test row:
    mean |SHAP| ranks features (global_importance), but collapses each one to
    a single number and cannot show a feature that pushes different customers
    in opposite directions (e.g. high vs. low value of a numeric feature)
    versus one that only ever pushes toward churn. shap.summary_plot draws
    onto the current pyplot figure rather than returning one, so the figure
    is created explicitly first and read back via plt.gcf() afterward.
    plot_size=None is required for that pre-set size to actually take effect
    — shap.summary_plot's default plot_size="auto" hardcodes its own width
    (fig.set_size_inches(8, ...)) regardless of the current figure's size,
    silently discarding whatever figsize was passed to plt.figure() above.
    rng=42 makes the point-jitter reproducible and opts out of shap's legacy
    global-NumPy-RNG fallback (SPEC 7), consistent with this project's
    random_state=42 convention everywhere else.
    """
    plt.figure(figsize=(14, 0.3 * len(feature_names) + 1.5))
    display_names = [display_feature_name(name) for name in feature_names]
    shap.summary_plot(
        shap_values,
        Xt,
        feature_names=display_names,
        max_display=len(display_names),
        plot_size=None,
        show=False,
        rng=42,
    )
    fig = plt.gcf()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _save_dependence_plots(
    dependence_by_feature: dict[str, dict[str, Any]],
    dev_v3_audit_set: set[str],
    contradicting_features: set[str],
    path: Path,
) -> None:
    """One scatter subplot per dependence-grid feature — SHAP value vs. raw
    (transformed) feature value.

    Renders top_k_dependence_features' test-side ranking unioned with
    threshold.py's own dev-side V3 audit set (dev_v3_audit_set) — panel
    titles mark which features are inherited from that dev audit set
    (audit-set) and which of those contradict their established EDA
    direction on the sealed test set (direction_sanity_check_test's
    violations) — CONTRADICTS is only ever set alongside audit-set, since
    only audit-set features are actually re-checked (no independent
    test-side k).
    """
    features = list(dependence_by_feature)
    fig, axes = plt.subplots(1, len(features), figsize=(5.0 * len(features), 4.5))
    axes_list = [axes] if len(features) == 1 else list(axes)
    for ax, feature in zip(axes_list, features, strict=True):
        dep = dependence_by_feature[feature]
        ax.scatter(
            dep["feature_values"], dep["shap_values"], s=8, alpha=0.5, color="tab:blue"
        )
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel(display_feature_name(feature))
        ax.set_ylabel("SHAP value")
        title = f"direction = {dep['direction']:.2f}"
        if feature in dev_v3_audit_set:
            title += " [audit-set]"
        if feature in contradicting_features:
            title += " [CONTRADICTS]"
        ax.set_title(title, fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _sorted_gap_pairs(
    rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]]
) -> list[tuple[str, float]]:
    """Every feature's |A - B| mean-signed-SHAP gap, (name, gap) sorted descending.

    Shared by both cohort pairs this module renders (FN-vs-TP, FP-vs-TN):
    both need "which features actually separate the two cohorts," not
    global_importance's overall-population rank, which can and does pick a
    different top-n (a feature both cohorts agree on contributes nothing to
    distinguishing one from the other). Returns every feature, not a top-n
    slice — the full ranking is what `explain.check_top_k_elbow` verifies a
    cutoff against, and what error_analysis.json persists so a drifted
    cutoff can be re-derived by hand straight from the file.
    """
    b_by_feature = {
        cast(str, r["feature"]): cast(float, r["mean_signed_shap"]) for r in rows_b
    }
    gaps = [
        (
            cast(str, r["feature"]),
            abs(
                cast(float, r["mean_signed_shap"])
                - b_by_feature.get(cast(str, r["feature"]), 0.0)
            ),
        )
        for r in rows_a
    ]
    gaps.sort(key=lambda t: t[1], reverse=True)
    return gaps


def _save_cohort_shap_plot(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
    top_features: list[str],
    path: Path,
    label_a: str = "FN cohort",
    label_b: str = "TP cohort",
    color_a: str = "tab:red",
    color_b: str = "tab:green",
    title: str = "FN vs. TP cohort — signed SHAP",
) -> None:
    """Signed mean SHAP, cohort A vs. cohort B, for `top_features`
    (_sorted_gap_pairs' top slice) — do the model's usual churn signals fire
    as strongly for one cohort as they do for the other, or not at all?
    Defaults render the FN-vs-TP pair; the FP-vs-TN pair passes its own
    labels/colors/title through the same plotting logic.
    """
    a_by_feature = {
        cast(str, r["feature"]): cast(float, r["mean_signed_shap"]) for r in rows_a
    }
    b_by_feature = {
        cast(str, r["feature"]): cast(float, r["mean_signed_shap"]) for r in rows_b
    }
    top = [
        (f, a_by_feature.get(f, 0.0), b_by_feature.get(f, 0.0)) for f in top_features
    ]

    fig, ax = plt.subplots(figsize=(10, 0.5 * len(top) + 1.5))
    y_pos = np.arange(len(top))
    ax.barh(y_pos - 0.2, [t[1] for t in top], height=0.4, label=label_a, color=color_a)
    ax.barh(
        y_pos + 0.2,
        [t[2] for t in top],
        height=0.4,
        label=label_b,
        color=color_b,
    )
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([display_feature_name(t[0]) for t in top])
    ax.invert_yaxis()
    ax.set_xlabel("mean signed SHAP")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _save_cohort_beeswarm_plots(
    shap_values: NDArray[np.float64],
    Xt: NDArray[np.float64],
    feature_names: list[str],
    mask_a: NDArray[np.bool_],
    mask_b: NDArray[np.bool_],
    top_features: list[str],
    path_a: Path,
    path_b: Path,
) -> None:
    """Beeswarms restricted to cohort A and cohort B, each showing only
    `top_features` — the cohort-vs-cohort bar chart above (e.g. FN vs. TP,
    or FP vs. TN) collapses each cohort to one signed mean per feature; this
    restores the individual spread within each cohort that a single mean
    can't show, exactly as the full-population beeswarm restores what
    global_importance's mean |SHAP| collapses away. Reuses
    _save_shap_beeswarm_plot unchanged, on a row-and-column subset of the
    same shap_values/Xt this module already computed once — no new SHAP call.
    """
    col_idx = [feature_names.index(f) for f in top_features]
    subset_names = [feature_names[i] for i in col_idx]
    _save_shap_beeswarm_plot(
        shap_values[mask_a][:, col_idx], Xt[mask_a][:, col_idx], subset_names, path_a
    )
    _save_shap_beeswarm_plot(
        shap_values[mask_b][:, col_idx], Xt[mask_b][:, col_idx], subset_names, path_b
    )


def _save_cohort_dependence_plot(
    feature_values: NDArray[np.float64],
    shap_values_feature: NDArray[np.float64],
    fn_mask: NDArray[np.bool_],
    tp_mask: NDArray[np.bool_],
    fp_mask: NDArray[np.bool_],
    tn_mask: NDArray[np.bool_],
    feature_display_name: str,
    path: Path,
) -> None:
    """Cohort-colored dependence scatter for one feature — raw (transformed)
    value on x, SHAP value on y — split into the same two panels as the
    cohort-means bar charts: actual churners (FN vs. TP) on the left, actual
    non-churners (FP vs. TN) on the right.

    The cohort means and cohort beeswarms above show that the FN/FP cohorts'
    SHAP values are diluted/collided relative to TP/TN, but neither shows
    *where in raw feature space* that happens — a mean or a per-cohort
    beeswarm can't say whether the two cohorts sit in the same region of the
    feature's range. This is the direct check: if FN and TP dots (or FP and
    TN dots) occupy the same x-range, the model has no boundary in this
    feature to draw between them — the "no loyalty/satisfaction signal, so
    these customer types overlap in the features available" claim made
    qualitatively above, shown instead of asserted. The larger cohort (TP,
    TN) is drawn first and the error cohort (FN, FP) drawn on top at higher
    alpha, so a numerically smaller error cohort stays visible against a
    denser background.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    axes[0].scatter(
        feature_values[tp_mask],
        shap_values_feature[tp_mask],
        s=12,
        alpha=0.35,
        color="tab:green",
        label="TP",
    )
    axes[0].scatter(
        feature_values[fn_mask],
        shap_values_feature[fn_mask],
        s=14,
        alpha=0.8,
        color="tab:red",
        label="FN",
    )
    axes[0].axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_xlabel(feature_display_name)
    axes[0].set_ylabel("SHAP value")
    axes[0].set_title("Actual churners: FN vs. TP")
    axes[0].legend()

    axes[1].scatter(
        feature_values[tn_mask],
        shap_values_feature[tn_mask],
        s=12,
        alpha=0.35,
        color="tab:blue",
        label="TN",
    )
    axes[1].scatter(
        feature_values[fp_mask],
        shap_values_feature[fp_mask],
        s=14,
        alpha=0.8,
        color="tab:orange",
        label="FP",
    )
    axes[1].axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel(feature_display_name)
    axes[1].set_title("Actual non-churners: FP vs. TN")
    axes[1].legend()

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _save_subgroup_dependence_plot(
    feature_values: NDArray[np.float64],
    shap_values_feature: NDArray[np.float64],
    mask_a: NDArray[np.bool_],
    mask_b: NDArray[np.bool_],
    label_a: str,
    label_b: str,
    color_a: str,
    color_b: str,
    feature_display_name: str,
    title: str,
    path: Path,
) -> None:
    """Single-panel dependence scatter restricted to two named subgroups,
    rather than the whole-population cohort split _save_cohort_dependence_plot
    draws.

    Not every follow-up question is answered by the same whole-cohort FN/TP
    vs. FP/TN structure — a feature's gap can be real and large within one
    specific subgroup while washing out to near-nothing in the full-cohort
    comparison that selects _save_cohort_dependence_plot's features in the
    first place (composition-masking, the same phenomenon the cohort-means
    section found for Contract Type, one level down). This exists for exactly
    that case: it takes whatever two boolean masks the caller has already
    used to establish a subgroup-level finding (e.g. a Mann-Whitney test) and
    renders the identical comparison as a scatter, so the finding is shown,
    not just asserted. `mask_b` (expected to be the larger, "normal" group)
    is drawn first at lower alpha so `mask_a` stays visible against it.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        feature_values[mask_b],
        shap_values_feature[mask_b],
        s=14,
        alpha=0.4,
        color=color_b,
        label=label_b,
    )
    ax.scatter(
        feature_values[mask_a],
        shap_values_feature[mask_a],
        s=18,
        alpha=0.85,
        color=color_a,
        label=label_a,
    )
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel(feature_display_name)
    ax.set_ylabel("SHAP value")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _draw_waterfall_panel(ax: Axes, row: dict[str, Any], label: str) -> None:
    """Draw one illustrative-case SHAP waterfall onto `ax`: cumulative bars
    from base_value through each top-k feature's signed contribution, an
    "other features" remainder bar, to the final prediction — explain.py::
    local_explanations' output rendered as the chart it names itself after.
    Legend uses SHAP's own E[f(x)]/f(x) notation (the expected value the
    explainer starts from, and this row's actual score) rather than bare
    "base"/"prediction" — the bars sum to exactly f(x) - E[f(x)], and naming
    them that way is what makes the sum-to relationship legible from the
    legend alone, not just from the axis.
    """
    base_value = cast(float, row["base_value"])
    prediction = cast(float, row["prediction"])
    top_features = cast(list[dict[str, Any]], row["top_features"])
    other_contribution = cast(float, row["other_contribution"])

    bar_labels: list[str] = []
    contributions: list[float] = []
    for feature in top_features:
        name = display_feature_name(cast(str, feature["feature"]))
        if "feature_value" in feature:
            name = f"{name} = {cast(float, feature['feature_value']):.2f}"
        bar_labels.append(name)
        contributions.append(cast(float, feature["shap_value"]))
    bar_labels.append("other features")
    contributions.append(other_contribution)

    y_positions = np.arange(len(contributions))
    cumulative = base_value
    for y_pos, contribution in zip(y_positions, contributions, strict=True):
        color = "tab:red" if contribution >= 0 else "tab:blue"
        ax.barh(y_pos, contribution, left=cumulative, color=color)
        cumulative += contribution

    ax.axvline(
        base_value,
        color="gray",
        linestyle="--",
        linewidth=0.8,
        label=f"E[f(x)] = {base_value:.3f}",
    )
    ax.axvline(
        prediction,
        color="black",
        linestyle="-",
        linewidth=1.0,
        label=f"f(x) = {prediction:.3f}",
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(bar_labels)
    ax.invert_yaxis()
    ax.set_title(label, fontsize=9)
    ax.legend(fontsize=7)


def _save_waterfall_plots(
    fn_rows: list[dict[str, Any]],
    fn_labels: list[str],
    fp_rows: list[dict[str, Any]],
    fp_labels: list[str],
    path: Path,
) -> None:
    """Two-row grid of illustrative-case SHAP waterfalls: false negatives on
    the top row, false positives on the bottom.

    Strictly illustrative, exactly as local_explanations' own docstring
    insists: a single case is an anecdote (n=1, no support) and must never be
    read as evidence in its own right — these panels exist only to make the
    cohort-level finding (_save_cohort_shap_plot) concrete with real cases,
    never to argue a point on their own. Two rows rather than one long row of
    FN-then-FP panels: it keeps each error type visually grouped and gives
    every panel roughly twice the width a single six-wide row would, since
    `fn_rows`/`fp_rows` are typically the same length as each other but half
    the combined count. `fn_rows`/`fn_labels` and `fp_rows`/`fp_labels` must
    each be paired and ordered consistently (a label identifies its panel as
    e.g. "Missed churner — FN (idx=17)" — which illustrative case it is, not
    a claim about it); the two sides may differ in length if fewer than the
    configured number of FN or FP cases exist in the sealed test set, so
    empty grid cells are turned off rather than assumed equal.
    """
    n_cols = max(len(fn_rows), len(fp_rows), 1)
    fig, axes = plt.subplots(2, n_cols, figsize=(5.5 * n_cols, 9.0), squeeze=False)

    for row_idx, (rows, labels) in enumerate(
        ((fn_rows, fn_labels), (fp_rows, fp_labels))
    ):
        for col_idx in range(n_cols):
            ax = axes[row_idx][col_idx]
            if col_idx >= len(rows):
                ax.axis("off")
                continue
            _draw_waterfall_panel(ax, rows[col_idx], labels[col_idx])

    fig.suptitle("SHAP Waterfall — Illustrative Cases", fontsize=12)
    fig.supxlabel("SHAP contribution (raw model score)")
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _save_confident_case_waterfalls(
    tp_row: dict[str, Any],
    tp_label: str,
    tn_row: dict[str, Any],
    tn_label: str,
    path: Path,
) -> None:
    """One-row, two-panel grid: the single most-confident true positive next
    to the single most-confident true negative.

    The FN/FP grid above never shows what a *correct* prediction looks like
    in the same chart type — a reader can only take the cohort-level
    "false positives look like true positives" finding (_save_cohort_shap_plot)
    on faith, or go compare an aggregate bar chart against an individual
    waterfall by eye. This is the direct, same-format check: one true
    positive and one true negative, drawn with the identical
    _draw_waterfall_panel used for every FN/FP case, so the comparison is
    apples to apples. Deliberately one panel per side, not `n_examples` —
    see _illustrative_indices' docstring for why more would not add anything.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    _draw_waterfall_panel(axes[0], tp_row, tp_label)
    _draw_waterfall_panel(axes[1], tn_row, tn_label)
    fig.suptitle("SHAP Waterfall — Confident Cases", fontsize=12)
    fig.supxlabel("SHAP contribution (raw model score)")
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _save_error_confidence_plot(confidence: dict[str, Any], path: Path) -> None:
    """FN/FP score histograms against t*, with the near-miss band shaded."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    threshold = cast(float, confidence["threshold"])
    band = cast(float, confidence["near_miss_band"])
    fn_scores = cast(list[float], confidence["fn_scores"])
    fp_scores = cast(list[float], confidence["fp_scores"])
    bins = np.linspace(0.0, 1.0, 21).tolist()
    if fn_scores:
        ax.hist(fn_scores, bins=bins, alpha=0.6, label="FN scores", color="tab:red")
    if fp_scores:
        ax.hist(fp_scores, bins=bins, alpha=0.6, label="FP scores", color="tab:orange")
    ax.axvline(threshold, color="black", linestyle="--", label=f"t* = {threshold:.3f}")
    ax.axvspan(
        threshold - band,
        threshold + band,
        color="gray",
        alpha=0.15,
        label="near-miss band",
    )
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("count")
    ax.set_title("Error confidence — sealed test")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _save_value_weighted_plot(rows: list[dict[str, Any]], path: Path) -> None:
    """FN/FP rate by MonthlyCharges decile."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    deciles = [cast(int, r["decile"]) for r in rows]
    fn_rates = [cast(float, r["fn_rate"]) for r in rows]
    fp_rates = [cast(float, r["fp_rate"]) for r in rows]
    width = 0.35
    x = np.arange(len(deciles))
    ax.bar(x - width / 2, fn_rates, width, label="FN rate", color="tab:red")
    ax.bar(x + width / 2, fp_rates, width, label="FP rate", color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in deciles])
    ax.set_xlabel("MonthlyCharges decile (1 = cheapest)")
    ax.set_ylabel("rate")
    ax.set_title("Value-weighted errors — sealed test")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _load_error_analysis_inputs(
    run_id: str,
    model_version: str,
    model_uri: str,
    cfg: DictConfig,
    raw_partition_override: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Check threshold provenance/screen status, load test/dev-OOF predictions, and join raw feature values.

    Reads reports/test_predictions.parquet (evaluate.py) + the dev-OOF vector
    (threshold.py's dev-OOF screen, fetched by run_id from calibrate.py's
    logged artifact) — the only route this module reaches evaluation data
    through.

    `raw_partition_override` is threaded only to the dev-OOF feature lookup,
    never the sealed-test one — see
    `_features_by_customer_id`'s own docstring for why.
    """
    mlflow.set_tracking_uri(resolve_tracking_uri(str(cfg.mlflow.tracking_uri)))

    validation_payload = load_threshold_validation(run_id, cfg)
    logged_model_id = resolve_logged_model_id(model_version, cfg)
    check_threshold_provenance(validation_payload, logged_model_id)
    check_threshold_screen_passed(validation_payload)
    manifest = load_training_manifest(run_id, cfg)
    committed_features = committed_features_from_manifest(manifest)
    model = load_fitted_model(model_uri, cfg)

    reports_dir = get_project_root() / str(cfg.paths.reports)
    test_predictions = pd.read_parquet(reports_dir / "test_predictions.parquet")
    stamped_model_ids = test_predictions["logged_model_id"].unique()
    if list(stamped_model_ids) != [logged_model_id]:
        raise ValueError(
            "reports/test_predictions.parquet is stamped with "
            f"logged_model_id={list(stamped_model_ids)!r}, which does not "
            f"match the model being analyzed (logged_model_id="
            f"{logged_model_id!r}) — this is a stale local copy left over "
            "from a different evaluate.py cycle (e.g. a rolled-back "
            "champion). Re-run models.evaluate for this model_version first."
        )
    dev_oof_predictions = load_dev_oof_predictions(run_id, cfg)

    test_features_df = _features_by_customer_id(test_predictions["customerid"])
    dev_features_df = _features_by_customer_id(
        dev_oof_predictions["customerid"], raw_partition_override
    )

    policy = load_policy_thresholds(cfg)
    base_threshold = resolve_policy_thresholds_by_scenario(policy)["base"]

    y_test = test_predictions["y_true"].to_numpy(dtype=float)
    proba_test = test_predictions["p_hat"].to_numpy(dtype=float)
    y_dev = dev_oof_predictions["y_true"].to_numpy(dtype=float)
    proba_dev = dev_oof_predictions["p_hat"].to_numpy(dtype=float)

    return {
        "run_id": run_id,
        "validation_payload": validation_payload,
        "committed_features": committed_features,
        "model": model,
        "reports_dir": reports_dir,
        "test_predictions": test_predictions,
        "test_features_df": test_features_df,
        "dev_features_df": dev_features_df,
        "base_threshold": base_threshold,
        "y_test": y_test,
        "proba_test": proba_test,
        "y_dev": y_dev,
        "proba_dev": proba_dev,
    }


def _compute_shap_diagnostics(
    model: Any,
    test_features_df: pd.DataFrame,
    committed_features: list[str],
    dev_v3_audit_set: list[str],
    ea_cfg: Any,
) -> dict[str, Any]:
    """Compute SHAP values once, global importance, dependence, the test-side
    direction-sanity re-audit, and binary effects.

    Everything downstream that needs SHAP reads from this one bundle —
    shap_values/feature_names/base_value/Xt_test/feature_index live across
    nearly the whole error-analysis pass, so they travel together rather
    than being re-threaded individually.

    `dev_v3_audit_set` is threshold.py's own V3 pre-seal top-k feature names
    (dev_oof_diagnostics.json's direction_check_feature_names) — direction_sanity_
    check_test re-audits exactly that set on the sealed test set, with no
    independent test-side k of its own (item 11): this checks whether what
    threshold.py already vetoed on dev still holds up on test, it does not
    re-derive its own veto surface. The dependence grid separately renders
    the broader union of that inherited set and this module's own
    top_k_dependence_features test-side ranking, for visual completeness.
    """
    preprocessor, booster = unwrap_calibrated_pipeline(model)
    shap_values, feature_names, base_value, Xt_test = compute_shap_values(
        preprocessor, booster, test_features_df[committed_features]
    )
    feature_index = {name: i for i, name in enumerate(feature_names)}

    importance_rows = global_importance(shap_values, feature_names)
    importance_by_feature = {
        cast(str, row["feature"]): cast(float, row["mean_abs_shap"])
        for row in importance_rows
    }
    top_k_dependence = int(ea_cfg.top_k_dependence_features)
    test_top_features = [
        cast(str, r["feature"]) for r in importance_rows[:top_k_dependence]
    ]
    top_k_dependence_elbow = check_top_k_elbow(
        [cast(float, r["mean_abs_shap"]) for r in importance_rows], top_k_dependence
    )
    if not top_k_dependence_elbow["valid"]:
        logger.warning(
            "top_k_dependence_features_elbow_drifted", **top_k_dependence_elbow
        )

    dependence_feature_set = sorted(
        set(test_top_features) | set(dev_v3_audit_set),
        key=lambda f: importance_by_feature[f],
        reverse=True,
    )
    dependence_by_feature = {
        feat: dependence_points(
            Xt_test[:, feature_index[feat]], shap_values[:, feature_index[feat]]
        )
        for feat in dependence_feature_set
    }
    directions = {
        feat: cast(float, dep["direction"])
        for feat, dep in dependence_by_feature.items()
    }
    direction_sanity_check_test = direction_sanity_check(
        dev_v3_audit_set, directions, EXPECTED_EDA_DIRECTIONS
    )
    contradicting_features = {
        cast(str, row["feature"]) for row in direction_sanity_check_test["violations"]
    }

    binary_features = [name for name in feature_names if name.startswith("binary__")]
    binary_dependence = {
        feat: binary_feature_effects(
            Xt_test[:, feature_index[feat]], shap_values[:, feature_index[feat]]
        )
        for feat in binary_features
    }

    return {
        "shap_values": shap_values,
        "feature_names": feature_names,
        "base_value": base_value,
        "Xt_test": Xt_test,
        "feature_index": feature_index,
        "importance_rows": importance_rows,
        "test_top_features": test_top_features,
        "dev_v3_audit_set": dev_v3_audit_set,
        "dependence_feature_set": dependence_feature_set,
        "top_k_dependence_elbow": top_k_dependence_elbow,
        "dependence_by_feature": dependence_by_feature,
        "binary_dependence": binary_dependence,
        "direction_sanity_check_test": direction_sanity_check_test,
        "contradicting_features": contradicting_features,
    }


def _compute_cohort_gap(
    shap_values: NDArray[np.float64],
    feature_names: list[str],
    mask_a: NDArray[np.bool_],
    mask_b: NDArray[np.bool_],
    top_k: int,
    elbow_warning_event: str,
) -> dict[str, Any]:
    """Compute cohort-A-vs-cohort-B signed SHAP and rank features by their |gap|.

    Generic: called once for FN-vs-TP, once for FP-vs-TN — each axis has its
    own plateau/elbow shape, so each gets its own top_k and its own elbow
    check rather than sharing one.
    """
    shap_rows_a = cohort_shap(shap_values, mask_a, feature_names)
    shap_rows_b = cohort_shap(shap_values, mask_b, feature_names)
    gap_ranking = _sorted_gap_pairs(shap_rows_a, shap_rows_b)
    top_features = [name for name, _ in gap_ranking[:top_k]]
    elbow = check_top_k_elbow([gap for _, gap in gap_ranking], top_k)
    if not elbow["valid"]:
        logger.warning(elbow_warning_event, **elbow)
    return {
        "shap_rows_a": shap_rows_a,
        "shap_rows_b": shap_rows_b,
        "gap_ranking": gap_ranking,
        "top_features": top_features,
        "elbow": elbow,
    }


def _annual_contract_subgroup_finding(
    Xt_test: NDArray[np.float64],
    shap_values: NDArray[np.float64],
    feature_index: dict[str, int],
    fn_mask: NDArray[np.bool_],
    tn_mask: NDArray[np.bool_],
) -> dict[str, Any]:
    """Confirmatory follow-up on Monthly Charges within annual/two-year-contract customers.

    Monthly Charges' full-cohort FN-vs-TP gap is small precisely because the
    60% month-to-month share of false negatives dilutes it — conditioning on
    contract type the way Contract Type's own cohort-means finding did
    reveals what the full-cohort comparison hides.
    """
    not_month_to_month = (
        Xt_test[:, feature_index["multi_cat__contract_type_Month-to-month"]] == 0
    )
    fn_annual_mask = fn_mask & not_month_to_month
    tn_annual_mask = tn_mask & not_month_to_month
    monthlycharges_col = Xt_test[:, feature_index["numeric__monthlycharges"]]
    monthlycharges_shap_col = shap_values[:, feature_index["numeric__monthlycharges"]]
    _, annual_monthlycharges_mwu_p = mannwhitneyu(
        monthlycharges_col[fn_annual_mask],
        monthlycharges_col[tn_annual_mask],
        alternative="greater",
    )
    finding = {
        "n_fn": int(fn_annual_mask.sum()),
        "n_tn": int(tn_annual_mask.sum()),
        "fn_median": float(np.median(monthlycharges_col[fn_annual_mask])),
        "tn_median": float(np.median(monthlycharges_col[tn_annual_mask])),
        "fn_mean_shap": float(monthlycharges_shap_col[fn_annual_mask].mean()),
        "tn_mean_shap": float(monthlycharges_shap_col[tn_annual_mask].mean()),
        "mannwhitney_p_fn_greater": float(annual_monthlycharges_mwu_p),
    }
    return {
        "finding": finding,
        "fn_annual_mask": fn_annual_mask,
        "tn_annual_mask": tn_annual_mask,
        "monthlycharges_col": monthlycharges_col,
        "monthlycharges_shap_col": monthlycharges_shap_col,
    }


def _compute_illustrative_cases(
    y_test: NDArray[np.float64],
    proba_test: NDArray[np.float64],
    base_threshold: float,
    shap_values: NDArray[np.float64],
    base_value: float,
    feature_names: list[str],
    Xt_test: NDArray[np.float64],
    ea_cfg: Any,
) -> dict[str, Any]:
    """Pick illustrative FN/FP/TP/TN cases and pre-slice their waterfall rows/labels.

    The offsets that carve local_explanation_rows/illustrative_labels into
    per-outcome groups are computed once, here, and the pre-sliced groups
    are what callers receive — never raw offsets a renderer would have to
    re-apply. That keeps the highest-risk arithmetic in this module (which
    row belongs to which outcome) in exactly one place.
    """
    illustrative = _illustrative_indices(
        y_test, proba_test, base_threshold, int(ea_cfg.n_local_explanation_examples)
    )
    illustrative_indices = (
        illustrative["fn"]
        + illustrative["fp"]
        + illustrative["tp"]
        + illustrative["tn"]
    )
    local_explanation_rows = local_explanations(
        shap_values,
        base_value,
        feature_names,
        illustrative_indices,
        int(ea_cfg.local_explanation_top_k),
        feature_values=Xt_test,
    )
    illustrative_labels = (
        [f"Missed churner — FN (idx={idx})" for idx in illustrative["fn"]]
        + [f"Over-flagged non-churner — FP (idx={idx})" for idx in illustrative["fp"]]
        + [f"Most confident churner — TP (idx={idx})" for idx in illustrative["tp"]]
        + [f"Most confident non-churner — TN (idx={idx})" for idx in illustrative["tn"]]
    )

    n_fn_illustrative = len(illustrative["fn"])
    n_fp_illustrative = len(illustrative["fp"])
    fn_end = n_fn_illustrative
    fp_end = fn_end + n_fp_illustrative
    tp_end = fp_end + len(illustrative["tp"])

    return {
        "illustrative": illustrative,
        "local_explanation_rows": local_explanation_rows,
        "illustrative_labels": illustrative_labels,
        "fn_rows": local_explanation_rows[:fn_end],
        "fn_labels": illustrative_labels[:fn_end],
        "fp_rows": local_explanation_rows[fn_end:fp_end],
        "fp_labels": illustrative_labels[fn_end:fp_end],
        "tp_row": local_explanation_rows[fp_end:tp_end][0],
        "tp_label": illustrative_labels[fp_end:tp_end][0],
        "tn_row": local_explanation_rows[tp_end:][0],
        "tn_label": illustrative_labels[tp_end:][0],
    }


def _compute_error_profiles(
    y_test: Any,
    proba_test: Any,
    base_threshold: float,
    test_features_df: pd.DataFrame,
    validation_payload: dict[str, Any],
    ea_cfg: Any,
) -> dict[str, Any]:
    """Compute the near-miss band, error-confidence profile, and value-weighted error profile."""
    near_miss_band = near_miss_band_from_validation(
        validation_payload, float(ea_cfg.near_miss_band)
    )
    confidence = error_confidence_profile(
        y_test, proba_test, base_threshold, near_miss_band
    )
    value_weighted_rows = value_weighted_error_profile(
        y_test,
        proba_test,
        test_features_df["monthlycharges"].to_numpy(dtype=float),
        base_threshold,
        int(ea_cfg.n_value_deciles),
    )
    fn_rate_top_value_decile = (
        value_weighted_rows[-1]["fn_rate"] if value_weighted_rows else float("nan")
    )
    return {
        "confidence": confidence,
        "value_weighted_rows": value_weighted_rows,
        "fn_rate_top_value_decile": fn_rate_top_value_decile,
    }


def _compute_error_concentration(
    y_dev: Any,
    proba_dev: Any,
    base_threshold: float,
    dev_features_df: pd.DataFrame,
    ea_cfg: Any,
) -> dict[str, Any]:
    """Run the dev-OOF error-concentration scan, once ranked by FNR and once by FPR."""
    numeric_features = list(FEATURE_SCHEMA.numeric)
    categorical_features = list(FEATURE_SCHEMA.binary) + list(FEATURE_SCHEMA.multi_cat)
    concentration_scan_args = (
        y_dev,
        proba_dev,
        base_threshold,
        dev_features_df,
        numeric_features,
        categorical_features,
        int(ea_cfg.n_quantile_bins),
        int(ea_cfg.min_cohort_support),
        int(ea_cfg.top_k_cohorts),
    )
    concentration_fnr_rows = error_concentration_scan(
        *concentration_scan_args, metric="fnr"
    )
    concentration_fpr_rows = error_concentration_scan(
        *concentration_scan_args, metric="fpr"
    )
    return {
        "concentration_fnr_rows": concentration_fnr_rows,
        "concentration_fpr_rows": concentration_fpr_rows,
    }


def _assemble_error_analysis_payload(
    model_version: str,
    run_id: str,
    base_threshold: float,
    shap_diag: dict[str, Any],
    fn_tp_gap: dict[str, Any],
    fp_tn_gap: dict[str, Any],
    subgroup: dict[str, Any],
    illustrative_bundle: dict[str, Any],
    error_profiles: dict[str, Any],
    concentration: dict[str, Any],
) -> dict[str, Any]:
    """Assemble error_analysis.json from every already-computed block.

    Does not carry threshold.py's dev_oof_diagnostics.json (V1/V2/V2b/V3)
    into this payload — register.py's model card is the sole reader of that
    data and fetches it directly via evaluate.load_dev_oof_diagnostics(run_id,
    cfg), off the same canonical artifact this run already resolved for its
    own direction_sanity_check_test re-audit, rather than through a copy
    re-keyed here that nothing else consumes.
    """
    return {
        "model_version": model_version,
        "run_id": run_id,
        "base_threshold": base_threshold,
        "error_confidence": error_profiles["confidence"],
        "value_weighted_errors": {
            "by_decile": error_profiles["value_weighted_rows"],
            "fn_rate_top_value_decile": error_profiles["fn_rate_top_value_decile"],
        },
        "error_concentration": {
            "dev_oof_top_fnr_cohorts": concentration["concentration_fnr_rows"],
            "dev_oof_top_fpr_cohorts": concentration["concentration_fpr_rows"],
        },
        "shap": {
            "global_importance": shap_diag["importance_rows"],
            "top_features": shap_diag["test_top_features"],
            "dependence_feature_set": shap_diag["dependence_feature_set"],
            "dependence": shap_diag["dependence_by_feature"],
            "binary_dependence": shap_diag["binary_dependence"],
            "cohort_shap": {
                "fn": fn_tp_gap["shap_rows_a"],
                "tp": fn_tp_gap["shap_rows_b"],
                "fp": fp_tn_gap["shap_rows_a"],
                "tn": fp_tn_gap["shap_rows_b"],
            },
            "cohort_top_features": fn_tp_gap["top_features"],
            "cohort_top_features_fp_tn": fp_tn_gap["top_features"],
            "cohort_gap_ranking_fn_tp": [
                {"feature": name, "gap": gap} for name, gap in fn_tp_gap["gap_ranking"]
            ],
            "cohort_gap_ranking_fp_tn": [
                {"feature": name, "gap": gap} for name, gap in fp_tn_gap["gap_ranking"]
            ],
            "local_explanations": illustrative_bundle["local_explanation_rows"],
        },
        "top_k_elbow_checks": {
            "dependence_features": shap_diag["top_k_dependence_elbow"],
            "cohort_gap_fn_tp": fn_tp_gap["elbow"],
            "cohort_gap_fp_tn": fp_tn_gap["elbow"],
        },
        "subgroup_findings": {
            "annual_contract_fn_vs_tn_monthlycharges": subgroup["finding"],
        },
        # The reported-only sealed-test re-audit of threshold.py's binding
        # pre-seal V3 veto (item 11: no independent test-side k). The binding
        # verdict itself (dev_oof_diagnostics.json's direction_sanity_result)
        # is not carried through here — register.py's model card fetches it
        # directly; see _assemble_error_analysis_payload's docstring.
        "direction_sanity_check_test": shap_diag["direction_sanity_check_test"],
    }


def _render_error_analysis_figures(
    shap_diag: dict[str, Any],
    fn_tp_gap: dict[str, Any],
    fp_tn_gap: dict[str, Any],
    masks: dict[str, NDArray[np.bool_]],
    subgroup: dict[str, Any],
    illustrative_bundle: dict[str, Any],
    error_profiles: dict[str, Any],
    cfg: DictConfig,
) -> dict[str, Path]:
    """Render all fifteen error-analysis figures and return their paths."""
    shap_values, Xt_test = shap_diag["shap_values"], shap_diag["Xt_test"]
    feature_names, feature_index = (
        shap_diag["feature_names"],
        shap_diag["feature_index"],
    )

    figures_dir = get_project_root() / str(cfg.paths.figures)
    paths = {
        "global_importance_path": figures_dir / "shap_global_importance.png",
        "beeswarm_path": figures_dir / "shap_beeswarm.png",
        "dependence_path": figures_dir / "shap_dependence.png",
        "cohort_shap_path": figures_dir / "shap_cohort_fn_vs_tp.png",
        "cohort_beeswarm_fn_path": figures_dir / "shap_cohort_beeswarm_fn.png",
        "cohort_beeswarm_tp_path": figures_dir / "shap_cohort_beeswarm_tp.png",
        "cohort_shap_fp_tn_path": figures_dir / "shap_cohort_fp_vs_tn.png",
        "cohort_beeswarm_fp_path": figures_dir / "shap_cohort_beeswarm_fp.png",
        "cohort_beeswarm_tn_path": figures_dir / "shap_cohort_beeswarm_tn.png",
        "cohort_dependence_tenure_path": figures_dir
        / "shap_cohort_dependence_tenure.png",
        "subgroup_dependence_monthlycharges_path": figures_dir
        / "shap_subgroup_dependence_monthlycharges_annual.png",
        "waterfall_path": figures_dir / "shap_waterfall_examples.png",
        "waterfall_confident_cases_path": figures_dir
        / "shap_waterfall_confident_cases.png",
        "error_confidence_path": figures_dir / "error_confidence.png",
        "value_weighted_path": figures_dir / "value_weighted_errors.png",
    }

    _save_shap_global_importance_plot(
        shap_diag["importance_rows"], paths["global_importance_path"]
    )
    _save_shap_beeswarm_plot(
        shap_values, Xt_test, feature_names, paths["beeswarm_path"]
    )
    _save_dependence_plots(
        shap_diag["dependence_by_feature"],
        set(shap_diag["dev_v3_audit_set"]),
        shap_diag["contradicting_features"],
        paths["dependence_path"],
    )
    _save_cohort_shap_plot(
        fn_tp_gap["shap_rows_a"],
        fn_tp_gap["shap_rows_b"],
        fn_tp_gap["top_features"],
        paths["cohort_shap_path"],
    )
    _save_cohort_beeswarm_plots(
        shap_values,
        Xt_test,
        feature_names,
        masks["fn_mask"],
        masks["tp_mask"],
        fn_tp_gap["top_features"],
        paths["cohort_beeswarm_fn_path"],
        paths["cohort_beeswarm_tp_path"],
    )
    _save_cohort_shap_plot(
        fp_tn_gap["shap_rows_a"],
        fp_tn_gap["shap_rows_b"],
        fp_tn_gap["top_features"],
        paths["cohort_shap_fp_tn_path"],
        label_a="FP cohort",
        label_b="TN cohort",
        color_a="tab:orange",
        color_b="tab:blue",
        title="FP vs. TN cohort — signed SHAP",
    )
    _save_cohort_beeswarm_plots(
        shap_values,
        Xt_test,
        feature_names,
        masks["fp_mask"],
        masks["tn_mask"],
        fp_tn_gap["top_features"],
        paths["cohort_beeswarm_fp_path"],
        paths["cohort_beeswarm_tn_path"],
    )
    _save_cohort_dependence_plot(
        Xt_test[:, feature_index["numeric__tenure"]],
        shap_values[:, feature_index["numeric__tenure"]],
        masks["fn_mask"],
        masks["tp_mask"],
        masks["fp_mask"],
        masks["tn_mask"],
        display_feature_name("numeric__tenure"),
        paths["cohort_dependence_tenure_path"],
    )
    # Renders the annual-contract FN-vs-TN Monthly Charges comparison already
    # computed and stats-tested by _annual_contract_subgroup_finding —
    # confirmatory, not a fresh fishing trip. Needs its own restricted plot
    # rather than reusing _save_cohort_dependence_plot's whole-cohort
    # structure precisely because the whole-cohort view is what hides this
    # finding in the first place.
    _save_subgroup_dependence_plot(
        subgroup["monthlycharges_col"],
        subgroup["monthlycharges_shap_col"],
        subgroup["fn_annual_mask"],
        subgroup["tn_annual_mask"],
        "FN (annual/two-year contract)",
        "TN (annual/two-year contract)",
        "tab:red",
        "tab:blue",
        display_feature_name("numeric__monthlycharges"),
        "Monthly Charges — annual/two-year-contract customers only: FN vs. TN",
        paths["subgroup_dependence_monthlycharges_path"],
    )
    _save_waterfall_plots(
        illustrative_bundle["fn_rows"],
        illustrative_bundle["fn_labels"],
        illustrative_bundle["fp_rows"],
        illustrative_bundle["fp_labels"],
        paths["waterfall_path"],
    )
    _save_confident_case_waterfalls(
        illustrative_bundle["tp_row"],
        illustrative_bundle["tp_label"],
        illustrative_bundle["tn_row"],
        illustrative_bundle["tn_label"],
        paths["waterfall_confident_cases_path"],
    )
    _save_error_confidence_plot(
        error_profiles["confidence"], paths["error_confidence_path"]
    )
    _save_value_weighted_plot(
        error_profiles["value_weighted_rows"], paths["value_weighted_path"]
    )

    return paths


def _log_error_analysis_run(
    model_version: str,
    loaded: dict[str, Any],
    shap_diag: dict[str, Any],
    concentration: dict[str, Any],
    error_analysis_payload: dict[str, Any],
    figure_paths: dict[str, Path],
    cfg: DictConfig,
) -> str:
    """Log every error-analysis artifact/metric onto a dedicated `error_analysis` run.

    Returns error_analysis_run_id, consumed after the run context closes
    (registry tagging).
    """
    test_predictions = loaded["test_predictions"]
    committed_features = loaded["committed_features"]
    test_features_df = loaded["test_features_df"]
    direction_sanity_check_test = shap_diag["direction_sanity_check_test"]

    ensure_experiment_metadata(cfg)
    model_id = resolve_logged_model_id(model_version, cfg)
    test_dataset = mlflow_dataset_from_pandas(
        pd.concat(
            [
                test_features_df[committed_features].reset_index(drop=True),
                test_predictions["y_true"].rename(TARGET_COL).reset_index(drop=True),
            ],
            axis=1,
        ),
        name="sealed_test",
        targets=TARGET_COL,
    )

    with mlflow.start_run(run_name="error_analysis") as run:
        set_run_description(_RUN_DESCRIPTION)
        error_analysis_run_id = str(run.info.run_id)
        mlflow.log_input(test_dataset, context="error_analysis")

        scalar_metrics: dict[str, float] = {
            "test_fn_near_miss_share": cast(
                float, error_analysis_payload["error_confidence"]["fn_near_miss_share"]
            ),
            "test_fn_rate_top_value_decile": cast(
                float,
                error_analysis_payload["value_weighted_errors"][
                    "fn_rate_top_value_decile"
                ],
            ),
            "direction_sanity_check_test_passed": (
                1.0 if direction_sanity_check_test["passed"] else 0.0
            ),
        }
        top_k_shap = int(cfg.error_analysis.top_k_dependence_features)
        for row in shap_diag["importance_rows"][:top_k_shap]:
            feature_key = _sanitize_metric_name(str(row["feature"]))
            scalar_metrics[f"shap_importance_{feature_key}"] = cast(
                float, row["mean_abs_shap"]
            )
        mlflow.log_metrics(scalar_metrics, model_id=model_id, dataset=test_dataset)

        mlflow.set_tag(
            "direction_sanity_check_test",
            "pass" if direction_sanity_check_test["passed"] else "fail",
        )

        mlflow.log_dict(error_analysis_payload, "error_analysis.json")
        mlflow.log_dict(
            {
                "dev_oof_top_fnr_cohorts": concentration["concentration_fnr_rows"],
                "dev_oof_top_fpr_cohorts": concentration["concentration_fpr_rows"],
            },
            "slices/error_concentration.json",
        )

        for path in figure_paths.values():
            mlflow.log_artifact(str(path), artifact_path="figures")

        with tempfile.TemporaryDirectory() as tmp_dir:
            shap_values_path = Path(tmp_dir) / "shap_values.parquet"
            shap_df = pd.DataFrame(
                shap_diag["shap_values"], columns=shap_diag["feature_names"]
            )
            shap_df.insert(0, "customerid", test_predictions["customerid"].to_numpy())
            shap_df["base_value"] = shap_diag["base_value"]
            shap_df.to_parquet(shap_values_path, index=False)
            mlflow.log_artifact(str(shap_values_path))

    return error_analysis_run_id


def _write_error_analysis_reports(
    model_version: str,
    error_analysis_run_id: str,
    error_analysis_payload: dict[str, Any],
    reports_dir: Path,
    cfg: DictConfig,
) -> None:
    """Write reports/error_analysis_receipt.json and mirror error_analysis.json to reports/.

    register.py, not this module, tags the model version with
    error_analysis_run_id (B1 moves every model-version tag write into
    register.py, since minting now happens after review) — this receipt is
    register.py's bootstrap pointer to this cycle's error_analysis run, the
    same first-invocation role calibrate_receipt.json already plays.
    """
    write_error_analysis_receipt(model_version, error_analysis_run_id, cfg)

    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "error_analysis.json", "w", encoding="utf-8") as f:
        json.dump(error_analysis_payload, f, indent=2, default=str)


def run_error_analysis_step(
    run_id: str,
    model_version: str,
    model_uri: str,
    cfg: DictConfig,
    raw_partition_override: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Run the full error-diagnosis pass and write reports/error_analysis.json.

    `raw_partition_override` is `None` by default so the rare/cold-start path
    is unaffected; a routine reserve-
    driven cycle passes `models/train/common.py::load_training_pool_bundle`'s
    own `raw_partition` so the dev-OOF SHAP lookup sources matured reserve
    customers' engineered feature values from `training_pool`, not their
    frozen `customers_raw` snapshot — see `_features_by_customer_id`'s own
    docstring.

    Takes `run_id`/`model_version`/`model_uri` already resolved by the
    caller (utils.mlflow.resolve_model_identifier — an explicit override,
    never an alias, or calibrate.py's receipt), checks threshold provenance
    (a stale threshold_validation.json would otherwise silently derive the
    near-miss band from the wrong model's CI), loads the champion, and reads
    the sealed-test and dev-OOF prediction artifacts — the only route this
    module reaches evaluation data through.

    Computes SHAP once on the sealed-test rows (`_compute_shap_diagnostics`)
    and derives every SHAP-based diagnostic from that single call: global
    importance, per-feature dependence, binary-feature effects, the reported-
    only `direction_sanity_check_test` re-audit of threshold.py's own V3
    pre-seal audit set (never a second gate — see that function's docstring),
    and — via `_compute_cohort_gap` — the FN-vs-TP and FP-vs-TN signed-SHAP
    cohort gaps (each ranked and sized independently; see that function's
    docstring for why). The annual-contract Monthly Charges subgroup finding
    (`_annual_contract_subgroup_finding`) and the illustrative FN/FP/TP/TN
    waterfall cases (`_compute_illustrative_cases`) reuse the same SHAP call.

    Runs the error-concentration scan on dev OOF only, twice — ranked by FNR
    and by FPR (`_compute_error_concentration`). Renders all figures
    (`_render_error_analysis_figures`), logs its own `error_analysis` MLflow
    run (a sibling of evaluate.py's `evaluation` run), and writes
    reports/error_analysis.json — register.py aborts without it.

    Each of the three top-k cutoffs used above (`top_k_dependence_features`,
    `top_k_cohort_gap_features`, `top_k_cohort_gap_features_fp_tn`) is
    checked against its own ranking via `explain.check_top_k_elbow`, with a
    warning — never a raise — if the plateau-then-elbow shape that justified
    it has drifted: none of these are re-derived when the champion changes,
    so a stale cutoff would otherwise silently narrow or widen a chart's
    feature set with nothing to flag it.
    """
    loaded = _load_error_analysis_inputs(
        run_id, model_version, model_uri, cfg, raw_partition_override
    )
    run_id = loaded["run_id"]
    base_threshold = loaded["base_threshold"]
    y_test, proba_test = loaded["y_test"], loaded["proba_test"]
    y_dev, proba_dev = loaded["y_dev"], loaded["proba_dev"]
    ea_cfg = cfg.error_analysis

    # Loaded before the SHAP pass, not after: direction_sanity_check_test
    # re-audits exactly the feature set threshold.py's V3 pre-seal screen
    # derived on dev (direction_check_feature_names), so that set must be in
    # hand before _compute_shap_diagnostics can build the dependence grid.
    dev_oof_diagnostics = load_dev_oof_diagnostics(run_id, cfg)

    shap_diag = _compute_shap_diagnostics(
        loaded["model"],
        loaded["test_features_df"],
        loaded["committed_features"],
        list(dev_oof_diagnostics["direction_check_feature_names"]),
        ea_cfg,
    )
    shap_values, feature_names = shap_diag["shap_values"], shap_diag["feature_names"]

    fn_mask = (y_test == 1) & (proba_test < base_threshold)
    tp_mask = (y_test == 1) & (proba_test >= base_threshold)
    fn_tp_gap = _compute_cohort_gap(
        shap_values,
        feature_names,
        fn_mask,
        tp_mask,
        int(ea_cfg.top_k_cohort_gap_features),
        "top_k_cohort_gap_features_elbow_drifted",
    )

    # Mirrors the FN-vs-TP pair for the two actual-non-churner outcomes: does a
    # false alarm's SHAP profile look like a real churner's (confident
    # failure), or is it a diluted push that only just crossed t*? Does NOT
    # reuse top_k_cohort_gap — this axis has its own plateau/elbow shape (see
    # top_k_cohort_gap_features_fp_tn's config comment) and it lands on a
    # different rank, so sharing one knob would silently mis-cut one pair.
    fp_mask = (y_test == 0) & (proba_test >= base_threshold)
    tn_mask = (y_test == 0) & (proba_test < base_threshold)
    fp_tn_gap = _compute_cohort_gap(
        shap_values,
        feature_names,
        fp_mask,
        tn_mask,
        int(ea_cfg.top_k_cohort_gap_features_fp_tn),
        "top_k_cohort_gap_features_fp_tn_elbow_drifted",
    )

    subgroup = _annual_contract_subgroup_finding(
        shap_diag["Xt_test"], shap_values, shap_diag["feature_index"], fn_mask, tn_mask
    )

    illustrative_bundle = _compute_illustrative_cases(
        y_test,
        proba_test,
        base_threshold,
        shap_values,
        shap_diag["base_value"],
        feature_names,
        shap_diag["Xt_test"],
        ea_cfg,
    )

    error_profiles = _compute_error_profiles(
        y_test,
        proba_test,
        base_threshold,
        loaded["test_features_df"],
        loaded["validation_payload"],
        ea_cfg,
    )

    concentration = _compute_error_concentration(
        y_dev, proba_dev, base_threshold, loaded["dev_features_df"], ea_cfg
    )

    error_analysis_payload = _assemble_error_analysis_payload(
        model_version,
        run_id,
        base_threshold,
        shap_diag,
        fn_tp_gap,
        fp_tn_gap,
        subgroup,
        illustrative_bundle,
        error_profiles,
        concentration,
    )

    masks = {
        "fn_mask": fn_mask,
        "tp_mask": tp_mask,
        "fp_mask": fp_mask,
        "tn_mask": tn_mask,
    }
    figure_paths = _render_error_analysis_figures(
        shap_diag,
        fn_tp_gap,
        fp_tn_gap,
        masks,
        subgroup,
        illustrative_bundle,
        error_profiles,
        cfg,
    )

    error_analysis_run_id = _log_error_analysis_run(
        model_version,
        loaded,
        shap_diag,
        concentration,
        error_analysis_payload,
        figure_paths,
        cfg,
    )

    _write_error_analysis_reports(
        model_version,
        error_analysis_run_id,
        error_analysis_payload,
        loaded["reports_dir"],
        cfg,
    )

    direction_sanity_check_test = shap_diag["direction_sanity_check_test"]
    logger.info(
        "error_analysis_step_done",
        run_id=run_id,
        error_analysis_run_id=error_analysis_run_id,
        model_version=model_version,
        direction_sanity_check_test_passed=direction_sanity_check_test["passed"],
    )

    return {
        "error_analysis_run_id": error_analysis_run_id,
        "model_version": model_version,
        "error_analysis": error_analysis_payload,
    }


if __name__ == "__main__":
    import sys

    import pandera as pa
    from dotenv import load_dotenv

    from telco_churn.utils.logging import configure_logging
    from telco_churn.utils.paths import activate_config, compose_config

    load_dotenv()
    configure_logging()

    try:
        cfg = compose_config(overrides=sys.argv[1:] or None)
        activate_config(cfg)
        cli_run_id, cli_model_version, cli_model_uri = resolve_model_identifier(
            cfg.error_analysis.run_id, cfg.error_analysis.model_version, cfg
        )
        result = run_error_analysis_step(
            cli_run_id, cli_model_version, cli_model_uri, cfg
        )
        logger.info(
            "error_analysis_step_done",
            model_version=result["model_version"],
            error_analysis_run_id=result["error_analysis_run_id"],
        )
    except FileNotFoundError as e:
        logger.error("error_analysis_data_not_found", error=str(e), exc_info=True)
        sys.exit(1)
    except pa.errors.SchemaError as e:
        logger.error("error_analysis_data_schema_invalid", error=str(e), exc_info=True)
        sys.exit(1)
    except ValueError as e:
        logger.error("error_analysis_invalid", error=str(e), exc_info=True)
        sys.exit(1)
    except AssertionError as e:
        logger.error("error_analysis_assertion_failed", error=str(e), exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error("error_analysis_failed", error=str(e), exc_info=True)
        sys.exit(1)

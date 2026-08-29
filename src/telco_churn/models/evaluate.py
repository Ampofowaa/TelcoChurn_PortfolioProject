"""Sealed-test evaluation for the rare/cold-start regime — a thin `__main__`
wrapper over `models/sealed_test.py`'s dataset-agnostic scoring/gate
primitives (Phase 10a-ii).

CLAUDE.md's modelling invariant: X_test/y_test are imported and used in
exactly two places — models/sealed_test.py (the sole importer of the test
side of telco_churn.data.split.partition()) and this module's own thin
wrapper over it. Every other module reaches test data solely through
persisted artifacts (reports/test_predictions.parquet), never through
telco_churn.data.split directly.

Resolves the model being evaluated by explicit run_id/version, never by
alias: models:/telco-churn-pipeline@challenger is a moving pointer, and a
future full-data retrain path could leave it pointing at a model trained on
the test set — which would then silently produce excellent, meaningless
sealed-test metrics. This module logs the resolved version into
reports/metrics.json so the metrics stay attributable to a specific artifact.

load_incumbent_proba/resolve_incumbent_summary/resolve_evaluation_champion
stay here rather than moving to models/sealed_test.py — they power only the
rare comparative regime's historical-alignment machinery (gates on an exact
customerid-set match against the champion's historical
test_predictions.parquet), which pipelines/performance_check.py (Phase 10b)
never needs, since its comparison cohort is a fresh reserve cohort every
cycle, not the sealed test set.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import mlflow
import mlflow.artifacts
import mlflow.tracking
import numpy as np
import pandas as pd
from mlflow.data.pandas_dataset import from_pandas as mlflow_dataset_from_pandas
from numpy.typing import NDArray
from omegaconf import DictConfig

from telco_churn.features.accessor import features_sha256
from telco_churn.features.build import TARGET_COL
from telco_churn.models.artifacts import (
    committed_features_from_manifest,
    load_fitted_model,
    load_threshold_validation,
    load_training_manifest,
    resolve_champion_version,
)
from telco_churn.models.diagnostics import (
    FAIRNESS_AXES,
    ROBUSTNESS_AXES,
    demographic_parity_difference_by_axis,
    equal_opportunity_difference_by_axis,
    sliced_calibration,
    sliced_decision_rates,
    sliced_ranking_metrics,
)
from telco_churn.models.economics import capacity_budget_check, ev_by_k
from telco_churn.models.gate import (
    check_threshold_provenance,
    check_threshold_screen_passed,
)
from telco_churn.models.plots import pr_curve_points, roc_curve_points
from telco_churn.models.policy_config import (
    costs_config_hash,
    load_costs_config,
    load_model_promotion_bars,
    load_policy_thresholds,
    resolve_policy_scenarios,
    resolve_policy_thresholds_by_scenario,
)
from telco_churn.models.sealed_test import (
    load_test_customer_ids,
    load_test_features,
    load_test_segment_lookup,
    sealed_test_business_impact,
    sealed_test_calibration_report,
    sealed_test_classification_report,
    sealed_test_decile_lift,
    sealed_test_fixed_recall_profile,
    sealed_test_promotion_decision,
    sealed_test_ranking_metrics,
    sealed_test_sensitivity_analysis,
    sliced_business_impact,
)
from telco_churn.utils.hashing import content_hash
from telco_churn.utils.logging import get_logger
from telco_churn.utils.mlflow import (
    ensure_experiment_metadata,
    resolve_logged_model_id,
    resolve_model_identifier,
    resolve_tracking_uri,
    set_run_description,
    write_eval_receipt,
)
from telco_churn.utils.paths import get_project_root

__all__ = [
    "demographic_parity_difference_by_axis",
    "equal_opportunity_difference_by_axis",
    "flatten_metrics_summary",
    "load_incumbent_proba",
    "resolve_evaluation_champion",
    "resolve_incumbent_summary",
    "resolve_logged_model_id",
    "run_evaluation_step",
    "sliced_calibration",
    "sliced_decision_rates",
    "sliced_ranking_metrics",
]

# ROBUSTNESS_AXES/FAIRNESS_AXES (ANALYSIS.md §0's V1/V2/V2b surface) live in
# diagnostics.py — shared with threshold.py's dev-OOF screen, which computes
# V1/V2/V2b on the dev-OOF surface; this module uses the same axes for its
# own sealed-test reporting slices.

# Below this many rows, or with only one class present, a segment's own
# calibration slope isn't estimable — skipped rather than reported as noise.
_MIN_SLICE_SIZE = 10

logger = get_logger(__name__)

_RUN_DESCRIPTION = (
    "Sealed-test evaluation for one registered model version - PR-AUC, "
    "recall/precision/F1 and confusion matrix per cost scenario, "
    "calibration transfer (BSS/ECE/slope), business-impact and EV "
    "sensitivity analysis, sliced fairness/robustness metrics. Computes the "
    "promotion gate verdict via gate.py::decide_promotion and persists "
    "promotion_decision.json. The sealed test set is touched here, and only "
    "here."
)


def load_incumbent_proba(
    champion_version: str,
    champion_eval_run_id: str,
    champion_data_content_hash: str,
    candidate_customer_ids: pd.Series,
    candidate_y_test: pd.Series,
    cfg: DictConfig,
) -> NDArray[np.float64]:
    """Read the champion's own historical sealed-test predictions rather than
    re-scoring it live.

    The champion's test_predictions.parquet (customerid, y_true, p_hat) was
    logged onto its own eval run — champion_eval_run_id, resolved by the
    caller via resolve_incumbent_summary — at the moment *it* was evaluated
    and promoted: an immutable, already-materialized artifact, not something
    this cycle recomputes. Reading it instead of loading the champion's
    fitted pipeline and re-running inference removes two live-coupling costs
    at once: no second model deserialization/inference pass every cycle, and
    no exposure to library/environment drift silently perturbing the
    champion's historical numbers between cycles — a second-order
    reproducibility gap resolve_evaluation_champion's explicit version pin
    doesn't by itself close, since pinning *which* version is champion still
    leaves *how it's scored* live unless the scoring itself is also frozen.

    Checked against the current processed-features file's own content hash
    first, before any download: customerid/label agreement alone can't rule
    out the feature pipeline having changed under an unchanged customer set
    (e.g. a new engineered column added to every row) — a candidate scored
    on a different feature space than the champion was is not a fair
    comparison even though nothing about the test partition's membership
    moved. Raises loudly, naming both hashes, rather than silently comparing
    across feature spaces.

    Reindexed onto candidate_customer_ids's exact row order rather than
    trusted to already match — the two vectors come from different
    evaluate.py runs (this cycle's and the champion's own), so alignment
    can't be assumed from row order alone the way it could when both were
    scored together in one process. Raises loudly, naming the champion
    version and its eval run, if the champion's recorded test customer set
    isn't identical to the candidate's (the canonical split moved since the
    champion was last evaluated — no fair paired comparison is possible
    without re-evaluating the champion against the current split) or if any
    shared customerid's recorded label disagrees (a same-customer label
    change should never happen on this project's static dataset and signals
    a deeper data-integrity problem, not a stale split).
    """
    candidate_data_content_hash = features_sha256()
    if candidate_data_content_hash != champion_data_content_hash:
        raise RuntimeError(
            f"Champion model version {champion_version!r}'s historical "
            f"sealed-test predictions (eval run {champion_eval_run_id!r}) "
            f"were computed against a different processed-features file "
            f"(data_content_hash {champion_data_content_hash!r}) than the "
            f"one on disk now ({candidate_data_content_hash!r}) — the "
            "feature pipeline has changed since the champion was last "
            "evaluated, so its historical predictions aren't a fair "
            "comparison even if the customer set still matches. Re-run "
            "models.evaluate for the champion's own run_id/model_version "
            "against the current features file, or pin "
            "evaluate.champion_version=none to compare cold-start instead."
        )

    mlflow.set_tracking_uri(resolve_tracking_uri(str(cfg.mlflow.tracking_uri)))
    local_path = mlflow.artifacts.download_artifacts(
        artifact_uri=f"runs:/{champion_eval_run_id}/test_predictions.parquet"
    )
    champion_predictions = pd.read_parquet(local_path)

    champion_ids = set(champion_predictions["customerid"])
    candidate_ids = set(candidate_customer_ids)
    if champion_ids != candidate_ids:
        raise RuntimeError(
            f"Champion model version {champion_version!r}'s historical "
            f"sealed-test predictions (eval run {champion_eval_run_id!r}) "
            "cover a different customer set than the current sealed test "
            "partition — the canonical split has moved since the champion "
            "was last evaluated. Re-run models.evaluate for the champion's "
            "own run_id/model_version against the current split, or pin "
            "evaluate.champion_version=none to compare cold-start instead."
        )

    aligned = champion_predictions.set_index("customerid").loc[candidate_customer_ids]
    if not np.array_equal(
        aligned["y_true"].to_numpy(dtype=np.int64),
        candidate_y_test.to_numpy(dtype=np.int64),
    ):
        raise RuntimeError(
            f"Champion model version {champion_version!r}'s historical "
            f"sealed-test labels (eval run {champion_eval_run_id!r}) "
            "disagree with the candidate's for at least one shared "
            "customerid — the same customer carries different churn labels "
            "between the two evaluation cycles, which should never happen "
            "on a static dataset."
        )

    proba: NDArray[np.float64] = aligned["p_hat"].to_numpy(dtype=np.float64)
    return proba


def resolve_incumbent_summary(
    champion_version: str, cfg: DictConfig
) -> dict[str, float | str]:
    """Read the champion's own gate-criteria tags and cost-config provenance,
    for side-by-side reporting.

    register.py tags every version it processes with the four gate-criteria
    tags and eval_run_id, unconditionally and regardless of the eventual
    promotion outcome (sourced from reports/eval_receipt.json on its first
    pass, from the tag itself thereafter — see register.py's own tag-
    resolution helpers). A champion is by construction a version register.py
    promoted, so it always carries them. Reading tags off the already-
    resolved version number, never re-reading the `champion` alias, matches
    load_incumbent_proba's own rule.

    costs_config_hash is a run tag, not a model-version tag (set on the
    evaluation run itself, not the registry entry — see _log_evaluation_run),
    so it's fetched via the version's own eval_run_id. Included so a reviewer
    can compare it against the candidate's own costs_config_hash tag: the
    recall-delta guardrail runs both models through the *current* t*, but
    this incumbent number was tagged under whatever costs.yaml was live at
    its own promotion — a different hash means it was never actually served
    under today's cost assumptions.

    data_content_hash is likewise a run tag, fetched the same way. It is also
    load_incumbent_proba's own input — that function refuses to compare
    against a champion whose historical predictions were computed against a
    different processed-features file, since a customerid/label match alone
    can't rule out the feature pipeline having changed under the same
    customer set.

    The returned eval_run_id is also load_incumbent_proba's own input — the
    caller resolves it once here rather than each function re-deriving it
    from a second get_model_version call.
    """
    mlflow.set_tracking_uri(resolve_tracking_uri(str(cfg.mlflow.tracking_uri)))
    registered_model_name = str(cfg.mlflow.registered_model_name)
    client = mlflow.tracking.MlflowClient()
    tags = client.get_model_version(registered_model_name, champion_version).tags

    missing = [
        key
        for key in (
            "test_pr_auc",
            "test_recall",
            "test_brier",
            "test_calibration_slope",
            "eval_run_id",
        )
        if key not in tags
    ]
    if missing:
        raise RuntimeError(
            f"Champion model version {champion_version!r} is missing gate-"
            f"criteria tags {missing} — every version evaluate.py has scored "
            "should carry these. Re-run models.evaluate for that version."
        )

    eval_run_tags = client.get_run(tags["eval_run_id"]).data.tags
    missing_run_tags = [
        key
        for key in ("costs_config_hash", "data_content_hash")
        if key not in eval_run_tags
    ]
    if missing_run_tags:
        raise RuntimeError(
            f"Champion model version {champion_version!r}'s evaluation run "
            f"{tags['eval_run_id']!r} is missing tag(s) {missing_run_tags} "
            "— every evaluation run sets them. Re-run models.evaluate for "
            "that version."
        )

    return {
        "version": champion_version,
        "pr_auc": float(tags["test_pr_auc"]),
        "recall": float(tags["test_recall"]),
        "brier": float(tags["test_brier"]),
        "calibration_slope": float(tags["test_calibration_slope"]),
        "costs_config_hash": eval_run_tags["costs_config_hash"],
        "data_content_hash": eval_run_tags["data_content_hash"],
        "eval_run_id": tags["eval_run_id"],
    }


def resolve_evaluation_champion(cfg: DictConfig) -> str | None:
    """Resolve the incumbent champion version to compare the candidate against.

    An explicit `evaluate.champion_version` override is read verbatim and
    never touches the `champion` alias — the alias is externally-mutable,
    moving registry state a DVC `cmd` string cannot declare as a dep
    (PROJECT_PLAN.md's undeclared-dependency note on the Phase 8 DAG), so a
    reproducible invocation resolves it once beforehand and passes the
    version in. The literal string "none" pins the cold-start regime
    explicitly, even if a champion happens to be live. Omitting the override
    (the config default) falls back to resolve_champion_version's live alias
    read — the right behaviour for interactive/notebook use, where reading
    "whichever champion is live right now" is exactly what's wanted.
    """
    override = cfg.evaluate.champion_version
    if override is None:
        return resolve_champion_version(cfg)
    if str(override).strip().lower() == "none":
        return None
    return str(override)


# ---------------------------------------------------------------------------
# Step 6: MLflow orchestration
# ---------------------------------------------------------------------------


def _save_pr_curve_plot(
    pr_points: list[dict[str, float]],
    prevalence: float,
    thresholds: dict[str, float],
    path: Path,
) -> None:
    """PR curve with the prevalence baseline and all three scenario thresholds marked."""
    fig, ax = plt.subplots(figsize=(6, 5))
    recalls = [p["recall"] for p in pr_points]
    precisions = [p["precision"] for p in pr_points]
    ax.plot(recalls, precisions, label="model", color="tab:blue")
    ax.axhline(
        prevalence,
        color="gray",
        linestyle="--",
        label=f"prevalence = {prevalence:.3f}",
    )
    for name, t in thresholds.items():
        idx = min(
            range(len(pr_points)), key=lambda i: abs(pr_points[i]["threshold"] - t)
        )
        ax.scatter(
            [pr_points[idx]["recall"]],
            [pr_points[idx]["precision"]],
            zorder=3,
            label=f"{name} t*={t:.3f}",
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve — sealed test")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _save_roc_curve_plot(roc_points: list[dict[str, float]], path: Path) -> None:
    """ROC curve — a labelled diagnostic, never the headline (ANALYSIS.md §0:
    ROC-AUC is optimistic under this project's class imbalance)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(
        [p["fpr"] for p in roc_points],
        [p["tpr"] for p in roc_points],
        label="model",
        color="tab:blue",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve — sealed test (diagnostic; not gated)")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _save_classification_report_plot(
    classification_rows: list[dict[str, Any]], path: Path
) -> None:
    """One panel per scenario: confusion matrix (counts + row-normalised %) beside
    positive-class precision/recall/F1, support, and contact rate — the honest
    replacement for sklearn.classification_report (ANALYSIS.md §0/plots.py)."""
    n = len(classification_rows)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    axes_list = [axes] if n == 1 else list(axes)
    for ax, row in zip(axes_list, classification_rows, strict=True):
        matrix = np.array([[row["tp"], row["fn"]], [row["fp"], row["tn"]]])
        ax.imshow(matrix, cmap="Blues")
        labels = [
            [
                f"TP={row['tp']:.0f}\n({row['tp_pct']:.1%})",
                f"FN={row['fn']:.0f}\n({row['fn_pct']:.1%})",
            ],
            [
                f"FP={row['fp']:.0f}\n({row['fp_pct']:.1%})",
                f"TN={row['tn']:.0f}\n({row['tn_pct']:.1%})",
            ],
        ]
        for i in range(2):
            for j in range(2):
                ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pred. churn", "Pred. stay"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Actual churn", "Actual stay"])
        ax.set_title(
            f"{row['scenario']} (t={row['threshold']:.3f})\n"
            f"recall={row['recall']:.2f} precision={row['precision']:.2f} "
            f"f1={row['f1']:.2f}\ncontact_rate={row['contact_rate']:.1%}",
            fontsize=9,
        )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _save_reliability_plot_test(bins: list[dict[str, float]], path: Path) -> None:
    """Sealed-test reliability diagram — a distinct filename from Phase 6's
    reliability_diagram.png (the dev-OOF one), so the before/after comparison
    is never destroyed by an overwrite."""
    fig, ax = plt.subplots(figsize=(5, 5))
    means = [b["mean_predicted"] for b in bins]
    obs = [b["observed_frequency"] for b in bins]
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    ax.plot(means, obs, marker="o", label="sealed test", color="tab:blue")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed churn frequency")
    ax.set_title("Reliability diagram — sealed test")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _save_ev_by_budget_plot(
    ev_curves: dict[str, list[dict[str, float]]],
    thresholds: dict[str, float],
    contact_capacity: float,
    path: Path,
) -> None:
    """EV-vs-K budget curve, one line per scenario, with t*'s K, the
    EV-maximising K, and the capacity limit marked — "the single most
    business-legible artifact this project can produce" (PROJECT_PLAN.md):
    at any budget, the business reads off its own number, no new model or
    threshold required.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, rows in ev_curves.items():
        ks = [row["k"] for row in rows]
        evs = [row["ev_cumulative"] for row in rows]
        (line,) = ax.plot(ks, evs, label=name)
        t = thresholds[name]
        # rows are k-ascending / threshold-descending; the first row whose
        # threshold has dropped to (or below) t* is where contacting "iff
        # proba >= t*" lands on this curve.
        t_star_row = next((row for row in rows if row["threshold"] <= t), rows[-1])
        ax.axvline(t_star_row["k"], color=line.get_color(), linestyle=":", alpha=0.6)
        best = max(rows, key=lambda row: row["ev_cumulative"])
        ax.scatter(
            [best["k"]],
            [best["ev_cumulative"]],
            marker="*",
            s=90,
            color=line.get_color(),
            zorder=3,
        )
    ax.axvline(
        contact_capacity,
        color="black",
        linestyle="--",
        label=f"capacity = {contact_capacity:.0f}",
    )
    ax.set_xlabel("Number contacted (K)")
    ax.set_ylabel("Cumulative expected value ($)")
    ax.set_title(
        "EV vs. number contacted — sealed test (dotted = t*, star = EV-maximising K)"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _save_breakeven_heatmap_plot(
    twoway_rows: list[dict[str, float]],
    retention_rate_values: list[float],
    cost_values: list[float],
    path: Path,
) -> None:
    """The r x c break-even grid with the EV = 0 contour — the chart that tells
    a stakeholder "at a $20 contact cost, this campaign is profitable as long
    as we retain at least X% of the churners we call" (PROJECT_PLAN.md),
    read straight off the picture rather than computed by hand.
    """
    grid = np.array(
        [
            [
                next(
                    row["ev"]
                    for row in twoway_rows
                    if row["retention_rate"] == r and row["cost"] == c
                )
                for c in cost_values
            ]
            for r in retention_rate_values
        ]
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    mesh = ax.pcolormesh(
        cost_values, retention_rate_values, grid, shading="auto", cmap="RdYlGn"
    )
    ax.contour(
        cost_values,
        retention_rate_values,
        grid,
        levels=[0.0],
        colors="black",
        linewidths=2,
    )
    fig.colorbar(mesh, ax=ax, label="Expected value per customer ($)")
    ax.set_xlabel(
        "Cost per customer contacted (c) — outreach + retention-offer cost, $"
    )
    ax.set_ylabel("Retention rate (r)")
    ax.set_title("Break-even frontier — sealed test (base scenario)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _save_tornado_plot(tornado_rows: list[dict[str, Any]], path: Path) -> None:
    """Inputs ranked by |ev_high - ev_low| under a symmetric perturbation —
    makes ANALYSIS.md §0's claim that r dominates visible at a glance instead
    of asking the reader to take it on trust.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    params = [cast(str, row["param"]) for row in tornado_rows]
    lows = [cast(float, row["ev_low"]) for row in tornado_rows]
    highs = [cast(float, row["ev_high"]) for row in tornado_rows]
    base_ev = cast(float, tornado_rows[0]["ev_base"])
    y_pos = np.arange(len(params))
    for i, (lo, hi) in enumerate(zip(lows, highs, strict=True)):
        ax.barh(i, hi - lo, left=min(lo, hi), color="tab:blue", alpha=0.7)
    ax.axvline(
        base_ev,
        color="gray",
        linestyle="--",
        linewidth=0.8,
        label=f"base EV/customer = {base_ev:.0f}",
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(params)
    ax.invert_yaxis()
    ax.set_xlabel("Expected value per customer ($)")
    ax.set_title("Sensitivity tornado — sealed test (base scenario)")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _save_gains_lift_plot(decile_rows: list[dict[str, float]], path: Path) -> None:
    """Cumulative gains curve and per-decile lift bars — "how much churn does
    the top-k% capture?", threshold-free and complementary to PR-AUC.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    deciles = [row["decile"] for row in decile_rows]
    capture = [row["cumulative_capture_rate"] for row in decile_rows]
    perfect = [row["perfect_capture_rate"] for row in decile_rows]
    ax1.plot(
        [0, *deciles], [0, *perfect], linestyle="--", color="green", label="perfect"
    )
    ax1.plot([0, *deciles], [0, *capture], marker="o", label="model")
    ax1.plot([0, 10], [0, 1], linestyle="--", color="gray", label="random")
    ax1.set_xlabel("Decile (top-K x 10%)")
    ax1.set_ylabel("Cumulative share of churners captured")
    ax1.set_title("Cumulative gains — sealed test")
    ax1.legend()

    lifts = [row["lift"] for row in decile_rows]
    ax2.bar(deciles, lifts, color="tab:blue")
    ax2.axhline(1.0, color="gray", linestyle="--", label="random")
    ax2.set_xlabel("Decile")
    ax2.set_ylabel("Lift over base rate")
    ax2.set_title("Decile lift — sealed test")
    ax2.legend()

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _load_and_score_candidate(
    run_id: str, model_version: str, model_uri: str, cfg: DictConfig
) -> dict[str, Any]:
    """Check threshold provenance/screen status, and score the sealed test set once."""
    mlflow.set_tracking_uri(resolve_tracking_uri(str(cfg.mlflow.tracking_uri)))
    registered_model_name = str(cfg.mlflow.registered_model_name)

    validation_payload = load_threshold_validation(run_id, cfg)
    logged_model_id = resolve_logged_model_id(model_version, cfg)
    check_threshold_provenance(validation_payload, logged_model_id)
    check_threshold_screen_passed(validation_payload)

    manifest = load_training_manifest(run_id, cfg)
    committed_features = committed_features_from_manifest(manifest)
    model = load_fitted_model(model_uri, cfg)

    X_test, y_test = load_test_features(committed_features)
    proba: NDArray[np.float64] = model.predict_proba(X_test)[:, 1]
    customer_ids = load_test_customer_ids()

    return {
        "registered_model_name": registered_model_name,
        "run_id": run_id,
        "X_test": X_test,
        "y_test": y_test,
        "proba": proba,
        "customer_ids": customer_ids,
    }


def _load_policy_context(cfg: DictConfig) -> dict[str, Any]:
    """Load the shipped policy thresholds/scenarios and the evaluate-wide bootstrap settings."""
    policy = load_policy_thresholds(cfg)
    scenarios = resolve_policy_scenarios(policy)
    thresholds = resolve_policy_thresholds_by_scenario(policy)
    return {
        "scenarios": scenarios,
        "thresholds": thresholds,
        "base_scenario": scenarios["base"],
        "base_threshold": thresholds["base"],
        "n_bootstrap": int(cfg.evaluate.n_bootstrap),
        "random_state": int(cfg.evaluate.random_state),
    }


def _compute_core_test_metrics(
    y_test: pd.Series,
    proba: NDArray[np.float64],
    policy_ctx: dict[str, Any],
    cfg: DictConfig,
) -> dict[str, Any]:
    """Compute ranking, per-scenario classification, fixed-recall, calibration, decile, and business-impact blocks."""
    thresholds, scenarios = policy_ctx["thresholds"], policy_ctx["scenarios"]
    n_bootstrap, random_state = policy_ctx["n_bootstrap"], policy_ctx["random_state"]

    ranking_metrics = sealed_test_ranking_metrics(
        y_test, proba, n_bootstrap, random_state
    )
    classification_rows = sealed_test_classification_report(
        y_test, proba, thresholds, n_bootstrap, random_state
    )
    fixed_recall_rows = sealed_test_fixed_recall_profile(
        y_test, proba, [float(r) for r in cfg.training_setup.fixed_recall_thresholds]
    )
    calibration_report = sealed_test_calibration_report(
        y_test, proba, cfg, n_bootstrap, random_state
    )
    decile_rows = sealed_test_decile_lift(y_test, proba)
    business_impact = sealed_test_business_impact(
        y_test, proba, scenarios, thresholds, n_bootstrap, random_state
    )
    if not business_impact["parameter_spread_dominates_sampling"]:
        logger.warning(
            "cost_scenario_spread_too_narrow",
            ev_spread=business_impact["ev_spread"],
            widest_within_scenario_ci_width=business_impact[
                "widest_within_scenario_ci_width"
            ],
            hint=(
                "EV bracket across cost scenarios is narrower than the widest "
                "within-scenario bootstrap CI — configs/costs.yaml's scenarios "
                "are too narrow to express the real uncertainty in r, hiding "
                "the EV bracket's dominant error source."
            ),
        )

    return {
        "ranking_metrics": ranking_metrics,
        "classification_rows": classification_rows,
        "fixed_recall_rows": fixed_recall_rows,
        "calibration_report": calibration_report,
        "decile_rows": decile_rows,
        "business_impact": business_impact,
    }


def _compute_sensitivity_block(
    y_test: pd.Series,
    proba: NDArray[np.float64],
    policy_ctx: dict[str, Any],
    cfg: DictConfig,
) -> dict[str, Any]:
    """Load costs config and compute the sealed-test sensitivity suite (tornado + break-even sweep)."""
    base_scenario, base_threshold = (
        policy_ctx["base_scenario"],
        policy_ctx["base_threshold"],
    )

    costs_cfg = load_costs_config(get_project_root() / str(cfg.paths.costs_config))
    retention_rate_values = [float(r) for r in costs_cfg.retention_rate_sweep]
    cost_values = [base_scenario.cost * m for m in (0.5, 1.0, 1.5, 2.0)]
    sensitivity = sealed_test_sensitivity_analysis(
        y_test,
        proba,
        base_scenario,
        base_threshold,
        retention_rate_values,
        cost_values,
        float(cfg.evaluate.tornado_pct_perturbation),
    )

    return {
        "costs_cfg": costs_cfg,
        "retention_rate_values": retention_rate_values,
        "cost_values": cost_values,
        "sensitivity": sensitivity,
    }


def _compute_sliced_diagnostics(
    y_test: pd.Series,
    proba: NDArray[np.float64],
    policy_ctx: dict[str, Any],
    cfg: DictConfig,
) -> dict[str, Any]:
    """Compute the sealed-test disaggregated slices and the fairness-difference summaries they feed."""
    base_scenario, base_threshold = (
        policy_ctx["base_scenario"],
        policy_ctx["base_threshold"],
    )
    n_bootstrap, random_state = policy_ctx["n_bootstrap"], policy_ctx["random_state"]

    all_axes = ROBUSTNESS_AXES + FAIRNESS_AXES
    test_segment_lookup = load_test_segment_lookup()
    test_ranking_slices = sliced_ranking_metrics(
        y_test, proba, test_segment_lookup, all_axes, n_bootstrap, random_state
    )
    test_decision_slices = sliced_decision_rates(
        y_test, proba, test_segment_lookup, all_axes, base_threshold
    )
    test_calibration_slices = sliced_calibration(
        y_test, proba, test_segment_lookup, all_axes, cfg, n_bootstrap, random_state
    )
    test_business_impact_slices = sliced_business_impact(
        y_test,
        proba,
        test_segment_lookup,
        FAIRNESS_AXES,
        base_scenario,
        base_threshold,
    )
    test_fairness_decision_rows = [
        row for row in test_decision_slices if row["axis"] in FAIRNESS_AXES
    ]
    test_equal_opportunity_by_axis = equal_opportunity_difference_by_axis(
        test_fairness_decision_rows
    )
    test_demographic_parity_by_axis = demographic_parity_difference_by_axis(
        test_fairness_decision_rows
    )
    test_equal_opportunity_diff = max(
        (v for v in test_equal_opportunity_by_axis.values() if not math.isnan(v)),
        default=float("nan"),
    )
    test_demographic_parity_diff = max(
        (v for v in test_demographic_parity_by_axis.values() if not math.isnan(v)),
        default=float("nan"),
    )

    return {
        "test_ranking_slices": test_ranking_slices,
        "test_decision_slices": test_decision_slices,
        "test_calibration_slices": test_calibration_slices,
        "test_business_impact_slices": test_business_impact_slices,
        "test_equal_opportunity_by_axis": test_equal_opportunity_by_axis,
        "test_demographic_parity_by_axis": test_demographic_parity_by_axis,
        "test_equal_opportunity_diff": test_equal_opportunity_diff,
        "test_demographic_parity_diff": test_demographic_parity_diff,
    }


def _compute_promotion_decision(
    run_id: str,
    y_test: pd.Series,
    proba: NDArray[np.float64],
    customer_ids: pd.Series,
    core_metrics: dict[str, Any],
    policy_ctx: dict[str, Any],
    cfg: DictConfig,
) -> dict[str, Any]:
    """Load the gate bars, resolve the incumbent, and call decide_promotion.

    Does not fetch threshold.py's dev-OOF diagnostics (V1/V2/V2b) — nothing
    here consumes them, and metrics.json no longer embeds a copy. Its sole
    reader, register.py's model card, fetches load_dev_oof_diagnostics(run_id,
    cfg) directly, off the same canonical MLflow artifact this run_id already
    points at — one owner, one read, not a chain of copies.

    The incumbent is resolved via resolve_evaluation_champion, not a bare
    resolve_champion_version(cfg) call — see that function's docstring for
    why the champion alias must be an explicit, caller-supplied override
    under `dvc repro` rather than a live read from inside this stage.

    incumbent_summary is resolved before incumbent_proba, not independently
    of it — both need the champion's eval_run_id and data_content_hash, and
    resolve_incumbent_summary already looks both up as part of reading the
    four gate-criteria tags, so load_incumbent_proba takes them as parameters
    rather than re-resolving either with a second registry round trip.
    """
    bars = load_model_promotion_bars(cfg)

    champion_version = resolve_evaluation_champion(cfg)
    incumbent_summary = (
        None
        if champion_version is None
        else resolve_incumbent_summary(champion_version, cfg)
    )
    incumbent_proba = (
        None
        if champion_version is None or incumbent_summary is None
        else load_incumbent_proba(
            champion_version,
            str(incumbent_summary["eval_run_id"]),
            str(incumbent_summary["data_content_hash"]),
            customer_ids,
            y_test,
            cfg,
        )
    )
    decision = sealed_test_promotion_decision(
        y_test,
        proba,
        incumbent_proba,
        core_metrics["ranking_metrics"],
        core_metrics["classification_rows"],
        core_metrics["calibration_report"],
        "base",
        bars,
        policy_ctx["n_bootstrap"],
        policy_ctx["random_state"],
    )

    return {
        "champion_version": champion_version,
        "incumbent_summary": incumbent_summary,
        "decision": decision,
    }


def _assemble_metrics_and_economics_payloads(
    model_version: str,
    run_id: str,
    y_test: pd.Series,
    proba: NDArray[np.float64],
    core_metrics: dict[str, Any],
    sliced: dict[str, Any],
    sensitivity_block: dict[str, Any],
    decision_result: dict[str, Any],
    policy_ctx: dict[str, Any],
) -> dict[str, Any]:
    """Assemble metrics.json, economics.json, and promotion_decision.json."""
    business_impact = core_metrics["business_impact"]
    decision = decision_result["decision"]

    metrics_payload: dict[str, Any] = {
        "model_version": model_version,
        "run_id": run_id,
        "champion_version": decision_result["champion_version"],
        "incumbent_summary": decision_result["incumbent_summary"],
        "ranking": core_metrics["ranking_metrics"],
        "classification": core_metrics["classification_rows"],
        "fixed_recall_profile": core_metrics["fixed_recall_rows"],
        "calibration": core_metrics["calibration_report"],
        "decile_lift": core_metrics["decile_rows"],
        "business_impact": business_impact,
        "sliced": {
            "test": {
                "ranking": sliced["test_ranking_slices"],
                "decision_rates": sliced["test_decision_slices"],
                "calibration": sliced["test_calibration_slices"],
                "business_impact": sliced["test_business_impact_slices"],
                "equal_opportunity_difference_by_axis": sliced[
                    "test_equal_opportunity_by_axis"
                ],
                "demographic_parity_difference_by_axis": sliced[
                    "test_demographic_parity_by_axis"
                ],
                "equal_opportunity_diff": sliced["test_equal_opportunity_diff"],
                "demographic_parity_diff": sliced["test_demographic_parity_diff"],
            },
        },
    }
    y_test_int = y_test.to_numpy(dtype=np.int64)
    ev_curves = {
        name: ev_by_k(proba, y_test_int, scenario)
        for name, scenario in policy_ctx["scenarios"].items()
    }

    costs_cfg = sensitivity_block["costs_cfg"]
    capacity_flags = capacity_budget_check(
        business_impact["scenarios"],
        float(costs_cfg.contact_capacity),
        float(costs_cfg.campaign_budget),
    )
    for name, flags in capacity_flags.items():
        if flags["over_capacity"] or flags["over_budget"]:
            logger.warning(
                "capacity_or_budget_exceeded",
                scenario=name,
                n_contacted=business_impact["scenarios"][name]["n_contacted"],
                contact_capacity=float(costs_cfg.contact_capacity),
                campaign_cost=business_impact["scenarios"][name]["campaign_cost"],
                campaign_budget=float(costs_cfg.campaign_budget),
                over_capacity=flags["over_capacity"],
                over_budget=flags["over_budget"],
                hint=(
                    "Implied contact count or spend at this scenario's shipped "
                    "threshold exceeds the retention team's operational limits — "
                    "the correct policy response is top-K-by-EV contact "
                    "selection (economics.json's ev_by_k), not a higher threshold."
                ),
            )

    economics_payload: dict[str, Any] = {
        "sensitivity": sensitivity_block["sensitivity"],
        "ev_by_k": ev_curves,
        "ev_treat_all_by_scenario": {
            name: row["ev_treat_all"]
            for name, row in business_impact["scenarios"].items()
        },
        "retention_rate_values_swept": sensitivity_block["retention_rate_values"],
        "cost_values_swept": sensitivity_block["cost_values"],
        "capacity_budget_check": capacity_flags,
    }
    promotion_decision_payload: dict[str, Any] = {
        **decision,
        "model_version": model_version,
        "metrics_content_hash": content_hash(metrics_payload),
    }

    return {
        "metrics_payload": metrics_payload,
        "ev_curves": ev_curves,
        "economics_payload": economics_payload,
        "promotion_decision_payload": promotion_decision_payload,
    }


def flatten_metrics_summary(
    metrics_payload: dict[str, Any],
    decision_payload: dict[str, Any],
    costs_hash: str,
) -> dict[str, Any]:
    """Flatten metrics.json + promotion_decision.json into the small, diffable cross-cycle surface.

    reports/metrics_summary.json is what a reviewer actually diffs across
    cycles: identity fields, the four gate criteria (PR-AUC, recall at the
    base scenario, Brier/BSS, calibration slope) with their CI bounds,
    test_ev_base, the §0 V2 fairness gaps (equal_opportunity/demographic_
    parity — CLAUDE.md names these as the one pair of sliced numbers that
    should be plottable across cycles, unlike the rest of `sliced`, which
    stays artifact-only), costs_config_hash, and — comparative regime only —
    the paired-bootstrap deltas gate.py judged them against. No t*:
    reports/policy/threshold.yaml is already its own DVC metrics entry (the
    threshold stage), so repeating it here would track the same fact twice.

    costs_hash is a parameter, not computed here, so this stays a pure
    function — the caller (_write_metrics_summary) resolves it the same way
    _log_evaluation_run/threshold.py's _assemble_threshold_payloads already
    do independently. Included alongside test_ev_base for the same reason
    _log_evaluation_run tags it on the run at all: a shift in test_ev_base
    between cycles is otherwise ambiguous between "the model changed" and
    "someone edited configs/costs.yaml" — this makes that call legible
    without cross-referencing the run's tags.

    Takes decision_payload (promotion_decision_payload) rather than a
    separate `regime` argument — decision_payload already carries `regime`
    as decide_promotion's own return field, and a second, independently-
    passed regime argument could disagree with the payload it was read from.
    Call after _log_evaluation_run, once decision_payload["eval_run_id"] has
    been stamped — not before.
    """
    classification_by_scenario = {
        cast(str, row["scenario"]): row for row in metrics_payload["classification"]
    }
    base_row = classification_by_scenario["base"]
    calibration = metrics_payload["calibration"]
    slope = calibration["calibration_slope"]
    ranking = metrics_payload["ranking"]
    sliced_test = metrics_payload["sliced"]["test"]
    regime = decision_payload["regime"]
    criteria = decision_payload["criteria"]

    summary: dict[str, Any] = {
        "model_version": metrics_payload["model_version"],
        "run_id": metrics_payload["run_id"],
        "eval_run_id": decision_payload["eval_run_id"],
        "champion_version": metrics_payload["champion_version"],
        "regime": regime,
        "gate": decision_payload["gate"],
        "costs_config_hash": costs_hash,
        "test_pr_auc": ranking["pr_auc"],
        "test_pr_auc_ci_lower": ranking["pr_auc_ci_lower"],
        "test_pr_auc_ci_upper": ranking["pr_auc_ci_upper"],
        "test_recall": base_row["recall"],
        "test_recall_ci_lower": base_row["recall_ci_lower"],
        "test_recall_ci_upper": base_row["recall_ci_upper"],
        "test_brier": calibration["brier"],
        "test_bss": calibration["bss"],
        "test_calibration_slope": slope["slope"],
        "test_calibration_slope_ci_lower": slope["slope_ci_lower"],
        "test_calibration_slope_ci_upper": slope["slope_ci_upper"],
        "test_ev_base": metrics_payload["business_impact"]["scenarios"]["base"]["ev"],
        "test_equal_opportunity_diff": sliced_test["equal_opportunity_diff"],
        "test_demographic_parity_diff": sliced_test["demographic_parity_diff"],
    }

    if regime == "comparative":
        for criterion_name in ("pr_auc", "recall", "brier"):
            entry = criteria[criterion_name]
            summary[f"{criterion_name}_delta_obs"] = entry["delta_obs"]
            summary[f"{criterion_name}_delta_ci_lower"] = entry["delta_ci"][0]
            summary[f"{criterion_name}_delta_ci_upper"] = entry["delta_ci"][1]

    return summary


def _write_metrics_summary(
    metrics_payload: dict[str, Any], decision_payload: dict[str, Any], cfg: DictConfig
) -> dict[str, Any]:
    """Compute and write reports/metrics_summary.json via flatten_metrics_summary."""
    costs_hash = costs_config_hash(get_project_root() / str(cfg.paths.costs_config))
    summary = flatten_metrics_summary(metrics_payload, decision_payload, costs_hash)
    reports_dir = get_project_root() / str(cfg.paths.reports)
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(
        reports_dir / "metrics_summary.json", "w", encoding="utf-8", newline="\n"
    ) as f:
        json.dump(summary, f, indent=2, default=str)
        f.write("\n")
    return summary


def _write_decile_lift_csv(
    decile_rows: list[dict[str, float]], cfg: DictConfig
) -> None:
    """Write reports/plots/decile_lift.csv from sealed_test_decile_lift's own output — no second computation."""
    plots_dir = get_project_root() / str(cfg.paths.plots)
    plots_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(decile_rows).to_csv(plots_dir / "decile_lift.csv", index=False)


def _render_evaluation_figures(
    y_test: pd.Series,
    proba: NDArray[np.float64],
    core_metrics: dict[str, Any],
    sensitivity_block: dict[str, Any],
    payloads: dict[str, Any],
    policy_ctx: dict[str, Any],
    cfg: DictConfig,
) -> dict[str, Any]:
    """Render all eight evaluation figures and return their paths."""
    thresholds = policy_ctx["thresholds"]
    classification_rows = core_metrics["classification_rows"]
    calibration_report = core_metrics["calibration_report"]
    decile_rows = core_metrics["decile_rows"]
    sensitivity = sensitivity_block["sensitivity"]
    retention_rate_values = sensitivity_block["retention_rate_values"]
    cost_values = sensitivity_block["cost_values"]
    ev_curves = payloads["ev_curves"]

    figures_dir = get_project_root() / str(cfg.paths.figures)
    pr_curve_path = figures_dir / "pr_curve_test.png"
    roc_curve_path = figures_dir / "roc_curve_test.png"
    classification_report_path = figures_dir / "classification_report_test.png"
    reliability_path = figures_dir / "reliability_diagram_test.png"
    ev_by_budget_path = figures_dir / "ev_by_budget.png"
    breakeven_heatmap_path = figures_dir / "breakeven_heatmap.png"
    sensitivity_tornado_path = figures_dir / "sensitivity_tornado.png"
    gains_lift_path = figures_dir / "gains_lift_test.png"

    _save_pr_curve_plot(
        pr_curve_points(y_test.tolist(), proba.tolist()),
        float(y_test.mean()),
        thresholds,
        pr_curve_path,
    )
    _save_roc_curve_plot(
        roc_curve_points(y_test.tolist(), proba.tolist()), roc_curve_path
    )
    _save_classification_report_plot(classification_rows, classification_report_path)
    _save_reliability_plot_test(
        cast(list[dict[str, float]], calibration_report["reliability_bins"]),
        reliability_path,
    )
    _save_ev_by_budget_plot(
        ev_curves,
        thresholds,
        float(sensitivity_block["costs_cfg"].contact_capacity),
        ev_by_budget_path,
    )
    _save_breakeven_heatmap_plot(
        cast(list[dict[str, float]], sensitivity["twoway"]),
        retention_rate_values,
        cost_values,
        breakeven_heatmap_path,
    )
    _save_tornado_plot(
        cast(list[dict[str, Any]], sensitivity["tornado"]), sensitivity_tornado_path
    )
    _save_gains_lift_plot(decile_rows, gains_lift_path)

    return {
        "pr_curve_path": pr_curve_path,
        "roc_curve_path": roc_curve_path,
        "classification_report_path": classification_report_path,
        "reliability_path": reliability_path,
        "ev_by_budget_path": ev_by_budget_path,
        "breakeven_heatmap_path": breakeven_heatmap_path,
        "sensitivity_tornado_path": sensitivity_tornado_path,
        "gains_lift_path": gains_lift_path,
    }


def _build_scalar_metrics(
    core_metrics: dict[str, Any],
    sliced: dict[str, Any],
    capacity_flags: dict[str, dict[str, float | bool]],
) -> dict[str, float]:
    """Build the flat test_* MLflow metric dict from the already-computed core/sliced blocks.

    Pure dict assembly — no MLflow calls, so it is a no-risk extraction on
    its own. capacity_flags is economics.capacity_budget_check's output
    (computed once in _assemble_metrics_and_economics_payloads and threaded
    through payloads, not recomputed here) — only its numeric excess fields
    become metrics; over_capacity/over_budget are that same sign as a bool
    and stay economics.json-only, since MLflow metrics must be numeric.
    """
    ranking_metrics = core_metrics["ranking_metrics"]
    calibration_report = core_metrics["calibration_report"]
    classification_rows = core_metrics["classification_rows"]
    business_impact = core_metrics["business_impact"]

    scalar_metrics: dict[str, float] = {
        "test_pr_auc": cast(float, ranking_metrics["pr_auc"]),
        "test_pr_auc_ci_lower": cast(float, ranking_metrics["pr_auc_ci_lower"]),
        "test_pr_auc_ci_upper": cast(float, ranking_metrics["pr_auc_ci_upper"]),
        "test_roc_auc": cast(float, ranking_metrics["roc_auc"]),
        "test_dummy_pr_auc_floor": cast(float, ranking_metrics["dummy_pr_auc_floor"]),
        "test_brier": cast(float, calibration_report["brier"]),
        "test_bss": cast(float, calibration_report["bss"]),
        "test_ece": cast(float, calibration_report["ece"]),
        "test_calibration_slope": cast(
            float, calibration_report["calibration_slope"]["slope"]
        ),
        "test_calibration_slope_ci_lower": cast(
            float, calibration_report["calibration_slope"]["slope_ci_lower"]
        ),
        "test_calibration_slope_ci_upper": cast(
            float, calibration_report["calibration_slope"]["slope_ci_upper"]
        ),
        "test_equal_opportunity_diff": sliced["test_equal_opportunity_diff"],
        "test_demographic_parity_diff": sliced["test_demographic_parity_diff"],
    }
    for row in classification_rows:
        scenario_name = cast(str, row["scenario"])
        for metric_key in ("recall", "precision", "f1", "contact_rate"):
            scalar_metrics[f"test_{metric_key}_{scenario_name}"] = cast(
                float, row[metric_key]
            )
        for metric_key in ("recall", "precision", "f1"):
            for bound in ("ci_lower", "ci_upper"):
                scalar_metrics[f"test_{metric_key}_{scenario_name}_{bound}"] = cast(
                    float, row[f"{metric_key}_{bound}"]
                )
    for scenario_name, row in business_impact["scenarios"].items():
        scalar_metrics[f"test_ev_{scenario_name}"] = cast(float, row["ev"])
        scalar_metrics[f"test_ev_{scenario_name}_ci_lower"] = cast(
            float, row["ev_ci_lower"]
        )
        scalar_metrics[f"test_ev_{scenario_name}_ci_upper"] = cast(
            float, row["ev_ci_upper"]
        )
        scalar_metrics[f"test_campaign_cost_{scenario_name}"] = cast(
            float, row["campaign_cost"]
        )
        scalar_metrics[f"test_retained_revenue_{scenario_name}"] = cast(
            float, row["retained_revenue"]
        )
        scalar_metrics[f"test_n_contacted_{scenario_name}"] = cast(
            float, row["n_contacted"]
        )
        scalar_metrics[f"test_break_even_retention_rate_{scenario_name}"] = cast(
            float, row["break_even_retention_rate"]
        )
        scalar_metrics[f"test_ev_treat_all_{scenario_name}"] = cast(
            float, row["ev_treat_all"]
        )
        scalar_metrics[f"test_ev_treat_none_{scenario_name}"] = cast(
            float, row["ev_treat_none"]
        )
    for scenario_name, flags in capacity_flags.items():
        scalar_metrics[f"test_capacity_excess_{scenario_name}"] = cast(
            float, flags["capacity_excess"]
        )
        scalar_metrics[f"test_budget_excess_{scenario_name}"] = cast(
            float, flags["budget_excess"]
        )
    return scalar_metrics


def _log_evaluation_run(
    model_version: str,
    loaded: dict[str, Any],
    core_metrics: dict[str, Any],
    sliced: dict[str, Any],
    sensitivity_block: dict[str, Any],
    payloads: dict[str, Any],
    figures: dict[str, Any],
    policy_ctx: dict[str, Any],
    cfg: DictConfig,
) -> tuple[str, pd.DataFrame]:
    """Log every evaluation artifact/metric onto a dedicated `evaluation` run.

    Returns (eval_run_id, test_predictions) — both consumed after the run
    context closes (registry tagging, the local reports/ mirror).
    """
    X_test, y_test, proba = loaded["X_test"], loaded["y_test"], loaded["proba"]
    decision = payloads["promotion_decision_payload"]
    metrics_payload = payloads["metrics_payload"]
    economics_payload = payloads["economics_payload"]
    costs_cfg = sensitivity_block["costs_cfg"]

    ensure_experiment_metadata(cfg)
    model_id = resolve_logged_model_id(model_version, cfg)
    test_dataset = mlflow_dataset_from_pandas(
        pd.concat(
            [X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1
        ),
        name="sealed_test",
        targets=TARGET_COL,
    )

    with mlflow.start_run(run_name="evaluation") as run:
        set_run_description(_RUN_DESCRIPTION)
        eval_run_id = run.info.run_id
        # Needed so a later re-log (e.g. the review notebook stamping `review`
        # onto this same payload) targets this run, not the evaluated
        # model's own training run — the two are never the same run.
        decision["eval_run_id"] = eval_run_id
        mlflow.log_input(test_dataset, context="evaluation")

        capacity_flags = economics_payload["capacity_budget_check"]
        scalar_metrics = _build_scalar_metrics(core_metrics, sliced, capacity_flags)
        mlflow.log_metrics(scalar_metrics, model_id=model_id, dataset=test_dataset)

        for scenario_name, scenario in policy_ctx["scenarios"].items():
            mlflow.log_params(
                {
                    f"cost_{scenario_name}_c": scenario.cost,
                    f"cost_{scenario_name}_r": scenario.retention_rate,
                    f"cost_{scenario_name}_ltv": scenario.ltv,
                    f"cost_{scenario_name}_arpu": scenario.arpu,
                }
            )
        mlflow.log_param("gross_margin", float(costs_cfg.gross_margin))
        mlflow.log_param("contact_capacity", int(costs_cfg.contact_capacity))
        mlflow.log_param("campaign_budget", float(costs_cfg.campaign_budget))
        mlflow.set_tag(
            "costs_config_hash",
            costs_config_hash(get_project_root() / str(cfg.paths.costs_config)),
        )
        # Fingerprints the processed-features file this candidate was scored
        # against, so a later comparative cycle (load_incumbent_proba) can
        # detect a feature-pipeline change even when the sealed test
        # partition's customerid membership happens to be unaffected by it.
        mlflow.set_tag("data_content_hash", features_sha256())

        mlflow.set_tag("gate_regime", decision["regime"])
        mlflow.set_tag("gate_result", decision["gate"])
        for criterion_name, criterion in decision["criteria"].items():
            mlflow.set_tag(
                f"gate_criterion_{criterion_name}",
                "pass" if criterion["passed"] else "fail",
            )

        mlflow.log_dict(metrics_payload, "metrics.json")
        mlflow.log_dict(economics_payload, "economics.json")
        mlflow.log_dict(decision, "promotion_decision.json")

        for path in (
            figures["pr_curve_path"],
            figures["roc_curve_path"],
            figures["classification_report_path"],
            figures["reliability_path"],
            figures["ev_by_budget_path"],
            figures["breakeven_heatmap_path"],
            figures["sensitivity_tornado_path"],
            figures["gains_lift_path"],
        ):
            mlflow.log_artifact(str(path), artifact_path="figures")

        with tempfile.TemporaryDirectory() as tmp_dir:
            predictions_path = Path(tmp_dir) / "test_predictions.parquet"
            test_predictions = pd.DataFrame(
                {
                    "customerid": loaded["customer_ids"],
                    "y_true": y_test.reset_index(drop=True),
                    "p_hat": proba,
                    # Stamped so error_analysis.py can detect a stale local
                    # reports/test_predictions.parquet copy left over from a
                    # different (e.g. rolled-back) model version — closes the
                    # stale-copy-on-hand-run-rollback hole.
                    "logged_model_id": model_id,
                }
            )
            test_predictions.to_parquet(predictions_path, index=False)
            mlflow.log_artifact(str(predictions_path))

    return eval_run_id, test_predictions


def _write_reports_mirror(
    payloads: dict[str, Any], test_predictions: pd.DataFrame, cfg: DictConfig
) -> None:
    """Mirror metrics.json/economics.json/promotion_decision.json/test_predictions.parquet to reports/.

    reports/dev_oof_predictions.parquet and dev_oof_diagnostics.json are not
    written here — threshold.py's dev-OOF screen (Phase 6's last step)
    already wrote both, and this module produces neither. This module also no
    longer fetches dev_oof_diagnostics.json itself (load_dev_oof_diagnostics
    stays exported for error_analysis.py/register.py, which resolve it
    directly rather than through a copy embedded in metrics.json).
    """
    reports_dir = get_project_root() / str(cfg.paths.reports)
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "metrics.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(payloads["metrics_payload"], f, indent=2, default=str)
        f.write("\n")
    with open(reports_dir / "economics.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(payloads["economics_payload"], f, indent=2, default=str)
        f.write("\n")
    with open(
        reports_dir / "promotion_decision.json", "w", encoding="utf-8", newline="\n"
    ) as f:
        json.dump(payloads["promotion_decision_payload"], f, indent=2, default=str)
        f.write("\n")
    test_predictions.to_parquet(reports_dir / "test_predictions.parquet", index=False)


def run_evaluation_step(
    run_id: str, model_version: str, model_uri: str, cfg: DictConfig
) -> dict[str, Any]:
    """Run the full sealed-test evaluation cycle and log it to a dedicated `evaluation` run.

    Takes `run_id`/`model_version`/`model_uri` already resolved by the
    caller (utils.mlflow.resolve_model_identifier — an explicit run_id/
    model_version override, never an alias, or calibrate.py's receipt),
    scores the sealed test set exactly once, computes every metrics/
    economics/slice block from that one probability vector, resolves the
    incumbent champion (if any) and reads its own historical predictions off
    its eval run rather than re-scoring it (load_incumbent_proba never
    touches the sealed test set a second time), and calls
    gate.py::decide_promotion — the single pass CLAUDE.md's "test set
    touched once" invariant permits.

    Logs a dedicated `evaluation` run (never appended to the dev model's own
    run — CLAUDE.md), mirrors metrics.json/economics.json/
    promotion_decision.json/test_predictions.parquet to reports/, writes
    reports/eval_receipt.json (register.py's bootstrap pointer to this
    cycle's eval run — register.py, not this module, tags the model version
    with eval_run_id and the four gate criteria, since minting/tagging now
    happens downstream of review), and renders all eight evaluation figures
    — this module builds them directly rather than a notebook, matching this
    project's convention that notebooks only display pipeline-produced
    figures, never render their own.
    """
    loaded = _load_and_score_candidate(run_id, model_version, model_uri, cfg)
    y_test, proba = loaded["y_test"], loaded["proba"]

    policy_ctx = _load_policy_context(cfg)
    core_metrics = _compute_core_test_metrics(y_test, proba, policy_ctx, cfg)
    sensitivity_block = _compute_sensitivity_block(y_test, proba, policy_ctx, cfg)
    sliced = _compute_sliced_diagnostics(y_test, proba, policy_ctx, cfg)
    decision_result = _compute_promotion_decision(
        loaded["run_id"],
        y_test,
        proba,
        loaded["customer_ids"],
        core_metrics,
        policy_ctx,
        cfg,
    )

    payloads = _assemble_metrics_and_economics_payloads(
        model_version,
        loaded["run_id"],
        y_test,
        proba,
        core_metrics,
        sliced,
        sensitivity_block,
        decision_result,
        policy_ctx,
    )
    figures = _render_evaluation_figures(
        y_test, proba, core_metrics, sensitivity_block, payloads, policy_ctx, cfg
    )

    eval_run_id, test_predictions = _log_evaluation_run(
        model_version,
        loaded,
        core_metrics,
        sliced,
        sensitivity_block,
        payloads,
        figures,
        policy_ctx,
        cfg,
    )

    write_eval_receipt(model_version, eval_run_id, cfg)
    _write_reports_mirror(payloads, test_predictions, cfg)
    _write_metrics_summary(
        payloads["metrics_payload"], payloads["promotion_decision_payload"], cfg
    )
    _write_decile_lift_csv(core_metrics["decile_rows"], cfg)

    decision = decision_result["decision"]
    logger.info(
        "evaluation_step_done",
        run_id=loaded["run_id"],
        eval_run_id=eval_run_id,
        model_version=model_version,
        champion_version=decision_result["champion_version"],
        gate_regime=decision["regime"],
        gate_result=decision["gate"],
    )

    return {
        "eval_run_id": eval_run_id,
        "model_version": model_version,
        "champion_version": decision_result["champion_version"],
        "metrics": payloads["metrics_payload"],
        "economics": payloads["economics_payload"],
        "promotion_decision": payloads["promotion_decision_payload"],
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
            cfg.evaluate.run_id, cfg.evaluate.model_version, cfg
        )
        result = run_evaluation_step(cli_run_id, cli_model_version, cli_model_uri, cfg)
        logger.info(
            "evaluation_step_done",
            model_version=result["model_version"],
            eval_run_id=result["eval_run_id"],
            gate_result=result["promotion_decision"]["gate"],
        )
    except FileNotFoundError as e:
        logger.error("evaluation_data_not_found", error=str(e), exc_info=True)
        sys.exit(1)
    except pa.errors.SchemaError as e:
        logger.error("evaluation_data_schema_invalid", error=str(e), exc_info=True)
        sys.exit(1)
    except ValueError as e:
        logger.error("evaluation_invalid", error=str(e), exc_info=True)
        sys.exit(1)
    except AssertionError as e:
        logger.error("evaluation_assertion_failed", error=str(e), exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error("evaluation_failed", error=str(e), exc_info=True)
        sys.exit(1)

"""Step 2: paired-bootstrap comparison + decision rule, plus non-gating diagnostics."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.metrics import (
    PrecisionRecallDisplay,
    average_precision_score,
    roc_auc_score,
)

from telco_churn.features.preprocessing import TENURE_COHORT_EDGES, TENURE_COHORT_LABELS
from telco_churn.models.diagnostics import (
    fixed_recall_profile,
    segment_bootstrap_delta,
    segment_oof_errors,
)
from telco_churn.models.train.common import _dvc_hash, _git_sha, _resolve_tracking_uri
from telco_churn.utils.logging import get_logger

__all__ = ["bootstrap_comparison", "run_comparison_step", "run_diagnostics_step"]

logger = get_logger(__name__)

# Flag-only characterization segments — never decide the model family (CLAUDE.md
# one-metric invariant).
_ROBUSTNESS_SEGMENTS: tuple[str, ...] = (
    "contract_type",
    "tenure_cohort",
    "internetservice",
)
_FAIRNESS_SEGMENTS: tuple[str, ...] = (
    "gender",
    "seniorcitizen",
    "has_partner",
    "dependents",
)


def bootstrap_comparison(
    scores_lgbm: list[float],
    scores_logreg: list[float],
    n_bootstrap: int,
    delta_threshold: float,
    random_state: int,
) -> dict[str, Any]:
    """Paired-bootstrap test on Δ = mean(AP_LGBM) − mean(AP_LogReg).

    Pairs fold i of LGBM against fold i of LogReg (same RSKF instance) so intra-fold
    variance cancels. Decision rule, evaluated on the 95% percentile bootstrap CI of
    Δ plus the materiality threshold delta_threshold:
      lgbm_win   — CI excludes 0 in LGBM's favour and delta_obs >= threshold.
      logreg_win — CI excludes 0 in LogReg's favour and |delta_obs| >= threshold.
      tie        — otherwise; LGBM is still adopted.

    Returns delta_obs, delta_ci_lower/upper, p_value (informational only — the CI is
    the test), decision ('lgbm'/'logreg'), decision_rule, n_bootstrap, and the raw
    bootstrap_deltas distribution (used for plotting by run_comparison_step).
    """
    rng = np.random.default_rng(random_state)
    diffs = np.array(scores_lgbm) - np.array(scores_logreg)
    delta_obs = float(diffs.mean())
    n = len(diffs)
    bootstrap_deltas = np.fromiter(
        (rng.choice(diffs, size=n, replace=True).mean() for _ in range(n_bootstrap)),
        dtype=float,
        count=n_bootstrap,
    )
    p_value = float((bootstrap_deltas <= 0).mean())
    ci_lower = float(np.percentile(bootstrap_deltas, 2.5))
    ci_upper = float(np.percentile(bootstrap_deltas, 97.5))

    if ci_lower > 0 and delta_obs >= delta_threshold:
        decision, decision_rule = "lgbm", "lgbm_win"
    elif ci_upper < 0 and delta_obs <= -delta_threshold:
        decision, decision_rule = "logreg", "logreg_win"
    else:
        decision, decision_rule = "lgbm", "tie"

    return {
        "delta_obs": round(delta_obs, 3),
        "delta_ci_lower": round(ci_lower, 3),
        "delta_ci_upper": round(ci_upper, 3),
        "p_value": round(p_value, 3),
        "decision": decision,
        "decision_rule": decision_rule,
        "n_bootstrap": n_bootstrap,
        "bootstrap_deltas": bootstrap_deltas,  # raw distribution; used for plotting in run_comparison_step
    }


def run_diagnostics_step(
    X_dev: pd.DataFrame,
    candidate_results: dict[str, dict[str, Any]],
    cfg: DictConfig,
) -> dict[str, list[dict[str, Any]]]:
    """Compute the fixed-recall profile, robustness/fairness segment flags, and
    per-segment paired-bootstrap delta CIs.

    Flag-only characterization for the analyst — never decides the model family
    (that's bootstrap_comparison's job alone). Must be called from inside
    run_comparison_step's active MLflow run — logs five artifacts onto that same
    run so the model card can read them by run_id without a second lookup.

    Returns {"fixed_recall": [...], "robustness": [...], "fairness": [...],
    "robustness_delta": [...], "fairness_delta": [...]}, each a flat row list — the
    first three tagged with their source candidate, the last two paired
    (LGBM-vs-LogReg delta per segment, not tagged per candidate).
    """
    recall_targets = list(cfg.training_setup.fixed_recall_thresholds)
    tenure_cohort = (
        pd.cut(
            X_dev["tenure"],
            bins=TENURE_COHORT_EDGES,
            labels=TENURE_COHORT_LABELS,
            include_lowest=True,
        )
        .astype(str)
        .rename("tenure_cohort")
    )
    segment_lookup: dict[str, pd.Series] = {
        "contract_type": X_dev["contract_type"],
        "tenure_cohort": tenure_cohort,
        "internetservice": X_dev["internetservice"],
        "gender": X_dev["gender"],
        "seniorcitizen": X_dev["seniorcitizen"],
        "has_partner": X_dev["has_partner"],
        "dependents": X_dev["dependents"],
    }

    fixed_recall_rows: list[dict[str, Any]] = []
    robustness_rows: list[dict[str, Any]] = []
    fairness_rows: list[dict[str, Any]] = []

    for candidate, result in candidate_results.items():
        for fr_row in fixed_recall_profile(
            result["oof_true"], result["oof_proba"], recall_targets
        ):
            fixed_recall_rows.append({"candidate": candidate, **fr_row})

        for segment_name in _ROBUSTNESS_SEGMENTS:
            for seg_row in segment_oof_errors(
                result["oof_true"], result["oof_proba"], segment_lookup[segment_name]
            ):
                robustness_rows.append({"candidate": candidate, **seg_row})

        for segment_name in _FAIRNESS_SEGMENTS:
            for seg_row in segment_oof_errors(
                result["oof_true"], result["oof_proba"], segment_lookup[segment_name]
            ):
                fairness_rows.append({"candidate": candidate, **seg_row})

    lgbm_res = candidate_results["lgbm_default"]
    logreg_res = candidate_results["logreg_cv"]
    n_segment_bootstrap = int(cfg.training_setup.segment_bootstrap_n_samples)
    random_state = int(cfg.random_seed)

    robustness_delta_rows: list[dict[str, Any]] = [
        row
        for segment_name in _ROBUSTNESS_SEGMENTS
        for row in segment_bootstrap_delta(
            lgbm_res["oof_true"],
            lgbm_res["oof_proba"],
            logreg_res["oof_proba"],
            segment_lookup[segment_name],
            n_bootstrap=n_segment_bootstrap,
            random_state=random_state,
        )
    ]
    fairness_delta_rows: list[dict[str, Any]] = [
        row
        for segment_name in _FAIRNESS_SEGMENTS
        for row in segment_bootstrap_delta(
            lgbm_res["oof_true"],
            lgbm_res["oof_proba"],
            logreg_res["oof_proba"],
            segment_lookup[segment_name],
            n_bootstrap=n_segment_bootstrap,
            random_state=random_state,
        )
    ]

    mlflow.log_text(
        pd.DataFrame(fixed_recall_rows).round(3).to_csv(index=False),
        "diagnostics/fixed_recall_profile.csv",
    )
    mlflow.log_text(
        pd.DataFrame(robustness_rows).round(3).to_csv(index=False),
        "diagnostics/segment_robustness.csv",
    )
    mlflow.log_text(
        pd.DataFrame(fairness_rows).round(3).to_csv(index=False),
        "diagnostics/segment_fairness.csv",
    )
    mlflow.log_text(
        pd.DataFrame(robustness_delta_rows).round(3).to_csv(index=False),
        "diagnostics/segment_robustness_delta.csv",
    )
    mlflow.log_text(
        pd.DataFrame(fairness_delta_rows).round(3).to_csv(index=False),
        "diagnostics/segment_fairness_delta.csv",
    )

    logger.info(
        "diagnostics_step_done",
        n_fixed_recall_rows=len(fixed_recall_rows),
        n_robustness_rows=len(robustness_rows),
        n_fairness_rows=len(fairness_rows),
        n_robustness_delta_rows=len(robustness_delta_rows),
        n_fairness_delta_rows=len(fairness_delta_rows),
    )

    return {
        "fixed_recall": fixed_recall_rows,
        "robustness": robustness_rows,
        "fairness": fairness_rows,
        "robustness_delta": robustness_delta_rows,
        "fairness_delta": fairness_delta_rows,
    }


def run_comparison_step(
    X_dev: pd.DataFrame,
    candidate_results: dict[str, dict[str, Any]],
    cfg: DictConfig,
) -> dict[str, Any]:
    """Run the paired-bootstrap decision on candidate_results and log a comparison run.

    Logs a 'model_comparison' MLflow run with bootstrap metrics, decision params,
    the comparison table/plots, and — via run_diagnostics_step inside the same
    run — the fixed-recall/fairness/robustness diagnostics.

    Returns bootstrap_comparison's dict (see its docstring) plus a "diagnostics" key
    holding run_diagnostics_step's return value.
    """
    ts = cfg.training_setup
    random_state = int(cfg.random_seed)

    lgbm_res = candidate_results["lgbm_default"]
    logreg_res = candidate_results["logreg_cv"]

    bootstrap = bootstrap_comparison(
        scores_lgbm=lgbm_res["pr_auc_scores"],
        scores_logreg=logreg_res["pr_auc_scores"],
        n_bootstrap=int(ts.bootstrap_n_samples),
        delta_threshold=float(ts.delta_threshold),
        random_state=random_state,
    )

    roc_lgbm = float(roc_auc_score(lgbm_res["oof_true"], lgbm_res["oof_proba"]))
    roc_logreg = float(roc_auc_score(logreg_res["oof_true"], logreg_res["oof_proba"]))
    pr_auc_oof_lgbm = float(
        average_precision_score(lgbm_res["oof_true"], lgbm_res["oof_proba"])
    )
    pr_auc_oof_logreg = float(
        average_precision_score(logreg_res["oof_true"], logreg_res["oof_proba"])
    )

    comparison_rows: list[dict[str, Any]] = []
    for cname, res in candidate_results.items():
        roc: float = (
            roc_lgbm
            if cname == "lgbm_default"
            else roc_logreg if cname == "logreg_cv" else float("nan")
        )
        pr_auc_oof: float = (
            pr_auc_oof_lgbm
            if cname == "lgbm_default"
            else pr_auc_oof_logreg if cname == "logreg_cv" else float("nan")
        )
        comparison_rows.append(
            {
                "candidate": cname,
                "cv_pr_auc_mean": round(res["pr_auc_mean"], 3),
                "cv_pr_auc_std": round(res["pr_auc_std"], 3),
                "pr_auc_oof": (
                    round(pr_auc_oof, 3) if not np.isnan(pr_auc_oof) else float("nan")
                ),
                "roc_auc": round(roc, 3) if not np.isnan(roc) else float("nan"),
                "train_time_s": round(res["train_time_s"], 2),
                "predict_time_s": round(res["predict_time_s"], 3),
            }
        )

    fig_pr, ax_pr = plt.subplots(figsize=(6, 5))
    for label, res in [("LightGBM", lgbm_res), ("LogReg", logreg_res)]:
        PrecisionRecallDisplay.from_predictions(
            res["oof_true"], res["oof_proba"], name=label, ax=ax_pr
        )
    ax_pr.set_title("OOF Precision-Recall Curves (dev set)")

    fig_bs, ax_bs = plt.subplots(figsize=(6, 4))
    ax_bs.hist(bootstrap["bootstrap_deltas"], bins=50, edgecolor="none", alpha=0.75)
    ax_bs.axvline(
        bootstrap["delta_obs"],
        color="C1",
        linestyle="--",
        label=f"Δ_obs = {bootstrap['delta_obs']:.3f}",
    )
    ax_bs.axvline(0.0, color="black", linestyle=":", label="Δ = 0 (null)")
    ax_bs.axvline(
        float(ts.delta_threshold),
        color="C2",
        linestyle="--",
        label=f"Δ* = {ts.delta_threshold}",
    )
    ax_bs.set_xlabel("Δ PR-AUC (LGBM − LogReg)")
    ax_bs.set_ylabel("Bootstrap resamples")
    ax_bs.set_title("Paired-Bootstrap Δ Distribution")
    ax_bs.legend()

    mlflow.set_tracking_uri(_resolve_tracking_uri(str(cfg.mlflow.tracking_uri)))
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run(run_name="model_comparison"):
        mlflow.set_tags(
            {
                "stage": "comparison",
                "git_sha": _git_sha(),
                "dvc_data_hash": _dvc_hash(cfg),
            }
        )
        mlflow.log_params(
            {
                "delta_threshold": float(ts.delta_threshold),
                "bootstrap_n_samples": int(ts.bootstrap_n_samples),
                "decision": bootstrap["decision"],
                "decision_rule": bootstrap["decision_rule"],
            }
        )
        mlflow.log_metrics(
            {
                "bootstrap_delta_obs": bootstrap["delta_obs"],
                "bootstrap_delta_ci_lower": bootstrap["delta_ci_lower"],
                "bootstrap_delta_ci_upper": bootstrap["delta_ci_upper"],
                "bootstrap_p_value": bootstrap["p_value"],
                "roc_auc_lgbm": round(roc_lgbm, 3),
                "roc_auc_logreg": round(roc_logreg, 3),
                "pr_auc_oof_lgbm": round(pr_auc_oof_lgbm, 3),
                "pr_auc_oof_logreg": round(pr_auc_oof_logreg, 3),
            }
        )
        mlflow.log_text(
            pd.DataFrame(comparison_rows).to_csv(index=False),
            "comparison/comparison_table.csv",
        )
        mlflow.log_figure(fig_pr, "comparison/pr_curves.png")
        mlflow.log_figure(fig_bs, "comparison/bootstrap_delta_dist.png")
        plt.close(fig_pr)
        plt.close(fig_bs)

        diagnostics = run_diagnostics_step(X_dev, candidate_results, cfg)

    logger.info(
        "comparison_step_done",
        decision=bootstrap["decision"],
        decision_rule=bootstrap["decision_rule"],
        delta_obs=bootstrap["delta_obs"],
        p_value=bootstrap["p_value"],
    )
    return {**bootstrap, "diagnostics": diagnostics}

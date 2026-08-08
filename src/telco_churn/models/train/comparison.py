"""Step 2: paired-bootstrap comparison + decision rule, plus non-gating diagnostics.

Not wired into the automated pipeline (models/train/__main__.py) — the family
decision it produces is frozen into common.py::COMMITTED_MODEL_FAMILY by a
human, via a reviewed code change, never recomputed live. Called only from
notebooks/03a-model-selection.ipynb, alongside candidates.py::run_candidate_step
— see that module's docstring and common.py::COMMITTED_MODEL_FAMILY for the
full rationale.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.metrics import average_precision_score, roc_auc_score

from telco_churn.features.preprocessing import TENURE_COHORT_EDGES, TENURE_COHORT_LABELS
from telco_churn.models.diagnostics import (
    fixed_recall_profile,
    segment_bootstrap_delta,
    segment_oof_errors,
)
from telco_churn.models.train.common import (
    _dvc_hash,
    _git_sha,
    _log_dev_input,
    _plot_bootstrap_delta,
    _plot_pr_curves,
)
from telco_churn.utils.logging import get_logger
from telco_churn.utils.mlflow import ensure_experiment_metadata, set_run_description
from telco_churn.utils.stats import paired_bootstrap_ci

__all__ = ["bootstrap_comparison", "run_comparison_step", "run_diagnostics_step"]

logger = get_logger(__name__)

_RUN_DESCRIPTION = (
    "Candidate comparison — Dummy / LogReg / LightGBM scored on one shared, "
    "paired RepeatedStratifiedKFold(5x3) over the development split. Decides "
    "model family via a pre-registered paired-bootstrap PR-AUC rule "
    "(materiality delta*=0.005); non-gating fixed-recall and "
    "fairness/robustness diagnostics logged alongside but never decide the "
    "family."
)

# Flag-only characterization segments — diagnostics reported alongside the
# comparison, never inputs to it; PR-AUC alone decides the model family.
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

# Family-review-only (notebooks/03a-model-selection.ipynb), not read from Hydra
# config: nothing in the automated pipeline calls run_comparison_step, so
# there is no CLI-override use case — same module-constant pattern as
# features/select.py::ABLATION_N_BOOTSTRAP.
_FAMILY_REVIEW_N_BOOTSTRAP: int = 10_000
_FAMILY_REVIEW_SEGMENT_N_BOOTSTRAP: int = 1_000


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
    result = paired_bootstrap_ci(scores_lgbm, scores_logreg, n_bootstrap, random_state)
    delta_obs = result["delta_obs"]
    ci_lower = result["delta_ci_lower"]
    ci_upper = result["delta_ci_upper"]

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
        "p_value": round(result["p_value"], 3),
        "decision": decision,
        "decision_rule": decision_rule,
        "n_bootstrap": n_bootstrap,
        "bootstrap_deltas": result["bootstrap_deltas"],
    }


def run_diagnostics_step(
    X_dev: pd.DataFrame,
    candidate_results: dict[str, dict[str, Any]],
    cfg: DictConfig,
    segment_n_bootstrap: int = _FAMILY_REVIEW_SEGMENT_N_BOOTSTRAP,
) -> dict[str, list[dict[str, Any]]]:
    """Compute the fixed-recall profile, robustness/fairness segment flags, and
    per-segment paired-bootstrap delta CIs.

    Flag-only characterization for the analyst — never decides the model family
    (that's bootstrap_comparison's job alone). Must be called from inside
    run_comparison_step's active MLflow run — logs five artifacts onto that same
    run so the model card can read them by run_id without a second lookup.

    segment_n_bootstrap defaults to the family-review constant above (notebook
    call sites take the default); overridable for fast test fixtures.

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
    n_segment_bootstrap = segment_n_bootstrap
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


def _build_comparison_table(
    candidate_results: dict[str, dict[str, Any]],
    roc_lgbm: float,
    roc_logreg: float,
    pr_auc_oof_lgbm: float,
    pr_auc_oof_logreg: float,
) -> list[dict[str, Any]]:
    """Build the per-candidate comparison table (CV mean/std, pooled OOF AP, ROC-AUC, timings)."""
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
    return comparison_rows


def _plot_comparison_figures(
    candidate_results: dict[str, dict[str, Any]],
    bootstrap: dict[str, Any],
    delta_threshold: float,
) -> dict[str, Any]:
    """Plot the OOF precision-recall curves and the paired-bootstrap Δ distribution."""
    lgbm_res = candidate_results["lgbm_default"]
    logreg_res = candidate_results["logreg_cv"]

    fig_pr = _plot_pr_curves(
        {"LightGBM": lgbm_res, "LogReg": logreg_res},
        title="OOF Precision-Recall Curves (dev set)",
    )

    fig_bs = _plot_bootstrap_delta(
        bootstrap["bootstrap_deltas"],
        bootstrap["delta_obs"],
        delta_threshold,
        title="Paired-Bootstrap Δ Distribution",
        xlabel="Δ PR-AUC (LGBM − LogReg)",
    )

    return {"fig_pr": fig_pr, "fig_bs": fig_bs}


def run_comparison_step(
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    candidate_results: dict[str, dict[str, Any]],
    cfg: DictConfig,
    n_bootstrap: int = _FAMILY_REVIEW_N_BOOTSTRAP,
    segment_n_bootstrap: int = _FAMILY_REVIEW_SEGMENT_N_BOOTSTRAP,
) -> dict[str, Any]:
    """Run the paired-bootstrap decision on candidate_results and log a comparison run.

    Logs a 'model_comparison' MLflow run with bootstrap metrics, decision params,
    the comparison table/plots, a dataset-lineage input (the dev partition, via
    common.py's _log_dev_input — the same accessor-resolved source
    candidates.py's per-candidate runs already log), and — via
    run_diagnostics_step inside the same run — the fixed-recall/fairness/
    robustness diagnostics.

    n_bootstrap/segment_n_bootstrap default to the family-review constants
    above (notebook call sites take the defaults); overridable for fast test
    fixtures. segment_n_bootstrap is threaded through to run_diagnostics_step.

    Returns bootstrap_comparison's dict (see its docstring) plus a "diagnostics" key
    holding run_diagnostics_step's return value, and "run_id" — the model_comparison
    run just logged, so the caller resolves it directly rather than re-querying
    MLflow by name/recency afterward.
    """
    ts = cfg.training_setup
    random_state = int(cfg.random_seed)

    lgbm_res = candidate_results["lgbm_default"]
    logreg_res = candidate_results["logreg_cv"]

    bootstrap = bootstrap_comparison(
        scores_lgbm=lgbm_res["pr_auc_scores"],
        scores_logreg=logreg_res["pr_auc_scores"],
        n_bootstrap=n_bootstrap,
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

    comparison_rows = _build_comparison_table(
        candidate_results, roc_lgbm, roc_logreg, pr_auc_oof_lgbm, pr_auc_oof_logreg
    )
    figures = _plot_comparison_figures(
        candidate_results, bootstrap, float(ts.delta_threshold)
    )
    fig_pr, fig_bs = figures["fig_pr"], figures["fig_bs"]

    ensure_experiment_metadata(cfg)

    with mlflow.start_run(run_name="model_comparison") as run:
        set_run_description(_RUN_DESCRIPTION)
        mlflow.set_tags(
            {
                "stage": "comparison",
                "git_sha": _git_sha(),
                "dvc_data_hash": _dvc_hash(cfg),
            }
        )
        _log_dev_input(X_dev, y_dev, context="training")
        mlflow.log_params(
            {
                "delta_threshold": float(ts.delta_threshold),
                "bootstrap_n_samples": n_bootstrap,
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

        diagnostics = run_diagnostics_step(
            X_dev, candidate_results, cfg, segment_n_bootstrap=segment_n_bootstrap
        )

    logger.info(
        "comparison_step_done",
        decision=bootstrap["decision"],
        decision_rule=bootstrap["decision_rule"],
        delta_obs=bootstrap["delta_obs"],
        p_value=bootstrap["p_value"],
        run_id=run.info.run_id,
    )
    return {**bootstrap, "diagnostics": diagnostics, "run_id": run.info.run_id}

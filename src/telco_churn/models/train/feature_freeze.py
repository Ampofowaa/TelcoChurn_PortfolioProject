"""Step 3: freeze the committed feature set via permutation-importance selection."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.metrics import PrecisionRecallDisplay, average_precision_score
from sklearn.model_selection import RepeatedStratifiedKFold

from telco_churn.features.build import FEATURE_SCHEMA
from telco_churn.features.select import (
    compute_shap_audit,
    flag_high_shap_dropouts,
    mint_committed_list,
    reduced_set_bootstrap_test,
    run_selection_cv,
)
from telco_churn.models.train.common import (
    _dvc_hash,
    _git_sha,
    lgbm_default_params,
)
from telco_churn.utils.logging import get_logger
from telco_churn.utils.mlflow import resolve_tracking_uri

__all__ = ["run_selection_step"]

logger = get_logger(__name__)


def run_selection_step(
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    candidate_results: dict[str, dict[str, Any]],
    cfg: DictConfig,
) -> dict[str, Any]:
    """Freeze the committed feature set via permutation-importance selection.

    Model-agnostic by construction (PR-AUC drop against whatever estimator is fit,
    not a tree-specific gain value) — but still fits LightGBM here since callers
    gate this on run_comparison_step's decision == 'lgbm'. The reduced set is the
    default; a paired-bootstrap test (select.reduced_set_bootstrap_test, the same
    mechanism as Step 2's model-family decision) only overrides it in favour of
    the full set when the full set wins materially and significantly.

    Logs a 'feature_selection' MLflow run with the permutation-importance table,
    per-fold stability, SHAP audit, any high-SHAP dropouts, the bootstrap-delta
    distribution, an OOF precision-recall curve (full vs. reduced set — same
    mechanism as run_comparison_step's pr_curves.png), and committed feature list
    as artifacts. Logs both full/reduced_cv_pr_auc_mean (mean of per-fold APs —
    what the paired bootstrap above actually tests) and full/reduced_pr_auc_oof
    (single AP on pooled, repeat-averaged OOF predictions — what the PR-curve
    legend shows) as separate metrics, mirroring run_comparison_step's
    cv_pr_auc_mean/pr_auc_oof split: the two are different statistics that need
    not agree, and only the former drives the decision. Also logs each fold's
    reduced-set score as the step-indexed metric reduced_cv_pr_auc_fold (mirroring
    candidates.run_candidate_step's cv_pr_auc_fold), so a caller can pair it against
    lgbm_default's own cv_pr_auc_fold history to compute a full-vs-reduced fold win
    rate — same paired folds the bootstrap test above uses.

    When decision == 'reduced', also fits and logs a second, complementary SHAP
    audit (selection/shap_importance_audit_committed.csv) restricted to
    committed_features only — the SHAP profile of the model that actually ships,
    since the primary shap_audit above always fits on the full candidate space
    (needed for flag_high_shap_dropouts) and so no longer describes the shipped
    model once features are dropped. Skipped when decision == 'full', where
    committed_features already equals the full set and shap_audit above already is
    that view.

    Returns {"decision": "reduced"|"full", "committed_features": [...],
    "permutation_importance_table": [...], "shap_audit": [...],
    "shap_audit_committed": [...] | None, "high_shap_dropouts": [...],
    "stability": [...], "bootstrap_test": {...}}.
    """
    random_state = int(cfg.random_seed)
    cc = cfg.training_setup
    sel = cfg.selection
    estimator_params = lgbm_default_params(cfg, random_state)

    binary = list(FEATURE_SCHEMA.binary)
    multi_cat = list(FEATURE_SCHEMA.multi_cat)
    numeric = list(FEATURE_SCHEMA.numeric)
    correlated_groups = tuple(tuple(group) for group in sel.correlated_groups)

    cv = RepeatedStratifiedKFold(
        n_splits=int(cc.cv_folds),
        n_repeats=int(cc.cv_repeats),
        random_state=random_state,
    )

    cv_result = run_selection_cv(
        X_dev,
        y_dev,
        cv,
        binary,
        multi_cat,
        numeric,
        estimator_params,
        n_repeats=int(sel.n_repeats),
        noise_floor_margin=float(sel.noise_floor_margin),
        correlated_groups=correlated_groups,
        inner_val_size=float(sel.inner_val_size),
        random_state=int(sel.random_state),
        n_jobs=int(cc.cv_n_jobs),
    )
    reduced_scores = cv_result["fold_scores"]
    reduced_mean = float(np.mean(reduced_scores))

    full_scores = candidate_results["lgbm_default"]["pr_auc_scores"]
    bootstrap_test = reduced_set_bootstrap_test(
        full_scores,
        reduced_scores,
        n_bootstrap=int(sel.bootstrap_n_samples),
        delta_threshold=float(cc.delta_threshold),
        random_state=int(sel.random_state),
    )

    committed_selector = mint_committed_list(
        X_dev,
        y_dev,
        binary,
        multi_cat,
        numeric,
        estimator_params,
        n_repeats=int(sel.n_repeats),
        noise_floor_margin=float(sel.noise_floor_margin),
        correlated_groups=correlated_groups,
        inner_val_size=float(sel.inner_val_size),
        random_state=int(sel.random_state),
    )

    decision = bootstrap_test["decision"]
    committed_features = (
        committed_selector.survivors_
        if decision == "reduced"
        else binary + multi_cat + numeric
    )

    shap_audit = compute_shap_audit(
        X_dev,
        y_dev,
        committed_features,
        binary,
        multi_cat,
        numeric,
        estimator_params,
    )
    high_shap_dropouts = flag_high_shap_dropouts(shap_audit)

    # shap_audit above always fits on the full candidate space (needed so a dropped
    # feature still gets a SHAP value for flag_high_shap_dropouts to compare against).
    # When the reduced set is what actually ships, that full-set fit no longer
    # describes the shipped model, so a second audit — fit only on committed_features
    # — is computed here as the ship-accurate complementary view. Skipped when
    # decision == "full" (committed_features already equals the full set there, so
    # shap_audit above already is that view) or when committed_features is empty
    # (no columns left to fit a model on).
    shap_audit_committed: pd.DataFrame | None = None
    if decision == "reduced" and committed_features:
        shap_audit_committed = compute_shap_audit(
            X_dev,
            y_dev,
            committed_features,
            [c for c in binary if c in committed_features],
            [c for c in multi_cat if c in committed_features],
            [c for c in numeric if c in committed_features],
            estimator_params,
        )

    full_res = candidate_results["lgbm_default"]
    full_pr_auc_oof = float(
        average_precision_score(full_res["oof_true"], full_res["oof_proba"])
    )
    reduced_pr_auc_oof = float(
        average_precision_score(cv_result["oof_true"], cv_result["oof_proba"])
    )

    fig_pr, ax_pr = plt.subplots(figsize=(6, 5))
    reduced_disp = PrecisionRecallDisplay.from_predictions(
        cv_result["oof_true"], cv_result["oof_proba"], name="Reduced set", ax=ax_pr
    )
    reduced_disp.line_.set_label(
        f"Reduced set (pooled OOF AP = {reduced_pr_auc_oof:.3f})"
    )
    reduced_disp.line_.set_alpha(0.6)
    full_disp = PrecisionRecallDisplay.from_predictions(
        full_res["oof_true"], full_res["oof_proba"], name="Full set", ax=ax_pr
    )
    full_disp.line_.set_label(f"Full set (pooled OOF AP = {full_pr_auc_oof:.3f})")
    ax_pr.legend()
    ax_pr.set_title("Pooled OOF PR Curves — full vs. reduced feature set")
    fig_pr.tight_layout()

    fig_bs, ax_bs = plt.subplots(figsize=(6, 4))
    ax_bs.hist(
        bootstrap_test["bootstrap_deltas"], bins=50, edgecolor="none", alpha=0.75
    )
    ax_bs.axvline(
        bootstrap_test["delta_obs"],
        color="C1",
        linestyle="--",
        label=f"Δ_obs = {bootstrap_test['delta_obs']:.3f}",
    )
    ax_bs.axvline(0.0, color="black", linestyle=":", label="Δ = 0 (null)")
    ax_bs.axvline(
        float(cc.delta_threshold),
        color="C2",
        linestyle="--",
        label=f"Δ* = {cc.delta_threshold}",
    )
    ax_bs.set_xlabel("Δ mean-of-fold PR-AUC (full − reduced)")
    ax_bs.set_ylabel("Bootstrap resamples")
    ax_bs.set_title("Paired-Bootstrap Δ Distribution — full vs. reduced feature set")
    ax_bs.legend()
    fig_bs.tight_layout()

    mlflow.set_tracking_uri(resolve_tracking_uri(str(cfg.mlflow.tracking_uri)))
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run(run_name="feature_selection"):
        mlflow.set_tags(
            {
                "stage": "selection",
                "git_sha": _git_sha(),
                "dvc_data_hash": _dvc_hash(cfg),
            }
        )
        mlflow.log_params(
            {
                "n_repeats": int(sel.n_repeats),
                "noise_floor_margin": float(sel.noise_floor_margin),
                "inner_val_size": float(sel.inner_val_size),
                "selection_random_state": int(sel.random_state),
                "delta_threshold": float(cc.delta_threshold),
                "decision": decision,
                "decision_rule": bootstrap_test["decision_rule"],
                "n_committed_features": len(committed_features),
                "n_full_features": len(binary) + len(multi_cat) + len(numeric),
                "n_high_shap_dropouts": len(high_shap_dropouts),
            }
        )
        mlflow.log_metrics(
            {
                "reduced_cv_pr_auc_mean": round(reduced_mean, 3),
                "full_cv_pr_auc_mean": round(float(np.mean(full_scores)), 3),
                "reduced_pr_auc_oof": round(reduced_pr_auc_oof, 3),
                "full_pr_auc_oof": round(full_pr_auc_oof, 3),
                "bootstrap_delta_obs": bootstrap_test["delta_obs"],
                "bootstrap_delta_ci_lower": bootstrap_test["delta_ci_lower"],
                "bootstrap_delta_ci_upper": bootstrap_test["delta_ci_upper"],
                "bootstrap_p_value": bootstrap_test["p_value"],
            }
        )
        for step, fold_score in enumerate(reduced_scores):
            mlflow.log_metric("reduced_cv_pr_auc_fold", round(fold_score, 3), step=step)
        mlflow.log_text(
            committed_selector.permutation_importance_table_.round(3).to_csv(
                index=False
            ),
            "selection/permutation_importance_table.csv",
        )
        mlflow.log_text(
            pd.DataFrame(cv_result["stability"]).round(3).to_csv(index=False),
            "selection/per_fold_stability.csv",
        )
        mlflow.log_text(
            shap_audit.round(3).to_csv(index=False),
            "selection/shap_importance_audit.csv",
        )
        if shap_audit_committed is not None:
            mlflow.log_text(
                shap_audit_committed.round(3).to_csv(index=False),
                "selection/shap_importance_audit_committed.csv",
            )
        mlflow.log_text(
            "\n".join(high_shap_dropouts), "selection/high_shap_dropouts.txt"
        )
        mlflow.log_text(
            "\n".join(committed_features), "selection/committed_features.txt"
        )
        mlflow.log_figure(fig_bs, "selection/bootstrap_delta_dist.png")
        mlflow.log_figure(fig_pr, "selection/pr_curves.png")
        plt.close(fig_bs)
        plt.close(fig_pr)

    logger.info(
        "selection_step_done",
        decision=decision,
        decision_rule=bootstrap_test["decision_rule"],
        n_committed=len(committed_features),
        reduced_cv_pr_auc_mean=round(reduced_mean, 3),
        full_cv_pr_auc_mean=round(float(np.mean(full_scores)), 3),
        delta_obs=bootstrap_test["delta_obs"],
        p_value=bootstrap_test["p_value"],
        high_shap_dropouts=high_shap_dropouts,
    )
    if high_shap_dropouts:
        logger.warning(
            "selection_high_shap_dropouts",
            features=high_shap_dropouts,
        )

    return {
        "decision": decision,
        "committed_features": committed_features,
        "permutation_importance_table": (
            committed_selector.permutation_importance_table_.to_dict("records")
        ),
        "shap_audit": shap_audit.to_dict("records"),
        "shap_audit_committed": (
            shap_audit_committed.to_dict("records")
            if shap_audit_committed is not None
            else None
        ),
        "high_shap_dropouts": high_shap_dropouts,
        "stability": cv_result["stability"],
        "bootstrap_test": bootstrap_test,
    }

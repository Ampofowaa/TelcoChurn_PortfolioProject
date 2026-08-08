"""Notebook-only: the mechanism that actually selects features.select.py::COMMITTED_FEATURES.

Decides — not merely reviews — the committed feature set, via four steps:
(1) score a full-feature LightGBM candidate and a permutation-reduced
candidate on the same paired CV folds; (2) bootstrap-test the paired Δ to
decide 'full' vs 'reduced'; (3) build the concrete recommended feature list
and its supporting evidence (permutation-importance table, correlated-group
importance, SHAP audit) consistent with that decision; (4) log everything to
a dedicated MLflow run. See run_feature_selection_step's own docstring for
the full step-by-step detail.

Not wired into the automated pipeline (models/train/__main__.py) — the
resulting decision is frozen into features/schema.py::COMMITTED_FEATURES
(and its sibling COMMITTED_FEATURES_DECISION/
COMMITTED_FEATURES_DECISION_RUN_ID) by a human, via a reviewed code change,
never recomputed live. Same footing as candidates.py/comparison.py's
model-family decision: called only from notebooks/03b-feature-selection.ipynb
§2, on a real trigger (a new engineered feature, a per-feature drift signal,
or a scheduled periodic review), not on every retrain. Lives beside
feature_freeze.py rather than inside it — feature_freeze.py is Step 3's
per-cycle diagnostic, auditing the already-frozen list every training cycle;
it never decides membership. This module is the one place membership
actually gets decided, and only when re-run deliberately.

See ANALYSIS.md §4b for the standing decision's rationale.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from omegaconf import DictConfig
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline

from telco_churn.features.build import FEATURE_SCHEMA
from telco_churn.features.preprocessing import build_preprocessor
from telco_churn.features.select import (
    ABLATION_N_BOOTSTRAP,
    compute_shap_audit,
    flag_high_shap_dropouts,
    reduced_set_bootstrap_test,
    run_selection_cv,
)
from telco_churn.models.train.common import (
    _dvc_hash,
    _git_sha,
    _log_dev_input,
    _plot_bootstrap_delta,
    _plot_permutation_importance,
    _plot_pr_curves,
    _plot_shap_audit,
    _plot_stability,
    cv_score_candidate,
    lgbm_default_params,
)
from telco_churn.utils.logging import get_logger
from telco_churn.utils.mlflow import resolve_tracking_uri, set_run_description

__all__ = ["run_feature_selection_step"]

logger = get_logger(__name__)

_REVIEW_EXPERIMENT_NAME = "telco-churn-feature-selection-review"

_RUN_DESCRIPTION = (
    "Feature-selection review — full-feature vs. permutation-reduced LightGBM "
    "scored on one shared, paired RepeatedStratifiedKFold over the "
    "development split. Decides the committed feature set via a "
    "pre-registered paired-bootstrap PR-AUC rule, mirroring "
    "train.comparison's model-family decision. Not wired into the automated "
    "pipeline — run only on a real trigger, never every retrain."
)

# Ablation-only (notebooks/03b-feature-selection.ipynb §2), not read from Hydra
# config: nothing in the automated pipeline calls run_feature_selection_step, so
# there is no CLI-override use case — same module-constant pattern
# candidates.py's _FAMILY_REVIEW_CV_FOLDS/_FAMILY_REVIEW_CV_REPEATS use for the
# adjacent model-family decision (100 paired folds either way).
_ABLATION_CV_FOLDS: int = 10
_ABLATION_CV_REPEATS: int = 10

# Majority-vote threshold (Meinshausen & Bühlmann 2010 "stability selection")
# for the ablation's own committed-feature recommendation when the decision is
# 'reduced': a feature must have survived at least half of run_selection_cv's
# per-fold PermutationImportanceSelector fits. Deliberately not
# mint_committed_list's own single all-dev fit — see PROJECT_PLAN.md's note on
# the "minting defect": a single fit can disagree with the 100-fold picture
# (survive on a fluke, or get dropped despite surviving most folds).
_STABILITY_THRESHOLD: float = 0.5


def run_feature_selection_step(
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    cfg: DictConfig,
    cv_folds: int = _ABLATION_CV_FOLDS,
    cv_repeats: int = _ABLATION_CV_REPEATS,
    n_bootstrap: int = ABLATION_N_BOOTSTRAP,
) -> dict[str, Any]:
    """Decide the committed feature set (full vs. reduced) and log a selection_review run.

    Four steps, in the order they actually run:

    1. Score two candidates on one shared RepeatedStratifiedKFold instance, so
       they're paired fold-for-fold rather than independently resampled — the
       same construction train.comparison.bootstrap_comparison uses for the
       model-family decision:
         - full  — [preprocessor -> LightGBM] on every FEATURE_SCHEMA column
           (cv_score_candidate).
         - reduced — features.select.run_selection_cv, which refits a fresh
           PermutationImportanceSelector *inside each fold's own training
           data* (never on the full dev set), so its scores are leakage-free.
           This same call also returns, per feature, a 100-fold-averaged
           importance_table (real_importance/decoy_importance/
           importance_floor) and a survival stability rate — evidence step 3
           reuses rather than recomputing.
    2. Decide 'full' vs 'reduced': features.select.reduced_set_bootstrap_test
       runs a percentile-bootstrap test on the paired Δ against a materiality
       threshold (cfg.training_setup.delta_threshold).
    3. Build the concrete recommendation implied by that decision, plus its
       supporting evidence:
         - 'full'    -> recommended_committed_features is the entire
           FEATURE_SCHEMA space.
         - 'reduced' -> recommended_committed_features is every feature whose
           step-1 stability clears _STABILITY_THRESHOLD (majority vote across
           the 100 folds) — never a second, separately-fit
           mint_committed_list call: a single all-dev fit is a different
           computation from 100 refits on different folds/seeds and can
           disagree with the fold-survival picture (PROJECT_PLAN.md's
           "minting defect" — survive on a fluke, or get dropped despite
           surviving most folds). Falls back to the single most-stable
           feature if stability alone would strand the recommendation at
           zero features.
       features.select.compute_shap_audit and flag_high_shap_dropouts then run
       against that recommended list — the same diagnostic surface
       feature_freeze.py logs every cycle, so a human reviewing this decision
       sees the full picture, not just the bare Δ. The logged
       permutation-importance table's survived column, the SHAP audit's
       committed flag, and recommended_committed_features all agree with one
       another by construction, since all three are derived from the same
       stability signal.
    4. Log everything to a 'selection_review' MLflow run in its own
       telco-churn-feature-selection-review experiment — deliberately
       separate from cfg.mlflow.experiment_name (telco-churn-training), so
       every review run adds one more dated, comparable entry without
       cluttering the regular per-cycle training runs.

    Logged artifacts, all at the run's root except the six figures (the run
    itself already is this review — no redundant stage-name prefix):
    per_fold_stability.csv, figures/bootstrap_delta_dist.png +
    figures/pr_curves.png (pooled OOF precision-recall curves, full vs.
    reduced — the same paired-evidence pattern comparison.py logs for the
    model-family decision), permutation_importance_table.csv
    (real_importance/decoy_importance/importance_floor/stability/survived) +
    figures/permutation_importance.png (colour = survived, black dashed line =
    decoy floor) + figures/per_fold_stability.png (fraction of folds survived
    per feature, colour = final survived flag — does the selector keep
    choosing this feature fold to fold, or does the survivor list bounce
    around), group_importance.json, shap_importance_audit.csv +
    figures/shap_importance_audit_survived.png (mean(|SHAP|) bar chart,
    colour = §3's survived flag) + figures/shap_importance_audit.png (same
    bar chart, colour = committed, high-SHAP dropouts flagged in gold — the
    reduced-vs-full comparison is visible directly on this second chart when
    the decision is 'reduced', since compute_shap_audit always scores the
    full candidate space regardless of what's committed), high_shap_dropouts.txt,
    and recommended_committed_features.txt.

    cv_folds/cv_repeats/n_bootstrap default to the module constants above
    (notebook call sites take the defaults); overridable for fast test fixtures.

    Returns reduced_set_bootstrap_test's dict (see its docstring) plus
    full_cv_pr_auc_mean, reduced_cv_pr_auc_mean, fold_win_rate, stability (the
    per-feature per-fold survival table), permutation_importance_table (with
    the stability-derived survived column described above),
    group_importance, shap_audit, high_shap_dropouts,
    recommended_committed_features, and run_id — the selection_review run just
    logged, so the caller resolves it directly rather than re-querying MLflow
    by name/recency afterward.
    """
    random_state = int(cfg.random_seed)
    sel = cfg.selection
    delta_threshold = float(cfg.training_setup.delta_threshold)
    estimator_params = lgbm_default_params(cfg, random_state)

    binary = list(FEATURE_SCHEMA.binary)
    multi_cat = list(FEATURE_SCHEMA.multi_cat)
    numeric = list(FEATURE_SCHEMA.numeric)
    correlated_groups = tuple(tuple(group) for group in sel.correlated_groups)

    cv = RepeatedStratifiedKFold(
        n_splits=cv_folds, n_repeats=cv_repeats, random_state=random_state
    )

    full_pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor(binary, multi_cat, numeric)),
            ("model", LGBMClassifier(**estimator_params)),
        ]
    )
    full_result = cv_score_candidate(full_pipeline, X_dev, y_dev, cv)

    cv_result = run_selection_cv(
        X_dev,
        y_dev,
        cv,
        binary,
        multi_cat,
        numeric,
        estimator_params,
        correlated_groups=correlated_groups,
        n_repeats=int(sel.n_repeats),
        noise_floor_margin=float(sel.noise_floor_margin),
        inner_val_size=float(sel.inner_val_size),
        random_state=int(sel.random_state),
        n_jobs=-1,
    )

    bootstrap = reduced_set_bootstrap_test(
        full_fold_scores=full_result["pr_auc_scores"],
        reduced_fold_scores=cv_result["fold_scores"],
        n_bootstrap=n_bootstrap,
        delta_threshold=delta_threshold,
        random_state=int(sel.random_state),
    )

    full_wins = sum(
        f > r
        for f, r in zip(
            full_result["pr_auc_scores"], cv_result["fold_scores"], strict=True
        )
    )
    n_folds = len(full_result["pr_auc_scores"])
    fold_win_rate = full_wins / n_folds

    # importance_table/group_importance are both cross-fold aggregates read
    # straight off the run_selection_cv call above — the same 100 folds that
    # produced the bootstrap decision, not a second, separately-fit
    # mint_committed_list estimate (see docstring: "same config, different
    # randomness" is not "the same computation").
    importance_table = pd.DataFrame(cv_result["importance_table"]).merge(
        pd.DataFrame(cv_result["stability"]), on="feature", how="left"
    )
    importance_table["survived"] = importance_table["stability"] >= _STABILITY_THRESHOLD

    if bootstrap["decision"] == "full":
        recommended_committed_features = binary + multi_cat + numeric
    else:
        survived_features = set(
            importance_table.loc[importance_table["survived"], "feature"]
        )
        recommended_committed_features = [
            f for f in (binary + multi_cat + numeric) if f in survived_features
        ]
        if not recommended_committed_features:
            # Never hand back zero features — the same guarantee
            # PermutationImportanceSelector.fit() gives survivors_, applied
            # here to the stability-based decision: keep the single
            # most-stable feature rather than an empty recommendation.
            fallback_feature = str(
                importance_table.sort_values("stability", ascending=False)[
                    "feature"
                ].iloc[0]
            )
            recommended_committed_features = [fallback_feature]
            importance_table.loc[
                importance_table["feature"] == fallback_feature, "survived"
            ] = True

    shap_audit = compute_shap_audit(
        X_dev,
        y_dev,
        recommended_committed_features,
        binary,
        multi_cat,
        numeric,
        estimator_params,
    )
    high_shap_dropouts = flag_high_shap_dropouts(shap_audit)

    fig_bs = _plot_bootstrap_delta(
        bootstrap["bootstrap_deltas"],
        bootstrap["delta_obs"],
        delta_threshold,
        title="Paired-Bootstrap Δ Distribution — Full vs. Reduced Feature Set",
        xlabel="Δ PR-AUC (full − reduced)",
    )
    fig_pr = _plot_pr_curves(
        {"Full": full_result, "Reduced": cv_result},
        title="OOF Precision-Recall Curves — Full vs. Reduced Feature Set",
    )

    mlflow.set_tracking_uri(resolve_tracking_uri(str(cfg.mlflow.tracking_uri)))
    mlflow.set_experiment(_REVIEW_EXPERIMENT_NAME)

    with mlflow.start_run(
        run_name=f"selection_review_{pd.Timestamp.now():%Y%m%d_%H%M%S}"
    ) as run:
        set_run_description(_RUN_DESCRIPTION)
        mlflow.set_tags(
            {
                "stage": "selection_review",
                "triggered_by": "manual",
                "git_sha": _git_sha(),
                "dvc_data_hash": _dvc_hash(cfg),
            }
        )
        _log_dev_input(X_dev, y_dev, context="review")
        mlflow.log_params(
            {
                "delta_threshold": delta_threshold,
                "bootstrap_n_samples": n_bootstrap,
                "cv_folds": cv_folds,
                "cv_repeats": cv_repeats,
                "decision": bootstrap["decision"],
                "decision_rule": bootstrap["decision_rule"],
                "n_recommended_features": len(recommended_committed_features),
                "n_high_shap_dropouts": len(high_shap_dropouts),
            }
        )
        mlflow.log_metrics(
            {
                "full_cv_pr_auc_mean": float(np.mean(full_result["pr_auc_scores"])),
                "reduced_cv_pr_auc_mean": float(np.mean(cv_result["fold_scores"])),
                "bootstrap_delta_obs": bootstrap["delta_obs"],
                "bootstrap_delta_ci_lower": bootstrap["delta_ci_lower"],
                "bootstrap_delta_ci_upper": bootstrap["delta_ci_upper"],
                "bootstrap_p_value": bootstrap["p_value"],
                "fold_win_rate": fold_win_rate,
            }
        )
        mlflow.log_text(
            pd.DataFrame(cv_result["stability"]).round(3).to_csv(index=False),
            "per_fold_stability.csv",
        )
        mlflow.log_figure(fig_bs, "figures/bootstrap_delta_dist.png")
        plt.close(fig_bs)
        mlflow.log_figure(fig_pr, "figures/pr_curves.png")
        plt.close(fig_pr)
        mlflow.log_text(
            importance_table.round(3).to_csv(index=False),
            "permutation_importance_table.csv",
        )
        fig_importance = _plot_permutation_importance(
            importance_table,
            title="Permutation Importance — This Review's Cross-Fold Fit",
        )
        mlflow.log_figure(fig_importance, "figures/permutation_importance.png")
        plt.close(fig_importance)
        fig_stability = _plot_stability(
            importance_table,
            title="Per-Fold Selection Stability — This Review",
        )
        mlflow.log_figure(fig_stability, "figures/per_fold_stability.png")
        plt.close(fig_stability)
        mlflow.log_dict(
            {
                "|".join(members): importance
                for members, importance in cv_result["group_importance"].items()
            },
            "group_importance.json",
        )
        mlflow.log_text(
            shap_audit.round(3).to_csv(index=False),
            "shap_importance_audit.csv",
        )
        fig_shap_survived = _plot_shap_audit(
            shap_audit.merge(importance_table[["feature", "survived"]], on="feature"),
            high_shap_dropouts=[],
            title="SHAP Audit — Coloured by §3 Permutation-Importance Survival",
            flag_column="survived",
            legend_labels=("Survived §3 floor", "Did not survive"),
        )
        mlflow.log_figure(
            fig_shap_survived, "figures/shap_importance_audit_survived.png"
        )
        plt.close(fig_shap_survived)
        fig_shap = _plot_shap_audit(
            shap_audit,
            high_shap_dropouts,
            title="SHAP Audit — This Review's Recommended Features",
        )
        mlflow.log_figure(fig_shap, "figures/shap_importance_audit.png")
        plt.close(fig_shap)
        mlflow.log_text("\n".join(high_shap_dropouts), "high_shap_dropouts.txt")
        mlflow.log_text(
            "\n".join(recommended_committed_features),
            "recommended_committed_features.txt",
        )

    logger.info(
        "selection_review_step_done",
        decision=bootstrap["decision"],
        decision_rule=bootstrap["decision_rule"],
        delta_obs=bootstrap["delta_obs"],
        p_value=bootstrap["p_value"],
        fold_win_rate=round(fold_win_rate, 3),
        n_recommended_features=len(recommended_committed_features),
        n_high_shap_dropouts=len(high_shap_dropouts),
        run_id=run.info.run_id,
    )

    return {
        **bootstrap,
        "full_cv_pr_auc_mean": float(np.mean(full_result["pr_auc_scores"])),
        "reduced_cv_pr_auc_mean": float(np.mean(cv_result["fold_scores"])),
        "fold_win_rate": fold_win_rate,
        "stability": cv_result["stability"],
        "permutation_importance_table": importance_table.to_dict("records"),
        "group_importance": cv_result["group_importance"],
        "shap_audit": shap_audit.to_dict("records"),
        "high_shap_dropouts": high_shap_dropouts,
        "recommended_committed_features": recommended_committed_features,
        "run_id": run.info.run_id,
    }

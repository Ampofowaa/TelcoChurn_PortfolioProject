"""Step 3: freeze the committed feature set and log the permutation-importance diagnostic."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from omegaconf import DictConfig

from telco_churn.features.build import COMMITTED_FEATURES, FEATURE_SCHEMA
from telco_churn.features.select import (
    compute_shap_audit,
    flag_high_shap_dropouts,
    mint_committed_list,
)
from telco_churn.models.train.common import (
    _dvc_hash,
    _git_sha,
    _log_dev_input,
    _plot_permutation_importance,
    _plot_shap_audit,
    lgbm_default_params,
)
from telco_churn.utils.logging import get_logger
from telco_churn.utils.mlflow import ensure_experiment_metadata, set_run_description

__all__ = ["run_feature_audit_step"]

logger = get_logger(__name__)

_RUN_DESCRIPTION = (
    "Feature selection diagnostic — permutation importance against a "
    "noise-decoy column, fit once on all of development. Committed feature "
    "set is read from features/schema.py::COMMITTED_FEATURES, frozen by the "
    "Step 3 ablation (notebooks/03b-feature-selection.ipynb §2, ANALYSIS.md "
    "§4b) rather than recomputed here; this run logs the permutation-"
    "importance table and a non-gating SHAP audit for the model actually "
    "being promoted this cycle."
)


def _mint_and_audit(
    X_dev: pd.DataFrame, y_dev: pd.Series, cfg: DictConfig
) -> dict[str, Any]:
    """Fit the all-dev permutation-importance selector and the SHAP audit against the full candidate space."""
    random_state = int(cfg.random_seed)
    sel = cfg.selection
    estimator_params = lgbm_default_params(cfg, random_state)

    binary = list(FEATURE_SCHEMA.binary)
    multi_cat = list(FEATURE_SCHEMA.multi_cat)
    numeric = list(FEATURE_SCHEMA.numeric)
    correlated_groups = tuple(tuple(group) for group in sel.correlated_groups)
    committed_features = list(COMMITTED_FEATURES)

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

    shap_audit = compute_shap_audit(
        X_dev, y_dev, committed_features, binary, multi_cat, numeric, estimator_params
    )
    high_shap_dropouts = flag_high_shap_dropouts(shap_audit)

    return {
        "committed_features": committed_features,
        "committed_selector": committed_selector,
        "shap_audit": shap_audit,
        "high_shap_dropouts": high_shap_dropouts,
    }


def _log_selection_run(
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    audits: dict[str, Any],
    cfg: DictConfig,
) -> None:
    """Log the permutation-importance table, SHAP audit, and rescue evidence onto a dedicated `feature_selection` run."""
    committed_selector = audits["committed_selector"]
    committed_features = audits["committed_features"]
    shap_audit = audits["shap_audit"]
    high_shap_dropouts = audits["high_shap_dropouts"]

    ensure_experiment_metadata(cfg)

    with mlflow.start_run(run_name="feature_selection"):
        set_run_description(_RUN_DESCRIPTION)
        mlflow.set_tags(
            {
                "stage": "selection",
                "git_sha": _git_sha(),
                "dvc_data_hash": _dvc_hash(cfg),
            }
        )
        _log_dev_input(X_dev, y_dev, context="training")
        mlflow.log_params(
            {
                "n_repeats": int(cfg.selection.n_repeats),
                "noise_floor_margin": float(cfg.selection.noise_floor_margin),
                "inner_val_size": float(cfg.selection.inner_val_size),
                "selection_random_state": int(cfg.selection.random_state),
                "n_committed_features": len(committed_features),
                "n_high_shap_dropouts": len(high_shap_dropouts),
            }
        )
        mlflow.log_text(
            committed_selector.permutation_importance_table_.round(3).to_csv(
                index=False
            ),
            "permutation_importance_table.csv",
        )
        fig_importance = _plot_permutation_importance(
            committed_selector.permutation_importance_table_,
            title="Permutation Importance — This Cycle's All-Dev Fit",
        )
        mlflow.log_figure(fig_importance, "figures/permutation_importance.png")
        plt.close(fig_importance)
        mlflow.log_text(
            shap_audit.round(3).to_csv(index=False),
            "shap_importance_audit.csv",
        )
        fig_shap = _plot_shap_audit(
            shap_audit,
            high_shap_dropouts,
            title="SHAP Audit — This Cycle's Committed Features",
        )
        mlflow.log_figure(fig_shap, "figures/shap_importance_audit.png")
        plt.close(fig_shap)
        mlflow.log_text("\n".join(high_shap_dropouts), "high_shap_dropouts.txt")
        mlflow.log_text("\n".join(committed_features), "committed_features.txt")
        mlflow.log_dict(
            {
                "|".join(members): importance
                for members, importance in committed_selector.group_importance_.items()
            },
            "group_importance.json",
        )


def run_feature_audit_step(
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    cfg: DictConfig,
) -> dict[str, Any]:
    """Audit the already-frozen committed feature set; log the permutation-importance/SHAP diagnostic.

    The committed feature set is read from features/schema.py::COMMITTED_FEATURES
    (alongside its sibling COMMITTED_FEATURES_DECISION/
    COMMITTED_FEATURES_DECISION_RUN_ID) — hand-maintained constants, not computed
    here. The keep-vs-reduce ablation that decides their contents
    (features.select.run_selection_cv + reduced_set_bootstrap_test, a
    paired-bootstrap test between a full-feature and a permutation-reduced
    LightGBM fit, ~85,000 scoring passes) is *not* called from this automated
    step — it lives in notebooks/03b-feature-selection.ipynb §2, run only on a
    deliberate trigger (a new engineered feature, a per-feature drift signal, or
    a scheduled periodic review), never every training cycle — see ANALYSIS.md
    §4b for the standing decision's rationale.

    What this step *does* run every cycle is the cheap diagnostic surface: one
    all-dev features.select.mint_committed_list fit for the permutation-importance
    table (including decide_survivors' correlated-group rescue — e.g. the
    demographic block [gender, seniorcitizen, has_partner, dependents], whose
    members fail the floor individually but whose joint importance is what a
    fairness-motivated removal would need to price) and a non-gating SHAP audit
    (features.select.compute_shap_audit / flag_high_shap_dropouts), scored against
    the *full* FEATURE_SCHEMA candidate space regardless of what's committed. This
    re-runs every cycle deliberately: those importances describe the model
    actually being promoted, not a stale snapshot from whenever COMMITTED_FEATURES
    was last decided.

    Logs a 'feature_selection' MLflow run with the permutation-importance table
    and its rendered chart (figures/permutation_importance.png — colour =
    survived, black dashed line = decoy floor), the SHAP audit and its rendered
    chart (figures/shap_importance_audit.png — features.select.compute_shap_audit's
    table plus the mean(|SHAP|) bar chart, colour = committed, high-SHAP dropouts
    flagged in gold), any high-SHAP dropouts, the committed feature list, and the
    correlated-group joint importances (group_importance.json) as artifacts — all
    at the run's root except the two figures, since the run itself already is
    this cycle's selection diagnostic; no redundant stage-name prefix.

    Returns {"committed_features": [...], "permutation_importance_table": [...],
    "shap_audit": [...], "high_shap_dropouts": [...]}.
    """
    audits = _mint_and_audit(X_dev, y_dev, cfg)
    _log_selection_run(X_dev, y_dev, audits, cfg)

    committed_features = audits["committed_features"]
    high_shap_dropouts = audits["high_shap_dropouts"]

    logger.info(
        "selection_step_done",
        n_committed=len(committed_features),
        high_shap_dropouts=high_shap_dropouts,
    )
    if high_shap_dropouts:
        logger.warning(
            "selection_high_shap_dropouts",
            features=high_shap_dropouts,
        )

    return {
        "committed_features": committed_features,
        "permutation_importance_table": (
            audits["committed_selector"].permutation_importance_table_.to_dict(
                "records"
            )
        ),
        "shap_audit": audits["shap_audit"].to_dict("records"),
        "high_shap_dropouts": high_shap_dropouts,
    }

"""Step 3 — permutation-importance feature selector (freezes the input space).

Selection lives inside a [tree_preprocessor -> selector -> model] Pipeline so
run_selection_cv refits it per fold — leak-free by construction. Survival is
decided against a synthetic noise-decoy column using the same
IMPORTANCE_NOISE_FLOOR_MARGIN rule as feature discovery (features/generate.py
Screen 4), so both stages share one threshold.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import average_precision_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from telco_churn.features.generate import IMPORTANCE_NOISE_FLOOR_MARGIN
from telco_churn.features.preprocessing import build_preprocessor

__all__ = [
    "DECOY_FEATURE",
    "DEFAULT_CORRELATED_GROUPS",
    "PermutationImportanceSelector",
    "compute_shap_audit",
    "decide_survivors",
    "flag_high_shap_dropouts",
    "mint_committed_list",
    "reduced_set_bootstrap_test",
    "run_selection_cv",
]

# Added only inside fit()'s importance computation — never appears in transform()'s
# output or the committed feature list.
DECOY_FEATURE = "_permutation_decoy"

# Correlated per the heatmap and VIF check (§8a/§8b notebooks/01-eda.ipynb); importance
# can concentrate on one member and starve the others below the decoy floor.
DEFAULT_CORRELATED_GROUPS: tuple[tuple[str, ...], ...] = (
    ("tenure", "totalcharges", "monthlycharges"),
    # internetservice_No propagates identically across all six add-on columns
    # (VIF = inf, §8b) whenever a customer has no internet service.
    (
        "internetservice",
        "onlinesecurity",
        "onlinebackup",
        "deviceprotection",
        "techsupport",
        "streamingtv",
        "streamingmovies",
    ),
)


def _build_column_map(
    output_columns: list[str], source_columns: list[str]
) -> dict[str, str]:
    """Map each preprocessed output column back to its source named feature.

    ColumnTransformer names columns "<transformer>__<feature>" (numeric) or
    "<transformer>__<feature>_<category>" (one-hot). Longest-match resolves
    both without a per-branch check.
    """
    mapping: dict[str, str] = {}
    for col in output_columns:
        remainder = col.split("__", 1)[1] if "__" in col else col
        candidates = [
            source
            for source in source_columns
            if remainder == source or remainder.startswith(f"{source}_")
        ]
        if not candidates:
            raise ValueError(
                f"Cannot map preprocessed column {col!r} back to a source feature"
            )
        mapping[col] = max(candidates, key=len)
    return mapping


def _permutation_importance_for_group(
    model: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cols: list[str],
    baseline: float,
    n_repeats: int,
    rng: np.random.Generator,
) -> float:
    """Mean PR-AUC drop from jointly shuffling `cols` (one shared row order per repeat)."""
    n = len(X_val)
    drops = np.empty(n_repeats, dtype=float)
    for i in range(n_repeats):
        X_perm = X_val.copy()
        order = rng.permutation(n)
        X_perm[cols] = X_val[cols].to_numpy()[order]
        score = average_precision_score(y_val, model.predict_proba(X_perm)[:, 1])
        drops[i] = baseline - score
    return float(drops.mean())


def _grouped_permutation_importance(
    model: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    groups: dict[str, list[str]],
    n_repeats: int,
    random_state: int,
) -> pd.Series:
    """Grouped permutation importance (PR-AUC drop) for every named group.

    Generalizes sklearn.inspection.permutation_importance, which only shuffles one
    column at a time, to shuffle every column in a group together — the standard
    correction for permutation importance under one-hot expansion or correlated
    clusters (Debeer & Strobl 2020's "grouped"/"collapsed" permutation importance).
    """
    rng = np.random.default_rng(random_state)
    baseline = average_precision_score(y_val, model.predict_proba(X_val)[:, 1])
    return pd.Series(
        {
            name: _permutation_importance_for_group(
                model, X_val, y_val, cols, baseline, n_repeats, rng
            )
            for name, cols in groups.items()
        }
    )


def decide_survivors(
    real_importance: pd.Series,
    decoy_importance: float,
    noise_floor_margin: float,
    correlated_groups: tuple[tuple[str, ...], ...],
    group_importance: dict[tuple[str, ...], float],
) -> pd.DataFrame:
    """Decide which features clear the decoy-referenced importance floor (Step 3 signal test).

    Pure decision function, separated from the model fit and permutation passes that
    produce real_importance/decoy_importance/group_importance, so the
    correlation-aware rescue is unit-testable without a model fit. group_importance
    is precomputed by the caller (requires the fitted model + held-out split) for
    any correlated_groups entry where every member individually fails.

    Returns one row per feature: feature, real_importance, decoy_importance,
    importance_floor, survived, rescued.
    """
    floor = max(decoy_importance, 0.0) + noise_floor_margin
    survived = real_importance > floor
    rescued = pd.Series(False, index=real_importance.index)

    for group in correlated_groups:
        members = tuple(c for c in group if c in real_importance.index)
        if len(members) < 2 or survived[list(members)].any():
            continue
        joint = group_importance.get(members)
        if joint is not None and joint > floor:
            top = real_importance[list(members)].idxmax()
            survived[top] = True
            rescued[top] = True

    return pd.DataFrame(
        {
            "feature": real_importance.index,
            "real_importance": real_importance.to_numpy(),
            "decoy_importance": decoy_importance,
            "importance_floor": floor,
            "survived": survived.to_numpy(),
            "rescued": rescued.to_numpy(),
        }
    ).reset_index(drop=True)


class PermutationImportanceSelector(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """sklearn transformer step placed after a fitted tree_preprocessor.

    Splits the data it receives into an inner train/held-out pair (the same
    construction as features/generate.py's candidate_importance), fits a LightGBM
    on the inner-train portion, and measures grouped permutation importance
    (PR-AUC drop) against the inner-held-out portion for every source named
    feature (one-hot dummies aggregated back to their source via column_map) plus
    a synthetic decoy column. Drops every dummy column whose source feature fails
    decide_survivors. Model-agnostic by construction — the signal is a PR-AUC
    drop against whatever estimator is fit, not a tree-specific gain value.

    Guarantees survivors_ is never empty: if every feature fails decide_survivors on a
    given fit (possible on small/noisy folds), the single highest-importance source
    feature is kept anyway so transform() never hands a caller a zero-column frame —
    a downstream model has nothing to fit on otherwise. This is a floor applied after
    decide_survivors, not a second decision path; decide_survivors' own floor logic
    is unchanged.
    """

    def __init__(
        self,
        binary: list[str],
        multi_cat: list[str],
        numeric: list[str],
        estimator_params: dict[str, Any],
        n_repeats: int = 50,
        noise_floor_margin: float = IMPORTANCE_NOISE_FLOOR_MARGIN,
        correlated_groups: tuple[tuple[str, ...], ...] = DEFAULT_CORRELATED_GROUPS,
        inner_val_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        self.binary = binary
        self.multi_cat = multi_cat
        self.numeric = numeric
        self.estimator_params = estimator_params
        self.n_repeats = n_repeats
        self.noise_floor_margin = noise_floor_margin
        self.correlated_groups = correlated_groups
        self.inner_val_size = inner_val_size
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series) -> PermutationImportanceSelector:
        """Fit on an inner train split; score grouped permutation importance on the rest."""
        source_columns = list(self.binary) + list(self.multi_cat) + list(self.numeric)
        column_map = _build_column_map(list(X.columns), source_columns)

        decoy_rng = np.random.default_rng(self.random_state)
        X_decoy = X.copy()
        X_decoy[DECOY_FEATURE] = decoy_rng.standard_normal(len(X))

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_decoy,
            y,
            test_size=self.inner_val_size,
            stratify=y,
            random_state=self.random_state,
        )

        model = LGBMClassifier(**self.estimator_params)
        model.fit(X_tr, y_tr)

        groups: dict[str, list[str]] = {DECOY_FEATURE: [DECOY_FEATURE]}
        for col, source in column_map.items():
            groups.setdefault(source, []).append(col)

        importance = _grouped_permutation_importance(
            model, X_val, y_val, groups, self.n_repeats, self.random_state
        )
        decoy_importance = float(importance[DECOY_FEATURE])
        real_importance = importance.drop(DECOY_FEATURE)
        floor = max(decoy_importance, 0.0) + self.noise_floor_margin

        group_importance: dict[tuple[str, ...], float] = {}
        for group in self.correlated_groups:
            members = tuple(c for c in group if c in real_importance.index)
            if len(members) < 2 or (real_importance[list(members)] > floor).any():
                continue
            joint_cols = [c for m in members for c in groups[m]]
            joint_result = _grouped_permutation_importance(
                model,
                X_val,
                y_val,
                {"_rescue_group": joint_cols},
                self.n_repeats,
                self.random_state,
            )
            group_importance[members] = float(joint_result["_rescue_group"])

        self.permutation_importance_table_ = decide_survivors(
            real_importance,
            decoy_importance,
            self.noise_floor_margin,
            self.correlated_groups,
            group_importance,
        )
        survived_lookup = dict(
            zip(
                self.permutation_importance_table_["feature"],
                self.permutation_importance_table_["survived"],
                strict=True,
            )
        )
        self.survivors_ = [c for c in source_columns if survived_lookup.get(c, False)]
        if not self.survivors_:
            # Every feature failed the decoy floor on this fold/fit — rather than hand
            # transform()'s caller a zero-column frame (which crashes the downstream
            # model fit outright), keep the single highest-scoring source feature as a
            # floor. A model needs at least one input; this is a last resort, not a
            # second decision path — decide_survivors' own floor logic is untouched.
            fallback_feature = str(real_importance.idxmax())
            self.survivors_ = [fallback_feature]
            self.permutation_importance_table_.loc[
                self.permutation_importance_table_["feature"] == fallback_feature,
                "survived",
            ] = True
        self._output_columns_ = [
            col for col, source in column_map.items() if source in set(self.survivors_)
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Drop every output column whose source feature did not survive selection."""
        return X[self._output_columns_]

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray[Any, Any]:
        """Return the surviving preprocessed output column names."""
        return np.array(self._output_columns_)


def _build_selection_pipeline(
    binary: list[str],
    multi_cat: list[str],
    numeric: list[str],
    estimator_params: dict[str, Any],
    n_repeats: int,
    noise_floor_margin: float,
    correlated_groups: tuple[tuple[str, ...], ...],
    inner_val_size: float,
    random_state: int,
) -> Pipeline:
    """Build [tree_preprocessor -> PermutationImportanceSelector -> LightGBM] as one Pipeline."""
    preprocessor = build_preprocessor(binary, multi_cat, numeric)
    preprocessor.set_output(transform="pandas")
    selector = PermutationImportanceSelector(
        binary=binary,
        multi_cat=multi_cat,
        numeric=numeric,
        estimator_params=estimator_params,
        n_repeats=n_repeats,
        noise_floor_margin=noise_floor_margin,
        correlated_groups=correlated_groups,
        inner_val_size=inner_val_size,
        random_state=random_state,
    )
    model = LGBMClassifier(**estimator_params)
    return Pipeline(
        [("preprocessor", preprocessor), ("selector", selector), ("model", model)]
    )


def _fit_and_score_selection_fold(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    binary: list[str],
    multi_cat: list[str],
    numeric: list[str],
    estimator_params: dict[str, Any],
    n_repeats: int,
    noise_floor_margin: float,
    correlated_groups: tuple[tuple[str, ...], ...],
    inner_val_size: float,
    random_state: int,
) -> tuple[float, list[str], np.ndarray[Any, Any]]:
    """Fit one fold's [preprocessor -> selector -> model] pipeline and score it.

    Returns (fold_pr_auc, survivors, val_proba). Builds a fresh pipeline (no
    object shared across folds), so this is safe to run concurrently: each call
    only reads its own X_tr/X_val slice and returns its own result.
    """
    pipeline = _build_selection_pipeline(
        binary,
        multi_cat,
        numeric,
        estimator_params,
        n_repeats,
        noise_floor_margin,
        correlated_groups,
        inner_val_size,
        random_state,
    )
    pipeline.fit(X_tr, y_tr)
    proba = pipeline.predict_proba(X_val)[:, 1]
    fold_score = float(average_precision_score(y_val, proba))
    survivors = list(pipeline.named_steps["selector"].survivors_)
    return fold_score, survivors, proba


def run_selection_cv(
    X: pd.DataFrame,
    y: pd.Series,
    cv: RepeatedStratifiedKFold,
    binary: list[str],
    multi_cat: list[str],
    numeric: list[str],
    estimator_params: dict[str, Any],
    n_repeats: int = 50,
    noise_floor_margin: float = IMPORTANCE_NOISE_FLOOR_MARGIN,
    correlated_groups: tuple[tuple[str, ...], ...] = DEFAULT_CORRELATED_GROUPS,
    inner_val_size: float = 0.2,
    random_state: int = 42,
    n_jobs: int = 1,
) -> dict[str, Any]:
    """Score the reduced feature set through CV and record each fold's survivors.

    Pass the same cv instance/params used for the full-feature candidate
    (train.run_candidate_step) so fold indices are identical and the per-fold
    scores are directly paired by reduced_set_bootstrap_test.

    n_jobs — folds run independently (each builds its own pipeline instance and
    fits on its own data), so joblib.Parallel(n_jobs=...) changes only wall-clock
    time, never the returned scores/survivors — see cv_score_candidate's n_jobs
    docstring for the same invariant applied to the Step 1 candidate CV.

    Returns fold_scores (reduced-set per-fold PR-AUC), fold_survivors (one list of
    surviving source columns per fold — the per-fold stability input), stability
    (per-feature survival rate across folds), and oof_proba/oof_true (reduced-set
    out-of-fold predictions, averaged across repeats the same way
    train.common.cv_score_candidate does for the full-feature candidates — lets
    run_selection_step render a full-vs-reduced OOF precision-recall curve using
    the same mechanism as Step 2's candidate comparison).
    """
    fold_indices = list(cv.split(X, y))
    fold_results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_and_score_selection_fold)(
            X.iloc[train_idx],
            y.iloc[train_idx],
            X.iloc[val_idx],
            y.iloc[val_idx],
            binary,
            multi_cat,
            numeric,
            estimator_params,
            n_repeats,
            noise_floor_margin,
            correlated_groups,
            inner_val_size,
            random_state,
        )
        for train_idx, val_idx in fold_indices
    )

    fold_scores = [r[0] for r in fold_results]
    fold_survivors = [r[1] for r in fold_results]

    oof_proba_sum = np.zeros(len(X))
    oof_counts = np.zeros(len(X), dtype=int)
    for (_, val_idx), (_, _, proba) in zip(fold_indices, fold_results, strict=True):
        oof_proba_sum[val_idx] += proba
        oof_counts[val_idx] += 1

    source_columns = list(binary) + list(multi_cat) + list(numeric)
    n_folds = len(fold_survivors)
    stability = [
        {
            "feature": col,
            "n_folds_survived": sum(col in survivors for survivors in fold_survivors),
            "n_folds": n_folds,
            "stability": (
                sum(col in survivors for survivors in fold_survivors) / n_folds
                if n_folds
                else float("nan")
            ),
        }
        for col in source_columns
    ]

    return {
        "fold_scores": fold_scores,
        "fold_survivors": fold_survivors,
        "stability": stability,
        "oof_proba": (oof_proba_sum / oof_counts).tolist(),
        "oof_true": y.tolist(),
    }


def mint_committed_list(
    X: pd.DataFrame,
    y: pd.Series,
    binary: list[str],
    multi_cat: list[str],
    numeric: list[str],
    estimator_params: dict[str, Any],
    n_repeats: int = 50,
    noise_floor_margin: float = IMPORTANCE_NOISE_FLOOR_MARGIN,
    correlated_groups: tuple[tuple[str, ...], ...] = DEFAULT_CORRELATED_GROUPS,
    inner_val_size: float = 0.2,
    random_state: int = 42,
) -> PermutationImportanceSelector:
    """Fit the selector once on all of development to freeze the committed feature list.

    Not a leakage risk despite the single all-dev fit: the boundary that matters is
    development vs. the sealed test, which this never touches. survivors_ on the
    returned selector is the frozen feature_columns.txt list;
    permutation_importance_table_ is its audit trail.
    """
    preprocessor = build_preprocessor(binary, multi_cat, numeric)
    preprocessor.set_output(transform="pandas")
    selector = PermutationImportanceSelector(
        binary=binary,
        multi_cat=multi_cat,
        numeric=numeric,
        estimator_params=estimator_params,
        n_repeats=n_repeats,
        noise_floor_margin=noise_floor_margin,
        correlated_groups=correlated_groups,
        inner_val_size=inner_val_size,
        random_state=random_state,
    )
    pipeline = Pipeline([("preprocessor", preprocessor), ("selector", selector)])
    pipeline.fit(X, y)
    fitted_selector: PermutationImportanceSelector = pipeline.named_steps["selector"]
    return fitted_selector


def compute_shap_audit(
    X: pd.DataFrame,
    y: pd.Series,
    committed_features: list[str],
    binary: list[str],
    multi_cat: list[str],
    numeric: list[str],
    estimator_params: dict[str, Any],
) -> pd.DataFrame:
    """Log a non-gating mean(|SHAP|) audit trail for every candidate feature.

    Fits one default-config LightGBM on the full feature space (all-dev), not just
    committed_features, so a feature decide_survivors dropped still gets a SHAP
    value — otherwise there is no way to see that the gate rejected something
    TreeSHAP ranks highly. This never decides keep/drop — permutation importance
    vs. the decoy is the sole gate — it only flags a mismatch worth a human look
    (flag_high_shap_dropouts).
    """
    source_columns = list(binary) + list(multi_cat) + list(numeric)

    preprocessor = build_preprocessor(binary, multi_cat, numeric)
    preprocessor.set_output(transform="pandas")
    model = LGBMClassifier(**estimator_params)
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(X[source_columns], y)

    Xt = pipeline.named_steps["preprocessor"].transform(X[source_columns])
    column_map = _build_column_map(list(Xt.columns), source_columns)
    explainer = shap.TreeExplainer(pipeline.named_steps["model"])
    shap_values = explainer.shap_values(Xt)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_abs = pd.Series(np.abs(shap_values).mean(axis=0), index=Xt.columns)
    per_feature = mean_abs.groupby(mean_abs.index.map(column_map)).sum()
    committed = set(committed_features)
    return (
        per_feature.rename("mean_abs_shap")
        .sort_values(ascending=False)
        .rename_axis("feature")
        .reset_index()
        .assign(committed=lambda df: df["feature"].isin(committed))
    )


def flag_high_shap_dropouts(shap_audit: pd.DataFrame) -> list[str]:
    """Return dropped features that would rank in the committed set's top-N by SHAP.

    shap_audit is compute_shap_audit's output: every candidate feature, sorted
    descending by mean_abs_shap, with a committed flag. N is the number of
    committed features, so this asks: does any rejected feature outrank a feature
    permutation importance actually kept? A non-empty result is not itself a
    reason to override decide_survivors — it is a prompt to look closer.
    """
    n_committed = int(shap_audit["committed"].sum())
    top_n = shap_audit.iloc[:n_committed]
    return [str(f) for f in top_n.loc[~top_n["committed"], "feature"]]


def reduced_set_bootstrap_test(
    full_fold_scores: list[float],
    reduced_fold_scores: list[float],
    n_bootstrap: int,
    delta_threshold: float,
    random_state: int,
) -> dict[str, Any]:
    """Paired-bootstrap test on Δ = mean(AP_full) − mean(AP_reduced).

    Mirrors train.comparison.bootstrap_comparison's Step 2 test: pairs fold i of
    the full-feature model against fold i of the reduced-feature model (both use
    the same RepeatedStratifiedKFold instance, so fold indices line up), which
    cancels intra-fold variance and gives more power than testing whether the
    reduced mean merely falls inside the full model's own unpaired CI.

    Default is the simpler reduced set; only a materially and significantly worse
    reduced set overrides it:
      material_full_win    — CI excludes 0 in full's favour and Δ >= threshold:
                              adopt the full set on the evidence.
      tie_immaterial        — CI includes 0, or excludes 0 but |Δ| < threshold:
                              practical tie — adopt the reduced set by default.
      material_reduced_win — CI excludes 0 in reduced's favour and Δ <= -threshold:
                              adopt the reduced set on the evidence.

    Returns delta_obs, delta_ci_lower/upper, p_value (informational only — the CI
    is the test), decision ('reduced'/'full'), decision_rule, n_bootstrap, and the
    raw bootstrap_deltas distribution (for plotting).
    """
    rng = np.random.default_rng(random_state)
    diffs = np.array(full_fold_scores) - np.array(reduced_fold_scores)
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
        decision, decision_rule = "full", "material_full_win"
    elif ci_upper < 0 and delta_obs <= -delta_threshold:
        decision, decision_rule = "reduced", "material_reduced_win"
    else:
        decision, decision_rule = "reduced", "tie_immaterial"

    return {
        "delta_obs": round(delta_obs, 3),
        "delta_ci_lower": round(ci_lower, 3),
        "delta_ci_upper": round(ci_upper, 3),
        "p_value": round(p_value, 3),
        "decision": decision,
        "decision_rule": decision_rule,
        "n_bootstrap": n_bootstrap,
        "bootstrap_deltas": bootstrap_deltas,
    }

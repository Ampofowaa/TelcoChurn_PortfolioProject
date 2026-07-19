"""Unit tests for telco_churn.features.select — Step 3 permutation-importance selector (B6)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from telco_churn.features.select import (
    PermutationImportanceSelector,
    _build_column_map,
    _grouped_permutation_importance,
    compute_shap_audit,
    decide_survivors,
    flag_high_shap_dropouts,
    mint_committed_list,
    reduced_set_bootstrap_test,
    run_selection_cv,
)

_BINARY = ["informative_bin", "noise_bin"]
_MULTI_CAT = ["informative_cat", "noise_cat"]
_NUMERIC = ["tenure", "totalcharges", "monthlycharges", "informative_num", "noise_num"]

_FAST_ESTIMATOR_PARAMS = {
    "n_estimators": 50,
    "num_leaves": 15,
    "min_child_samples": 10,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": 1,
    "verbose": -1,
}


def _make_synthetic_data(
    n: int = 400, seed: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic dev-like frame with planted signal columns and planted pure noise.

    tenure/totalcharges/monthlycharges mirror the real correlated trio (totalcharges
    accrues from monthlycharges over tenure); informative_* columns carry real
    signal into churn; noise_* columns are independent of the target by construction.
    """
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 72, size=n)
    monthlycharges = rng.uniform(20, 120, size=n)
    totalcharges = tenure * monthlycharges + rng.normal(0, 5, size=n)
    informative_num = rng.normal(0, 1, size=n)
    noise_num = rng.normal(0, 1, size=n)
    informative_bin = rng.choice(["Yes", "No"], size=n)
    noise_bin = rng.choice(["Yes", "No"], size=n)
    informative_cat = rng.choice(["A", "B", "C"], size=n)
    noise_cat = rng.choice(["A", "B", "C"], size=n)

    logit = (
        0.08 * tenure
        + 1.5 * informative_num
        + 1.2 * (informative_bin == "Yes").astype(float)
        + 1.2 * (informative_cat == "A").astype(float)
        - 4.0
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = pd.Series(rng.binomial(1, prob), name="churn")

    X = pd.DataFrame(
        {
            "tenure": tenure,
            "totalcharges": totalcharges,
            "monthlycharges": monthlycharges,
            "informative_num": informative_num,
            "noise_num": noise_num,
            "informative_bin": informative_bin,
            "noise_bin": noise_bin,
            "informative_cat": informative_cat,
            "noise_cat": noise_cat,
        }
    )
    return X, y


@pytest.fixture
def synthetic_data() -> tuple[pd.DataFrame, pd.Series]:
    return _make_synthetic_data()


# ---------------------------------------------------------------------------
# _build_column_map
# ---------------------------------------------------------------------------


def test_build_column_map_resolves_numeric_and_ohe_dummies() -> None:
    """Numeric passthrough and one-hot dummy columns both resolve to their source."""
    mapping = _build_column_map(
        ["numeric__tenure", "binary__gender_Male", "multi_cat__contract_type_One year"],
        ["tenure", "gender", "contract_type"],
    )
    assert mapping == {
        "numeric__tenure": "tenure",
        "binary__gender_Male": "gender",
        "multi_cat__contract_type_One year": "contract_type",
    }


def test_build_column_map_unmappable_column_raises() -> None:
    """A preprocessed column with no matching source feature raises rather than
    silently attributing its importance to the wrong (or no) feature.
    """
    with pytest.raises(ValueError, match="Cannot map"):
        _build_column_map(["numeric__unknown_col"], ["tenure"])


# ---------------------------------------------------------------------------
# _grouped_permutation_importance
# ---------------------------------------------------------------------------


def test_grouped_permutation_importance_ranks_signal_above_noise() -> None:
    """Shuffling a genuinely informative column drops held-out PR-AUC more than
    shuffling an unrelated noise column.
    """
    rng = np.random.default_rng(0)
    n = 800
    X = pd.DataFrame({"signal": rng.normal(size=n), "noise": rng.normal(size=n)})
    logit = 3.0 * X["signal"] - 0.2
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = pd.Series(rng.binomial(1, prob))

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    model = LGBMClassifier(**_FAST_ESTIMATOR_PARAMS)
    model.fit(X_tr, y_tr)

    importance = _grouped_permutation_importance(
        model,
        X_val,
        y_val,
        {"signal": ["signal"], "noise": ["noise"]},
        n_repeats=15,
        random_state=42,
    )
    assert importance["signal"] > importance["noise"]


def test_grouped_permutation_importance_joint_group_covers_all_members() -> None:
    """A joint group's importance reflects shuffling every member column together."""
    rng = np.random.default_rng(0)
    n = 500
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    logit = 2.0 * X["a"] + 2.0 * X["b"]
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = pd.Series(rng.binomial(1, prob))

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    model = LGBMClassifier(**_FAST_ESTIMATOR_PARAMS)
    model.fit(X_tr, y_tr)

    importance = _grouped_permutation_importance(
        model, X_val, y_val, {"ab": ["a", "b"]}, n_repeats=15, random_state=42
    )
    assert importance["ab"] > 0.0


# ---------------------------------------------------------------------------
# decide_survivors — pure decision function
# ---------------------------------------------------------------------------


def test_decide_survivors_keeps_signal_drops_noise() -> None:
    """A feature whose real importance clears the decoy floor survives; one that
    doesn't is dropped.
    """
    real_importance = pd.Series({"signal": 0.10, "noise": 0.001})
    table = decide_survivors(
        real_importance,
        decoy_importance=0.001,
        noise_floor_margin=0.005,
        correlated_groups=(),
        group_importance={},
    ).set_index("feature")
    assert table.loc["signal", "survived"]
    assert not table.loc["noise", "survived"]


def test_decide_survivors_returns_expected_columns() -> None:
    """Output has one row per feature with the documented columns."""
    real_importance = pd.Series({"a": 0.05, "b": 0.001})
    table = decide_survivors(real_importance, 0.001, 0.005, (), {})
    assert len(table) == 2
    assert set(table.columns) == {
        "feature",
        "real_importance",
        "decoy_importance",
        "importance_floor",
        "survived",
        "rescued",
    }


def test_decide_survivors_rescues_correlated_group_when_combined_signal_clears() -> (
    None
):
    """Two correlated features that both fail individually are rescued when their
    jointly-permuted importance clears the decoy floor.
    """
    real_importance = pd.Series({"a": 0.004, "b": 0.004, "c": 0.0})
    table = decide_survivors(
        real_importance,
        decoy_importance=0.0,
        noise_floor_margin=0.005,
        correlated_groups=(("a", "b"),),
        group_importance={("a", "b"): 0.02},
    ).set_index("feature")

    assert table["survived"].sum() == 1
    assert table["rescued"].sum() == 1
    rescued_feature = table.index[table["rescued"]][0]
    assert rescued_feature in ("a", "b")
    assert not table.loc["c", "survived"]


def test_decide_survivors_no_rescue_when_a_member_already_survives() -> None:
    """No rescue fires when at least one group member already survives individually."""
    real_importance = pd.Series({"a": 0.10, "b": 0.0, "c": 0.0})
    table = decide_survivors(
        real_importance,
        decoy_importance=0.0,
        noise_floor_margin=0.005,
        correlated_groups=(("a", "b"),),
        group_importance={("a", "b"): 0.20},
    ).set_index("feature")
    assert table.loc["a", "survived"]
    assert not table.loc["a", "rescued"]
    assert not table.loc["b", "survived"]
    assert table["rescued"].sum() == 0


def test_decide_survivors_no_rescue_when_group_has_no_real_signal() -> None:
    """A correlated group with no real signal stays dropped — rescue is not a floor."""
    real_importance = pd.Series({"a": 0.001, "b": 0.001})
    table = decide_survivors(
        real_importance,
        decoy_importance=0.0,
        noise_floor_margin=0.005,
        correlated_groups=(("a", "b"),),
        group_importance={("a", "b"): 0.002},
    ).set_index("feature")
    assert not table["survived"].any()
    assert not table["rescued"].any()


# ---------------------------------------------------------------------------
# PermutationImportanceSelector — end-to-end fit/transform
# ---------------------------------------------------------------------------


def test_permutation_importance_selector_keeps_informative_drops_noise(
    synthetic_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """On synthetic data with planted signal and planted pure noise, informative
    columns survive and noise columns are dropped.
    """
    from telco_churn.features.preprocessing import build_preprocessor

    X, y = synthetic_data
    preprocessor = build_preprocessor(_BINARY, _MULTI_CAT, _NUMERIC)
    preprocessor.set_output(transform="pandas")
    Xt = preprocessor.fit_transform(X, y)

    selector = PermutationImportanceSelector(
        binary=_BINARY,
        multi_cat=_MULTI_CAT,
        numeric=_NUMERIC,
        estimator_params=_FAST_ESTIMATOR_PARAMS,
        n_repeats=15,
        noise_floor_margin=0.005,
        random_state=42,
    )
    selector.fit(Xt, y)

    assert "informative_num" in selector.survivors_
    assert "informative_bin" in selector.survivors_
    assert "informative_cat" in selector.survivors_
    assert "noise_num" not in selector.survivors_
    assert "noise_bin" not in selector.survivors_
    assert "noise_cat" not in selector.survivors_


def test_permutation_importance_selector_transform_drops_noise_output_columns(
    synthetic_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """transform() keeps only output columns whose source feature survived."""
    from telco_churn.features.preprocessing import build_preprocessor

    X, y = synthetic_data
    preprocessor = build_preprocessor(_BINARY, _MULTI_CAT, _NUMERIC)
    preprocessor.set_output(transform="pandas")
    Xt = preprocessor.fit_transform(X, y)

    selector = PermutationImportanceSelector(
        binary=_BINARY,
        multi_cat=_MULTI_CAT,
        numeric=_NUMERIC,
        estimator_params=_FAST_ESTIMATOR_PARAMS,
        n_repeats=15,
        noise_floor_margin=0.005,
        random_state=42,
    )
    selector.fit(Xt, y)
    reduced = selector.transform(Xt)

    assert not any(col.startswith("numeric__noise_num") for col in reduced.columns)
    assert list(reduced.columns) == list(selector.get_feature_names_out())
    assert set(reduced.columns) <= set(Xt.columns)


def test_permutation_importance_selector_table_covers_all_features(
    synthetic_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """permutation_importance_table_ has one row per source feature, not per output column."""
    from telco_churn.features.preprocessing import build_preprocessor

    X, y = synthetic_data
    preprocessor = build_preprocessor(_BINARY, _MULTI_CAT, _NUMERIC)
    preprocessor.set_output(transform="pandas")
    Xt = preprocessor.fit_transform(X, y)

    selector = PermutationImportanceSelector(
        binary=_BINARY,
        multi_cat=_MULTI_CAT,
        numeric=_NUMERIC,
        estimator_params=_FAST_ESTIMATOR_PARAMS,
        n_repeats=10,
        noise_floor_margin=0.005,
        random_state=42,
    )
    selector.fit(Xt, y)

    all_source_cols = set(_BINARY + _MULTI_CAT + _NUMERIC)
    assert set(selector.permutation_importance_table_["feature"]) == all_source_cols


def test_permutation_importance_selector_all_noise_falls_back_to_one_feature() -> None:
    """When every feature is pure noise, none individually clears decide_survivors'
    floor, but the fallback keeps exactly one feature rather than returning empty —
    a zero-column transform() output would crash any downstream model fit.
    """
    from telco_churn.features.preprocessing import build_preprocessor

    rng = np.random.default_rng(1)
    n = 300
    X = pd.DataFrame(
        {
            "noise_num": rng.normal(size=n),
            "noise_bin": rng.choice(["Yes", "No"], size=n),
            "noise_cat": rng.choice(["A", "B", "C"], size=n),
        }
    )
    y = pd.Series(rng.integers(0, 2, size=n), name="churn")

    preprocessor = build_preprocessor(["noise_bin"], ["noise_cat"], ["noise_num"])
    preprocessor.set_output(transform="pandas")
    Xt = preprocessor.fit_transform(X, y)

    selector = PermutationImportanceSelector(
        binary=["noise_bin"],
        multi_cat=["noise_cat"],
        numeric=["noise_num"],
        estimator_params=_FAST_ESTIMATOR_PARAMS,
        n_repeats=20,
        noise_floor_margin=0.005,
        correlated_groups=(),
        random_state=42,
    )
    selector.fit(Xt, y)

    assert len(selector.survivors_) == 1
    assert selector.survivors_[0] in {"noise_num", "noise_bin", "noise_cat"}
    table = selector.permutation_importance_table_.set_index("feature")
    assert table.loc[selector.survivors_[0], "survived"]
    assert table["survived"].sum() == 1

    reduced = selector.transform(Xt)
    assert reduced.shape[1] > 0


def test_permutation_importance_selector_empty_dataframe_raises() -> None:
    """Fitting on a zero-row frame raises rather than silently selecting nothing."""
    Xt = pd.DataFrame({"numeric__tenure": pd.Series(dtype=float)})
    y = pd.Series(dtype=int)
    selector = PermutationImportanceSelector(
        binary=[],
        multi_cat=[],
        numeric=["tenure"],
        estimator_params=_FAST_ESTIMATOR_PARAMS,
        n_repeats=5,
        random_state=42,
    )
    # train_test_split(..., stratify=y) raises on an empty y before LightGBM's own
    # empty-input error would be reached.
    with pytest.raises((ValueError, IndexError)):
        selector.fit(Xt, y)


# ---------------------------------------------------------------------------
# run_selection_cv — leak-free-by-refit + per-fold stability
# ---------------------------------------------------------------------------


def test_run_selection_cv_fits_selector_once_per_fold(
    synthetic_data: tuple[pd.DataFrame, pd.Series],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The selector's fit is called exactly once per CV fold — proof selection is
    refit inside each fold's training portion, never on the full development set.
    """
    import telco_churn.features.select as select_module

    X, y = synthetic_data
    cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=1, random_state=42)

    call_count = 0
    original_fit = select_module.PermutationImportanceSelector.fit

    def _counting_fit(self, X_fit, y_fit):  # type: ignore[no-untyped-def]
        nonlocal call_count
        call_count += 1
        return original_fit(self, X_fit, y_fit)

    monkeypatch.setattr(
        select_module.PermutationImportanceSelector, "fit", _counting_fit
    )

    run_selection_cv(
        X,
        y,
        cv,
        binary=_BINARY,
        multi_cat=_MULTI_CAT,
        numeric=_NUMERIC,
        estimator_params=_FAST_ESTIMATOR_PARAMS,
        n_repeats=10,
        noise_floor_margin=0.005,
        random_state=42,
    )

    assert call_count == cv.get_n_splits()


def test_run_selection_cv_uses_distinct_seed_per_fold(
    synthetic_data: tuple[pd.DataFrame, pd.Series],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each fold's selector gets its own child seed, not a shared random_state —
    otherwise the decoy draw and every group's permutation shuffle repeat
    identically on every fold, weakening the noise-floor estimate's independence.
    Determinism is still required: re-running with the same base random_state
    must reproduce the same per-fold seeds.
    """
    import telco_churn.features.select as select_module

    X, y = synthetic_data
    cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=2, random_state=42)

    original_init = select_module.PermutationImportanceSelector.__init__

    def _record_seed(seeds: list[int]):  # type: ignore[no-untyped-def]
        def _init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            seeds.append(kwargs["random_state"])
            return original_init(self, *args, **kwargs)

        return _init

    seeds_run1: list[int] = []
    monkeypatch.setattr(
        select_module.PermutationImportanceSelector,
        "__init__",
        _record_seed(seeds_run1),
    )
    run_selection_cv(
        X,
        y,
        cv,
        binary=_BINARY,
        multi_cat=_MULTI_CAT,
        numeric=_NUMERIC,
        estimator_params=_FAST_ESTIMATOR_PARAMS,
        n_repeats=10,
        noise_floor_margin=0.005,
        random_state=42,
    )

    assert len(seeds_run1) == cv.get_n_splits()
    assert len(set(seeds_run1)) == len(seeds_run1)

    seeds_run2: list[int] = []
    monkeypatch.setattr(
        select_module.PermutationImportanceSelector,
        "__init__",
        _record_seed(seeds_run2),
    )
    run_selection_cv(
        X,
        y,
        cv,
        binary=_BINARY,
        multi_cat=_MULTI_CAT,
        numeric=_NUMERIC,
        estimator_params=_FAST_ESTIMATOR_PARAMS,
        n_repeats=10,
        noise_floor_margin=0.005,
        random_state=42,
    )

    assert seeds_run2 == seeds_run1


def test_run_selection_cv_returns_one_score_and_survivor_list_per_fold(
    synthetic_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = synthetic_data
    cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=1, random_state=42)

    result = run_selection_cv(
        X,
        y,
        cv,
        binary=_BINARY,
        multi_cat=_MULTI_CAT,
        numeric=_NUMERIC,
        estimator_params=_FAST_ESTIMATOR_PARAMS,
        n_repeats=10,
        noise_floor_margin=0.005,
        random_state=42,
    )

    n_folds = cv.get_n_splits()
    assert len(result["fold_scores"]) == n_folds
    assert len(result["fold_survivors"]) == n_folds
    assert all(0.0 <= s <= 1.0 for s in result["fold_scores"])


def test_run_selection_cv_returns_oof_predictions_aligned_to_row_order(
    synthetic_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """oof_proba/oof_true cover every dev row, in the original row order, with every
    row scored exactly once across the (non-repeated) CV split.
    """
    X, y = synthetic_data
    cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=1, random_state=42)

    result = run_selection_cv(
        X,
        y,
        cv,
        binary=_BINARY,
        multi_cat=_MULTI_CAT,
        numeric=_NUMERIC,
        estimator_params=_FAST_ESTIMATOR_PARAMS,
        n_repeats=10,
        noise_floor_margin=0.005,
        random_state=42,
    )

    assert len(result["oof_proba"]) == len(X)
    assert result["oof_true"] == y.tolist()
    assert all(0.0 <= p <= 1.0 for p in result["oof_proba"])


def test_run_selection_cv_stability_covers_all_features_and_is_bounded(
    synthetic_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = synthetic_data
    cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=1, random_state=42)

    result = run_selection_cv(
        X,
        y,
        cv,
        binary=_BINARY,
        multi_cat=_MULTI_CAT,
        numeric=_NUMERIC,
        estimator_params=_FAST_ESTIMATOR_PARAMS,
        n_repeats=10,
        noise_floor_margin=0.005,
        random_state=42,
    )

    all_source_cols = set(_BINARY + _MULTI_CAT + _NUMERIC)
    stability = {row["feature"]: row for row in result["stability"]}
    assert set(stability) == all_source_cols
    for row in stability.values():
        assert row["n_folds"] == cv.get_n_splits()
        assert 0.0 <= row["stability"] <= 1.0


def test_run_selection_cv_parallel_matches_sequential(
    synthetic_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """n_jobs=2 returns bit-identical results to n_jobs=1 — folds build their own
    pipeline instance and touch no shared state, so running them concurrently
    changes wall-clock time only, never the scores/survivors.
    """
    X, y = synthetic_data
    common_kwargs = dict(
        binary=_BINARY,
        multi_cat=_MULTI_CAT,
        numeric=_NUMERIC,
        estimator_params=_FAST_ESTIMATOR_PARAMS,
        n_repeats=10,
        noise_floor_margin=0.005,
        random_state=42,
    )

    cv1 = RepeatedStratifiedKFold(n_splits=2, n_repeats=2, random_state=42)
    sequential = run_selection_cv(X, y, cv1, n_jobs=1, **common_kwargs)

    cv2 = RepeatedStratifiedKFold(n_splits=2, n_repeats=2, random_state=42)
    parallel = run_selection_cv(X, y, cv2, n_jobs=2, **common_kwargs)

    assert sequential["fold_scores"] == parallel["fold_scores"]
    assert sequential["fold_survivors"] == parallel["fold_survivors"]
    assert sequential["oof_proba"] == parallel["oof_proba"]


# ---------------------------------------------------------------------------
# mint_committed_list — fit once on all-dev
# ---------------------------------------------------------------------------


def test_mint_committed_list_returns_fitted_selector(
    synthetic_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = synthetic_data
    selector = mint_committed_list(
        X,
        y,
        binary=_BINARY,
        multi_cat=_MULTI_CAT,
        numeric=_NUMERIC,
        estimator_params=_FAST_ESTIMATOR_PARAMS,
        n_repeats=15,
        noise_floor_margin=0.005,
        random_state=42,
    )
    assert isinstance(selector, PermutationImportanceSelector)
    assert isinstance(selector.survivors_, list)
    assert set(selector.permutation_importance_table_["feature"]) == set(
        _BINARY + _MULTI_CAT + _NUMERIC
    )


# ---------------------------------------------------------------------------
# compute_shap_audit — non-gating diagnostic
# ---------------------------------------------------------------------------


def test_compute_shap_audit_returns_ranked_table_for_all_features(
    synthetic_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """The SHAP audit covers every candidate feature (not just committed ones), ranks
    by mean|SHAP|, and flags which survived selection — without deciding keep/drop
    itself.
    """
    X, y = synthetic_data
    all_features = _BINARY + _MULTI_CAT + _NUMERIC
    committed = ["informative_num", "informative_bin", "informative_cat", "tenure"]

    table = compute_shap_audit(
        X,
        y,
        committed_features=committed,
        binary=_BINARY,
        multi_cat=_MULTI_CAT,
        numeric=_NUMERIC,
        estimator_params=_FAST_ESTIMATOR_PARAMS,
    )

    assert set(table["feature"]) == set(all_features)
    assert set(table.columns) == {"feature", "mean_abs_shap", "committed"}
    assert (table["mean_abs_shap"] >= 0).all()
    assert list(table["mean_abs_shap"]) == sorted(table["mean_abs_shap"], reverse=True)
    assert set(table.loc[table["committed"], "feature"]) == set(committed)


# ---------------------------------------------------------------------------
# flag_high_shap_dropouts — non-gating diagnostic
# ---------------------------------------------------------------------------


def test_flag_high_shap_dropouts_empty_when_committed_are_top_ranked() -> None:
    shap_audit = pd.DataFrame(
        {
            "feature": ["a", "b", "c", "d"],
            "mean_abs_shap": [0.4, 0.3, 0.2, 0.1],
            "committed": [True, True, False, False],
        }
    )
    assert flag_high_shap_dropouts(shap_audit) == []


def test_flag_high_shap_dropouts_flags_dropped_feature_outranking_a_committed_one() -> (
    None
):
    shap_audit = pd.DataFrame(
        {
            "feature": ["a", "b", "c", "d"],
            "mean_abs_shap": [0.4, 0.3, 0.2, 0.1],
            "committed": [True, False, True, False],
        }
    )
    assert flag_high_shap_dropouts(shap_audit) == ["b"]


def test_flag_high_shap_dropouts_empty_when_all_features_committed() -> None:
    shap_audit = pd.DataFrame(
        {
            "feature": ["a", "b"],
            "mean_abs_shap": [0.4, 0.3],
            "committed": [True, True],
        }
    )
    assert flag_high_shap_dropouts(shap_audit) == []


# ---------------------------------------------------------------------------
# reduced_set_bootstrap_test — paired-difference keep-vs-reduce decision
# ---------------------------------------------------------------------------


def test_reduced_set_bootstrap_test_full_features_win() -> None:
    """CI fully above 0 and Δ clears Δ*: the full set wins on the evidence."""
    full_scores = [0.5] * 15
    reduced_scores = [0.4] * 15
    result = reduced_set_bootstrap_test(
        full_scores,
        reduced_scores,
        n_bootstrap=500,
        delta_threshold=0.01,
        random_state=42,
    )
    assert result["decision"] == "full"
    assert result["decision_rule"] == "full_features_win"
    assert result["delta_obs"] > 0


def test_reduced_set_bootstrap_test_below_threshold_ties_reduced() -> None:
    """Δ excludes 0 but is below Δ*: a tie — the reduced set is adopted by default."""
    full_scores = [0.403] * 15
    reduced_scores = [0.400] * 15
    result = reduced_set_bootstrap_test(
        full_scores,
        reduced_scores,
        n_bootstrap=500,
        delta_threshold=0.01,
        random_state=42,
    )
    assert result["decision"] == "reduced"
    assert result["decision_rule"] == "tie"


def test_reduced_set_bootstrap_test_ci_includes_zero_ties_reduced() -> None:
    """A CI straddling 0 is also a tie — the reduced set is adopted, not the full set."""
    diffs = [
        0.03,
        -0.02,
        0.04,
        -0.03,
        0.02,
        -0.04,
        0.03,
        -0.02,
        0.00,
        0.02,
        -0.03,
        0.04,
        -0.02,
        0.03,
        -0.03,
    ]
    reduced_scores = [0.40] * len(diffs)
    full_scores = [0.40 + d for d in diffs]
    result = reduced_set_bootstrap_test(
        full_scores,
        reduced_scores,
        n_bootstrap=500,
        delta_threshold=0.01,
        random_state=42,
    )
    assert result["decision"] == "reduced"
    assert result["decision_rule"] == "tie"
    assert result["delta_ci_lower"] < 0 < result["delta_ci_upper"]


def test_reduced_set_bootstrap_test_reduced_features_win() -> None:
    """CI fully below 0 and |Δ| clears Δ* in the reduced set's favour."""
    full_scores = [0.4] * 15
    reduced_scores = [0.5] * 15
    result = reduced_set_bootstrap_test(
        full_scores,
        reduced_scores,
        n_bootstrap=500,
        delta_threshold=0.01,
        random_state=42,
    )
    assert result["decision"] == "reduced"
    assert result["decision_rule"] == "reduced_features_win"


def test_reduced_set_bootstrap_test_ci_contains_obs() -> None:
    """The 95% CI must bracket the observed Δ."""
    full_scores = [0.5 + i * 0.01 for i in range(15)]
    reduced_scores = [0.4 + i * 0.01 for i in range(15)]
    result = reduced_set_bootstrap_test(
        full_scores,
        reduced_scores,
        n_bootstrap=1000,
        delta_threshold=0.01,
        random_state=42,
    )
    assert result["delta_ci_lower"] <= result["delta_obs"] <= result["delta_ci_upper"]


def test_reduced_set_bootstrap_test_return_keys() -> None:
    """All eight expected keys are present."""
    scores = [0.5] * 5
    result = reduced_set_bootstrap_test(
        scores, scores, n_bootstrap=100, delta_threshold=0.01, random_state=0
    )
    assert set(result) == {
        "delta_obs",
        "delta_ci_lower",
        "delta_ci_upper",
        "p_value",
        "decision",
        "decision_rule",
        "n_bootstrap",
        "bootstrap_deltas",
    }


def test_reduced_set_bootstrap_test_n_bootstrap_preserved() -> None:
    """n_bootstrap in the result matches the argument."""
    scores = [0.5] * 5
    result = reduced_set_bootstrap_test(
        scores, scores, n_bootstrap=250, delta_threshold=0.01, random_state=0
    )
    assert result["n_bootstrap"] == 250


def test_reduced_set_bootstrap_test_equal_scores_p_value_is_one() -> None:
    """Identical fold scores give Δ = 0 on every resample, so p_value = 1.0."""
    scores = [0.5] * 15
    result = reduced_set_bootstrap_test(
        scores, scores, n_bootstrap=500, delta_threshold=0.01, random_state=42
    )
    assert result["p_value"] == 1.0
    assert result["delta_obs"] == 0.0
    assert result["decision_rule"] == "tie"

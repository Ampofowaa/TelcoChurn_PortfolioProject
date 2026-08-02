"""Unit tests for src/telco_churn/features/generate.py.

Synthetic data strategy
-----------------------
``discovery_data`` plants a strong linear signal in ``candidate_good``
(value ≈ churn_label × 20 + small noise) so a DecisionTreeClassifier can
distinguish churners from non-churners reliably in both OOF cross-validation
and permutation-importance contexts.  ``decoy`` is pure Gaussian noise —
a well-fitted tree assigns it near-zero importance.

DecisionTreeClassifier is used for all model-dependent tests because:
  - Scale-invariant — compatible with the unscaled numeric branch of
    build_preprocessor (no StandardScaler in the numeric pipeline).
  - Detects planted signals reliably in small synthetic datasets.
  - Fast — avoids LightGBM boot overhead in CI.
DummyClassifier is NOT used: it ignores all features so permutation
importance is meaningless noise for both candidate and decoy.
LogisticRegression is NOT used: poor convergence without StandardScaler on
mixed-scale features (build_preprocessor's numeric branch has no scaler).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score
from sklearn.tree import DecisionTreeClassifier

from telco_churn.features.generate import (
    MIN_PR_AUC_DELTA,
    AdoptionDecision,
    BackwardEliminationResult,
    ImportanceResult,
    LapRecord,
    RedundancyResult,
    adoption_gate,
    backward_elimination,
    bootstrap_pr_auc_ci,
    candidate_importance,
    oof_predictions,
    profile_false_negatives,
    redundancy_screen,
    serving_available,
    write_provenance,
)
from telco_churn.features.preprocessing import build_preprocessor

# ---------------------------------------------------------------------------
# Module-scoped fixtures — shared across tests to avoid redundant model fits
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def discovery_data() -> tuple[pd.DataFrame, pd.Series]:
    """400-row balanced dataset with planted signal in ``candidate_good``.

    candidate_good ≈ y * 20 + N(0,1) — strongly separates classes.
    decoy          ~ N(0,1)           — pure noise; model assigns ~0 importance.
    """
    rng = np.random.default_rng(42)
    n = 400
    y = pd.Series(np.repeat([0, 1], n // 2), name="churn")
    X = pd.DataFrame(
        {
            "monthlycharges": rng.uniform(18.0, 119.0, size=n),
            "candidate_good": y.to_numpy() * 20.0 + rng.normal(0, 1, size=n),
            "decoy": rng.normal(0, 1, size=n),
        }
    )
    return X, y


@pytest.fixture(scope="module")
def numeric_pre() -> object:
    """Tree-family ColumnTransformer covering the three discovery_data columns."""
    return build_preprocessor(
        binary=[],
        multi_cat=[],
        numeric=["monthlycharges", "candidate_good", "decoy"],
    )


@pytest.fixture(scope="module")
def dt() -> DecisionTreeClassifier:
    """Fast, scale-invariant classifier reused across model-dependent tests."""
    return DecisionTreeClassifier(random_state=42)


@pytest.fixture(scope="module")
def oof_proba(
    discovery_data: tuple[pd.DataFrame, pd.Series],
    numeric_pre: object,
    dt: DecisionTreeClassifier,
) -> np.ndarray:
    """OOF predicted probabilities computed once and shared across multiple tests."""
    X, y = discovery_data
    return oof_predictions(X, y, numeric_pre, model=dt)


# ---------------------------------------------------------------------------
# Gate-input factory helpers — keep individual tests declarative
# ---------------------------------------------------------------------------


def _ok_serving() -> tuple[bool, str]:
    return (True, "all input columns available at serving time")


def _fail_serving(col: str = "future_col") -> tuple[bool, str]:
    return (False, f"columns not available at serving time: ['{col}']")


def _ok_redundancy() -> RedundancyResult:
    return RedundancyResult(
        flagged=False,
        max_corr=0.10,
        max_cramers_v=None,
        flag_reason=None,
    )


def _flagged_redundancy() -> RedundancyResult:
    return RedundancyResult(
        flagged=True,
        max_corr=0.97,
        max_cramers_v=None,
        flag_reason="|corr|=0.970 > 0.95 with 'existing_col'",
    )


def _ok_importance(cand: float = 0.08, floor: float = 0.01) -> ImportanceResult:
    return ImportanceResult(
        candidate_importance=cand,
        noise_floor=floor,
        above_floor=cand > floor,
    )


def _fail_importance(cand: float = 0.005, floor: float = 0.01) -> ImportanceResult:
    return ImportanceResult(
        candidate_importance=cand,
        noise_floor=floor,
        above_floor=False,
    )


def _make_lap(lap: int = 1, decision: str = "adopted") -> LapRecord:
    return LapRecord(
        lap=lap,
        blind_spot="long_tenure_mtm",
        candidate="is_long_month_to_month",
        hypothesis="H1",
        eda_anchor="§2 bivariate analysis",
        max_corr=None,
        max_cramers_v=None,
        prior_pr_auc=0.60,
        new_pr_auc=0.63,
        delta_pr_auc=0.03,
        prior_ci_low=0.55,
        prior_ci_high=0.65,
        sub_recall_delta=0.12,
        candidate_importance_score=0.04,
        noise_floor=0.01,
        decision=decision,
        rejection_screen=None,
        rejection_reason=None,
    )


# ---------------------------------------------------------------------------
# oof_predictions
# ---------------------------------------------------------------------------


def test_oof_predictions_shape_matches_input(
    discovery_data: tuple[pd.DataFrame, pd.Series],
    oof_proba: np.ndarray,
) -> None:
    """Output length equals the number of input rows."""
    X, _ = discovery_data
    assert len(oof_proba) == len(X)


def test_oof_predictions_values_in_unit_interval(oof_proba: np.ndarray) -> None:
    """All OOF predicted probabilities are in [0, 1]."""
    assert float(oof_proba.min()) >= 0.0
    assert float(oof_proba.max()) <= 1.0


def test_oof_predictions_planted_signal_yields_high_pr_auc(
    discovery_data: tuple[pd.DataFrame, pd.Series],
    oof_proba: np.ndarray,
) -> None:
    """PR-AUC on OOF probabilities exceeds 0.90 when candidate_good encodes y × 20."""
    _, y = discovery_data
    pr_auc = average_precision_score(y, oof_proba)
    assert pr_auc > 0.90


# ---------------------------------------------------------------------------
# profile_false_negatives
# ---------------------------------------------------------------------------


def test_profile_fn_returns_required_columns() -> None:
    """Output DataFrame always contains the six canonical columns."""
    X = pd.DataFrame(
        {"contract_type": ["Month-to-month", "Two year"], "tenure": [5, 60]}
    )
    y = pd.Series([1, 0])
    result = profile_false_negatives(X, y, np.array([0.3, 0.1]))
    required = {"feature", "subgroup", "fn_rate", "fn_count", "total_positives", "size"}
    assert required.issubset(set(result.columns))


def test_profile_fn_empty_df_returns_empty() -> None:
    """Empty input DataFrame returns an empty result with the required columns."""
    result = profile_false_negatives(
        pd.DataFrame(), pd.Series([], dtype=int), np.array([])
    )
    assert result.empty
    assert "fn_rate" in result.columns


def test_profile_fn_no_positives_returns_empty() -> None:
    """All-negative label vector returns an empty result — no FNs are possible."""
    X = pd.DataFrame({"tenure": [10, 20, 30]})
    y = pd.Series([0, 0, 0])
    assert profile_false_negatives(X, y, np.array([0.3, 0.2, 0.1])).empty


def test_profile_fn_categorical_groups_by_value() -> None:
    """Categorical column produces one row per distinct value that contains positives."""
    X = pd.DataFrame(
        {"contract_type": ["Month-to-month", "Two year", "Month-to-month"]}
    )
    y = pd.Series([1, 1, 0])
    # Row 0: y=1, proba=0.3 < 0.5 → FN; Row 1: y=1, proba=0.9 ≥ 0.5 → TP
    proba = np.array([0.3, 0.9, 0.2])
    result = profile_false_negatives(X, y, proba, threshold=0.5)
    mtm = result[
        (result["feature"] == "contract_type")
        & (result["subgroup"] == "Month-to-month")
    ]
    assert len(mtm) == 1
    assert float(mtm["fn_rate"].iloc[0]) == pytest.approx(1.0)


def test_profile_fn_sorted_by_fn_rate_descending() -> None:
    """Rows are sorted from highest to lowest fn_rate."""
    X = pd.DataFrame(
        {
            "contract_type": [
                "Month-to-month",
                "Two year",
                "Month-to-month",
                "Two year",
            ]
        }
    )
    y = pd.Series([1, 1, 1, 1])
    # MTM: 2 positives, 2 FNs → fn_rate 1.0; Two year: 2 positives, 0 FNs → 0.0
    proba = np.array([0.2, 0.9, 0.1, 0.8])
    result = profile_false_negatives(X, y, proba, threshold=0.5)
    fn_rates = result["fn_rate"].tolist()
    assert fn_rates == sorted(fn_rates, reverse=True)


# ---------------------------------------------------------------------------
# serving_available
# ---------------------------------------------------------------------------


def test_serving_available_returns_true_for_known_columns() -> None:
    """Columns that are a subset of SERVING_COLS return (True, …)."""
    available, reason = serving_available(["tenure", "monthlycharges"])
    assert available is True
    assert "available" in reason


def test_serving_available_returns_false_for_leaked_column() -> None:
    """A column absent from SERVING_COLS causes immediate (False, …) rejection."""
    available, reason = serving_available(["tenure", "oof_score_from_prior_model"])
    assert available is False
    assert "oof_score_from_prior_model" in reason


def test_serving_available_custom_serving_cols_respected() -> None:
    """Custom serving_cols set overrides the default SERVING_COLS constant."""
    ok, _ = serving_available(["tenure"], serving_cols=frozenset({"tenure"}))
    assert ok is True
    fail, _ = serving_available(["tenure"], serving_cols=frozenset({"monthlycharges"}))
    assert fail is False


# ---------------------------------------------------------------------------
# redundancy_screen
# ---------------------------------------------------------------------------


def test_redundancy_screen_empty_adopted_not_flagged() -> None:
    """Empty adopted set (first lap) always returns flagged=False with all None metrics."""
    candidate = pd.Series([1.0, 2.0, 3.0], name="new_feat")
    result = redundancy_screen(candidate, pd.DataFrame())
    assert result.flagged is False
    assert result.max_corr is None
    assert result.max_cramers_v is None


def test_redundancy_screen_collinear_numeric_flagged() -> None:
    """Numeric candidate nearly identical to an adopted column is flagged."""
    rng = np.random.default_rng(0)
    base = np.arange(100, dtype=float)
    candidate = pd.Series(base + rng.normal(0, 0.001, size=100), name="near_clone")
    adopted = pd.DataFrame({"existing": base})
    result = redundancy_screen(candidate, adopted)
    assert result.flagged is True
    assert result.max_corr is not None
    assert result.max_corr > 0.95


def test_redundancy_screen_independent_numeric_not_flagged() -> None:
    """Numeric candidate orthogonal to the adopted set is not flagged."""
    rng = np.random.default_rng(1)
    candidate = pd.Series(rng.normal(size=200), name="independent")
    adopted = pd.DataFrame({"existing": rng.normal(size=200)})
    result = redundancy_screen(candidate, adopted)
    assert result.flagged is False


def test_redundancy_screen_high_cramers_v_categorical_flagged() -> None:
    """Categorical candidate near-identical to an adopted categorical is flagged."""
    vals = ["A", "B", "C"] * 100
    candidate = pd.Series(vals, name="cat_new")
    adopted = pd.DataFrame({"cat_existing": vals})
    result = redundancy_screen(candidate, adopted, cramers_v_threshold=0.50)
    assert result.flagged is True
    assert result.max_cramers_v is not None
    assert result.max_cramers_v > 0.50


def test_redundancy_screen_numeric_candidate_vs_cat_only_adopted() -> None:
    """Numeric candidate with no numeric adopted columns: pairwise corr skipped, not flagged."""
    candidate = pd.Series([1.0, 2.0, 3.0], name="num_feat")
    adopted = pd.DataFrame({"cat_col": ["A", "B", "C"]})
    result = redundancy_screen(candidate, adopted)
    assert result.flagged is False
    assert result.max_corr is None


# ---------------------------------------------------------------------------
# candidate_importance
# ---------------------------------------------------------------------------


def test_candidate_importance_planted_signal_above_decoy(
    discovery_data: tuple[pd.DataFrame, pd.Series],
    numeric_pre: object,
    dt: DecisionTreeClassifier,
) -> None:
    """Planted-signal candidate has permutation importance strictly above the decoy floor."""
    X, y = discovery_data
    result = candidate_importance(
        X,
        y,
        numeric_pre,
        candidate_col="candidate_good",
        decoy_col="decoy",
        n_repeats=5,
        model=dt,
    )
    assert result.above_floor is True
    assert result.candidate_importance > result.noise_floor


def test_candidate_importance_returns_importance_result_type(
    discovery_data: tuple[pd.DataFrame, pd.Series],
    numeric_pre: object,
    dt: DecisionTreeClassifier,
) -> None:
    """Return type is ImportanceResult with the three required typed fields."""
    X, y = discovery_data
    result = candidate_importance(
        X,
        y,
        numeric_pre,
        candidate_col="candidate_good",
        decoy_col="decoy",
        n_repeats=5,
        model=dt,
    )
    assert isinstance(result, ImportanceResult)
    assert isinstance(result.candidate_importance, float)
    assert isinstance(result.noise_floor, float)
    assert isinstance(result.above_floor, bool)


def test_candidate_importance_pure_noise_candidate_is_below_floor(
    discovery_data: tuple[pd.DataFrame, pd.Series],
    numeric_pre: object,
    dt: DecisionTreeClassifier,
) -> None:
    """Roles swapped: score the pure-noise column as the candidate against the
    planted-signal column as the reference floor. A noise candidate cannot
    clear a floor set by a column the model actually relies on -- drives
    candidate_importance's own above_floor=False branch directly, rather than
    only through adoption_gate's ImportanceResult(above_floor=False) fixture."""
    X, y = discovery_data
    result = candidate_importance(
        X,
        y,
        numeric_pre,
        candidate_col="decoy",
        decoy_col="candidate_good",
        n_repeats=5,
        model=dt,
    )
    assert result.above_floor is False
    assert result.candidate_importance <= result.noise_floor


# ---------------------------------------------------------------------------
# adoption_gate
# ---------------------------------------------------------------------------


def test_adoption_gate_screen1_rejects_unavailable_feature() -> None:
    """Screen 1 rejects immediately when a required column is absent at serving time."""
    decision = adoption_gate(
        _fail_serving("oof_score"),
        _ok_redundancy(),
        prior_pr_auc=0.60,
        new_pr_auc=0.63,
        sub_recall_delta=0.15,
        importance_result=_ok_importance(),
    )
    assert decision.adopted is False
    assert decision.rejection_screen == 1
    assert "oof_score" in (decision.rejection_reason or "")


def test_adoption_gate_screen2_rejects_redundant_with_failing_swap() -> None:
    """Screen 2 rejects when the candidate is redundant and does not beat the swap recall baseline."""
    decision = adoption_gate(
        _ok_serving(),
        _flagged_redundancy(),
        prior_pr_auc=0.60,
        new_pr_auc=0.63,
        sub_recall_delta=0.05,  # does not exceed swap_sub_recall_delta=0.10
        importance_result=_ok_importance(),
        swap_sub_recall_delta=0.10,
    )
    assert decision.adopted is False
    assert decision.rejection_screen == 2


def test_adoption_gate_screen2_redundant_no_swap_continues() -> None:
    """A redundant candidate with no swap baseline is not auto-rejected at screen 2."""
    decision = adoption_gate(
        _ok_serving(),
        _flagged_redundancy(),
        prior_pr_auc=0.60,
        new_pr_auc=0.63,
        sub_recall_delta=0.15,
        importance_result=_ok_importance(),
        swap_sub_recall_delta=None,
    )
    assert decision.adopted is True


def test_adoption_gate_screen2_null_recall_delta_with_swap_rejects() -> None:
    """sub_recall_delta=None with a swap baseline rejects at screen 2 — unknown recall
    cannot clear a redundancy flag; treat as failing the swap comparison."""
    decision = adoption_gate(
        _ok_serving(),
        _flagged_redundancy(),
        prior_pr_auc=0.60,
        new_pr_auc=0.63,
        sub_recall_delta=None,
        importance_result=_ok_importance(),
        swap_sub_recall_delta=0.05,
    )
    assert decision.adopted is False
    assert decision.rejection_screen == 2


def test_adoption_gate_null_recall_delta_no_swap_passes_screen2() -> None:
    """sub_recall_delta=None without a swap baseline does not trigger Screen 2 rejection."""
    decision = adoption_gate(
        _ok_serving(),
        _flagged_redundancy(),
        prior_pr_auc=0.60,
        new_pr_auc=0.63,
        sub_recall_delta=None,
        importance_result=_ok_importance(),
        swap_sub_recall_delta=None,
    )
    assert decision.adopted is True


def test_adoption_gate_screen3_rejects_pr_auc_below_prior() -> None:
    """Screen 3 rejects when new PR-AUC falls below the prior PR-AUC."""
    decision = adoption_gate(
        _ok_serving(),
        _ok_redundancy(),
        prior_pr_auc=0.60,
        new_pr_auc=0.55,  # below prior 0.60
        sub_recall_delta=0.15,
        importance_result=_ok_importance(),
    )
    assert decision.adopted is False
    assert decision.rejection_screen == 3


def test_adoption_gate_screen3_flat_pr_auc_rejects() -> None:
    """Flat PR-AUC equal to the prior rejects at screen 3 — no strict improvement means no adoption."""
    prior_pr_auc = 0.63
    decision = adoption_gate(
        _ok_serving(),
        _ok_redundancy(),
        prior_pr_auc=prior_pr_auc,
        new_pr_auc=prior_pr_auc,  # identical to prior
        sub_recall_delta=0.12,
        importance_result=_ok_importance(),
    )
    assert decision.adopted is False
    assert decision.rejection_screen == 3


def test_adoption_gate_screen3_rejects_below_min_delta() -> None:
    """Screen 3 rejects when PR-AUC improves but the delta is below MIN_PR_AUC_DELTA."""
    decision = adoption_gate(
        _ok_serving(),
        _ok_redundancy(),
        prior_pr_auc=0.60,
        new_pr_auc=0.60 + MIN_PR_AUC_DELTA / 2,  # positive delta but below threshold
        sub_recall_delta=0.10,
        importance_result=_ok_importance(),
    )
    assert decision.adopted is False
    assert decision.rejection_screen == 3
    assert "below minimum" in (decision.rejection_reason or "")


def test_adoption_gate_screen3_improvement_passes() -> None:
    """New PR-AUC above prior by at least MIN_PR_AUC_DELTA is a clear adoption signal."""
    decision = adoption_gate(
        _ok_serving(),
        _ok_redundancy(),
        prior_pr_auc=0.60,
        new_pr_auc=0.72,
        sub_recall_delta=0.12,
        importance_result=_ok_importance(),
    )
    assert decision.adopted is True


def test_adoption_gate_screen4_rejects_below_noise_floor() -> None:
    """Screen 4 rejects when permutation importance does not exceed the decoy noise floor."""
    decision = adoption_gate(
        _ok_serving(),
        _ok_redundancy(),
        prior_pr_auc=0.60,
        new_pr_auc=0.63,
        sub_recall_delta=0.15,
        importance_result=_fail_importance(),
    )
    assert decision.adopted is False
    assert decision.rejection_screen == 4


def test_adoption_gate_adopts_when_all_screens_pass() -> None:
    """Candidate that passes all four screens is adopted with no rejection metadata."""
    decision = adoption_gate(
        _ok_serving(),
        _ok_redundancy(),
        prior_pr_auc=0.60,
        new_pr_auc=0.63,
        sub_recall_delta=0.15,
        importance_result=_ok_importance(),
    )
    assert decision.adopted is True
    assert decision.rejection_screen is None
    assert decision.rejection_reason is None


def test_adoption_gate_returns_adoption_decision_type() -> None:
    """Return type is AdoptionDecision in both adopted and rejected cases."""
    accepted = adoption_gate(
        _ok_serving(),
        _ok_redundancy(),
        prior_pr_auc=0.60,
        new_pr_auc=0.63,
        sub_recall_delta=0.15,
        importance_result=_ok_importance(),
    )
    rejected = adoption_gate(
        _fail_serving(),
        _ok_redundancy(),
        prior_pr_auc=0.60,
        new_pr_auc=0.63,
        sub_recall_delta=0.15,
        importance_result=_ok_importance(),
    )
    assert isinstance(accepted, AdoptionDecision)
    assert isinstance(rejected, AdoptionDecision)


# ---------------------------------------------------------------------------
# bootstrap_pr_auc_ci
# ---------------------------------------------------------------------------


def test_bootstrap_ci_lower_le_upper() -> None:
    """CI lower bound is always ≤ upper bound regardless of input quality."""
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 50)
    scores = rng.uniform(0.0, 1.0, size=100).astype(np.float64)
    lo, hi = bootstrap_pr_auc_ci(
        y.astype(np.int_), scores, n_iterations=200, random_state=0
    )
    assert lo <= hi


def test_bootstrap_ci_perfect_predictions_tight_near_one() -> None:
    """Near-perfect predictions yield a CI tightly concentrated above 0.95."""
    n = 200
    y = np.repeat([0, 1], n // 2)
    scores = (
        y.astype(float) + np.random.default_rng(0).normal(0, 0.01, size=n)
    ).astype(np.float64)
    lo, _ = bootstrap_pr_auc_ci(
        y.astype(np.int_), scores, n_iterations=500, random_state=42
    )
    assert lo > 0.95


def test_bootstrap_ci_reproducible_with_same_seed() -> None:
    """Two calls with the same random_state produce identical lower and upper bounds."""
    y = np.repeat([0, 1], 50).astype(np.int_)
    scores = np.random.default_rng(5).uniform(0.0, 1.0, size=100).astype(np.float64)
    ci1 = bootstrap_pr_auc_ci(y, scores, n_iterations=200, random_state=99)
    ci2 = bootstrap_pr_auc_ci(y, scores, n_iterations=200, random_state=99)
    assert ci1 == ci2


def test_bootstrap_ci_handles_single_class_resamples() -> None:
    """Function returns valid bounds even when some resamples land on a single class."""
    # Heavy imbalance + small n maximises single-class resample probability
    y = np.array([0, 0, 0, 0, 0, 1], dtype=np.int_)
    scores = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.9], dtype=np.float64)
    lo, hi = bootstrap_pr_auc_ci(y, scores, n_iterations=500, random_state=0)
    assert 0.0 <= lo <= hi <= 1.0


def test_bootstrap_ci_all_single_class_returns_zero_zero() -> None:
    """n=1 forces every resample to be single-class; guard returns (0.0, 0.0) not IndexError."""
    y = np.array([1], dtype=np.int_)
    scores = np.array([0.9], dtype=np.float64)
    lo, hi = bootstrap_pr_auc_ci(y, scores, n_iterations=50, random_state=0)
    assert lo == 0.0
    assert hi == 0.0


# ---------------------------------------------------------------------------
# write_provenance
# ---------------------------------------------------------------------------


def test_write_provenance_creates_json_file(tmp_path: Path) -> None:
    """write_provenance creates the specified output JSON file."""
    out = tmp_path / "reports" / "provenance.json"
    write_provenance([], [], out, discovery_threshold=0.27)
    assert out.exists()


def test_write_provenance_json_top_level_keys(tmp_path: Path) -> None:
    """JSON output has run_config, adopted_features, and laps at the top level."""
    out = tmp_path / "provenance.json"
    write_provenance(
        [_make_lap()], ["is_long_month_to_month"], out, discovery_threshold=0.27
    )
    payload = json.loads(out.read_text())
    assert "run_config" in payload
    assert "adopted_features" in payload
    assert "laps" in payload
    assert payload["adopted_features"] == ["is_long_month_to_month"]
    assert len(payload["laps"]) == 1


def test_write_provenance_run_config_surfaces_gate_constants(tmp_path: Path) -> None:
    """run_config contains all gate threshold constants at their module-level values."""
    out = tmp_path / "provenance.json"
    write_provenance([], [], out, discovery_threshold=0.27)
    cfg = json.loads(out.read_text())["run_config"]
    assert "CORR_THRESHOLD" in cfg
    assert "CRAMERS_V_THRESHOLD" in cfg
    assert "IMPORTANCE_NOISE_FLOOR_MARGIN" in cfg
    assert "VIF_THRESHOLD" not in cfg
    assert "MIN_SUB_GAIN" not in cfg
    assert "MIN_PR_AUC_DELTA" in cfg
    assert "discovery_threshold" in cfg


def test_write_provenance_creates_companion_md(tmp_path: Path) -> None:
    """A provenance.md summary is written alongside the JSON."""
    out = tmp_path / "provenance.json"
    write_provenance(
        [_make_lap()], ["is_long_month_to_month"], out, discovery_threshold=0.27
    )
    md = tmp_path / "provenance.md"
    assert md.exists()
    assert "is_long_month_to_month" in md.read_text()


def test_write_provenance_empty_laps_writes_valid_json(tmp_path: Path) -> None:
    """write_provenance with no laps produces a valid JSON file with empty lists."""
    out = tmp_path / "provenance.json"
    write_provenance([], [], out, discovery_threshold=0.27)
    payload = json.loads(out.read_text())
    assert payload["laps"] == []
    assert payload["adopted_features"] == []


# ---------------------------------------------------------------------------
# backward_elimination
# ---------------------------------------------------------------------------


def test_backward_elimination_empty_adopted_returns_empty(
    discovery_data: tuple[pd.DataFrame, pd.Series],
    dt: DecisionTreeClassifier,
) -> None:
    """Empty adopted list returns an empty result list without fitting any model."""
    X, y = discovery_data
    results = backward_elimination(
        X=X,
        y=y,
        adopted=[],
        full_pr_auc=0.7,
        build_preprocessor_fn=build_preprocessor,
        binary=[],
        multi_cat=[],
        numeric=list(X.columns),
        model=dt,
    )
    assert results == []


def test_backward_elimination_keeps_useful_feature(
    discovery_data: tuple[pd.DataFrame, pd.Series],
    oof_proba: np.ndarray,
    dt: DecisionTreeClassifier,
) -> None:
    """Removing candidate_good hurts PR-AUC by at least MIN_PR_AUC_DELTA — it is kept."""
    X, y = discovery_data
    full_pr_auc = float(average_precision_score(y, oof_proba))

    results = backward_elimination(
        X=X,
        y=y,
        adopted=["candidate_good"],
        full_pr_auc=full_pr_auc,
        build_preprocessor_fn=build_preprocessor,
        binary=[],
        multi_cat=[],
        numeric=list(X.columns),
        model=dt,
    )
    assert len(results) == 1
    r = results[0]
    assert r.feature == "candidate_good"
    assert r.keep is True
    assert r.delta >= MIN_PR_AUC_DELTA


def test_backward_elimination_drops_noise_feature(
    discovery_data: tuple[pd.DataFrame, pd.Series],
    oof_proba: np.ndarray,
    dt: DecisionTreeClassifier,
) -> None:
    """Removing the decoy does not hurt PR-AUC by MIN_PR_AUC_DELTA — it is dropped."""
    X, y = discovery_data
    full_pr_auc = float(average_precision_score(y, oof_proba))

    results = backward_elimination(
        X=X,
        y=y,
        adopted=["decoy"],
        full_pr_auc=full_pr_auc,
        build_preprocessor_fn=build_preprocessor,
        binary=[],
        multi_cat=[],
        numeric=list(X.columns),
        model=dt,
    )
    assert len(results) == 1
    r = results[0]
    assert r.feature == "decoy"
    assert r.keep is False


def test_backward_elimination_result_fields_and_delta_consistency(
    discovery_data: tuple[pd.DataFrame, pd.Series],
    oof_proba: np.ndarray,
    dt: DecisionTreeClassifier,
) -> None:
    """BackwardEliminationResult has typed fields; delta equals full_pr_auc - pr_auc_without."""
    X, y = discovery_data
    full_pr_auc = float(average_precision_score(y, oof_proba))

    results = backward_elimination(
        X=X,
        y=y,
        adopted=["decoy"],
        full_pr_auc=full_pr_auc,
        build_preprocessor_fn=build_preprocessor,
        binary=[],
        multi_cat=[],
        numeric=list(X.columns),
        model=dt,
    )
    r = results[0]
    assert isinstance(r, BackwardEliminationResult)
    assert isinstance(r.pr_auc_without, float)
    assert isinstance(r.delta, float)
    assert isinstance(r.keep, bool)
    assert r.delta == pytest.approx(full_pr_auc - r.pr_auc_without)

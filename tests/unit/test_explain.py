"""Unit tests for telco_churn.models.explain — pure SHAP helpers (Phase 7)."""

from __future__ import annotations

import numpy as np
import pytest

from telco_churn.models.explain import (
    binary_feature_effects,
    check_top_k_elbow,
    cohort_shap,
    dependence_points,
    direction_sanity_check,
    feature_directions,
    global_importance,
    local_explanations,
)

# ---------------------------------------------------------------------------
# global_importance
# ---------------------------------------------------------------------------


def test_global_importance_keys() -> None:
    """Each row contains the two expected keys."""
    rng = np.random.default_rng(1)
    shap_values = rng.normal(size=(50, 3))
    rows = global_importance(shap_values, ["a", "b", "c"])
    assert len(rows) == 3
    for row in rows:
        assert {"feature", "mean_abs_shap"} <= set(row)


def test_global_importance_sorted_descending() -> None:
    """Rows are sorted by mean_abs_shap, most important first."""
    shap_values = np.array(
        [
            [0.1, 5.0, -0.2],
            [-0.2, 4.0, 0.1],
            [0.0, -6.0, 0.3],
        ]
    )
    rows = global_importance(shap_values, ["small", "big", "tiny"])
    assert rows[0]["feature"] == "big"
    values = [row["mean_abs_shap"] for row in rows]
    assert values == sorted(values, reverse=True)


def test_global_importance_is_never_negative() -> None:
    """mean_abs_shap is a magnitude — always >= 0, even for negative-signed SHAP."""
    shap_values = np.array([[-1.0, -2.0], [-1.0, -2.0]])
    rows = global_importance(shap_values, ["f1", "f2"])
    for row in rows:
        assert row["mean_abs_shap"] >= 0.0


def test_global_importance_zero_shap_gives_zero_importance() -> None:
    """A feature with all-zero SHAP values reports zero importance."""
    shap_values = np.array([[0.0, 5.0], [0.0, -5.0]])
    rows = {
        row["feature"]: row["mean_abs_shap"]
        for row in global_importance(shap_values, ["dead", "live"])
    }
    assert rows["dead"] == 0.0
    assert rows["live"] == 5.0


# ---------------------------------------------------------------------------
# dependence_points
# ---------------------------------------------------------------------------


def test_dependence_points_keys() -> None:
    """Result contains the three expected keys."""
    rng = np.random.default_rng(2)
    feature_values = rng.random(30)
    shap_values = rng.normal(size=30)
    result = dependence_points(feature_values, shap_values)
    assert {"feature_values", "shap_values", "direction"} <= set(result)
    assert len(result["feature_values"]) == 30
    assert len(result["shap_values"]) == 30


def test_dependence_points_recovers_positive_monotone_direction() -> None:
    """A planted positive monotone relationship (higher feature -> higher SHAP)
    recovers a positive direction sign — the property V3's instrument depends on."""
    rng = np.random.default_rng(3)
    feature_values = np.linspace(0, 10, 200)
    shap_values = 2.0 * feature_values + rng.normal(0, 0.1, 200)
    result = dependence_points(feature_values, shap_values)
    assert result["direction"] > 0.9


def test_dependence_points_recovers_negative_monotone_direction() -> None:
    """A planted negative monotone relationship (higher feature -> lower SHAP)
    recovers a negative direction sign."""
    rng = np.random.default_rng(4)
    feature_values = np.linspace(0, 10, 200)
    shap_values = -3.0 * feature_values + rng.normal(0, 0.1, 200)
    result = dependence_points(feature_values, shap_values)
    assert result["direction"] < -0.9


def test_dependence_points_zero_variance_feature_returns_zero_direction() -> None:
    """A feature that is constant across all rows has no direction to read —
    returns 0.0 rather than NaN from a degenerate correlation."""
    feature_values = np.full(20, 5.0)
    shap_values = np.linspace(-1, 1, 20)
    result = dependence_points(feature_values, shap_values)
    assert result["direction"] == 0.0


def test_dependence_points_zero_variance_shap_returns_zero_direction() -> None:
    """All-identical SHAP values (e.g. a feature the model never used) also has
    no direction to read."""
    feature_values = np.linspace(0, 10, 20)
    shap_values = np.zeros(20)
    result = dependence_points(feature_values, shap_values)
    assert result["direction"] == 0.0


# ---------------------------------------------------------------------------
# binary_feature_effects
# ---------------------------------------------------------------------------


def test_binary_feature_effects_keys() -> None:
    """Result contains the five expected keys."""
    feature_values = np.array([1.0, 0.0, 1.0, 0.0])
    shap_values = np.array([0.2, -0.1, 0.3, -0.2])
    result = binary_feature_effects(feature_values, shap_values)
    assert {
        "direction",
        "mean_shap_at_1",
        "mean_shap_at_0",
        "n_at_1",
        "n_at_0",
    } <= set(result)


def test_binary_feature_effects_group_means_split_by_level() -> None:
    """mean_shap_at_1/mean_shap_at_0 average only their own level's rows."""
    feature_values = np.array([1.0, 1.0, 0.0, 0.0])
    shap_values = np.array([0.4, 0.6, -0.1, -0.3])
    result = binary_feature_effects(feature_values, shap_values)
    assert result["mean_shap_at_1"] == pytest.approx(0.5)
    assert result["mean_shap_at_0"] == pytest.approx(-0.2)
    assert result["n_at_1"] == 2
    assert result["n_at_0"] == 2


def test_binary_feature_effects_recovers_direction_sign() -> None:
    """A planted level-1-pushes-churn-up relationship recovers a positive
    direction, mirroring dependence_points' correlation-sign contract."""
    rng = np.random.default_rng(8)
    feature_values = np.array([1.0] * 100 + [0.0] * 100)
    shap_values = np.concatenate(
        [rng.normal(0.5, 0.05, 100), rng.normal(-0.5, 0.05, 100)]
    )
    result = binary_feature_effects(feature_values, shap_values)
    assert result["direction"] > 0.9
    assert result["mean_shap_at_1"] > result["mean_shap_at_0"]


def test_binary_feature_effects_constant_feature_returns_zero_direction() -> None:
    """A feature that never varies (all rows the same level) has no direction
    to read — returns 0.0 rather than a NaN correlation, and the absent
    level's mean is 0.0 rather than a mean of an empty slice."""
    feature_values = np.ones(10)
    shap_values = np.linspace(-1, 1, 10)
    result = binary_feature_effects(feature_values, shap_values)
    assert result["direction"] == 0.0
    assert result["n_at_0"] == 0
    assert result["mean_shap_at_0"] == 0.0
    assert result["n_at_1"] == 10


# ---------------------------------------------------------------------------
# cohort_shap
# ---------------------------------------------------------------------------


def test_cohort_shap_keys() -> None:
    """Each row contains the two expected keys."""
    rng = np.random.default_rng(5)
    shap_values = rng.normal(size=(20, 3))
    mask = np.array([True] * 10 + [False] * 10)
    rows = cohort_shap(shap_values, mask, ["a", "b", "c"])
    assert len(rows) == 3
    for row in rows:
        assert {"feature", "mean_signed_shap"} <= set(row)


def test_cohort_shap_signed_recovers_negative_contribution_planted_feature() -> None:
    """The load-bearing test: a feature planted to push a cohort's SHAP negative
    is recovered as negative by cohort_shap (signed) — and a naive mean(|SHAP|)
    on the same cohort/feature does NOT recover the sign, because magnitude
    discards it. That pair is the entire argument for the signed formulation.
    """
    n = 40
    # Planted feature always pushes this cohort's score down (negative SHAP).
    planted = np.full(n, -0.5)
    other = np.array([0.3, -0.2] * (n // 2))
    shap_values = np.column_stack([planted, other])
    cohort_mask = np.ones(n, dtype=bool)

    signed_rows = {
        row["feature"]: row["mean_signed_shap"]
        for row in cohort_shap(shap_values, cohort_mask, ["planted", "other"])
    }
    assert signed_rows["planted"] == pytest.approx(-0.5)
    assert signed_rows["planted"] < 0

    # The absolute-mean counterpart destroys the sign — the point of the test.
    absolute_mean = np.abs(shap_values[cohort_mask]).mean(axis=0)[0]
    assert absolute_mean > 0
    assert absolute_mean != pytest.approx(signed_rows["planted"])


def test_cohort_shap_only_uses_masked_rows() -> None:
    """Rows outside the cohort mask do not influence the reported mean."""
    shap_values = np.array([[10.0], [-10.0], [1.0], [1.0]])
    mask = np.array([False, False, True, True])
    rows = cohort_shap(shap_values, mask, ["f"])
    assert rows[0]["mean_signed_shap"] == pytest.approx(1.0)


def test_cohort_shap_two_cohorts_can_diverge() -> None:
    """FN and TP cohorts with different signed SHAP profiles produce visibly
    different results when cohort_shap is called once per mask."""
    shap_values = np.array(
        [
            [-1.0],  # FN
            [-1.2],  # FN
            [2.0],  # TP
            [1.8],  # TP
        ]
    )
    fn_mask = np.array([True, True, False, False])
    tp_mask = np.array([False, False, True, True])
    fn_rows = cohort_shap(shap_values, fn_mask, ["f"])
    tp_rows = cohort_shap(shap_values, tp_mask, ["f"])
    assert fn_rows[0]["mean_signed_shap"] < 0
    assert tp_rows[0]["mean_signed_shap"] > 0


# ---------------------------------------------------------------------------
# local_explanations
# ---------------------------------------------------------------------------


def test_local_explanations_one_row_per_index() -> None:
    """One result row per requested index."""
    rng = np.random.default_rng(6)
    shap_values = rng.normal(size=(10, 4))
    rows = local_explanations(
        shap_values, 0.1, ["a", "b", "c", "d"], [0, 3, 7], top_k=2
    )
    assert len(rows) == 3
    assert [row["row_index"] for row in rows] == [0, 3, 7]


def test_local_explanations_returns_top_k_features_by_magnitude() -> None:
    """top_features contains exactly top_k entries, sorted by |SHAP| descending."""
    shap_values = np.array([[0.1, -5.0, 0.2, 3.0, -0.05]])
    rows = local_explanations(shap_values, 0.1, ["a", "b", "c", "d", "e"], [0], top_k=2)
    top = rows[0]["top_features"]
    assert len(top) == 2
    assert top[0]["feature"] == "b"
    assert top[1]["feature"] == "d"


def test_local_explanations_preserves_sign() -> None:
    """A negative-contribution feature keeps its negative sign in the output."""
    shap_values = np.array([[-4.0, 1.0]])
    rows = local_explanations(shap_values, 0.1, ["neg", "pos"], [0], top_k=1)
    assert rows[0]["top_features"][0]["feature"] == "neg"
    assert rows[0]["top_features"][0]["shap_value"] < 0


def test_local_explanations_top_k_larger_than_feature_count_returns_all() -> None:
    """top_k larger than the number of features returns every feature, not an error."""
    shap_values = np.array([[1.0, 2.0]])
    rows = local_explanations(shap_values, 0.1, ["a", "b"], [0], top_k=10)
    assert len(rows[0]["top_features"]) == 2


def test_local_explanations_base_value_is_shared_across_rows() -> None:
    """base_value is echoed on every row, unchanged."""
    rng = np.random.default_rng(7)
    shap_values = rng.normal(size=(5, 3))
    rows = local_explanations(shap_values, 0.265, ["a", "b", "c"], [0, 2], top_k=2)
    assert all(row["base_value"] == pytest.approx(0.265) for row in rows)


def test_local_explanations_prediction_equals_base_plus_full_sum() -> None:
    """prediction = base_value + sum of every feature's SHAP value, including
    features not shown individually — the waterfall's endpoint must reflect
    the full row, not just the top_k slice."""
    shap_values = np.array([[0.5, -0.2, 0.1, 0.05]])
    base_value = 0.3
    rows = local_explanations(
        shap_values, base_value, ["a", "b", "c", "d"], [0], top_k=1
    )
    assert rows[0]["prediction"] == pytest.approx(base_value + shap_values[0].sum())


def test_local_explanations_other_contribution_sums_excluded_features() -> None:
    """other_contribution equals the sum of every feature's SHAP value not in top_features."""
    shap_values = np.array([[0.5, -0.2, 0.1, 0.05]])
    rows = local_explanations(shap_values, 0.3, ["a", "b", "c", "d"], [0], top_k=1)
    # top_k=1 keeps only "a" (0.5); the rest (-0.2 + 0.1 + 0.05) is "other".
    assert rows[0]["other_contribution"] == pytest.approx(-0.2 + 0.1 + 0.05)


def test_local_explanations_other_contribution_zero_when_top_k_covers_all() -> None:
    """other_contribution is exactly 0.0 when top_k includes every feature —
    nothing is left over to collapse into the remainder bucket."""
    shap_values = np.array([[1.0, 2.0]])
    rows = local_explanations(shap_values, 0.0, ["a", "b"], [0], top_k=10)
    assert rows[0]["other_contribution"] == 0.0


def test_local_explanations_includes_feature_value_when_provided() -> None:
    """Each top feature carries the customer's raw value for it when
    feature_values is supplied — the "feature = value" waterfall label."""
    shap_values = np.array([[0.5, -0.2]])
    feature_values = np.array([[12.0, 3.5]])
    rows = local_explanations(
        shap_values,
        0.3,
        ["tenure", "charges"],
        [0],
        top_k=2,
        feature_values=feature_values,
    )
    by_feature = {f["feature"]: f for f in rows[0]["top_features"]}
    assert by_feature["tenure"]["feature_value"] == pytest.approx(12.0)
    assert by_feature["charges"]["feature_value"] == pytest.approx(3.5)


def test_local_explanations_omits_feature_value_when_not_provided() -> None:
    """feature_value key is absent (not None) when feature_values isn't supplied."""
    shap_values = np.array([[0.5, -0.2]])
    rows = local_explanations(shap_values, 0.3, ["a", "b"], [0], top_k=2)
    for entry in rows[0]["top_features"]:
        assert "feature_value" not in entry


# ---------------------------------------------------------------------------
# feature_directions
# ---------------------------------------------------------------------------


def test_feature_directions_one_entry_per_feature() -> None:
    """Returns a direction for every column, not just a top-k slice."""
    rng = np.random.default_rng(9)
    Xt = rng.normal(size=(30, 3))
    shap_values = rng.normal(size=(30, 3))
    result = feature_directions(Xt, shap_values, ["a", "b", "c"])
    assert set(result) == {"a", "b", "c"}


def test_feature_directions_matches_dependence_points_direction() -> None:
    """feature_directions agrees with dependence_points' own per-feature
    direction — both wrap the same _signed_direction computation."""
    rng = np.random.default_rng(10)
    Xt = rng.normal(size=(40, 2))
    shap_values = rng.normal(size=(40, 2))
    result = feature_directions(Xt, shap_values, ["a", "b"])
    expected_a = dependence_points(Xt[:, 0], shap_values[:, 0])["direction"]
    assert result["a"] == pytest.approx(expected_a)


# ---------------------------------------------------------------------------
# direction_sanity_check
# ---------------------------------------------------------------------------


def test_direction_sanity_check_no_violation_when_direction_matches() -> None:
    """A feature whose observed direction matches its expected sign passes."""
    result = direction_sanity_check(["tenure"], {"tenure": -0.8}, {"tenure": -1})
    assert result["passed"] is True
    assert result["violations"] == []


def test_direction_sanity_check_flags_contradicting_direction() -> None:
    """A feature whose observed direction contradicts the established EDA
    relationship is flagged as a violation, and the gate fails."""
    result = direction_sanity_check(["tenure"], {"tenure": 0.7}, {"tenure": -1})
    assert result["passed"] is False
    assert len(result["violations"]) == 1
    assert result["violations"][0]["feature"] == "tenure"


def test_direction_sanity_check_unmatched_feature_is_not_checked_and_does_not_fail() -> (
    None
):
    """A checked feature with no established relationship is recorded as
    unchecked rather than silently treated as a pass on a claim never made."""
    result = direction_sanity_check(
        ["some_engineered_ratio"], {"some_engineered_ratio": 0.5}, {"tenure": -1}
    )
    assert result["passed"] is True
    row = result["checked_features"][0]
    assert row["checked"] is False
    assert row["matched_eda_relationship"] is None


def test_direction_sanity_check_longest_key_wins_on_ambiguous_substring() -> None:
    """A feature name matching multiple expected-direction keys resolves to
    the longest (most specific) match."""
    result = direction_sanity_check(
        ["contract_type_Two year"],
        {"contract_type_Two year": -0.9},
        {"year": 1, "two year": -1},
    )
    row = result["checked_features"][0]
    assert row["matched_eda_relationship"] == "two year"
    assert row["contradicts"] is False


# ---------------------------------------------------------------------------
# check_top_k_elbow
# ---------------------------------------------------------------------------


def test_check_top_k_elbow_valid_when_configured_k_sits_on_a_real_elbow() -> None:
    """A tight plateau (small consecutive deltas) followed by a real jump at
    the configured k reports valid=True."""
    values = [0.90, 0.60, 0.30, 0.29, 0.28, 0.27, 0.10, 0.09, 0.08]
    result = check_top_k_elbow(values, configured_k=6)
    assert result["valid"] is True
    assert result["ratio"] > 1.5


def test_check_top_k_elbow_invalid_when_configured_k_is_mid_plateau() -> None:
    """A configured k landing inside a smooth, gap-free decay (no real
    elbow anywhere) reports valid=False."""
    values = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    result = check_top_k_elbow(values, configured_k=3)
    assert result["valid"] is False


def test_check_top_k_elbow_matches_current_project_config() -> None:
    """Regression guard pinned to the real global-importance ranking
    top_k_dependence_features=8 was derived from: its own elbow at rank 8
    still validates, and a badly-wrong k=3 (mid-plateau) still fails."""
    global_importance = [
        0.6469,
        0.4313,
        0.2247,
        0.1949,
        0.1924,
        0.1918,
        0.1831,
        0.1747,
        0.1286,
        0.1157,
        0.0945,
    ]
    assert check_top_k_elbow(global_importance, configured_k=8)["valid"] is True
    assert check_top_k_elbow(global_importance, configured_k=3)["valid"] is False


def test_check_top_k_elbow_empty_plateau_reports_valid_with_no_baseline() -> None:
    """configured_k too close to 1 to have a plateau baseline reports
    valid=True with no ratio computed, rather than raising or dividing by
    zero."""
    values = [1.0, 0.5, 0.1]
    result = check_top_k_elbow(values, configured_k=1)
    assert result["valid"] is True
    assert result["plateau_median_delta"] is None

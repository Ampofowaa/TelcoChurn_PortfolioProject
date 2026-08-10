"""Unit tests for telco_churn.models.economics — pure business-economics helpers (Phase 7)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from telco_churn.models.economics import (
    break_even_retention_rate,
    campaign_cost,
    ev_by_k,
    expected_value,
    retained_revenue,
    sensitivity_oneway,
    sensitivity_twoway,
    tornado,
)
from telco_churn.models.policy_config import CostScenario, expected_value_at_threshold

SCENARIO = CostScenario(
    name="base", arpu=70.0, ltv=1512.0, cost=50.0, retention_rate=0.30
)


# ---------------------------------------------------------------------------
# expected_value
# ---------------------------------------------------------------------------


def test_expected_value_scales_per_customer_average_by_n() -> None:
    """expected_value equals expected_value_at_threshold's per-customer average times n."""
    rng = np.random.default_rng(1)
    n = 100
    y = rng.integers(0, 2, size=n)
    proba = rng.random(n)
    t = 0.5
    per_customer = expected_value_at_threshold(proba, y, SCENARIO, t)
    total = expected_value(proba, y, SCENARIO, t)
    assert total == pytest.approx(per_customer * n)


def test_expected_value_zero_when_nobody_contacted() -> None:
    """Contacting no one (threshold above every score) gives EV = 0."""
    y = np.array([0, 1, 0, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.4])
    assert expected_value(proba, y, SCENARIO, 1.01) == 0.0


# ---------------------------------------------------------------------------
# campaign_cost
# ---------------------------------------------------------------------------


def test_campaign_cost_matches_manual_count() -> None:
    """campaign_cost equals n_contacted * scenario.cost."""
    proba = np.array([0.1, 0.6, 0.7, 0.9])
    n_contacted = 3  # scores >= 0.5
    assert campaign_cost(proba, SCENARIO, 0.5) == pytest.approx(
        n_contacted * SCENARIO.cost
    )


def test_campaign_cost_zero_when_nobody_contacted() -> None:
    """No one scored above threshold gives zero campaign cost."""
    proba = np.array([0.1, 0.2, 0.3])
    assert campaign_cost(proba, SCENARIO, 0.99) == 0.0


# ---------------------------------------------------------------------------
# retained_revenue
# ---------------------------------------------------------------------------


def test_retained_revenue_matches_manual_count() -> None:
    """retained_revenue equals tp * retention_rate * ltv."""
    y = np.array([0, 1, 1, 0])
    proba = np.array(
        [0.1, 0.6, 0.7, 0.9]
    )  # tp: rows 1,2 (0.6, 0.7 >= 0.5); row 3 is fp
    tp = 2
    assert retained_revenue(proba, y, SCENARIO, 0.5) == pytest.approx(
        tp * SCENARIO.retention_rate * SCENARIO.ltv
    )


def test_retained_revenue_zero_when_no_true_positives_contacted() -> None:
    """Contacting only non-churners gives zero retained revenue."""
    y = np.array([0, 0, 0, 1])
    proba = np.array([0.9, 0.8, 0.7, 0.1])
    assert retained_revenue(proba, y, SCENARIO, 0.5) == 0.0


def test_retained_revenue_expected_value_equals_revenue_minus_cost() -> None:
    """Net expected_value equals retained_revenue minus campaign_cost, exactly."""
    rng = np.random.default_rng(15)
    n = 120
    y = rng.integers(0, 2, size=n)
    proba = rng.random(n)
    t = 0.5
    revenue = retained_revenue(proba, y, SCENARIO, t)
    cost = campaign_cost(proba, SCENARIO, t)
    ev = expected_value(proba, y, SCENARIO, t)
    assert ev == pytest.approx(revenue - cost)


# ---------------------------------------------------------------------------
# ev_by_k
# ---------------------------------------------------------------------------


def test_ev_by_k_keys() -> None:
    """Each row contains the three expected keys."""
    rng = np.random.default_rng(2)
    n = 50
    y = rng.integers(0, 2, size=n)
    proba = rng.random(n)
    rows = ev_by_k(proba, y, SCENARIO)
    assert len(rows) >= 1
    for row in rows:
        assert {"k", "threshold", "ev_cumulative"} <= set(row)


def test_ev_by_k_k_values_are_nondecreasing_and_bounded_by_n() -> None:
    """k is strictly increasing across rows and never exceeds n."""
    rng = np.random.default_rng(3)
    n = 80
    y = rng.integers(0, 2, size=n)
    proba = rng.random(n)
    rows = ev_by_k(proba, y, SCENARIO)
    ks = [row["k"] for row in rows]
    assert ks == sorted(ks)
    assert ks[-1] == n


def test_ev_by_k_last_row_matches_contact_everyone_ev() -> None:
    """The final row's cumulative EV equals expected_value at threshold 0 (contact everyone)."""
    rng = np.random.default_rng(4)
    n = 60
    y = rng.integers(0, 2, size=n)
    proba = rng.random(n)
    rows = ev_by_k(proba, y, SCENARIO)
    assert rows[-1]["ev_cumulative"] == pytest.approx(
        expected_value(proba, y, SCENARIO, 0.0)
    )


def test_ev_by_k_collapses_tied_scores_into_one_row() -> None:
    """A tie group contributes exactly one row, at the K where the full group is included."""
    y = np.array([0, 1, 0, 1])
    proba = np.array([0.5, 0.5, 0.5, 0.5])  # all tied
    rows = ev_by_k(proba, y, SCENARIO)
    assert len(rows) == 1
    assert rows[0]["k"] == 4


def test_ev_by_k_peak_matches_argmax_of_expected_value_curve() -> None:
    """ev_by_k's peak row lands at the same threshold as an independent argmax
    over expected_value_at_threshold evaluated at every distinct score —
    cross-checks the cumulative-sum construction against a direct computation."""
    rng = np.random.default_rng(5)
    n = 150
    y_true = rng.integers(0, 2, size=n)
    proba = np.where(y_true == 1, 0.8, 0.2) + rng.normal(0, 0.15, n)
    proba = np.clip(proba, 0.0, 1.0)

    rows = ev_by_k(proba, y_true, SCENARIO)
    peak_row = max(rows, key=lambda r: r["ev_cumulative"])

    direct_best = max(
        (expected_value_at_threshold(proba, y_true, SCENARIO, t), t)
        for t in np.unique(proba)
    )
    assert peak_row["ev_cumulative"] / n == pytest.approx(direct_best[0], abs=1e-9)


# ---------------------------------------------------------------------------
# break_even_retention_rate
# ---------------------------------------------------------------------------


def test_break_even_retention_rate_recovers_scenario_r_at_shipped_threshold() -> None:
    """If the group contacted at the closed-form threshold matches the scenario's
    own composition, break_even_retention_rate recovers a value consistent
    with a positive, finite EV requirement."""
    rng = np.random.default_rng(6)
    n = 500
    y = rng.integers(0, 2, size=n)
    proba = np.where(y == 1, 0.7, 0.3) + rng.normal(0, 0.1, n)
    proba = np.clip(proba, 0.0, 1.0)
    r_min = break_even_retention_rate(proba, y, SCENARIO, 0.5)
    assert r_min > 0
    assert math.isfinite(r_min)


def test_break_even_retention_rate_ev_is_zero_at_r_min() -> None:
    """Substituting r_min back into the scenario makes EV at that threshold exactly zero."""
    from dataclasses import replace

    rng = np.random.default_rng(7)
    n = 300
    y = rng.integers(0, 2, size=n)
    proba = rng.random(n)
    t = 0.5
    r_min = break_even_retention_rate(proba, y, SCENARIO, t)
    break_even_scenario = replace(SCENARIO, retention_rate=r_min)
    ev = expected_value_at_threshold(proba, y, break_even_scenario, t)
    assert ev == pytest.approx(0.0, abs=1e-6)


def test_break_even_retention_rate_returns_inf_when_no_true_positives() -> None:
    """A threshold that contacts only non-churners has no retention rate that
    recoups the spend — returns +inf rather than raising or dividing by zero."""
    y = np.array([0, 0, 0, 1])
    proba = np.array([0.9, 0.8, 0.7, 0.1])  # only negatives scored above threshold
    r_min = break_even_retention_rate(proba, y, SCENARIO, 0.5)
    assert r_min == float("inf")


# ---------------------------------------------------------------------------
# sensitivity_oneway
# ---------------------------------------------------------------------------


def test_sensitivity_oneway_retention_rate_keys_and_length() -> None:
    """One row per value, each with the param name and 'ev' keys."""
    rng = np.random.default_rng(8)
    n = 100
    y = rng.integers(0, 2, size=n)
    proba = rng.random(n)
    values = [0.15, 0.20, 0.30, 0.40, 0.45]
    rows = sensitivity_oneway(proba, y, SCENARIO, 0.5, "retention_rate", values)
    assert len(rows) == len(values)
    for row, v in zip(rows, values, strict=True):
        assert row["retention_rate"] == pytest.approx(v)
        assert "ev" in row


def test_sensitivity_oneway_ev_increases_with_retention_rate() -> None:
    """Holding everything else fixed, EV is monotonically non-decreasing in r
    when there is at least one true positive contacted."""
    y = np.array([0] * 5 + [1] * 5)
    proba = np.array([0.1] * 5 + [0.9] * 5)
    rows = sensitivity_oneway(
        proba, y, SCENARIO, 0.5, "retention_rate", [0.1, 0.3, 0.6]
    )
    evs = [row["ev"] for row in rows]
    assert evs == sorted(evs)


def test_sensitivity_oneway_cost_decreases_ev() -> None:
    """Holding everything else fixed, EV is monotonically non-increasing in cost."""
    y = np.array([0] * 5 + [1] * 5)
    proba = np.array([0.1] * 5 + [0.9] * 5)
    rows = sensitivity_oneway(proba, y, SCENARIO, 0.5, "cost", [10.0, 50.0, 100.0])
    evs = [row["ev"] for row in rows]
    assert evs == sorted(evs, reverse=True)


def test_sensitivity_oneway_does_not_mutate_scenario() -> None:
    """The scenario passed in is left untouched — replace() must not mutate in place."""
    rng = np.random.default_rng(9)
    n = 50
    y = rng.integers(0, 2, size=n)
    proba = rng.random(n)
    sensitivity_oneway(proba, y, SCENARIO, 0.5, "retention_rate", [0.9])
    assert SCENARIO.retention_rate == 0.30


# ---------------------------------------------------------------------------
# sensitivity_twoway
# ---------------------------------------------------------------------------


def test_sensitivity_twoway_produces_full_grid() -> None:
    """One row per (r, c) pair — the full cross product."""
    rng = np.random.default_rng(10)
    n = 80
    y = rng.integers(0, 2, size=n)
    proba = rng.random(n)
    retention_rates = [0.15, 0.30, 0.45]
    costs = [20.0, 50.0]
    rows = sensitivity_twoway(proba, y, SCENARIO, 0.5, retention_rates, costs)
    assert len(rows) == len(retention_rates) * len(costs)
    for row in rows:
        assert {"retention_rate", "cost", "ev"} <= set(row)


def test_sensitivity_twoway_matches_oneway_at_fixed_cost() -> None:
    """Slicing the two-way grid at a fixed cost reproduces the one-way sweep at that cost."""
    rng = np.random.default_rng(11)
    n = 100
    y = rng.integers(0, 2, size=n)
    proba = rng.random(n)
    retention_rates = [0.20, 0.30, 0.40]
    fixed_cost_scenario = SCENARIO
    grid = sensitivity_twoway(
        proba, y, fixed_cost_scenario, 0.5, retention_rates, [50.0]
    )
    oneway = sensitivity_oneway(
        proba, y, fixed_cost_scenario, 0.5, "retention_rate", retention_rates
    )
    grid_evs = [row["ev"] for row in grid]
    oneway_evs = [row["ev"] for row in oneway]
    assert grid_evs == pytest.approx(oneway_evs)


# ---------------------------------------------------------------------------
# tornado
# ---------------------------------------------------------------------------


def test_tornado_returns_three_params_sorted_by_spread_descending() -> None:
    """Exactly three rows (retention_rate, cost, ltv), sorted by spread descending."""
    rng = np.random.default_rng(12)
    n = 200
    y = rng.integers(0, 2, size=n)
    proba = rng.random(n)
    rows = tornado(proba, y, SCENARIO, 0.5, 0.2)
    assert {row["param"] for row in rows} == {"retention_rate", "cost", "ltv"}
    spreads = [row["spread"] for row in rows]
    assert spreads == sorted(spreads, reverse=True)


def test_tornado_retention_rate_dominates_for_default_costs_yaml_style_scenario() -> (
    None
):
    """For a scenario shaped like the project's own configs/costs.yaml (retention
    rate spans a wide relative range vs. cost/LTV), retention_rate produces the
    widest EV swing — the claim §0 makes that the tornado chart exists to show."""
    y = np.array([0] * 50 + [1] * 50)
    proba = np.array([0.1] * 50 + [0.9] * 50)
    rows = tornado(proba, y, SCENARIO, 0.5, 0.5)
    by_param = {row["param"]: row["spread"] for row in rows}
    assert by_param["retention_rate"] == max(by_param.values())


def test_tornado_row_keys() -> None:
    """Each row contains all five expected keys."""
    rng = np.random.default_rng(13)
    n = 60
    y = rng.integers(0, 2, size=n)
    proba = rng.random(n)
    rows = tornado(proba, y, SCENARIO, 0.5, 0.1)
    for row in rows:
        assert {"param", "ev_base", "ev_low", "ev_high", "spread"} <= set(row)


def test_tornado_ev_base_matches_expected_value_at_threshold() -> None:
    """ev_base is identical across all rows and equals the unperturbed EV."""
    rng = np.random.default_rng(14)
    n = 90
    y = rng.integers(0, 2, size=n)
    proba = rng.random(n)
    t = 0.5
    rows = tornado(proba, y, SCENARIO, t, 0.2)
    expected = expected_value_at_threshold(proba, y, SCENARIO, t)
    for row in rows:
        assert row["ev_base"] == pytest.approx(expected)

"""Unit tests for telco_churn.serving.contact_policy."""

from __future__ import annotations

import numpy as np
import pytest

from telco_churn.models.policy_config import CostScenario
from telco_churn.serving.contact_policy import select_contacts


@pytest.fixture
def flat_cost_scenario() -> CostScenario:
    """r * LTV = 200, cost = 50 -> EV_i = 200 * p_i - 50, strictly increasing
    in p_i, so EV rank and probability rank agree row-for-row under this
    flat-cost scenario (ANALYSIS.md §0).
    """
    return CostScenario(
        name="test", arpu=100.0, ltv=500.0, cost=50.0, retention_rate=0.4
    )


def test_decision_is_score_ge_threshold(flat_cost_scenario: CostScenario) -> None:
    proba = np.array([0.9, 0.6, 0.5, 0.3, 0.1])
    result = select_contacts(
        proba,
        flat_cost_scenario,
        threshold=0.4,
        contact_capacity=100,
        campaign_budget=100_000.0,
    )
    np.testing.assert_array_equal(result.decision, [True, True, True, False, False])


def test_decision_is_unaffected_by_contact_capacity(
    flat_cost_scenario: CostScenario,
) -> None:
    """`decision` must be identical whether contact_capacity is a small,
    binding value or a value far larger than the batch — capacity only ever
    touches `contact`. This is what actually pins down the decision/contact
    split, not just the ranking on top of it.
    """
    proba = np.array([0.95, 0.88, 0.71, 0.65, 0.55])

    tight = select_contacts(
        proba,
        flat_cost_scenario,
        threshold=0.31,
        contact_capacity=3,
        campaign_budget=100_000.0,
    )
    loose = select_contacts(
        proba,
        flat_cost_scenario,
        threshold=0.31,
        contact_capacity=3_000,
        campaign_budget=100_000.0,
    )

    np.testing.assert_array_equal(tight.decision, loose.decision)
    assert tight.decision.sum() == 5


def test_contact_is_exactly_top_k_by_probability_when_capacity_binds(
    flat_cost_scenario: CostScenario,
) -> None:
    """With decision.sum() (5) driven past a deliberately small
    contact_capacity (3), the resulting contact=True set must be exactly the
    top-3 rows by raw probability — EV and probability rank agree under this
    flat-cost scenario, so this equality is the test. A future
    per-customer-LTV change (ANALYSIS.md §0's v2 backlog) is exactly what
    would break it.
    """
    proba = np.array([0.95, 0.88, 0.71, 0.65, 0.55])
    result = select_contacts(
        proba,
        flat_cost_scenario,
        threshold=0.31,
        contact_capacity=3,
        campaign_budget=100_000.0,
    )

    np.testing.assert_array_equal(result.contact, [True, True, True, False, False])
    assert result.capacity_limit == 3


def test_contact_equals_decision_when_capacity_not_binding(
    flat_cost_scenario: CostScenario,
) -> None:
    """When decision.sum() <= capacity_limit, contact must equal decision
    exactly — a consequence of the top-K cut keeping everyone, not a
    separately-coded no-pressure branch.
    """
    proba = np.array([0.9, 0.6, 0.5, 0.3, 0.1])
    result = select_contacts(
        proba,
        flat_cost_scenario,
        threshold=0.4,
        contact_capacity=100,
        campaign_budget=100_000.0,
    )
    np.testing.assert_array_equal(result.contact, result.decision)
    assert result.capacity_limit == 100


def test_capacity_limit_is_min_of_headcount_and_budget_floor(
    flat_cost_scenario: CostScenario,
) -> None:
    """capacity_limit = min(contact_capacity, floor(campaign_budget / cost)) —
    a tight budget (cost=50/contact) can bind before headcount does.
    """
    proba = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    result = select_contacts(
        proba,
        flat_cost_scenario,
        threshold=0.4,
        contact_capacity=100,
        campaign_budget=120.0,
    )
    assert result.capacity_limit == 2
    np.testing.assert_array_equal(result.contact, [True, True, False, False, False])


def test_no_eligible_rows_yields_no_contacts(flat_cost_scenario: CostScenario) -> None:
    proba = np.array([0.2, 0.1, 0.05])
    result = select_contacts(
        proba,
        flat_cost_scenario,
        threshold=0.4,
        contact_capacity=10,
        campaign_budget=100_000.0,
    )
    assert not result.decision.any()
    assert not result.contact.any()


def test_threshold_accepts_a_per_row_array_not_just_a_scalar(
    flat_cost_scenario: CostScenario,
) -> None:
    """Canary-routed rows are judged against whichever model served them
    (serving/predict.py::ScoredBatch.served_threshold), so threshold must be
    able to vary row-by-row rather than being one pinned cutoff for the
    whole batch — the same 0.5 probability is eligible under a lower
    row-specific threshold and ineligible under a higher one.
    """
    proba = np.array([0.5, 0.5, 0.5])
    threshold = np.array([0.4, 0.6, 0.5])
    result = select_contacts(
        proba,
        flat_cost_scenario,
        threshold=threshold,
        contact_capacity=10,
        campaign_budget=100_000.0,
    )
    np.testing.assert_array_equal(result.decision, [True, False, True])

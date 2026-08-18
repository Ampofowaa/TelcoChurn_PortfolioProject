"""Capacity-constrained contact policy — the serving-time decision mechanism
that turns a scored batch into who actually gets called this cycle.

`threshold.py` flags the trigger: when the implied contact rate exceeds the
retention team's capacity, the correct response is not a higher threshold
(which abandons the Bayes-optimal `t*` cut) but top-K by expected value
`p * r * LTV - c`. That response is serving-time, not evaluation-time, since
evaluation has no notion of operational capacity to consult — so it lives
here rather than in models/.

Under the current cost model (c, r, LTV all scenario-level constants), EV is
a strictly increasing function of p alone, so top-K-by-EV and top-K-by-score
are rank-identical — the divergence only appears once LTV varies per
customer (ANALYSIS.md §0's leading v2 item). Implementing the actual policy
now still matters: the capacity constraint and the K-cut are real today,
even though the ranking they operate over doesn't yet differ from a plain
score sort.

**Eligibility is evaluated per row against whichever model actually served
that row** (`serving/predict.py::ScoredBatch.served_threshold` — champion's
threshold for a champion-served row, challenger's own threshold for a
canary-routed one), never against a single pinned threshold. Pinning
eligibility to champion's threshold regardless of which model produced the
score would make a canary-routed customer's odds of passing eligibility
depend partly on an arbitrary hash-bucket assignment rather than purely on
their risk under the model that actually scored them, and would make
`predict.py`'s `predictions_above_threshold_total` Prometheus counter (which
already uses each row's own served threshold) disagree with the real
contact/no-contact outcome for canary-routed rows.

Per-row eligibility does not, by itself, make canary measurement immune to
capacity/budget interference between the champion and challenger cohorts —
see `ANALYSIS.md` §9, item 14.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from telco_churn.models.policy_config import CostScenario, expected_value_per_customer

__all__ = ["ContactDecision", "select_contacts"]


@dataclass(frozen=True)
class ContactDecision:
    """One batch's policy output: who the model+threshold flags, and who the
    capacity-constrained cycle actually has room to contact.

    `decision` answers "does the model+threshold say this customer is worth
    contacting" and never changes meaning. `contact` answers "will the
    retention team actually call them this cycle" and is the only field
    capacity ever touches. Kept as two separate arrays rather than one field
    reinterpreted per branch, so a caller filtering on either gets a fixed,
    unambiguous meaning regardless of whether capacity bound this call.
    """

    decision: NDArray[np.bool_]
    contact: NDArray[np.bool_]
    capacity_limit: int


def select_contacts(
    proba: NDArray[np.float64],
    scenario: CostScenario,
    threshold: NDArray[np.float64] | float,
    contact_capacity: int,
    campaign_budget: float,
) -> ContactDecision:
    """Rank the model+threshold's contact set by expected value, then cut it
    to whatever the cycle's operational capacity/budget actually allows.

    One unconditional computation, not a two-branch mechanism: `decision` is
    `proba >= threshold`, untouched by capacity. `contact` is `True` for the
    top `capacity_limit` of the decision-true set by descending EV, `False`
    for everyone else (including every decision-false row). When
    `decision.sum() <= capacity_limit`, the top-`capacity_limit` cut keeps
    everyone anyway — `contact` comes out identical to `decision` as a
    consequence of the ranking, not via a separate no-pressure branch.

    `threshold` is a per-row array in normal serving use — each row's own
    served model's threshold (see module docstring) — but a scalar is
    accepted too (broadcasts via numpy elementwise comparison) for callers
    with one threshold shared across the whole batch, e.g. tests.

    Ties in EV rank exactly where probability ties (flat costs), and are
    broken by original row order via a stable sort — deterministic across
    repeated calls on the same input, without introducing a customerid
    parameter this function doesn't otherwise need.
    """
    decision = proba >= threshold
    capacity_limit = min(
        contact_capacity, int(np.floor(campaign_budget / scenario.cost))
    )

    ev = expected_value_per_customer(proba, scenario)
    eligible = np.flatnonzero(decision)
    ranked = eligible[np.argsort(-ev[eligible], kind="stable")]

    contact = np.zeros_like(decision)
    contact[ranked[:capacity_limit]] = True

    return ContactDecision(
        decision=decision, contact=contact, capacity_limit=capacity_limit
    )

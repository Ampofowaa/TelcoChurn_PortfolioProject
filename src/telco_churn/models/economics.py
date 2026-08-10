"""Pure business-economics helpers for the sealed-test evaluation.

No I/O, no MLflow, no estimator — arrays, a CostScenario, and cost parameters
in; dollars and dicts out. models/evaluate.py is the only caller: it builds
reports/economics.json and the EV/sensitivity figures from these.

Reuses models.policy_config.CostScenario and expected_value_at_threshold
rather than redefining p·r·LTV − c a second time — two implementations would
agree only until one of them changed. That function returns the *per-customer
average* EV; the dollar-figure helpers below (expected_value, campaign_cost,
ev_by_k) return *totals* instead — the unit stakeholders actually want — kept
separate so neither call site has to multiply/divide by n itself.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from telco_churn.models.policy_config import CostScenario, expected_value_at_threshold

__all__ = [
    "break_even_retention_rate",
    "campaign_cost",
    "ev_by_k",
    "expected_value",
    "retained_revenue",
    "sensitivity_oneway",
    "sensitivity_twoway",
    "tornado",
]


def expected_value(
    proba: NDArray[np.float64],
    y: NDArray[np.int_],
    scenario: CostScenario,
    threshold: float,
) -> float:
    """Total dollar expected value from contacting everyone scored >= threshold.

    expected_value_at_threshold's per-customer average, scaled back up by n —
    the headline dollar figure a README or model card quotes.
    """
    return expected_value_at_threshold(proba, y, scenario, threshold) * len(y)


def campaign_cost(
    proba: NDArray[np.float64], scenario: CostScenario, threshold: float
) -> float:
    """Total campaign cost: n_contacted * scenario.cost.

    What the retention team actually spends running the campaign at this
    threshold — distinct from expected_value, which nets that spend against
    retained revenue.
    """
    n_contacted = int(np.sum(proba >= threshold))
    return float(n_contacted * scenario.cost)


def retained_revenue(
    proba: NDArray[np.float64],
    y: NDArray[np.int_],
    scenario: CostScenario,
    threshold: float,
) -> float:
    """Gross retained revenue: tp * scenario.retention_rate * scenario.ltv.

    campaign_cost's complement — net expected_value is their difference, and
    reporting both separately (rather than only the net figure) is what lets
    a stakeholder read "this costs $C to run and returns $R", the sentence
    Finance asks for, instead of just the number that nets them together.
    """
    contacted = proba >= threshold
    tp = int(np.sum(contacted & (y == 1)))
    return float(tp * scenario.retention_rate * scenario.ltv)


def ev_by_k(
    proba: NDArray[np.float64], y: NDArray[np.int_], scenario: CostScenario
) -> list[dict[str, float]]:
    """Cumulative dollar EV from contacting the top-K highest-scored customers.

    Rank customers by score and plot cumulative EV against K: at any budget
    the business reads off its own number, and the curve's unconstrained peak
    is a free check on the closed-form threshold (it should land at
    K = the count implied by t*).

    Same sorted-cumulative-sum construction as threshold.py's
    expected_value_curve (O(n log n)), denominated in contact count K instead
    of probability threshold. One row per distinct observed probability,
    matching that function's tie handling — a tied group can only be
    contacted all-or-nothing, so a K strictly inside one would be fiction.
    """
    order = np.argsort(-proba, kind="stable")
    proba_sorted = proba[order]
    y_sorted = y[order]
    n = len(y_sorted)

    tp_cum = np.cumsum(y_sorted)
    k_cum = np.arange(1, n + 1)
    fp_cum = k_cum - tp_cum
    ev_cumulative = (
        tp_cum * (scenario.retention_rate * scenario.ltv - scenario.cost)
        - fp_cum * scenario.cost
    )

    keep = np.r_[np.diff(proba_sorted) != 0, True]
    return [
        {"k": int(k), "threshold": float(t), "ev_cumulative": float(ev)}
        for k, t, ev in zip(
            k_cum[keep], proba_sorted[keep], ev_cumulative[keep], strict=True
        )
    ]


def break_even_retention_rate(
    proba: NDArray[np.float64],
    y: NDArray[np.int_],
    scenario: CostScenario,
    threshold: float,
) -> float:
    """The retention rate r_min at which contacting at `threshold` breaks even (EV = 0).

    Solved in closed form from ev(t) = tp·(r·LTV − c) − fp·c = 0:
    r_min = c·n_contacted / (tp·LTV). The campaign is worth running iff the
    true r >= r_min, and unlike every other number in this module, r_min is
    falsifiable after a single real campaign. tp=0 returns +inf (no retention
    rate recoups the spend) rather than dividing by zero.
    """
    contacted = proba >= threshold
    n_contacted = int(np.sum(contacted))
    tp = int(np.sum(contacted & (y == 1)))
    if tp == 0:
        return float("inf")
    return float(scenario.cost * n_contacted / (tp * scenario.ltv))


def sensitivity_oneway(
    proba: NDArray[np.float64],
    y: NDArray[np.int_],
    scenario: CostScenario,
    threshold: float,
    param: Literal["retention_rate", "cost"],
    values: list[float],
) -> list[dict[str, float]]:
    """EV at the fixed shipped threshold as one cost parameter varies, others held fixed.

    `param` is "retention_rate" or "cost". Swaps in each value via
    dataclasses.replace so no other field of `scenario` drifts. r is believed
    to be the most consequential parameter in the deployment decision; this
    sweep lets that claim be checked rather than taken on trust.
    """
    rows: list[dict[str, float]] = []
    for value in values:
        varied = (
            replace(scenario, retention_rate=float(value))
            if param == "retention_rate"
            else replace(scenario, cost=float(value))
        )
        rows.append(
            {
                param: float(value),
                "ev": expected_value_at_threshold(proba, y, varied, threshold),
            }
        )
    return rows


def sensitivity_twoway(
    proba: NDArray[np.float64],
    y: NDArray[np.int_],
    scenario: CostScenario,
    threshold: float,
    retention_rates: list[float],
    costs: list[float],
) -> list[dict[str, float]]:
    """EV at the fixed shipped threshold across a retention_rate x cost grid.

    The break-even frontier's underlying data — one row per (r, c) pair. The
    EV = 0 contour is whatever a plotting call draws through this grid, not
    computed here; this function only produces the grid.
    """
    return [
        {
            "retention_rate": float(r),
            "cost": float(c),
            "ev": expected_value_at_threshold(
                proba,
                y,
                replace(scenario, retention_rate=float(r), cost=float(c)),
                threshold,
            ),
        }
        for r in retention_rates
        for c in costs
    ]


def tornado(
    proba: NDArray[np.float64],
    y: NDArray[np.int_],
    scenario: CostScenario,
    threshold: float,
    pct_perturbation: float,
) -> list[dict[str, float | str]]:
    """Rank cost parameters by how far EV moves under a symmetric ± perturbation, one at a time.

    Perturbs retention_rate, cost, and ltv by pct_perturbation in each
    direction (others held at scenario's shipped values) and sorts by the
    resulting EV spread — making the "r dominates" claim visible rather than
    asserted in prose. arpu is excluded: it determines ltv and cost through
    resolve_scenario, so perturbing it too would double-count that assumption.
    """
    base_ev = expected_value_at_threshold(proba, y, scenario, threshold)
    rows: list[dict[str, float | str]] = []
    for param in ("retention_rate", "cost", "ltv"):
        base_value = float(getattr(scenario, param))
        low_value = base_value * (1 - pct_perturbation)
        high_value = base_value * (1 + pct_perturbation)
        if param == "retention_rate":
            low = replace(scenario, retention_rate=low_value)
            high = replace(scenario, retention_rate=high_value)
        elif param == "cost":
            low = replace(scenario, cost=low_value)
            high = replace(scenario, cost=high_value)
        else:
            low = replace(scenario, ltv=low_value)
            high = replace(scenario, ltv=high_value)
        ev_low = expected_value_at_threshold(proba, y, low, threshold)
        ev_high = expected_value_at_threshold(proba, y, high, threshold)
        rows.append(
            {
                "param": param,
                "ev_base": base_ev,
                "ev_low": ev_low,
                "ev_high": ev_high,
                "spread": abs(ev_high - ev_low),
            }
        )
    rows.sort(key=lambda row: row["spread"], reverse=True)
    return rows

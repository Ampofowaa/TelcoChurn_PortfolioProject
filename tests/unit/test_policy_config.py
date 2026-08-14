"""Unit tests for telco_churn.models.policy_config."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from telco_churn.models.policy_config import (
    CostScenario,
    costs_config_hash,
    expected_value_at_threshold,
    load_model_promotion_bars,
    resolve_policy_scenarios,
    resolve_policy_thresholds_by_scenario,
)
from telco_churn.models.threshold import expected_value_curve
from telco_churn.utils.paths import get_project_root


@pytest.fixture
def base_scenario() -> CostScenario:
    """A simple, hand-computable scenario: t* = 100 / (0.5 * 500) = 0.4."""
    return CostScenario(
        name="test", arpu=100.0, ltv=500.0, cost=100.0, retention_rate=0.5
    )


@pytest.fixture
def policy_fixture() -> OmegaConf:
    """Same shape as configs/policy/threshold.yaml's `scenarios` block."""
    return OmegaConf.create(
        {
            "scenarios": {
                "conservative": {
                    "threshold": 0.27,
                    "costs": {"c": 22.0, "r": 0.20, "ltv": 408.0, "arpu": 56.7},
                },
                "base": {
                    "threshold": 0.39,
                    "costs": {"c": 67.76, "r": 0.30, "ltv": 573.12, "arpu": 79.6},
                },
                "optimistic": {
                    "threshold": 0.50,
                    "costs": {"c": 134.78, "r": 0.40, "ltv": 678.24, "arpu": 94.2},
                },
            }
        }
    )


# ---------------------------------------------------------------------------
# costs_config_hash
# ---------------------------------------------------------------------------


def test_costs_config_hash_changes_iff_costs_yaml_changes(tmp_path: Path) -> None:
    """Same resolved content -> same hash; a resolved-value difference -> a
    different hash.

    Pins configs/policy/threshold.yaml's provenance without a model stamp:
    same hash means the model changed, a different hash means the cost
    assumptions did.
    """
    path_a = tmp_path / "costs_a.yaml"
    path_a.write_text("gross_margin: 0.60\n", encoding="utf-8")
    path_b = tmp_path / "costs_b.yaml"
    path_b.write_text("gross_margin: 0.60\n", encoding="utf-8")
    path_c = tmp_path / "costs_c.yaml"
    path_c.write_text("gross_margin: 0.65\n", encoding="utf-8")

    assert costs_config_hash(path_a) == costs_config_hash(path_b)
    assert costs_config_hash(path_a) != costs_config_hash(path_c)


def test_costs_config_hash_ignores_cosmetic_formatting(tmp_path: Path) -> None:
    """A comment, re-indentation, or a CRLF/LF line-ending swap must not change
    the hash — only a resolved-value change may, or checking the same
    costs.yaml out on a different OS becomes a false provenance mismatch
    against the hash already pinned in configs/policy/threshold.yaml.
    """
    path_plain = tmp_path / "costs_plain.yaml"
    path_plain.write_text("gross_margin: 0.60\nhorizon_months: 12\n", encoding="utf-8")
    path_commented = tmp_path / "costs_commented.yaml"
    path_commented.write_text(
        "# gross margin assumption\n"
        "gross_margin: 0.60\n"
        "\n"
        "horizon_months: 12  # months\n",
        encoding="utf-8",
    )
    path_crlf = tmp_path / "costs_crlf.yaml"
    path_crlf.write_bytes(b"gross_margin: 0.60\r\nhorizon_months: 12\r\n")

    plain_hash = costs_config_hash(path_plain)
    assert costs_config_hash(path_commented) == plain_hash
    assert costs_config_hash(path_crlf) == plain_hash


# ---------------------------------------------------------------------------
# expected_value_at_threshold
# ---------------------------------------------------------------------------


def test_expected_value_at_threshold_agrees_with_curve(
    base_scenario: CostScenario,
) -> None:
    """expected_value_at_threshold must agree with expected_value_curve at
    each of the curve's own threshold points — the invariant that keeps the
    scalar (used for dev_ev_at_t_star and by Phase 7's economics.py) from
    drifting apart from the curve it is a single point on.
    """
    proba = np.array([0.9, 0.7, 0.4, 0.2])
    y = np.array([1, 0, 1, 0])

    thresholds, ev = expected_value_curve(proba, y, base_scenario)

    for t, expected_ev in zip(thresholds, ev, strict=True):
        assert expected_value_at_threshold(
            proba, y, base_scenario, float(t)
        ) == pytest.approx(expected_ev)


def test_expected_value_at_threshold_hand_computed(
    base_scenario: CostScenario,
) -> None:
    """Same 4-row example as threshold's test_expected_value_curve_hand_computed,
    evaluated directly at t=0.4 rather than read off the curve: contacts rows
    with proba >= 0.4 (0.9, 0.7, 0.4) -> y = [1, 0, 1], 2 TP + 1 FP ->
    (2*150 - 1*100) / 4 = 50.0.
    """
    proba = np.array([0.9, 0.7, 0.4, 0.2])
    y = np.array([1, 0, 1, 0])

    assert expected_value_at_threshold(proba, y, base_scenario, 0.4) == pytest.approx(
        50.0
    )


# ---------------------------------------------------------------------------
# resolve_policy_scenarios / resolve_policy_thresholds_by_scenario
# ---------------------------------------------------------------------------


def test_resolve_policy_scenarios_one_per_scenario(policy_fixture: OmegaConf) -> None:
    """One CostScenario per named scenario, with fields read from `costs`."""
    scenarios = resolve_policy_scenarios(policy_fixture)
    assert set(scenarios) == {"conservative", "base", "optimistic"}
    base = scenarios["base"]
    assert base.name == "base"
    assert base.cost == pytest.approx(67.76)
    assert base.retention_rate == pytest.approx(0.30)
    assert base.ltv == pytest.approx(573.12)
    assert base.arpu == pytest.approx(79.6)


def test_resolve_policy_thresholds_by_scenario(policy_fixture: OmegaConf) -> None:
    """One threshold per named scenario, matching the policy's own `threshold` field."""
    thresholds = resolve_policy_thresholds_by_scenario(policy_fixture)
    assert thresholds == {"conservative": 0.27, "base": 0.39, "optimistic": 0.50}


# ---------------------------------------------------------------------------
# load_model_promotion_bars
# ---------------------------------------------------------------------------


def test_load_model_promotion_bars_reads_real_config() -> None:
    """Reads the actual configs/model_promotion.yaml shipped with the project."""
    from telco_churn.utils.paths import compose_config

    cfg = compose_config()
    bars = load_model_promotion_bars(cfg)
    assert bars.pr_auc_bar == pytest.approx(0.60)
    assert bars.recall_bar == pytest.approx(0.65)
    assert bars.calibration_slope_band[0] == pytest.approx(0.80)
    assert bars.calibration_slope_band[1] == pytest.approx(1.25)
    assert bars.pr_auc_materiality_threshold == pytest.approx(0.005)
    assert bars.brier_non_inferiority_margin == pytest.approx(0.005)
    assert bars.recall_non_inferiority_margin == pytest.approx(0.01)


def test_load_model_promotion_bars_matches_yaml_file() -> None:
    """Cross-check against the real configs/model_promotion.yaml — the
    ANALYSIS.md §0 pre-registered gate policy — rather than hardcoding the
    bar values, so this test doesn't go stale (or silently start asserting
    the wrong thing) the moment the policy is revised.
    """
    cfg = OmegaConf.create(
        {"paths": {"model_promotion_config": "configs/model_promotion.yaml"}}
    )
    raw = OmegaConf.load(get_project_root() / "configs" / "model_promotion.yaml")

    bars = load_model_promotion_bars(cfg)

    assert bars.pr_auc_bar == pytest.approx(float(raw.pr_auc_bar))
    assert bars.recall_bar == pytest.approx(float(raw.recall_bar))
    lo, hi = bars.calibration_slope_band
    assert lo == pytest.approx(float(raw.calibration_slope_band[0]))
    assert hi == pytest.approx(float(raw.calibration_slope_band[1]))
    assert bars.pr_auc_materiality_threshold == pytest.approx(
        float(raw.pr_auc_materiality_threshold)
    )
    assert bars.brier_non_inferiority_margin == pytest.approx(
        float(raw.brier_non_inferiority_margin)
    )
    assert bars.recall_non_inferiority_margin == pytest.approx(
        float(raw.recall_non_inferiority_margin)
    )

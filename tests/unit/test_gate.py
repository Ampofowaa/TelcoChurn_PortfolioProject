"""Unit tests for telco_churn.models.gate — the promotion gate (ANALYSIS.md §0), no MLflow, no filesystem."""

from __future__ import annotations

import pytest

from telco_churn.models.gate import (
    GateBars,
    GateInputs,
    decide_promotion,
    record_review,
)

COLD_START = "cold_start"
COMPARATIVE = "comparative"

# Local test fixture values — deliberately not imported from configs/ or from
# gate.py: the mechanism under test must hold for whatever bars it is handed,
# not merely for today's policy numbers. If ANALYSIS.md §0's bars change,
# these tests should not need to.
PR_AUC_BAR = 0.60
RECALL_BAR = 0.65
PR_AUC_MATERIALITY_THRESHOLD = 0.005
BRIER_NON_INFERIORITY_MARGIN = 0.005
RECALL_NON_INFERIORITY_MARGIN = 0.03

BARS = GateBars(
    pr_auc_bar=PR_AUC_BAR,
    recall_bar=RECALL_BAR,
    calibration_slope_band=(0.80, 1.25),
    pr_auc_materiality_threshold=PR_AUC_MATERIALITY_THRESHOLD,
    brier_non_inferiority_margin=BRIER_NON_INFERIORITY_MARGIN,
    recall_non_inferiority_margin=RECALL_NON_INFERIORITY_MARGIN,
)


def _passing_cold_start_inputs(**overrides: float) -> GateInputs:
    defaults: dict[str, float] = {
        "pr_auc": PR_AUC_BAR + 0.05,
        "recall": RECALL_BAR + 0.05,
        "bss": 0.10,
        "calibration_slope": 1.0,
        "calibration_slope_ci_lower": 0.90,
        "calibration_slope_ci_upper": 1.10,
    }
    defaults.update(overrides)
    return GateInputs(**defaults)


def _passing_comparative_inputs(**overrides: float) -> GateInputs:
    defaults: dict[str, float] = {
        "pr_auc": 0.65,
        "recall": RECALL_BAR + 0.05,
        "bss": 0.10,
        "calibration_slope": 1.0,
        "calibration_slope_ci_lower": 0.90,
        "calibration_slope_ci_upper": 1.10,
        "pr_auc_delta_obs": PR_AUC_MATERIALITY_THRESHOLD + 0.01,
        "pr_auc_delta_ci_lower": 0.001,
        "pr_auc_delta_ci_upper": 0.02,
        "brier_delta_obs": -0.01,
        "brier_delta_ci_lower": -0.02,
        "brier_delta_ci_upper": 0.0,
        "recall_delta_obs": 0.01,
        "recall_delta_ci_lower": -0.005,
        "recall_delta_ci_upper": 0.03,
    }
    defaults.update(overrides)
    return GateInputs(**defaults)


# ---------------------------------------------------------------------------
# GateBars
# ---------------------------------------------------------------------------


def test_gate_bars_has_no_defaults() -> None:
    """Every field is required — constructing without a value raises, so a
    caller can never silently fall back to an unstated policy number."""
    with pytest.raises(TypeError):
        GateBars(pr_auc_bar=0.60)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Cold-start regime
# ---------------------------------------------------------------------------


def test_cold_start_all_criteria_passing_gives_pass() -> None:
    """A model clearing every bar passes the cold-start gate."""
    decision = decide_promotion(_passing_cold_start_inputs(), COLD_START, BARS)
    assert decision["regime"] == "cold_start"
    assert decision["gate"] == "pass"
    assert decision["review"] == "pending"


def test_cold_start_failing_pr_auc_fails_selection_and_gate() -> None:
    """PR-AUC below the bar fails selection, and the gate overall."""
    decision = decide_promotion(
        _passing_cold_start_inputs(pr_auc=0.50), COLD_START, BARS
    )
    assert decision["criteria"]["pr_auc"]["passed"] is False
    assert decision["gate"] == "fail"


def test_cold_start_recall_guardrail_vetoes_a_pr_auc_passing_model() -> None:
    """The guardrail's veto power under test: PR-AUC clears its bar but recall
    does not — the model is still rejected overall."""
    decision = decide_promotion(
        _passing_cold_start_inputs(recall=0.40), COLD_START, BARS
    )
    assert decision["criteria"]["pr_auc"]["passed"] is True
    assert decision["criteria"]["recall"]["passed"] is False
    assert decision["gate"] == "fail"


def test_cold_start_bss_guardrail_vetoes() -> None:
    """A non-positive Brier skill score vetoes even with PR-AUC passing."""
    decision = decide_promotion(_passing_cold_start_inputs(bss=-0.01), COLD_START, BARS)
    assert decision["criteria"]["brier_skill_score"]["passed"] is False
    assert decision["gate"] == "fail"


def test_cold_start_bss_exactly_zero_fails() -> None:
    """The bar is strict '> 0' — exactly zero does not pass."""
    decision = decide_promotion(_passing_cold_start_inputs(bss=0.0), COLD_START, BARS)
    assert decision["criteria"]["brier_skill_score"]["passed"] is False


@pytest.mark.parametrize(
    ("ci_lower", "ci_upper"),
    [
        (1.30, 1.50),  # entirely above the band
        (0.40, 0.70),  # entirely below the band
    ],
)
def test_cold_start_calibration_slope_guardrail_vetoes_on_material_evidence(
    ci_lower: float, ci_upper: float
) -> None:
    """A slope CI lying entirely outside the band vetoes, in either direction."""
    decision = decide_promotion(
        _passing_cold_start_inputs(
            calibration_slope_ci_lower=ci_lower, calibration_slope_ci_upper=ci_upper
        ),
        COLD_START,
        BARS,
    )
    assert decision["criteria"]["calibration_slope"]["passed"] is False
    assert decision["gate"] == "fail"


def test_cold_start_calibration_slope_ci_merely_touching_band_does_not_veto() -> None:
    """A CI wide enough to touch the band edge, but not entirely outside it, is
    not evidence of harm — burden of proof stays with the accuser."""
    decision = decide_promotion(
        _passing_cold_start_inputs(
            calibration_slope_ci_lower=0.70, calibration_slope_ci_upper=1.30
        ),
        COLD_START,
        BARS,
    )
    assert decision["criteria"]["calibration_slope"]["passed"] is True


def test_cold_start_no_guardrail_alone_can_flip_a_failed_selection_to_pass() -> None:
    """Guardrails can only ever veto: failing PR-AUC fails the gate regardless
    of how well every guardrail does."""
    decision = decide_promotion(
        _passing_cold_start_inputs(
            pr_auc=0.10,
            recall=0.99,
            bss=0.99,
            calibration_slope_ci_lower=0.99,
            calibration_slope_ci_upper=1.01,
        ),
        COLD_START,
        BARS,
    )
    assert decision["gate"] == "fail"


@pytest.mark.parametrize(
    ("field", "good_value", "bad_value"),
    [
        ("recall", RECALL_BAR + 0.10, 0.0),
        ("bss", 0.5, -0.5),
    ],
)
def test_cold_start_no_guardrail_value_alone_flips_rejection_into_promotion(
    field: str, good_value: float, bad_value: float
) -> None:
    """Property check: across the guardrail input space, no value of recall or
    Brier alone turns a failing gate into a passing one when selection failed."""
    decision_selection_failed = decide_promotion(
        _passing_cold_start_inputs(pr_auc=0.10, **{field: good_value}), COLD_START, BARS
    )
    assert decision_selection_failed["gate"] == "fail"
    decision_still_failed = decide_promotion(
        _passing_cold_start_inputs(pr_auc=0.10, **{field: bad_value}), COLD_START, BARS
    )
    assert decision_still_failed["gate"] == "fail"


# ---------------------------------------------------------------------------
# Comparative regime
# ---------------------------------------------------------------------------


def test_comparative_all_criteria_passing_gives_pass() -> None:
    """A challenger with a materially, confidently better PR-AUC and passing
    guardrails is promoted."""
    decision = decide_promotion(_passing_comparative_inputs(), COMPARATIVE, BARS)
    assert decision["regime"] == "comparative"
    assert decision["gate"] == "pass"


def test_comparative_ci_straddling_zero_is_not_promoted_despite_delta_above_materiality() -> (
    None
):
    """Δ_obs clears the materiality threshold, but the CI straddles zero — not
    statistically distinguishable from noise, so no promotion."""
    decision = decide_promotion(
        _passing_comparative_inputs(
            pr_auc_delta_obs=PR_AUC_MATERIALITY_THRESHOLD + 0.01,
            pr_auc_delta_ci_lower=-0.001,
            pr_auc_delta_ci_upper=0.02,
        ),
        COMPARATIVE,
        BARS,
    )
    assert decision["criteria"]["pr_auc"]["passed"] is False
    assert decision["gate"] == "fail"


def test_comparative_tight_ci_below_materiality_is_not_promoted() -> None:
    """A tight, statistically clear CI whose Δ_obs is below the materiality
    threshold is a real-but-trivial gain — not promoted."""
    decision = decide_promotion(
        _passing_comparative_inputs(
            pr_auc_delta_obs=PR_AUC_MATERIALITY_THRESHOLD - 0.001,
            pr_auc_delta_ci_lower=0.0001,
            pr_auc_delta_ci_upper=0.0009,
        ),
        COMPARATIVE,
        BARS,
    )
    assert decision["criteria"]["pr_auc"]["passed"] is False
    assert decision["gate"] == "fail"


def test_comparative_pr_auc_below_absolute_bar_vetoes_despite_material_delta() -> None:
    """A candidate whose own PR-AUC sits below the cold-start bar is not
    promoted even with a materially, confidently better Δ vs the incumbent —
    the absolute floor is a second necessary condition, not a fallback the Δ
    can substitute for. Guards against bar creep: if `pr_auc_bar` is later
    raised past a weak incumbent's own PR-AUC, comparative candidates must
    still clear the new floor, not merely out-pace the weak incumbent."""
    decision = decide_promotion(
        _passing_comparative_inputs(pr_auc=PR_AUC_BAR - 0.05),
        COMPARATIVE,
        BARS,
    )
    assert decision["criteria"]["pr_auc"]["passed"] is False
    assert decision["gate"] == "fail"


def test_comparative_pr_auc_at_bar_with_material_delta_passes() -> None:
    """The floor is inclusive (`>=`), matching the cold-start rule."""
    decision = decide_promotion(
        _passing_comparative_inputs(pr_auc=PR_AUC_BAR),
        COMPARATIVE,
        BARS,
    )
    assert decision["criteria"]["pr_auc"]["passed"] is True
    assert decision["gate"] == "pass"


def test_comparative_recall_guardrail_vetoes_a_pr_auc_passing_challenger() -> None:
    """The guardrail is absolute in the comparative regime too — it does not
    relax just because selection passed."""
    decision = decide_promotion(
        _passing_comparative_inputs(recall=0.10), COMPARATIVE, BARS
    )
    assert decision["criteria"]["pr_auc"]["passed"] is True
    assert decision["criteria"]["recall"]["passed"] is False
    assert decision["gate"] == "fail"


def test_comparative_recall_guardrail_vetoes_on_material_regression_despite_clearing_floor() -> (
    None
):
    """A candidate can clear the absolute recall floor while still being a
    large regression from the incumbent (recall 0.90 -> 0.66 both clear 0.65)
    — the non-inferiority check catches what the absolute floor alone
    cannot."""
    decision = decide_promotion(
        _passing_comparative_inputs(
            recall=RECALL_BAR + 0.01,
            recall_delta_obs=-0.20,
            recall_delta_ci_lower=-0.25,
            recall_delta_ci_upper=-0.15,
        ),
        COMPARATIVE,
        BARS,
    )
    assert decision["criteria"]["recall"]["passed"] is False
    assert decision["gate"] == "fail"


def test_comparative_recall_guardrail_does_not_veto_for_want_of_evidence() -> None:
    """Non-inferiority, burden of proof on the accuser: a CI overlapping the
    margin (not entirely below it) does not veto, even if the point estimate
    is on the worse side."""
    decision = decide_promotion(
        _passing_comparative_inputs(
            recall=RECALL_BAR + 0.05,
            recall_delta_obs=-0.02,
            recall_delta_ci_lower=-0.05,
            recall_delta_ci_upper=0.01,
        ),
        COMPARATIVE,
        BARS,
    )
    assert decision["criteria"]["recall"]["passed"] is True
    assert decision["gate"] == "pass"


def test_comparative_missing_recall_delta_raises() -> None:
    """Calling the comparative regime without the required recall Δ fields
    raises rather than silently treating None as some default."""
    incomplete = GateInputs(
        pr_auc=0.65,
        recall=0.70,
        bss=0.1,
        calibration_slope=1.0,
        calibration_slope_ci_lower=0.9,
        calibration_slope_ci_upper=1.1,
        pr_auc_delta_obs=0.01,
        pr_auc_delta_ci_lower=0.001,
        pr_auc_delta_ci_upper=0.02,
        brier_delta_obs=-0.01,
        brier_delta_ci_lower=-0.02,
        brier_delta_ci_upper=0.0,
    )
    with pytest.raises(ValueError, match="recall_delta"):
        decide_promotion(incomplete, COMPARATIVE, BARS)


def test_comparative_brier_guardrail_vetoes_when_ci_entirely_above_margin() -> None:
    """A Brier Δ CI lying entirely above the non-inferiority margin vetoes —
    positive evidence of material calibration harm."""
    decision = decide_promotion(
        _passing_comparative_inputs(
            brier_delta_obs=0.01,
            brier_delta_ci_lower=BRIER_NON_INFERIORITY_MARGIN + 0.001,
            brier_delta_ci_upper=0.02,
        ),
        COMPARATIVE,
        BARS,
    )
    assert decision["criteria"]["brier"]["passed"] is False
    assert decision["gate"] == "fail"


def test_comparative_brier_guardrail_does_not_veto_for_want_of_evidence() -> None:
    """Non-inferiority, burden of proof on the accuser: a CI overlapping the
    margin (not entirely above it) does not veto, even if the point estimate
    is on the worse side."""
    decision = decide_promotion(
        _passing_comparative_inputs(
            brier_delta_obs=0.006,
            brier_delta_ci_lower=-0.01,
            brier_delta_ci_upper=0.02,
        ),
        COMPARATIVE,
        BARS,
    )
    assert decision["criteria"]["brier"]["passed"] is True


def test_comparative_bss_guardrail_vetoes_despite_passing_brier_delta() -> None:
    """A non-positive BSS vetoes even when the Brier Δ vs the incumbent shows
    no material regression — the absolute floor is a second necessary
    condition, not subsumed by the non-inferiority check. Guards against
    biocreep: a lineage that is merely non-inferior to its immediate
    predecessor every cycle can still drift below a fixed reference over
    many cycles with nothing else to catch it."""
    decision = decide_promotion(
        _passing_comparative_inputs(bss=-0.01),
        COMPARATIVE,
        BARS,
    )
    assert decision["criteria"]["brier"]["passed"] is False
    assert decision["gate"] == "fail"


def test_comparative_bss_exactly_zero_fails() -> None:
    """The bar is strict '> 0', matching the cold-start rule."""
    decision = decide_promotion(
        _passing_comparative_inputs(bss=0.0),
        COMPARATIVE,
        BARS,
    )
    assert decision["criteria"]["brier"]["passed"] is False


def test_comparative_calibration_slope_guardrail_is_absolute_not_relative() -> None:
    """The champion's own slope drifting outside the band vetoes the champion's
    own promotion cycle logic identically to cold start — it is a property of
    the artifact, not a competition against the incumbent."""
    decision = decide_promotion(
        _passing_comparative_inputs(
            calibration_slope_ci_lower=1.30, calibration_slope_ci_upper=1.50
        ),
        COMPARATIVE,
        BARS,
    )
    assert decision["criteria"]["calibration_slope"]["passed"] is False
    assert decision["gate"] == "fail"


def test_comparative_missing_pr_auc_delta_raises() -> None:
    """Calling the comparative regime without the required Δ fields raises
    rather than silently treating None as some default."""
    incomplete = GateInputs(
        pr_auc=0.65,
        recall=0.70,
        bss=0.1,
        calibration_slope=1.0,
        calibration_slope_ci_lower=0.9,
        calibration_slope_ci_upper=1.1,
    )
    with pytest.raises(ValueError, match="pr_auc_delta"):
        decide_promotion(incomplete, COMPARATIVE, BARS)


def test_comparative_missing_brier_delta_raises() -> None:
    """Same guard for the Brier delta fields."""
    incomplete = GateInputs(
        pr_auc=0.65,
        recall=0.70,
        bss=0.1,
        calibration_slope=1.0,
        calibration_slope_ci_lower=0.9,
        calibration_slope_ci_upper=1.1,
        pr_auc_delta_obs=0.01,
        pr_auc_delta_ci_lower=0.001,
        pr_auc_delta_ci_upper=0.02,
    )
    with pytest.raises(ValueError, match="brier_delta"):
        decide_promotion(incomplete, COMPARATIVE, BARS)


@pytest.mark.parametrize(
    ("field", "good_value", "bad_value"),
    [
        ("recall", RECALL_BAR + 0.10, 0.0),
        ("brier_delta_ci_lower", -0.02, BRIER_NON_INFERIORITY_MARGIN + 0.01),
    ],
)
def test_comparative_no_guardrail_value_alone_flips_a_failed_selection_to_pass(
    field: str, good_value: float, bad_value: float
) -> None:
    """Same property as cold start, in the comparative regime: no guardrail
    value can rescue a selection that failed on Δ."""
    failing_selection_kwargs = {
        "pr_auc_delta_obs": 0.0001,
        "pr_auc_delta_ci_lower": -0.01,
        "pr_auc_delta_ci_upper": 0.01,
    }
    decision_good = decide_promotion(
        _passing_comparative_inputs(**{field: good_value}, **failing_selection_kwargs),
        COMPARATIVE,
        BARS,
    )
    assert decision_good["gate"] == "fail"
    decision_bad = decide_promotion(
        _passing_comparative_inputs(**{field: bad_value}, **failing_selection_kwargs),
        COMPARATIVE,
        BARS,
    )
    assert decision_bad["gate"] == "fail"


# ---------------------------------------------------------------------------
# record_review
# ---------------------------------------------------------------------------


def test_record_review_stamps_all_fields() -> None:
    """The returned dict carries the verdict, notes, approver, timestamp, and
    the direction sanity check's veto outcome."""
    decision = decide_promotion(_passing_cold_start_inputs(), COLD_START, BARS)
    stamped = record_review(
        decision,
        verdict="approved",
        notes="No segment collapse; fairness gaps within tolerance.",
        approver="r.frimpong",
        reviewed_at="2026-07-19T00:00:00Z",
        direction_sanity_check_fired=False,
    )
    assert stamped["review"] == "approved"
    assert stamped["approver"] == "r.frimpong"
    assert stamped["reviewed_at"] == "2026-07-19T00:00:00Z"
    assert stamped["direction_sanity_check_fired"] is False


def test_record_review_does_not_mutate_original_decision() -> None:
    """record_review returns a new dict — the caller's original decision object
    (and, structurally, whatever pending-review copy is on disk before this
    call) is left untouched."""
    decision = decide_promotion(_passing_cold_start_inputs(), COLD_START, BARS)
    original_review = decision["review"]
    record_review(
        decision,
        verdict="rejected",
        notes="",
        approver="a",
        reviewed_at="t",
        direction_sanity_check_fired=False,
    )
    assert decision["review"] == original_review


def test_record_review_preserves_gate_verdict_fields() -> None:
    """The original gate/regime/criteria fields survive the stamp unchanged —
    record_review adds fields, it does not overwrite the gate's own verdict."""
    decision = decide_promotion(_passing_cold_start_inputs(), COLD_START, BARS)
    stamped = record_review(
        decision,
        verdict="approved",
        notes="",
        approver="a",
        reviewed_at="t",
        direction_sanity_check_fired=False,
    )
    assert stamped["gate"] == decision["gate"]
    assert stamped["regime"] == decision["regime"]
    assert stamped["criteria"] == decision["criteria"]

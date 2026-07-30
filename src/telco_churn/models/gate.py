"""The promotion gate — ANALYSIS.md §0's success criterion, implemented once.

Pure: no I/O, no MLflow, no file reads. `decide_promotion` takes numbers and
returns a verdict; `evaluate.py` calls it the moment the sealed-test metrics
exist and persists the result to reports/promotion_decision.json.
`models/register.py` **reads** that persisted verdict rather than calling
this module again — two independent evaluations of one rule is exactly how
a rule becomes two rules.

The five pre-registered policy numbers (`GateBars`) live in
configs/model_promotion.yaml, not in this module — that file is the single
source of truth, loaded directly (bypassing Hydra's CLI-override composition,
the same pattern configs/costs.yaml already uses) so the bars cannot be
silently changed by a command-line override the way an ordinary Hydra config
key can. `evaluate.py` loads that file and constructs the `GateBars` this
module applies; gate.py never reads a bar's value from anywhere itself, and
has no fallback copy of the numbers to go stale against the config.

Scope: this module implements the automated four-criterion gate (PR-AUC
selection; recall, Brier, and calibration-slope guardrails) — §0's "Success
criterion" table — plus V3 (SHAP direction sanity), the sole remaining
pre-registered human-review veto. V3 is computed in error_analysis.py via
explain.py, on the sealed test set, and stamped by a human via record_review
below, which only writes that stamp onto an already-persisted decision — it
does not compute V3 itself.

V1 (segment collapse), V2 (fairness disparity), and V2b (per-group
calibration) are computed in threshold.py's dev-OOF screen via diagnostics.py
and reported by evaluate.py alongside the sealed-test metrics — for the model
card and for the reviewer's notes — but they do not gate promotion. Two
reasons, not one: the 1,409-row sealed test set cannot power a subgroup
conclusion at all (some segments carry on the order of ten churners), and
dev-set evidence should not be what gates a decision that is otherwise
test-set-centered. Continuous production monitoring (Phase 10's
performance_check.py, Phase 13's dashboards) is where subgroup/fairness
enforcement actually lives going forward — production volume accumulates the
statistical power a single pre-launch snapshot cannot supply.

Guardrails may only veto a model selection has already admitted; none can
ever promote one. This is what keeps PR-AUC the sole selection signal: if
PR-AUC and Brier disagree, PR-AUC still wins the *selection*, and Brier's only
recourse is to veto the result outright.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

__all__ = [
    "GateBars",
    "GateInputs",
    "decide_promotion",
    "record_review",
    "slope_passes",
]


@dataclass(frozen=True)
class GateBars:
    """The five pre-registered policy numbers ANALYSIS.md §0 fixes as bars/margins.

    No defaults: the canonical values live in configs/model_promotion.yaml,
    not here — a default here would be a second copy of a policy value that
    goes stale the moment the config changes and no caller happens to exercise
    the fallback. Construct this from the loaded config at the call site
    (evaluate.py); tests construct their own to probe the mechanism
    independent of whatever the current policy happens to be.
    """

    pr_auc_bar: float
    recall_bar: float
    calibration_slope_band: tuple[float, float]
    pr_auc_materiality_threshold: float
    brier_non_inferiority_margin: float


@dataclass(frozen=True)
class GateInputs:
    """One candidate artifact's gate-relevant statistics.

    pr_auc/recall/bss/calibration_slope(+CI) are absolute properties of this
    artifact alone and are what the cold-start regime bars against. The
    *_delta_* fields are only meaningful in the comparative regime — they are
    a paired-bootstrap Δ = this candidate − the incumbent, computed by the
    caller (evaluate.py, via utils.stats.paired_bootstrap_metric_ci for PR-AUC
    and paired_bootstrap_ci for Brier) over the same evaluation rows. gate.py
    never sees the underlying probability vectors and cannot compute a Δ-with-
    CI itself — the pairing lives in the resampled rows, which is exactly why
    these are interval-valued fields handed in, not bare scalars decide_promotion
    derives on its own.
    """

    pr_auc: float
    recall: float
    bss: float
    calibration_slope: float
    calibration_slope_ci_lower: float
    calibration_slope_ci_upper: float
    pr_auc_delta_obs: float | None = None
    pr_auc_delta_ci_lower: float | None = None
    pr_auc_delta_ci_upper: float | None = None
    brier_delta_obs: float | None = None
    brier_delta_ci_lower: float | None = None
    brier_delta_ci_upper: float | None = None


def slope_passes(ci_lower: float, ci_upper: float, band: tuple[float, float]) -> bool:
    """Veto iff the CI lies entirely outside `band` — a tolerance region, not a
    bar: only positive evidence of material miscalibration vetoes, never a
    merely wide-but-overlapping estimate. Public: threshold.py's dev-OOF
    pre-seal screen reuses this exact check, rather than reimplementing the
    same band logic a second time."""
    band_lower, band_upper = band
    entirely_outside = ci_upper < band_lower or ci_lower > band_upper
    return not entirely_outside


def _cold_start_decision(candidate: GateInputs, bars: GateBars) -> dict[str, Any]:
    """§0's cold-start regime: absolute bars on the point estimate, no incumbent to pair against."""
    pr_auc_passed = candidate.pr_auc >= bars.pr_auc_bar
    recall_passed = candidate.recall >= bars.recall_bar
    bss_passed = candidate.bss > 0.0
    slope_passed = slope_passes(
        candidate.calibration_slope_ci_lower,
        candidate.calibration_slope_ci_upper,
        bars.calibration_slope_band,
    )

    criteria = {
        "pr_auc": {
            "role": "selection",
            "passed": pr_auc_passed,
            "value": candidate.pr_auc,
            "bar": bars.pr_auc_bar,
        },
        "recall": {
            "role": "guardrail",
            "passed": recall_passed,
            "value": candidate.recall,
            "bar": bars.recall_bar,
        },
        "brier_skill_score": {
            "role": "guardrail",
            "passed": bss_passed,
            "value": candidate.bss,
            "bar": 0.0,
        },
        "calibration_slope": {
            "role": "guardrail",
            "passed": slope_passed,
            "value": candidate.calibration_slope,
            "ci": [
                candidate.calibration_slope_ci_lower,
                candidate.calibration_slope_ci_upper,
            ],
            "band": list(bars.calibration_slope_band),
        },
    }
    gate_passed = pr_auc_passed and recall_passed and bss_passed and slope_passed
    return {"regime": "cold_start", "criteria": criteria, "gate_passed": gate_passed}


def _comparative_decision(candidate: GateInputs, bars: GateBars) -> dict[str, Any]:
    """§0's comparative regime: a paired-bootstrap Δ with a materiality threshold
    for selection; the guardrails stay absolute, exactly as in cold start."""
    if (
        candidate.pr_auc_delta_obs is None
        or candidate.pr_auc_delta_ci_lower is None
        or candidate.pr_auc_delta_ci_upper is None
    ):
        raise ValueError(
            "Comparative regime requires candidate.pr_auc_delta_obs/"
            "pr_auc_delta_ci_lower/pr_auc_delta_ci_upper — compute them via "
            "utils.stats.paired_bootstrap_metric_ci over the shared evaluation rows."
        )
    if (
        candidate.brier_delta_obs is None
        or candidate.brier_delta_ci_lower is None
        or candidate.brier_delta_ci_upper is None
    ):
        raise ValueError(
            "Comparative regime requires candidate.brier_delta_obs/"
            "brier_delta_ci_lower/brier_delta_ci_upper — compute them via "
            "utils.stats.paired_bootstrap_ci over the shared evaluation rows."
        )

    pr_auc_passed = (
        candidate.pr_auc_delta_ci_lower > 0.0
        and candidate.pr_auc_delta_obs >= bars.pr_auc_materiality_threshold
    )
    recall_passed = candidate.recall >= bars.recall_bar
    slope_passed = slope_passes(
        candidate.calibration_slope_ci_lower,
        candidate.calibration_slope_ci_upper,
        bars.calibration_slope_band,
    )
    # Non-inferiority, burden of proof on the accuser: veto only on positive
    # evidence the challenger's Brier is materially worse (CI entirely above
    # the margin), never merely for failing to prove it is not.
    brier_passed = not (
        candidate.brier_delta_ci_lower > bars.brier_non_inferiority_margin
    )

    criteria = {
        "pr_auc": {
            "role": "selection",
            "passed": pr_auc_passed,
            "delta_obs": candidate.pr_auc_delta_obs,
            "delta_ci": [
                candidate.pr_auc_delta_ci_lower,
                candidate.pr_auc_delta_ci_upper,
            ],
            "materiality_threshold": bars.pr_auc_materiality_threshold,
        },
        "recall": {
            "role": "guardrail",
            "passed": recall_passed,
            "value": candidate.recall,
            "bar": bars.recall_bar,
        },
        "brier": {
            "role": "guardrail",
            "passed": brier_passed,
            "delta_obs": candidate.brier_delta_obs,
            "delta_ci": [
                candidate.brier_delta_ci_lower,
                candidate.brier_delta_ci_upper,
            ],
            "non_inferiority_margin": bars.brier_non_inferiority_margin,
        },
        "calibration_slope": {
            "role": "guardrail",
            "passed": slope_passed,
            "value": candidate.calibration_slope,
            "ci": [
                candidate.calibration_slope_ci_lower,
                candidate.calibration_slope_ci_upper,
            ],
            "band": list(bars.calibration_slope_band),
        },
    }
    gate_passed = pr_auc_passed and recall_passed and brier_passed and slope_passed
    return {"regime": "comparative", "criteria": criteria, "gate_passed": gate_passed}


def decide_promotion(
    candidate: GateInputs, incumbent: GateInputs | None, bars: GateBars
) -> dict[str, Any]:
    """§0's promotion gate: PR-AUC selection plus three veto-only guardrails.

    `incumbent is None` selects the cold-start regime (absolute bars — there is
    nothing to compare against yet); otherwise the comparative regime (a
    paired-bootstrap Δ against a materiality threshold). `incumbent`'s own
    metrics are not read here, and it is not echoed into the result — the
    comparative Δ fields on `candidate` are already computed relative to it by
    the caller, so `incumbent` is accepted only to make the regime switch
    explicit at the call site rather than an implicit `None`-field check on
    `candidate`. `bars` is the caller's resolved configs/model_promotion.yaml — this
    function applies it, it does not know or care where the numbers came from.

    Returns a dict with `regime`, `gate` ("pass"/"fail"), `review` ("pending"
    — a human stamps this via record_review), and a per-criterion breakdown
    (`criteria`) recording each one's role, pass/fail, and the values it was
    judged on. Guardrails can only ever veto: `gate` is "pass" iff selection
    passed *and* every guardrail passed.
    """
    result = (
        _cold_start_decision(candidate, bars)
        if incumbent is None
        else _comparative_decision(candidate, bars)
    )
    return {
        "regime": result["regime"],
        "gate": "pass" if result["gate_passed"] else "fail",
        "review": "pending",
        "criteria": result["criteria"],
    }


def record_review(
    decision: dict[str, Any],
    verdict: Literal["approved", "rejected"],
    notes: str,
    approver: str,
    reviewed_at: str,
    direction_sanity_check_fired: bool,
) -> dict[str, Any]:
    """Stamp the human review verdict onto a persisted gate decision.

    Pure: `decision` is the dict evaluate.py already wrote to
    reports/promotion_decision.json (with `review: "pending"`) — this function
    reads/writes nothing itself and returns an updated copy. The notebook's
    closing cell owns loading and re-saving the file; `register.py` refuses
    to run without `review: "approved"` here.

    `direction_sanity_check_fired` is the *outcome* of the direction sanity
    check (computed in error_analysis.py from SHAP dependence points) — True
    means the criterion fired (a sign flip was found), not that it passed.
    This function only records it; it does not decide `verdict` from it —
    that judgment call is the reviewer's, not this function's. Segment
    collapse, per-group calibration, and fairness disparity are reported
    diagnostics (see the module docstring), not veto criteria — they never
    gated `verdict`, so they are not stamped here alongside the one outcome
    that does.
    """
    return {
        **decision,
        "review": verdict,
        "review_notes": notes,
        "approver": approver,
        "reviewed_at": reviewed_at,
        "direction_sanity_check_fired": direction_sanity_check_fired,
    }

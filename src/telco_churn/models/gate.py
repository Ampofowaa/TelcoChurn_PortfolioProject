"""The promotion gate — the champion/challenger promotion success criterion, implemented once.

Pure: no I/O, no MLflow, no file reads. `decide_promotion` takes numbers and
returns a verdict; `evaluate.py` calls it the moment the sealed-test metrics
exist and persists the result to reports/promotion_decision.json.
`models/register.py` **reads** that persisted verdict rather than calling
this module again — two independent evaluations of one rule is how a rule
becomes two rules.

The six pre-registered policy numbers (`GateBars`) live in
configs/model_promotion.yaml, not here — loaded directly (bypassing Hydra's
CLI-override composition, like configs/costs.yaml) so they can't be silently
changed by a command-line override. `evaluate.py` loads that file and
constructs the `GateBars` this module applies; gate.py has no fallback copy
of the numbers to go stale against the config.

Scope: this module implements the automated four-criterion gate (PR-AUC
selection; recall, Brier, and calibration-slope guardrails) plus V3 (SHAP
direction sanity), the sole remaining pre-registered human-review veto. V3
is computed in error_analysis.py via explain.py; the human verdict is
recorded separately, via `record_review` below, as its own append-only
promotion_review.json document — never fused onto decide_promotion's own
output, which has exactly one author (evaluate.py) and must stay that way
for a DVC-tracked out to be valid.

V1 (segment collapse), V2 (fairness disparity), and V2b (per-group
calibration) are computed in threshold.py's dev-OOF screen via diagnostics.py
and reported by evaluate.py for the model card and reviewer's notes, but they
do not gate promotion: the 1,409-row sealed test set can't power a subgroup
conclusion (some segments carry on the order of ten churners), and dev-set
evidence shouldn't gate a decision that is otherwise test-set-centered.
Continuous production monitoring (scheduled performance checks and ongoing
dashboards) is where subgroup/fairness enforcement lives going forward, once
production volume supplies the statistical power a single pre-launch
snapshot can't.

Guardrails may only veto a model selection has already admitted; none can
ever promote one. This keeps PR-AUC the sole selection signal — if PR-AUC and
Brier disagree, PR-AUC still wins the *selection*, and Brier's only recourse
is to veto the result outright.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

__all__ = [
    "GateBars",
    "GateInputs",
    "check_threshold_provenance",
    "check_threshold_screen_passed",
    "decide_promotion",
    "record_review",
    "slope_passes",
]


@dataclass(frozen=True)
class GateBars:
    """The six pre-registered policy numbers this gate checks candidates against (bars/margins).

    No defaults: the canonical values live in configs/model_promotion.yaml —
    a default here would be a second copy that goes stale the moment the
    config changes. Construct this from the loaded config at the call site
    (evaluate.py); tests construct their own to probe the mechanism
    independent of the current policy.
    """

    pr_auc_bar: float
    recall_bar: float
    calibration_slope_band: tuple[float, float]
    pr_auc_materiality_threshold: float
    brier_non_inferiority_margin: float
    recall_non_inferiority_margin: float


@dataclass(frozen=True)
class GateInputs:
    """One candidate artifact's gate-relevant statistics.

    pr_auc/recall/bss/calibration_slope(+CI) are absolute properties of this
    artifact and are what the cold-start regime bars against. The `*_delta_*`
    fields are only meaningful in the comparative regime — a paired-bootstrap
    Δ = this candidate − the incumbent, computed by the caller (evaluate.py,
    via utils.stats.paired_bootstrap_metric_ci / paired_bootstrap_ci) over the
    same evaluation rows. gate.py never sees the underlying probability
    vectors, so these are interval-valued fields handed in, not scalars it
    derives itself.
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
    recall_delta_obs: float | None = None
    recall_delta_ci_lower: float | None = None
    recall_delta_ci_upper: float | None = None


def slope_passes(ci_lower: float, ci_upper: float, band: tuple[float, float]) -> bool:
    """Veto iff the CI lies entirely outside `band`.

    `band` is a tolerance region, not a bar: only positive evidence of
    material miscalibration vetoes, never a merely wide-but-overlapping
    estimate. Public so threshold.py's dev-OOF pre-seal screen can reuse this
    exact check rather than reimplementing the band logic.
    """
    band_lower, band_upper = band
    entirely_outside = ci_upper < band_lower or ci_lower > band_upper
    return not entirely_outside


def _cold_start_decision(candidate: GateInputs, bars: GateBars) -> dict[str, Any]:
    """The cold-start regime: absolute bars on the point estimate, no incumbent to pair against."""
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
    """The comparative regime: a paired-bootstrap Δ with a materiality threshold for selection.

    PR-AUC selection requires *both* the absolute cold-start floor
    (`candidate.pr_auc >= bars.pr_auc_bar`) and the paired-bootstrap Δ vs the
    incumbent — not the Δ alone. Without the floor, a champion lineage's
    absolute PR-AUC is only ever checked against its immediate predecessor,
    never against the currently pre-registered bar: if `bars.pr_auc_bar` is
    later raised in configs/model_promotion.yaml, an incumbent promoted under
    the old, lower bar would let every subsequent comparative candidate skip
    the new floor forever, so long as each one keeps clearing its
    predecessor by the materiality threshold. The floor closes that
    loophole; calibration slope was already absolute in this regime.

    Recall stays absolute (`candidate.recall >= bars.recall_bar`, unchanged
    by regime — the business floor doesn't lower just because the incumbent
    has slipped) but additionally picks up a non-inferiority veto, the same
    shape as Brier's: a candidate clearing the absolute floor can still be a
    large regression from what the incumbent currently delivers (recall 0.90
    -> 0.66 both clear a 0.65 floor), and nothing upstream of this check ever
    compares the two. Vetoes only on positive evidence of material harm (the
    delta CI lies entirely below `-bars.recall_non_inferiority_margin`),
    never merely for failing to prove non-inferiority — same burden of proof
    as Brier's guardrail.

    Brier requires *both* the absolute cold-start floor (`candidate.bss >
    0.0`, vs. the DummyClassifier(prior) baseline) and the non-inferiority Δ
    vs the incumbent — not the Δ alone. A non-inferiority-only check
    tolerates a small regression *every* cycle by design (so trivial Brier
    noise never blocks a real PR-AUC gain), which means a champion lineage
    can drift downward indefinitely, one tolerated dip at a time, without
    ever being checked against a fixed reference — the same failure mode
    pharmacovigilance calls "biocreep": a chain of non-inferiority
    comparisons against only the most recent predecessor, never re-anchored
    to a fixed baseline, can compound into total drift even though every
    individual step passed. The absolute floor is what re-anchors it.
    """
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
    if (
        candidate.recall_delta_obs is None
        or candidate.recall_delta_ci_lower is None
        or candidate.recall_delta_ci_upper is None
    ):
        raise ValueError(
            "Comparative regime requires candidate.recall_delta_obs/"
            "recall_delta_ci_lower/recall_delta_ci_upper — compute them via "
            "utils.stats.paired_bootstrap_metric_ci over the shared evaluation "
            "rows, at the shipped operating threshold."
        )

    pr_auc_passed = (
        candidate.pr_auc >= bars.pr_auc_bar
        and candidate.pr_auc_delta_ci_lower > 0.0
        and candidate.pr_auc_delta_obs >= bars.pr_auc_materiality_threshold
    )
    slope_passed = slope_passes(
        candidate.calibration_slope_ci_lower,
        candidate.calibration_slope_ci_upper,
        bars.calibration_slope_band,
    )
    # Absolute floor, same as cold start: re-anchors the lineage against a
    # fixed reference so repeated tolerated dips (below) can't compound into
    # total drift with nothing ever catching it.
    bss_passed = candidate.bss > 0.0
    # Non-inferiority, burden of proof on the accuser: veto only on positive
    # evidence the challenger's Brier is materially worse (CI entirely above
    # the margin), never merely for failing to prove it is not.
    brier_delta_passed = not (
        candidate.brier_delta_ci_lower > bars.brier_non_inferiority_margin
    )
    brier_passed = bss_passed and brier_delta_passed
    # Same non-inferiority shape as Brier, mirrored for direction: recall is
    # better when higher, so material harm is a delta CI lying entirely
    # *below* the negative margin, not above it.
    recall_regressed = (
        candidate.recall_delta_ci_upper < -bars.recall_non_inferiority_margin
    )
    recall_passed = candidate.recall >= bars.recall_bar and not recall_regressed

    criteria = {
        "pr_auc": {
            "role": "selection",
            "passed": pr_auc_passed,
            "value": candidate.pr_auc,
            "bar": bars.pr_auc_bar,
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
            "delta_obs": candidate.recall_delta_obs,
            "delta_ci": [
                candidate.recall_delta_ci_lower,
                candidate.recall_delta_ci_upper,
            ],
            "non_inferiority_margin": bars.recall_non_inferiority_margin,
        },
        "brier": {
            "role": "guardrail",
            "passed": brier_passed,
            "bss_value": candidate.bss,
            "bss_bar": 0.0,
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
    candidate: GateInputs,
    regime: Literal["cold_start", "comparative"],
    bars: GateBars,
) -> dict[str, Any]:
    """The promotion gate: PR-AUC selection plus three veto-only guardrails.

    `regime` selects cold-start (absolute bars) vs. comparative (a
    paired-bootstrap Δ against a materiality threshold) explicitly — the
    caller (evaluate.py) already knows which regime applies from whether an
    incumbent champion exists, so this function takes that decision as a
    value rather than inferring it from an incumbent object it would
    otherwise have to accept and then never read: the comparative Δ fields on
    `candidate` are already computed relative to the incumbent by the caller.
    `bars` is the caller's resolved configs/model_promotion.yaml; this
    function applies it without knowing or caring where the numbers came
    from.

    Returns a dict with `regime`, `gate` ("pass"/"fail"), and a per-criterion
    breakdown (`criteria`) recording each one's role, pass/fail, and judged
    values. `gate` is "pass" iff selection passed *and* every guardrail
    passed. Carries no `review` field — the human verdict is a separate
    document with a separate author, recorded via `record_review` below, not
    a field this pure function could ever fill in.
    """
    result = (
        _cold_start_decision(candidate, bars)
        if regime == "cold_start"
        else _comparative_decision(candidate, bars)
    )
    return {
        "regime": result["regime"],
        "gate": "pass" if result["gate_passed"] else "fail",
        "criteria": result["criteria"],
    }


def record_review(
    promotion_review: dict[str, Any] | None,
    decision: dict[str, Any],
    verdict: Literal["approved", "rejected"],
    notes: str,
    approver: str,
    reviewed_at: str,
) -> dict[str, Any]:
    """Append one human-review entry to promotion_review.json's entries list.

    Pure: `promotion_review` is the dict already logged on the eval run
    (`runs:/<eval_run_id>/promotion_review.json`), or None if this cycle has
    no review yet — this function reads/writes nothing itself and returns an
    updated copy. `decision` is evaluate.py's already-persisted
    promotion_decision.json, read only for `eval_run_id`/
    `metrics_content_hash` — the two fields that bind this document to a
    specific evaluation without ever needing `model_version`, which does not
    exist yet at review time (register.py mints it afterward). The review CLI
    (models/review.py) is the sole caller — it owns loading the prior
    document (if any) and re-logging the result, and never touches the
    registry itself — register.py reads only `entries[-1]`.

    Append-only, never overwritten in place: a reviewer who reconsiders logs
    a second entry rather than editing the first, so the full review history
    for a cycle survives (e.g. an initial "rejected" later followed by
    "approved" once a concern is addressed) — register.py acts on the latest
    entry only, but every earlier one stays auditable. No
    `direction_sanity_check_fired` field — that was a machine fact
    (error_analysis.py's V3 outcome) duplicated into a human document that
    has no business restating it; a reviewer or auditor traces it through
    error_analysis_run_id instead. Segment collapse, per-group calibration,
    and fairness disparity are reported diagnostics (see module docstring),
    not veto criteria — they are not stamped here either.
    """
    existing_entries = list(promotion_review["entries"]) if promotion_review else []
    entry = {
        "verdict": verdict,
        "notes": notes,
        "approver": approver,
        "reviewed_at": reviewed_at,
    }
    return {
        "eval_run_id": decision["eval_run_id"],
        "metrics_content_hash": decision["metrics_content_hash"],
        "entries": [*existing_entries, entry],
    }


def check_threshold_provenance(
    validation_payload: dict[str, Any], logged_model_id: str
) -> None:
    """Raise ValueError if the threshold's model stamp doesn't match the model being evaluated.

    threshold.py splits the derived threshold into a model-independent policy
    file (configs/policy/threshold.yaml — a pure function of costs.yaml,
    carrying no model stamp) and a model-dependent validation artifact
    (threshold_validation.json). "A re-calibration invalidates a previously-
    derived threshold" (threshold.py's own docstring) is aspirational until
    something checks it — this is that check. Applying a threshold derived
    against a different calibration map would otherwise produce plausible,
    wrong numbers with nothing raised.

    Compares on logged_model_id, not (run_id, model_version): a LoggedModel
    is the actual scored artifact, and model_run_id is kept in
    validation_payload as a locator only (see threshold.py's payload
    assembly) — no longer load-bearing here.
    """
    stamped_model_id = str(validation_payload["logged_model_id"])
    if stamped_model_id != logged_model_id:
        raise ValueError(
            "threshold_validation.json's model stamp "
            f"(logged_model_id={stamped_model_id!r}) does not match the model "
            f"being evaluated (logged_model_id={logged_model_id!r}) — the "
            "threshold was derived against a different calibration map. "
            "Re-run models.threshold before evaluating."
        )


def check_threshold_screen_passed(validation_payload: dict[str, Any]) -> None:
    """Raise RuntimeError if threshold.py's dev-OOF pre-seal screen failed.

    validation_payload["failures"] is the model-dependent half of
    threshold.py's dev-OOF pre-seal screen (calibration_slope +
    v3_direction_sanity; V1/V2/V2b are reported-only and never appear here)
    — this is an independent re-check at every downstream reader
    (evaluate.py, error_analysis.py, register.py), not a replacement for the
    RuntimeError run_threshold_step itself already raises when the screen
    fails. No override flag: a failed screen means the artifact is not
    trustworthy, and nothing downstream may proceed against it regardless.
    """
    failures = validation_payload["failures"]
    if failures:
        clauses = "; ".join(f"{f['criterion']}: {f['detail']}" for f in failures)
        raise RuntimeError(
            "threshold_validation.json's dev-OOF pre-seal screen failed "
            f"({len(failures)} criterion/criteria): {clauses}. Re-run "
            "models.threshold (and models.calibrate first, if needed) "
            "before evaluating or running error analysis."
        )

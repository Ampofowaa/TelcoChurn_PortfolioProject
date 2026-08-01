"""MLflow Model Registry alias flip — acts on the promotion decision evaluate.py persisted.

register.py reads the gate's verdict; it does not recompute it, and it mints
no new version. The decision was already made by gate.py::decide_promotion,
called from evaluate.py at the moment the sealed-test metrics existed, and
persisted onto evaluate.py's own `evaluation` run — the human review is
stamped onto that same artifact (config-gated via cfg.register.require_review;
Phase 10's weekly retrain sets it False since the automated gate is what a
no-human-in-the-loop cycle uses). Two independent evaluations of one rule are
how a rule drifts into two rules.

Every cross-cycle read this module makes is resolved by explicit run_id, via
tags evaluate.py/error_analysis.py set on the model version being registered
(eval_run_id, error_analysis_run_id — the same pattern calibrate.py's
logged_model_id tag already establishes), never a local reports/ path. A fixed local path
reflects whichever run last executed evaluate.py/error_analysis.py on this
machine, not necessarily this model_version's own cycle — exactly the
staleness failure mode CLAUDE.md's drift_reference.json rule and
evaluate.py's own load_dev_oof_diagnostics already avoid for the same reason.
reports/ is still written (metrics.json, promotion_decision.json,
error_analysis.json, drift_reference.json, model_card.json, ...) as a
human-inspection mirror by the steps that produce them — this module just
never trusts it as a source of truth.

Step order is deliberate — cheapest and most decisive checks run first, and
nothing mutates before it has to:
    1. If the candidate's promotion_status tag is already "promoted"/
       "rejected" (not "pending"), this is a re-invocation of an
       already-decided cycle — log it and stop, without resolving
       eval_run_id/error_analysis_run_id or reading anything else, and without
       repeating any check or re-emitting model_promoted/model_rejected
       (Phase 10's orchestrated, retried flow makes this a real scenario,
       not a hypothetical one: a re-emission would silently reset the
       promoted_at-keyed bake-in window on a promotion that already
       happened).
    2. Resolve eval_run_id from the model version's own tag and load
       promotion_decision.json/metrics.json from that run. Assert the
       decision was computed for this model_version, and that a fresh hash
       of metrics.json matches the one the decision stamped.
    3. If require_review is set, assert review == "approved".
    4. If the gate itself failed, tag rejected and stop — before the
       manifest gate, the error_analysis.json gate, or the smoke check,
       none of which can change that outcome.
    5. Manifest gate: the four per-cycle artifacts must be present.
    6. Resolve error_analysis_run_id from the model version's own tag and load
       error_analysis.json from that run; assert it too was computed for
       this model_version (the model card's fairness section is assembled
       from it, never hand-transcribed, so a stale error_analysis.json from
       a different cycle must not be silently attributed to this one).
    7. Phase 1 smoke check, pre-flip, by explicit version URI: an
       environment-parity check first (raises EnvironmentMismatchError and
       leaves promotion_status untouched on a mismatch — a stale checking
       environment is not evidence the model is bad), then schema/output-
       range/golden-parity checks (tag rejected on failure), then a
       non-gating latency/size diagnostic.
    8. Immediately before flipping — not earlier — re-resolve champion and
       assert it still matches the decision's incumbent, closing the race
       window between evaluation and promotion.
    9. Phase 2: flip, reload through the alias, reconfirm golden parity.
       Roll back via rollback_champion() on failure.
   10. Build and log drift_reference.json and model_card.json onto the
       promoted run.
   11. Tag promoted, append a promotion_log entry, log model_promoted with
       promoted_at.

MLflow surface: register.py opens no run of its own — it acts on the
registry (alias flips, version tags) and on the registered run (artifacts,
via MlflowClient calls that target an existing run_id directly, never
mlflow.start_run). It computes no metrics; if it is ever tempted to, that is
a decision leaking in that belongs in gate.py.

Champion history: every promotion and rollback appends one entry to a
promotion_log tag on the *registered model* (see _append_promotion_event /
champion_history) — an ordered, append-only record of what champion was,
in sequence, so "what was champion before this one" is a direct index
rather than a tag-scan-and-max() reconstruction over model versions.
"""

from __future__ import annotations

import json
import re
import time
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import mlflow
import mlflow.artifacts
import mlflow.pyfunc
import mlflow.sklearn
import mlflow.tracking
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from numpy.typing import NDArray
from omegaconf import DictConfig

from telco_churn.models.calibrate import (
    committed_features_from_manifest,
    load_dev_features,
    load_training_manifest,
)
from telco_churn.models.diagnostics import FAIRNESS_AXES, ROBUSTNESS_AXES
from telco_churn.models.drift_reference import build_reference
from telco_churn.models.evaluate import content_hash, resolve_champion_version
from telco_churn.models.threshold import load_dev_oof_predictions
from telco_churn.utils.logging import get_logger
from telco_churn.utils.mlflow import ensure_experiment_metadata
from telco_churn.utils.paths import get_project_root

__all__ = [
    "EnvironmentMismatchError",
    "champion_history",
    "check_environment_parity",
    "promote_to_alias",
    "rollback_champion",
    "run_registration_step",
]

logger = get_logger(__name__)

# The four per-cycle artifacts every registered run must carry — a champion
# that cannot describe itself must not become promotable (CLAUDE.md).
_MANIFEST_ARTIFACTS = (
    "feature_space.txt",
    "feature_columns.txt",
    "preprocessing.pkl",
    "training_manifest.json",
)


class EnvironmentMismatchError(RuntimeError):
    """The checking environment disagrees with the one the model was logged under.

    Distinct from a smoke-check failure: this means the *verification*
    cannot be trusted, not that the *model* is broken. Callers must not tag
    promotion_status on this exception — the version stays exactly where it
    was (pending, if this fires during the initial smoke check), on the same
    "no verdict was reached, so don't fabricate one" logic that makes
    pending the crash-safe mint-time default in the first place.
    """


def check_environment_parity(model_uri: str, packages: list[str]) -> dict[str, str]:
    """Compare installed `packages` against the registered model's own logged pip requirements.

    The golden-parity tolerance (cfg.register.golden_atol) is only
    meaningful if these haven't drifted from what calibrate.py logged the
    model under — a numpy/scikit-learn/lightgbm patch release shifts
    floating-point output in the same 1e-7-1e-9 range as that tolerance,
    with no actual bug present. Raises EnvironmentMismatchError naming every
    disagreeing package; returns the matched versions on success.
    """
    requirements_path = mlflow.pyfunc.get_model_dependencies(model_uri, format="pip")
    with open(requirements_path, encoding="utf-8") as f:
        requirements_text = f.read()

    matched: dict[str, str] = {}
    mismatches: dict[str, tuple[str, str]] = {}
    for package in packages:
        match = re.search(
            rf"^{re.escape(package)}==([^\s;]+)",
            requirements_text,
            re.MULTILINE | re.IGNORECASE,
        )
        if match is None:
            continue
        pinned = match.group(1)
        installed = importlib_metadata.version(package)
        if pinned == installed:
            matched[package] = installed
        else:
            mismatches[package] = (pinned, installed)

    if mismatches:
        detail = ", ".join(
            f"{pkg} (logged={pinned}, installed={installed})"
            for pkg, (pinned, installed) in mismatches.items()
        )
        raise EnvironmentMismatchError(
            "Environment mismatch against the registered model's logged "
            f"dependencies: {detail}. The golden-parity tolerance is only "
            "meaningful under a matching environment — fix the environment "
            "and re-run rather than trusting this check."
        )
    return matched


def promote_to_alias(
    registered_model_name: str, version: str, alias: str = "champion"
) -> None:
    """Point `alias` at `version`.

    A parameterized standalone step, not an inline hard-coded
    set_registered_model_alias(name, "champion", ...) call — Phase 10's
    shadow/canary staging reuses this exact seam to point a `shadow` alias
    at a staged candidate before the `champion` flip, without touching this
    module.
    """
    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(registered_model_name, alias, version)


_CRITERION_LABELS = {
    "pr_auc": "PR-AUC",
    "recall": "recall",
    "brier_skill_score": "BSS",
    "brier": "Brier",
    "calibration_slope": "calib slope",
}


def _describe_criterion(key: str, criterion: dict[str, Any]) -> str:
    """Render one gate.py criterion as a short human-readable clause.

    Field shape differs by regime (gate.py's cold_start vs. comparative
    criteria dicts) and by whether the criterion carries a point value/bar
    or a paired-bootstrap delta/CI — branches on field presence rather than
    regime, since calibration_slope's shape is identical in both.
    """
    label = _CRITERION_LABELS.get(key, key)
    status = "ok" if criterion["passed"] else "FAIL"
    if "delta_obs" in criterion:
        lo, hi = criterion["delta_ci"]
        threshold = criterion.get(
            "materiality_threshold", criterion.get("non_inferiority_margin")
        )
        return (
            f"{label} Δ{criterion['delta_obs']:+.3f} [{lo:+.3f}, {hi:+.3f}] "
            f"vs {threshold:.3f} ({status})"
        )
    if "ci" in criterion:
        lo, hi = criterion["ci"]
        band_lo, band_hi = criterion["band"]
        return (
            f"{label} {criterion['value']:.3f} [{lo:.3f}, {hi:.3f}] "
            f"band [{band_lo:.2f}, {band_hi:.2f}] ({status})"
        )
    return f"{label} {criterion['value']:.3f} vs bar {criterion['bar']:.3f} ({status})"


def _rejection_description(decision: dict[str, Any], reason: str) -> str:
    """One-line human-readable summary for a rejected model version's description field.

    `reason` distinguishes the three rejection paths in run_registration_step
    — a gate failure, a post-registration smoke-check failure, or a post-
    alias-flip parity failure — because they mean operationally different
    things (a metrics problem vs. a serving-artifact problem vs. a rollback
    that already happened) and must not collapse into one indistinguishable
    "rejected" string the way the promotion_status tag alone does.
    """
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    if reason == "gate_fail":
        clauses = "; ".join(
            _describe_criterion(key, criterion)
            for key, criterion in decision["criteria"].items()
        )
        return f"Rejected {today} ({decision['regime']} regime) — {clauses}"
    if reason == "smoke_check_failed":
        return (
            f"Rejected {today} — gate passed ({decision['regime']}) but failed "
            "post-registration smoke check (schema/range or golden-prediction "
            "parity)."
        )
    if reason == "post_flip_parity_failed":
        return (
            f"Rejected {today} — gate passed ({decision['regime']}) but failed "
            "golden-prediction parity after the champion alias flip; champion "
            "rolled back to the prior version."
        )
    raise ValueError(f"unknown rejection reason: {reason!r}")


def _promotion_description(
    decision: dict[str, Any], previous_champion_version: str | None
) -> str:
    """One-line human-readable summary for a promoted model version's description field."""
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    clauses = "; ".join(
        _describe_criterion(key, criterion)
        for key, criterion in decision["criteria"].items()
    )
    incumbent_note = (
        f"Replaced champion v{previous_champion_version}."
        if previous_champion_version is not None
        else "First champion — no incumbent."
    )
    return (
        f"Promoted {today} ({decision['regime']} regime) — {clauses} {incumbent_note}"
    )


def _append_promotion_event(registered_model_name: str, event: dict[str, Any]) -> None:
    """Append one entry to the champion promotion/rollback history.

    Stored as a single JSON-array tag on the *registered model* itself, not
    a per-version tag — this history is not "which version did this?", it's
    "what did the champion alias do, in order?", so it needs one stable
    home that survives across every version and every training cycle
    rather than being scattered per-version. This is what makes
    champion_history() a direct query instead of the tag-scan-and-max()
    reconstruction rollback_champion() has to fall back to when no
    explicit target_version is given.
    """
    client = mlflow.tracking.MlflowClient()
    try:
        existing = client.get_registered_model(registered_model_name).tags.get(
            "promotion_log", "[]"
        )
    except MlflowException:
        existing = "[]"
    history: list[dict[str, Any]] = json.loads(existing)
    history.append(event)
    client.set_registered_model_tag(
        registered_model_name, "promotion_log", json.dumps(history)
    )


def champion_history(registered_model_name: str) -> list[dict[str, Any]]:
    """Return every champion promotion/rollback, oldest first.

    Each entry is {"action": "promoted" | "rolled_back", "version",
    "previous_champion_version" | "rolled_back_from", "at"}. "What was
    champion before this one" is history[-2] — a direct index into an
    append-only log, not a scan over model-version tags.
    """
    client = mlflow.tracking.MlflowClient()
    try:
        raw = client.get_registered_model(registered_model_name).tags.get(
            "promotion_log", "[]"
        )
    except MlflowException:
        raw = "[]"
    return list(json.loads(raw))


def rollback_champion(
    registered_model_name: str, target_version: str | None = None
) -> str:
    """Roll `champion` back to a version tagged promotion_status=promoted.

    Never by version arithmetic (N-1): a rejected or crashed evaluation
    cycle can leave a higher-numbered version in the registry with no
    champion alias and no promoted tag — indistinguishable from a
    legitimate former champion by version number alone. Selecting on the
    tag is the one safe query.

    With target_version, rolls back to exactly that version — after
    confirming it carries promotion_status=promoted, so an explicit
    override still can't be pointed at a version that was never a
    validated champion.

    Without target_version, undoes the most recent promotion: the
    highest-versioned promoted candidate that is not whatever `champion`
    currently points at. The exclusion matters because promotion_status is
    never cleared once set — the current champion is itself tagged
    promoted, so a bare max() over all promoted versions resolves right
    back to the version being rolled back from, a silent no-op, in exactly
    the scenario this function exists for: an earlier champion (still
    tagged promoted) needs to be restored after a later one has already
    served for a while. This is this rule's single implementation, shared
    by both the automated post-flip abort path (below, where the new
    candidate isn't tagged promoted yet, so the exclusion is a no-op there)
    and a manual emergency rollback invocation, so the two can never drift
    into two rules.
    """
    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name='{registered_model_name}'")
    # ModelVersion.version is an int on the wire — normalize to str immediately
    # so dict keys and the target_version/alias comparisons below never
    # silently mismatch on type.
    promoted = {
        str(v.version): v
        for v in versions
        if v.tags.get("promotion_status") == "promoted"
    }
    if not promoted:
        raise RuntimeError(
            f"No version of {registered_model_name!r} is tagged "
            "promotion_status=promoted — nothing to roll back to."
        )

    try:
        current_champion_version: str | None = str(
            client.get_model_version_by_alias(registered_model_name, "champion").version
        )
    except MlflowException:
        current_champion_version = None

    if target_version is not None:
        if target_version not in promoted:
            raise RuntimeError(
                f"Version {target_version!r} of {registered_model_name!r} is "
                "not tagged promotion_status=promoted — refusing to roll "
                "back to a version that was never a validated champion."
            )
        target = promoted[target_version]
    else:
        candidates = [
            v for v in promoted.values() if str(v.version) != current_champion_version
        ]
        if not candidates:
            raise RuntimeError(
                f"No promoted version of {registered_model_name!r} other "
                f"than the current champion (version "
                f"{current_champion_version!r}) exists — nothing earlier to "
                "roll back to."
            )
        target = max(candidates, key=lambda v: int(v.version))

    promote_to_alias(registered_model_name, str(target.version), "champion")
    _append_promotion_event(
        registered_model_name,
        {
            "action": "rolled_back",
            "version": str(target.version),
            "rolled_back_from": current_champion_version,
            "at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
        },
    )
    logger.info(
        "champion_rollback",
        registered_model_name=registered_model_name,
        rolled_back_to_version=target.version,
    )
    return str(target.version)


def _check_manifest_artifacts(run_id: str) -> None:
    """Abort if the registered run is missing any of the four per-cycle artifacts."""
    client = mlflow.tracking.MlflowClient()
    present = {a.path for a in client.list_artifacts(run_id)}
    missing = [name for name in _MANIFEST_ARTIFACTS if name not in present]
    if missing:
        raise RuntimeError(
            f"Run {run_id!r} is missing per-cycle artifact(s) {missing} — a "
            "champion that cannot describe itself must not become promotable."
        )


def _smoke_check_schema_and_range(
    model: Any, feature_columns: list[str], X_sample: pd.DataFrame
) -> None:
    """Assert the model's input schema is satisfied and its output is a valid probability."""
    missing_cols = [c for c in feature_columns if c not in X_sample.columns]
    if missing_cols:
        raise AssertionError(
            f"Registered model's committed feature columns {missing_cols} "
            "are not present in the scored sample."
        )
    proba: NDArray[np.float64] = model.predict_proba(X_sample[feature_columns])[:, 1]
    if proba.ndim != 1:
        raise AssertionError(f"predict_proba output is not 1-D: shape={proba.shape}")
    if not np.all(np.isfinite(proba)):
        raise AssertionError("predict_proba output contains non-finite values.")
    if not np.all((proba >= 0.0) & (proba <= 1.0)):
        raise AssertionError("predict_proba output is not within [0, 1].")


def _check_golden_parity(model: Any, golden: dict[str, Any], atol: float) -> None:
    """Assert the model reproduces calibrate.py's in-memory reference scores on the golden rows.

    golden["rows"] and golden["p_hat"] come from golden_predictions.json,
    captured by a different process (calibrate.py, in memory, before
    pickling) than the one doing this comparison — never regenerated here,
    or the check would be circular.
    """
    golden_rows = pd.DataFrame(golden["rows"])
    reloaded_scores = model.predict_proba(golden_rows)[:, 1]
    if not np.allclose(reloaded_scores, golden["p_hat"], rtol=0, atol=atol):
        raise AssertionError(
            "Golden-parity check failed: the registered model's reloaded "
            "scores differ from calibrate.py's in-memory reference scores "
            "beyond tolerance — the serialized model is not safe to serve."
        )


def _log_smoke_diagnostics(
    model: Any, golden_rows: pd.DataFrame, model_uri: str, run_id: str
) -> None:
    """Log non-gating per-row latency and model-size diagnostics onto the registered run.

    A relative regression tripwire across cycles, never a gate — measured
    offline on dev/CI hardware, not in the serving container, and a frozen
    spec makes a size/latency move almost always measurement noise, not a
    real problem to block on.
    """
    start = time.perf_counter()
    model.predict_proba(golden_rows)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    ms_per_row = elapsed_ms / max(len(golden_rows), 1)

    local_model_dir = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)
    model_size_bytes = sum(
        f.stat().st_size for f in Path(local_model_dir).rglob("*") if f.is_file()
    )

    client = mlflow.tracking.MlflowClient()
    client.log_metric(run_id, "smoke_inference_ms_per_row", ms_per_row)
    client.log_metric(run_id, "model_size_bytes", float(model_size_bytes))


def _build_drift_reference_inputs(
    run_id: str, eval_run_id: str, cfg: DictConfig
) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    """Union dev-OOF and sealed-test predictions — all rows scored out-of-sample.

    Never the champion's own scores on its training data: in-sample scores
    are systematically sharper, and baselining on them would report a
    permanent phantom drift signal against honest production scores.

    Both halves are fetched by explicit run_id, never a local reports/ path:
    the dev-OOF half from calibrate.py's own MLflow artifact (via
    threshold.load_dev_oof_predictions, keyed on the training run_id), and
    the sealed-test half from evaluate.py's `evaluation` run (keyed on
    eval_run_id, resolved from the model version's own eval_run_id tag) —
    a fixed local path would reflect whichever run last executed
    threshold.py/evaluate.py on this machine, not necessarily this
    model_version's own cycle.
    """
    dev_oof = load_dev_oof_predictions(run_id, cfg)
    test_predictions_path = mlflow.artifacts.download_artifacts(
        artifact_uri=f"runs:/{eval_run_id}/test_predictions.parquet"
    )
    test_predictions = pd.read_parquet(test_predictions_path)
    combined = pd.concat(
        [dev_oof[["y_true", "p_hat"]], test_predictions[["y_true", "p_hat"]]],
        ignore_index=True,
    )
    oos_proba = combined["p_hat"].to_numpy(dtype=np.float64)
    y_combined = combined["y_true"].to_numpy(dtype=np.int_)
    return oos_proba, y_combined


def _summarize_for_stakeholders(
    metrics: dict[str, Any], base_scenario: str = "base"
) -> str:
    """One short, plain-language paragraph — no metric jargon — computed
    from the same base-scenario numbers technical_appendix reports in raw
    form. Leads the card because a reader who isn't already fluent in
    PR-AUC/calibration-slope needs this before anything else is useful to
    them; the rest of the artifact is reference detail for whoever wants to
    verify the claim.
    """
    base_row: dict[str, Any] = next(
        (
            row
            for row in metrics.get("classification", [])
            if row.get("scenario") == base_scenario
        ),
        {},
    )
    base_impact = (
        metrics.get("business_impact", {}).get("scenarios", {}).get(base_scenario, {})
    )
    recall = base_row.get("recall")
    precision = base_row.get("precision")
    contact_rate = base_impact.get("contact_rate")
    ev = base_impact.get("ev")
    campaign_cost = base_impact.get("campaign_cost")

    if None in (recall, precision, contact_rate, ev, campaign_cost):
        return (
            "Summary unavailable — a base-scenario figure this summary is "
            "computed from is missing from metrics.json."
        )

    return (
        f"This model identifies {recall:.0%} of customers who go on to "
        f"cancel. Of the customers it recommends contacting "
        f"({contact_rate:.0%} of the customer base), {precision:.0%} "
        "actually do churn — the remainder are contacted as a "
        "precautionary cost, not a false alarm to be eliminated. Under the "
        f"base cost scenario, following its recommendations is projected "
        f"to net ${ev:,.0f} more than contacting no one, at a campaign "
        f"cost of ${campaign_cost:,.0f}. See business_case.expected_impact "
        "for the conservative and aggressive scenarios, and "
        "risks_and_limitations before treating these figures as "
        "guaranteed."
    )


def _summarize_error_patterns(error_analysis: dict[str, Any], git_sha: str) -> str:
    """Plain-language "where the model struggles" insight, computed from
    error_analysis.py's already-logged value-weighted, confidence, and
    cohort-scan output. Narrated rather than dumped as raw cohort rows —
    those still live in technical_appendix.error_concentration for a
    reviewer who wants the full ranked list — and derived fresh each cycle
    from already-logged figures rather than hand-authored, so it can't go
    stale the way a fixed narrative would under a future retrain.

    The two closing sentences are the exception: they state dataset-schema
    facts (no feature here measures loyalty/satisfaction; phone-only,
    no-internet customers are structurally under-signaled) that are true
    regardless of which model version is serving, so hardcoding them carries
    none of the staleness risk the rest of this function is written to
    avoid. The specific behavioral findings tied to *this* champion's
    measured SHAP dependence — the spend x contract-length interaction, the
    width of tenure's score swing, the manual-flag targeting rule — are
    exactly the kind of figure that would go stale, so they are not
    transcribed here; ANALYSIS.md §7 Business takeaways is pointed to
    instead. Unlike the (already fully inline, merely supplementary) §9
    citations elsewhere in this card, §7's targeting recommendations are not
    duplicated anywhere else in the card, so the pointer is load-bearing —
    it is pinned to this run's git_sha (training_manifest.json) rather than
    dropped, since ANALYSIS.md §7 is rewritten every cycle to describe
    whichever model is *currently* champion and a bare "ANALYSIS.md §7"
    citation on an old model_card.json would silently come to describe a
    later model's findings, not this one's.
    """
    fn_rate_top_value_decile = error_analysis.get("value_weighted_errors", {}).get(
        "fn_rate_top_value_decile"
    )
    fnr_cohorts = error_analysis.get("error_concentration", {}).get(
        "dev_oof_top_fnr_cohorts", []
    )
    fpr_cohorts = error_analysis.get("error_concentration", {}).get(
        "dev_oof_top_fpr_cohorts", []
    )
    top_fnr = fnr_cohorts[0] if fnr_cohorts else None
    top_fpr = fpr_cohorts[0] if fpr_cohorts else None
    fn_near_miss_share = error_analysis.get("error_confidence", {}).get(
        "fn_near_miss_share"
    )
    fp_near_miss_share = error_analysis.get("error_confidence", {}).get(
        "fp_near_miss_share"
    )

    if None in (
        fn_rate_top_value_decile,
        top_fnr,
        top_fpr,
        fn_near_miss_share,
        fp_near_miss_share,
    ):
        return (
            "Error-pattern summary unavailable — one or more of the "
            "figures it is computed from is missing from error_analysis.json."
        )
    assert top_fnr is not None
    assert top_fpr is not None

    return (
        "The model's biggest blind spot among its highest-value customers: "
        f"it misses {fn_rate_top_value_decile:.0%} of churners in the most "
        "expensive customer decile. Across all customers, the group it "
        f"misses most often is {top_fnr['feature']}={top_fnr['value']} "
        f"({top_fnr['rate']:.0%} missed), and the group it flags "
        f"unnecessarily most often is {top_fpr['feature']}={top_fpr['value']} "
        f"({top_fpr['rate']:.0%} false-alarm rate). Most of these are not "
        f"close calls: only {fn_near_miss_share:.0%} of missed churners and "
        f"{fp_near_miss_share:.0%} of false alarms were near the cutoff, so "
        "adjusting the threshold would not have fixed most of them — these "
        "are pattern misses, not borderline decisions. See "
        "technical_appendix.error_concentration for the full ranked list. "
        "Two structural gaps behind these numbers: no feature in this "
        "dataset measures customer loyalty or satisfaction directly, and "
        "phone-only customers with no internet service are nearly invisible "
        "to the model since almost every signal it relies on is "
        "internet-service-based. See ANALYSIS.md §7 Business takeaways "
        f"as of commit {git_sha[:8]} for this model version's specific "
        "targeting and modelling-fix recommendations, tied to its measured "
        "SHAP behavior and not duplicated here to avoid staleness under "
        "retrain — that section describes whichever model was champion "
        "when it was last written, so the commit pin is what ties it back "
        "to this model version specifically."
    )


def _summarize_model_drivers(error_analysis: dict[str, Any], top_k: int = 3) -> str:
    """Plain-language "what drives the score" insight, computed from
    error_analysis.py's already-logged SHAP global importance — a genuine
    business insight (how the model reasons), not a caveat, so it lives in
    business_case rather than risks_and_limitations. Derived fresh each
    cycle from already-logged values, never hand-authored.
    """
    ranked = error_analysis.get("shap", {}).get("global_importance", [])
    if not ranked:
        return (
            "Driver summary unavailable — global_importance is missing "
            "from error_analysis.json."
        )
    top_names = [str(row["feature"]) for row in ranked[:top_k]]
    return (
        "The model's score is driven mainly by: " + ", ".join(top_names) + ". "
        "See technical_appendix.factors and the SHAP detail in "
        "error_analysis.json (linked via this run's error_analysis_run_id "
        "tag) for the full ranking and per-feature direction."
    )


def _summarize_promotion_context(
    decision: dict[str, Any], champion_version: Any
) -> str:
    """Plain-language "is this a first deployment or a replacement, and by how
    much did it win" — computed from gate.py's own regime/criteria
    breakdown, never re-derived. Reaching this card means decision["gate"]
    was "pass" (run_registration_step returns early on "fail"), so a
    comparative-regime candidate has already cleared every guardrail; this
    only needs to narrate the selection criterion (PR-AUC) that a business
    reader cares about, not re-litigate the guardrails.
    """
    regime = decision.get("regime")
    criteria = decision.get("criteria", {})
    if regime == "cold_start":
        return (
            "This is the first model promoted to production for this "
            "project — there was no existing champion to compare against, "
            "so the decision rests on this candidate's own numbers clearing "
            "the pre-registered bars (ANALYSIS.md §0), not on beating a "
            "prior version."
        )
    if regime == "comparative":
        pr_auc_delta = criteria.get("pr_auc", {}).get("delta_obs")
        pr_auc_ci = criteria.get("pr_auc", {}).get("delta_ci")
        if pr_auc_delta is None or not pr_auc_ci:
            return (
                "Promotion-context summary unavailable — comparative-regime "
                "PR-AUC delta figures are missing from "
                "promotion_decision.json."
            )
        return (
            f"This model replaces the previous production model (version "
            f"{champion_version}). Reaching this card means it cleared every "
            "pre-registered bar in ANALYSIS.md §0, including beating that "
            f"incumbent's PR-AUC by {pr_auc_delta:+.4f} (95% bootstrap CI "
            f"[{pr_auc_ci[0]:+.4f}, {pr_auc_ci[1]:+.4f}], entirely above "
            "zero and above the pre-registered materiality threshold) "
            "without its Brier score becoming materially worse. See "
            "governance.promotion_context for the full per-criterion "
            "breakdown."
        )
    return (
        f"Promotion-context summary unavailable — unrecognized regime "
        f"{regime!r} in promotion_decision.json."
    )


def _build_business_case_section(
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    economics: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, Any]:
    """Build model_card.json's business_case section."""
    return {
        "what_it_predicts": (
            "Whether a specific customer cancels within the current "
            "billing cycle (about 30 days) — a binary, already-happened "
            "outcome (the IBM Telco dataset's own `Churn` column, "
            "recoded 0/1), not a prediction of future intent to leave "
            "or dissatisfaction. One score per customer per scoring "
            "cycle (monthly by default), not per household or account. "
            "See ANALYSIS.md §0 for the full problem framing."
        ),
        "why_this_exists": (
            "Rank-order and score existing telecom customers by "
            "probability of churn, to prioritize retention outreach "
            "within a fixed campaign budget."
        ),
        "promotion_rationale": _summarize_promotion_context(
            decision, metrics.get("champion_version")
        ),
        "who_uses_it": (
            "Retention/marketing operations teams and the analysts supporting them."
        ),
        "how_to_use_the_output": (
            "The model returns a probability and a recommended contact "
            "decision at the shipped threshold. Use expected_impact's "
            "scenarios to compare cost assumptions before committing "
            "campaign budget — the recommended contact list changes "
            "with the scenario chosen."
        ),
        "expected_impact": {
            "scenarios": metrics.get("business_impact", {}).get("scenarios"),
            "ev_by_k_logged": "ev_by_k" in economics,
        },
        "recommended_next_steps": (
            f"See ANALYSIS.md §7 Business takeaways and §10 "
            f"Recommendations & Next Steps, both as of commit "
            f"{str(manifest.get('git_sha', 'unknown'))[:8]}, for this "
            "model version's specific targeting/modelling-fix "
            "recommendations and the project's deployment/monitoring "
            "commitments — these are human judgment calls, kept in one "
            "place and not duplicated here. Both sections are rewritten "
            "each cycle to describe whichever model is then champion, "
            "so the commit pin is what ties them back to this version "
            "rather than a later one."
        ),
    }


def _build_risks_and_limitations_section(
    manifest: dict[str, Any], error_analysis: dict[str, Any]
) -> dict[str, Any]:
    """Build model_card.json's risks_and_limitations section."""
    return {
        "out_of_scope": (
            "Do not use this model's score to make a decision that "
            "penalizes a specific customer — denying them something, "
            "charging them more, downgrading their service — without a "
            "person reviewing that decision first (what's formally "
            "called an 'adverse action'). And don't assume the numbers "
            "on this card apply to any customer base other than the one "
            "the model was built and tested on: a single historical "
            "snapshot of one telecom's customers (the IBM Telco "
            "dataset), not a different company, market, or time period "
            "(what's formally called the 'training distribution')."
        ),
        "ethical_considerations": {
            "sensitive_attributes_used_as_factors": (
                "Gender, senior-citizen status, partner/dependents "
                "status ARE model inputs — not excluded. This is "
                "deliberate: the model drives a retention offer, not a "
                "credit or employment decision; three of the four carry "
                "real predictive signal; and fairness is enforced by "
                "measuring outcomes across these groups (below), not by "
                "hiding them. See ANALYSIS.md §4a for the full "
                "reasoning."
            ),
            "risk_of_misuse": (
                "A churn score is a prediction about behavior, not a "
                "judgment of a customer's worth or trustworthiness — "
                "treating it as the latter is a misuse of this tool; "
                "see out_of_scope above."
            ),
            "known_gap": (
                "The claim 'this model treats customer subgroups "
                "fairly' rests on evidence from the development data, "
                "not the sealed test set. The held-out test set is "
                "simply too small to trust every subgroup on its own — "
                "its thinnest single slice (two-year contracts, a "
                "different segment reported alongside these axes) has "
                "under ten churners in it, and the same sample-size "
                "problem affects the fairness comparisons below too, "
                "just less severely. Test-set numbers are reported in "
                "technical_appendix for reference but were not the "
                "basis for the promotion decision."
            ),
        },
        "known_limitations": {
            "cost_parameters_are_assumed_not_measured": (
                "The dollar figures in this card assume that a "
                "contacted customer who was about to churn can be "
                "retained a certain fraction of the time (the "
                "'retention rate') — a reasonable industry benchmark, "
                "not something measured from this company's own past "
                "campaigns, because no such data exists yet. Customer "
                "lifetime value is also a scenario-level average, not a "
                "number specific to each customer."
            ),
            "where_the_model_struggles": _summarize_error_patterns(
                error_analysis, str(manifest.get("git_sha", "unknown"))
            ),
            "prevalence_drift_invalidates_calibration_and_threshold": (
                "The model's probabilities and the cutoff it uses to "
                "flag a customer are both tuned to today's churn rate "
                "(about 26.5%). If the real share of customers who "
                "churn changes meaningfully over time, this model's "
                "numbers need to be re-verified, not simply assumed to "
                "still hold."
            ),
            "no_shadow_or_canary_validation": (
                "Before this version started serving, it was checked "
                "offline against historical data — it was not first "
                "tried out on a small slice of live traffic or run "
                "silently alongside the previous model for comparison "
                "(what's formally called 'shadow' or 'canary' "
                "deployment), because there is no live customer feed "
                "for this project to do that with."
            ),
            "see_also": (
                "Two more limitations worth knowing: this model predicts "
                "who will cancel, not who would actually respond to a "
                "retention offer — the expected-value figures above "
                "assume a benchmark response rate rather than a "
                "measured one. And they also assume enough capacity to "
                "contact everyone the model flags; if the retention "
                "team can't act on that many customers, the real "
                "expected value is lower. This card covers the "
                "limitations most relevant to day-to-day use of the "
                "model's recommendations — the project's full technical "
                "documentation (ANALYSIS.md §9) covers additional, more "
                "technical limitations for a reviewer with access to it."
            ),
        },
    }


def _build_governance_section(
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    decision: dict[str, Any],
    calibration_method: str,
    run_id: str,
    eval_run_id: str,
    error_analysis_run_id: str,
    model_version: str,
    registered_model_name: str,
    alias: str,
    promoted_at: str,
) -> dict[str, Any]:
    """Build model_card.json's governance section."""
    return {
        "model_details": {
            "registered_model_name": registered_model_name,
            "version": model_version,
            "alias": alias,
            "promoted_at": promoted_at,
            "model_type": (
                f"{manifest.get('model_family', 'unknown')} + "
                f"{calibration_method} CalibratedClassifierCV"
            ),
            "maintainer": "Richlove Frimpong (see README.md § Author for contact)",
            "hyperparameters_and_provenance": (
                "Full hyperparameters, git SHA, DVC data hash, and the "
                "family-comparison paired-Δ are in training_manifest.json "
                "on this run — not duplicated here."
            ),
            "run_id": run_id,
            "eval_run_id": eval_run_id,
            "error_analysis_run_id": error_analysis_run_id,
            "mlflow_ui": (
                "Every artifact this card points to (training_manifest.json, "
                "calibration_summary.json, error_analysis.json, metrics.json, "
                "economics.json, promotion_decision.json, drift_reference.json) "
                "lives on one of the three run IDs above. Open the MLflow UI "
                "(README.md § Quick Start, step 8 — Browse experiment runs) "
                "and search a run ID there to browse its artifacts directly."
            ),
        },
        "monitoring": {
            "drift_reference_captured": True,
            "note": (
                "A drift-monitoring baseline (feature distributions, "
                "out-of-sample score distribution, churn prevalence) was "
                "captured at promotion; see drift_reference.json on this "
                "run."
            ),
        },
        "promotion_context": {
            "regime": decision.get("regime"),
            "previous_champion_version": metrics.get("champion_version"),
            "criteria": decision.get("criteria"),
        },
        "human_review": {
            "review": decision.get("review"),
            "review_notes": decision.get("review_notes"),
            "reviewed_at": decision.get("reviewed_at"),
        },
    }


def _build_technical_appendix_section(
    manifest: dict[str, Any], metrics: dict[str, Any], error_analysis: dict[str, Any]
) -> dict[str, Any]:
    """Build model_card.json's technical_appendix section.

    n_test/evaluation_data_description/n_dev/training_data_description are
    computed here, their only consumer, rather than in the parent assembler.
    """
    dev_oof_flags = error_analysis.get("dev_oof_diagnostics_carried_through", {})
    calibration = metrics.get("calibration", {})
    sliced = metrics.get("sliced", {})

    classification_rows = metrics.get("classification", [])
    n_test = (
        int(sum(classification_rows[0].get(k, 0) for k in ("tp", "fp", "fn", "tn")))
        if classification_rows
        else None
    )
    evaluation_data_description = (
        f"Sealed test set: n={n_test:,}, stratified by customerid, held out "
        "before feature discovery and touched once."
        if n_test is not None
        else (
            "Sealed test set description unavailable — classification rows "
            "missing from metrics.json."
        )
    )

    n_dev = manifest.get("tuning_summary", {}).get("n_final_fit")
    training_data_description = (
        f"{n_dev:,}-row development partition — the same rows and model "
        "whose metrics are reported here."
        if n_dev is not None
        else (
            "Development-partition description unavailable — n_final_fit "
            "missing from training_manifest.json."
        )
    )

    return {
        "evaluation_data": {
            "description": evaluation_data_description,
            "dummy_pr_auc_floor": metrics.get("ranking", {}).get("dummy_pr_auc_floor"),
        },
        "training_data": {
            "description": training_data_description,
            "feature_columns": manifest.get("feature_selection", {}).get(
                "model_features"
            ),
        },
        "performance_vs_baselines": {
            "pr_auc": metrics.get("ranking", {}).get("pr_auc"),
            "pr_auc_ci": [
                metrics.get("ranking", {}).get("pr_auc_ci_lower"),
                metrics.get("ranking", {}).get("pr_auc_ci_upper"),
            ],
            "classification_at_shipped_thresholds": metrics.get("classification"),
        },
        "fixed_recall_profile": metrics.get("fixed_recall_profile"),
        "calibration_quality": {
            "brier": calibration.get("brier"),
            "bss": calibration.get("bss"),
            "calibration_slope": calibration.get("calibration_slope", {}).get("slope"),
        },
        "factors": {
            "robustness_axes": list(ROBUSTNESS_AXES),
            "fairness_axes": list(FAIRNESS_AXES),
            "rationale": (
                "Robustness axes (contract type, tenure cohort, internet "
                "service) are the segments where the business impact of "
                "a ranking collapse is largest. Fairness axes (gender, "
                "senior citizen, partner/dependents status) are checked "
                "because they are protected or quasi-protected "
                "attributes present in the IBM Telco schema — an empty "
                "flag list below means these axes were examined and "
                "cleared, not that they were not examined."
            ),
        },
        "fairness_robustness": {
            "flags": {
                "v1_segment_collapse_flagged": dev_oof_flags.get("v1_flagged", []),
                "v2_equal_opportunity_flagged": dev_oof_flags.get(
                    "v2_equal_opportunity_flagged", {}
                ),
                "v2_demographic_parity_flagged": dev_oof_flags.get(
                    "v2_demographic_parity_flagged", {}
                ),
                "v2b_calibration_flagged": dev_oof_flags.get("v2b_flagged", []),
                "v3_direction_sanity_violations": error_analysis.get(
                    "direction_sanity_check", {}
                ).get("violations", []),
            },
            "disaggregated_results": {
                "dev_oof": sliced.get("dev_oof_diagnostics", {}),
                "sealed_test": sliced.get("test", {}),
            },
        },
        "error_concentration": error_analysis.get("error_concentration", {}),
    }


def _assemble_model_card(
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    economics: dict[str, Any],
    error_analysis: dict[str, Any],
    decision: dict[str, Any],
    calibration_method: str,
    run_id: str,
    eval_run_id: str,
    error_analysis_run_id: str,
    model_version: str,
    registered_model_name: str,
    alias: str,
    promoted_at: str,
) -> dict[str, Any]:
    """Assemble model_card.json strictly from already-computed artifacts.

    Shaped as a business memo, not Mitchell et al.'s academic model-card
    sectioning: executive_summary leads, business_case and
    risks_and_limitations follow in the order a stakeholder actually reads
    them, governance carries provenance/ownership plus the promotion-decision
    audit trail (regime, per-criterion breakdown, human review verdict — all
    read from decision, never recomputed), and technical_appendix
    holds everything an ML engineer or auditor would want to verify but a
    business reader does not need first. No hyperparameters, hashes, or raw
    statistical detail — those stay in training_manifest.json
    (governance.model_details.hyperparameters_and_provenance points there
    rather than duplicating any of it). Baselines throughout are
    treat-none/treat-all and the DummyClassifier prevalence floor, never
    LogReg (that family comparison lives in ANALYSIS.md §4 and
    training_manifest.json's own paired-Δ field).
    """
    return {
        "executive_summary": _summarize_for_stakeholders(metrics),
        "business_case": _build_business_case_section(
            manifest, metrics, economics, decision
        ),
        "risks_and_limitations": _build_risks_and_limitations_section(
            manifest, error_analysis
        ),
        "governance": _build_governance_section(
            manifest,
            metrics,
            decision,
            calibration_method,
            run_id,
            eval_run_id,
            error_analysis_run_id,
            model_version,
            registered_model_name,
            alias,
            promoted_at,
        ),
        "technical_appendix": _build_technical_appendix_section(
            manifest, metrics, error_analysis
        ),
    }


def _check_already_decided(
    current_version: Any, model_version: str
) -> dict[str, Any] | None:
    """Short-circuit a re-invocation of an already-decided cycle.

    Cheapest check first — needs nothing else resolved or read.
    """
    current_status = current_version.tags.get("promotion_status")
    if current_status in ("promoted", "rejected"):
        logger.info(
            "registration_already_decided",
            model_version=model_version,
            promotion_status=current_status,
        )
        return {
            "model_version": model_version,
            "promotion_status": current_status,
            "already_decided": True,
        }
    return None


def _load_and_verify_evaluation_artifacts(
    current_version: Any, model_version: str
) -> dict[str, Any]:
    """Resolve eval_run_id from the version's own tag and load+verify its decision/metrics.

    Never a local reports/ path (module docstring).
    """
    eval_run_id = current_version.tags.get("eval_run_id")
    if not eval_run_id:
        raise RuntimeError(
            f"Model version {model_version!r} has no eval_run_id tag — "
            "evaluate.py's registration step sets this; re-run "
            "models.evaluate before registering."
        )

    decision = mlflow.artifacts.load_dict(
        f"runs:/{eval_run_id}/promotion_decision.json"
    )
    decision_model_version = str(decision["model_version"])
    if decision_model_version != str(model_version):
        raise ValueError(
            f"promotion_decision.json (run {eval_run_id!r}) was computed for "
            f"model_version={decision_model_version!r}, not {model_version!r} "
            "— refusing to act on a decision for a different artifact."
        )

    metrics = mlflow.artifacts.load_dict(f"runs:/{eval_run_id}/metrics.json")
    stamped_metrics_hash = str(decision.get("metrics_content_hash"))
    if content_hash(metrics) != stamped_metrics_hash:
        raise ValueError(
            "promotion_decision.json's metrics_content_hash does not match a "
            f"fresh hash of metrics.json on run {eval_run_id!r} — the "
            "decision was computed against a different metrics.json than the "
            "one currently logged (a stale decision from a previous cycle, "
            "or metrics.json regenerated independently onto the same run). "
            "Refusing to act on a decision that cannot be verified against "
            "its own evidence."
        )

    return {"eval_run_id": eval_run_id, "decision": decision, "metrics": metrics}


def _check_review_approval(cfg: DictConfig, decision: dict[str, Any]) -> None:
    """Raise if a stamped human review is required but absent."""
    if bool(cfg.register.require_review) and decision.get("review") != "approved":
        raise RuntimeError(
            f"promotion_decision.json's review is {decision.get('review')!r}, "
            "not 'approved' — register.require_review is set and refuses to "
            "act without a stamped human review."
        )


def _tag_rejected(
    client: Any,
    registered_model_name: str,
    model_version: str,
    decision: dict[str, Any],
    reason: str,
) -> None:
    """Tag, describe, and log a rejection — the caller still decides return vs. raise.

    Shared by all three reject sites (gate_fail, smoke_check_failed,
    post_flip_parity_failed), which differ in control flow: gate_fail
    returns normally, the other two raise from inside an except block. This
    helper only ever tags/logs — unifying the return-vs-raise decision here
    too would either swallow a traceback or return instead of propagating.
    """
    client.set_model_version_tag(
        registered_model_name, model_version, "promotion_status", "rejected"
    )
    client.update_model_version(
        registered_model_name,
        model_version,
        description=_rejection_description(decision, reason),
    )
    logger.info(
        "model_rejected",
        model_version=model_version,
        regime=decision["regime"],
        reason=reason,
    )


def _load_and_verify_error_analysis(
    client: Any, current_version: Any, model_version: str
) -> dict[str, Any]:
    """Resolve error_analysis_run_id from the version's own tag and load+verify error_analysis.json.

    Verified against this model_version too, so a stale error_analysis.json
    from a different cycle can't be silently attributed to this one.
    """
    error_analysis_run_id = current_version.tags.get("error_analysis_run_id")
    if not error_analysis_run_id:
        raise RuntimeError(
            f"Model version {model_version!r} has no error_analysis_run_id "
            "tag — error_analysis.py's registration step sets this; the "
            "model card's fairness/robustness section cannot be assembled "
            "without its error_analysis.json. Re-run models.error_analysis "
            "before registering."
        )
    error_analysis_run_artifacts = {
        a.path for a in client.list_artifacts(error_analysis_run_id)
    }
    if "error_analysis.json" not in error_analysis_run_artifacts:
        raise RuntimeError(
            f"Run {error_analysis_run_id!r} (tagged error_analysis_run_id on "
            f"model version {model_version!r}) has no error_analysis.json "
            "artifact — the model card's fairness/robustness section cannot "
            "be assembled without it."
        )
    error_analysis = mlflow.artifacts.load_dict(
        f"runs:/{error_analysis_run_id}/error_analysis.json"
    )
    error_analysis_model_version = str(error_analysis.get("model_version"))
    if error_analysis_model_version != str(model_version):
        raise ValueError(
            f"error_analysis.json (run {error_analysis_run_id!r}) was "
            f"computed for model_version={error_analysis_model_version!r}, "
            f"not {model_version!r} — refusing to assemble a model card from "
            "diagnostics for a different artifact."
        )
    return {
        "error_analysis_run_id": error_analysis_run_id,
        "error_analysis": error_analysis,
    }


def _run_pre_flip_checks(
    cfg: DictConfig,
    client: Any,
    registered_model_name: str,
    model_version: str,
    committed_features: list[str],
    X_dev: pd.DataFrame,
    run_id: str,
    decision: dict[str, Any],
) -> dict[str, Any]:
    """Phase 1: pre-flip checks, by explicit version URI.

    Tags rejected and re-raises on a genuine smoke-check failure — the only
    reject path here that isn't a plain abort.
    """
    model_uri = f"models:/{registered_model_name}/{model_version}"

    environment_packages = list(cfg.register.environment_packages)
    check_environment_parity(model_uri, environment_packages)

    model = mlflow.sklearn.load_model(model_uri)
    golden = mlflow.artifacts.load_dict(
        f"runs:/{run_id}/calibration/golden_predictions.json"
    )
    calibration_summary = mlflow.artifacts.load_dict(
        f"runs:/{run_id}/calibration/calibration_summary.json"
    )
    golden_atol = float(cfg.register.golden_atol)

    try:
        _smoke_check_schema_and_range(model, committed_features, X_dev.head(5))
        _check_golden_parity(model, golden, golden_atol)
    except AssertionError:
        _tag_rejected(
            client, registered_model_name, model_version, decision, "smoke_check_failed"
        )
        raise

    _log_smoke_diagnostics(model, pd.DataFrame(golden["rows"]), model_uri, run_id)

    return {
        "golden": golden,
        "calibration_summary": calibration_summary,
        "golden_atol": golden_atol,
    }


def _reverify_incumbent(cfg: DictConfig, metrics: dict[str, Any]) -> Any:
    """Re-verify the incumbent immediately before flipping, not earlier.

    `metrics` was already loaded (and hash-verified) by
    _load_and_verify_evaluation_artifacts — reused here rather than
    re-read, so there is exactly one load of
    runs:/<eval_run_id>/metrics.json per invocation.
    """
    recorded_champion_version = metrics.get("champion_version")
    if recorded_champion_version is not None:
        live_champion_version = resolve_champion_version(cfg)
        if live_champion_version != str(recorded_champion_version):
            raise RuntimeError(
                f"champion moved from {recorded_champion_version!r} "
                f"(recorded at evaluation time) to {live_champion_version!r} "
                "since evaluation — refusing to flip against a stale "
                "comparison."
            )
    return recorded_champion_version


def _flip_and_confirm(
    client: Any,
    registered_model_name: str,
    model_version: str,
    alias: str,
    golden: dict[str, Any],
    golden_atol: float,
    decision: dict[str, Any],
) -> None:
    """Phase 2: flip the alias, then confirm parity through the alias itself.

    Rolls the alias back, tags rejected, and re-raises on a post-flip parity
    failure — the alias must never point at a model that failed its own
    serving check.
    """
    promote_to_alias(registered_model_name, model_version, alias)
    reloaded = mlflow.sklearn.load_model(f"models:/{registered_model_name}@{alias}")
    try:
        _check_golden_parity(reloaded, golden, golden_atol)
    except AssertionError:
        rollback_champion(registered_model_name)
        _tag_rejected(
            client,
            registered_model_name,
            model_version,
            decision,
            "post_flip_parity_failed",
        )
        raise


def _build_and_log_drift_reference(
    client: Any,
    cfg: DictConfig,
    run_id: str,
    eval_run_id: str,
    X_dev: pd.DataFrame,
    reports_dir: Path,
) -> None:
    """Build the drift reference, log it onto the promoted run, and mirror it to reports/."""
    oos_proba, y_combined = _build_drift_reference_inputs(run_id, eval_run_id, cfg)
    drift_reference = build_reference(
        X_dev, oos_proba, y_combined, n_bins=int(cfg.register.drift_reference_n_bins)
    )
    client.log_dict(run_id, drift_reference, "registration/drift_reference.json")
    with open(reports_dir / "drift_reference.json", "w", encoding="utf-8") as f:
        json.dump(drift_reference, f, indent=2, default=str)


def _build_and_log_model_card(
    client: Any,
    cfg: DictConfig,
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    error_analysis: dict[str, Any],
    decision: dict[str, Any],
    calibration_summary: dict[str, Any],
    run_id: str,
    eval_run_id: str,
    error_analysis_run_id: str,
    model_version: str,
    registered_model_name: str,
    alias: str,
    reports_dir: Path,
) -> dict[str, Any]:
    """Assemble the model card, log it onto the promoted run, and mirror it to reports/."""
    eval_run_artifacts = {a.path for a in client.list_artifacts(eval_run_id)}
    economics = (
        mlflow.artifacts.load_dict(f"runs:/{eval_run_id}/economics.json")
        if "economics.json" in eval_run_artifacts
        else {}
    )
    model_card = _assemble_model_card(
        manifest,
        metrics,
        economics,
        error_analysis,
        decision,
        str(calibration_summary.get("method", "unknown")),
        run_id,
        eval_run_id,
        error_analysis_run_id,
        model_version,
        registered_model_name,
        alias,
        datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
    )
    client.log_dict(run_id, model_card, "registration/model_card.json")
    with open(reports_dir / "model_card.json", "w", encoding="utf-8") as f:
        json.dump(model_card, f, indent=2, default=str)
    return model_card


def _finalize_promotion(
    client: Any,
    registered_model_name: str,
    model_version: str,
    decision: dict[str, Any],
    model_card: dict[str, Any],
    recorded_champion_version: Any,
) -> str:
    """Tag the version promoted, describe it, and append the promotion event."""
    promoted_at = str(model_card["governance"]["model_details"]["promoted_at"])
    client.set_model_version_tag(
        registered_model_name, model_version, "promotion_status", "promoted"
    )
    client.update_model_version(
        registered_model_name,
        model_version,
        description=_promotion_description(decision, recorded_champion_version),
    )
    _append_promotion_event(
        registered_model_name,
        {
            "action": "promoted",
            "version": model_version,
            "previous_champion_version": recorded_champion_version,
            "at": promoted_at,
        },
    )
    return promoted_at


def run_registration_step(model_version: str, cfg: DictConfig) -> dict[str, Any]:
    """Act on evaluate.py's persisted promotion decision for `model_version`.

    See the module docstring for the full step order. Returns a dict
    describing the outcome: {"model_version", "promotion_status", ...}.
    Raises on any abort that is not a genuine pass/fail verdict (missing
    manifest artifacts, missing error_analysis.json, an environment
    mismatch, a stale incumbent) — those leave promotion_status untouched.
    A genuine smoke-check failure tags "rejected" and re-raises.
    """
    ensure_experiment_metadata(cfg)
    registered_model_name = str(cfg.mlflow.registered_model_name)
    alias = str(cfg.register.alias)
    client = mlflow.tracking.MlflowClient()

    current_version = client.get_model_version(registered_model_name, model_version)
    already_decided = _check_already_decided(current_version, model_version)
    if already_decided is not None:
        return already_decided

    eval_artifacts = _load_and_verify_evaluation_artifacts(
        current_version, model_version
    )
    eval_run_id = eval_artifacts["eval_run_id"]
    decision = eval_artifacts["decision"]
    metrics = eval_artifacts["metrics"]

    _check_review_approval(cfg, decision)

    if decision["gate"] == "fail":
        _tag_rejected(
            client, registered_model_name, model_version, decision, "gate_fail"
        )
        return {"model_version": model_version, "promotion_status": "rejected"}

    run_id = str(current_version.run_id)
    _check_manifest_artifacts(run_id)

    ea_artifacts = _load_and_verify_error_analysis(
        client, current_version, model_version
    )
    error_analysis_run_id = ea_artifacts["error_analysis_run_id"]
    error_analysis = ea_artifacts["error_analysis"]

    manifest = load_training_manifest(run_id, cfg)
    committed_features = committed_features_from_manifest(manifest)
    X_dev, _y_dev = load_dev_features(committed_features)

    pre_flip = _run_pre_flip_checks(
        cfg,
        client,
        registered_model_name,
        model_version,
        committed_features,
        X_dev,
        run_id,
        decision,
    )

    recorded_champion_version = _reverify_incumbent(cfg, metrics)

    _flip_and_confirm(
        client,
        registered_model_name,
        model_version,
        alias,
        pre_flip["golden"],
        pre_flip["golden_atol"],
        decision,
    )

    # --- On promotion: drift_reference.json + model_card.json ---
    # reports_dir is write-only from here on — a human-inspection mirror,
    # never a source this module reads back from (module docstring).
    reports_dir = get_project_root() / str(cfg.paths.reports)
    reports_dir.mkdir(parents=True, exist_ok=True)

    _build_and_log_drift_reference(client, cfg, run_id, eval_run_id, X_dev, reports_dir)

    model_card = _build_and_log_model_card(
        client,
        cfg,
        manifest,
        metrics,
        error_analysis,
        decision,
        pre_flip["calibration_summary"],
        run_id,
        eval_run_id,
        error_analysis_run_id,
        model_version,
        registered_model_name,
        alias,
        reports_dir,
    )

    promoted_at = _finalize_promotion(
        client,
        registered_model_name,
        model_version,
        decision,
        model_card,
        recorded_champion_version,
    )

    logger.info(
        "model_promoted",
        model_version=model_version,
        regime=decision["regime"],
        alias=alias,
        promoted_at=promoted_at,
    )

    return {
        "model_version": model_version,
        "promotion_status": "promoted",
        "alias": alias,
        "promoted_at": promoted_at,
    }


if __name__ == "__main__":
    import sys

    from dotenv import load_dotenv

    from telco_churn.utils.logging import configure_logging
    from telco_churn.utils.paths import compose_config

    load_dotenv()
    configure_logging()

    try:
        cfg = compose_config(overrides=sys.argv[1:] or None)
        cli_model_version = cfg.register.model_version
        if cli_model_version is None:
            raise ValueError(
                "register.model_version is required, e.g. `python -m "
                "telco_churn.models.register register.model_version=2`"
            )
        result = run_registration_step(str(cli_model_version), cfg)
        logger.info("registration_step_done", **result)
    except EnvironmentMismatchError as e:
        logger.error("registration_environment_mismatch", error=str(e), exc_info=True)
        sys.exit(1)
    except ValueError as e:
        logger.error("registration_invalid", error=str(e), exc_info=True)
        sys.exit(1)
    except RuntimeError as e:
        logger.error("registration_blocked", error=str(e), exc_info=True)
        sys.exit(1)
    except AssertionError as e:
        logger.error("registration_smoke_check_failed", error=str(e), exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error("registration_failed", error=str(e), exc_info=True)
        sys.exit(1)

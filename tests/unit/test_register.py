"""Unit tests for telco_churn.models.register.

Every test registers a real (small, fast — LogisticRegression, not the full
LightGBM pipeline) model version against a tmp-scoped MLflow experiment, with
the artifacts register.py's checks expect logged onto the run. register.py
reads its cross-cycle inputs (promotion_decision.json, metrics.json,
economics.json, test_predictions.parquet, error_analysis.json) by explicit
run_id, resolved from eval_run_id/error_analysis_run_id tags on the model version —
mirroring evaluate.py/error_analysis.py's real behavior — never from a local
reports/ path, so these tests log real 'evaluation'/'error_analysis' MLflow
runs rather than writing local fixture files. reports/ is still sandboxed to
tmp_path (register.py mirrors its writes there), matching the pattern
registration_cfg uses in test_calibrate.py. load_dev_features/
load_training_manifest/resolve_champion_version are monkeypatched on the
register module's own namespace (patch-where-used, since register.py imports
them by name).
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import mlflow.artifacts
import mlflow.sklearn
import mlflow.tracking
import numpy as np
import pandas as pd
import pytest
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import LogisticRegression

import telco_churn.models.register as register
from telco_churn.models.evaluate import content_hash
from telco_churn.utils.mlflow import write_error_analysis_receipt, write_eval_receipt

_COMMITTED_FEATURES = ["tenure", "monthlycharges"]


@pytest.fixture(scope="module")
def register_mlflow_uri(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Point MLflow at a module-scoped tmp SQLite store with an explicit
    artifact_location — inlines conftest.py::mlflow_test_experiment's logic
    rather than requesting it, since that fixture depends on the
    function-scoped tmp_path.

    Sharing the experiment/tracking store across the whole file (skipping the
    ~2.5-3.5s SQLite-store bootstrap per test) is safe *only* because
    register_cfg below gives every test its own uniquely-named registered
    model — register.py's rollback_champion/register_model_version operate
    per registered_model_name, and several tests here (e.g.
    test_rollback_champion_raises_when_nothing_promoted) assert on that
    namespace being otherwise empty, which a shared name across tests would
    break. Do not revert registered_model_name to a fixed string without
    also reverting this fixture to function scope.
    """
    tmp_path = tmp_path_factory.mktemp("register_mlflow")
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    artifact_location = (tmp_path / "artifacts").as_uri()
    experiment_id = mlflow.create_experiment(
        "test_register", artifact_location=artifact_location
    )
    mlflow.set_experiment(experiment_id=experiment_id)
    return tracking_uri


@pytest.fixture
def register_cfg(
    register_mlflow_uri: str, tmp_path: Path, request: pytest.FixtureRequest
) -> DictConfig:
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    registered_model_name = "test-telco-churn-pipeline-" + re.sub(
        r"[^A-Za-z0-9_-]", "_", request.node.name
    )
    return OmegaConf.create(
        {
            "mlflow": {
                "tracking_uri": register_mlflow_uri,
                "experiment_name": "test_register",
                "registered_model_name": registered_model_name,
            },
            "paths": {"reports": str(reports_dir)},
            "register": {
                "model_version": None,
                "require_review": True,
                "alias": "champion",
                "golden_atol": 1e-9,
                "environment_packages": [],
                "drift_reference_n_bins": 5,
            },
        }
    )


@pytest.fixture
def X_golden() -> pd.DataFrame:
    rng = np.random.RandomState(42)
    return pd.DataFrame(
        {
            "tenure": rng.randint(1, 72, size=8),
            "monthlycharges": rng.uniform(20, 120, size=8),
        }
    )


@pytest.fixture
def fitted_model(X_golden: pd.DataFrame) -> LogisticRegression:
    y = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    model = LogisticRegression()
    model.fit(X_golden, y)
    return model


def _register_test_version(
    registered_model_name: str,
    fitted_model: LogisticRegression,
    X_golden: pd.DataFrame,
    promotion_status: str | None,
    include_manifest_artifacts: bool = True,
    include_golden: bool = True,
    perturb_golden: bool = False,
) -> tuple[str, str]:
    """Register `fitted_model`, logging the artifacts register.py's checks expect.

    Returns (version, run_id).
    """
    signature = infer_signature(X_golden, fitted_model.predict_proba(X_golden))
    with mlflow.start_run(run_name="tuning_study") as run:
        run_id = run.info.run_id
        if include_manifest_artifacts:
            mlflow.log_text("dummy", "feature_space.txt")
            mlflow.log_text("dummy", "feature_columns.txt")
            mlflow.log_text("dummy", "preprocessing.pkl")
            mlflow.log_dict(
                {
                    "model_family": "lightgbm",
                    "feature_selection": {"model_features": _COMMITTED_FEATURES},
                },
                "training_manifest.json",
            )
        if include_golden:
            golden_p_hat = fitted_model.predict_proba(X_golden)[:, 1]
            if perturb_golden:
                golden_p_hat = np.clip(golden_p_hat + 0.4, 0.0, 1.0)
            mlflow.log_dict(
                {
                    "purpose": (
                        "serving-parity fixture — reproducibility only; "
                        "scores are in-sample and are not performance evidence"
                    ),
                    "customerid": [f"cust-{i:04d}" for i in range(len(X_golden))],
                    "rows": X_golden.to_dict(orient="records"),
                    "p_hat": golden_p_hat.tolist(),
                },
                "calibration/golden_predictions.json",
            )
            mlflow.log_dict(
                {"method": "sigmoid"},
                "calibration/calibration_summary.json",
            )
        # _build_drift_reference_inputs unconditionally fetches this via
        # threshold.load_dev_oof_predictions(run_id, cfg) on the full-pass
        # path — must exist on every registered run, not just the ones that
        # reach promotion.
        dev_oof = pd.DataFrame(
            {"customerid": ["a", "b"], "y_true": [0, 1], "p_hat": [0.2, 0.8]}
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            dev_oof_path = Path(tmp_dir) / "dev_oof_predictions.parquet"
            dev_oof.to_parquet(dev_oof_path, index=False)
            mlflow.log_artifact(str(dev_oof_path), artifact_path="calibration")
        model_info = mlflow.sklearn.log_model(
            sk_model=fitted_model,
            name="calibrated_model",
            signature=signature,
            input_example=X_golden,
            registered_model_name=registered_model_name,
        )
    version = str(model_info.registered_model_version)
    if promotion_status is not None:
        client = mlflow.tracking.MlflowClient()
        client.set_model_version_tag(
            registered_model_name, version, "promotion_status", promotion_status
        )
    return version, run_id


def _log_evaluation_and_error_analysis(
    registered_model_name: str,
    version: str,
    *,
    decision_model_version: str | None = None,
    gate: str = "pass",
    review: str | None = "approved",
    regime: str = "cold_start",
    champion_version: str | None = None,
    include_error_analysis: bool = True,
    error_analysis_model_version: str | None = None,
    tag_version: bool = True,
) -> tuple[str, str | None]:
    """Log a real 'evaluation' run (promotion_decision.json/metrics.json/
    economics.json/test_predictions.parquet), a real 'error_analysis' run
    (error_analysis.json), and — if `review` is not None — a real
    promotion_review.json (one entry, that verdict) on the eval run. Then,
    if `tag_version`, tags `version` with eval_run_id/error_analysis_run_id
    — mirroring what register.py itself does at mint/first-pass (B1:
    evaluate.py/error_analysis.py write no model-version tags themselves
    anymore), since register.py resolves both exclusively from tag-or-
    receipt, never a local reports/ path.

    decision_model_version/error_analysis_model_version default to `version`
    but can be overridden to simulate a stale/mismatched artifact (a
    promotion_decision.json or error_analysis.json computed for a different
    model_version than the one being registered).
    include_error_analysis=False skips the 'error_analysis' run and its
    error_analysis_run_id tag entirely, simulating error_analysis.py never
    having run. review=None skips promotion_review.json entirely, simulating
    "no review recorded yet" (require_review blocks on this). tag_version=False
    skips both tag writes, for tests exercising register.py's own receipt-
    bootstrap resolution instead (its first-pass tagging is exactly what
    this fixture otherwise pre-empts).

    Returns (eval_run_id, error_analysis_run_id | None).
    """
    decision_model_version = decision_model_version or version
    error_analysis_model_version = error_analysis_model_version or version
    client = mlflow.tracking.MlflowClient()

    metrics_payload = {
        "champion_version": champion_version,
        "ranking": {"pr_auc": 0.65, "dummy_pr_auc_floor": 0.265},
        # A "base" row is load-bearing: register.py's _tag_gate_criteria (B1)
        # reads test_recall off it when tagging every processed version, not
        # only promoted ones — matching evaluate.py's old unconditional-tag
        # behavior.
        "classification": [{"scenario": "base", "recall": 0.70}],
        "fixed_recall_profile": [],
        "calibration": {"brier": 0.13, "bss": 0.3, "calibration_slope": {"slope": 1.0}},
        "business_impact": {"scenarios": {}},
    }
    promotion_decision_payload = {
        "model_version": decision_model_version,
        "gate": gate,
        "regime": regime,
        "criteria": {},
        "metrics_content_hash": content_hash(metrics_payload),
    }
    test_predictions = pd.DataFrame(
        {"customerid": ["c", "d"], "y_true": [1, 0], "p_hat": [0.7, 0.1]}
    )

    with mlflow.start_run(run_name="evaluation") as run:
        eval_run_id = run.info.run_id
        mlflow.log_dict(metrics_payload, "metrics.json")
        mlflow.log_dict({}, "economics.json")
        mlflow.log_dict(promotion_decision_payload, "promotion_decision.json")
        with tempfile.TemporaryDirectory() as tmp_dir:
            predictions_path = Path(tmp_dir) / "test_predictions.parquet"
            test_predictions.to_parquet(predictions_path, index=False)
            mlflow.log_artifact(str(predictions_path))
        if review is not None:
            promotion_review_payload = {
                "eval_run_id": eval_run_id,
                "metrics_content_hash": promotion_decision_payload[
                    "metrics_content_hash"
                ],
                "entries": [
                    {
                        "verdict": review,
                        "notes": "",
                        "approver": "test",
                        "reviewed_at": "t",
                    }
                ],
            }
            mlflow.log_dict(promotion_review_payload, "promotion_review.json")
    if tag_version:
        client.set_model_version_tag(
            registered_model_name, version, "eval_run_id", eval_run_id
        )

    error_analysis_run_id: str | None = None
    if include_error_analysis:
        error_analysis_payload = {
            "model_version": error_analysis_model_version,
            "dev_oof_diagnostics_carried_through": {
                "v1_flagged": [],
                "v2_equal_opportunity_flagged": {},
                "v2_demographic_parity_flagged": {},
                "v2b_flagged": [],
            },
            "direction_sanity_check": {"violations": []},
            "error_concentration": {},
        }
        with mlflow.start_run(run_name="error_analysis") as run:
            error_analysis_run_id = run.info.run_id
            mlflow.log_dict(error_analysis_payload, "error_analysis.json")
        if tag_version:
            client.set_model_version_tag(
                registered_model_name,
                version,
                "error_analysis_run_id",
                error_analysis_run_id,
            )

    return eval_run_id, error_analysis_run_id


def _patch_calibrate_helpers(
    monkeypatch: pytest.MonkeyPatch, X_dev: pd.DataFrame
) -> None:
    def _fake_load_training_manifest(run_id: str, cfg: DictConfig) -> dict[str, Any]:
        return {
            "model_family": "lightgbm",
            "feature_selection": {"model_features": _COMMITTED_FEATURES},
        }

    def _fake_load_dev_features(
        committed_features: list[str],
    ) -> tuple[pd.DataFrame, pd.Series]:
        return X_dev[committed_features], pd.Series([0] * len(X_dev))

    monkeypatch.setattr(
        register, "load_training_manifest", _fake_load_training_manifest
    )
    monkeypatch.setattr(register, "load_dev_features", _fake_load_dev_features)


@pytest.fixture
def X_dev(X_golden: pd.DataFrame) -> pd.DataFrame:
    """A dev-partition stand-in bigger than the golden sample, same columns."""
    rng = np.random.RandomState(7)
    extra = pd.DataFrame(
        {
            "tenure": rng.randint(1, 72, size=20),
            "monthlycharges": rng.uniform(20, 120, size=20),
        }
    )
    return pd.concat([X_golden, extra], ignore_index=True)


# ---------------------------------------------------------------------------
# register_challenger — the calibrate.py registration extraction (PR B1)
# ---------------------------------------------------------------------------


def test_register_challenger_mints_pending_and_points_challenger_alias(
    register_cfg: DictConfig,
    fitted_model: LogisticRegression,
    X_golden: pd.DataFrame,
) -> None:
    """register_challenger — moved out of calibrate.py in B1 — mints the
    version, tags it pending immediately, and points `challenger` at it only
    after reload parity passes."""
    registered_name = str(register_cfg.mlflow.registered_model_name)
    signature = infer_signature(X_golden, fitted_model.predict_proba(X_golden))
    with mlflow.start_run(run_name="tuning_study"):
        model_info = mlflow.sklearn.log_model(
            sk_model=fitted_model,
            name="calibrated_model",
            signature=signature,
            input_example=X_golden,
            registered_model_name=registered_name,
        )
    in_memory_preds = fitted_model.predict_proba(X_golden)

    version = register.register_challenger(
        register_cfg, model_info, X_golden, in_memory_preds
    )

    client = mlflow.tracking.MlflowClient()
    tags = client.get_model_version(registered_name, version).tags
    assert tags["promotion_status"] == "pending"
    assert tags["logged_model_id"] == model_info.model_id
    assert tags["training_data_scope"] == "dev"
    challenger = client.get_model_version_by_alias(registered_name, "challenger")
    assert str(challenger.version) == version


def test_register_challenger_leaves_pending_and_no_alias_on_parity_failure(
    register_cfg: DictConfig,
    fitted_model: LogisticRegression,
    X_golden: pd.DataFrame,
) -> None:
    """A reload-parity mismatch raises, but the mint-time `pending` tag
    written just before the check must survive — the fail-safe default a
    crash between mint and verdict is supposed to get for free — and
    `challenger` must never point at the unverified version."""
    registered_name = str(register_cfg.mlflow.registered_model_name)
    signature = infer_signature(X_golden, fitted_model.predict_proba(X_golden))
    with mlflow.start_run(run_name="tuning_study"):
        model_info = mlflow.sklearn.log_model(
            sk_model=fitted_model,
            name="calibrated_model",
            signature=signature,
            input_example=X_golden,
            registered_model_name=registered_name,
        )
    wrong_preds = np.zeros((len(X_golden), 2))

    with pytest.raises(AssertionError, match="Reload parity"):
        register.register_challenger(register_cfg, model_info, X_golden, wrong_preds)

    client = mlflow.tracking.MlflowClient()
    version = str(model_info.registered_model_version)
    tags = client.get_model_version(registered_name, version).tags
    assert tags["promotion_status"] == "pending"
    with pytest.raises(MlflowException):
        client.get_model_version_by_alias(registered_name, "challenger")


# ---------------------------------------------------------------------------
# Reads-not-recomputes, version mismatch, idempotency, fail-fast ordering
# ---------------------------------------------------------------------------


def test_register_aborts_on_model_version_mismatch(
    register_cfg: DictConfig,
    fitted_model: LogisticRegression,
    X_golden: pd.DataFrame,
) -> None:
    registered_name = str(register_cfg.mlflow.registered_model_name)
    version, _run_id = _register_test_version(
        registered_name, fitted_model, X_golden, promotion_status="pending"
    )
    _log_evaluation_and_error_analysis(
        registered_name, version, decision_model_version="999"
    )

    with pytest.raises(ValueError, match="different artifact"):
        register.run_registration_step(version, register_cfg)


def test_register_idempotent_on_already_promoted(
    register_cfg: DictConfig,
    fitted_model: LogisticRegression,
    X_golden: pd.DataFrame,
    X_dev: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second invocation against an already-decided version is a no-op —
    it must not re-run the smoke check or re-emit model_promoted.

    Deliberately does not tag eval_run_id/error_analysis_run_id at all: the
    already-decided short-circuit must fire before resolving either, so a
    re-invocation needs no readable evaluation artifacts to succeed.
    """
    registered_name = str(register_cfg.mlflow.registered_model_name)
    version, _run_id = _register_test_version(
        registered_name, fitted_model, X_golden, promotion_status="promoted"
    )
    _patch_calibrate_helpers(monkeypatch, X_dev)

    check_calls: list[str] = []

    def _record_call(*args: Any, **kwargs: Any) -> dict[str, str]:
        check_calls.append("called")
        return {}

    monkeypatch.setattr(register, "check_environment_parity", _record_call)

    result = register.run_registration_step(version, register_cfg)

    assert result["already_decided"] is True
    assert result["promotion_status"] == "promoted"
    assert check_calls == []


def test_register_fail_fast_skips_smoke_check_on_gate_fail(
    register_cfg: DictConfig,
    fitted_model: LogisticRegression,
    X_golden: pd.DataFrame,
    X_dev: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A gate: fail decision must never reach the manifest gate or the smoke
    check — cheapest, most decisive check first.
    """
    registered_name = str(register_cfg.mlflow.registered_model_name)
    version, _run_id = _register_test_version(
        registered_name,
        fitted_model,
        X_golden,
        promotion_status="pending",
        include_manifest_artifacts=False,  # would fail the manifest gate if reached
    )
    _log_evaluation_and_error_analysis(registered_name, version, gate="fail")
    _patch_calibrate_helpers(monkeypatch, X_dev)

    smoke_check_calls: list[str] = []

    def _record_call(*args: Any, **kwargs: Any) -> dict[str, str]:
        smoke_check_calls.append("called")
        return {}

    monkeypatch.setattr(register, "check_environment_parity", _record_call)

    result = register.run_registration_step(version, register_cfg)

    assert result["promotion_status"] == "rejected"
    assert smoke_check_calls == []

    client = mlflow.tracking.MlflowClient()
    tags = client.get_model_version(registered_name, version).tags
    assert tags["promotion_status"] == "rejected"


def test_register_requires_approved_review(
    register_cfg: DictConfig,
    fitted_model: LogisticRegression,
    X_golden: pd.DataFrame,
) -> None:
    registered_name = str(register_cfg.mlflow.registered_model_name)
    version, _run_id = _register_test_version(
        registered_name, fitted_model, X_golden, promotion_status="pending"
    )
    _log_evaluation_and_error_analysis(registered_name, version, review=None)

    with pytest.raises(RuntimeError, match="review"):
        register.run_registration_step(version, register_cfg)


def test_register_human_rejected_review_tags_rejected_not_pending(
    register_cfg: DictConfig,
    fitted_model: LogisticRegression,
    X_golden: pd.DataFrame,
) -> None:
    """A human 'rejected' verdict routes through _tag_rejected like the other
    three reject sites (gate_fail, smoke_check_failed, post_flip_parity_failed)
    — the live defect B1 fixes: left at `pending`, a deliberate human
    rejection would be indistinguishable from a crash artifact."""
    registered_name = str(register_cfg.mlflow.registered_model_name)
    version, _run_id = _register_test_version(
        registered_name, fitted_model, X_golden, promotion_status="pending"
    )
    _log_evaluation_and_error_analysis(registered_name, version, review="rejected")

    result = register.run_registration_step(version, register_cfg)

    assert result["promotion_status"] == "rejected"
    client = mlflow.tracking.MlflowClient()
    tags = client.get_model_version(registered_name, version).tags
    assert tags["promotion_status"] == "rejected"
    assert "human review" in str(
        client.get_model_version(registered_name, version).description
    )


def test_register_resolves_eval_and_error_analysis_run_id_from_receipts_on_first_pass(
    register_cfg: DictConfig,
    fitted_model: LogisticRegression,
    X_golden: pd.DataFrame,
    X_dev: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no eval_run_id/error_analysis_run_id tags yet — this version's
    first registration pass — register.py bootstraps both from
    reports/eval_receipt.json/reports/error_analysis_receipt.json (the
    pointers evaluate.py/error_analysis.py write in the real pipeline) and
    tags the version itself, so a full pass still completes and a re-run
    resolves purely from the now-present tags."""
    registered_name = str(register_cfg.mlflow.registered_model_name)
    version, _run_id = _register_test_version(
        registered_name, fitted_model, X_golden, promotion_status="pending"
    )
    eval_run_id, error_analysis_run_id = _log_evaluation_and_error_analysis(
        registered_name, version, tag_version=False
    )
    assert error_analysis_run_id is not None
    write_eval_receipt(version, eval_run_id, register_cfg)
    write_error_analysis_receipt(version, error_analysis_run_id, register_cfg)
    _patch_calibrate_helpers(monkeypatch, X_dev)
    monkeypatch.setattr(register, "check_environment_parity", lambda *a, **k: {})

    client = mlflow.tracking.MlflowClient()
    assert "eval_run_id" not in client.get_model_version(registered_name, version).tags

    result = register.run_registration_step(version, register_cfg)

    assert result["promotion_status"] == "promoted"
    tags = client.get_model_version(registered_name, version).tags
    assert tags["eval_run_id"] == eval_run_id
    assert tags["error_analysis_run_id"] == error_analysis_run_id
    assert tags["test_pr_auc"] == "0.65"
    assert tags["test_recall"] == "0.7"


def test_register_receipt_for_different_model_version_raises(
    register_cfg: DictConfig,
    fitted_model: LogisticRegression,
    X_golden: pd.DataFrame,
) -> None:
    """A stale reports/eval_receipt.json — written for a different model
    version, e.g. left over from an earlier cycle on the same machine —
    must not be silently trusted for this one."""
    registered_name = str(register_cfg.mlflow.registered_model_name)
    version, _run_id = _register_test_version(
        registered_name, fitted_model, X_golden, promotion_status="pending"
    )
    eval_run_id, _ea_run_id = _log_evaluation_and_error_analysis(
        registered_name, version, tag_version=False
    )
    write_eval_receipt("999", eval_run_id, register_cfg)

    with pytest.raises(RuntimeError, match="eval_receipt"):
        register.run_registration_step(version, register_cfg)


# ---------------------------------------------------------------------------
# Manifest gate + error_analysis gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "missing_artifact",
    [
        "feature_space.txt",
        "feature_columns.txt",
        "preprocessing.pkl",
        "training_manifest.json",
    ],
)
def test_register_manifest_gate_blocks_on_any_missing_artifact(
    register_cfg: DictConfig,
    fitted_model: LogisticRegression,
    X_golden: pd.DataFrame,
    X_dev: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    missing_artifact: str,
) -> None:
    registered_name = str(register_cfg.mlflow.registered_model_name)
    signature = infer_signature(X_golden, fitted_model.predict_proba(X_golden))
    all_artifacts = [
        "feature_space.txt",
        "feature_columns.txt",
        "preprocessing.pkl",
        "training_manifest.json",
    ]
    with mlflow.start_run(run_name="tuning_study"):
        for name in all_artifacts:
            if name != missing_artifact:
                mlflow.log_text("dummy", name)
        model_info = mlflow.sklearn.log_model(
            sk_model=fitted_model,
            name="calibrated_model",
            signature=signature,
            input_example=X_golden,
            registered_model_name=registered_name,
        )
    version = str(model_info.registered_model_version)
    client = mlflow.tracking.MlflowClient()
    client.set_model_version_tag(
        registered_name, version, "promotion_status", "pending"
    )
    _log_evaluation_and_error_analysis(registered_name, version)
    _patch_calibrate_helpers(monkeypatch, X_dev)

    with pytest.raises(RuntimeError, match="missing per-cycle artifact"):
        register.run_registration_step(version, register_cfg)

    # Abort, not rejection: nothing decided about the candidate's fitness.
    tags = client.get_model_version(registered_name, version).tags
    assert tags["promotion_status"] == "pending"


def test_register_aborts_when_error_analysis_missing(
    register_cfg: DictConfig,
    fitted_model: LogisticRegression,
    X_golden: pd.DataFrame,
    X_dev: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registered_name = str(register_cfg.mlflow.registered_model_name)
    version, _run_id = _register_test_version(
        registered_name, fitted_model, X_golden, promotion_status="pending"
    )
    _log_evaluation_and_error_analysis(
        registered_name, version, include_error_analysis=False
    )
    _patch_calibrate_helpers(monkeypatch, X_dev)

    # No error_analysis_run_id tag (error_analysis.py never ran) and no
    # reports/error_analysis_receipt.json (B1: register.py's only bootstrap
    # channel) — read_error_analysis_receipt raises naming the producing
    # command.
    with pytest.raises(FileNotFoundError, match="models.error_analysis"):
        register.run_registration_step(version, register_cfg)

    client = mlflow.tracking.MlflowClient()
    tags = client.get_model_version(registered_name, version).tags
    assert tags["promotion_status"] == "pending"


# ---------------------------------------------------------------------------
# Environment mismatch: leaves pending, not rejected
# ---------------------------------------------------------------------------


def test_register_environment_mismatch_leaves_pending(
    register_cfg: DictConfig,
    fitted_model: LogisticRegression,
    X_golden: pd.DataFrame,
    X_dev: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registered_name = str(register_cfg.mlflow.registered_model_name)
    version, _run_id = _register_test_version(
        registered_name, fitted_model, X_golden, promotion_status="pending"
    )
    _log_evaluation_and_error_analysis(registered_name, version)
    _patch_calibrate_helpers(monkeypatch, X_dev)

    def _raise_mismatch(*args: Any, **kwargs: Any) -> dict[str, str]:
        raise register.EnvironmentMismatchError(
            "numpy mismatch: logged=2.0.0 installed=2.1.0"
        )

    monkeypatch.setattr(register, "check_environment_parity", _raise_mismatch)

    with pytest.raises(register.EnvironmentMismatchError):
        register.run_registration_step(version, register_cfg)

    client = mlflow.tracking.MlflowClient()
    tags = client.get_model_version(registered_name, version).tags
    # The discriminating assertion: not "rejected".
    assert tags["promotion_status"] == "pending"

    model = client.get_registered_model(registered_name)
    assert "champion" not in model.aliases


# ---------------------------------------------------------------------------
# Smoke-check failures: pre-flip rejects without flipping; post-flip rolls back
# ---------------------------------------------------------------------------


def test_register_pre_flip_golden_parity_failure_rejects_without_flipping(
    register_cfg: DictConfig,
    fitted_model: LogisticRegression,
    X_golden: pd.DataFrame,
    X_dev: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registered_name = str(register_cfg.mlflow.registered_model_name)
    version, _run_id = _register_test_version(
        registered_name,
        fitted_model,
        X_golden,
        promotion_status="pending",
        perturb_golden=True,  # golden_predictions.json disagrees with the model
    )
    _log_evaluation_and_error_analysis(registered_name, version)
    _patch_calibrate_helpers(monkeypatch, X_dev)
    monkeypatch.setattr(register, "check_environment_parity", lambda *a, **k: {})

    with pytest.raises(AssertionError, match="Golden-parity check failed"):
        register.run_registration_step(version, register_cfg)

    client = mlflow.tracking.MlflowClient()
    tags = client.get_model_version(registered_name, version).tags
    assert tags["promotion_status"] == "rejected"

    model = client.get_registered_model(registered_name)
    assert "champion" not in model.aliases


def test_register_post_flip_failure_rolls_back(
    register_cfg: DictConfig,
    fitted_model: LogisticRegression,
    X_golden: pd.DataFrame,
    X_dev: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simulate a post-flip re-load that disagrees with golden — the abort
    path must roll champion back to the prior promoted version, not leave it
    on the just-failed candidate.
    """
    registered_name = str(register_cfg.mlflow.registered_model_name)
    # A legitimate prior champion.
    prior_version, _prior_run_id = _register_test_version(
        registered_name, fitted_model, X_golden, promotion_status="promoted"
    )
    client = mlflow.tracking.MlflowClient()
    register.promote_to_alias(registered_name, prior_version, "champion")

    candidate_version, _run_id = _register_test_version(
        registered_name, fitted_model, X_golden, promotion_status="pending"
    )
    _log_evaluation_and_error_analysis(registered_name, candidate_version)
    _patch_calibrate_helpers(monkeypatch, X_dev)
    monkeypatch.setattr(register, "check_environment_parity", lambda *a, **k: {})

    # Pre-flip parity passes (golden matches candidate); post-flip parity fails
    # because the reloaded-by-alias model diverges from the reference.
    call_count = {"n": 0}
    real_check = register._check_golden_parity

    def _flaky_parity(model: Any, golden: dict[str, Any], atol: float) -> None:
        call_count["n"] += 1
        if call_count["n"] == 1:
            real_check(model, golden, atol)
        else:
            raise AssertionError(
                "Golden-parity check failed: simulated post-flip drift."
            )

    monkeypatch.setattr(register, "_check_golden_parity", _flaky_parity)

    with pytest.raises(AssertionError):
        register.run_registration_step(candidate_version, register_cfg)

    tags = client.get_model_version(registered_name, candidate_version).tags
    assert tags["promotion_status"] == "rejected"

    model = client.get_registered_model(registered_name)
    assert str(model.aliases["champion"]) == prior_version

    history = register.champion_history(registered_name)
    assert history[-1] == {
        "action": "rolled_back",
        "version": prior_version,
        "rolled_back_from": candidate_version,
        "at": history[-1]["at"],
    }


def test_register_post_flip_failure_unsets_alias_on_cold_start(
    register_cfg: DictConfig,
    fitted_model: LogisticRegression,
    X_golden: pd.DataFrame,
    X_dev: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same post-flip parity failure as above, but the first-ever promotion —
    no prior champion exists. The abort path must unset `champion` (not
    raise, and not leave it pointing at the failed candidate) and still tag
    the candidate rejected.
    """
    registered_name = str(register_cfg.mlflow.registered_model_name)
    candidate_version, _run_id = _register_test_version(
        registered_name, fitted_model, X_golden, promotion_status="pending"
    )
    _log_evaluation_and_error_analysis(registered_name, candidate_version)
    _patch_calibrate_helpers(monkeypatch, X_dev)
    monkeypatch.setattr(register, "check_environment_parity", lambda *a, **k: {})

    call_count = {"n": 0}
    real_check = register._check_golden_parity

    def _flaky_parity(model: Any, golden: dict[str, Any], atol: float) -> None:
        call_count["n"] += 1
        if call_count["n"] == 1:
            real_check(model, golden, atol)
        else:
            raise AssertionError(
                "Golden-parity check failed: simulated post-flip drift."
            )

    monkeypatch.setattr(register, "_check_golden_parity", _flaky_parity)

    with pytest.raises(AssertionError):
        register.run_registration_step(candidate_version, register_cfg)

    client = mlflow.tracking.MlflowClient()
    tags = client.get_model_version(registered_name, candidate_version).tags
    assert tags["promotion_status"] == "rejected"

    model = client.get_registered_model(registered_name)
    assert "champion" not in model.aliases

    history = register.champion_history(registered_name)
    assert history[-1] == {
        "action": "rolled_back",
        "version": None,
        "rolled_back_from": candidate_version,
        "at": history[-1]["at"],
    }


# ---------------------------------------------------------------------------
# Golden round trip is genuine, not circular
# ---------------------------------------------------------------------------


def test_register_reads_golden_reference_does_not_regenerate_it(
    register_cfg: DictConfig,
    fitted_model: LogisticRegression,
    X_golden: pd.DataFrame,
    X_dev: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A golden_predictions.json whose scores were perturbed must fail
    registration — proving register.py reads the reference rather than
    reproducing a fresh (and therefore always-agreeing) one.
    """
    registered_name = str(register_cfg.mlflow.registered_model_name)
    version, _run_id = _register_test_version(
        registered_name,
        fitted_model,
        X_golden,
        promotion_status="pending",
        perturb_golden=True,
    )
    _log_evaluation_and_error_analysis(registered_name, version)
    _patch_calibrate_helpers(monkeypatch, X_dev)
    monkeypatch.setattr(register, "check_environment_parity", lambda *a, **k: {})

    with pytest.raises(AssertionError, match="Golden-parity check failed"):
        register.run_registration_step(version, register_cfg)


# ---------------------------------------------------------------------------
# Incumbent staleness (comparative regime)
# ---------------------------------------------------------------------------


def test_register_aborts_on_stale_incumbent(
    register_cfg: DictConfig,
    fitted_model: LogisticRegression,
    X_golden: pd.DataFrame,
    X_dev: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registered_name = str(register_cfg.mlflow.registered_model_name)
    version, _run_id = _register_test_version(
        registered_name, fitted_model, X_golden, promotion_status="pending"
    )
    _log_evaluation_and_error_analysis(
        registered_name, version, regime="comparative", champion_version="7"
    )
    _patch_calibrate_helpers(monkeypatch, X_dev)
    monkeypatch.setattr(register, "check_environment_parity", lambda *a, **k: {})
    # champion has moved to a different version than the decision recorded.
    monkeypatch.setattr(register, "resolve_champion_version", lambda cfg: "8")

    with pytest.raises(RuntimeError, match="champion moved"):
        register.run_registration_step(version, register_cfg)

    client = mlflow.tracking.MlflowClient()
    tags = client.get_model_version(registered_name, version).tags
    assert tags["promotion_status"] == "pending"


# ---------------------------------------------------------------------------
# _build_drift_reference_inputs — dev-OOF union sealed-test, never in-sample
# ---------------------------------------------------------------------------


def test_build_drift_reference_inputs_unions_dev_oof_and_sealed_test(
    register_cfg: DictConfig,
    fitted_model: LogisticRegression,
    X_golden: pd.DataFrame,
) -> None:
    """The drift score baseline must be dev-OOF union sealed-test — never the
    champion scoring its own training data. _register_test_version logs a
    2-row dev_oof (y_true=[0, 1], p_hat=[0.2, 0.8]) at
    calibration/dev_oof_predictions.parquet on `run_id`;
    _log_evaluation_and_error_analysis logs a 2-row test_predictions
    (y_true=[1, 0], p_hat=[0.7, 0.1]) at the root of `eval_run_id`. Assert the
    function concatenates both halves — dev-OOF first, then sealed-test, per
    its own pd.concat call order — with no row dropped and no row doubled.
    """
    registered_name = str(register_cfg.mlflow.registered_model_name)
    version, run_id = _register_test_version(
        registered_name, fitted_model, X_golden, promotion_status="pending"
    )
    eval_run_id, _ = _log_evaluation_and_error_analysis(registered_name, version)

    oos_proba, y_combined = register._build_drift_reference_inputs(
        run_id, eval_run_id, register_cfg
    )

    # 2 dev-OOF rows + 2 sealed-test rows -- neither half dropped, neither doubled.
    assert oos_proba.shape == (4,)
    assert y_combined.shape == (4,)
    np.testing.assert_array_equal(oos_proba, [0.2, 0.8, 0.7, 0.1])
    np.testing.assert_array_equal(y_combined, [0, 1, 1, 0])
    assert oos_proba.dtype == np.float64
    assert y_combined.dtype == np.int_


def test_build_drift_reference_inputs_row_count_equals_sum_of_both_halves(
    register_cfg: DictConfig,
    fitted_model: LogisticRegression,
    X_golden: pd.DataFrame,
) -> None:
    """Regression guard against a future refactor that unions on a join key
    instead of a plain concat — row count must be additive, not deduplicated,
    since dev-OOF and sealed-test are disjoint customer sets by construction."""
    registered_name = str(register_cfg.mlflow.registered_model_name)
    version, run_id = _register_test_version(
        registered_name, fitted_model, X_golden, promotion_status="pending"
    )
    eval_run_id, _ = _log_evaluation_and_error_analysis(registered_name, version)

    dev_oof = register.load_dev_oof_predictions(run_id, register_cfg)
    test_predictions_path = mlflow.artifacts.download_artifacts(
        artifact_uri=f"runs:/{eval_run_id}/test_predictions.parquet"
    )
    test_predictions = pd.read_parquet(test_predictions_path)

    oos_proba, _y_combined = register._build_drift_reference_inputs(
        run_id, eval_run_id, register_cfg
    )

    assert len(oos_proba) == len(dev_oof) + len(test_predictions)


# ---------------------------------------------------------------------------
# _describe_criterion — bar-only, CI-based, and delta-based criterion shapes
# ---------------------------------------------------------------------------


def test_describe_criterion_renders_bar_only_shape() -> None:
    """cold_start-regime criteria (pr_auc, recall) carry value+bar, no ci/delta."""
    text = register._describe_criterion(
        "pr_auc", {"passed": True, "value": 0.42, "bar": 0.30}
    )
    assert text == "PR-AUC 0.420 vs bar 0.300 (ok)"


def test_describe_criterion_renders_bar_only_failure_status() -> None:
    """A failed bar-only criterion renders (FAIL), not (ok)."""
    text = register._describe_criterion(
        "recall", {"passed": False, "value": 0.55, "bar": 0.70}
    )
    assert "(FAIL)" in text
    assert text.startswith("recall 0.550")


def test_describe_criterion_renders_ci_based_shape() -> None:
    """cold_start calibration_slope carries value+ci+band, no bar/delta."""
    text = register._describe_criterion(
        "calibration_slope",
        {"passed": True, "value": 0.95, "ci": [0.85, 1.05], "band": [0.8, 1.2]},
    )
    assert text == "calib slope 0.950 [0.850, 1.050] band [0.80, 1.20] (ok)"


def test_describe_criterion_renders_delta_based_shape_with_materiality_threshold() -> (
    None
):
    """comparative-regime pr_auc carries delta_obs/delta_ci + materiality_threshold."""
    text = register._describe_criterion(
        "pr_auc",
        {
            "passed": True,
            "value": 0.44,
            "bar": 0.30,
            "delta_obs": 0.02,
            "delta_ci": [0.01, 0.03],
            "materiality_threshold": 0.01,
        },
    )
    assert text == "PR-AUC Δ+0.020 [+0.010, +0.030] vs 0.010 (ok)"


def test_describe_criterion_renders_delta_based_shape_with_non_inferiority_margin() -> (
    None
):
    """comparative-regime recall carries delta_obs/delta_ci + non_inferiority_margin
    instead of materiality_threshold — _describe_criterion's threshold lookup must
    fall back to it rather than rendering a missing/None threshold."""
    text = register._describe_criterion(
        "recall",
        {
            "passed": False,
            "value": 0.60,
            "bar": 0.70,
            "delta_obs": -0.05,
            "delta_ci": [-0.08, -0.02],
            "non_inferiority_margin": -0.03,
        },
    )
    assert text == "recall Δ-0.050 [-0.080, -0.020] vs -0.030 (FAIL)"


def test_describe_criterion_unmapped_key_falls_back_to_raw_key() -> None:
    """A criterion key with no _CRITERION_LABELS entry renders as itself, not KeyError."""
    text = register._describe_criterion(
        "some_future_metric", {"passed": True, "value": 1.0, "bar": 0.5}
    )
    assert text.startswith("some_future_metric ")


# ---------------------------------------------------------------------------
# _summarize_promotion_context — model_card.json's promotion-narrative helper
# ---------------------------------------------------------------------------


def test_summarize_promotion_context_cold_start_regime() -> None:
    """cold_start never compares against a champion_version, regardless of what's passed."""
    text = register._summarize_promotion_context(
        {"regime": "cold_start", "criteria": {}}, champion_version=None
    )
    assert "first model promoted" in text
    assert "no existing champion" in text


def test_summarize_promotion_context_comparative_regime_with_delta() -> None:
    """comparative regime narrates the PR-AUC delta and CI against the incumbent version."""
    decision = {
        "regime": "comparative",
        "criteria": {"pr_auc": {"delta_obs": 0.015, "delta_ci": [0.005, 0.025]}},
    }
    text = register._summarize_promotion_context(decision, champion_version="7")
    assert "version 7" in text
    assert "+0.0150" in text
    assert "[+0.0050, +0.0250]" in text


def test_summarize_promotion_context_comparative_regime_missing_delta_figures() -> None:
    """comparative regime with no pr_auc delta_obs/delta_ci logged renders the
    unavailable fallback rather than crashing on a missing key."""
    decision = {"regime": "comparative", "criteria": {}}
    text = register._summarize_promotion_context(decision, champion_version="7")
    assert "unavailable" in text


def test_summarize_promotion_context_unrecognized_regime() -> None:
    """An unrecognized regime string renders its own unavailable fallback, naming the regime."""
    decision = {"regime": "canary", "criteria": {}}
    text = register._summarize_promotion_context(decision, champion_version=None)
    assert "unavailable" in text
    assert "'canary'" in text


# ---------------------------------------------------------------------------
# _summarize_model_drivers — model_card.json's SHAP-driven business_case insight
# ---------------------------------------------------------------------------


def test_summarize_model_drivers_lists_top_k_features_by_rank_order() -> None:
    """Feature names are taken in the SHAP global_importance's own rank order,
    truncated to top_k — never re-sorted or re-ranked here."""
    error_analysis = {
        "shap": {
            "global_importance": [
                {"feature": "contract_type", "importance": 0.30},
                {"feature": "tenure", "importance": 0.22},
                {"feature": "monthlycharges", "importance": 0.18},
                {"feature": "internetservice", "importance": 0.10},
            ]
        }
    }
    text = register._summarize_model_drivers(error_analysis, top_k=3)
    assert "contract_type, tenure, monthlycharges" in text
    assert "internetservice" not in text


def test_summarize_model_drivers_empty_global_importance_renders_fallback() -> None:
    """A missing/empty global_importance renders the unavailable fallback rather
    than an empty driver list."""
    text = register._summarize_model_drivers({"shap": {"global_importance": []}})
    assert "unavailable" in text

    text_missing_key = register._summarize_model_drivers({"shap": {}})
    assert "unavailable" in text_missing_key


# ---------------------------------------------------------------------------
# Full pass: alias flip, drift_reference.json, model_card.json, promoted tag
# ---------------------------------------------------------------------------


def test_register_full_pass_promotes_and_logs_artifacts(
    register_cfg: DictConfig,
    fitted_model: LogisticRegression,
    X_golden: pd.DataFrame,
    X_dev: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registered_name = str(register_cfg.mlflow.registered_model_name)
    version, run_id = _register_test_version(
        registered_name, fitted_model, X_golden, promotion_status="pending"
    )
    _log_evaluation_and_error_analysis(registered_name, version)
    _patch_calibrate_helpers(monkeypatch, X_dev)
    monkeypatch.setattr(register, "check_environment_parity", lambda *a, **k: {})

    result = register.run_registration_step(version, register_cfg)

    assert result["promotion_status"] == "promoted"
    assert "promoted_at" in result

    client = mlflow.tracking.MlflowClient()
    tags = client.get_model_version(registered_name, version).tags
    assert tags["promotion_status"] == "promoted"
    model = client.get_registered_model(registered_name)
    assert str(model.aliases["champion"]) == version

    history = register.champion_history(registered_name)
    assert history == [
        {
            "action": "promoted",
            "version": version,
            "previous_champion_version": None,
            "at": result["promoted_at"],
        }
    ]

    drift_reference = mlflow.artifacts.load_dict(
        f"runs:/{run_id}/registration/drift_reference.json"
    )
    assert "features" in drift_reference
    assert "score" in drift_reference
    assert "prevalence" in drift_reference
    assert (Path(str(register_cfg.paths.reports)) / "drift_reference.json").exists()

    model_card = mlflow.artifacts.load_dict(
        f"runs:/{run_id}/registration/model_card.json"
    )
    assert model_card["governance"]["model_details"]["version"] == version
    assert model_card["governance"]["model_details"]["alias"] == "champion"
    assert (Path(str(register_cfg.paths.reports)) / "model_card.json").exists()

    # Rollback property: re-promote a second version, and the first's
    # drift_reference.json must still be retrievable from its own run.
    second_version, second_run_id = _register_test_version(
        registered_name, fitted_model, X_golden, promotion_status="pending"
    )
    _log_evaluation_and_error_analysis(registered_name, second_version)
    second_result = register.run_registration_step(second_version, register_cfg)
    still_there = mlflow.artifacts.load_dict(
        f"runs:/{run_id}/registration/drift_reference.json"
    )
    assert still_there == drift_reference

    history_after_second = register.champion_history(registered_name)
    assert len(history_after_second) == 2
    assert history_after_second[-1] == {
        "action": "promoted",
        "version": second_version,
        "previous_champion_version": None,
        "at": second_result["promoted_at"],
    }


def test_register_rolls_back_alias_when_post_flip_finalization_fails(
    register_cfg: DictConfig,
    fitted_model: LogisticRegression,
    X_golden: pd.DataFrame,
    X_dev: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure between the alias flip and the promoted tag (drift_reference/
    model_card build, or the tag write itself) must not leave `champion`
    pointing at a version that isn't tagged promoted. _complete_promotion
    rolls the alias back to the prior promoted champion and re-raises,
    leaving promotion_status untouched (pending, not rejected — this isn't a
    gate failure) so a retry after fixing the underlying issue is valid.

    Sets up a real incumbent champion first (rather than a cold start) so the
    retry also exercises _reverify_incumbent's race check: before the fix,
    a crashed first attempt would leave `champion` pointing at the candidate
    with no promoted tag, so a retry would see live_champion == candidate but
    metrics["champion_version"] == the original incumbent, mismatch, and
    raise forever. With the rollback, the retry sees a consistent pair and
    completes.
    """
    registered_name = str(register_cfg.mlflow.registered_model_name)
    _patch_calibrate_helpers(monkeypatch, X_dev)
    monkeypatch.setattr(register, "check_environment_parity", lambda *a, **k: {})

    incumbent_version, _incumbent_run_id = _register_test_version(
        registered_name, fitted_model, X_golden, promotion_status="pending"
    )
    _log_evaluation_and_error_analysis(registered_name, incumbent_version)
    incumbent_result = register.run_registration_step(incumbent_version, register_cfg)
    assert incumbent_result["promotion_status"] == "promoted"

    candidate_version, _candidate_run_id = _register_test_version(
        registered_name, fitted_model, X_golden, promotion_status="pending"
    )
    _log_evaluation_and_error_analysis(
        registered_name, candidate_version, champion_version=incumbent_version
    )

    original_build_model_card = register._build_and_log_model_card

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("simulated MLflow write failure")

    monkeypatch.setattr(register, "_build_and_log_model_card", _boom)

    with pytest.raises(RuntimeError, match="simulated MLflow write failure"):
        register.run_registration_step(candidate_version, register_cfg)

    client = mlflow.tracking.MlflowClient()
    candidate_tags = client.get_model_version(registered_name, candidate_version).tags
    assert candidate_tags["promotion_status"] == "pending"

    model = client.get_registered_model(registered_name)
    assert str(model.aliases["champion"]) == incumbent_version

    history = register.champion_history(registered_name)
    assert history[-1]["action"] == "rolled_back"
    assert history[-1]["version"] == incumbent_version
    assert history[-1]["rolled_back_from"] == candidate_version

    # Retry after the transient failure is fixed: champion/incumbent are
    # consistent again (rolled back to incumbent_version), so this must
    # complete cleanly rather than tripping _reverify_incumbent.
    monkeypatch.setattr(
        register, "_build_and_log_model_card", original_build_model_card
    )

    retry_result = register.run_registration_step(candidate_version, register_cfg)

    assert retry_result["promotion_status"] == "promoted"
    candidate_tags_after_retry = client.get_model_version(
        registered_name, candidate_version
    ).tags
    assert candidate_tags_after_retry["promotion_status"] == "promoted"
    model_after_retry = client.get_registered_model(registered_name)
    assert str(model_after_retry.aliases["champion"]) == candidate_version


# ---------------------------------------------------------------------------
# Emergency rollback: highest promoted, never highest version number
# ---------------------------------------------------------------------------


def test_rollback_champion_selects_highest_promoted_not_highest_version(
    register_cfg: DictConfig,
    fitted_model: LogisticRegression,
    X_golden: pd.DataFrame,
) -> None:
    registered_name = str(register_cfg.mlflow.registered_model_name)
    legit_version, _run_id = _register_test_version(
        registered_name, fitted_model, X_golden, promotion_status="promoted"
    )
    # A later, higher-numbered version that was evaluated and rejected.
    rejected_version, _run_id2 = _register_test_version(
        registered_name, fitted_model, X_golden, promotion_status="rejected"
    )
    assert int(rejected_version) > int(legit_version)

    result = register.rollback_champion(registered_name)

    assert result == legit_version
    client = mlflow.tracking.MlflowClient()
    model = client.get_registered_model(registered_name)
    assert str(model.aliases["champion"]) == legit_version


def test_rollback_champion_unsets_alias_when_nothing_promoted(
    register_cfg: DictConfig,
    fitted_model: LogisticRegression,
    X_golden: pd.DataFrame,
) -> None:
    """No promoted version exists anywhere (a from-scratch registry) — there is
    nothing to restore, so this is a first-class unset, not an error. This is
    also the shape rollback_champion() sees from the automated post-flip
    abort path during a first-ever (cold-start) promotion, where the just-
    flipped candidate isn't tagged promoted yet and no earlier champion ever
    existed.
    """
    registered_name = str(register_cfg.mlflow.registered_model_name)
    _register_test_version(
        registered_name, fitted_model, X_golden, promotion_status="rejected"
    )

    result = register.rollback_champion(registered_name)

    assert result is None
    client = mlflow.tracking.MlflowClient()
    model = client.get_registered_model(registered_name)
    assert "champion" not in model.aliases

    history = register.champion_history(registered_name)
    assert history[-1] == {
        "action": "rolled_back",
        "version": None,
        "rolled_back_from": None,
        "at": history[-1]["at"],
    }


def test_rollback_champion_reraises_non_not_found_mlflow_errors(
    register_cfg: DictConfig,
    fitted_model: LogisticRegression,
    X_golden: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transient/auth/server failure while reading the current champion
    alias must propagate, not be read as 'no champion currently set' —
    silently swallowing it risks rolling back a healthy champion under a
    wrong premise about what is currently serving.
    """
    registered_name = str(register_cfg.mlflow.registered_model_name)
    _register_test_version(
        registered_name, fitted_model, X_golden, promotion_status="promoted"
    )

    def _raise_transient(
        self: mlflow.tracking.MlflowClient, name: str, alias: str
    ) -> None:
        raise MlflowException("temporarily unavailable", error_code=INTERNAL_ERROR)

    monkeypatch.setattr(
        mlflow.tracking.MlflowClient, "get_model_version_by_alias", _raise_transient
    )

    with pytest.raises(MlflowException, match="temporarily unavailable"):
        register.rollback_champion(registered_name)


def test_rollback_champion_skips_current_champion_when_also_promoted(
    register_cfg: DictConfig,
    fitted_model: LogisticRegression,
    X_golden: pd.DataFrame,
) -> None:
    """A no-argument rollback must undo the most recent promotion, not re-select it.

    Once the later champion is itself tagged promoted (the normal end state
    of a successful registration), a bare max() over all promoted versions
    would resolve right back to it — a silent no-op. The current champion
    alias must be excluded from the candidate set.
    """
    registered_name = str(register_cfg.mlflow.registered_model_name)
    earlier_version, _run_id = _register_test_version(
        registered_name, fitted_model, X_golden, promotion_status="promoted"
    )
    later_version, _run_id2 = _register_test_version(
        registered_name, fitted_model, X_golden, promotion_status="promoted"
    )
    assert int(later_version) > int(earlier_version)
    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(registered_name, "champion", later_version)

    result = register.rollback_champion(registered_name)

    assert result == earlier_version
    model = client.get_registered_model(registered_name)
    assert str(model.aliases["champion"]) == earlier_version


def test_rollback_champion_with_explicit_target_version(
    register_cfg: DictConfig,
    fitted_model: LogisticRegression,
    X_golden: pd.DataFrame,
) -> None:
    registered_name = str(register_cfg.mlflow.registered_model_name)
    earlier_version, _run_id = _register_test_version(
        registered_name, fitted_model, X_golden, promotion_status="promoted"
    )
    later_version, _run_id2 = _register_test_version(
        registered_name, fitted_model, X_golden, promotion_status="promoted"
    )
    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(registered_name, "champion", later_version)

    result = register.rollback_champion(registered_name, target_version=earlier_version)

    assert result == earlier_version
    model = client.get_registered_model(registered_name)
    assert str(model.aliases["champion"]) == earlier_version


def test_rollback_champion_rejects_unpromoted_target_version(
    register_cfg: DictConfig,
    fitted_model: LogisticRegression,
    X_golden: pd.DataFrame,
) -> None:
    registered_name = str(register_cfg.mlflow.registered_model_name)
    _register_test_version(
        registered_name, fitted_model, X_golden, promotion_status="promoted"
    )
    rejected_version, _run_id = _register_test_version(
        registered_name, fitted_model, X_golden, promotion_status="rejected"
    )

    with pytest.raises(RuntimeError, match="not tagged promotion_status=promoted"):
        register.rollback_champion(registered_name, target_version=rejected_version)

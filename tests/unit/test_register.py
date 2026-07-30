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

import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mlflow
import mlflow.artifacts
import mlflow.sklearn
import mlflow.tracking
import numpy as np
import pandas as pd
import pytest
from mlflow.models import infer_signature
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import LogisticRegression

import telco_churn.models.register as register
from telco_churn.models.evaluate import content_hash

_COMMITTED_FEATURES = ["tenure", "monthlycharges"]


@pytest.fixture
def register_mlflow_uri(mlflow_test_experiment: Callable[[str], str]) -> str:
    return mlflow_test_experiment("test_register")


@pytest.fixture
def register_cfg(register_mlflow_uri: str, tmp_path: Path) -> DictConfig:
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    return OmegaConf.create(
        {
            "mlflow": {
                "tracking_uri": register_mlflow_uri,
                "experiment_name": "test_register",
                "registered_model_name": "test-telco-churn-pipeline",
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
    review: str = "approved",
    regime: str = "cold_start",
    champion_version: str | None = None,
    include_error_analysis: bool = True,
    error_analysis_model_version: str | None = None,
) -> tuple[str, str | None]:
    """Log a real 'evaluation' run (promotion_decision.json/metrics.json/
    economics.json/test_predictions.parquet) and a real 'error_analysis' run
    (error_analysis.json), then tag `version` with
    eval_run_id/error_analysis_run_id — mirroring what
    evaluate.py/error_analysis.py actually do, since register.py now
    resolves both exclusively from those tags, never a local reports/ path.

    decision_model_version/error_analysis_model_version default to `version`
    but can be overridden to simulate a stale/mismatched artifact (a
    promotion_decision.json or error_analysis.json computed for a different
    model_version than the one being registered).
    include_error_analysis=False skips the 'error_analysis' run and its
    error_analysis_run_id tag entirely, simulating error_analysis.py never
    having run.

    Returns (eval_run_id, error_analysis_run_id | None).
    """
    decision_model_version = decision_model_version or version
    error_analysis_model_version = error_analysis_model_version or version
    client = mlflow.tracking.MlflowClient()

    metrics_payload = {
        "champion_version": champion_version,
        "ranking": {"pr_auc": 0.65, "dummy_pr_auc_floor": 0.265},
        "classification": [],
        "fixed_recall_profile": [],
        "calibration": {"brier": 0.13, "bss": 0.3, "calibration_slope": {"slope": 1.0}},
        "business_impact": {"scenarios": {}},
    }
    promotion_decision_payload = {
        "model_version": decision_model_version,
        "gate": gate,
        "review": review,
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
    _log_evaluation_and_error_analysis(registered_name, version, review="pending")

    with pytest.raises(RuntimeError, match="review"):
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

    with pytest.raises(RuntimeError, match="error_analysis.json"):
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


def test_rollback_champion_raises_when_nothing_promoted(
    register_cfg: DictConfig,
    fitted_model: LogisticRegression,
    X_golden: pd.DataFrame,
) -> None:
    registered_name = str(register_cfg.mlflow.registered_model_name)
    _register_test_version(
        registered_name, fitted_model, X_golden, promotion_status="rejected"
    )

    with pytest.raises(RuntimeError, match="nothing to roll back to"):
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

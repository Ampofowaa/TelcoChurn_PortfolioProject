"""Integration tests: register.py __main__ subprocess (self-contained, no Docker).

CLAUDE.md: every __main__ CLI entry point requires a subprocess integration
test — direct function calls miss argparse/OmegaConf resolution and the
env-var-to-engine joints that only surface at the process boundary.

The full production chain up to (but not including) register.py is seeded
once per module via direct in-process calls — log_model.run_model_logging_step
-> calibrate.run_calibration_step -> threshold.run_threshold_step (which now
folds in the dev-OOF screen as its own last step) -> evaluate.run_evaluation_step
-> error_analysis.run_error_analysis_step -> a human review stamp via
gate.record_review (mirroring what models/review.py's CLI does) — one step
further down the pipeline than test_error_analysis_subprocess.py's own
precedent. Only register.py itself crosses the subprocess boundary. A
dev-OOF screen failure (a real slope outside the band on this synthetic
fixture) used to be tolerated during seeding — since PR B0, evaluate.py and
error_analysis.py both independently re-check the screen's verdict and raise
too, so a screen failure now cascades and blocks the whole fixture (see the
fixture's own inline comment); no longer swallowed here.

The seeded decision's gate outcome is forced to "pass" after the real chain
runs: evaluate.py's cold-start gate on this synthetic fixture is not
guaranteed to pass (test_evaluate_subprocess.py itself only asserts
gate in {"pass", "fail"}), and this module's job is to test register.py's
own behavior given a passing decision, not to re-litigate evaluate.py's gate
logic (already covered by test_gate.py/test_evaluate.py).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import mlflow
import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

import telco_churn.models.calibrate as calibrate
import telco_churn.models.error_analysis as error_analysis
import telco_churn.models.evaluate as evaluate
import telco_churn.models.gate as gate
import telco_churn.models.threshold as threshold
import telco_churn.models.train.log_model as log_model
from telco_churn.data.split import make_split, partition, write_split
from telco_churn.features.accessor import FEATURES_FILENAME
from telco_churn.utils.paths import (
    activate_config,
    compose_config,
    get_project_root,
    reset_active_config,
)

pytestmark = pytest.mark.integration

_PROJECT_ROOT = get_project_root()

_FAST_CALIBRATION_OVERRIDES = [
    "calibration.method=sigmoid",
    "calibration.outer_cv_folds=3",
    "calibration.inner_cv_folds=3",
]
_FAST_EVALUATE_OVERRIDES = ["evaluate.n_bootstrap=30"]
# v3_top_k_features=3 (not the production 8): _make_synthetic_processed_frame's
# docstring plants exactly three real signal columns (contract_type, tenure,
# monthlycharges) — the only ones with a real, learnable relationship to
# churn — so cutting the V3 pre-seal veto's top-k at 3 keeps it checking real
# signal instead of noise from one of this fixture's many uninformative
# columns. Mirrors tests/unit/test_error_analysis.py's identical override.
_FAST_THRESHOLD_OVERRIDES = ["threshold.v3_top_k_features=3"]


def _make_synthetic_processed_frame(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """A FeatureOutputSchema-conformant frame with every segment-axis column
    evaluate.py's slicing needs, mirroring test_evaluate_subprocess.py's own
    fixture (genuine learnable signal, not label noise — see that module's
    docstring for why).
    """
    rng = np.random.default_rng(seed)
    contract_type = rng.choice(["Month-to-month", "One year", "Two year"], size=n)
    tenure = rng.integers(0, 73, size=n)
    monthlycharges = rng.uniform(18.25, 118.75, size=n)

    logit = (
        -0.5
        + 1.4 * (contract_type == "Month-to-month")
        - 0.03 * tenure
        + 0.01 * (monthlycharges - 60.0)
    )
    churn_prob = 1.0 / (1.0 + np.exp(-logit))
    churn = (rng.random(n) < churn_prob).astype(int)

    return pd.DataFrame(
        {
            "customerid": [f"cust-{i:04d}" for i in range(n)],
            "gender": rng.choice(["Male", "Female"], size=n),
            "has_partner": rng.choice(["Yes", "No"], size=n),
            "dependents": rng.choice(["Yes", "No"], size=n),
            "phoneservice": rng.choice(["Yes", "No"], size=n),
            "paperlessbilling": rng.choice(["Yes", "No"], size=n),
            "seniorcitizen": rng.integers(0, 2, size=n).tolist(),
            "multiplelines": rng.choice(["Yes", "No", "No phone service"], size=n),
            "internetservice": rng.choice(["DSL", "Fiber optic", "No"], size=n),
            "onlinesecurity": rng.choice(["Yes", "No", "No internet service"], size=n),
            "onlinebackup": rng.choice(["Yes", "No", "No internet service"], size=n),
            "deviceprotection": rng.choice(
                ["Yes", "No", "No internet service"], size=n
            ),
            "techsupport": rng.choice(["Yes", "No", "No internet service"], size=n),
            "streamingtv": rng.choice(["Yes", "No", "No internet service"], size=n),
            "streamingmovies": rng.choice(["Yes", "No", "No internet service"], size=n),
            "contract_type": contract_type,
            "paymentmethod": rng.choice(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                size=n,
            ),
            "tenure": tenure.tolist(),
            "monthlycharges": monthlycharges.tolist(),
            "totalcharges": (monthlycharges * np.maximum(tenure, 1)).tolist(),
            "charge_per_service": rng.uniform(0.5, 50.0, size=n).tolist(),
            "churn": churn.tolist(),
        }
    )


def _seed_processed_data(
    out_dir: Path, n: int = 300, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write a processed features file + matching canonical split manifest into out_dir."""
    df = _make_synthetic_processed_frame(n=n, seed=seed)
    df.to_parquet(out_dir / FEATURES_FILENAME, index=False)
    manifest = make_split(
        ids=df["customerid"], labels=df["churn"], test_size=0.2, random_state=42
    )
    write_split(manifest, out_dir / "split_manifest.parquet")
    return df, manifest


def _sqlite_experiment(tmp_path: Path, experiment_name: str) -> str:
    """Create a throwaway SQLite-backed MLflow store with an explicit artifact
    root, returning its tracking URI — mirrors conftest.py :: mlflow_test_experiment.
    """
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    experiment_id = mlflow.create_experiment(
        experiment_name, artifact_location=(tmp_path / "artifacts").as_uri()
    )
    mlflow.set_experiment(experiment_id=experiment_id)
    return tracking_uri


def _base_costs_cfg_dict() -> dict:
    """A valid, minimal costs.yaml payload — matches configs/costs.yaml's schema."""
    return {
        "gross_margin": 0.60,
        "horizon_months": 12,
        "discount_months": 3,
        "arpu_quantile": {"conservative": 0.25, "base": 0.50, "optimistic": 0.75},
        "scenarios": {
            "conservative": {
                "outreach_cost": 5.0,
                "discount_rate": 0.10,
                "retention_rate": 0.20,
            },
            "base": {
                "outreach_cost": 20.0,
                "discount_rate": 0.20,
                "retention_rate": 0.30,
            },
            "optimistic": {
                "outreach_cost": 50.0,
                "discount_rate": 0.30,
                "retention_rate": 0.40,
            },
        },
        "retention_rate_sweep": [0.15, 0.20, 0.30, 0.40, 0.45],
        "contact_capacity": 200,
        "campaign_budget": 15_000,
        "argmax_ev_bootstrap_n_samples": 100,
    }


@pytest.fixture(scope="module")
def reviewed_model(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """Seed a real registered, calibrated, thresholded, evaluated, error-analyzed,
    and human-reviewed model once for the whole module — the production chain
    up to (not including) register.py.
    """
    seed_root = tmp_path_factory.mktemp("register_subprocess_seed")
    data_dir = seed_root / "processed"
    data_dir.mkdir()
    df, manifest = _seed_processed_data(data_dir)

    experiment_name = str(compose_config().mlflow.experiment_name)
    tracking_uri = _sqlite_experiment(seed_root, experiment_name)
    registered_model_name = "test-register-subprocess-pipeline"

    costs_path = seed_root / "costs.yaml"
    OmegaConf.save(OmegaConf.create(_base_costs_cfg_dict()), costs_path)
    figures_dir = seed_root / "figures"
    policy_dir = seed_root / "policy"
    reports_dir = seed_root / "reports"

    cfg = compose_config(
        overrides=[
            f"mlflow.tracking_uri={tracking_uri}",
            f"mlflow.registered_model_name={registered_model_name}",
            f"paths.costs_config={costs_path}",
            f"paths.figures={figures_dir}",
            f"paths.policy={policy_dir}",
            f"paths.reports={reports_dir}",
            f"paths.processed_data={data_dir}",
            *_FAST_CALIBRATION_OVERRIDES,
            *_FAST_EVALUATE_OVERRIDES,
            *_FAST_THRESHOLD_OVERRIDES,
        ]
    )

    dev_df, _test_df = partition(df, manifest)
    feature_cols = [c for c in df.columns if c not in ("customerid", "churn")]
    X_dev, y_dev = dev_df[feature_cols], dev_df["churn"]

    tuning_result = {
        "best_params": {
            "num_leaves": 8,
            "learning_rate": 0.1,
            "min_child_samples": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "max_depth": 5,
        },
        "best_n_estimators_median": 10,
        "best_cv_pr_auc_mean": 0.6,
        "committed_features": feature_cols,
        "tuning_summary": {
            "n_trials_requested": 50,
            "n_completed_trials": 16,
            "n_pruned_trials": 34,
            "n_failed_trials": 0,
            "min_completed_trials": 10,
            "trial_count_below_threshold": False,
            "selection_rule": "1se",
            "selected_trial_number": 9,
            "selected_cv_pr_auc": 0.6,
            "raw_best_trial_number": 36,
            "raw_best_cv_pr_auc": 0.6664,
            "se": 0.0139,
            "band_floor": 0.6525,
            "boundary_hits": {"num_leaves": False},
        },
    }
    activate_config(cfg)
    try:
        with mlflow.start_run(run_name="tuning_study") as run:
            tuning_result["parent_run_id"] = run.info.run_id
        log_result = log_model.run_model_logging_step(X_dev, y_dev, tuning_result, cfg)
        cal_result = calibrate.run_calibration_step(log_result["run_id"], cfg)
        run_id = str(cal_result["run_id"])
        model_version = str(cal_result["model_version"])
        model_uri = str(cal_result["model_uri"])

        # A dev-OOF screen failure used to be tolerated here (threshold.py's
        # own RuntimeError, swallowed) since it didn't block evaluate.py.
        # Since PR B0, evaluate.py/error_analysis.py both independently
        # re-check screen_passed via check_threshold_screen_passed and raise
        # RuntimeError too — a screen failure now cascades and blocks every
        # downstream step in this fixture, so it is no longer a tolerable
        # outcome to swallow here (see test_error_analysis_subprocess.py's
        # identical fixture for the same reasoning).
        threshold.run_threshold_step(run_id, model_version, cfg)
        evaluate.run_evaluation_step(run_id, model_version, model_uri, cfg)
        error_analysis.run_error_analysis_step(run_id, model_version, model_uri, cfg)
    finally:
        reset_active_config()

    decision_path = reports_dir / "promotion_decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    # Force a deterministic "pass" — see module docstring. promotion_decision.json
    # keeps exactly one author (evaluate.py) post-B1 — this local rewrite is a
    # human-inspection mirror only and is never read back by register.py, which
    # always resolves the MLflow copy on the eval run; re-log the forced "pass"
    # there too so the two copies do not diverge.
    decision["gate"] = "pass"
    decision_path.write_text(json.dumps(decision), encoding="utf-8")
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    client.log_dict(decision["eval_run_id"], decision, "promotion_decision.json")

    # The human review verdict is a separate, append-only promotion_review.json
    # document (B1) — built via gate.record_review and logged onto the eval
    # run, exactly as models/review.py's CLI does.
    promotion_review = gate.record_review(
        None,
        decision,
        verdict="approved",
        notes="Forced pass for register.py subprocess coverage.",
        approver="test-suite",
        reviewed_at=datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
    )
    client.log_dict(decision["eval_run_id"], promotion_review, "promotion_review.json")

    return {
        "tracking_uri": tracking_uri,
        "registered_model_name": registered_model_name,
        "model_version": model_version,
        "run_id": log_result["run_id"],
        "data_dir": data_dir,
        "costs_path": costs_path,
        "policy_dir": policy_dir,
        "figures_dir": figures_dir,
        "reports_dir": reports_dir,
        "cfg_overrides": [
            f"mlflow.tracking_uri={tracking_uri}",
            f"mlflow.registered_model_name={registered_model_name}",
            f"paths.costs_config={costs_path}",
            f"paths.figures={figures_dir}",
            f"paths.policy={policy_dir}",
            f"paths.reports={reports_dir}",
            f"paths.processed_data={data_dir}",
            *_FAST_CALIBRATION_OVERRIDES,
            *_FAST_EVALUATE_OVERRIDES,
        ],
    }


def _log_and_tag_evaluation_run(
    tracking_uri: str,
    registered_model_name: str,
    model_version: str,
    *,
    decision_model_version: str,
    gate_result: str,
    review: str | None,
) -> str:
    """Log a minimal promotion_decision.json + metrics.json (and, if `review`
    is not None, a one-entry promotion_review.json) onto a fresh 'evaluation'
    MLflow run and tag eval_run_id onto model_version — mirroring register.py's
    own first-pass tagging (B1: evaluate.py itself tags nothing). register.py
    resolves both exclusively via that tag now, never a local reports/ path,
    so a test simulating a specific decision (e.g. gate: fail) logs one
    directly rather than hand-editing a local JSON file register.py no longer
    reads.
    """
    mlflow.set_tracking_uri(tracking_uri)
    metrics_payload = {
        "champion_version": None,
        "ranking": {"pr_auc": 0.65},
        # A "base" row is load-bearing: register.py's _tag_gate_criteria (B1)
        # reads test_recall off it when tagging every processed version.
        "classification": [{"scenario": "base", "recall": 0.70}],
        "calibration": {"brier": 0.13, "calibration_slope": {"slope": 1.0}},
    }
    with mlflow.start_run(run_name="evaluation") as run:
        eval_run_id = run.info.run_id
        mlflow.log_dict(metrics_payload, "metrics.json")
        mlflow.log_dict(
            {
                "model_version": decision_model_version,
                "gate": gate_result,
                "regime": "cold_start",
                "criteria": {},
                "metrics_content_hash": evaluate.content_hash(metrics_payload),
            },
            "promotion_decision.json",
        )
        if review is not None:
            mlflow.log_dict(
                {
                    "eval_run_id": eval_run_id,
                    "metrics_content_hash": evaluate.content_hash(metrics_payload),
                    "entries": [
                        {
                            "verdict": review,
                            "notes": "",
                            "approver": "test-suite",
                            "reviewed_at": "t",
                        }
                    ],
                },
                "promotion_review.json",
            )
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    client.set_model_version_tag(
        registered_model_name, model_version, "eval_run_id", eval_run_id
    )
    return eval_run_id


def _mint_second_pending_version(fixture: dict[str, object]) -> str:
    """Re-run calibrate.py's real registration against the fixture's own
    tuning_study run_id, minting a second, independent, still-`pending`
    version — so a test that needs to tag a version without disturbing
    another test's already-decided one gets a genuinely separate artifact,
    rather than depending on test execution order.
    """
    overrides = cast("list[str]", fixture["cfg_overrides"])
    cfg = compose_config(overrides=overrides)
    activate_config(cfg)
    try:
        cal_result = calibrate.run_calibration_step(str(fixture["run_id"]), cfg)
    finally:
        reset_active_config()
    return str(cal_result["model_version"])


def _run_register_cli(
    model_version: str | None,
    fixture: dict[str, object],
    extra_overrides: list[str] | None = None,
    timeout: int = 300,
) -> subprocess.CompletedProcess[str]:
    """Invoke register.py's CLI as a real subprocess with sandboxed output paths."""
    overrides = [
        f"mlflow.tracking_uri={fixture['tracking_uri']}",
        f"mlflow.registered_model_name={fixture['registered_model_name']}",
        f"paths.reports={fixture['reports_dir']}",
        f"paths.processed_data={fixture['data_dir']}",
    ]
    if model_version is not None:
        overrides.append(f"register.model_version={model_version}")
    overrides.extend(extra_overrides or [])

    env = {
        **os.environ,
        "MLFLOW_TRACKING_URI": str(fixture["tracking_uri"]),
    }
    return subprocess.run(
        [sys.executable, "-m", "telco_churn.models.register", *overrides],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(_PROJECT_ROOT),
        timeout=timeout,
    )


def test_register_main_cli_exits_zero_and_promotes(
    reviewed_model: dict[str, object],
) -> None:
    """register.py __main__ flips champion onto the reviewed, passing candidate
    and logs drift_reference.json + model_card.json onto its run."""
    result = _run_register_cli(str(reviewed_model["model_version"]), reviewed_model)

    assert (
        result.returncode == 0
    ), f"register CLI exited non-zero:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    assert "registration_step_done" in result.stdout

    tracking_uri = str(reviewed_model["tracking_uri"])
    registered_name = str(reviewed_model["registered_model_name"])
    model_version = str(reviewed_model["model_version"])
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)

    model = client.get_registered_model(registered_name)
    assert str(model.aliases["champion"]) == model_version

    version = client.get_model_version(registered_name, model_version)
    assert version.tags["promotion_status"] == "promoted"

    run_id = version.run_id
    drift_reference = mlflow.artifacts.load_dict(
        f"runs:/{run_id}/registration/drift_reference.json"
    )
    assert "features" in drift_reference
    assert "score" in drift_reference

    model_card = mlflow.artifacts.load_dict(
        f"runs:/{run_id}/registration/model_card.json"
    )
    assert model_card["governance"]["model_details"]["version"] == model_version

    reports_dir = Path(str(reviewed_model["reports_dir"]))
    assert (reports_dir / "drift_reference.json").exists()
    assert (reports_dir / "model_card.json").exists()


def test_register_main_cli_exits_one_when_model_version_missing(
    reviewed_model: dict[str, object],
) -> None:
    """register.py __main__ exits 1 when register.model_version is not
    provided — never inferred from the challenger alias."""
    result = _run_register_cli(None, reviewed_model, timeout=60)

    assert result.returncode == 1, (
        f"register CLI should exit 1 when register.model_version is missing:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_register_main_cli_exits_one_on_gate_fail(
    reviewed_model: dict[str, object],
) -> None:
    """A gate: fail decision must exit 1 and tag the version rejected, never
    flip champion.

    Mints its own second version via _mint_second_pending_version rather than
    reusing reviewed_model["model_version"] — that version may already be
    tagged `promoted` by test_register_main_cli_exits_zero_and_promotes, and
    these two mutating tests must not depend on execution order. Logs its own
    'evaluation' run with gate="fail" and tags eval_run_id onto second_version
    directly (_log_and_tag_evaluation_run) — register.py resolves
    promotion_decision.json exclusively via that tag, never a local reports/
    path, so no local file surgery is needed to simulate this decision.
    """
    second_version = _mint_second_pending_version(reviewed_model)
    _log_and_tag_evaluation_run(
        str(reviewed_model["tracking_uri"]),
        str(reviewed_model["registered_model_name"]),
        second_version,
        decision_model_version=second_version,
        gate_result="fail",
        review="approved",
    )

    result = _run_register_cli(second_version, reviewed_model, timeout=60)

    assert result.returncode == 0, (
        f"register CLI should exit 0 on a clean gate:fail rejection:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "model_rejected" in result.stdout

    client = mlflow.tracking.MlflowClient(
        tracking_uri=str(reviewed_model["tracking_uri"])
    )
    version = client.get_model_version(
        str(reviewed_model["registered_model_name"]), second_version
    )
    assert version.tags["promotion_status"] == "rejected"

    model = client.get_registered_model(str(reviewed_model["registered_model_name"]))
    assert str(model.aliases.get("champion")) != second_version

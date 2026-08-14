"""Integration tests: calibrate.py __main__ subprocess (self-contained, no Docker).

CLAUDE.md: every __main__ CLI entry point requires a subprocess integration
test — direct function calls miss argparse/OmegaConf resolution and the
env-var-to-engine joints that only surface at the process boundary. Mirrors
test_train_subprocess.py's pattern: a throwaway SQLite-backed MLflow store,
synthetic processed data seeded via env var, no Docker required.

The tuning_study parent run (what calibrate.py calibrates) is seeded via a
direct in-process call to log_model.run_model_logging_step — the real
production chain's Step 5, not a hand-built substitute — so only calibrate.py
itself crosses the subprocess boundary.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

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


def _make_synthetic_processed_frame(n: int = 150, seed: int = 0) -> pd.DataFrame:
    """A FeatureOutputSchema-conformant frame, large enough to survive every CV split."""
    rng = np.random.default_rng(seed)
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
            "contract_type": rng.choice(
                ["Month-to-month", "One year", "Two year"], size=n
            ),
            "paymentmethod": rng.choice(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                size=n,
            ),
            "tenure": rng.integers(0, 73, size=n).tolist(),
            "monthlycharges": rng.uniform(18.25, 118.75, size=n).tolist(),
            "totalcharges": rng.uniform(18.25, 8684.8, size=n).tolist(),
            "charge_per_service": rng.uniform(0.5, 50.0, size=n).tolist(),
            "churn": rng.integers(0, 2, size=n).tolist(),
        }
    )


def _seed_processed_data(
    out_dir: Path, n: int = 150, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write a processed features file + matching canonical split manifest into out_dir.

    Returns (df, manifest) — the manifest must be passed explicitly to
    partition() later; partition()'s default reads load_split(), which is the
    real project's split_manifest.parquet, not this seeded one.
    """
    df = _make_synthetic_processed_frame(n=n, seed=seed)
    df.to_parquet(out_dir / FEATURES_FILENAME, index=False)
    manifest = make_split(
        ids=df["customerid"], labels=df["churn"], test_size=0.2, random_state=42
    )
    write_split(manifest, out_dir / "split_manifest.parquet")
    return df, manifest


def _seed_tuning_study_run(
    df: pd.DataFrame,
    manifest: pd.DataFrame,
    cfg: object,
    trial_count_below_threshold: bool = False,
) -> str:
    """Seed a real tuning_study run via the production Step 5 chain, returning its run_id."""
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
            "trial_count_below_threshold": trial_count_below_threshold,
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
    with mlflow.start_run(run_name="tuning_study") as run:
        tuning_result["parent_run_id"] = run.info.run_id
    # run_model_logging_step's training-manifest step reads the processed-
    # features path via features_sha256() -> load_config(), not the `cfg`
    # parameter directly — activate_config() is what makes that internal
    # read see paths.processed_data below instead of falling back to the
    # real project's own datasets/processed/, which doesn't exist in CI.
    activate_config(cfg)
    try:
        result = log_model.run_model_logging_step(X_dev, y_dev, tuning_result, cfg)
    finally:
        reset_active_config()
    return str(result["run_id"])


def _sqlite_experiment(tmp_path: Path, experiment_name: str) -> str:
    """Create a throwaway SQLite-backed MLflow store with an explicit artifact
    root, returning its tracking URI — mirrors conftest.py :: mlflow_test_experiment.

    set_experiment(experiment_id=...) after create_experiment is load-bearing:
    without it, a run started directly in this (parent) process — as
    _seed_tuning_study_run does — lands in the Default experiment (id "0")
    instead, and the child subprocess's later mlflow.start_run(run_id=...)
    fails with an experiment-ID mismatch when it resumes that run.
    """
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    experiment_id = mlflow.create_experiment(
        experiment_name, artifact_location=(tmp_path / "artifacts").as_uri()
    )
    mlflow.set_experiment(experiment_id=experiment_id)
    return tracking_uri


def test_calibrate_main_cli_exits_zero_and_logs(tmp_path: Path) -> None:
    """calibrate.py __main__ calibrates and logs the pipeline — and performs
    no registry write of its own (B1: register.py's mint step, run as its
    own separate CLI step afterward, is what registers the challenger)."""
    data_dir = tmp_path / "processed"
    data_dir.mkdir()
    df, manifest = _seed_processed_data(data_dir)
    reports_dir = tmp_path / "reports"

    experiment_name = str(compose_config().mlflow.experiment_name)
    tracking_uri = _sqlite_experiment(tmp_path, experiment_name)
    cfg = compose_config(
        overrides=[
            f"mlflow.tracking_uri={tracking_uri}",
            f"paths.processed_data={data_dir}",
            *_FAST_CALIBRATION_OVERRIDES,
        ]
    )
    run_id = _seed_tuning_study_run(df, manifest, cfg)

    env = {
        **os.environ,
        "MLFLOW_TRACKING_URI": tracking_uri,
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "telco_churn.models.calibrate",
            f"calibration.run_id={run_id}",
            f"paths.processed_data={data_dir}",
            f"paths.reports={reports_dir}",
            *_FAST_CALIBRATION_OVERRIDES,
        ],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(_PROJECT_ROOT),
        timeout=300,
    )

    assert (
        result.returncode == 0
    ), f"calibrate CLI exited non-zero:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    assert "calibration_step_done" in result.stdout

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    assert client.search_registered_models() == []

    run = client.get_run(run_id)
    assert run.data.tags["calibrated_model_id"]
    assert run.data.tags["calibrated_model_uri"]

    receipt = json.loads(
        (reports_dir / "calibrate_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["run_id"] == run_id
    assert receipt["logged_model_id"] == run.data.tags["calibrated_model_id"]
    assert receipt["model_uri"] == run.data.tags["calibrated_model_uri"]
    assert "model_version" not in receipt


def test_calibrate_main_cli_exits_one_when_run_id_missing(tmp_path: Path) -> None:
    """calibrate.py __main__ exits 1 when calibration.run_id is not provided —
    never inferred from 'latest'.
    """
    env = {
        **os.environ,
        "MLFLOW_TRACKING_URI": f"sqlite:///{tmp_path / 'mlflow.db'}",
    }
    result = subprocess.run(
        [sys.executable, "-m", "telco_churn.models.calibrate"],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(_PROJECT_ROOT),
        timeout=60,
    )

    assert result.returncode == 1, (
        f"calibrate CLI should exit 1 when calibration.run_id is missing:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_calibrate_main_cli_exits_one_on_low_trial_count(tmp_path: Path) -> None:
    """calibrate.py __main__ exits 1 when trial_count_below_threshold is true
    and no override is set — a data-quality gate, not a performance comparison.
    """
    data_dir = tmp_path / "processed"
    data_dir.mkdir()
    df, manifest = _seed_processed_data(data_dir)

    experiment_name = str(compose_config().mlflow.experiment_name)
    tracking_uri = _sqlite_experiment(tmp_path, experiment_name)
    cfg = compose_config(
        overrides=[
            f"mlflow.tracking_uri={tracking_uri}",
            f"paths.processed_data={data_dir}",
            *_FAST_CALIBRATION_OVERRIDES,
        ]
    )
    run_id = _seed_tuning_study_run(df, manifest, cfg, trial_count_below_threshold=True)

    env = {
        **os.environ,
        "MLFLOW_TRACKING_URI": tracking_uri,
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "telco_churn.models.calibrate",
            f"calibration.run_id={run_id}",
            f"paths.processed_data={data_dir}",
            *_FAST_CALIBRATION_OVERRIDES,
        ],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(_PROJECT_ROOT),
        timeout=300,
    )

    assert result.returncode == 1, (
        f"calibrate CLI should exit 1 on trial_count_below_threshold:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    assert client.search_registered_models() == []

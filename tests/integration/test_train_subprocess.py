"""Integration tests: train.py __main__ subprocess (self-contained, no Docker)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from telco_churn.data.split import make_split, write_split
from telco_churn.utils.paths import get_project_root

pytestmark = pytest.mark.integration

_PROJECT_ROOT = get_project_root()

# Fast-path Hydra CLI overrides — keep the full Steps 1-5 pipeline (candidates,
# selection, Optuna tuning, registration) to well under a minute instead of the
# ~5-minute real-config run, without touching any production config file.
_FAST_OVERRIDES = [
    "training_setup.cv_folds=2",
    "training_setup.cv_repeats=1",
    "training_setup.bootstrap_n_samples=200",
    "training.candidate.n_estimators=10",
    # num_leaves/min_child_samples are sized for ~5,600 real dev rows; on this
    # ~150-row synthetic fold (~75 rows/fold, further carved for early stopping in
    # Step 4) the real defaults starve every leaf and LightGBM fits a featureless
    # stump ("Forced splits file... maximum feature index in dataset is -1").
    "training.candidate.num_leaves=5",
    "training.candidate.min_child_samples=3",
    "logreg.Cs=2",
    "logreg.cv_folds=2",
    "logreg.max_iter=50",
    "selection.n_repeats=3",
    "selection.bootstrap_n_samples=200",
    "tuning.n_trials=2",
    "tuning.cv_folds=2",
    "tuning.n_estimators_ceiling=20",
    "tuning.early_stopping_rounds=3",
    "tuning.n_startup_trials=1",
    "tuning.search_space.num_leaves.low=2",
    "tuning.search_space.num_leaves.high=10",
    "tuning.search_space.min_child_samples.low=2",
    "tuning.search_space.min_child_samples.high=10",
    "tuning.search_space.max_depth.low=2",
    "tuning.search_space.max_depth.high=4",
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


def _seed_processed_data(out_dir: Path, n: int = 150, seed: int = 0) -> None:
    """Write a processed CSV + matching canonical split manifest into out_dir."""
    df = _make_synthetic_processed_frame(n=n, seed=seed)
    df.to_csv(out_dir / "telco_churn_processed.csv", index=False)
    manifest = make_split(
        ids=df["customerid"], labels=df["churn"], test_size=0.2, random_state=42
    )
    write_split(manifest, out_dir / "split_manifest.parquet")


def test_train_main_cli_exits_zero(tmp_path: Path) -> None:
    """train.py __main__ runs the full Steps 1-5 pipeline end-to-end and exits 0.

    CLAUDE.md: every __main__ entry point requires a subprocess integration test
    covering the full composition path — here that's compose_config() ->
    _load_dev_features() -> run_candidate_step -> run_comparison_step ->
    run_selection_step -> run_tuning_step -> run_registration_step. Fast-path
    Hydra CLI overrides (_FAST_OVERRIDES) keep this well under a minute; MLflow
    points at a throwaway local SQLite store so no Docker server is required and
    the real experiment history is never touched.
    """
    data_dir = tmp_path / "processed"
    data_dir.mkdir()
    _seed_processed_data(data_dir)
    mlflow_db = tmp_path / "mlflow.db"

    env = {
        **os.environ,
        "PROCESSED_DATA_DIR": str(data_dir),
        "MLFLOW_TRACKING_URI": f"sqlite:///{mlflow_db}",
    }
    result = subprocess.run(
        [sys.executable, "-m", "telco_churn.models.train", *_FAST_OVERRIDES],
        env=env,
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
        timeout=300,
    )

    assert (
        result.returncode == 0
    ), f"train CLI exited non-zero:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    assert "registration_step_done" in result.stdout


def test_train_main_cli_exits_one_when_processed_data_missing(tmp_path: Path) -> None:
    """train.py __main__ exits 1 when the processed CSV does not exist yet.

    The realistic failure mode of running `python -m telco_churn.models.train`
    before `make features` / `make split`.
    """
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    mlflow_db = tmp_path / "mlflow.db"

    env = {
        **os.environ,
        "PROCESSED_DATA_DIR": str(empty_dir),
        "MLFLOW_TRACKING_URI": f"sqlite:///{mlflow_db}",
    }
    result = subprocess.run(
        [sys.executable, "-m", "telco_churn.models.train"],
        env=env,
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
        timeout=60,
    )

    assert result.returncode == 1, (
        f"train CLI should exit 1 when the processed CSV is missing:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )

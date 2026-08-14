"""Integration tests: review.py __main__ subprocess (self-contained, no Docker).

CLAUDE.md: every __main__ CLI entry point requires a subprocess integration
test — direct function calls miss argparse/OmegaConf resolution and the
env-var-to-engine joints that only surface at the process boundary. Mirrors
test_calibrate_subprocess.py's pattern: a throwaway SQLite-backed MLflow
store, no Docker required.

The 'evaluation' run (what review.py stamps) is seeded via a direct
in-process mlflow.log_dict call — the real chain's evaluate.py output, not a
hand-built substitute for the artifact shape — so only review.py itself
crosses the subprocess boundary.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import mlflow
import mlflow.artifacts
import mlflow.tracking
import pytest

from telco_churn.utils.hashing import content_hash
from telco_churn.utils.paths import compose_config, get_project_root

pytestmark = pytest.mark.integration

_PROJECT_ROOT = get_project_root()


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


def _seed_evaluation_run() -> str:
    """Seed a real 'evaluation' run with promotion_decision.json, returning its run_id."""
    metrics_payload = {"ranking": {"pr_auc": 0.65}}
    with mlflow.start_run(run_name="evaluation") as run:
        eval_run_id = str(run.info.run_id)
        mlflow.log_dict(
            {
                "regime": "cold_start",
                "gate": "pass",
                "criteria": {},
                "model_version": None,
                "eval_run_id": eval_run_id,
                "metrics_content_hash": content_hash(metrics_payload),
            },
            "promotion_decision.json",
        )
    return eval_run_id


def test_review_main_cli_exits_zero_and_stamps_promotion_review(
    tmp_path: Path,
) -> None:
    """review.py __main__ appends a review entry and logs promotion_review.json."""
    experiment_name = str(compose_config().mlflow.experiment_name)
    tracking_uri = _sqlite_experiment(tmp_path, experiment_name)
    eval_run_id = _seed_evaluation_run()

    env = {**os.environ, "MLFLOW_TRACKING_URI": tracking_uri}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "telco_churn.models.review",
            f"mlflow.tracking_uri={tracking_uri}",
            f"review.eval_run_id={eval_run_id}",
            "review.verdict=approved",
            'review.approver="J. Doe"',
            "review.notes=looks good",
        ],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(_PROJECT_ROOT),
        timeout=60,
    )

    assert (
        result.returncode == 0
    ), f"review CLI exited non-zero:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    assert "review_step_done" in result.stdout

    mlflow.set_tracking_uri(tracking_uri)
    promotion_review = mlflow.artifacts.load_dict(
        f"runs:/{eval_run_id}/promotion_review.json"
    )
    assert len(promotion_review["entries"]) == 1
    assert promotion_review["entries"][0]["verdict"] == "approved"


def test_review_main_cli_exits_one_when_verdict_missing(tmp_path: Path) -> None:
    """review.py __main__ exits 1 when review.verdict is not provided."""
    experiment_name = str(compose_config().mlflow.experiment_name)
    tracking_uri = _sqlite_experiment(tmp_path, experiment_name)
    eval_run_id = _seed_evaluation_run()

    env = {**os.environ, "MLFLOW_TRACKING_URI": tracking_uri}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "telco_churn.models.review",
            f"mlflow.tracking_uri={tracking_uri}",
            f"review.eval_run_id={eval_run_id}",
            'review.approver="J. Doe"',
        ],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(_PROJECT_ROOT),
        timeout=60,
    )

    assert result.returncode == 1, (
        f"review CLI should exit 1 when review.verdict is missing:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_review_main_cli_exits_one_when_notes_blank(tmp_path: Path) -> None:
    """review.py __main__ exits 1 when review.notes is blank — an empty
    audit-trail entry is refused the same as a missing verdict/approver."""
    experiment_name = str(compose_config().mlflow.experiment_name)
    tracking_uri = _sqlite_experiment(tmp_path, experiment_name)
    eval_run_id = _seed_evaluation_run()

    env = {**os.environ, "MLFLOW_TRACKING_URI": tracking_uri}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "telco_churn.models.review",
            f"mlflow.tracking_uri={tracking_uri}",
            f"review.eval_run_id={eval_run_id}",
            "review.verdict=approved",
            'review.approver="J. Doe"',
            'review.notes="   "',
        ],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(_PROJECT_ROOT),
        timeout=60,
    )

    assert result.returncode == 1, (
        f"review CLI should exit 1 when review.notes is blank:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )

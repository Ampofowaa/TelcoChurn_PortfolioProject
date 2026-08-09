"""Unit tests for telco_churn.models.review — the human promotion-review CLI (PR B1).

Each test logs a real 'evaluation' run (promotion_decision.json) against a
tmp-scoped MLflow experiment, then calls run_review_step directly — the
__main__ CLI's argument resolution (eval_run_id/verdict/approver overrides,
receipt fallback) is covered by tests/integration/test_review_subprocess.py,
per CLAUDE.md's rule that only the subprocess boundary exercises argparse/
OmegaConf/dotenv wiring.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow
import mlflow.artifacts
import mlflow.tracking
import pytest
from omegaconf import DictConfig, OmegaConf

from telco_churn.models.evaluate import content_hash
from telco_churn.models.review import run_review_step


@pytest.fixture
def review_mlflow_uri(tmp_path_factory: pytest.TempPathFactory) -> str:
    tmp_path = tmp_path_factory.mktemp("review_mlflow")
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    artifact_location = (tmp_path / "artifacts").as_uri()
    experiment_id = mlflow.create_experiment(
        "test_review", artifact_location=artifact_location
    )
    mlflow.set_experiment(experiment_id=experiment_id)
    return tracking_uri


@pytest.fixture
def review_cfg(review_mlflow_uri: str, tmp_path: Path) -> DictConfig:
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    return OmegaConf.create(
        {
            "mlflow": {"tracking_uri": review_mlflow_uri},
            "paths": {"reports": str(reports_dir)},
        }
    )


def _log_evaluation_run() -> str:
    """Log a minimal real 'evaluation' run with promotion_decision.json, return its run_id."""
    metrics_payload = {"ranking": {"pr_auc": 0.65}}
    with mlflow.start_run(run_name="evaluation") as run:
        eval_run_id = str(run.info.run_id)
        decision_payload: dict[str, Any] = {
            "regime": "cold_start",
            "gate": "pass",
            "criteria": {},
            "model_version": None,
            "eval_run_id": eval_run_id,
            "metrics_content_hash": content_hash(metrics_payload),
        }
        mlflow.log_dict(decision_payload, "promotion_decision.json")
        return eval_run_id


def test_run_review_step_first_call_creates_document_with_one_entry(
    review_cfg: DictConfig,
) -> None:
    eval_run_id = _log_evaluation_run()

    result = run_review_step(
        eval_run_id,
        "approved",
        "looks good",
        "r.frimpong",
        "2026-08-09T00:00:00Z",
        review_cfg,
    )

    assert result["eval_run_id"] == eval_run_id
    assert len(result["entries"]) == 1
    entry = result["entries"][0]
    assert entry["verdict"] == "approved"
    assert entry["approver"] == "r.frimpong"
    assert entry["notes"] == "looks good"
    assert entry["reviewed_at"] == "2026-08-09T00:00:00Z"


def test_run_review_step_logs_promotion_review_onto_the_eval_run(
    review_cfg: DictConfig,
) -> None:
    eval_run_id = _log_evaluation_run()

    run_review_step(eval_run_id, "approved", "", "a", "t", review_cfg)

    logged = mlflow.artifacts.load_dict(f"runs:/{eval_run_id}/promotion_review.json")
    assert len(logged["entries"]) == 1
    assert logged["entries"][0]["verdict"] == "approved"


def test_run_review_step_mirrors_to_reports_dir(review_cfg: DictConfig) -> None:
    eval_run_id = _log_evaluation_run()

    run_review_step(eval_run_id, "approved", "", "a", "t", review_cfg)

    reports_dir = Path(str(review_cfg.paths.reports))
    mirrored = json.loads((reports_dir / "promotion_review.json").read_text())
    assert mirrored["eval_run_id"] == eval_run_id
    assert len(mirrored["entries"]) == 1


def test_run_review_step_second_call_appends_rather_than_overwrites(
    review_cfg: DictConfig,
) -> None:
    """A second review — e.g. an initial rejection later followed up with an
    approval — appends to the existing entries list rather than clobbering
    it, so the full review history for this cycle survives."""
    eval_run_id = _log_evaluation_run()
    run_review_step(eval_run_id, "rejected", "needs fix", "a", "t1", review_cfg)

    result = run_review_step(eval_run_id, "approved", "fixed", "b", "t2", review_cfg)

    assert len(result["entries"]) == 2
    assert result["entries"][0]["verdict"] == "rejected"
    assert result["entries"][1]["verdict"] == "approved"
    logged = mlflow.artifacts.load_dict(f"runs:/{eval_run_id}/promotion_review.json")
    assert len(logged["entries"]) == 2

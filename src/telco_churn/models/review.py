"""Human promotion-review CLI — stamps a verdict onto promotion_review.json, independent of Jupyter.

Wraps gate.py::record_review. Resolves the eval run to review from an
explicit override or evaluate.py's reports/eval_receipt.json by default (the
same first-invocation bootstrap role calibrate_receipt.json already plays
for threshold.py/evaluate.py/error_analysis.py — see utils/mlflow.py's
module docstring), fetches promotion_review.json's current state from that
run (if any), appends one entry, re-logs the document onto the eval run, and
mirrors it to reports/ for human inspection only — register.py always
resolves the MLflow copy, never this mirror.

Never touches the registry: a full manual cycle pre-Phase-10 is three
commands, not one — `dvc repro` (or the equivalent module-by-module run)
produces promotion_decision.json/error_analysis.json, this CLI stamps the
verdict, and register.py's own CLI reads the stamped verdict and mints/
flips/rejects. Phase 10's Prefect resume task collapses the last two into
one task by calling run_review_step and then register.py's mint/flip logic
in turn, with no change to either script's internals — only to what
triggers them.
"""

from __future__ import annotations

import json
from typing import Any, Literal

import mlflow
import mlflow.artifacts
import mlflow.tracking
from omegaconf import DictConfig

from telco_churn.models.gate import record_review
from telco_churn.utils.logging import get_logger
from telco_churn.utils.mlflow import read_eval_receipt, resolve_tracking_uri
from telco_churn.utils.paths import get_project_root

__all__ = ["run_review_step"]

logger = get_logger(__name__)


def run_review_step(
    eval_run_id: str,
    verdict: Literal["approved", "rejected"],
    notes: str,
    approver: str,
    reviewed_at: str,
    cfg: DictConfig,
) -> dict[str, Any]:
    """Append one review entry to `eval_run_id`'s promotion_review.json and re-log it.

    Reads promotion_decision.json on the same run for `eval_run_id`/
    `metrics_content_hash` — record_review binds the review document to
    those, never to `model_version`, which does not exist yet at review
    time (register.py mints it only after review, on a pass).
    """
    mlflow.set_tracking_uri(resolve_tracking_uri(str(cfg.mlflow.tracking_uri)))
    client = mlflow.tracking.MlflowClient()

    decision = mlflow.artifacts.load_dict(
        f"runs:/{eval_run_id}/promotion_decision.json"
    )
    eval_run_artifacts = {a.path for a in client.list_artifacts(eval_run_id)}
    existing_promotion_review = (
        mlflow.artifacts.load_dict(f"runs:/{eval_run_id}/promotion_review.json")
        if "promotion_review.json" in eval_run_artifacts
        else None
    )

    promotion_review = record_review(
        existing_promotion_review, decision, verdict, notes, approver, reviewed_at
    )

    client.log_dict(eval_run_id, promotion_review, "promotion_review.json")

    reports_dir = get_project_root() / str(cfg.paths.reports)
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "promotion_review.json", "w", encoding="utf-8") as f:
        json.dump(promotion_review, f, indent=2, default=str)

    logger.info(
        "review_recorded",
        eval_run_id=eval_run_id,
        verdict=verdict,
        approver=approver,
        entry_count=len(promotion_review["entries"]),
    )

    return promotion_review


if __name__ == "__main__":
    import sys
    from datetime import UTC, datetime

    from dotenv import load_dotenv

    from telco_churn.utils.logging import configure_logging
    from telco_churn.utils.paths import activate_config, compose_config

    load_dotenv()
    configure_logging()

    def _resolve_eval_run_id(cfg: DictConfig) -> str:
        """Return review.eval_run_id if given, else evaluate.py's receipt's eval_run_id."""
        if cfg.review.eval_run_id is not None:
            return str(cfg.review.eval_run_id)
        return str(read_eval_receipt(cfg)["eval_run_id"])

    def _print_review_guidance(eval_run_id: str) -> None:
        """Log where to find this cycle's diagnostics and what inputs are required.

        Points at the MLflow run, not notebooks/05-evaluation-and-error-
        analysis.ipynb and not the local reports/ mirror — neither is a
        reliable pointer for a reviewer who isn't the same person, on the
        same machine, that ran evaluate.py/error_analysis.py: the notebook
        only reflects this cycle's numbers if someone happened to
        re-execute it against this exact eval_run_id, and reports/ is
        gitignored, per-machine scratch (.gitignore:229), not something a
        different reviewer can reach. The MLflow run is the one artifact
        guaranteed reachable by anyone pointed at the shared tracking URI —
        the server proxies artifact reads through its own API regardless of
        which machine produced them (docker-compose.yml's
        --artifacts-destination). Logged unconditionally, before the
        required-field checks below, so a first-time user sees it even on a
        run that fails validation.
        """
        logger.info(
            "review_guidance",
            eval_run_id=eval_run_id,
            where_to_look=(
                "Before deciding, inspect this cycle's evaluation run in "
                f"MLflow (`mlflow ui`, run_id={eval_run_id}) — metrics.json, "
                "economics.json, promotion_decision.json, "
                "error_analysis.json, and figures/ are all logged there."
            ),
            required_inputs=(
                "review.verdict — 'approved' or 'rejected' (required); "
                'review.approver — your name, e.g. "J. Doe" (required); '
                "review.notes — a one-line reason for the verdict (required, "
                "becomes part of the permanent audit trail)."
            ),
        )

    def _require_verdict(cfg: DictConfig) -> Literal["approved", "rejected"]:
        """Return review.verdict, raising if it was not supplied or is invalid."""
        verdict = cfg.review.verdict
        if verdict not in ("approved", "rejected"):
            raise ValueError(
                "review.verdict is required and must be 'approved' or "
                "'rejected', e.g. `python -m telco_churn.models.review "
                'review.verdict=approved review.approver="J. Doe"`'
            )
        return "approved" if verdict == "approved" else "rejected"

    def _require_approver(cfg: DictConfig) -> str:
        """Return review.approver, raising if the CLI override was not supplied."""
        if cfg.review.approver is None:
            raise ValueError(
                'review.approver is required, e.g. `review.approver="J. Doe"`'
            )
        return str(cfg.review.approver)

    def _require_notes(cfg: DictConfig) -> str:
        """Return review.notes, raising if it was not supplied or is blank.

        Notes become part of the permanent, append-only audit trail
        (CLAUDE.md § MLflow Model Registry) — an empty note defeats that
        purpose as surely as a missing one, so both are refused alike.
        """
        notes = cfg.review.notes
        if notes is None or not str(notes).strip():
            raise ValueError(
                "review.notes is required and must be non-empty — it becomes "
                'part of the permanent audit trail, e.g. `review.notes="PR-AUC '
                "and recall both clear the bar; no fairness or calibration "
                'concerns."`'
            )
        return str(notes)

    try:
        cfg = compose_config(overrides=sys.argv[1:] or None)
        activate_config(cfg)
        cli_eval_run_id = _resolve_eval_run_id(cfg)
        _print_review_guidance(cli_eval_run_id)
        cli_verdict = _require_verdict(cfg)
        cli_approver = _require_approver(cfg)
        cli_notes = _require_notes(cfg)
        result = run_review_step(
            cli_eval_run_id,
            cli_verdict,
            cli_notes,
            cli_approver,
            datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
            cfg,
        )
        logger.info(
            "review_step_done",
            eval_run_id=cli_eval_run_id,
            verdict=result["entries"][-1]["verdict"],
        )
    except FileNotFoundError as e:
        logger.error("review_data_not_found", error=str(e), exc_info=True)
        sys.exit(1)
    except ValueError as e:
        logger.error("review_invalid", error=str(e), exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error("review_failed", error=str(e), exc_info=True)
        sys.exit(1)

"""Manual outcome-recording CLI — the write path for `prediction_outcomes`.

Designed and built now even though nothing on this project's static dataset
will ever call it in production — the same "real mechanism, no live
traffic" discipline `PROJECT_PLAN.md`'s Phase 10a-i section applies to
`batch_predict.py`'s shadow scoring and the monthly gate's demonstrative
scoring surface. In a real deployment this is what a CRM webhook or a
nightly reconciliation job would invoke; here it is called manually, for
Phase 10b's verification step (seeding `prediction_outcomes` with a matured
cohort of known labels) and never scheduled or wired into Prefect, because
nothing real exists on this dataset to trigger it.

Unlike `serving/prediction_log.py`'s fail-open `BackgroundTasks` write, this
has no live request path to protect — it raises normally on failure, the
standard boundary-error-handling rule (`CLAUDE.md`), since nothing about a
label-recording CLI needs the fail-open discipline built to keep a Postgres
hiccup from touching a live `/predict` response.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from omegaconf import DictConfig
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine

from telco_churn.data.tables import prediction_outcomes

__all__ = ["build_outcome_row", "write_outcomes"]

OutcomeSource = Literal["crm_sync", "manual", "synthetic_seed"]


def build_outcome_row(
    customerid: str,
    churned: bool,
    observed_at: datetime,
    source: OutcomeSource,
) -> dict[str, Any]:
    """One `prediction_outcomes` row.

    `outcome_id`/`recorded_at` are server-generated (Identity PK,
    `server_default=func.now()` — see `data/tables.py::prediction_outcomes`)
    and are never set here.
    """
    return {
        "customerid": customerid,
        "churned": churned,
        "observed_at": observed_at,
        "source": source,
    }


def write_outcomes(rows: list[dict[str, Any]], engine: Engine) -> int:
    """Insert already-built `prediction_outcomes` rows,
    `ON CONFLICT (customerid, observed_at, source) DO NOTHING`.

    The unique constraint on that triple (`data/tables.py`'s
    `uq_prediction_outcomes_customer_observed_source`) is what makes a
    re-run of this module's CLI safe rather than a silent duplicate — this
    matters specifically because the CLI has no live trigger to guard
    against accidental re-invocation (a typo'd argument, a re-run after an
    ambiguous exit) the way `prediction_log.py`'s request-scoped write does;
    there is no upstream caller in a position to notice and dedupe.

    Raises normally on failure — no fail-open discipline here, unlike
    `prediction_log.py`'s `BackgroundTasks` write: there is no live request
    path this write could disrupt.
    """
    if not rows:
        return 0
    stmt = pg_insert(prediction_outcomes).values(rows)
    stmt = stmt.on_conflict_do_nothing(
        index_elements=["customerid", "observed_at", "source"]
    )
    with engine.begin() as conn:
        result = conn.execute(stmt)
    return int(result.rowcount)


if __name__ == "__main__":
    import sys

    from dotenv import load_dotenv

    from telco_churn.utils.db import apply_migrations, get_engine
    from telco_churn.utils.logging import configure_logging, get_logger
    from telco_churn.utils.paths import activate_config, compose_config

    load_dotenv()
    configure_logging()
    logger = get_logger(__name__)

    def _require_customerid(cfg: DictConfig) -> str:
        if cfg.outcomes.customerid is None:
            raise ValueError(
                "outcomes.customerid is required, e.g. `python -m "
                "telco_churn.serving.outcomes outcomes.customerid=serve-0001 "
                "outcomes.churned=true "
                "outcomes.observed_at=2026-08-22T00:00:00+00:00`"
            )
        return str(cfg.outcomes.customerid)

    def _require_churned(cfg: DictConfig) -> bool:
        if cfg.outcomes.churned is None:
            raise ValueError(
                "outcomes.churned is required ('true' or 'false'), e.g. "
                "`outcomes.churned=true`"
            )
        return bool(cfg.outcomes.churned)

    def _require_observed_at(cfg: DictConfig) -> datetime:
        if cfg.outcomes.observed_at is None:
            raise ValueError(
                "outcomes.observed_at is required (ISO 8601), e.g. "
                "`outcomes.observed_at=2026-08-22T00:00:00+00:00`"
            )
        return datetime.fromisoformat(str(cfg.outcomes.observed_at))

    def _require_source(cfg: DictConfig) -> OutcomeSource:
        source = str(cfg.outcomes.source)
        if source == "crm_sync":
            return "crm_sync"
        if source == "manual":
            return "manual"
        if source == "synthetic_seed":
            return "synthetic_seed"
        raise ValueError(
            "outcomes.source must be one of 'crm_sync', 'manual', "
            f"'synthetic_seed' — got {source!r}"
        )

    try:
        cli_cfg = compose_config(overrides=sys.argv[1:] or None)
        activate_config(cli_cfg)
        apply_migrations()
        cli_customerid = _require_customerid(cli_cfg)
        cli_churned = _require_churned(cli_cfg)
        cli_observed_at = _require_observed_at(cli_cfg)
        cli_source = _require_source(cli_cfg)
        cli_row = build_outcome_row(
            cli_customerid, cli_churned, cli_observed_at, cli_source
        )
        cli_written = write_outcomes([cli_row], get_engine())
        logger.info(
            "outcome_recorded",
            customerid=cli_customerid,
            churned=cli_churned,
            observed_at=cli_observed_at.isoformat(),
            source=cli_source,
            written=cli_written,
        )
    except ValueError as e:
        logger.error("outcome_invalid", error=str(e), exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error("outcome_recording_failed", error=str(e), exc_info=True)
        sys.exit(1)

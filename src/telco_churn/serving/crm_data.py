"""Seeded generator for customers_crm — a simulated "current state" table
derived from customers_raw, never touching the read-only raw table itself.

customer_lookup.py::lookup_customers reads customers_crm, not customers_raw,
so GET /customer/{customerid} and /predict/batch's ID-resolution path have a
source genuinely distinct from the frozen training snapshot — see
prediction_logging_plan.md Part A for the full rationale.

Pure generation (generate_crm_rows) is separated from I/O (load_crm), the
same split drift_reference.py's pure builder / register.py's I/O already
uses. Schema creation is Alembic's job (utils/db.py::apply_migrations), not
this module's — load_crm expects customers_crm to already exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from telco_churn.serving.customer_lookup import LOOKUP_COLUMNS
from telco_churn.utils.logging import get_logger

__all__ = [
    "CrmGenerationParams",
    "generate_crm_rows",
    "load_crm",
]

logger = get_logger(__name__)

_RAW_TABLE = "customers_raw"
_CRM_TABLE = "customers_crm"

# Real customers don't downgrade a contract term without churning outright —
# only the plausible upgrade direction is ever rolled.
_CONTRACT_UPGRADE: dict[str, str] = {
    "Month-to-month": "One year",
    "One year": "Two year",
}


@dataclass(frozen=True)
class CrmGenerationParams:
    """Seeded nudge knobs, resolved from configs/serving/api.yaml's crm: block."""

    random_state: int
    tenure_advance_min_months: int
    tenure_advance_max_months: int
    contract_upgrade_probability: float
    totalcharges_noise_scale: float


def generate_crm_rows(df: pd.DataFrame, params: CrmGenerationParams) -> pd.DataFrame:
    """Light, plausible "current state" nudges per customer — never a
    different customer.

    Deterministic under a fixed random_state: the same input frame and the
    same params always produce the same nudges, so reloading is a safe
    truncate-and-reload rather than needing upsert logic. Only
    crm_snapshot_at genuinely changes between runs, since it stamps
    wall-clock generation time rather than being part of the seeded draw.

    tenure advances by a seeded uniform +1..+6 months; contract upgrades
    (never downgrades) with contract_upgrade_probability; totalcharges
    carries the old balance forward, adds tenure_delta x monthlycharges, and
    applies a small seeded noise factor. Everything else — service flags,
    internetservice, demographics — is held fixed, since most account
    attributes don't change month to month.
    """
    rng = np.random.default_rng(params.random_state)
    n = len(df)
    out = df[["customerid", *LOOKUP_COLUMNS]].copy()

    tenure = pd.to_numeric(out["tenure"]).to_numpy(dtype=np.int64)
    monthlycharges = pd.to_numeric(out["monthlycharges"]).to_numpy(dtype=np.float64)
    # NaN for the 11 zero-tenure customers (no first bill yet) — treated as 0
    # rather than dropped, since the nudge below gives them their first real
    # tenure and thus their first real accrued balance.
    old_totalcharges = (
        pd.to_numeric(out["totalcharges"]).fillna(0.0).to_numpy(dtype=np.float64)
    )

    tenure_delta = rng.integers(
        params.tenure_advance_min_months,
        params.tenure_advance_max_months + 1,
        size=n,
    )
    out["tenure"] = tenure + tenure_delta

    upgrade_roll = rng.random(n) < params.contract_upgrade_probability
    upgrade_target = out["contract_type"].map(_CONTRACT_UPGRADE)
    out["contract_type"] = np.where(
        upgrade_roll & upgrade_target.notna(), upgrade_target, out["contract_type"]
    )

    noise = rng.normal(loc=1.0, scale=params.totalcharges_noise_scale, size=n)
    accrued = tenure_delta * monthlycharges
    out["totalcharges"] = np.round((old_totalcharges + accrued) * noise, 2)

    out["crm_snapshot_at"] = datetime.now(UTC)
    return out


def load_crm(df: pd.DataFrame, engine: Engine) -> int:
    """Truncate-and-reload customers_crm from a generated DataFrame.

    Safe as a full replace rather than an upsert: generate_crm_rows is
    deterministic under a fixed seed, so a re-run yields the same nudges
    every time — the same idempotent, safe-to-re-run property
    data/ingest.py already has for customers_raw.

    Does not create customers_crm itself — the table is expected to already
    exist (utils/db.py::apply_migrations, run once at CLI start, or a test
    fixture's own equivalent call), the same convention data/ingest.py::ingest
    now follows for customers_raw.
    """
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE TABLE {_CRM_TABLE}"))
        df.to_sql(
            _CRM_TABLE,
            conn,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=1000,
        )
    return len(df)


if __name__ == "__main__":
    import sys

    from dotenv import load_dotenv

    from telco_churn.utils.db import apply_migrations, get_engine
    from telco_churn.utils.logging import configure_logging
    from telco_churn.utils.paths import activate_config, compose_config

    load_dotenv()
    configure_logging()

    cfg = compose_config(overrides=sys.argv[1:] or None)
    activate_config(cfg)

    try:
        # Belt-and-braces schema creation for a Postgres instance that never saw
        # docker-compose.yml's docker-entrypoint-initdb.d mount — CI's
        # testcontainers, Phase 12's RDS. A no-op once already at head.
        apply_migrations()
        engine = get_engine()
        raw = pd.read_sql_table(_RAW_TABLE, engine)
        params = CrmGenerationParams(
            random_state=int(cfg.serving.crm.random_state),
            tenure_advance_min_months=int(cfg.serving.crm.tenure_advance_min_months),
            tenure_advance_max_months=int(cfg.serving.crm.tenure_advance_max_months),
            contract_upgrade_probability=float(
                cfg.serving.crm.contract_upgrade_probability
            ),
            totalcharges_noise_scale=float(cfg.serving.crm.totalcharges_noise_scale),
        )
        crm_rows = generate_crm_rows(raw, params)
        n = load_crm(crm_rows, engine)
        logger.info("customers_crm_loaded", rows_loaded=n)
    except Exception as e:
        logger.error("crm_data_failed", error=str(e), exc_info=True)
        sys.exit(1)

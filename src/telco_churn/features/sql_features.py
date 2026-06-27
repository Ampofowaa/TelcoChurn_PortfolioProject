"""Builds SQL feature views in Postgres from the sql/features/ directory."""

from __future__ import annotations

import time
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.engine import Engine

from telco_churn.utils.logging import get_logger

logger = get_logger(__name__)

# Execution order matters: customer_features joins the preceding view.
_SQL_FILES = [
    "charge_per_service.sql",
    "customer_features.sql",
]


def build_sql_features(engine: Engine, sql_dir: Path) -> None:
    """Create or replace the two feature views in dependency order.

    Runs charge_per_service → customer_features via SQLAlchemy.
    Each file uses CREATE OR REPLACE VIEW, so the operation is idempotent.
    sql_dir must be supplied by the caller — resolve from configs/config.yaml
    (cfg.paths.sql_features) so the location is environment-overridable.
    """
    with engine.begin() as conn:
        for filename in _SQL_FILES:
            sql_path = sql_dir / filename
            t0 = time.perf_counter()
            try:
                conn.execute(text(sql_path.read_text(encoding="utf-8")))
            except Exception as exc:
                raise RuntimeError(f"Failed executing {filename}") from exc
            duration_ms = round((time.perf_counter() - t0) * 1000)
            logger.info("feature view created", file=filename, duration_ms=duration_ms)


if __name__ == "__main__":
    import sys

    from dotenv import load_dotenv

    from telco_churn.utils.db import get_engine
    from telco_churn.utils.logging import configure_logging
    from telco_churn.utils.paths import load_config

    load_dotenv()
    configure_logging()

    cfg = load_config()

    try:
        build_sql_features(get_engine(), sql_dir=Path(cfg.paths.sql_features))
    except Exception as e:
        logger.error("sql_features failed", error=str(e), exc_info=True)
        sys.exit(1)

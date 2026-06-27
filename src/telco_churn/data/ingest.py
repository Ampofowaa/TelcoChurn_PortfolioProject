"""Idempotent CSV-to-Postgres loader for the Telco Churn raw dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from telco_churn.data.schema import RawSchema
from telco_churn.data.validate import ValidationError, validate_raw
from telco_churn.utils.db import get_engine
from telco_churn.utils.logging import get_logger
from telco_churn.utils.paths import get_project_root

__all__ = [
    "load_raw_csv",
    "setup_schema",
    "ingest",
]

logger = get_logger(__name__)

# Path to the authoritative DDL — column types and PRIMARY KEY live here only.
_SQL_SCHEMA = get_project_root() / "sql" / "schema" / "001_create_raw.sql"

# Derived from RawSchema via the public Pandera API so inheritance and metaclass
# processing are respected — frozenset(__annotations__) misses inherited fields.
_REQUIRED_COLUMNS: frozenset[str] = frozenset(RawSchema.to_schema().columns.keys())


def load_raw_csv(path: Path) -> pd.DataFrame:
    """Load and type-coerce the raw Telco CSV.

    Coerces TotalCharges to numeric (whitespace in source → NaN for the 11
    zero-tenure customers) and encodes Churn Yes/No as a binary integer
    column named 'churn'.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Raw CSV not found at {path}. Run `make data` to download it."
        )
    df = pd.read_csv(path, encoding="utf-8")
    df.columns = df.columns.str.lower()
    df = df.rename(columns={"partner": "has_partner", "contract": "contract_type"})
    actual = set(df.columns)
    missing = _REQUIRED_COLUMNS - actual
    extra = actual - _REQUIRED_COLUMNS
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing: {sorted(missing)}")
        if extra:
            parts.append(f"unexpected: {sorted(extra)}")
        raise ValueError(f"CSV column mismatch — {'; '.join(parts)}")
    df["totalcharges"] = pd.to_numeric(df["totalcharges"], errors="coerce")
    df["churn"] = (df["churn"] == "Yes").astype(int)
    return df


def setup_schema(engine: Engine) -> None:
    """Create the customers_raw table if it does not exist.

    Executes 001_create_raw.sql which uses CREATE TABLE IF NOT EXISTS and
    declares customerid as PRIMARY KEY. Unlike to_sql(if_exists='replace'),
    this never drops the table, so the PK constraint is never silently lost.
    """
    ddl = _SQL_SCHEMA.read_text()
    with engine.begin() as conn:
        conn.execute(text(ddl))


def _load_staging(df: pd.DataFrame, engine: Engine) -> None:
    """Bulk-load the full dataset into a throwaway staging table.

    No constraints on the staging table — this is the fast path. Column types
    are inferred by pandas; no PRIMARY KEY or NOT NULL is declared. Speed is
    the only goal here; correctness is enforced by the main table at merge time.
    Any leftover staging table from a previous failed run is replaced.
    """
    df.to_sql(
        "customers_raw_staging",
        engine,
        if_exists="replace",
        index=False,
        method="multi",
        chunksize=1000,
    )
    logger.info("staging_loaded", csv_rows=len(df), table="customers_raw_staging")


def _merge_from_staging(update_cols: list[str], engine: Engine) -> int:
    """Merge customers_raw_staging into customers_raw, then drop the staging table.

    The INSERT … ON CONFLICT DO UPDATE and DROP TABLE run in a single
    transaction: either the full merge commits or nothing changes in the main
    table and the staging table survives intact for a retry.

    Column names in update_cols come from load_raw_csv() — not user input —
    so the f-string SET clause is not a SQL injection risk.

    Returns the DB-reported row count (inserts + updates) from the merge
    statement — not the CSV row count, which can differ if the source file
    is partial or if CHECK constraints reject rows mid-transaction.
    """
    set_clause = ", ".join(f"{col} = EXCLUDED.{col}" for col in update_cols)
    with engine.begin() as conn:
        result = conn.execute(
            text(
                f"INSERT INTO customers_raw "
                f"SELECT * FROM customers_raw_staging "
                f"ON CONFLICT (customerid) DO UPDATE SET {set_clause}"
            )
        )
        conn.execute(text("DROP TABLE IF EXISTS customers_raw_staging"))
    return int(result.rowcount)


def ingest(path: Path, engine: Engine | None = None) -> int:
    """Load the raw Telco CSV into the customers_raw Postgres table.

    Uses the industry-standard staging table pattern:
      1. Bulk-load into customers_raw_staging (no constraints, fast).
      2. MERGE from staging into customers_raw via INSERT … ON CONFLICT DO UPDATE.
      3. Drop the staging table inside the same transaction as the merge.

    This mirrors the dbt incremental model pattern used in production warehouses
    (Snowflake / BigQuery MERGE). The main table's PRIMARY KEY is never dropped;
    the merge is fully atomic. Returns the number of rows loaded.
    """
    if engine is None:
        engine = get_engine()
    df = load_raw_csv(path)
    validate_raw(df, strict=True)
    setup_schema(engine)
    update_cols = [c for c in df.columns if c != "customerid"]
    csv_rows = len(df)
    _load_staging(df, engine)
    n = _merge_from_staging(update_cols, engine)
    if n != csv_rows:
        raise RuntimeError(
            f"Merge row count mismatch: DB reported {n} rows processed "
            f"but CSV contained {csv_rows} — check Postgres logs for constraint violations."
        )
    logger.info("merge_complete", db_rows=n, csv_rows=csv_rows, table="customers_raw")
    return n


if __name__ == "__main__":
    import argparse
    import sys

    from dotenv import load_dotenv

    from telco_churn.utils.logging import configure_logging
    from telco_churn.utils.paths import load_config

    load_dotenv()
    configure_logging()

    cfg = load_config()
    default_csv = Path(cfg.paths.raw_data)

    parser = argparse.ArgumentParser(description="Ingest raw Telco CSV into Postgres.")
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=default_csv,
        help=f"Path to the raw CSV file (default: {default_csv})",
    )
    args = parser.parse_args()

    try:
        ingest(path=args.csv_path)
    except ValueError as e:
        logger.error("ingest_schema_mismatch", error=str(e), exc_info=True)
        sys.exit(1)
    except ValidationError as e:
        logger.error(
            "ingest_blocked_by_validation",
            errors=len(e.result.errors),
            exc_info=True,
        )
        sys.exit(1)
    except RuntimeError as e:
        logger.error("ingest_row_count_mismatch", error=str(e), exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error("ingest_failed", error=str(e), exc_info=True)
        sys.exit(1)

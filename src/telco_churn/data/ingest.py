"""Idempotent CSV-to-Postgres loader for the Telco Churn raw dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from telco_churn.data.checks import MAX_NULL_RATE, MIN_ROWS, frame_checksum
from telco_churn.data.schema import RawSchema
from telco_churn.data.validate import ValidationError, validate_raw
from telco_churn.utils.db import get_engine
from telco_churn.utils.logging import get_logger
from telco_churn.utils.paths import get_project_root

__all__ = [
    "IngestReceipt",
    "load_raw_csv",
    "setup_schema",
    "ingest",
]


@dataclass(frozen=True)
class IngestReceipt:
    """Outcome of one ingest() call — the DVC `ingest` stage's file stand-in for the Postgres write.

    DVC outs must be files; the real result of ingest() is rows in Postgres,
    which cannot itself be a dep or an out. This receipt is the hashable
    artifact that stands in for that side effect.
    """

    rows_loaded: int
    csv_rows: int
    null_counts: dict[str, int]
    frame_checksum: str


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

    The INSERT names its columns explicitly on both the target and the SELECT
    side rather than relying on `INSERT INTO customers_raw SELECT * FROM
    customers_raw_staging`, which Postgres maps positionally: it would insert
    correctly only by coincidence, as long as the staging table's column
    order (inherited from the CSV header) happened to match customers_raw's
    declared DDL order. Several columns share an identical type with no
    distinguishing CHECK constraint (e.g. has_partner/dependents/
    phoneservice/paperlessbilling, all VARCHAR(3) Yes/No flags), so a
    positional drift between the two orderings would silently swap values
    between them instead of raising. Naming both column lists makes the
    mapping order-independent — a future DDL or CSV-header reorder becomes a
    no-op, and a genuinely missing column raises immediately instead of
    shifting values into its neighbor.

    Returns the DB-reported row count (inserts + updates) from the merge
    statement — not the CSV row count, which can differ if the source file
    is partial or if CHECK constraints reject rows mid-transaction.
    """
    all_cols = ["customerid", *update_cols]
    col_list = ", ".join(all_cols)
    set_clause = ", ".join(f"{col} = EXCLUDED.{col}" for col in update_cols)
    with engine.begin() as conn:
        result = conn.execute(
            text(
                f"INSERT INTO customers_raw ({col_list}) "
                f"SELECT {col_list} FROM customers_raw_staging "
                f"ON CONFLICT (customerid) DO UPDATE SET {set_clause}"
            )
        )
        conn.execute(text("DROP TABLE IF EXISTS customers_raw_staging"))
    return int(result.rowcount)


def ingest(
    path: Path,
    engine: Engine | None = None,
    min_rows: int = MIN_ROWS,
    max_null_rate: float = MAX_NULL_RATE,
) -> IngestReceipt:
    """Load the raw Telco CSV into the customers_raw Postgres table.

    Uses the industry-standard staging table pattern:
      1. Bulk-load into customers_raw_staging (no constraints, fast).
      2. MERGE from staging into customers_raw via INSERT … ON CONFLICT DO UPDATE.
      3. Drop the staging table inside the same transaction as the merge.

    This mirrors the dbt incremental model pattern used in production warehouses
    (Snowflake / BigQuery MERGE). The main table's PRIMARY KEY is never dropped;
    the merge is fully atomic. Returns an IngestReceipt describing the load.

    min_rows/max_null_rate feed validate_raw's gate-5 thresholds and default to
    the same checks.py constants validate_raw itself falls back to, so a direct
    call — a unit test, a one-off script — needs no config. The __main__ entry
    point below overrides both from config: validation.min_rows /
    validation.max_null_rate, so an operator-tunable value in configs/config.yaml
    actually reaches the ingest path, not just validate.py's own CLI.
    """
    if engine is None:
        engine = get_engine()
    df = load_raw_csv(path)
    validate_raw(df, strict=True, min_rows=min_rows, max_null_rate=max_null_rate)
    setup_schema(engine)
    update_cols = [c for c in df.columns if c != "customerid"]
    csv_rows = len(df)
    _load_staging(df, engine)
    n = _merge_from_staging(update_cols, engine)
    if n != csv_rows:
        # A CHECK/NOT NULL violation would raise IntegrityError out of _merge_from_staging's
        # engine.begin() before this line is ever reached — Postgres fails the whole
        # INSERT...SELECT atomically, it does not skip individual rows. validate_raw already
        # rejects the same violations upstream (Gate 1 mirrors the DDL's CHECK constraints;
        # Gate 2 rejects duplicate customerid). So a mismatch here, with no exception raised,
        # points at a staging/merge invariant break instead — e.g. a bug in _load_staging.
        raise RuntimeError(
            f"Merge row count mismatch: DB reported {n} rows processed "
            f"but CSV contained {csv_rows} — investigate _load_staging/_merge_from_staging, "
            "not per-row constraint violations."
        )
    logger.info("merge_complete", db_rows=n, csv_rows=csv_rows, table="customers_raw")
    return IngestReceipt(
        rows_loaded=n,
        csv_rows=csv_rows,
        null_counts={col: int(df[col].isna().sum()) for col in df.columns},
        frame_checksum=frame_checksum(df),
    )


if __name__ == "__main__":
    import json
    import sys
    from dataclasses import asdict

    from dotenv import load_dotenv

    from telco_churn.utils.logging import configure_logging
    from telco_churn.utils.paths import activate_config, compose_config

    load_dotenv()
    configure_logging()

    cfg = compose_config(overrides=sys.argv[1:] or None)
    activate_config(cfg)
    csv_path = get_project_root() / cfg.paths.raw_data

    try:
        receipt = ingest(
            path=csv_path,
            min_rows=int(cfg.validation.min_rows),
            max_null_rate=float(cfg.validation.max_null_rate),
        )
        receipt_path = get_project_root() / "reports" / "ingest_receipt.json"
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(
            json.dumps(asdict(receipt), indent=2, sort_keys=True) + "\n",
            newline="\n",
        )
        logger.info("ingest_receipt_written", path=str(receipt_path))
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

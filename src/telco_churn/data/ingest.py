"""Idempotent CSV-to-Postgres loader for the Telco Churn raw dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import types
from sqlalchemy.engine import Engine

from telco_churn.utils.db import get_engine
from telco_churn.utils.logging import get_logger

logger = get_logger(__name__)

RAW_CSV = Path("datasets/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Explicit column types keep the DB schema aligned with 001_create_raw.sql.
_DTYPE_MAP: dict[str, types.TypeEngine[Any]] = {
    "customerid": types.VARCHAR(20),
    "gender": types.VARCHAR(10),
    "seniorcitizen": types.SmallInteger(),
    "has_partner": types.VARCHAR(3),
    "dependents": types.VARCHAR(3),
    "tenure": types.SmallInteger(),
    "phoneservice": types.VARCHAR(3),
    "multiplelines": types.VARCHAR(25),
    "internetservice": types.VARCHAR(25),
    "onlinesecurity": types.VARCHAR(25),
    "onlinebackup": types.VARCHAR(25),
    "deviceprotection": types.VARCHAR(25),
    "techsupport": types.VARCHAR(25),
    "streamingtv": types.VARCHAR(25),
    "streamingmovies": types.VARCHAR(25),
    "contract_type": types.VARCHAR(20),
    "paperlessbilling": types.VARCHAR(3),
    "paymentmethod": types.VARCHAR(45),
    "monthlycharges": types.Numeric(8, 2),
    "totalcharges": types.Numeric(10, 2),
    "churn": types.SmallInteger(),
}


def load_raw_csv(path: Path = RAW_CSV) -> pd.DataFrame:
    """Load and type-coerce the raw Telco CSV.

    Coerces TotalCharges to numeric (whitespace in source → NaN for the 11
    zero-tenure customers) and encodes Churn Yes/No as a binary integer
    column named 'churn'.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Raw CSV not found at {path}. Run `make data` to download it."
        )
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["churn"] = (df["Churn"] == "Yes").astype(int)
    df = df.drop(columns=["Churn"])
    df.columns = df.columns.str.lower()
    return df.rename(columns={"partner": "has_partner", "contract": "contract_type"})


def ingest(path: Path = RAW_CSV, engine: Engine | None = None) -> int:
    """Load the raw Telco CSV into the customers_raw Postgres table.

    The table is replaced on each call, making the operation idempotent.
    Returns the number of rows loaded.
    """
    if engine is None:
        engine = get_engine()
    df = load_raw_csv(path)
    df.to_sql(
        "customers_raw",
        engine,
        if_exists="replace",
        index=False,
        dtype=_DTYPE_MAP,  # pyright: ignore[reportArgumentType]  # pandas-stubs DtypeArg omits dict
        method="multi",
        chunksize=1000,
    )
    n = len(df)
    logger.info("ingested", rows=n, table="customers_raw")
    return n


if __name__ == "__main__":
    import sys

    from dotenv import load_dotenv

    from telco_churn.utils.logging import configure_logging

    load_dotenv()
    configure_logging()
    try:
        ingest()
    except Exception as e:
        logger.error("ingest failed", error=str(e))
        sys.exit(1)

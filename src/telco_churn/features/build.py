"""Feature engineering: column group definitions and CLI entry point."""

from __future__ import annotations

import pandas as pd
import pandera as pa
from sqlalchemy import text
from sqlalchemy.engine import Engine

from telco_churn.features.schema import (
    COMMITTED_FEATURES,
    COMMITTED_FEATURES_DECISION,
    COMMITTED_FEATURES_DECISION_RUN_ID,
    FEATURE_SCHEMA,
    CustomerFeaturesSchema,
    FeatureOutputSchema,
    FeatureSchema,
)
from telco_churn.utils.logging import get_logger

__all__ = [
    "COMMITTED_FEATURES",
    "COMMITTED_FEATURES_DECISION",
    "COMMITTED_FEATURES_DECISION_RUN_ID",
    "FEATURE_SCHEMA",
    "FeatureSchema",
    "TARGET_COL",
    "SQL_FEATURE_COLS",
    "build_feature_query",
    "load_customer_features",
    "build_feature_df",
]

logger = get_logger(__name__)

TARGET_COL = "churn"

SQL_FEATURE_COLS: list[str] = (
    ["customerid"]
    + list(FEATURE_SCHEMA.binary)
    + list(FEATURE_SCHEMA.multi_cat)
    + list(FEATURE_SCHEMA.numeric)
    + [TARGET_COL]
)


def build_feature_query(reserve_months: list[int] | None = None) -> str:
    """Build the customer_features SELECT, scoped to the given reserve months.

    `customer_features` is sourced from `training_pool`, which holds both the
    original CSV-seeded population (`reserve_month IS NULL`) and every
    past/future reserve cohort
    (`reserve_month` 1..6). Every training query includes the seeded
    population; `reserve_months=None` (the default v1/cold-start path) selects
    *only* that population — today's exact customers_raw-derived behavior,
    unaffected by any reserve cohort that has since matured. A non-empty list
    additionally folds in those matured cohorts (the fold-forward training
    query, §D1/§D4) — this is what Phase 10b's training_cycle.py will call for
    a routine retrain cycle; nothing in this repo calls it with a non-None
    argument yet.

    reserve_months values are cast through int() before interpolation — they
    come from this project's own fold-forward bookkeeping (reserve_manifest.parquet
    cohort numbers 1..6), never request/user input, and the cast both
    documents that and rejects a non-numeric value loudly rather than
    building an unintended query string.
    """
    cols = ", ".join(SQL_FEATURE_COLS)
    if not reserve_months:
        return f"SELECT {cols} FROM customer_features WHERE reserve_month IS NULL"
    months = ", ".join(str(int(m)) for m in reserve_months)
    return (
        f"SELECT {cols} FROM customer_features "
        f"WHERE reserve_month IS NULL OR reserve_month IN ({months})"
    )


def load_customer_features(
    engine: Engine, reserve_months: list[int] | None = None
) -> pd.DataFrame:
    """Read customer_features from Postgres, scoped to build_feature_query's filter.

    Returns exactly SQL_FEATURE_COLS — training_pool_id/reserve_month (the
    join key and the fold-forward filter column) never leave this function,
    since build_feature_df's schemas describe the customerid+feature+churn
    shape only.
    """
    return pd.read_sql_query(text(build_feature_query(reserve_months)), engine)


@pa.check_input(CustomerFeaturesSchema)  # type: ignore[untyped-decorator]
@pa.check_output(FeatureOutputSchema)  # type: ignore[untyped-decorator]
def build_feature_df(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and return the feature DataFrame from the customer_features SQL view.

    df must contain all columns produced by the customer_features SQL view plus a
    churn column. customerid and churn pass through unchanged. Extracting y and
    selecting feature columns before training is the caller's responsibility (train.py).

    Sorted by customerid: customer_features is a SQL view with no ORDER BY, so
    its row order is whatever Postgres's query plan happens to produce for that
    JOIN — not guaranteed stable across different loads of the same data. Every
    downstream consumer (CV fold assignment in particular) splits by row
    position, so an unpinned order makes "reproducible given random_state=42"
    false in practice. Same fix data/split.py already applies to the manifest,
    for the same reason.
    """
    return df.sort_values("customerid", kind="stable").reset_index(drop=True)


def _reject_if_empty(df: pd.DataFrame) -> None:
    """Raise if df has zero rows — refuses to cache an empty features Parquet.

    Pandera's per-column checks on CustomerFeaturesSchema/FeatureOutputSchema
    pass vacuously on a zero-row frame, so a broken upstream SQL join would
    otherwise write (and DVC would cache) an empty artifact undetected.
    """
    if df.empty:
        raise ValueError(
            "build_feature_df returned zero rows — refusing to write an "
            "empty features Parquet"
        )


if __name__ == "__main__":
    import sys

    from dotenv import load_dotenv
    from sqlalchemy.exc import SQLAlchemyError

    from telco_churn.features.accessor import features_path
    from telco_churn.features.sql_features import build_sql_features
    from telco_churn.utils.db import get_engine
    from telco_churn.utils.logging import configure_logging
    from telco_churn.utils.paths import (
        activate_config,
        compose_config,
        get_project_root,
    )

    load_dotenv()
    configure_logging()

    cfg = compose_config(overrides=sys.argv[1:] or None)
    activate_config(cfg)
    sql_dir = get_project_root() / cfg.paths.sql_features

    try:
        engine = get_engine()
        build_sql_features(engine, sql_dir=sql_dir)

        df_raw = load_customer_features(engine)
        df_out = build_feature_df(df_raw)
        _reject_if_empty(df_out)

        out_path = features_path()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_parquet(out_path, index=False)

        feature_cols = (
            list(FEATURE_SCHEMA.binary)
            + list(FEATURE_SCHEMA.multi_cat)
            + list(FEATURE_SCHEMA.numeric)
        )
        logger.info(
            "feature dataframe saved",
            path=str(out_path),
            rows=int(df_out.shape[0]),
            feature_cols=len(feature_cols),
        )
    except pa.errors.SchemaError as e:
        logger.error("feature_build_schema_invalid", error=str(e), exc_info=True)
        sys.exit(1)
    except SQLAlchemyError as e:
        logger.error("feature_build_db_error", error=str(e), exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error("feature_build_failed", error=str(e), exc_info=True)
        sys.exit(1)

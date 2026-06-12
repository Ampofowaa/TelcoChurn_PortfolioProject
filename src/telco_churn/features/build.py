"""Feature engineering: column group definitions (SQL + Python features) and CLI entry point."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pandera as pa

from telco_churn.features.schema import CustomerFeaturesSchema, FeatureOutputSchema
from telco_churn.utils.logging import get_logger

__all__ = [
    "BINARY_STR_COLS",
    "BINARY_INT_COLS",
    "MULTI_CAT_COLS",
    "NUMERIC_COLS",
    "PYTHON_ENGINEERED_COLS",
    "TARGET_COL",
    "SQL_FEATURE_COLS",
    "build_feature_df",
]

logger = get_logger(__name__)

BINARY_STR_COLS: list[str] = [
    "gender",
    "has_partner",
    "dependents",
    "phoneservice",
    "paperlessbilling",
]

BINARY_INT_COLS: list[str] = [
    "seniorcitizen",
    "is_long_month_to_month",  # Python-engineered: H1
]

MULTI_CAT_COLS: list[str] = [
    "multiplelines",
    "internetservice",
    "onlinesecurity",
    "onlinebackup",
    "deviceprotection",
    "techsupport",
    "streamingtv",
    "streamingmovies",
    "contract_type",
    "paymentmethod",
    "tenure_cohort",  # SQL-engineered: tenure_buckets.sql
    "fiber_contract",  # Python-engineered: H3a
    "dsl_contract",  # Python-engineered: H3b
]

NUMERIC_COLS: list[str] = [
    "tenure",
    "monthlycharges",
    "totalcharges",
    "charge_per_service",  # SQL-engineered: charge_per_service.sql
    "monthly_to_total_ratio",  # Python-engineered: H2
]

# Absent from the customer_features SQL view — excluded from the SELECT in __main__.
PYTHON_ENGINEERED_COLS: frozenset[str] = frozenset(
    {
        "is_long_month_to_month",
        "monthly_to_total_ratio",
        "fiber_contract",
        "dsl_contract",
    }
)

TARGET_COL = "churn"

# Columns to read from the customer_features SQL view — all feature columns minus
# the Python-engineered ones (absent from the view) plus customerid and target.
SQL_FEATURE_COLS: list[str] = ["customerid"] + [
    c
    for c in BINARY_STR_COLS
    + BINARY_INT_COLS
    + MULTI_CAT_COLS
    + NUMERIC_COLS
    + [TARGET_COL]
    if c not in PYTHON_ENGINEERED_COLS
]

# Phase 5 replaces PYTHON_ENGINEERED_COLS with provenance-based source sets, making
# this filter unnecessary. Until then, guard that PYTHON_ENGINEERED_COLS stays in sync:
# a column removed from the typed lists without being removed here would silently keep
# being excluded from SQL_FEATURE_COLS, causing a missing-column bug at DB read time.
assert PYTHON_ENGINEERED_COLS <= set(
    BINARY_STR_COLS + BINARY_INT_COLS + MULTI_CAT_COLS + NUMERIC_COLS
), (
    "PYTHON_ENGINEERED_COLS references columns absent from typed feature lists — "
    "update the typed lists or remove the stale entry"
)


def _add_python_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute H1, H2, H3a, and H3b engineered features on a copy of df.

    NaN in monthly_to_total_ratio (11 zero-tenure rows) is preserved intentionally —
    handled downstream by SimpleImputer(strategy='median') in the training pipeline.
    totalcharges=0.0 (non-null zero) is also coerced to NaN before division — a 0.0
    with positive monthlycharges is a pipeline data bug, not a real customer state;
    treating it as missing gives the same downstream imputation path as the NaN rows.
    H3a/H3b are 4-level categoricals, not binary flags — churn rates span 54.6%→7.2%
    (fiber) and 32.2%→1.9% (DSL) across contract tiers, a range binary flags cannot
    represent.
    """
    df = df.copy()
    df["is_long_month_to_month"] = (
        (df["tenure"] > 24) & (df["contract_type"] == "Month-to-month")
    ).astype(int)
    safe_total = df["totalcharges"].replace(0.0, float("nan"))
    df["monthly_to_total_ratio"] = df["monthlycharges"] / safe_total
    df["fiber_contract"] = np.where(
        df["internetservice"] == "Fiber optic",
        df["contract_type"] + "_Fiber optic",
        "Not Fiber optic",
    )
    df["dsl_contract"] = np.where(
        df["internetservice"] == "DSL",
        df["contract_type"] + "_DSL",
        "Not DSL",
    )
    return df


@pa.check_input(CustomerFeaturesSchema)  # type: ignore[untyped-decorator]
@pa.check_output(FeatureOutputSchema)  # type: ignore[untyped-decorator]
def build_feature_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add the four Python-engineered columns (H1, H2, H3a, and H3b) and return the augmented DataFrame.

    df must contain all columns produced by the customer_features SQL view plus a
    churn column. customerid and churn pass through unchanged. Extracting y and
    selecting feature columns before training is the caller's responsibility
    (train.py).
    """
    return _add_python_features(df)


if __name__ == "__main__":
    import sys

    from dotenv import load_dotenv
    from omegaconf import OmegaConf
    from sqlalchemy.exc import SQLAlchemyError

    from telco_churn.features.sql_features import build_sql_features
    from telco_churn.utils.db import get_engine
    from telco_churn.utils.logging import configure_logging
    from telco_churn.utils.paths import get_project_root

    load_dotenv()
    configure_logging()

    cfg = OmegaConf.load(get_project_root() / "configs" / "config.yaml")
    processed_dir = Path(cfg.paths.processed_data)
    sql_dir = Path(cfg.paths.sql_features)

    try:
        engine = get_engine()
        build_sql_features(engine, sql_dir=sql_dir)

        df_raw = pd.read_sql_table(
            "customer_features", engine, columns=SQL_FEATURE_COLS
        )
        df_out = build_feature_df(df_raw)

        out_path = processed_dir / "telco_churn_processed.csv"
        processed_dir.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(out_path, index=False)

        feature_cols = BINARY_STR_COLS + BINARY_INT_COLS + MULTI_CAT_COLS + NUMERIC_COLS
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

"""Feature engineering: column group definitions and CLI entry point."""

from __future__ import annotations

import pandas as pd
import pandera as pa

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


if __name__ == "__main__":
    import sys

    from dotenv import load_dotenv
    from sqlalchemy.exc import SQLAlchemyError

    from telco_churn.data.schema import RawSchema
    from telco_churn.data.validate import ValidationError, validate_clean
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

        df_raw = pd.read_sql_table(
            "customer_features", engine, columns=SQL_FEATURE_COLS
        )
        df_out = build_feature_df(df_raw)

        # Median-fill totalcharges for this check only — the shipped parquet keeps
        # NaN, since real imputation must stay train-only inside the ColumnTransformer
        # (CLAUDE.md: SimpleImputer fit per-fold). This just exercises CleanedSchema's
        # totalcharges-non-null and totalcharges>=monthlycharges invariants so a future
        # SQL/feature-build regression fails the build instead of only a manual notebook run.
        #
        # Narrowed to RawSchema's own columns before validating: CleanedSchema is
        # RawSchema (Config.strict = True — "unexpected columns fail loudly") plus the
        # totalcharges tweaks, so it knows nothing about SQL-engineered columns like
        # charge_per_service. df_out carries those too; validating it unfiltered fails
        # strict mode on every adopted feature, not just an actual raw-column regression.
        clean_check_df = df_out[list(RawSchema.to_schema().columns.keys())].copy()
        clean_check_df["totalcharges"] = clean_check_df["totalcharges"].fillna(
            clean_check_df["totalcharges"].median()
        )
        validate_clean(
            clean_check_df,
            strict=True,
            reports_dir=get_project_root() / cfg.paths.validation_reports,
            min_rows=int(cfg.validation.min_rows),
            max_null_rate=float(cfg.validation.max_null_rate),
        )

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
    except ValidationError as e:
        logger.error(
            "feature_build_clean_validation_failed",
            errors=len(e.result.errors),
            warnings=len(e.result.warnings),
            exc_info=True,
        )
        sys.exit(1)
    except SQLAlchemyError as e:
        logger.error("feature_build_db_error", error=str(e), exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error("feature_build_failed", error=str(e), exc_info=True)
        sys.exit(1)

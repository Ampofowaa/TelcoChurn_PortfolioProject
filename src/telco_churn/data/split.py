"""Canonical dev/test split — sealed as a customerid partition before feature discovery.

The split depends only on (customerid, churn), never on any engineered feature, so
it is established once, right after raw-data validation, before feature discovery
or feature engineering run. See docs/canonical-split-refactor-tasks.md for the
motivating refactor: this module is the single place the split is defined; every
downstream consumer (feature discovery, models/train/, evaluate.py) imports it
rather than redefining it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from telco_churn.utils.logging import get_logger
from telco_churn.utils.paths import get_project_root, load_config

__all__ = [
    "DEV",
    "TEST",
    "SPLIT_COL",
    "make_split",
    "write_split",
    "load_split",
    "dev_ids",
    "test_ids",
    "partition",
]

logger = get_logger(__name__)

DEV = "dev"
TEST = "test"
SPLIT_COL = "split"


def _default_manifest_path() -> Path:
    """Return the canonical split manifest path, anchored to the project root."""
    cfg = load_config()
    return (
        Path(get_project_root() / cfg.paths.processed_data) / "split_manifest.parquet"
    )


def make_split(
    ids: pd.Series,
    labels: pd.Series,
    test_size: float,
    random_state: int,
) -> pd.DataFrame:
    """Return a stratified dev/test partition of ids, by label.

    Depends only on (customerid, churn) — not on any engineered feature — so it can
    be established before feature discovery or feature engineering run. ids/labels
    are sorted by customerid before splitting (not just the output), so the result
    is independent of the caller's row order: sklearn's stratified split is
    positionally order-sensitive even with a fixed random_state — the same random
    permutation applied to a different row order selects different individuals —
    and sources like an unordered SQL `SELECT *` make no row-order guarantee.
    Re-running with the same inputs and random_state always produces a
    byte-identical manifest, regardless of the order ids/labels arrived in.

    Raises ValueError when labels has fewer than 2 classes — this covers both the
    single-class stratification guard and the empty-frame edge case, since an empty
    Series has 0 unique values.
    """
    if labels.nunique() < 2:
        raise ValueError(
            "make_split requires at least 2 classes in `labels` to stratify; got "
            f"{labels.nunique()}."
        )
    ordered = pd.DataFrame(
        {"customerid": ids.to_numpy(), "label": labels.to_numpy()}
    ).sort_values("customerid", kind="stable")
    ids_dev, ids_test = train_test_split(
        ordered["customerid"],
        test_size=test_size,
        random_state=random_state,
        stratify=ordered["label"],
    )
    manifest = pd.concat(
        [
            pd.DataFrame({"customerid": ids_dev.to_numpy(), SPLIT_COL: DEV}),
            pd.DataFrame({"customerid": ids_test.to_numpy(), SPLIT_COL: TEST}),
        ],
        ignore_index=True,
    )
    return manifest.sort_values("customerid").reset_index(drop=True)


def write_split(df: pd.DataFrame, path: Path) -> None:
    """Write the split manifest to a Parquet file, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_split(path: Path | None = None) -> pd.DataFrame:
    """Load the canonical split manifest.

    path defaults to the canonical location under cfg.paths.processed_data; pass an
    explicit path in tests to avoid touching the real project artifact.
    """
    resolved = path if path is not None else _default_manifest_path()
    if not resolved.exists():
        raise FileNotFoundError(
            f"Split manifest not found at {resolved}. Run `make split` first."
        )
    return pd.read_parquet(resolved)


def dev_ids(manifest: pd.DataFrame | None = None) -> pd.Series:
    """Return the customerid Series for the dev partition.

    manifest defaults to load_split(); pass an in-memory manifest in tests.
    """
    resolved = manifest if manifest is not None else load_split()
    return resolved.loc[resolved[SPLIT_COL] == DEV, "customerid"].reset_index(drop=True)


def test_ids(
    manifest: pd.DataFrame | None = None,  # noqa: PT028 — not a pytest test
) -> pd.Series:
    """Return the customerid Series for the test partition.

    manifest defaults to load_split(); pass an in-memory manifest in tests.
    Only evaluate.py may consume this — see CLAUDE.md modelling invariants:
    'test set touched once.'
    """
    resolved = manifest if manifest is not None else load_split()
    return resolved.loc[resolved[SPLIT_COL] == TEST, "customerid"].reset_index(
        drop=True
    )


def partition(
    df: pd.DataFrame, manifest: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join an arbitrary customerid-keyed frame to the manifest; return (dev_df, test_df).

    manifest defaults to load_split(); pass an in-memory manifest in tests. Raises
    ValueError if any row in df fails to match a manifest customerid — a reset index
    or coerced dtype could otherwise silently misalign the partition.
    """
    resolved = manifest if manifest is not None else load_split()
    merged = df.merge(resolved, on="customerid", how="inner", validate="one_to_one")
    if len(merged) != len(df):
        raise ValueError(
            f"partition(): {len(df) - len(merged)} row(s) in df did not match the "
            "split manifest by customerid."
        )
    dev_df = (
        merged.loc[merged[SPLIT_COL] == DEV]
        .drop(columns=[SPLIT_COL])
        .reset_index(drop=True)
    )
    test_df = (
        merged.loc[merged[SPLIT_COL] == TEST]
        .drop(columns=[SPLIT_COL])
        .reset_index(drop=True)
    )
    return dev_df, test_df


_RAW_TABLE = "customers_raw"

if __name__ == "__main__":
    import sys

    from dotenv import load_dotenv
    from sqlalchemy.exc import SQLAlchemyError

    from telco_churn.data.validate import ValidationError, validate_raw
    from telco_churn.utils.db import get_engine
    from telco_churn.utils.logging import configure_logging
    from telco_churn.utils.paths import compose_config

    load_dotenv()
    configure_logging()

    cfg = compose_config()

    try:
        engine = get_engine()
        df = pd.read_sql_table(_RAW_TABLE, engine)
        validate_raw(df, strict=True)
        manifest = make_split(
            ids=df["customerid"],
            labels=df["churn"],
            test_size=float(cfg.training_setup.test_size),
            random_state=int(cfg.random_seed),
        )
        out_path = _default_manifest_path()
        write_split(manifest, out_path)
        n_dev = int((manifest[SPLIT_COL] == DEV).sum())
        n_test = int((manifest[SPLIT_COL] == TEST).sum())
        logger.info(
            "split_manifest_written",
            path=str(out_path),
            n_dev=n_dev,
            n_test=n_test,
        )
    except OSError as e:
        logger.error("split_engine_config_error", error=str(e), exc_info=True)
        sys.exit(1)
    except ValueError as e:
        logger.error("split_invalid", error=str(e), exc_info=True)
        sys.exit(1)
    except ValidationError as e:
        logger.error(
            "split_blocked_by_validation",
            errors=len(e.result.errors),
            exc_info=True,
        )
        sys.exit(1)
    except SQLAlchemyError as e:
        logger.error("split_db_error", error=str(e), exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error("split_failed", error=str(e), exc_info=True)
        sys.exit(1)

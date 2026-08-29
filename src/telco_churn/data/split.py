"""Canonical dev/test split — sealed as a customerid partition before feature discovery.

The split depends only on (customerid, churn), never on any engineered feature, so
it is established once, right after raw-data validation, before feature discovery
or feature engineering run. This module is the single place the split is defined;
every downstream consumer (feature discovery, models/train/, evaluate.py) imports
it rather than redefining it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from telco_churn.utils.logging import get_logger
from telco_churn.utils.paths import get_project_root, load_config

__all__ = [
    "DEV",
    "TEST",
    "SPLIT_COL",
    "RESERVE_COL",
    "make_split",
    "write_split",
    "load_split",
    "dev_ids",
    "test_ids",
    "partition",
    "make_reserve",
    "write_reserve",
    "load_reserve",
    "reserve_ids",
    "sealed_test_ids",
]

logger = get_logger(__name__)

DEV = "dev"
TEST = "test"
SPLIT_COL = "split"
RESERVE_COL = "reserve_month"


def _default_manifest_path() -> Path:
    """Return the canonical split manifest path, anchored to the project root."""
    cfg = load_config()
    return (
        Path(get_project_root() / cfg.paths.processed_data) / "split_manifest.parquet"
    )


def _default_reserve_manifest_path() -> Path:
    """Return the canonical reserve manifest path, anchored to the project root."""
    cfg = load_config()
    return (
        Path(get_project_root() / cfg.paths.processed_data) / "reserve_manifest.parquet"
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
            f"Split manifest not found at {resolved}. Run `dvc repro split` first."
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


def make_reserve(
    ids: pd.Series,
    labels: pd.Series,
    n_months: int,
    fraction: float,
    random_state: int,
) -> pd.DataFrame:
    """Return a nullable cohort-month assignment for every id in `ids`.

    `ids`/`labels` are the *test*-partition customerid/label pair — this is the
    reserve/sealed-test subdivision of `test`, not a third top-level partition.
    A `fraction` slice of `ids`, stratified by `labels`,
    is carved into `n_months` roughly-equal, roughly-churn-balanced monthly cohorts
    (1..n_months); the remainder gets `reserve_month = NULL` — "sealed test, never
    reserved."

    Uses its own random draws, scoped only to this function — never the RandomState
    object make_split()'s train_test_split call constructs internally — so a reserve
    edit can never retroactively redraw the (dev, test) partition (§D4's "own
    independent random-state stream" safeguard). Deterministic under a fixed
    `random_state`: same inputs and seed always produce a byte-identical manifest.

    Cohort assignment is a per-label round-robin (not sklearn's StratifiedKFold),
    deliberately — StratifiedKFold raises when a class has fewer members than
    `n_months`, which a small synthetic test fixture routinely does; round-robin
    degrades gracefully (an under-represented class simply leaves some cohorts
    without one of its members) while still keeping each cohort close to the
    dataset's overall churn prevalence, same as `n_months` folds would.

    Raises ValueError when labels has fewer than 2 classes, `n_months` < 1, or
    `fraction` is not in (0, 1).
    """
    if n_months < 1:
        raise ValueError(f"make_reserve requires n_months >= 1; got {n_months}.")
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"make_reserve requires 0 < fraction < 1; got {fraction}.")
    if labels.nunique() < 2:
        raise ValueError(
            "make_reserve requires at least 2 classes in `labels` to stratify; got "
            f"{labels.nunique()}."
        )
    ordered = (
        pd.DataFrame({"customerid": ids.to_numpy(), "label": labels.to_numpy()})
        .sort_values("customerid", kind="stable")
        .reset_index(drop=True)
    )

    reserve_pool_ids, _sealed_ids = train_test_split(
        ordered["customerid"],
        train_size=fraction,
        random_state=random_state,
        stratify=ordered["label"],
    )
    reserve_pool_mask = ordered["customerid"].isin(set(reserve_pool_ids))
    reserve_pool = ordered.loc[reserve_pool_mask].reset_index(drop=True)

    rng = np.random.RandomState(random_state)
    cohort = np.empty(len(reserve_pool), dtype="int16")
    for label_value in sorted(reserve_pool["label"].unique()):
        group_idx = np.flatnonzero(reserve_pool["label"].to_numpy() == label_value)
        shuffled = rng.permutation(group_idx)
        cohort[shuffled] = (np.arange(len(shuffled)) % n_months) + 1
    reserve_pool = reserve_pool.assign(**{RESERVE_COL: cohort})

    manifest = ordered.merge(
        reserve_pool[["customerid", RESERVE_COL]], on="customerid", how="left"
    )
    manifest[RESERVE_COL] = manifest[RESERVE_COL].astype("Int16")
    return (
        manifest[["customerid", RESERVE_COL]]
        .sort_values("customerid")
        .reset_index(drop=True)
    )


def write_reserve(df: pd.DataFrame, path: Path) -> None:
    """Write the reserve manifest to a Parquet file, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_reserve(path: Path | None = None) -> pd.DataFrame:
    """Load the canonical reserve manifest.

    path defaults to the canonical location under cfg.paths.processed_data; pass an
    explicit path in tests to avoid touching the real project artifact.
    """
    resolved = path if path is not None else _default_reserve_manifest_path()
    if not resolved.exists():
        raise FileNotFoundError(
            f"Reserve manifest not found at {resolved}. Run `dvc repro split` first."
        )
    return pd.read_parquet(resolved)


def reserve_ids(manifest: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return the reserve cohort-month assignment: customerid, reserve_month.

    manifest defaults to load_reserve(); pass an in-memory manifest in tests.
    `reserve_month` is NULL for a customerid that is sealed test and never reserved.

    This is the release *schedule* only, not a data source — the actual training
    rows for a matured cohort come from prediction_log ⋈ prediction_outcomes,
    never from re-reading this manifest for feature values.
    """
    resolved = manifest if manifest is not None else load_reserve()
    return resolved.reset_index(drop=True)


def sealed_test_ids(
    split_manifest: pd.DataFrame | None = None,
    reserve_manifest: pd.DataFrame | None = None,
) -> pd.Series:
    """Return test_ids() minus every customerid the reserve manifest marks reserved.

    Both manifests default to their canonical on-disk location; pass in-memory
    manifests in tests. Carries a defensive assertion —
    `len(sealed_test_ids()) < len(test_ids())` — unconditionally true from the
    moment a reserve manifest exists (fraction > 0 by construction), so a silent
    no-op in reserve_ids() (a join miss, a swallowed missing-manifest error) fails
    loudly here rather than quietly serving the full test set mislabeled as the
    shrunk one.
    """
    test = test_ids(split_manifest)
    reserved = reserve_ids(reserve_manifest)
    reserved_ids = set(reserved.loc[reserved[RESERVE_COL].notna(), "customerid"])
    sealed = test[~test.isin(reserved_ids)].reset_index(drop=True)
    assert len(sealed) < len(test), (
        "sealed_test_ids() returned the full test set unchanged — reserve_ids() "
        "produced no reserved customerids. This indicates a broken reserve "
        "manifest, not a legitimate zero-reserve state."
    )
    return sealed


_RAW_TABLE = "customers_raw"

if __name__ == "__main__":
    import sys

    from dotenv import load_dotenv
    from sqlalchemy.exc import SQLAlchemyError

    from telco_churn.utils.db import get_engine
    from telco_churn.utils.logging import configure_logging
    from telco_churn.utils.paths import activate_config, compose_config

    load_dotenv()
    configure_logging()

    cfg = compose_config(overrides=sys.argv[1:] or None)
    activate_config(cfg)

    try:
        engine = get_engine()
        df = pd.read_sql_table(_RAW_TABLE, engine)
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

        # Written once, here, alongside split_manifest.parquet — never at any of
        # reserve_ids()'s own call sites.
        test_customerids = manifest.loc[manifest[SPLIT_COL] == TEST, "customerid"]
        test_labels = (
            df.set_index("customerid")
            .loc[test_customerids, "churn"]
            .reset_index(drop=True)
        )
        reserve_manifest = make_reserve(
            ids=test_customerids.reset_index(drop=True),
            labels=test_labels,
            n_months=int(cfg.training_setup.reserve_months),
            fraction=float(cfg.training_setup.reserve_fraction),
            random_state=int(cfg.training_setup.reserve_random_state),
        )
        reserve_out_path = _default_reserve_manifest_path()
        write_reserve(reserve_manifest, reserve_out_path)
        n_reserved = int(reserve_manifest[RESERVE_COL].notna().sum())
        n_sealed = int(reserve_manifest[RESERVE_COL].isna().sum())
        logger.info(
            "reserve_manifest_written",
            path=str(reserve_out_path),
            n_reserved=n_reserved,
            n_sealed=n_sealed,
        )
    except OSError as e:
        logger.error("split_engine_config_error", error=str(e), exc_info=True)
        sys.exit(1)
    except ValueError as e:
        logger.error("split_invalid", error=str(e), exc_info=True)
        sys.exit(1)
    except SQLAlchemyError as e:
        logger.error("split_db_error", error=str(e), exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error("split_failed", error=str(e), exc_info=True)
        sys.exit(1)

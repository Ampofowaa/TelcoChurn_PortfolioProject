"""Unit tests for telco_churn.data.split — the canonical dev/test partition."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from telco_churn.data.split import (
    DEV,
    SPLIT_COL,
    TEST,
    dev_ids,
    load_split,
    make_split,
    partition,
    write_split,
)
from telco_churn.data.split import test_ids as get_test_ids  # avoid pytest collection

_N = 200  # enough rows to stratify comfortably at 20% test size


@pytest.fixture
def ids_labels() -> tuple[pd.Series, pd.Series]:
    """Synthetic (customerid, churn) pair — ~30% churn prevalence, 200 rows."""
    rng = np.random.default_rng(0)
    ids = pd.Series([f"cust-{i:04d}" for i in range(_N)], name="customerid")
    labels = pd.Series(rng.choice([0, 1], size=_N, p=[0.7, 0.3]), name="churn")
    return ids, labels


@pytest.fixture
def manifest(ids_labels: tuple[pd.Series, pd.Series]) -> pd.DataFrame:
    """A manifest produced by make_split, reused by the accessor-helper tests."""
    ids, labels = ids_labels
    return make_split(ids, labels, test_size=0.2, random_state=42)


# ---------------------------------------------------------------------------
# make_split
# ---------------------------------------------------------------------------


def test_make_split_sizes(ids_labels: tuple[pd.Series, pd.Series]) -> None:
    """20% test_size produces an ~80/20 row split."""
    ids, labels = ids_labels
    result = make_split(ids, labels, test_size=0.2, random_state=42)
    n_dev = int((result[SPLIT_COL] == DEV).sum())
    n_test = int((result[SPLIT_COL] == TEST).sum())
    assert n_dev + n_test == _N
    assert n_test == pytest.approx(_N * 0.2, abs=1)


def test_make_split_disjoint(ids_labels: tuple[pd.Series, pd.Series]) -> None:
    """Dev and test customer IDs never overlap."""
    ids, labels = ids_labels
    result = make_split(ids, labels, test_size=0.2, random_state=42)
    dev_set = set(result.loc[result[SPLIT_COL] == DEV, "customerid"])
    test_set = set(result.loc[result[SPLIT_COL] == TEST, "customerid"])
    assert dev_set.isdisjoint(test_set)


def test_make_split_full_coverage(ids_labels: tuple[pd.Series, pd.Series]) -> None:
    """Every input customerid appears exactly once in the manifest."""
    ids, labels = ids_labels
    result = make_split(ids, labels, test_size=0.2, random_state=42)
    assert set(result["customerid"]) == set(ids)
    assert result["customerid"].is_unique


def test_make_split_stratified(ids_labels: tuple[pd.Series, pd.Series]) -> None:
    """Dev/test churn rate stays within tolerance of the overall prevalence."""
    ids, labels = ids_labels
    result = make_split(ids, labels, test_size=0.2, random_state=42)
    label_by_id = dict(zip(ids, labels, strict=True))
    dev_rate = (
        result.loc[result[SPLIT_COL] == DEV, "customerid"].map(label_by_id).mean()
    )
    test_rate = (
        result.loc[result[SPLIT_COL] == TEST, "customerid"].map(label_by_id).mean()
    )
    overall_rate = labels.mean()
    assert dev_rate == pytest.approx(overall_rate, abs=0.05)
    assert test_rate == pytest.approx(overall_rate, abs=0.05)


def test_make_split_deterministic(ids_labels: tuple[pd.Series, pd.Series]) -> None:
    """Same seed produces a byte-identical manifest across repeated calls."""
    ids, labels = ids_labels
    a = make_split(ids, labels, test_size=0.2, random_state=42)
    b = make_split(ids, labels, test_size=0.2, random_state=42)
    pd.testing.assert_frame_equal(a, b)


def test_make_split_order_invariant(ids_labels: tuple[pd.Series, pd.Series]) -> None:
    """Row order of the input does not affect the resulting partition.

    Guards against unordered data sources (e.g. an unordered SQL SELECT, which
    makes no row-order guarantee) producing a different dev/test split than a
    CSV-ordered source, even with the same random_state — sklearn's stratified
    split is positionally order-sensitive despite a fixed seed.
    """
    ids, labels = ids_labels
    baseline = make_split(ids, labels, test_size=0.2, random_state=42)

    shuffle_idx = np.random.default_rng(123).permutation(len(ids))
    ids_shuffled = ids.iloc[shuffle_idx].reset_index(drop=True)
    labels_shuffled = labels.iloc[shuffle_idx].reset_index(drop=True)
    shuffled = make_split(ids_shuffled, labels_shuffled, test_size=0.2, random_state=42)

    pd.testing.assert_frame_equal(baseline, shuffled)


def test_make_split_single_class_raises() -> None:
    """A single-class label column fails the stratification guard loudly."""
    ids = pd.Series([f"cust-{i:04d}" for i in range(10)])
    labels = pd.Series([0] * 10)
    with pytest.raises(ValueError, match="at least 2 classes"):
        make_split(ids, labels, test_size=0.2, random_state=42)


def test_make_split_empty_raises() -> None:
    """An empty frame fails the same stratification guard (0 unique classes)."""
    ids = pd.Series([], dtype=str)
    labels = pd.Series([], dtype=int)
    with pytest.raises(ValueError, match="at least 2 classes"):
        make_split(ids, labels, test_size=0.2, random_state=42)


# ---------------------------------------------------------------------------
# write_split / load_split round trip
# ---------------------------------------------------------------------------


def test_write_load_round_trip(manifest: pd.DataFrame, tmp_path: Path) -> None:
    """A written manifest loads back identically (content + dtypes)."""
    path = tmp_path / "split_manifest.parquet"
    write_split(manifest, path)
    loaded = load_split(path)
    pd.testing.assert_frame_equal(loaded, manifest)


def test_write_split_creates_parent_dirs(
    manifest: pd.DataFrame, tmp_path: Path
) -> None:
    """write_split creates missing parent directories."""
    path = tmp_path / "nested" / "dir" / "split_manifest.parquet"
    write_split(manifest, path)
    assert path.exists()


def test_load_split_missing_raises(tmp_path: Path) -> None:
    """Loading a nonexistent manifest fails loudly rather than returning empty."""
    with pytest.raises(FileNotFoundError, match="Run `dvc repro split` first"):
        load_split(tmp_path / "does_not_exist.parquet")


# ---------------------------------------------------------------------------
# dev_ids / test_ids / partition
# ---------------------------------------------------------------------------


def test_dev_ids_test_ids_disjoint_and_cover(manifest: pd.DataFrame) -> None:
    """dev_ids/test_ids partition the manifest exactly, with no overlap."""
    d = dev_ids(manifest)
    t = get_test_ids(manifest)
    assert set(d).isdisjoint(set(t))
    assert set(d) | set(t) == set(manifest["customerid"])
    assert len(d) + len(t) == len(manifest)


def test_partition_splits_arbitrary_frame(manifest: pd.DataFrame) -> None:
    """partition() joins an arbitrary customerid-keyed frame to the manifest."""
    df = pd.DataFrame(
        {
            "customerid": manifest["customerid"],
            "some_feature": range(len(manifest)),
        }
    )
    dev_df, test_df = partition(df, manifest)
    assert SPLIT_COL not in dev_df.columns
    assert SPLIT_COL not in test_df.columns
    assert set(dev_df["customerid"]) == set(dev_ids(manifest))
    assert set(test_df["customerid"]) == set(get_test_ids(manifest))
    assert len(dev_df) + len(test_df) == len(df)


def test_partition_unmatched_customerid_raises(manifest: pd.DataFrame) -> None:
    """A customerid absent from the manifest fails loudly instead of silently dropping."""
    df = pd.DataFrame(
        {
            "customerid": [*manifest["customerid"].tolist(), "unknown-cust"],
            "some_feature": range(len(manifest) + 1),
        }
    )
    with pytest.raises(ValueError, match="did not match the split manifest"):
        partition(df, manifest)


# ---------------------------------------------------------------------------
# __main__ CLI — reads from Postgres (customers_raw), so the subprocess
# integration test lives in tests/integration/test_split_postgres.py
# alongside the other DB-backed __main__ tests.
# ---------------------------------------------------------------------------

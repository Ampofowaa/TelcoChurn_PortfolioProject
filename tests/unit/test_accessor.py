"""Unit tests for telco_churn.features.accessor — the processed-features accessor."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pyarrow
import pytest
from pandera.errors import SchemaError, SchemaErrors

from telco_churn.features.accessor import (
    FEATURES_FILENAME,
    features_path,
    features_sha256,
    load_features,
)
from telco_churn.utils.paths import get_project_root, load_config


def _make_feature_row(
    tenure: int = 12,
    monthlycharges: float = 29.85,
    totalcharges: float = 358.20,
    churn: int = 0,
) -> dict[str, object]:
    """Minimal row matching FeatureOutputSchema."""
    return {
        "customerid": "1111-AAAAA",
        "gender": "Male",
        "seniorcitizen": 0,
        "has_partner": "Yes",
        "dependents": "No",
        "tenure": tenure,
        "phoneservice": "Yes",
        "multiplelines": "No",
        "internetservice": "DSL",
        "onlinesecurity": "Yes",
        "onlinebackup": "No",
        "deviceprotection": "No",
        "techsupport": "No",
        "streamingtv": "No",
        "streamingmovies": "No",
        "contract_type": "Month-to-month",
        "paperlessbilling": "Yes",
        "paymentmethod": "Electronic check",
        "monthlycharges": monthlycharges,
        "totalcharges": totalcharges,
        "churn": churn,
        "charge_per_service": monthlycharges / 3.0,
    }


@pytest.fixture
def valid_features_df() -> pd.DataFrame:
    """Two-row DataFrame covering both churn classes, matching FeatureOutputSchema."""
    return pd.DataFrame(
        [
            _make_feature_row(tenure=12, churn=0),
            _make_feature_row(tenure=36, churn=1),
        ]
    )


# ---------------------------------------------------------------------------
# features_path
# ---------------------------------------------------------------------------


def test_features_path_default_matches_configured_processed_dir() -> None:
    """With no override, features_path() resolves under cfg.paths.processed_data,
    anchored to the project root — never a bare relative path."""
    cfg = load_config()
    expected = get_project_root() / str(cfg.paths.processed_data) / FEATURES_FILENAME
    assert features_path() == expected


def test_features_path_explicit_override_returned_unchanged(tmp_path: Path) -> None:
    """An explicit path bypasses the canonical project location entirely — the
    override tests use to avoid touching the real project artifact."""
    explicit = tmp_path / "custom_features.parquet"
    assert features_path(explicit) == explicit


# ---------------------------------------------------------------------------
# load_features
# ---------------------------------------------------------------------------


def test_load_features_valid_file_round_trips(
    tmp_path: Path, valid_features_df: pd.DataFrame
) -> None:
    """A schema-conformant Parquet file loads and validates cleanly, preserving
    row count and columns."""
    path = tmp_path / "features.parquet"
    valid_features_df.to_parquet(path, index=False)

    loaded = load_features(path)

    assert len(loaded) == len(valid_features_df)
    assert set(valid_features_df.columns).issubset(set(loaded.columns))


def test_load_features_missing_column_raises(
    tmp_path: Path, valid_features_df: pd.DataFrame
) -> None:
    """A file missing a required column (e.g. the SQL-engineering step was
    skipped) fails schema validation loudly, not silently."""
    path = tmp_path / "features.parquet"
    valid_features_df.drop(columns=["charge_per_service"]).to_parquet(path, index=False)

    with pytest.raises((SchemaError, SchemaErrors)):
        load_features(path)


def test_load_features_wrong_dtype_raises(
    tmp_path: Path, valid_features_df: pd.DataFrame
) -> None:
    """A column that cannot satisfy its declared dtype (tenure as text) fails
    schema validation."""
    bad_df = valid_features_df.copy()
    bad_df["tenure"] = "twelve"
    path = tmp_path / "features.parquet"
    bad_df.to_parquet(path, index=False)

    with pytest.raises((SchemaError, SchemaErrors)):
        load_features(path)


def test_load_features_empty_file_raises(tmp_path: Path) -> None:
    """A zero-byte file is not a valid Parquet container (no magic bytes/footer)
    and fails loudly at the pyarrow read boundary rather than silently
    returning an empty frame."""
    path = tmp_path / "features.parquet"
    path.write_text("")

    with pytest.raises(pyarrow.lib.ArrowInvalid):
        load_features(path)


def test_load_features_zero_row_file_loads_empty_frame(
    tmp_path: Path, valid_features_df: pd.DataFrame
) -> None:
    """A zero-row Parquet file loads and validates cleanly, unlike the
    equivalent CSV case: Parquet stores each column's dtype in its schema, so a
    zero-row file round-trips with the same dtypes as the DataFrame that wrote
    it — there is no header-only ambiguity for pandera's coerce=False check to
    reject the way an all-`object`-dtype CSV read would be."""
    path = tmp_path / "features.parquet"
    valid_features_df.iloc[:0].to_parquet(path, index=False)

    loaded = load_features(path)

    assert loaded.empty
    assert set(valid_features_df.columns).issubset(set(loaded.columns))


def test_load_features_missing_file_raises(tmp_path: Path) -> None:
    """A nonexistent path fails loudly (FileNotFoundError from pandas), not
    with a schema error that would misattribute the cause."""
    with pytest.raises(FileNotFoundError):
        load_features(tmp_path / "does-not-exist.parquet")


# ---------------------------------------------------------------------------
# features_sha256
# ---------------------------------------------------------------------------


def test_features_sha256_matches_manual_hash(
    tmp_path: Path, valid_features_df: pd.DataFrame
) -> None:
    """features_sha256 is exactly hashlib.sha256 over the file's raw bytes —
    no normalization, no re-serialization."""
    path = tmp_path / "features.parquet"
    valid_features_df.to_parquet(path, index=False)

    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert features_sha256(path) == expected


def test_features_sha256_stable_across_repeated_calls(
    tmp_path: Path, valid_features_df: pd.DataFrame
) -> None:
    """Hashing the same unchanged file twice returns the same digest."""
    path = tmp_path / "features.parquet"
    valid_features_df.to_parquet(path, index=False)

    assert features_sha256(path) == features_sha256(path)


def test_features_sha256_changes_when_content_changes(
    tmp_path: Path, valid_features_df: pd.DataFrame
) -> None:
    """A content edit (e.g. a re-run of the feature pipeline on updated raw
    data) changes the digest — the consistency check evaluate.py relies
    on to catch a stale or mismatched features file."""
    path = tmp_path / "features.parquet"
    valid_features_df.to_parquet(path, index=False)
    original_hash = features_sha256(path)

    changed_df = valid_features_df.copy()
    changed_df.loc[0, "monthlycharges"] = 999.0
    changed_df.to_parquet(path, index=False)

    assert features_sha256(path) != original_hash

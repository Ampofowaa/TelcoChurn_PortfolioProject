"""Single accessor for the processed features table — path, format, and content hash.

Reads telco_churn_features.parquet, written by features/build.py's __main__
(`make features`) — the file Phase 8's dvc.yaml will declare as the `features`
stage's output once that wiring lands. Every consumer (train/common.py,
calibrate.py, evaluate.py, threshold.py) reads through load_features() rather
than constructing the path/format inline, so a future format change is a
one-function edit instead of a rewrite across modules.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from telco_churn.features.schema import FeatureOutputSchema
from telco_churn.utils.paths import get_project_root, load_config

__all__ = ["FEATURES_FILENAME", "features_path", "load_features", "features_sha256"]

FEATURES_FILENAME = "telco_churn_features.parquet"


def _default_features_path() -> Path:
    """Return the canonical processed-features path, anchored to the project root."""
    cfg = load_config()
    return Path(get_project_root() / cfg.paths.processed_data) / FEATURES_FILENAME


def features_path(path: Path | None = None) -> Path:
    """Return the processed-features path.

    path defaults to the canonical project location; pass an explicit path in
    tests to avoid touching the real project artifact.
    """
    return path if path is not None else _default_features_path()


def load_features(path: Path | None = None) -> pd.DataFrame:
    """Load and schema-validate the processed features table.

    path defaults to the canonical project location; pass an explicit path in
    tests. Validated against FeatureOutputSchema so a schema-drifted or stale
    file fails loudly here rather than downstream in a consumer.
    """
    df = pd.read_parquet(features_path(path))
    return FeatureOutputSchema.validate(df)


def features_sha256(path: Path | None = None) -> str:
    """Return the sha256 content hash of the processed features file on disk.

    Called by every models/train/* module and evaluate.py to populate
    training_manifest.json's data_content_hash — verifying the same features
    file backs training, calibration, and evaluation within one cycle.
    Independent of DVC's own content hash, which is unavailable until the DVC
    pipeline wrap tracks this file.
    """
    return hashlib.sha256(features_path(path).read_bytes()).hexdigest()

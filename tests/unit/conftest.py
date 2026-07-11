"""Shared fixtures for unit tests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from helpers import make_row

from telco_churn.features.build import TARGET_COL
from telco_churn.models.train.common import _FEATURE_COLS


@pytest.fixture
def mlflow_test_experiment(tmp_path: Path) -> Callable[[str], str]:
    """Factory: point MLflow at a tmp-scoped SQLite store with an explicit
    artifact_location, returning the tracking URI once the named experiment is
    active.

    A SQLite-backed tracking store with no explicit artifact root defaults run
    artifacts to './mlruns' relative to CWD — the repo root under pytest — so
    every models/train/* test that opened a real MLflow run leaked artifacts
    into the tracked mlruns/ directory even though its tracking metadata lived
    in a discarded tmp_path SQLite file. create_experiment(artifact_location=...)
    is required rather than set_experiment(name): set_experiment cannot set an
    artifact_location on an experiment that doesn't exist yet, and by the time
    it exists, the default has already been applied.
    """

    def _make(experiment_name: str) -> str:
        tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
        mlflow.set_tracking_uri(tracking_uri)
        artifact_location = (tmp_path / "artifacts").as_uri()
        experiment_id = mlflow.create_experiment(
            experiment_name, artifact_location=artifact_location
        )
        mlflow.set_experiment(experiment_id=experiment_id)
        return tracking_uri

    return _make


@pytest.fixture
def valid_raw_df() -> pd.DataFrame:
    """Two-row DataFrame with valid values matching the raw Telco schema.

    Rows have distinct totalcharges (358.20 vs 500.00) so imputation tests
    exercise a real median computation rather than a degenerate identical-value case.
    """
    return pd.DataFrame(
        [make_row("1111-AAAAA"), make_row("2222-BBBBB", totalcharges=500.00)]
    )


@pytest.fixture
def zero_tenure_df(valid_raw_df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame including the expected zero-tenure / NULL totalcharges row."""
    extra = make_row("9999-ZZZZZ")
    extra["tenure"] = 0
    extra["totalcharges"] = float("nan")
    return pd.concat([valid_raw_df, pd.DataFrame([extra])], ignore_index=True)


@pytest.fixture
def empty_telco_df() -> pd.DataFrame:
    """Empty DataFrame with the full Telco schema columns."""
    return pd.DataFrame(columns=list(make_row().keys()))


@pytest.fixture
def large_valid_df() -> pd.DataFrame:
    """1 001-row DataFrame to clear the Gate 5 row-count threshold."""
    rows = [make_row(f"cust-{i:04d}") for i in range(1_001)]
    return pd.DataFrame(rows)


_TRAIN_N = 120  # enough rows to stratify comfortably at 20% test size


@pytest.fixture
def feature_df() -> pd.DataFrame:
    """Synthetic feature DataFrame matching _FEATURE_COLS + customerid + churn.

    Shared across the models/train/* unit test modules (candidates, comparison,
    tuning, log_model) — one synthetic frame shape for the whole training
    pipeline test suite.
    """
    rng = np.random.default_rng(0)
    n = _TRAIN_N
    return pd.DataFrame(
        {
            "customerid": [f"cust-{i:04d}" for i in range(n)],
            "gender": rng.choice(["Male", "Female"], size=n),
            "has_partner": rng.choice(["Yes", "No"], size=n),
            "dependents": rng.choice(["Yes", "No"], size=n),
            "phoneservice": rng.choice(["Yes", "No"], size=n),
            "paperlessbilling": rng.choice(["Yes", "No"], size=n),
            "seniorcitizen": rng.integers(0, 2, size=n).tolist(),
            "multiplelines": rng.choice(["Yes", "No", "No phone service"], size=n),
            "internetservice": rng.choice(["DSL", "Fiber optic", "No"], size=n),
            "onlinesecurity": rng.choice(["Yes", "No", "No internet service"], size=n),
            "onlinebackup": rng.choice(["Yes", "No", "No internet service"], size=n),
            "deviceprotection": rng.choice(
                ["Yes", "No", "No internet service"], size=n
            ),
            "techsupport": rng.choice(["Yes", "No", "No internet service"], size=n),
            "streamingtv": rng.choice(["Yes", "No", "No internet service"], size=n),
            "streamingmovies": rng.choice(["Yes", "No", "No internet service"], size=n),
            "contract_type": rng.choice(
                ["Month-to-month", "One year", "Two year"], size=n
            ),
            "paymentmethod": rng.choice(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                size=n,
            ),
            "tenure": rng.integers(0, 73, size=n).tolist(),
            "monthlycharges": rng.uniform(18.25, 118.75, size=n).tolist(),
            "totalcharges": rng.uniform(18.25, 8684.8, size=n).tolist(),
            "charge_per_service": rng.uniform(0.5, 50.0, size=n).tolist(),
            "churn": rng.integers(0, 2, size=n).tolist(),
        }
    )


@pytest.fixture
def dev_split(feature_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """X, y built directly from the synthetic fixture — no split needed for CV tests."""
    return feature_df[_FEATURE_COLS], feature_df[TARGET_COL]

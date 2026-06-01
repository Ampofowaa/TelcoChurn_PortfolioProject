"""Shared fixtures for unit tests."""

from __future__ import annotations

import pandas as pd
import pytest
from helpers import make_row


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

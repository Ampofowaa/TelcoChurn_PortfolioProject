"""Unit tests for telco_churn.data.ingest parsing helpers."""

from pathlib import Path

import pandas as pd
import pytest

from telco_churn.data.ingest import load_raw_csv

_MINIMAL_CSV = """\
customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges,Churn
7590-VHVEG,Female,0,Yes,No,1,No,No phone service,DSL,No,Yes,No,No,No,No,Month-to-month,Yes,Electronic check,29.85,29.85,No
5575-GNVDE,Male,0,No,No,34,Yes,No,DSL,Yes,No,Yes,No,No,No,One year,No,Mailed check,56.95,1889.50,No
3668-QPYBK,Male,0,No,No,2,Yes,No,DSL,Yes,Yes,No,No,No,No,Month-to-month,Yes,Mailed check,53.85,108.15,Yes
zero-tenure,Female,0,No,No,0,No,No phone service,DSL,No,No,No,No,No,No,Month-to-month,Yes,Electronic check,20.00, ,No
"""


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    """Four-row CSV fixture including one zero-tenure whitespace TotalCharges."""
    path = tmp_path / "sample.csv"
    path.write_text(_MINIMAL_CSV)
    return path


def test_load_returns_dataframe(sample_csv: Path) -> None:
    """load_raw_csv returns a DataFrame with the expected row count."""
    df = load_raw_csv(sample_csv)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4


def test_total_charges_whitespace_coerced_to_nan(sample_csv: Path) -> None:
    """Whitespace TotalCharges (zero-tenure row) is coerced to NaN, not dropped."""
    df = load_raw_csv(sample_csv)
    zero_tenure = df.loc[df["customerid"] == "zero-tenure", "totalcharges"]
    assert len(zero_tenure) == 1
    assert pd.isna(zero_tenure.iloc[0])


def test_churn_encoded_as_binary_int(sample_csv: Path) -> None:
    """Churn Yes → 1, No → 0; only values are 0 and 1."""
    df = load_raw_csv(sample_csv)
    assert set(df["churn"].unique()).issubset({0, 1})
    assert df.loc[df["customerid"] == "3668-QPYBK", "churn"].iloc[0] == 1
    assert df.loc[df["customerid"] == "7590-VHVEG", "churn"].iloc[0] == 0


def test_original_churn_column_dropped(sample_csv: Path) -> None:
    """The raw 'Churn' text column is replaced by binary 'churn'."""
    df = load_raw_csv(sample_csv)
    assert "Churn" not in df.columns
    assert "churn" in df.columns


def test_all_columns_lowercase(sample_csv: Path) -> None:
    """All column names are lowercased to avoid quoted-identifier friction in SQL."""
    df = load_raw_csv(sample_csv)
    assert all(col == col.lower() for col in df.columns)


def test_keyword_columns_renamed(sample_csv: Path) -> None:
    """'partner' and 'contract' are renamed to avoid SQL keyword collisions."""
    df = load_raw_csv(sample_csv)
    assert "has_partner" in df.columns
    assert "contract_type" in df.columns
    assert "partner" not in df.columns
    assert "contract" not in df.columns


def test_missing_file_raises(tmp_path: Path) -> None:
    """load_raw_csv raises FileNotFoundError with an actionable message."""
    with pytest.raises(FileNotFoundError, match="make data"):
        load_raw_csv(tmp_path / "does_not_exist.csv")

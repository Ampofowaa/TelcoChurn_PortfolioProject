"""Unit tests for telco_churn.serving.crm_data — pure customers_crm generator (Phase 9 gap-fill)."""

from __future__ import annotations

import pandas as pd
import pytest

from telco_churn.serving.crm_data import CrmGenerationParams, generate_crm_rows
from telco_churn.serving.customer_lookup import LOOKUP_COLUMNS

_PARAMS = CrmGenerationParams(
    random_state=42,
    tenure_advance_min_months=1,
    tenure_advance_max_months=6,
    contract_upgrade_probability=0.5,
    totalcharges_noise_scale=0.02,
)


def _raw_df(n: int = 50) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customerid": [f"cust-{i:04d}" for i in range(n)],
            "gender": ["Female" if i % 2 == 0 else "Male" for i in range(n)],
            "seniorcitizen": [0] * n,
            "has_partner": ["Yes" if i % 3 == 0 else "No" for i in range(n)],
            "dependents": ["No"] * n,
            "tenure": [0 if i < 5 else (i % 70) for i in range(n)],
            "phoneservice": ["Yes"] * n,
            "multiplelines": ["No"] * n,
            "internetservice": [
                "DSL" if i % 2 == 0 else "Fiber optic" for i in range(n)
            ],
            "onlinesecurity": ["No"] * n,
            "onlinebackup": ["Yes"] * n,
            "deviceprotection": ["No"] * n,
            "techsupport": ["No"] * n,
            "streamingtv": ["No"] * n,
            "streamingmovies": ["No"] * n,
            "contract_type": [
                ["Month-to-month", "One year", "Two year"][i % 3] for i in range(n)
            ],
            "paperlessbilling": ["Yes"] * n,
            "paymentmethod": ["Electronic check"] * n,
            "monthlycharges": [20.0 + i for i in range(n)],
            "totalcharges": [
                None if i < 5 else float(20.0 + i) * (i % 70) for i in range(n)
            ],
            "churn": [0] * n,
        }
    )


def test_output_has_customerid_lookup_columns_and_snapshot_timestamp() -> None:
    df = _raw_df()

    out = generate_crm_rows(df, _PARAMS)

    assert set(out.columns) == {"customerid", *LOOKUP_COLUMNS, "crm_snapshot_at"}
    assert "churn" not in out.columns


def test_generation_is_deterministic_under_a_fixed_seed() -> None:
    df = _raw_df()

    out1 = generate_crm_rows(df, _PARAMS).drop(columns=["crm_snapshot_at"])
    out2 = generate_crm_rows(df, _PARAMS).drop(columns=["crm_snapshot_at"])

    pd.testing.assert_frame_equal(out1, out2)


def test_tenure_advances_within_the_configured_bounds() -> None:
    df = _raw_df()

    out = generate_crm_rows(df, _PARAMS)

    delta = out["tenure"].to_numpy() - df["tenure"].to_numpy()
    assert delta.min() >= _PARAMS.tenure_advance_min_months
    assert delta.max() <= _PARAMS.tenure_advance_max_months


def test_contract_never_moves_backward() -> None:
    """Two year -> one year, or any downgrade, must never occur."""
    df = _raw_df()
    rank = {"Month-to-month": 0, "One year": 1, "Two year": 2}

    out = generate_crm_rows(df, _PARAMS)

    old_rank = df["contract_type"].map(rank)
    new_rank = out["contract_type"].map(rank)
    assert (new_rank >= old_rank).all()


def test_zero_tenure_customers_get_a_real_totalcharges() -> None:
    """The 11-zero-tenure-customer NaN case: no longer zero tenure after the
    nudge, so totalcharges must become a real, non-null accrued balance."""
    df = _raw_df()

    out = generate_crm_rows(df, _PARAMS)

    zero_tenure_mask = df["tenure"] == 0
    assert zero_tenure_mask.sum() > 0
    nudged = out.loc[zero_tenure_mask, "totalcharges"]
    assert nudged.notna().all()
    assert (nudged > 0).all()


def test_unrelated_fields_are_held_fixed() -> None:
    df = _raw_df()
    held_fixed = [
        "gender",
        "seniorcitizen",
        "has_partner",
        "dependents",
        "phoneservice",
        "multiplelines",
        "internetservice",
        "onlinesecurity",
        "onlinebackup",
        "deviceprotection",
        "techsupport",
        "streamingtv",
        "streamingmovies",
        "paperlessbilling",
        "paymentmethod",
        "monthlycharges",
    ]

    out = generate_crm_rows(df, _PARAMS)

    for col in held_fixed:
        pd.testing.assert_series_equal(out[col], df[col], check_names=False)


def test_zero_contract_upgrade_probability_never_upgrades() -> None:
    df = _raw_df()
    params = CrmGenerationParams(
        random_state=42,
        tenure_advance_min_months=1,
        tenure_advance_max_months=6,
        contract_upgrade_probability=0.0,
        totalcharges_noise_scale=0.02,
    )

    out = generate_crm_rows(df, params)

    pd.testing.assert_series_equal(
        out["contract_type"], df["contract_type"], check_names=False
    )


@pytest.mark.parametrize("random_state", [1, 2, 3])
def test_a_different_seed_changes_the_nudges(random_state: int) -> None:
    df = _raw_df()
    other_params = CrmGenerationParams(
        random_state=random_state,
        tenure_advance_min_months=1,
        tenure_advance_max_months=6,
        contract_upgrade_probability=0.5,
        totalcharges_noise_scale=0.02,
    )

    out_default = generate_crm_rows(df, _PARAMS)
    out_other = generate_crm_rows(df, other_params)

    assert not out_default["tenure"].equals(out_other["tenure"])

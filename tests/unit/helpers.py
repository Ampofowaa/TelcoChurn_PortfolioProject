"""Shared test utility functions for the unit test suite."""

from __future__ import annotations


def make_row(
    customer_id: str = "1111-AAAAA", totalcharges: float = 358.20
) -> dict[str, object]:
    """Return a valid single-row dict matching the customers_raw schema."""
    return {
        "customerid": customer_id,
        "gender": "Male",
        "seniorcitizen": 0,
        "has_partner": "Yes",
        "dependents": "No",
        "tenure": 12,
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
        "monthlycharges": 29.85,
        "totalcharges": totalcharges,
        "churn": 0,
    }

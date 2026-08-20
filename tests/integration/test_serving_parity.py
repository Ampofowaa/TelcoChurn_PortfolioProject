"""Integration test: serving-parity through the full HTTP path.

tests/unit/test_predict.py::test_predict_single_matches_golden_predictions_json
already pins this at the predict.py function boundary (PredictRequest ->
_assemble_model_input -> ColumnTransformer -> predict_proba), using inline
feature payloads. This file extends the same invariant one layer up, through
the parts that unit test deliberately bypasses: real HTTP request/response
(de)serialization, and POST /predict/batch's ID-only resolution path
(Pydantic validation -> Postgres lookup -> _assemble_model_input ->
predict_proba).

POST /predict/batch's ID-only path no longer resolves against
calibrate.py's frozen golden_predictions.json values — customer_lookup.py
resolves against customers_crm, a seeded "current state" derivation that
deliberately diverges from the training-time snapshot
(prediction_logging_plan.md Part A), so a customerid-only request and the
literal golden feature values are no longer expected to agree bit-for-bit.
What the HTTP/Postgres path must still guarantee is self-consistency: an
ID-only batch request and an inline request built from that same
customer's current GET /customer/{id} row must score identically — anything
else means the Postgres lookup -> _assemble_model_input -> predict_proba
chain silently dropped or altered a value in transit.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from telco_churn.serving.app import app

pytestmark = pytest.mark.integration


def test_predict_batch_id_only_matches_inline_prediction_of_the_same_crm_row(
    serving_env: dict[str, Any], serving_champion: dict[str, Any]
) -> None:
    """ID-only /predict/batch resolution must score identically to
    submitting that same customer's current customers_crm row (as returned
    by GET /customer/{id}) inline via /predict — the Postgres lookup ->
    _assemble_model_input -> predict_proba chain must not silently alter a
    value in transit, even though customers_crm's values themselves
    (deliberately) no longer match calibrate.py's frozen golden reference.
    """
    with TestClient(app) as client:
        id_only_payload = [
            {"customerid": cid} for cid in serving_champion["golden_customerids"]
        ]
        id_only_response = client.post("/predict/batch", json=id_only_payload)
        assert id_only_response.status_code == 200
        id_only_body = id_only_response.json()
        assert id_only_body["errors"] == []
        results_by_id = {r["customerid"]: r for r in id_only_body["results"]}

        for customerid in serving_champion["golden_customerids"]:
            lookup_response = client.get(f"/customer/{customerid}")
            assert lookup_response.status_code == 200
            features = lookup_response.json()["features"]

            inline_response = client.post(
                "/predict", json={**features, "include_explanation": False}
            )
            assert inline_response.status_code == 200

            assert results_by_id[customerid]["probability"] == pytest.approx(
                inline_response.json()["probability"], abs=1e-9
            )
            assert results_by_id[customerid]["model_version"] == (
                inline_response.json()["model_version"]
            )


def test_predict_batch_id_only_decision_matches_served_threshold(
    serving_env: dict[str, Any], serving_champion: dict[str, Any]
) -> None:
    """decision/decision_threshold must agree with the served
    probability/threshold pair for every ID-resolved row — the invariant
    serving/app.py::predict_batch_endpoint's docstring documents (each row's
    eligibility is judged against whichever model actually served it, never
    a value pinned ahead of time).
    """
    payload = [{"customerid": cid} for cid in serving_champion["golden_customerids"]]

    with TestClient(app) as client:
        response = client.post("/predict/batch", json=payload)

    assert response.status_code == 200
    body = response.json()
    for result in body["results"]:
        assert result["threshold"] == pytest.approx(serving_champion["threshold"])
        assert result["decision_threshold"] == result["threshold"]
        assert result["decision"] == (result["probability"] >= result["threshold"])

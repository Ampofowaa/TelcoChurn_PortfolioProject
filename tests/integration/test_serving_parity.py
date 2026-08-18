"""Integration test: serving-parity through the full HTTP path.

tests/unit/test_predict.py::test_predict_single_matches_golden_predictions_json
already pins this at the predict.py function boundary (PredictRequest ->
_assemble_model_input -> ColumnTransformer -> predict_proba). This file
extends the same invariant one layer up, through the parts that unit test
deliberately bypasses: real HTTP request/response (de)serialization, and
POST /predict/batch's ID-only resolution path (Pydantic validation ->
Postgres lookup -> _assemble_model_input -> predict_proba). A served
probability that drifts from calibrate.py's own logged golden_predictions.json
reference at more than cfg.register.golden_atol means something on this
path — not just the model itself — has silently changed the served number.

register.golden_atol (not a locally-chosen tolerance) is the same bound
register.py's own pre-promotion smoke check uses to decide whether a
reloaded model is still the artifact it claims to be (see CLAUDE.md's MLflow
Model Registry section, "Validate, then promote").
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from telco_churn.serving.app import app

pytestmark = pytest.mark.integration


def test_predict_batch_id_only_matches_golden_predictions_at_register_golden_atol(
    serving_env: dict[str, Any], serving_champion: dict[str, Any]
) -> None:
    """Every golden customer, requested by customerid alone (forcing the
    Postgres ID-resolution path, not an inline feature payload), must score
    within cfg.register.golden_atol of calibrate.py's own logged reference —
    the same golden_predictions.json artifact serving_champion (conftest.py)
    logged onto the champion's own run.
    """
    payload = [{"customerid": cid} for cid in serving_champion["golden_customerids"]]
    atol = serving_champion["golden_atol"]

    with TestClient(app) as client:
        response = client.post("/predict/batch", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["errors"] == []
    results_by_id = {r["customerid"]: r for r in body["results"]}

    for customerid, expected_p_hat in zip(
        serving_champion["golden_customerids"],
        serving_champion["golden_p_hat"],
        strict=True,
    ):
        result = results_by_id[customerid]
        assert result["probability"] == pytest.approx(expected_p_hat, abs=atol)
        assert result["model_version"] == serving_champion["model_version"]


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

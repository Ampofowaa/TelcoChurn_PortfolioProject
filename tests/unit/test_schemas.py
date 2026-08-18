"""Unit tests for telco_churn.serving.schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from telco_churn.features.schema import FEATURE_SCHEMA
from telco_churn.serving.schemas import (
    BatchPredictionError,
    BatchPredictionResult,
    BatchPredictItem,
    BatchPredictResponse,
    CustomerFeatures,
    Explanation,
    FeatureContribution,
    PredictRequest,
    PredictResponse,
)

_FULL_FEATURE_ROW = {
    "gender": "Female",
    "seniorcitizen": 0,
    "has_partner": "Yes",
    "dependents": "No",
    "tenure": 5,
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
    "monthlycharges": 70.5,
    "totalcharges": 350.25,
}


def test_customer_features_accepts_a_full_valid_row() -> None:
    features = CustomerFeatures.model_validate(
        {**_FULL_FEATURE_ROW, "customerid": "A-0001"}
    )
    assert features.customerid == "A-0001"
    assert features.gender == "Female"


def test_customer_features_totalcharges_is_nullable() -> None:
    """Zero-tenure customers have no totalcharges yet — must not be forced non-null."""
    features = CustomerFeatures.model_validate(
        {**_FULL_FEATURE_ROW, "totalcharges": None}
    )
    assert features.totalcharges is None


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("gender", "Other"),
        ("has_partner", "Maybe"),
        ("multiplelines", "Sometimes"),
        ("onlinesecurity", "Unknown"),
        ("internetservice", "Satellite"),
        ("contract_type", "Three year"),
        ("paymentmethod", "Cash"),
    ],
)
def test_customer_features_rejects_out_of_domain_categoricals(
    field: str, bad_value: str
) -> None:
    row = {**_FULL_FEATURE_ROW, field: bad_value}
    with pytest.raises(ValidationError):
        CustomerFeatures.model_validate(row)


@pytest.mark.parametrize("field", ["tenure", "monthlycharges", "totalcharges"])
def test_customer_features_rejects_negative_numerics(field: str) -> None:
    row = {**_FULL_FEATURE_ROW, field: -1}
    with pytest.raises(ValidationError):
        CustomerFeatures.model_validate(row)


def test_customer_features_round_trips_into_predict_request() -> None:
    """A GET /customer/{id} response must construct a valid PredictRequest with
    zero remapping — the whole point of sharing one schema.
    """
    looked_up = CustomerFeatures.model_validate(
        {**_FULL_FEATURE_ROW, "customerid": "A-0001"}
    )
    request = PredictRequest.model_validate(looked_up.model_dump())
    assert request.customerid == "A-0001"
    assert request.include_explanation is True


def test_customer_features_field_set_matches_feature_schema_raw_columns() -> None:
    """CustomerFeatures is hand-typed and must not silently drift from FEATURE_SCHEMA.

    charge_per_service is the one FEATURE_SCHEMA column deliberately excluded:
    it is computed server-side by predict.py's _assemble_model_input, never
    supplied by a caller. If a future feature-selection review adds a new raw
    column to FEATURE_SCHEMA/COMMITTED_FEATURES without a matching edit here,
    this catches it before a live request KeyErrors inside _assemble_model_input.
    """
    raw_columns = {
        *FEATURE_SCHEMA.binary,
        *FEATURE_SCHEMA.multi_cat,
        *FEATURE_SCHEMA.numeric,
    } - {"charge_per_service"}
    wire_fields = set(CustomerFeatures.model_fields) - {"customerid"}

    only_in_feature_schema = raw_columns - wire_fields
    only_in_customer_features = wire_fields - raw_columns

    assert not only_in_feature_schema, (
        "Columns in FEATURE_SCHEMA but missing from CustomerFeatures: "
        f"{only_in_feature_schema}"
    )
    assert not only_in_customer_features, (
        "Fields on CustomerFeatures but missing from FEATURE_SCHEMA: "
        f"{only_in_customer_features}"
    )


def test_predict_response_never_carries_contact_or_capacity_limit() -> None:
    response = PredictResponse(
        customerid="A-0001",
        probability=0.5,
        threshold=0.39,
        decision=True,
        model_version="1",
    )
    dumped = response.model_dump()
    assert "contact" not in dumped
    assert "capacity_limit" not in dumped


def test_predict_response_include_explanation_false_omits_explanation_entirely() -> (
    None
):
    response = PredictResponse(
        customerid="A-0001",
        probability=0.5,
        threshold=0.39,
        decision=True,
        model_version="1",
        explanation=None,
    )
    dumped = response.model_dump(exclude_none=True)
    assert "explanation" not in dumped


def test_predict_response_includes_explanation_when_present() -> None:
    explanation = Explanation(
        base_value=-0.2,
        top_features=[
            FeatureContribution(feature="tenure", shap_value=0.3, customer_value=5)
        ],
    )
    response = PredictResponse(
        customerid="A-0001",
        probability=0.5,
        threshold=0.39,
        decision=True,
        model_version="1",
        explanation=explanation,
    )
    dumped = response.model_dump(exclude_none=True)
    assert dumped["explanation"]["top_features"][0]["feature"] == "tenure"


def test_batch_predict_item_validates_id_only_shape() -> None:
    item = BatchPredictItem.model_validate({"customerid": "A-0001"})
    assert item.customerid == "A-0001"
    assert item.gender is None


def test_batch_predict_item_validates_inline_features_shape() -> None:
    item = BatchPredictItem.model_validate(_FULL_FEATURE_ROW)
    assert item.customerid is None
    assert item.gender == "Female"


def test_batch_predict_item_validates_id_with_partial_override_shape() -> None:
    item = BatchPredictItem.model_validate(
        {"customerid": "A-0001", "monthlycharges": 55.0}
    )
    assert item.customerid == "A-0001"
    assert item.monthlycharges == 55.0
    assert item.gender is None


def test_batch_predict_item_still_rejects_out_of_domain_categoricals() -> None:
    with pytest.raises(ValidationError):
        BatchPredictItem.model_validate({"customerid": "A-0001", "gender": "Other"})


def test_batch_predict_response_always_carries_capacity_limit() -> None:
    response = BatchPredictResponse(
        capacity_limit=3,
        results=[
            BatchPredictionResult(
                index=0,
                customerid="A-0001",
                probability=0.9,
                threshold=0.39,
                decision_threshold=0.39,
                decision=True,
                contact=True,
                model_version="1",
                served_source="champion",
            )
        ],
        errors=[
            BatchPredictionError(
                index=1,
                customerid=None,
                reason="validation_error",
                detail="gender: not a valid enumeration member",
            )
        ],
    )
    dumped = response.model_dump()
    assert dumped["capacity_limit"] == 3
    assert dumped["results"][0]["contact"] is True
    assert dumped["errors"][0]["reason"] == "validation_error"


def test_batch_prediction_error_reason_is_restricted_to_the_two_known_classes() -> None:
    with pytest.raises(ValidationError):
        BatchPredictionError(
            index=0, customerid=None, reason="something_else", detail="oops"
        )

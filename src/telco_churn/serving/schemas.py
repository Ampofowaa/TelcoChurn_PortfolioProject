"""Pydantic v2 request/response models for the serving API.

Field-level constraints mirror features/schema.py's Pandera CustomerFeaturesSchema
so the wire contract and the training-time validation contract cannot silently
diverge. The Yes/No-family fields import their allowed-value sets directly from
data/schema.py (YES_NO, YES_NO_NO_PHONE, YES_NO_NO_INTERNET) rather than
re-typing them; gender/internetservice/contract_type/paymentmethod have no such
shared constant to import (Pandera itself declares them inline), so those stay
Literal types here, matching Pandera's own inline-literal treatment.

CustomerFeatures is the single 19-raw-field-plus-customerid shape used both as
PredictRequest's base and as the GET /customer/{customerid} response model —
one schema, not two independently maintained ones, so a lookup response can be
round-tripped straight into POST /predict with no remapping.

BatchPredictItem is a distinct, all-optional mirror of the same fields: a batch
item may arrive as an ID-only reference, a full inline feature set, or an ID
with a partial override, and CustomerFeatures's required fields cannot express
that. The categorical isin-checks are shared between the two models via
_CategoricalFieldValidators rather than duplicated.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from telco_churn.data.schema import YES_NO, YES_NO_NO_INTERNET, YES_NO_NO_PHONE

__all__ = [
    "CustomerFeatures",
    "CustomerLookupResponse",
    "PredictRequest",
    "FeatureContribution",
    "Explanation",
    "PredictResponse",
    "BatchPredictItem",
    "BatchPredictionResult",
    "BatchPredictionError",
    "BatchPredictResponse",
]

_INTERNET_SERVICE = Literal["DSL", "Fiber optic", "No"]
_CONTRACT_TYPE = Literal["Month-to-month", "One year", "Two year"]
_PAYMENT_METHOD = Literal[
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]


def _validate_isin(value: str, allowed: frozenset[str], field_name: str) -> str:
    """Raise unless value is a member of allowed; used by the shared field validators below."""
    if value not in allowed:
        raise ValueError(f"{field_name} must be one of {sorted(allowed)}")
    return value


class _CategoricalFieldValidators:
    """Mixin providing the Yes/No-family isin checks shared by CustomerFeatures
    and BatchPredictItem. Not a BaseModel itself — picked up via MRO when a
    concrete model class combining this mixin with BaseModel is built, so the
    same validator logic runs against both the required-field and
    all-optional-field variants without being written twice. Each validator
    passes None through unchanged, since BatchPredictItem's fields are
    Optional and an absent field carries no value to check.
    """

    @field_validator(
        "has_partner",
        "dependents",
        "phoneservice",
        "paperlessbilling",
        check_fields=False,
    )
    @classmethod
    def _validate_yes_no(cls, v: str | None, info: ValidationInfo) -> str | None:
        return v if v is None else _validate_isin(v, YES_NO, str(info.field_name))

    @field_validator("multiplelines", check_fields=False)
    @classmethod
    def _validate_multiplelines(cls, v: str | None) -> str | None:
        return v if v is None else _validate_isin(v, YES_NO_NO_PHONE, "multiplelines")

    @field_validator(
        "onlinesecurity",
        "onlinebackup",
        "deviceprotection",
        "techsupport",
        "streamingtv",
        "streamingmovies",
        check_fields=False,
    )
    @classmethod
    def _validate_yes_no_no_internet(
        cls, v: str | None, info: ValidationInfo
    ) -> str | None:
        return (
            v
            if v is None
            else _validate_isin(v, YES_NO_NO_INTERNET, str(info.field_name))
        )


class CustomerFeatures(_CategoricalFieldValidators, BaseModel):
    """A customer's 19 raw feature values, plus a pass-through customerid.

    customerid is a request/response identifier, never a model feature — it
    rides alongside the 19 declared fields without being fed into
    predict_proba, the same pass-through separation
    features/schema.py::CustomerFeaturesSchema already enforces for the
    training-time DataFrame. Optional because a prospective customer scored
    in manual/what-if mode has no customerid yet.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    customerid: str | None = None
    gender: Literal["Male", "Female"]
    seniorcitizen: Literal[0, 1]
    has_partner: str
    dependents: str
    tenure: int = Field(ge=0)
    phoneservice: str
    multiplelines: str
    internetservice: _INTERNET_SERVICE
    onlinesecurity: str
    onlinebackup: str
    deviceprotection: str
    techsupport: str
    streamingtv: str
    streamingmovies: str
    contract_type: _CONTRACT_TYPE
    paperlessbilling: str
    paymentmethod: _PAYMENT_METHOD
    monthlycharges: float = Field(ge=0.0)
    totalcharges: float | None = Field(default=None, ge=0.0)


class CustomerLookupResponse(BaseModel):
    """GET /customer/{customerid}'s response: the looked-up feature set plus
    provenance, not bare CustomerFeatures.

    crm_snapshot_at stays off CustomerFeatures itself — that schema is the
    model's actual feature contract, and a provenance timestamp isn't a
    feature. Wrapping it here instead is what lets the Streamlit Lookup tab
    say something genuinely true ("CRM snapshot as of ...") about a row that
    now comes from customers_crm, not the frozen training snapshot.
    """

    features: CustomerFeatures
    crm_snapshot_at: datetime


class PredictRequest(CustomerFeatures):
    """POST /predict's request body: CustomerFeatures plus two fields that
    are request options, not customer attributes — neither is ever fed into
    predict_proba.
    """

    include_explanation: bool = True
    # /predict/batch reports a third value, "partial_override" — Score a
    # Customer never does: detecting "fetched, then edited" here would need
    # client-side diff logic purely to populate a label, for a distinction
    # not worth the cost. A bare API caller
    # that never sets this leaves it None rather than defaulting to a guess.
    resolution_kind: Literal["id_only", "full_inline"] | None = None


class FeatureContribution(BaseModel):
    """One feature's contribution to a single local SHAP explanation."""

    feature: str
    shap_value: float
    customer_value: Any


class Explanation(BaseModel):
    """A single prediction's local SHAP breakdown: base_value plus the top-K
    contributing features by absolute magnitude.

    Reconstructs the champion's pre-calibration score
    (base_value + sum(shap_value)), not the calibrated probability returned
    alongside it — TreeExplainer cannot see through CalibratedClassifierCV's
    sigmoid/isotonic map.
    """

    base_value: float
    top_features: list[FeatureContribution]


class PredictResponse(BaseModel):
    """POST /predict's response: probability, threshold, decision — no
    contact/capacity_limit, since a single customer in isolation always fits
    under any capacity >= 1 and has no population to be ranked against.
    """

    customerid: str | None
    probability: float
    threshold: float
    decision: bool
    model_version: str
    explanation: Explanation | None = None


class BatchPredictItem(_CategoricalFieldValidators, BaseModel):
    """One row of a POST /predict/batch request body.

    Every field optional so the same model validates all three item shapes:
    ID-only ({"customerid": "..."}), full inline features (no customerid
    needed), or an ID with a partial feature override. Resolving which shape
    a given item is — and looking up Postgres when fields are missing — is
    the caller's (serving/predict.py's) job; this model only accepts the
    wire shape.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    customerid: str | None = None
    gender: Literal["Male", "Female"] | None = None
    seniorcitizen: Literal[0, 1] | None = None
    has_partner: str | None = None
    dependents: str | None = None
    tenure: int | None = Field(default=None, ge=0)
    phoneservice: str | None = None
    multiplelines: str | None = None
    internetservice: _INTERNET_SERVICE | None = None
    onlinesecurity: str | None = None
    onlinebackup: str | None = None
    deviceprotection: str | None = None
    techsupport: str | None = None
    streamingtv: str | None = None
    streamingmovies: str | None = None
    contract_type: _CONTRACT_TYPE | None = None
    paperlessbilling: str | None = None
    paymentmethod: _PAYMENT_METHOD | None = None
    monthlycharges: float | None = Field(default=None, ge=0.0)
    totalcharges: float | None = Field(default=None, ge=0.0)


class BatchPredictionResult(BaseModel):
    """One scored row of a POST /predict/batch response.

    Carries both decision (model+threshold flag) and contact (whether the
    capacity-constrained cycle actually has room), never one field
    reinterpreted per branch — decision never changes meaning, and contact is
    the only field capacity ever touches.

    threshold and decision_threshold are always equal — both are whichever
    model actually served this row's own calibrated bar (matches
    predict.py::ScoredBatch.served_threshold). Kept as two separate fields
    rather than collapsed into one: threshold is a per-row diagnostic,
    decision_threshold documents specifically what decision was evaluated
    against, and a future change could reintroduce a divergence between them
    without a wire-format break. decision/contact are computed per row
    against that row's own served threshold, not pinned to the champion's
    policy regardless of which model produced probability for a given row —
    see serving/app.py::predict_batch_endpoint. contact_capacity is still
    one campaign-wide pool shared by champion- and challenger-routed rows —
    see ANALYSIS.md §9, item 14 for the capacity-interference limitation
    that remains.

    served_source is "champion" for every row unless canary actually routed
    that row to the challenger (predict.py::ScoredBatch.served_source) —
    shadow mode never changes it, since shadow only dual-scores for
    comparison/logging and never serves the challenger's opinion. Without
    this field, model_version alone can't tell a reader which model a mixed
    canary batch's rows came from without cross-referencing Model Info.
    """

    index: int
    customerid: str | None
    probability: float
    threshold: float
    decision_threshold: float
    decision: bool
    contact: bool
    model_version: str
    served_source: str


class BatchPredictionError(BaseModel):
    """One failed row of a POST /predict/batch response.

    reason distinguishes "malformed" (validation_error) from "well-formed but
    unresolvable" (resolution_error — a supplied customerid with no matching
    Postgres row) — different failure classes a caller debugging a batch
    needs to tell apart.
    """

    index: int
    customerid: str | None
    reason: Literal["validation_error", "resolution_error"]
    detail: str


class BatchPredictResponse(BaseModel):
    """POST /predict/batch's response: partial success, not fail-the-whole-batch.

    capacity_limit is always present (informational even when it didn't
    bind), unlike PredictResponse, which never carries it at all.
    """

    capacity_limit: int
    results: list[BatchPredictionResult]
    errors: list[BatchPredictionError]

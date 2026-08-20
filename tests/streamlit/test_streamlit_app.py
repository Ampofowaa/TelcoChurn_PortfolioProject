"""Headless AppTest smoke test for telco_churn.ui.streamlit_app.

Scoped to CLAUDE.md's UI-testing note: this is smoke coverage only (the app
imports and runs without exception; the Lookup tab renders against a mocked
GET /customer/{id} response; the Manual tab's widget set matches
features/schema.py's constraint sets), not a substitute for exercising the
tabs in a real browser before the phase is called done.

No real HTTP or MLflow calls: requests.request is stubbed per test (the
sole HTTP boundary streamlit_app.py calls through), and
resolve_champion_version is stubbed to None (cold start) so the "About this
model" tab — which every tab renders on each run, st.tabs bodies are not
conditionally executed — degrades gracefully instead of touching a real
MLflow tracking server.
"""

from __future__ import annotations

from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

from telco_churn.features.schema import CustomerFeaturesSchema
from telco_churn.serving.schemas import CustomerFeatures
from telco_churn.utils.paths import get_project_root

_APP_PATH = str(get_project_root() / "src" / "telco_churn" / "ui" / "streamlit_app.py")
# Generous: the first run in a fresh process pays for importing streamlit,
# altair, mlflow, and composing Hydra's config tree — AppTest's own 3s
# default is tuned for an already-warm process, not this cold start.
_RUN_TIMEOUT_SECONDS = 60

_CUSTOMER_ROW: dict[str, Any] = {
    "customerid": "7590-VHVEG",
    "gender": "Female",
    "seniorcitizen": 0,
    "has_partner": "Yes",
    "dependents": "No",
    "tenure": 1,
    "phoneservice": "No",
    "multiplelines": "No phone service",
    "internetservice": "DSL",
    "onlinesecurity": "No",
    "onlinebackup": "Yes",
    "deviceprotection": "No",
    "techsupport": "No",
    "streamingtv": "No",
    "streamingmovies": "No",
    "contract_type": "Month-to-month",
    "paperlessbilling": "Yes",
    "paymentmethod": "Electronic check",
    "monthlycharges": 29.85,
    "totalcharges": 29.85,
}

_CRM_SNAPSHOT_AT = "2026-08-15T00:00:00+00:00"
# The UI reformats this to match utils/logging.py's structlog timestamp
# style ("YYYY-MM-DD HH:MM:SS UTC") — see streamlit_app.py::_format_crm_snapshot.
_CRM_SNAPSHOT_AT_DISPLAY = "2026-08-15 00:00:00 UTC"
# GET /customer/{id}'s actual response shape: CustomerLookupResponse{features,
# crm_snapshot_at} — never the bare CustomerFeatures dict _CUSTOMER_ROW is,
# since customers_crm (prediction_logging_plan.md Part A) always carries a
# provenance timestamp alongside the feature values.
_CUSTOMER_LOOKUP_RESPONSE: dict[str, Any] = {
    "features": _CUSTOMER_ROW,
    "crm_snapshot_at": _CRM_SNAPSHOT_AT,
}


class _FakeResponse:
    """Minimal requests.Response stand-in — only the surface streamlit_app.py reads."""

    def __init__(self, status_code: int, payload: Any) -> None:
        self.status_code = status_code
        self._payload = payload

    @property
    def ok(self) -> bool:
        return self.status_code < 400

    @property
    def text(self) -> str:
        return str(self._payload)

    def json(self) -> Any:
        return self._payload


@pytest.fixture(autouse=True)
def _stub_champion_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "telco_churn.models.artifacts.resolve_champion_version", lambda cfg: None
    )


def test_app_runs_without_exception() -> None:
    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=_RUN_TIMEOUT_SECONDS)
    assert not at.exception


def test_lookup_tab_prefills_from_mocked_customer_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_request(method: str, url: str, **kwargs: Any) -> _FakeResponse:
        assert method == "GET"
        assert url.endswith(f"/customer/{_CUSTOMER_ROW['customerid']}")
        return _FakeResponse(200, _CUSTOMER_LOOKUP_RESPONSE)

    monkeypatch.setattr("requests.request", fake_request)

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=_RUN_TIMEOUT_SECONDS)
    at.text_input(key="lookup_customerid").input(_CUSTOMER_ROW["customerid"])
    at.button(key="lookup_fetch").click().run(timeout=_RUN_TIMEOUT_SECONDS)

    assert not at.exception
    assert at.success[0].value == (
        f"Loaded customer {_CUSTOMER_ROW['customerid']} "
        f"(CRM snapshot as of {_CRM_SNAPSHOT_AT_DISPLAY})"
    )
    # A successful fetch bumps the form's key-rotation nonce (see
    # _render_lookup_tab's comment on why: a same-key widget can survive a
    # programmatic overwrite in a real browser even though the state-level
    # value is correct), so the field keys are no longer the bare
    # "lookup_<field>" — discover the live ones rather than hardcoding a
    # specific nonce value.
    gender_widget = next(w for w in at.selectbox if w.key.endswith("_gender"))
    tenure_widget = next(w for w in at.number_input if w.key.endswith("_tenure"))
    assert gender_widget.value == _CUSTOMER_ROW["gender"]
    assert tenure_widget.value == _CUSTOMER_ROW["tenure"]


def test_lookup_tab_shows_warning_on_unknown_customer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "requests.request",
        lambda method, url, **kwargs: _FakeResponse(404, "not found"),
    )

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=_RUN_TIMEOUT_SECONDS)
    at.text_input(key="lookup_customerid").input("no-such-id")
    at.button(key="lookup_fetch").click().run(timeout=_RUN_TIMEOUT_SECONDS)

    assert not at.exception
    assert any("no-such-id" in w.value for w in at.warning)


def test_lookup_tab_fetch_works_without_a_prior_commit_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Typing an ID and clicking Fetch in the same interaction (no separate
    Enter/blur run first) must still fetch — regresses the `disabled=not
    customerid` bug, where the button stayed disabled (and so un-clickable)
    until a prior rerun had already committed the typed value.
    """

    def fake_request(method: str, url: str, **kwargs: Any) -> _FakeResponse:
        return _FakeResponse(200, _CUSTOMER_LOOKUP_RESPONSE)

    monkeypatch.setattr("requests.request", fake_request)

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=_RUN_TIMEOUT_SECONDS)
    at.text_input(key="lookup_customerid").input(_CUSTOMER_ROW["customerid"])
    at.button(key="lookup_fetch").click().run(timeout=_RUN_TIMEOUT_SECONDS)

    assert not at.exception
    assert at.success[0].value == (
        f"Loaded customer {_CUSTOMER_ROW['customerid']} "
        f"(CRM snapshot as of {_CRM_SNAPSHOT_AT_DISPLAY})"
    )


def test_lookup_tab_fetch_with_blank_customerid_warns_instead_of_erroring() -> None:
    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=_RUN_TIMEOUT_SECONDS)
    at.button(key="lookup_fetch").click().run(timeout=_RUN_TIMEOUT_SECONDS)

    assert not at.exception
    assert any("Enter a customer ID" in w.value for w in at.warning)


def test_lookup_tab_clear_resets_the_customerid_box(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The customer ID box must come back empty after Clear — regressed by
    a text_input whose rendered value can survive a session_state pop +
    rerun in a real browser even though the state-level value reports "".
    The fix (a reset-counter-suffixed key) forces a fresh widget, which
    this test checks for directly rather than trusting `.value` alone.
    """

    def fake_request(method: str, url: str, **kwargs: Any) -> _FakeResponse:
        return _FakeResponse(200, _CUSTOMER_LOOKUP_RESPONSE)

    monkeypatch.setattr("requests.request", fake_request)

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=_RUN_TIMEOUT_SECONDS)
    at.text_input(key="lookup_customerid").input(_CUSTOMER_ROW["customerid"])
    at.button(key="lookup_fetch").click().run(timeout=_RUN_TIMEOUT_SECONDS)

    at.button(key="lookup_clear").click().run(timeout=_RUN_TIMEOUT_SECONDS)

    assert not at.exception
    assert "lookup_prefill" not in at.session_state
    remaining_keys = {w.key for w in at.text_input}
    assert "lookup_customerid" not in remaining_keys
    new_key = next(k for k in remaining_keys if k.startswith("lookup_customerid_"))
    assert at.text_input(key=new_key).value == ""


def test_score_tab_widget_options_match_schema_constraint_sets() -> None:
    """The failure mode this guards: a hand-typed option list in the UI
    silently drifting from features/schema.py's own Pandera constraint,
    producing a dropdown that lets a user submit a value the API 422s on.
    """
    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=_RUN_TIMEOUT_SECONDS)
    assert not at.exception

    schema = CustomerFeaturesSchema.to_schema()

    def isin_options(field: str) -> set[str]:
        for check in schema.columns[field].checks:
            allowed = getattr(check, "statistics", {}).get("allowed_values")
            if allowed:
                return {str(v) for v in allowed}
        raise AssertionError(f"{field!r} has no isin constraint")

    for field in ("gender", "internetservice", "contract_type", "paymentmethod"):
        widget = at.selectbox(key=f"lookup_{field}")
        assert set(widget.options) == isin_options(field)

    for field in ("has_partner", "dependents", "phoneservice", "paperlessbilling"):
        widget = at.selectbox(key=f"lookup_{field}")
        assert set(widget.options) == {"Yes", "No"}

    assert set(at.selectbox(key="lookup_multiplelines").options) == {
        "Yes",
        "No",
        "No phone service",
    }
    for field in (
        "onlinesecurity",
        "onlinebackup",
        "deviceprotection",
        "techsupport",
        "streamingtv",
        "streamingmovies",
    ):
        assert set(at.selectbox(key=f"lookup_{field}").options) == {
            "Yes",
            "No",
            "No internet service",
        }


def test_score_tab_renders_every_raw_field() -> None:
    """Every one of CustomerFeatures's 19 raw fields (all but customerid,
    the pass-through, and charge_per_service, which is never a user input)
    is represented by a widget. `WidgetList.__call__(key=...)` itself raises
    KeyError on a missing key and Streamlit raises on a duplicate one (which
    test_app_runs_without_exception already guards), so existence here is
    what needs asserting — uniqueness comes for free from those two.
    """
    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=_RUN_TIMEOUT_SECONDS)
    assert not at.exception

    raw_fields = {
        name for name in CustomerFeatures.model_fields if name != "customerid"
    }
    selectbox_fields = raw_fields - {
        "seniorcitizen",
        "tenure",
        "monthlycharges",
        "totalcharges",
    }
    for field in selectbox_fields:
        assert at.selectbox(key=f"lookup_{field}") is not None, field

    assert at.toggle(key="lookup_seniorcitizen") is not None
    for field in ("tenure", "monthlycharges", "totalcharges"):
        assert at.number_input(key=f"lookup_{field}") is not None, field


def test_bulk_drill_into_customer_sends_a_flat_predict_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test: _drill_into_prediction's GET /customer/{id} call
    returns CustomerLookupResponse{features, crm_snapshot_at} (Part A), not
    bare CustomerFeatures — the payload it builds for POST /predict must
    unwrap ["features"] first. Previously it spread the whole response in
    (`{**resp.json(), "include_explanation": True}`), producing
    `{"features": {...}, "crm_snapshot_at": ..., "include_explanation": true}`
    — a shape PredictRequest rejects with 422 "field required" on every real
    field, since none of them exist at the top level.
    """
    predict_payloads: list[dict[str, Any]] = []

    def fake_request(method: str, url: str, **kwargs: Any) -> _FakeResponse:
        customerid = _CUSTOMER_ROW["customerid"]
        if method == "POST" and url.endswith("/predict/batch"):
            return _FakeResponse(
                200,
                {
                    "capacity_limit": 10,
                    "results": [
                        {
                            "index": 0,
                            "customerid": customerid,
                            "probability": 0.3,
                            "threshold": 0.4,
                            "decision_threshold": 0.4,
                            "decision": False,
                            "contact": False,
                            "model_version": "1",
                            "served_source": "champion",
                        }
                    ],
                    "errors": [],
                },
            )
        if method == "GET" and url.endswith(f"/customer/{customerid}"):
            return _FakeResponse(200, _CUSTOMER_LOOKUP_RESPONSE)
        if method == "POST" and url.endswith("/predict"):
            payload = kwargs.get("json", {})
            predict_payloads.append(payload)
            return _FakeResponse(
                200,
                {
                    "customerid": payload.get("customerid"),
                    "probability": 0.3,
                    "threshold": 0.4,
                    "decision": False,
                    "model_version": "1",
                    "explanation": None,
                },
            )
        raise AssertionError(f"unexpected request: {method} {url}")

    monkeypatch.setattr("requests.request", fake_request)

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=_RUN_TIMEOUT_SECONDS)

    csv_bytes = f"customerid\n{_CUSTOMER_ROW['customerid']}\n".encode()
    at.file_uploader(key="bulk_csv").set_value(("customers.csv", csv_bytes, "text/csv"))
    at.run(timeout=_RUN_TIMEOUT_SECONDS)

    at.button(key="bulk_score").click().run(timeout=_RUN_TIMEOUT_SECONDS)
    assert not at.exception

    at.button(key="bulk_drill_button").click().run(timeout=_RUN_TIMEOUT_SECONDS)

    assert not at.exception
    assert len(predict_payloads) == 1
    sent = predict_payloads[0]
    assert "features" not in sent, f"payload must be flattened, not wrapped: {sent}"
    assert sent["gender"] == _CUSTOMER_ROW["gender"]
    assert sent["tenure"] == _CUSTOMER_ROW["tenure"]

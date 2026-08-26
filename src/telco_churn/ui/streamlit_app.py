"""Streamlit UI — a thin client over the FastAPI serving app (serving/app.py).

No model, preprocessor, or database is loaded in this process: every
prediction/lookup goes over HTTP to `API_BASE_URL` (defaulting to
`http://localhost:8000` for a bare `streamlit run` outside Compose;
`docker-compose.yml` sets it to `http://fastapi:8000` for the containerised
pairing). The one exception is the "Model Info" tab, which reads
`registration/model_card.json` directly off the champion's MLflow run — the
same artifact-by-run-id resolution every other stage in this project uses,
never a second copy proxied through the API.

One form over the same 19 raw-input fields, plus two supporting
capabilities: **Score a Customer** (default) — `GET /customer/{customerid}`
optionally prefills the form for a real account, and every field stays
editable either way. A former second tab let a blank form stand in for a
prospect who hasn't signed up yet; that reused a model trained and tested
only on real customers' observed histories to answer a genuinely different,
unvalidated question, so a blank/no-match form is now explicitly flagged as
a hypothetical scenario rather than presented as an equally-valid
prediction (see the in-form caveat in `_render_lookup_tab`) — manual entry
for a *real* account that isn't in the lookup system yet is still fully
supported, that's what the caveat's tenure check excludes. Calls
`POST /predict` and renders probability + SHAP. **Batch Prediction** maps
parsed CSV rows into `POST /predict/batch`'s three item shapes
client-side — no second batch-validation implementation. **Model Info**
renders the champion's `model_card.json` as readable sections, fetched on
demand (a button, not an eager call) since `st.tabs` runs every tab's body
on every rerun regardless of which is active.

Widget option lists are read off `features/schema.py`'s Pandera constraints
(`CustomerFeaturesSchema`) and `data/schema.py`'s shared Yes/No-family
constants, never re-typed here — the same schema-as-source-of-truth
discipline `serving/schemas.py` already applies to the wire contract.
"""

from __future__ import annotations

import json
import os
import re
from datetime import date, datetime
from typing import Any

import altair as alt
import mlflow
import mlflow.artifacts
import pandas as pd
import requests
import streamlit as st

from telco_churn.data.schema import YES_NO, YES_NO_NO_INTERNET, YES_NO_NO_PHONE
from telco_churn.features.schema import CustomerFeaturesSchema
from telco_churn.models.artifacts import resolve_champion_version
from telco_churn.serving.schemas import CustomerFeatures
from telco_churn.utils.mlflow import resolve_model_run_id, resolve_tracking_uri
from telco_churn.utils.paths import activate_config, compose_config

__all__ = ["API_BASE_URL", "main"]

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
# Display-only — API_BASE_URL is `http://fastapi:8000` in Compose (Docker's
# internal service DNS), reachable from this process but not from the host
# browser rendering the page. API_DISPLAY_URL is the host-facing equivalent
# (docker-compose.yml sets it to `http://localhost:${API_PORT}`, matching
# that service's published port mapping) so the "Serving API" caption below
# is an actual clickable link instead of a dead one. Falls back to
# API_BASE_URL itself, which is already host-reachable outside Compose (a
# bare `streamlit run` against a bare `uvicorn` run, both on localhost).
_API_DISPLAY_URL = os.environ.get("API_DISPLAY_URL", API_BASE_URL)
_REQUEST_TIMEOUT_SECONDS = 30.0
# Mirrors configs/serving/api.yaml's poll_interval_seconds default — the
# champion this tab reads about hot-reloads on roughly that cadence, so this
# poll shouldn't lag it by much. Only gates the cheap "which version is
# champion" registry lookup — the model_card.json artifact fetch itself is
# cached per-version with no TTL (see _load_model_card_for_version).
_CHAMPION_VERSION_POLL_TTL_SECONDS = 30

# blue<->red diverging pair (dataviz palette): red for SHAP contributions
# that push the score toward churn, blue for contributions that push away.
_SHAP_INCREASE_COLOR = "#e34948"
_SHAP_DECREASE_COLOR = "#2a78d6"

_YES_NO_FIELDS = ("has_partner", "dependents", "phoneservice")
_YES_NO_NO_PHONE_FIELDS = ("multiplelines",)
_YES_NO_NO_INTERNET_FIELDS = (
    "onlinesecurity",
    "onlinebackup",
    "deviceprotection",
    "techsupport",
    "streamingtv",
    "streamingmovies",
)
# Human-readable labels, keyed by raw field name — the single translation
# table shared by the form widgets' labels and the SHAP bar chart's feature
# names, so a label only ever needs updating in one place.
_FIELD_LABELS: dict[str, str] = {
    "gender": "Gender",
    "seniorcitizen": "Senior citizen",
    "has_partner": "Has partner",
    "dependents": "Dependents",
    "tenure": "Tenure (months)",
    "phoneservice": "Phone service",
    "multiplelines": "Multiple lines",
    "internetservice": "Internet service",
    "onlinesecurity": "Online security",
    "onlinebackup": "Online backup",
    "deviceprotection": "Device protection",
    "techsupport": "Tech support",
    "streamingtv": "Streaming TV",
    "streamingmovies": "Streaming movies",
    "contract_type": "Contract type",
    "paperlessbilling": "Paperless billing",
    "paymentmethod": "Payment method",
    "monthlycharges": "Monthly charges ($)",
    "totalcharges": "Total charges ($)",
    "charge_per_service": "Charge per service ($)",
}


# ---------------------------------------------------------------------------
# Schema-sourced widget options
# ---------------------------------------------------------------------------


def _isin_options(field_name: str) -> list[str]:
    """Allowed values for `field_name`, read off CustomerFeaturesSchema's own
    isin constraint rather than a hand-typed option list.
    """
    schema = CustomerFeaturesSchema.to_schema()
    for check in schema.columns[field_name].checks:
        allowed = getattr(check, "statistics", {}).get("allowed_values")
        if allowed:
            return sorted(str(v) for v in allowed)
    raise ValueError(f"{field_name!r} has no isin constraint in CustomerFeaturesSchema")


def _select_index(options: list[str], value: Any) -> int:
    """Index of `value` in `options`, or 0 when absent (no prefill, or an
    unrecognized value)."""
    try:
        return options.index(str(value))
    except ValueError:
        return 0


# ---------------------------------------------------------------------------
# HTTP boundary
# ---------------------------------------------------------------------------


def _api_request(method: str, path: str, **kwargs: Any) -> requests.Response | None:
    """The sole HTTP call site the rest of this module goes through.

    Surfaces a connection failure as a Streamlit error and returns None
    rather than raising, so every caller has one uniform "did this work"
    check instead of a try/except at each of the four call sites.
    """
    try:
        return requests.request(
            method, f"{API_BASE_URL}{path}", timeout=_REQUEST_TIMEOUT_SECONDS, **kwargs
        )
    except requests.RequestException as exc:
        st.error(f"Could not reach the API at {API_BASE_URL}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Shared form + prediction rendering (Score a Customer tab)
# ---------------------------------------------------------------------------


def render_customer_form(
    key_prefix: str, prefill: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Render the 19-raw-field customer form and return the current widget values.

    Three columns, grouped by theme rather than raw field count (7/7/5 —
    identity+account / internet+add-ons / contract+billing) — narrower per
    field than a 2-column split (each column claims a third of the page
    instead of half, so a "Yes"/"No" selectbox isn't stretched across ~450px
    of mostly empty space) and shorter overall (a 7-row tallest column
    instead of an 11-row one), which also brings the Predict button — always
    rendered immediately below this call — closer to the top of the tab.

    `charge_per_service` (the 20th, SQL-engineered feature) gets no widget —
    it is never a user input in either entry mode, only ever computed by
    build_feature_df/compute_charge_per_service.
    """
    prefill = prefill or {}

    def default(field: str, fallback: Any) -> Any:
        value = prefill.get(field)
        return fallback if value is None else value

    values: dict[str, Any] = {}
    col1, col2, col3 = st.columns(3)

    with col1:
        gender_opts = _isin_options("gender")
        values["gender"] = st.selectbox(
            _FIELD_LABELS["gender"],
            gender_opts,
            index=_select_index(gender_opts, default("gender", "Female")),
            key=f"{key_prefix}_gender",
        )
        values["seniorcitizen"] = int(
            st.toggle(
                _FIELD_LABELS["seniorcitizen"],
                value=bool(default("seniorcitizen", 0)),
                key=f"{key_prefix}_seniorcitizen",
            )
        )
        for field in _YES_NO_FIELDS:
            opts = sorted(YES_NO)
            values[field] = st.selectbox(
                _FIELD_LABELS[field],
                opts,
                index=_select_index(opts, default(field, "No")),
                key=f"{key_prefix}_{field}",
            )
        values["tenure"] = int(
            st.number_input(
                _FIELD_LABELS["tenure"],
                min_value=0,
                step=1,
                value=int(default("tenure", 0)),
                key=f"{key_prefix}_tenure",
            )
        )
        for field in _YES_NO_NO_PHONE_FIELDS:
            opts = sorted(YES_NO_NO_PHONE)
            values[field] = st.selectbox(
                _FIELD_LABELS[field],
                opts,
                index=_select_index(opts, default(field, "No phone service")),
                key=f"{key_prefix}_{field}",
            )

    with col2:
        internet_opts = _isin_options("internetservice")
        values["internetservice"] = st.selectbox(
            _FIELD_LABELS["internetservice"],
            internet_opts,
            index=_select_index(internet_opts, default("internetservice", "No")),
            key=f"{key_prefix}_internetservice",
        )
        for field in _YES_NO_NO_INTERNET_FIELDS:
            opts = sorted(YES_NO_NO_INTERNET)
            values[field] = st.selectbox(
                _FIELD_LABELS[field],
                opts,
                index=_select_index(opts, default(field, "No internet service")),
                key=f"{key_prefix}_{field}",
            )

    with col3:
        contract_opts = _isin_options("contract_type")
        values["contract_type"] = st.selectbox(
            _FIELD_LABELS["contract_type"],
            contract_opts,
            index=_select_index(
                contract_opts, default("contract_type", "Month-to-month")
            ),
            key=f"{key_prefix}_contract_type",
        )
        opts = sorted(YES_NO)
        values["paperlessbilling"] = st.selectbox(
            _FIELD_LABELS["paperlessbilling"],
            opts,
            index=_select_index(opts, default("paperlessbilling", "No")),
            key=f"{key_prefix}_paperlessbilling",
        )
        payment_opts = _isin_options("paymentmethod")
        values["paymentmethod"] = st.selectbox(
            _FIELD_LABELS["paymentmethod"],
            payment_opts,
            index=_select_index(
                payment_opts, default("paymentmethod", "Electronic check")
            ),
            key=f"{key_prefix}_paymentmethod",
        )
        monthly_default = float(default("monthlycharges", 0.0))
        values["monthlycharges"] = float(
            st.number_input(
                _FIELD_LABELS["monthlycharges"],
                min_value=0.0,
                value=monthly_default,
                key=f"{key_prefix}_monthlycharges",
            )
        )
        total_prefill = prefill.get("totalcharges")
        total_default = (
            float(total_prefill)
            if total_prefill is not None
            else round(monthly_default * max(int(default("tenure", 0)), 1), 2)
        )
        values["totalcharges"] = float(
            st.number_input(
                _FIELD_LABELS["totalcharges"],
                min_value=0.0,
                value=total_default,
                key=f"{key_prefix}_totalcharges",
                help=(
                    "Defaults to an estimate (monthly charge x tenure) when "
                    "no account was fetched — edit if an actual billed "
                    "total is known."
                ),
            )
        )
    return values


def _parse_transformed_name(transformed_name: str) -> tuple[str, str]:
    """Split a ColumnTransformer output name into (raw field, one-hot level).

    e.g. 'multi_cat__contract_type_Month-to-month' -> ('contract_type', 'Month-to-month');
    'numeric__tenure' -> ('tenure', '') — no one-hot level for a numeric column.
    An unrecognized remainder (no matching raw field) comes back as (remainder, '').
    """
    remainder = transformed_name.split("__", 1)[-1]
    matches = [
        field
        for field in _FIELD_LABELS
        if remainder == field or remainder.startswith(field + "_")
    ]
    if not matches:
        return remainder, ""
    field = max(matches, key=len)
    return field, remainder[len(field) + 1 :]


def _humanize_feature_name(transformed_name: str) -> str:
    """Translate a ColumnTransformer output name (e.g. 'multi_cat__contract_type_Month-to-month')
    into a retention-agent-readable label ('Contract type: Month-to-month').

    Display concern only — the API's explanation payload stays the faithful,
    typed record of what the model actually saw (see this module's docstring).
    """
    field, level = _parse_transformed_name(transformed_name)
    label = _FIELD_LABELS.get(field, field)
    if not level:
        return label
    if field == "seniorcitizen" and level in ("0", "1"):
        level = "Yes" if level == "1" else "No"
    return f"{label}: {level}"


def _display_value(field: str, raw_value: Any) -> str:
    """The customer's actual value for `field`, in the same words the form uses."""
    if field == "seniorcitizen":
        return "Yes" if int(raw_value) == 1 else "No"
    return str(raw_value)


def _impact_label(relative_magnitude: float) -> str:
    """Coarse High/Medium/Low bucket, relative to the largest factor in this
    customer's own explanation. SHAP units are a pre-calibration log-odds
    scale with no meaningful absolute reading for a retail audience — a
    within-chart relative comparison is the only one that means anything
    without a global reference distribution.
    """
    if relative_magnitude >= 0.66:
        return "High impact"
    if relative_magnitude >= 0.33:
        return "Medium impact"
    return "Low impact"


def _render_shap_chart(explanation: dict[str, Any], raw_values: dict[str, Any]) -> None:
    """One bar per raw customer attribute.

    A multi-valued field (e.g. `contract_type`) becomes several independent
    one-hot columns in `top_features` — showing each as its own bar reads as
    contradictory to a non-technical viewer (e.g. "Month-to-month" and
    "Two year" both listed at once). Grouping every column that traces back
    to the same raw field into one bar, labelled with the value this
    customer actually has (from the submitted form, not the encoded 0/1),
    removes that ambiguity. SHAP's decimal units carry no meaningful scale
    for a retail audience, so the axis is hidden in favour of a coarse
    High/Medium/Low badge relative to the largest factor shown here.
    """
    rows = explanation.get("top_features", [])
    if not rows:
        return

    grouped: dict[str, float] = {}
    order: list[str] = []
    for r in rows:
        field, _level = _parse_transformed_name(r["feature"])
        if field not in grouped:
            grouped[field] = 0.0
            order.append(field)
        grouped[field] += r["shap_value"]

    def label_for(field: str) -> str:
        base = _FIELD_LABELS.get(field, field)
        raw_value = raw_values.get(field)
        if raw_value is None:
            return base
        return f"{base}: {_display_value(field, raw_value)}"

    values = [grouped[field] for field in order]
    max_abs = max((abs(v) for v in values), default=0.0) or 1.0

    chart_df = pd.DataFrame(
        {
            "feature": [label_for(field) for field in order],
            "shap_value": values,
            "direction": [
                "Increases risk" if v >= 0 else "Decreases risk" for v in values
            ],
            "impact": [_impact_label(abs(v) / max_abs) for v in values],
        }
    ).sort_values("shap_value")

    # labelLimit=0 (Vega-Lite's "no limit") — the default ~100px cap was
    # truncating labels like "Contract type: Month-to-month" mid-word ("Mo…"),
    # which also silently ate the impact badge whenever it had been appended
    # to the same string (always the last part to fit, so always the first
    # cut). The domain is padded 30% past the longest bar so the impact
    # label — now its own text mark, not string-concatenated — has room to
    # render past the bar's tip instead of being clipped at the plot edge.
    base = alt.Chart(chart_df).encode(
        x=alt.X(
            "shap_value:Q",
            axis=None,
            scale=alt.Scale(domain=[-max_abs * 1.3, max_abs * 1.3]),
        ),
        y=alt.Y("feature:N", sort=None, title=None, axis=alt.Axis(labelLimit=0)),
    )
    bars = base.mark_bar().encode(
        color=alt.Color(
            "direction:N",
            scale=alt.Scale(
                domain=["Increases risk", "Decreases risk"],
                range=[_SHAP_INCREASE_COLOR, _SHAP_DECREASE_COLOR],
            ),
            legend=alt.Legend(title=None),
        ),
        tooltip=[
            alt.Tooltip("feature:N", title="Factor"),
            alt.Tooltip("direction:N", title="Direction"),
            alt.Tooltip("impact:N", title="Relative impact"),
        ],
    )
    # `align` is a static mark property in this Altair version, not a
    # conditional `.encode()` channel (removed from the generated schema —
    # `.encode(align=...)` now raises `TypeError: unexpected keyword
    # argument 'align'`), so the two label directions are split into two
    # sign-filtered layers instead of one conditionally-aligned one.
    labels_positive = (
        base.transform_filter("datum.shap_value >= 0")
        .mark_text(color="#c9ced6", align="left", dx=5)
        .encode(text="impact:N")
    )
    labels_negative = (
        base.transform_filter("datum.shap_value < 0")
        .mark_text(color="#c9ced6", align="right", dx=-5)
        .encode(text="impact:N")
    )
    st.altair_chart(bars + labels_positive + labels_negative, use_container_width=True)
    st.caption(
        "Bar length shows how strongly each factor pushed this prediction, "
        "not an exact amount — read it as a ranked list of reasons, not a "
        "precise breakdown."
    )


def _render_prediction_response(
    data: dict[str, Any], raw_values: dict[str, Any]
) -> None:
    probability = float(data["probability"])
    threshold = float(data["threshold"])
    decision = bool(data["decision"])
    col_probability, col_risk = st.columns(2)
    with col_probability:
        st.metric("Churn probability", f"{probability:.1%}")
    with col_risk:
        st.metric("Risk level", _risk_label(probability, threshold))
    st.progress(min(max(probability, 0.0), 1.0))
    st.write("**Contact recommended**" if decision else "**No action needed**")
    explanation = data.get("explanation")
    if explanation:
        _render_shap_chart(explanation, raw_values)


def _run_single_prediction(payload: dict[str, Any]) -> None:
    """`payload` doubles as the raw-value source for `_render_shap_chart`'s
    per-attribute labels (e.g. "Contract type: Month-to-month") — it already
    carries every raw field the form submitted, so no second lookup is needed.
    """
    resp = _api_request("POST", "/predict", json=payload)
    if resp is None:
        return
    if not resp.ok:
        st.error(f"API error {resp.status_code}: {resp.text}")
        return
    _render_prediction_response(resp.json(), payload)


# ---------------------------------------------------------------------------
# Tab 1 — Score a Customer (default)
# ---------------------------------------------------------------------------


def _format_crm_snapshot(iso_timestamp: str) -> str:
    """Render crm_snapshot_at as 'YYYY-MM-DD HH:MM:SS UTC' — the same format
    utils/logging.py's structlog TimeStamper stamps every ingest/crm_data
    log line with, so a timestamp reads the same whether it came from a
    terminal or this UI. Falls back to the raw ISO string on a parse
    failure; a display nicety is never worth breaking the page over.
    """
    try:
        return datetime.fromisoformat(iso_timestamp).strftime("%Y-%m-%d %H:%M:%S UTC")
    except ValueError:
        return iso_timestamp


def _render_lookup_tab() -> None:
    """One form, two ways in: fetch a real customer by ID to prefill it, or
    fill it in yourself. Manual entry stays legitimate for a real account
    the lookup system doesn't have yet — what changes is that the caveat
    below now says so explicitly instead of a second tab silently treating
    a hand-typed profile as an equally-valid prediction. Editing a
    *fetched* customer's fields is a different, already-legitimate what-if
    (a real relationship's plan hypothetically changed, not a relationship
    that never existed), so it gets no caveat regardless of what's edited.
    """
    # No st.subheader here: the tab bar already labels this "Score a
    # Customer" — restating it would just push the actual instruction
    # further down the page for no added clarity.
    st.caption(
        "Enter a Customer ID and click Fetch to load their current record, "
        "or fill in the form yourself. Click Predict for a churn risk "
        "score and explanation. This never changes the customer's "
        "account details."
    )
    # `key` incorporates a reset counter, bumped on both Clear and a
    # successful Fetch. Deleting a widget's session-state entry (or simply
    # never touching it) and passing a new value=/index= reports the new
    # value correctly at the state level, but a widget's rendered value can
    # still survive that in the browser — the frontend keeps its own copy
    # and doesn't always resync a same-key widget it already rendered once.
    # Confirmed the hard way: fetching a *second* customer into an
    # already-rendered form left stale fields showing (and fed the SHAP
    # chart stale inputs too, since it explains whatever `form_values`
    # currently holds). Changing the key forces old widgets to be torn down
    # and fresh ones mounted, which is the reliable fix — same trick as the
    # customer-ID field below, just applied to the whole form. Two
    # independent nonces because they bump on different events: the ID
    # field must not remount on Fetch (the user just typed that value
    # directly; only overwriting a widget programmatically needs a
    # remount), but the form fields must.
    id_nonce = st.session_state.get("lookup_id_nonce", 0)
    customerid_key = (
        "lookup_customerid" if id_nonce == 0 else f"lookup_customerid_{id_nonce}"
    )
    customerid = st.text_input("Customer ID (optional)", key=customerid_key)
    col_fetch, col_clear = st.columns([1, 1])
    with col_fetch:
        # Never `disabled=not customerid`: that disables the button using
        # this run's value, before the browser has sent up what was just
        # typed — Streamlit only applies that keystroke on the next rerun,
        # so a disabled button silently swallows the click that would have
        # triggered it, and the user has to hit Enter/Tab first to "unstick"
        # it. Validating after the click removes that dead click entirely.
        fetch_clicked = st.button("Fetch customer", key="lookup_fetch")
    with col_clear:
        clear_clicked = st.button("Clear", key="lookup_clear")

    form_nonce = st.session_state.get("lookup_form_nonce", 0)
    form_key_prefix = "lookup" if form_nonce == 0 else f"lookup_{form_nonce}"

    if clear_clicked:
        st.session_state.pop("lookup_prefill", None)
        st.session_state["lookup_id_nonce"] = id_nonce + 1
        st.session_state["lookup_form_nonce"] = form_nonce + 1
        st.rerun()

    if fetch_clicked:
        if not customerid:
            st.warning("Enter a customer ID to fetch.")
        else:
            resp = _api_request("GET", f"/customer/{customerid}")
            if resp is None:
                pass
            elif resp.status_code == 404:
                st.warning(f"No customer found with ID '{customerid}'.")
                st.session_state.pop("lookup_prefill", None)
            elif resp.ok:
                st.session_state["lookup_prefill"] = resp.json()
                st.session_state["lookup_form_nonce"] = form_nonce + 1
                st.rerun()
            else:
                st.error(f"API error {resp.status_code}: {resp.text}")

    lookup_response = st.session_state.get("lookup_prefill")
    prefill = lookup_response.get("features") if lookup_response else None
    if lookup_response is not None and prefill:
        st.success(
            f"Loaded customer {prefill.get('customerid')} "
            f"(CRM snapshot as of "
            f"{_format_crm_snapshot(lookup_response.get('crm_snapshot_at'))})"
        )

    form_values = render_customer_form(form_key_prefix, prefill=prefill)

    # Zero tenure with no fetched account is still a real-shaped state (the
    # training data has zero-tenure customers — a brand-new real signup
    # looks exactly like this). A nonzero tenure with no fetched account has
    # no real referent: nobody has "8 months on this plan" without a real
    # relationship on file, so that combination is answering a counterfactual
    # the model's validation never covered, not predicting a real instance.
    if not prefill and int(form_values.get("tenure", 0)) > 0:
        st.info(
            "No matching account loaded, and tenure is set above 0 — this "
            "scores a hypothetical scenario, not a validated prediction. "
            "The model was trained and tested on real customers' actual "
            "histories, not hand-typed combinations of feature values."
        )

    if st.button("Predict", key="lookup_predict"):
        payload = {
            **form_values,
            "customerid": (prefill.get("customerid") if prefill else customerid)
            or None,
            "include_explanation": True,
            # id_only whenever a Fetch preceded this submission — regardless
            # of whether the fetched fields were then edited — full_inline
            # for manual entry with no prior Fetch. No "partial_override"
            # here: Score a Customer never detects "fetched, then edited";
            # the edited values are still fully captured in
            # feature_snapshot either way.
            "resolution_kind": (
                "id_only" if lookup_response is not None else "full_inline"
            ),
        }
        _run_single_prediction(payload)


# ---------------------------------------------------------------------------
# Bulk CSV upload — a capability, not a third entry mode
# ---------------------------------------------------------------------------


def _normalize_bulk_csv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase headers and apply the same two renames data/ingest.py's
    load_raw_csv applies to the raw IBM CSV (partner -> has_partner,
    contract -> contract_type), so an upload of the original Telco file
    format works with no manual remapping. churn is dropped if present —
    it's the historical label, never a model input.
    """
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    df = df.rename(columns={"partner": "has_partner", "contract": "contract_type"})
    return df.drop(columns=[c for c in ("churn",) if c in df.columns])


def _rows_from_bulk_csv(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Map each parsed row into whichever of /predict/batch's three item
    shapes its columns happen to fill — an ID-only column set, full inline
    features, or an ID with a partial override all fall out of the same
    "keep only known columns, drop empty cells" rule.
    """
    known_columns = [name for name in CustomerFeatures.model_fields]
    present = [c for c in df.columns if c in known_columns]
    trimmed = df[present]
    records: list[dict[str, Any]] = json.loads(trimmed.to_json(orient="records"))
    return records


def _drill_into_prediction(
    customerid: str | None, submitted_row: dict[str, Any]
) -> None:
    """Re-score one row from a scored batch via single /predict — reusing
    Lookup's exact rendering path rather than a second, batch-shaped
    explanation feature (/predict/batch never computes SHAP).
    """
    if customerid:
        resp = _api_request("GET", f"/customer/{customerid}")
        if resp is not None and resp.ok:
            features = resp.json().get("features", {})
            payload = {**features, "include_explanation": True}
            _run_single_prediction(payload)
            return
    payload = {**submitted_row, "customerid": customerid, "include_explanation": True}
    _run_single_prediction(payload)


def _risk_label(probability: float, threshold: float) -> str:
    """Coarse High/Medium/Low bucket, anchored to this row's own served
    threshold `t*` rather than a fixed global split.

    `t*` is the cost-optimal cut (§0) — everyone at or above it is already
    "worth contacting" by the model's own economics, so "Low" is defined as
    exactly the not-flagged population (`probability < threshold`, which
    always agrees with `decision=False`). Medium/High then split the flagged
    population in half, purely as a within-that-population degree-of-risk
    read — both always mean "at or above t*," never "maybe." A fixed
    global split (e.g. `_impact_label`'s 0.66/0.33) would let a customer
    sitting just above `t*` land in "Medium" while `decision=True`, reading
    as equivocal about a call the model already made.
    """
    if probability < threshold:
        return "Low"
    midpoint = threshold + (1.0 - threshold) / 2
    if probability < midpoint:
        return "Medium"
    return "High"


def _bulk_results_display_df(
    results: list[dict[str, Any]], show_diagnostics: bool
) -> pd.DataFrame:
    """Analyst-facing columns by default — plain ID, probability as a
    percentage, a risk label, and the actual contact decision. The raw
    per-row model internals (both threshold columns, the `decision` flag
    before capacity, `model_version`) are a data-scientist diagnostic view,
    opted into rather than shown by default — see the discussion this
    responds to: a marketing analyst has no use for `decision_threshold`.
    """
    if show_diagnostics:
        return pd.DataFrame(results)
    return pd.DataFrame(
        {
            "Customer ID": [r["customerid"] or f"Row {r['index']}" for r in results],
            "Churn Probability": [f"{r['probability']:.0%}" for r in results],
            "Risk Level": [
                _risk_label(r["probability"], r["threshold"]) for r in results
            ],
            "Contact This Cycle": ["Yes" if r["contact"] else "No" for r in results],
        }
    )


def _bulk_drill_options(
    results: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Selector label per scored row: the customer ID alone whenever it
    uniquely identifies one row — disambiguated with the row number only for
    the two cases that actually need it, a missing ID (an inline-feature row
    with no `customerid`) or a repeated one (the same account scored twice,
    e.g. under different what-if inputs). Row numbers are noise for the
    common case, so they no longer appear unconditionally.
    """
    id_counts: dict[str | None, int] = {}
    for r in results:
        id_counts[r["customerid"]] = id_counts.get(r["customerid"], 0) + 1

    options: dict[str, dict[str, Any]] = {}
    for r in results:
        customerid = r["customerid"]
        label = (
            customerid
            if customerid is not None and id_counts[customerid] == 1
            else f"{customerid or '(no id)'} (row {r['index']})"
        )
        options[label] = r
    return options


def _render_bulk_tab() -> None:
    st.subheader("Bulk CSV upload")
    st.caption(
        "Upload a CSV of customers to score in one call. Each row can be: "
        "just a `customerid` (their current details are looked up "
        "automatically), a fully filled-in profile (no `customerid` "
        "needed), or a `customerid` plus a few fields to override — useful "
        "for testing what happens if a specific real customer's plan "
        "changed. Rows of different types can be mixed in the same file. "
        "Capped at the API's configured batch size."
    )
    uploaded = st.file_uploader("CSV file", type=["csv"], key="bulk_csv")
    if uploaded is None:
        return

    try:
        raw_df = pd.read_csv(uploaded)
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        st.error(f"Could not parse the uploaded CSV: {exc}")
        return

    normalized = _normalize_bulk_csv_columns(raw_df)
    rows = _rows_from_bulk_csv(normalized)
    max_size = int(_get_cfg().serving.batch.max_size)
    over_limit = len(rows) > max_size
    if over_limit:
        st.error(
            f"{len(rows)} row(s) parsed — exceeds the API's batch limit of "
            f"{max_size}. Split the file and upload in smaller batches."
        )
    else:
        st.write(f"{len(rows)} row(s) parsed (limit: {max_size}).")

    if st.button("Score batch", key="bulk_score", disabled=not rows or over_limit):
        resp = _api_request("POST", "/predict/batch", json=rows)
        if resp is None:
            pass
        elif not resp.ok:
            st.error(f"API error {resp.status_code}: {resp.text}")
        else:
            st.session_state["bulk_result"] = resp.json()
            st.session_state["bulk_rows"] = rows
            st.session_state["bulk_scored_at"] = date.today().isoformat()

    data = st.session_state.get("bulk_result")
    if not data:
        return

    scored_at = st.session_state.get("bulk_scored_at", date.today().isoformat())

    st.metric("Capacity limit this cycle", data["capacity_limit"])
    results = data.get("results", [])
    if results:
        st.markdown("##### Results")
        show_diagnostics = st.checkbox(
            "Show model diagnostics (probability threshold, raw decision "
            "flag, model version)",
            key="bulk_show_diagnostics",
        )
        results_df = _bulk_results_display_df(results, show_diagnostics)
        st.dataframe(results_df, hide_index=True, use_container_width=True)
        download_df = results_df.drop(columns=["index"], errors="ignore")
        st.download_button(
            "Download results (CSV)",
            data=download_df.to_csv(index=False).encode("utf-8"),
            file_name=f"batch_predictions_{scored_at}.csv",
            mime="text/csv",
            key="bulk_download_results",
        )
        capacity_excluded = [r for r in results if r["decision"] and not r["contact"]]
        if capacity_excluded:
            detail = (
                " — see rows where decision is true and contact is false."
                if show_diagnostics
                else "."
            )
            st.caption(
                f"{len(capacity_excluded)} customer(s) were flagged by the "
                "model but not contacted this cycle (capacity reached)"
                f"{detail}"
            )

    errors = data.get("errors", [])
    if errors:
        st.markdown("##### Errors")
        errors_df = pd.DataFrame(errors)
        st.dataframe(errors_df, hide_index=True, use_container_width=True)
        st.download_button(
            "Download errors (CSV)",
            data=errors_df.to_csv(index=False).encode("utf-8"),
            file_name=f"batch_prediction_errors_{scored_at}.csv",
            mime="text/csv",
            key="bulk_download_errors",
        )

    if not results:
        return
    st.markdown("##### Drill into one customer")
    submitted_rows: list[dict[str, Any]] = st.session_state.get("bulk_rows", [])
    options = _bulk_drill_options(results)
    choice = st.selectbox(
        "Select a scored row", list(options.keys()), key="bulk_drill_choice"
    )
    if st.button("View explanation", key="bulk_drill_button"):
        chosen = options[choice]
        submitted_row = submitted_rows[chosen["index"]]
        _drill_into_prediction(chosen["customerid"], submitted_row)


# ---------------------------------------------------------------------------
# About this model
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def _get_cfg() -> Any:
    cfg = compose_config()
    activate_config(cfg)
    return cfg


@st.cache_data(ttl=_CHAMPION_VERSION_POLL_TTL_SECONDS, show_spinner=False)
def _resolve_champion_version_cached() -> str | None:
    """Current `champion` alias's version, re-polled every cache window.

    Deliberately split from `_load_model_card_for_version`: this is a cheap
    registry lookup (no artifact fetch), so polling it on a short TTL costs
    nothing even on the cycles where the champion hasn't changed — never a
    fixed local path, so a champion rollback is reflected here without
    redeploying the UI (drift_reference.json's same resolve-through-the-alias
    discipline).
    """
    cfg = _get_cfg()
    mlflow.set_tracking_uri(resolve_tracking_uri(str(cfg.mlflow.tracking_uri)))
    return resolve_champion_version(cfg)


@st.cache_data(show_spinner="Loading model card...")
def _load_model_card_for_version(version: str) -> dict[str, Any] | None:
    """`model_card.json` for one specific champion version.

    Cached with no TTL: `model_card.json` is written once at promotion and
    never changes for a given version, so keying the cache on `version`
    alone is enough — a champion rollback resolves to a different version
    and naturally busts the cache, with no timer needed.
    """
    cfg = _get_cfg()
    run_id = resolve_model_run_id(version, cfg)
    try:
        card: dict[str, Any] = mlflow.artifacts.load_dict(
            f"runs:/{run_id}/registration/model_card.json"
        )
    except Exception:
        return None
    return card


# model_card.json's prose mixes stakeholder narrative with sentences
# addressed to an engineer with repo/MLflow access ("See ANALYSIS.md §7...",
# "See technical_appendix.error_concentration..."). Those are dead pointers
# for anyone on this page — stripped here rather than at the source, since
# the underlying card stays the correct, complete artifact for whoever does
# have that access (register.py's own spec, untouched).
_INTERNAL_REFERENCE_PATTERN = re.compile(
    r"\s*See (?:ANALYSIS\.md|README\.md|technical_appendix|training_manifest\.json|"
    r"business_case\.[\w.]+|risks_and_limitations)[^.]*\."
)


def _strip_internal_references(text: str) -> str:
    return _INTERNAL_REFERENCE_PATTERN.sub("", text).strip()


# register.py's versioned internal names (technical_appendix.fairness_
# robustness.flags — v1/v2/v2b/v3 track ANALYSIS.md §0's guardrail
# numbering) for the reader of this tab, who has no reason to know what
# "v2" refers to. Falls back to the raw key for any future flag this map
# doesn't yet cover, rather than raising.
_FAIRNESS_FLAG_LABELS = {
    "v1_segment_collapse_flagged": "segment performance collapse",
    "v2_equal_opportunity_flagged": "equal opportunity gap",
    "v2_demographic_parity_flagged": "demographic parity gap",
    "v2b_calibration_flagged": "per-group calibration collapse",
    "v3_direction_sanity_violations": "direction-sanity violation",
}


def _render_about_model_tab() -> None:
    """Gated behind a button rather than fetched unconditionally.

    `st.tabs` renders every tab's body on every script run, active tab or
    not — so an eager fetch here would cost a champion-alias lookup plus a
    model_card.json download (63KB over HTTP from MLflow) on every single
    page load and interaction, for a tab most sessions never open. Once
    `about_model_requested` is set, it stays set for the session (`st.button`
    only returns True on the run it was clicked), so subsequent reruns keep
    the section visible without re-prompting — the underlying calls are
    `st.cache_data`-cached anyway, so re-rendering after the first load costs
    nothing extra.
    """
    st.subheader("About the current deployed model")

    requested = st.session_state.get("about_model_requested", False)
    if not requested:
        st.caption(
            "Loads governance details from MLflow on demand, not on every page render."
        )
        if st.button("Load model details", key="about_model_load"):
            st.session_state["about_model_requested"] = True
        else:
            return

    try:
        version = _resolve_champion_version_cached()
    except Exception as exc:  # MLflow unreachable — a real, expected failure mode
        st.error(f"Could not reach MLflow to resolve the champion model: {exc}")
        return

    if version is None:
        st.info("No champion model is currently registered.")
        return

    try:
        card = _load_model_card_for_version(version)
    except Exception as exc:  # MLflow unreachable — a real, expected failure mode
        st.error(f"Could not reach MLflow to resolve the champion model: {exc}")
        return

    if card is None:
        st.warning(f"Model card not available for champion version {version}.")
        return

    # Curated for this tab's actual audience — retention/marketing staff and
    # portfolio reviewers, per the card's own `who_uses_it` field — neither
    # of whom can act on a raw fairness-score JSON blob or a precision/
    # recall table at five decision thresholds. That detail stays in the
    # underlying model_card.json artifact (and MLflow) for whoever has the
    # access and the reason to want it; this view answers "what does it do,
    # is it worth trusting, what should I watch out for."
    model_details = card.get("governance", {}).get("model_details", {})
    st.caption(
        f"Deployed model version {model_details.get('version', version)} — "
        f"promoted {model_details.get('promoted_at', 'unknown date')}"
    )

    business_case = card.get("business_case", {})

    st.markdown("#### Model Purpose")
    st.write(business_case.get("why_this_exists", ""))
    st.write(
        _strip_internal_references(card.get("executive_summary", "Not available."))
    )
    # what_it_predicts's technical framing (label definition, dataset
    # column name) mostly restates executive_summary in denser, more
    # dataset-specific language — kept in the underlying model_card.json
    # for whoever reads the raw artifact, but not repeated here. The one
    # operationally useful fact in it (customer-level, not household;
    # scored on a recurring cycle) survives as its own light-touch note.
    st.caption(
        "One score per customer per scoring cycle, not per household or account."
    )

    st.markdown("#### Console Users")
    st.write(business_case.get("who_uses_it", "Not available."))

    st.markdown("#### Expected Business Impact")
    base_scenario = (
        business_case.get("expected_impact", {}).get("scenarios", {}).get("base")
    )
    if base_scenario:
        col1, col2, col3 = st.columns(3)
        col1.metric("Projected net benefit", f"${base_scenario['ev']:,.0f}")
        col2.metric("Customers contacted", f"{base_scenario['contact_rate']:.0%}")
        col3.metric("Campaign cost", f"${base_scenario['campaign_cost']:,.0f}")
        st.caption(
            "Under the base cost scenario — see the executive summary above "
            "for the model's catch rate and precision at this threshold."
        )
    else:
        st.caption("Not available for this promotion cycle.")

    st.markdown("#### Known Limitations")
    limitations = card.get("risks_and_limitations", {}).get("known_limitations", {})
    for key, text in limitations.items():
        st.write(
            f"**{key.replace('_', ' ').capitalize()}:** {_strip_internal_references(text)}"
        )

    st.markdown("#### Fairness Check")
    flags = (
        card.get("technical_appendix", {})
        .get("fairness_robustness", {})
        .get("flags", {})
    )
    flagged = [k for k, v in flags.items() if v]
    if not flags:
        st.caption("Not available for this promotion cycle.")
    elif flagged:
        readable = ", ".join(_FAIRNESS_FLAG_LABELS.get(k, k) for k in flagged)
        # Bold plain text, not st.warning: a colored/smaller-font alert box
        # reads as "something's wrong," but this is a reviewed, documented
        # finding (ANALYSIS.md §7 — judged consistent with real churn-risk
        # differences across these groups, not model bias, and tracked
        # rather than blocking), not an operational error. Emphasis without
        # the alarm styling.
        st.write(f"**Potential fairness gaps flagged in: {readable}.**")
    else:
        st.write("**No fairness gaps flagged across the segments checked.**")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Telco Churn Retention Console", layout="wide")
    st.title("Telco Churn Retention Console")
    st.caption(f"Serving API: [{_API_DISPLAY_URL}/docs]({_API_DISPLAY_URL}/docs)")

    tab_lookup, tab_bulk, tab_about = st.tabs(
        ["Score a Customer", "Batch Prediction", "Model Info"]
    )
    with tab_lookup:
        _render_lookup_tab()
    with tab_bulk:
        _render_bulk_tab()
    with tab_about:
        _render_about_model_tab()


main()

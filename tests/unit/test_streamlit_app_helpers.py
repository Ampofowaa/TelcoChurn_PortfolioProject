"""Unit tests for telco_churn.ui.streamlit_app's pure display-formatting
helpers (no Streamlit ScriptRunContext needed — see tests/streamlit/ for the
headless AppTest smoke suite that exercises the app end to end).
"""

from __future__ import annotations

import pytest

from telco_churn.ui.streamlit_app import (
    _bulk_drill_options,
    _bulk_results_display_df,
    _display_value,
    _humanize_feature_name,
    _impact_label,
    _parse_transformed_name,
    _risk_label,
)


@pytest.mark.parametrize(
    ("transformed_name", "expected"),
    [
        (
            "multi_cat__contract_type_Month-to-month",
            ("contract_type", "Month-to-month"),
        ),
        ("multi_cat__contract_type_Two year", ("contract_type", "Two year")),
        ("numeric__tenure", ("tenure", "")),
        ("binary__seniorcitizen_1", ("seniorcitizen", "1")),
        (
            "unknown_prefix__totally_unrecognized_column",
            ("totally_unrecognized_column", ""),
        ),
    ],
)
def test_parse_transformed_name_splits_field_and_level(
    transformed_name: str, expected: tuple[str, str]
) -> None:
    assert _parse_transformed_name(transformed_name) == expected


def test_humanize_feature_name_translates_seniorcitizen_binary_level() -> None:
    assert _humanize_feature_name("binary__seniorcitizen_1") == "Senior citizen: Yes"
    assert _humanize_feature_name("binary__seniorcitizen_0") == "Senior citizen: No"


def test_humanize_feature_name_falls_back_to_raw_remainder_when_unmatched() -> None:
    assert _humanize_feature_name("unknown_prefix__mystery_col") == "mystery_col"


def test_display_value_translates_seniorcitizen_int_to_yes_no() -> None:
    assert _display_value("seniorcitizen", 1) == "Yes"
    assert _display_value("seniorcitizen", 0) == "No"


def test_display_value_passes_other_fields_through_as_str() -> None:
    assert _display_value("contract_type", "Month-to-month") == "Month-to-month"
    assert _display_value("tenure", 12) == "12"


@pytest.mark.parametrize(
    ("relative_magnitude", "expected"),
    [
        (1.0, "High impact"),
        (0.66, "High impact"),
        (0.5, "Medium impact"),
        (0.1, "Low impact"),
    ],
)
def test_impact_label_buckets_by_relative_magnitude(
    relative_magnitude: float, expected: str
) -> None:
    assert _impact_label(relative_magnitude) == expected


@pytest.mark.parametrize(
    ("probability", "threshold", "expected"),
    [
        (0.3, 0.4, "Low"),
        (0.39, 0.4, "Low"),
        (0.4, 0.4, "Medium"),
        (0.69, 0.4, "Medium"),
        (0.7, 0.4, "High"),
        (1.0, 0.4, "High"),
    ],
)
def test_risk_label_buckets_relative_to_served_threshold(
    probability: float, threshold: float, expected: str
) -> None:
    assert _risk_label(probability, threshold) == expected


def test_risk_label_low_bucket_always_agrees_with_a_negative_decision() -> None:
    threshold = 0.4
    for probability in (0.0, 0.1, 0.39):
        assert probability < threshold
        assert _risk_label(probability, threshold) == "Low"


_BULK_RESULTS = [
    {
        "index": 0,
        "customerid": "CUST-1",
        "probability": 0.82,
        "threshold": 0.4,
        "decision_threshold": 0.4,
        "decision": True,
        "contact": True,
        "model_version": "3",
        "served_source": "champion",
    },
    {
        "index": 1,
        "customerid": None,
        "probability": 0.1,
        "threshold": 0.4,
        "decision_threshold": 0.4,
        "decision": False,
        "contact": False,
        "model_version": "4",
        "served_source": "challenger",
    },
]


def test_bulk_results_display_df_default_view_is_analyst_facing() -> None:
    df = _bulk_results_display_df(_BULK_RESULTS, show_diagnostics=False)
    assert list(df.columns) == [
        "Customer ID",
        "Churn Probability",
        "Risk Level",
        "Contact This Cycle",
    ]
    assert df["Customer ID"].tolist() == ["CUST-1", "Row 1"]
    assert df["Churn Probability"].tolist() == ["82%", "10%"]
    assert df["Risk Level"].tolist() == ["High", "Low"]
    assert df["Contact This Cycle"].tolist() == ["Yes", "No"]


def test_bulk_results_display_df_diagnostics_view_is_raw_results() -> None:
    df = _bulk_results_display_df(_BULK_RESULTS, show_diagnostics=True)
    assert list(df.columns) == [
        "index",
        "customerid",
        "probability",
        "threshold",
        "decision_threshold",
        "decision",
        "contact",
        "model_version",
        "served_source",
    ]


def test_bulk_drill_options_labels_unique_customerid_plainly() -> None:
    options = _bulk_drill_options(_BULK_RESULTS)
    assert set(options.keys()) == {"CUST-1", "(no id) (row 1)"}
    assert options["CUST-1"]["index"] == 0
    assert options["(no id) (row 1)"]["index"] == 1


def test_bulk_drill_options_disambiguates_repeated_customerid_with_row_number() -> None:
    results = [
        {**_BULK_RESULTS[0], "index": 0, "customerid": "CUST-9"},
        {**_BULK_RESULTS[0], "index": 1, "customerid": "CUST-9"},
    ]
    options = _bulk_drill_options(results)
    assert set(options.keys()) == {"CUST-9 (row 0)", "CUST-9 (row 1)"}
    assert options["CUST-9 (row 0)"]["index"] == 0
    assert options["CUST-9 (row 1)"]["index"] == 1

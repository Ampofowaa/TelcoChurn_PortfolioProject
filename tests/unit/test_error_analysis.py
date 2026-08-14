"""Unit tests for telco_churn.models.error_analysis — pure error-diagnosis
helpers plus run_error_analysis_step's orchestration (Phase 7)."""

from __future__ import annotations

import inspect
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import mlflow
import numpy as np
import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf

import telco_churn.models.artifacts as artifacts
import telco_churn.models.calibrate as calibrate
import telco_churn.models.error_analysis as error_analysis
import telco_churn.models.evaluate as evaluate
import telco_churn.models.register as register
import telco_churn.models.threshold as threshold
import telco_churn.models.train.log_model as log_model
from telco_churn.data.split import make_split, partition, write_split
from telco_churn.features.accessor import FEATURES_FILENAME
from telco_churn.features.schema import FEATURE_SCHEMA
from telco_churn.models.error_analysis import (
    _COLUMN_DISPLAY_NAMES,
    _illustrative_indices,
    _sanitize_metric_name,
    _sorted_gap_pairs,
    display_feature_name,
    error_concentration_scan,
    error_confidence_profile,
    near_miss_band_from_validation,
    value_weighted_error_profile,
)
from telco_churn.utils.paths import activate_config, compose_config, reset_active_config

# ---------------------------------------------------------------------------
# error_confidence_profile
# ---------------------------------------------------------------------------


def test_error_confidence_profile_planted_near_miss_fns_report_high_share() -> None:
    """FNs planted just below t* (within the near-miss band) report a high
    near-miss share — the threshold-artifact case."""
    threshold = 0.50
    y = [1] * 10 + [0] * 10
    proba = [threshold - 0.01] * 10 + [0.1] * 10
    result = error_confidence_profile(y, proba, threshold, near_miss_band=0.05)
    assert result["n_fn"] == 10
    assert result["fn_near_miss_share"] == pytest.approx(1.0)


def test_error_confidence_profile_planted_confident_fns_report_low_share() -> None:
    """FNs planted near zero (far from t*) report a low near-miss share — the
    model-failure case."""
    threshold = 0.50
    y = [1] * 10 + [0] * 10
    proba = [0.02] * 10 + [0.1] * 10
    result = error_confidence_profile(y, proba, threshold, near_miss_band=0.05)
    assert result["n_fn"] == 10
    assert result["fn_near_miss_share"] == pytest.approx(0.0)


def test_error_confidence_profile_empty_fn_cohort_reports_nan_not_raise() -> None:
    """No false negatives at all reports NaN near-miss share rather than
    raising or dividing by zero."""
    y = [0, 0, 0, 1]
    proba = [0.1, 0.2, 0.3, 0.9]  # the one churner is correctly caught
    result = error_confidence_profile(y, proba, threshold=0.5, near_miss_band=0.05)
    assert result["n_fn"] == 0
    assert np.isnan(result["fn_near_miss_share"])


def test_error_confidence_profile_fp_scores_isolated_from_fn_scores() -> None:
    """FP and FN score lists only contain their own error type's scores."""
    y = [1, 0, 1, 0]
    proba = [0.1, 0.9, 0.8, 0.2]  # row0 FN, row1 FP, row2 TP, row3 TN
    result = error_confidence_profile(y, proba, threshold=0.5, near_miss_band=0.05)
    assert result["fn_scores"] == pytest.approx([0.1])
    assert result["fp_scores"] == pytest.approx([0.9])


# ---------------------------------------------------------------------------
# value_weighted_error_profile
# ---------------------------------------------------------------------------


def test_value_weighted_error_profile_fn_rate_rises_with_value_decile() -> None:
    """FN rate rises with MonthlyCharges decile on a fixture where high-value
    churners are systematically missed."""
    rng = np.random.default_rng(0)
    n = 500
    value = rng.uniform(18.0, 120.0, size=n)
    y = (rng.random(n) < 0.4).astype(int)
    # Correctly catch every churner except: miss high-value churners
    # (value above the median) with much higher probability than low-value ones.
    miss_prob = np.where(value >= np.median(value), 0.9, 0.05)
    missed = (rng.random(n) < miss_prob) & (y == 1)
    proba = np.where(y == 1, 0.9, 0.1)
    proba = np.where(missed, 0.1, proba)

    rows = value_weighted_error_profile(
        y.tolist(), proba.tolist(), value.tolist(), threshold=0.5, n_deciles=5
    )
    fn_rates = [row["fn_rate"] for row in rows]
    assert fn_rates[0] < fn_rates[-1]


def test_value_weighted_error_profile_decile_count_and_ascending_value() -> None:
    """Deciles are value-ordered ascending; mean_value increases monotonically."""
    rng = np.random.default_rng(1)
    n = 300
    value = rng.uniform(10.0, 200.0, size=n)
    y = rng.integers(0, 2, size=n)
    proba = rng.random(n)
    rows = value_weighted_error_profile(
        y.tolist(), proba.tolist(), value.tolist(), threshold=0.5, n_deciles=10
    )
    means = [row["mean_value"] for row in rows]
    assert means == sorted(means)


def test_value_weighted_error_profile_zero_positives_in_decile_gives_nan() -> None:
    """A decile with zero actual positives reports NaN fn_rate, not a
    division-by-zero error."""
    value = list(range(20))
    y = [0] * 20  # nobody churns
    proba = [0.1] * 20
    rows = value_weighted_error_profile(y, proba, value, threshold=0.5, n_deciles=4)
    assert all(np.isnan(row["fn_rate"]) for row in rows)
    assert all(row["n_pos"] == 0 for row in rows)
    assert all(row["n_neg"] == row["n"] for row in rows)


def test_value_weighted_error_profile_n_pos_n_neg_persisted_and_consistent() -> None:
    """n_pos/n_neg are persisted per decile (not just re-derivable from n),
    sum to n, and match an independent recomputation of the same bins."""
    rng = np.random.default_rng(2)
    n = 200
    value = rng.uniform(10.0, 200.0, size=n)
    y = rng.integers(0, 2, size=n)
    proba = rng.random(n)
    rows = value_weighted_error_profile(
        y.tolist(), proba.tolist(), value.tolist(), threshold=0.5, n_deciles=5
    )

    bins = pd.qcut(value, q=5)
    for row, category in zip(rows, bins.categories, strict=True):
        mask = np.asarray(bins == category)
        assert row["n_pos"] + row["n_neg"] == row["n"]
        assert row["n_pos"] == int((y[mask] == 1).sum())
        assert row["n_neg"] == int((y[mask] == 0).sum())


# ---------------------------------------------------------------------------
# error_concentration_scan
# ---------------------------------------------------------------------------


def test_error_concentration_scan_finds_planted_cohort_on_non_preregistered_axis() -> (
    None
):
    """A planted high-FNR cohort on an axis that is not one of the seven
    pre-registered V1/V2/V2b axes is found — the whole point of the scan."""
    rng = np.random.default_rng(2)
    n = 400
    # "paymentmethod" is not a V1/V2/V2b axis (contract_type, tenure_cohort,
    # internetservice, gender, seniorcitizen, has_partner, dependents).
    paymentmethod = rng.choice(["Electronic check", "Mailed check"], size=n)
    other_col = rng.integers(0, 2, size=n)
    y = (rng.random(n) < 0.4).astype(int)

    # Electronic check churners are almost always missed; mailed check churners
    # are almost always caught.
    miss_prob = np.where(paymentmethod == "Electronic check", 0.95, 0.05)
    missed = (rng.random(n) < miss_prob) & (y == 1)
    proba = np.where(y == 1, 0.9, 0.1)
    proba = np.where(missed, 0.1, proba)

    feature_frame = pd.DataFrame({"paymentmethod": paymentmethod, "other": other_col})
    rows = error_concentration_scan(
        y.tolist(),
        proba.tolist(),
        threshold=0.5,
        feature_frame=feature_frame,
        numeric_features=[],
        categorical_features=["paymentmethod", "other"],
        n_quantile_bins=5,
        min_support=10,
        top_k=5,
        metric="fnr",
    )
    assert rows[0]["feature"] == "paymentmethod"
    assert rows[0]["value"] == "Electronic check"
    assert rows[0]["metric"] == "fnr"
    assert rows[0]["rate"] > 0.8


def test_error_concentration_scan_finds_planted_high_fpr_cohort() -> None:
    """The FPR counterpart: a planted high-false-alarm cohort is found when
    metric='fpr', proving the scan isn't hard-wired to FN rate specifically."""
    rng = np.random.default_rng(5)
    n = 400
    paymentmethod = rng.choice(["Electronic check", "Mailed check"], size=n)
    y = (rng.random(n) < 0.4).astype(int)

    # Electronic check non-churners are almost always false-alarmed on;
    # mailed check non-churners are almost never false-alarmed on.
    false_alarm_prob = np.where(paymentmethod == "Electronic check", 0.95, 0.05)
    false_alarmed = (rng.random(n) < false_alarm_prob) & (y == 0)
    proba = np.where(y == 1, 0.9, 0.1)
    proba = np.where(false_alarmed, 0.9, proba)

    feature_frame = pd.DataFrame({"paymentmethod": paymentmethod})
    rows = error_concentration_scan(
        y.tolist(),
        proba.tolist(),
        threshold=0.5,
        feature_frame=feature_frame,
        numeric_features=[],
        categorical_features=["paymentmethod"],
        n_quantile_bins=5,
        min_support=10,
        top_k=5,
        metric="fpr",
    )
    assert rows[0]["value"] == "Electronic check"
    assert rows[0]["metric"] == "fpr"
    assert rows[0]["rate"] > 0.8


def test_error_concentration_scan_fnr_and_fpr_rank_independently() -> None:
    """A cohort can rank high on one metric and low on the other — the two
    scans must not be the same ranking read two ways."""
    rng = np.random.default_rng(6)
    n = 400
    group = rng.choice(["A", "B"], size=n)
    y = (rng.random(n) < 0.4).astype(int)

    # Group A: near-total misses (high FNR), but false alarms are rare (low FPR).
    # Group B: churners are always caught (zero FNR), but non-churners are
    # almost always false-alarmed on (high FPR).
    miss_prob_a = 0.95
    false_alarm_prob_b = 0.95
    proba = np.where(y == 1, 0.9, 0.1)
    miss_a = (group == "A") & (y == 1) & (rng.random(n) < miss_prob_a)
    alarm_b = (group == "B") & (y == 0) & (rng.random(n) < false_alarm_prob_b)
    proba = np.where(miss_a, 0.1, proba)
    proba = np.where(alarm_b, 0.9, proba)

    feature_frame = pd.DataFrame({"group": group})
    fnr_rows = {
        row["value"]: row["rate"]
        for row in error_concentration_scan(
            y.tolist(),
            proba.tolist(),
            threshold=0.5,
            feature_frame=feature_frame,
            numeric_features=[],
            categorical_features=["group"],
            n_quantile_bins=5,
            min_support=10,
            top_k=5,
            metric="fnr",
        )
    }
    fpr_rows = {
        row["value"]: row["rate"]
        for row in error_concentration_scan(
            y.tolist(),
            proba.tolist(),
            threshold=0.5,
            feature_frame=feature_frame,
            numeric_features=[],
            categorical_features=["group"],
            n_quantile_bins=5,
            min_support=10,
            top_k=5,
            metric="fpr",
        )
    }
    assert fnr_rows["A"] > fnr_rows["B"]
    assert fpr_rows["B"] > fpr_rows["A"]


def test_error_concentration_scan_drops_cohorts_below_min_support() -> None:
    """A cohort with fewer than min_support rows is excluded entirely."""
    y = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    proba = [0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
    feature_frame = pd.DataFrame({"tiny_group": ["rare"] * 2 + ["common"] * 8})
    rows = error_concentration_scan(
        y,
        proba,
        threshold=0.5,
        feature_frame=feature_frame,
        numeric_features=[],
        categorical_features=["tiny_group"],
        n_quantile_bins=5,
        min_support=5,
        top_k=5,
        metric="fnr",
    )
    values = [row["value"] for row in rows]
    assert "rare" not in values


def test_error_concentration_scan_returns_top_k_sorted_descending() -> None:
    """Results are capped at top_k and sorted by rate descending."""
    rng = np.random.default_rng(3)
    n = 300
    group = rng.integers(0, 6, size=n).astype(str)
    y = (rng.random(n) < 0.5).astype(int)
    proba = rng.random(n)
    feature_frame = pd.DataFrame({"group": group})
    rows = error_concentration_scan(
        y.tolist(),
        proba.tolist(),
        threshold=0.5,
        feature_frame=feature_frame,
        numeric_features=[],
        categorical_features=["group"],
        n_quantile_bins=5,
        min_support=5,
        top_k=2,
        metric="fnr",
    )
    assert len(rows) <= 2
    rates = [row["rate"] for row in rows]
    assert rates == sorted(rates, reverse=True)


def test_error_concentration_scan_bins_numeric_features() -> None:
    """A numeric feature is quantile-binned rather than treated as one giant cohort."""
    rng = np.random.default_rng(4)
    n = 200
    numeric_col = rng.uniform(0, 100, size=n)
    y = (rng.random(n) < 0.5).astype(int)
    proba = rng.random(n)
    feature_frame = pd.DataFrame({"numeric_col": numeric_col})
    rows = error_concentration_scan(
        y.tolist(),
        proba.tolist(),
        threshold=0.5,
        feature_frame=feature_frame,
        numeric_features=["numeric_col"],
        categorical_features=[],
        n_quantile_bins=4,
        min_support=5,
        top_k=10,
        metric="fnr",
    )
    assert all(row["type"] == "numeric_bin" for row in rows)
    assert len({row["value"] for row in rows}) > 1


# ---------------------------------------------------------------------------
# _sorted_gap_pairs
# ---------------------------------------------------------------------------


def test_sorted_gap_pairs_ranks_by_absolute_a_minus_b_gap() -> None:
    """The feature with the largest |A - B| gap ranks first, regardless of
    which cohort's own mean is larger in absolute terms."""
    rows_a = [
        {"feature": "small_gap", "mean_signed_shap": 0.5},
        {"feature": "big_gap", "mean_signed_shap": -0.9},
    ]
    rows_b = [
        {"feature": "small_gap", "mean_signed_shap": 0.45},
        {"feature": "big_gap", "mean_signed_shap": 0.1},
    ]
    result = _sorted_gap_pairs(rows_a, rows_b)
    assert [name for name, _ in result] == ["big_gap", "small_gap"]
    assert result[0][1] == pytest.approx(1.0)


def test_sorted_gap_pairs_returns_every_feature_not_just_a_slice() -> None:
    """Every feature comes back, most-separating first — callers slice the
    top-n themselves; the full ranking is what a drifted-elbow re-derivation
    and error_analysis.json's persisted ranking both need."""
    rows_a = [{"feature": f"f{i}", "mean_signed_shap": float(i)} for i in range(5)]
    rows_b = [{"feature": f"f{i}", "mean_signed_shap": 0.0} for i in range(5)]
    result = _sorted_gap_pairs(rows_a, rows_b)
    assert [name for name, _ in result] == ["f4", "f3", "f2", "f1", "f0"]


def test_sorted_gap_pairs_missing_b_feature_treated_as_zero() -> None:
    """A feature present in cohort A but absent from cohort B's rows is
    scored against an implicit B mean of 0.0, not dropped or errored."""
    rows_a = [{"feature": "a_only", "mean_signed_shap": 0.6}]
    result = _sorted_gap_pairs(rows_a, [])
    assert result == [("a_only", 0.6)]


# ---------------------------------------------------------------------------
# _illustrative_indices
# ---------------------------------------------------------------------------

_ILLUSTRATIVE_Y = np.array([1, 1, 1, 0, 0, 0, 1, 0], dtype=float)
_ILLUSTRATIVE_PROBA = np.array(
    [0.10, 0.20, 0.60, 0.05, 0.90, 0.40, 0.05, 0.99], dtype=float
)


def test_illustrative_indices_fn_fp_pick_most_confidently_wrong_first() -> None:
    """FN indices come back ascending by score (the furthest-below-threshold
    miss first); FP indices come back descending (the furthest-above-
    threshold false alarm first) — each the most legible waterfall story."""
    result = _illustrative_indices(
        _ILLUSTRATIVE_Y, _ILLUSTRATIVE_PROBA, threshold=0.5, n_examples=3
    )
    assert result["fn"] == [6, 0, 1]
    assert result["fp"] == [7, 4]


def test_illustrative_indices_tp_tn_return_exactly_one_regardless_of_n_examples() -> (
    None
):
    """TP/TN each return exactly the single most-confident correct case, not
    n_examples — a fixed reference pair for the FN/FP panels, not a third
    and fourth multi-panel row."""
    result = _illustrative_indices(
        _ILLUSTRATIVE_Y, _ILLUSTRATIVE_PROBA, threshold=0.5, n_examples=3
    )
    assert result["tp"] == [2]
    assert result["tn"] == [3]


# ---------------------------------------------------------------------------
# near_miss_band_from_validation
# ---------------------------------------------------------------------------


def test_near_miss_band_from_validation_uses_ci_half_width() -> None:
    """The band is half the width of argmax_ev_bootstrap_ci, not the fallback."""
    payload = {"argmax_ev_bootstrap_ci": [0.30, 0.40]}
    assert near_miss_band_from_validation(payload, fallback=0.05) == pytest.approx(0.05)


def test_near_miss_band_from_validation_falls_back_when_ci_missing() -> None:
    """A validation payload with no CI key returns the caller-supplied fallback."""
    assert near_miss_band_from_validation({}, fallback=0.05) == 0.05


def test_near_miss_band_from_validation_falls_back_when_ci_empty() -> None:
    """An explicitly empty/falsy CI value also falls back rather than unpacking
    a two-tuple out of nothing."""
    assert (
        near_miss_band_from_validation({"argmax_ev_bootstrap_ci": []}, fallback=0.07)
        == 0.07
    )


def test_near_miss_band_from_validation_narrow_ci_gives_narrow_band() -> None:
    """A tighter bootstrap CI (a more stable empirical argmax) yields a
    tighter near-miss band than a wide one."""
    narrow = near_miss_band_from_validation(
        {"argmax_ev_bootstrap_ci": [0.38, 0.40]}, fallback=0.05
    )
    wide = near_miss_band_from_validation(
        {"argmax_ev_bootstrap_ci": [0.20, 0.60]}, fallback=0.05
    )
    assert narrow < wide


# ---------------------------------------------------------------------------
# _sanitize_metric_name
# ---------------------------------------------------------------------------


def test_sanitize_metric_name_strips_characters_mlflow_rejects() -> None:
    """One-hot feature names carrying spaces and parentheses (e.g. a
    OneHotEncoder level like "Bank transfer (automatic)") no longer break
    mlflow.log_metrics — the regression this module hit in practice."""
    name = "multi_cat__paymentmethod_Bank transfer (automatic)"
    sanitized = _sanitize_metric_name(name)
    assert "(" not in sanitized
    assert ")" not in sanitized


def test_sanitize_metric_name_leaves_allowed_characters_untouched() -> None:
    """Alphanumerics, underscores, dashes, periods, spaces, and slashes are
    MLflow's own allow-list and pass through unchanged."""
    name = "abc_XYZ-123. /"
    assert _sanitize_metric_name(name) == name


# ---------------------------------------------------------------------------
# display_feature_name
# ---------------------------------------------------------------------------


def testdisplay_feature_name_numeric() -> None:
    """A numeric feature's raw ColumnTransformer name maps to its pretty
    column label with no value suffix."""
    assert display_feature_name("numeric__monthlycharges") == "Monthly Charges"


def testdisplay_feature_name_binary_one_hot() -> None:
    """A binary one-hot feature renders as 'Column: Category'."""
    assert display_feature_name("binary__gender_Male") == "Gender: Male"


def testdisplay_feature_name_multi_cat_column_with_underscore() -> None:
    """A multi_cat column whose raw name itself contains an underscore
    (contract_type) still splits at the right point — the regression this
    function exists to avoid versus a naive first-underscore split."""
    name = "multi_cat__contract_type_Month-to-month"
    assert display_feature_name(name) == "Contract Type: Month-to-month"


def testdisplay_feature_name_category_with_spaces_and_parens() -> None:
    """A category value containing spaces and parentheses (a real OneHotEncoder
    level) is preserved verbatim after the column-name split."""
    name = "multi_cat__paymentmethod_Bank transfer (automatic)"
    assert display_feature_name(name) == "Payment Method: Bank transfer (automatic)"


def testdisplay_feature_name_charge_per_service_column_with_underscore() -> None:
    """A numeric column whose own name contains an underscore is looked up
    whole, not split."""
    assert display_feature_name("numeric__charge_per_service") == "Charge Per Service"


def testdisplay_feature_name_unrecognised_falls_back_to_raw() -> None:
    """An unknown prefix or a name with no '__' separator returns unchanged —
    a display helper must never raise on an unexpected feature."""
    assert display_feature_name("some_other_thing") == "some_other_thing"
    assert (
        display_feature_name("unknown__made_up_feature") == "unknown__made_up_feature"
    )


def test_column_display_names_covers_every_feature_schema_column() -> None:
    """_COLUMN_DISPLAY_NAMES must stay in lockstep with FEATURE_SCHEMA — a
    column added to the schema with no display-name entry would silently fall
    back to its raw snake_case form on every chart."""
    schema_columns = set(FEATURE_SCHEMA.binary + FEATURE_SCHEMA.multi_cat) | set(
        FEATURE_SCHEMA.numeric
    )
    assert schema_columns == set(_COLUMN_DISPLAY_NAMES)


# ---------------------------------------------------------------------------
# Structural guard — mirrors CLAUDE.md's "test set touched once" invariant
# ---------------------------------------------------------------------------


def test_error_analysis_module_never_imports_data_split() -> None:
    """error_analysis.py must not import telco_churn.data.split — the module
    reaches evaluation data solely through reports/test_predictions.parquet
    and reports/dev_oof_predictions.parquet, both written by evaluate.py."""
    source = inspect.getsource(error_analysis)
    assert "data.split" not in source


# ---------------------------------------------------------------------------
# run_error_analysis_step — orchestration
#
# Mirrors test_calibrate.py's direct run_calibration_step tests: seed the
# real production chain (log_model -> calibrate -> threshold -> evaluate) in
# a throwaway SQLite MLflow store, then call run_error_analysis_step directly
# rather than only through tests/integration/test_error_analysis_subprocess.py's
# subprocess boundary. Unlike that subprocess call, this exercises the
# orchestrator in-process, so it is the only place coverage.py sees its body.
# ---------------------------------------------------------------------------

_FAST_CALIBRATION_OVERRIDES = [
    "calibration.method=sigmoid",
    "calibration.outer_cv_folds=3",
    "calibration.inner_cv_folds=3",
]
# Small enough to keep evaluate.run_evaluation_step's many bootstrap passes
# fast when seeding the fixture; error_analysis.py's own SHAP pass has no
# bootstrap knob of its own.
_FAST_EVALUATE_OVERRIDES = ["evaluate.n_bootstrap=30"]
# threshold.n_bootstrap defaults to 1000 (configs/threshold/default.yaml) and
# is refit once per FAIRNESS_AXES group (8 groups) inside run_threshold_step's
# V2b calibration-collapse check (sliced_calibration -> calibration_slope's
# LogisticRegression bootstrap) -- ~8k unreduced refits. test_threshold.py's
# own fixture already reduces this to 200; mirrored here since this fixture
# nests the same call inside a larger chain. v3_top_k_features=3 (not the
# production 8): _make_synthetic_processed_frame's docstring plants exactly
# three real signal columns (contract_type, tenure, monthlycharges) — the
# only ones with a real, learnable relationship to churn — so cutting the V3
# pre-seal veto's top-k at 3 keeps it checking real signal instead of noise
# from one of this fixture's many uninformative columns.
_FAST_THRESHOLD_OVERRIDES = [
    "threshold.n_bootstrap=50",
    "threshold.v3_top_k_features=3",
]


def _make_synthetic_processed_frame(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """A FeatureOutputSchema-conformant frame with every segment-axis column
    evaluate.py's slicing needs, sized so each axis's groups clear its
    minimum slice size on both the dev and test partitions.

    churn is drawn from a logistic function of contract_type/tenure/
    monthlycharges — a real, learnable relationship, so FN/FP cohorts are
    non-degenerate and the SHAP direction-sanity check has something to check.
    """
    rng = np.random.default_rng(seed)
    contract_type = rng.choice(["Month-to-month", "One year", "Two year"], size=n)
    tenure = rng.integers(0, 73, size=n)
    monthlycharges = rng.uniform(18.25, 118.75, size=n)

    logit = (
        -0.5
        + 1.4 * (contract_type == "Month-to-month")
        - 0.03 * tenure
        + 0.01 * (monthlycharges - 60.0)
    )
    churn_prob = 1.0 / (1.0 + np.exp(-logit))
    churn = (rng.random(n) < churn_prob).astype(int)

    return pd.DataFrame(
        {
            "customerid": [f"cust-{i:04d}" for i in range(n)],
            "gender": rng.choice(["Male", "Female"], size=n),
            "has_partner": rng.choice(["Yes", "No"], size=n),
            "dependents": rng.choice(["Yes", "No"], size=n),
            "phoneservice": rng.choice(["Yes", "No"], size=n),
            "paperlessbilling": rng.choice(["Yes", "No"], size=n),
            "seniorcitizen": rng.integers(0, 2, size=n).tolist(),
            "multiplelines": rng.choice(["Yes", "No", "No phone service"], size=n),
            "internetservice": rng.choice(["DSL", "Fiber optic", "No"], size=n),
            "onlinesecurity": rng.choice(["Yes", "No", "No internet service"], size=n),
            "onlinebackup": rng.choice(["Yes", "No", "No internet service"], size=n),
            "deviceprotection": rng.choice(
                ["Yes", "No", "No internet service"], size=n
            ),
            "techsupport": rng.choice(["Yes", "No", "No internet service"], size=n),
            "streamingtv": rng.choice(["Yes", "No", "No internet service"], size=n),
            "streamingmovies": rng.choice(["Yes", "No", "No internet service"], size=n),
            "contract_type": contract_type,
            "paymentmethod": rng.choice(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                size=n,
            ),
            "tenure": tenure.tolist(),
            "monthlycharges": monthlycharges.tolist(),
            "totalcharges": (monthlycharges * np.maximum(tenure, 1)).tolist(),
            "charge_per_service": rng.uniform(0.5, 50.0, size=n).tolist(),
            "churn": churn.tolist(),
        }
    )


def _seed_processed_data(
    out_dir: Path, n: int = 300, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write a processed features file + matching canonical split manifest into out_dir."""
    df = _make_synthetic_processed_frame(n=n, seed=seed)
    df.to_parquet(out_dir / FEATURES_FILENAME, index=False)
    manifest = make_split(
        ids=df["customerid"], labels=df["churn"], test_size=0.2, random_state=42
    )
    write_split(manifest, out_dir / "split_manifest.parquet")
    return df, manifest


def _base_costs_cfg_dict() -> dict[str, Any]:
    """A valid, minimal costs.yaml payload — matches configs/costs.yaml's
    schema, including contact_capacity (evaluate.py's EV-by-budget figure
    reads it), with a reduced bootstrap count for speed."""
    return {
        "gross_margin": 0.60,
        "horizon_months": 12,
        "discount_months": 3,
        "arpu_quantile": {"conservative": 0.25, "base": 0.50, "optimistic": 0.75},
        "scenarios": {
            "conservative": {
                "outreach_cost": 5.0,
                "discount_rate": 0.10,
                "retention_rate": 0.20,
            },
            "base": {
                "outreach_cost": 20.0,
                "discount_rate": 0.20,
                "retention_rate": 0.30,
            },
            "optimistic": {
                "outreach_cost": 50.0,
                "discount_rate": 0.30,
                "retention_rate": 0.40,
            },
        },
        "retention_rate_sweep": [0.15, 0.20, 0.30, 0.40, 0.45],
        "contact_capacity": 200,
        "campaign_budget": 15_000,
        "argmax_ev_bootstrap_n_samples": 100,
    }


@pytest.fixture(scope="module")
def evaluated_model(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[dict[str, object]]:
    """Seed a real registered, calibrated, thresholded, and evaluated model —
    the production chain up to (not including) error_analysis.py — via direct
    in-process calls, then leave the config installed so the test body's own
    call into run_error_analysis_step resolves the same synthetic features.

    Self-contained: a throwaway SQLite MLflow store, no Docker — mirrors
    tests/integration/test_error_analysis_subprocess.py::evaluated_model,
    which stops at the same point for the same reason but hands off to a
    subprocess instead of a direct call.

    Module-scoped: every test in this file only reads this fixture's output
    (run_error_analysis_step is re-invoked, deterministically, by each test
    body) rather than mutating it, so the full train->calibrate->threshold->
    evaluate chain — ~65s of real model fitting — is built once per module
    instead of once per test. activate_config's installed config is a
    process-global, not tied to any fixture scope, so nothing here depends on
    tmp_path/monkeypatch's function scoping the way the old env-var version did.
    """
    tmp_path = tmp_path_factory.mktemp("evaluated_model")
    data_dir = tmp_path / "processed"
    data_dir.mkdir()
    df, manifest = _seed_processed_data(data_dir)

    # Must match cfg.mlflow.experiment_name's own default (not an arbitrary
    # name): log_model.run_model_logging_step calls mlflow.set_experiment(cfg
    # .mlflow.experiment_name) internally before resuming the parent run this
    # fixture opens below, so a mismatched name here switches the active
    # experiment out from under that run and start_run raises.
    experiment_name = str(compose_config().mlflow.experiment_name)
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    experiment_id = mlflow.create_experiment(
        experiment_name, artifact_location=(tmp_path / "artifacts").as_uri()
    )
    mlflow.set_experiment(experiment_id=experiment_id)

    registered_model_name = "test-error-analysis-pipeline"
    costs_path = tmp_path / "costs.yaml"
    OmegaConf.save(OmegaConf.create(_base_costs_cfg_dict()), costs_path)
    figures_dir = tmp_path / "figures"
    policy_dir = tmp_path / "policy"
    reports_dir = tmp_path / "reports"

    cfg = compose_config(
        overrides=[
            f"mlflow.tracking_uri={tracking_uri}",
            f"mlflow.registered_model_name={registered_model_name}",
            f"paths.costs_config={costs_path}",
            f"paths.figures={figures_dir}",
            f"paths.policy={policy_dir}",
            f"paths.reports={reports_dir}",
            f"paths.processed_data={data_dir}",
            *_FAST_CALIBRATION_OVERRIDES,
            *_FAST_EVALUATE_OVERRIDES,
            *_FAST_THRESHOLD_OVERRIDES,
        ]
    )
    activate_config(cfg)

    dev_df, _test_df = partition(df, manifest)
    feature_cols = [c for c in df.columns if c not in ("customerid", "churn")]
    X_dev, y_dev = dev_df[feature_cols], dev_df["churn"]

    tuning_result: dict[str, Any] = {
        "best_params": {
            "num_leaves": 8,
            "learning_rate": 0.1,
            "min_child_samples": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "max_depth": 5,
        },
        "best_n_estimators_median": 10,
        "best_cv_pr_auc_mean": 0.6,
        "committed_features": feature_cols,
        "tuning_summary": {
            "n_trials_requested": 50,
            "n_completed_trials": 16,
            "n_pruned_trials": 34,
            "n_failed_trials": 0,
            "min_completed_trials": 10,
            "trial_count_below_threshold": False,
            "selection_rule": "1se",
            "selected_trial_number": 9,
            "selected_cv_pr_auc": 0.6,
            "raw_best_trial_number": 36,
            "raw_best_cv_pr_auc": 0.6664,
            "se": 0.0139,
            "band_floor": 0.6525,
            "boundary_hits": {"num_leaves": False},
        },
    }
    with mlflow.start_run(run_name="tuning_study") as run:
        tuning_result["parent_run_id"] = run.info.run_id
    log_result = log_model.run_model_logging_step(X_dev, y_dev, tuning_result, cfg)
    cal_result = calibrate.run_calibration_step(log_result["run_id"], cfg)
    run_id = str(cal_result["run_id"])
    model_uri = str(cal_result["model_uri"])
    # B1's call-site decoupling: calibrate.py no longer registers anything
    # itself, so this fixture mints the challenger the same way register.py's
    # own mint-mode CLI would, in-process.
    model_version = register.register_challenger(
        cfg, run_id, model_uri, str(cal_result["logged_model_id"])
    )

    threshold.run_threshold_step(run_id, model_version, cfg)
    evaluate.run_evaluation_step(run_id, model_version, model_uri, cfg)

    try:
        yield {
            "cfg": cfg,
            "run_id": run_id,
            "model_version": model_version,
            "model_uri": model_uri,
            "data_dir": data_dir,
            "reports_dir": reports_dir,
            "figures_dir": figures_dir,
            "policy_dir": policy_dir,
            "costs_path": costs_path,
            "tracking_uri": tracking_uri,
            "registered_model_name": registered_model_name,
        }
    finally:
        reset_active_config()


@pytest.fixture(autouse=True)
def _reactivate_evaluated_model_config(
    request: pytest.FixtureRequest,
) -> Iterator[None]:
    """Re-install evaluated_model's cfg for each test body that uses it.

    evaluated_model is module-scoped, so its own activate_config() call is
    undone by tests/unit/conftest.py's autouse _reset_active_config (function-
    scoped, and same-scope conftest fixtures instantiate before same-scope
    fixtures declared in the test module itself — so it runs after
    evaluated_model's setup but before every test body). Re-installing here,
    at the same function scope but declared in this file, runs after that
    reset and is what makes the config survive for the test body. Guarded on
    request.fixturenames so tests that don't use evaluated_model — most of
    this module — never pay for building it.
    """
    if "evaluated_model" not in request.fixturenames:
        yield
        return
    cfg = cast(DictConfig, request.getfixturevalue("evaluated_model")["cfg"])
    activate_config(cfg)
    try:
        yield
    finally:
        reset_active_config()


def test_run_error_analysis_step_returns_expected_keys_and_writes_report(
    evaluated_model: dict[str, object],
) -> None:
    """run_error_analysis_step computes SHAP on the sealed test set, scans dev
    OOF for error concentration, and writes reports/error_analysis.json — the
    artifact register.py aborts without."""
    cfg = cast(DictConfig, evaluated_model["cfg"])
    run_id = str(evaluated_model["run_id"])
    model_version = str(evaluated_model["model_version"])
    model_uri = str(evaluated_model["model_uri"])

    result = error_analysis.run_error_analysis_step(
        run_id, model_version, model_uri, cfg
    )

    assert result["model_version"] == model_version
    assert result["error_analysis_run_id"]
    payload = cast(dict[str, Any], result["error_analysis"])
    for key in (
        "error_confidence",
        "value_weighted_errors",
        "error_concentration",
        "shap",
        "top_k_elbow_checks",
        "subgroup_findings",
        "direction_sanity_check_test",
    ):
        assert key in payload, f"error_analysis payload missing field: {key}"
    assert "dev_oof_diagnostics_carried_through" not in payload

    reports_dir = Path(str(evaluated_model["reports_dir"]))
    on_disk = json.loads(
        (reports_dir / "error_analysis.json").read_text(encoding="utf-8")
    )
    assert on_disk["model_version"] == model_version


def test_run_error_analysis_step_shap_payload_has_expected_shape(
    evaluated_model: dict[str, object],
) -> None:
    """The shap sub-payload carries every field the notebook and model_card
    assembly (register.py) read, including both cohort-gap rankings."""
    cfg = cast(DictConfig, evaluated_model["cfg"])
    run_id = str(evaluated_model["run_id"])
    model_version = str(evaluated_model["model_version"])
    model_uri = str(evaluated_model["model_uri"])

    result = error_analysis.run_error_analysis_step(
        run_id, model_version, model_uri, cfg
    )
    payload = cast(dict[str, Any], result["error_analysis"])
    shap_payload = payload["shap"]

    assert set(shap_payload) >= {
        "global_importance",
        "top_features",
        "dependence_feature_set",
        "dependence",
        "binary_dependence",
        "cohort_shap",
        "cohort_top_features",
        "cohort_top_features_fp_tn",
        "cohort_gap_ranking_fn_tp",
        "cohort_gap_ranking_fp_tn",
        "local_explanations",
    }
    assert set(shap_payload["cohort_shap"]) == {"fn", "tp", "fp", "tn"}
    assert "passed" in payload["direction_sanity_check_test"]
    assert set(payload["top_k_elbow_checks"]) == {
        "dependence_features",
        "cohort_gap_fn_tp",
        "cohort_gap_fp_tn",
    }


def test_run_error_analysis_step_dependence_grid_includes_dev_v3_audit_set(
    evaluated_model: dict[str, object],
) -> None:
    """The dependence grid's feature set is top_k_dependence_features' own
    test-side ranking unioned with threshold.py's dev-side V3 audit set —
    sorted by test mean-|SHAP| descending, with no independent test-side k
    for the inherited half (item 11/12)."""
    cfg = cast(DictConfig, evaluated_model["cfg"])
    run_id = str(evaluated_model["run_id"])
    model_version = str(evaluated_model["model_version"])
    model_uri = str(evaluated_model["model_uri"])

    result = error_analysis.run_error_analysis_step(
        run_id, model_version, model_uri, cfg
    )
    payload = cast(dict[str, Any], result["error_analysis"])
    shap_payload = payload["shap"]

    dev_oof_diagnostics = artifacts.load_dev_oof_diagnostics(run_id, cfg)
    dev_audit_set = set(dev_oof_diagnostics["direction_check_feature_names"])

    dependence_feature_set = shap_payload["dependence_feature_set"]
    assert dev_audit_set <= set(dependence_feature_set)
    assert set(shap_payload["top_features"]) <= set(dependence_feature_set)

    importance_by_feature = {
        row["feature"]: row["mean_abs_shap"]
        for row in shap_payload["global_importance"]
    }
    mean_abs_shap_values = [importance_by_feature[f] for f in dependence_feature_set]
    assert mean_abs_shap_values == sorted(mean_abs_shap_values, reverse=True)


def test_run_error_analysis_step_writes_all_figures(
    evaluated_model: dict[str, object],
) -> None:
    """Every figure the notebook embeds is written to paths.figures, not left
    to be regenerated ad hoc."""
    cfg = cast(DictConfig, evaluated_model["cfg"])
    run_id = str(evaluated_model["run_id"])
    model_version = str(evaluated_model["model_version"])
    model_uri = str(evaluated_model["model_uri"])

    error_analysis.run_error_analysis_step(run_id, model_version, model_uri, cfg)

    figures_dir = Path(str(evaluated_model["figures_dir"]))
    for filename in (
        "shap_global_importance.png",
        "shap_beeswarm.png",
        "shap_dependence.png",
        "shap_cohort_fn_vs_tp.png",
        "shap_cohort_beeswarm_fn.png",
        "shap_cohort_beeswarm_tp.png",
        "shap_cohort_fp_vs_tn.png",
        "shap_cohort_beeswarm_fp.png",
        "shap_cohort_beeswarm_tn.png",
        "shap_cohort_dependence_tenure.png",
        "shap_subgroup_dependence_monthlycharges_annual.png",
        "shap_waterfall_examples.png",
        "shap_waterfall_confident_cases.png",
        "error_confidence.png",
        "value_weighted_errors.png",
    ):
        assert (figures_dir / filename).exists(), f"missing figure: {filename}"


def test_run_error_analysis_step_logs_mlflow_run_and_direction_tag(
    evaluated_model: dict[str, object],
) -> None:
    """Logs its own error_analysis run (a sibling of evaluate.py's evaluation
    run) tagged with the test-side direction-sanity re-audit verdict, with
    error_analysis.json and the figures/ folder attached as artifacts."""
    cfg = cast(DictConfig, evaluated_model["cfg"])
    run_id = str(evaluated_model["run_id"])
    model_version = str(evaluated_model["model_version"])
    model_uri = str(evaluated_model["model_uri"])

    result = error_analysis.run_error_analysis_step(
        run_id, model_version, model_uri, cfg
    )

    client = mlflow.tracking.MlflowClient(
        tracking_uri=str(evaluated_model["tracking_uri"])
    )
    run = client.get_run(str(result["error_analysis_run_id"]))
    assert run.info.run_name == "error_analysis"
    assert run.data.tags.get("direction_sanity_check_test") in {"pass", "fail"}

    artifact_paths = {a.path for a in client.list_artifacts(run.info.run_id)}
    assert "error_analysis.json" in artifact_paths
    assert "shap_values.parquet" in artifact_paths
    # list_artifacts is non-recursive — a subfolder like "figures/" shows up
    # as one directory entry, not its individual file paths.
    assert "figures" in artifact_paths
    figures_artifact_paths = {
        a.path for a in client.list_artifacts(run.info.run_id, "figures")
    }
    assert "figures/shap_global_importance.png" in figures_artifact_paths


def test_run_error_analysis_step_raises_when_prediction_artifacts_missing(
    evaluated_model: dict[str, object], tmp_path: Path
) -> None:
    """A reports/ directory without evaluate.py's prediction artifacts fails
    loudly rather than silently reaching for the test split some other way —
    mirrors test_error_analysis_subprocess.py's exit-1 CLI case, one layer
    closer to the code that actually raises."""
    run_id = str(evaluated_model["run_id"])
    model_version = str(evaluated_model["model_version"])
    model_uri = str(evaluated_model["model_uri"])
    empty_reports_dir = tmp_path / "empty_reports"
    empty_reports_dir.mkdir()
    cfg_with_empty_reports = compose_config(
        overrides=[
            f"mlflow.tracking_uri={evaluated_model['tracking_uri']}",
            f"mlflow.registered_model_name={evaluated_model['registered_model_name']}",
            f"paths.costs_config={evaluated_model['costs_path']}",
            f"paths.figures={evaluated_model['figures_dir']}",
            f"paths.policy={evaluated_model['policy_dir']}",
            f"paths.reports={empty_reports_dir}",
            *_FAST_CALIBRATION_OVERRIDES,
            *_FAST_EVALUATE_OVERRIDES,
            *_FAST_THRESHOLD_OVERRIDES,
        ]
    )

    with pytest.raises(FileNotFoundError):
        error_analysis.run_error_analysis_step(
            run_id, model_version, model_uri, cfg_with_empty_reports
        )


def test_run_error_analysis_step_raises_on_missing_dev_oof_diagnostics_key(
    evaluated_model: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dev_oof_diagnostics.json missing direction_check_feature_names raises
    KeyError — a genuine pipeline defect, never silently rendered as an empty
    V3 audit set. This is now the only key this module reads directly off
    threshold.py's artifact — the V1/V2/V2b/V3 flags are no longer carried
    through error_analysis.json; register.py's model card fetches those
    itself, directly, at promotion time."""
    cfg = cast(DictConfig, evaluated_model["cfg"])
    run_id = str(evaluated_model["run_id"])
    model_version = str(evaluated_model["model_version"])
    model_uri = str(evaluated_model["model_uri"])

    def _incomplete_dev_oof_diagnostics(
        _run_id: str, _cfg: DictConfig
    ) -> dict[str, Any]:
        return {}  # direction_check_feature_names deliberately omitted

    monkeypatch.setattr(
        error_analysis, "load_dev_oof_diagnostics", _incomplete_dev_oof_diagnostics
    )

    with pytest.raises(KeyError, match="direction_check_feature_names"):
        error_analysis.run_error_analysis_step(run_id, model_version, model_uri, cfg)


def test_run_error_analysis_step_raises_on_stale_test_predictions_stamp(
    evaluated_model: dict[str, object],
) -> None:
    """reports/test_predictions.parquet stamped with a different logged_model_id
    than the model being analyzed raises — the stale-copy-on-hand-run-rollback
    hole the stamp exists to close."""
    cfg = cast(DictConfig, evaluated_model["cfg"])
    run_id = str(evaluated_model["run_id"])
    model_version = str(evaluated_model["model_version"])
    model_uri = str(evaluated_model["model_uri"])
    reports_dir = Path(str(evaluated_model["reports_dir"]))

    predictions_path = reports_dir / "test_predictions.parquet"
    original = pd.read_parquet(predictions_path)
    stale = original.copy()
    stale["logged_model_id"] = "a-different-logged-model-id"
    stale.to_parquet(predictions_path, index=False)

    try:
        with pytest.raises(ValueError, match="a-different-logged-model-id"):
            error_analysis.run_error_analysis_step(
                run_id, model_version, model_uri, cfg
            )
    finally:
        # evaluated_model is module-scoped and shared with later tests —
        # restore the real stamp so this test's mutation doesn't poison them.
        original.to_parquet(predictions_path, index=False)

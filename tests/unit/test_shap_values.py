"""Unit tests for telco_churn.models.shap_values — the sole shap-importing module under models/ (Phase 8)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from telco_churn.features.preprocessing import build_preprocessor
from telco_churn.models.shap_values import (
    build_tree_explainer,
    compute_shap_values,
    explain_with_explainer,
    unwrap_calibrated_pipeline,
)

_BINARY = ["informative_bin"]
_MULTI_CAT: list[str] = []
_NUMERIC = ["informative_num", "noise_num"]

_FAST_ESTIMATOR_PARAMS = {
    "n_estimators": 30,
    "num_leaves": 7,
    "min_child_samples": 5,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": 1,
    "verbose": -1,
}


def _make_synthetic_data(
    n: int = 200, seed: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    informative_num = rng.normal(0, 1, size=n)
    noise_num = rng.normal(0, 1, size=n)
    informative_bin = rng.choice(["Yes", "No"], size=n)

    logit = 1.5 * informative_num + 1.2 * (informative_bin == "Yes").astype(float) - 0.5
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = pd.Series(rng.binomial(1, prob), name="churn")
    X = pd.DataFrame(
        {
            "informative_num": informative_num,
            "noise_num": noise_num,
            "informative_bin": informative_bin,
        }
    )
    return X, y


@pytest.fixture
def fitted_calibrated_pipeline() -> tuple[CalibratedClassifierCV, pd.DataFrame]:
    """A real, small CalibratedClassifierCV(ensemble=False) over [preprocessor -> LGBMClassifier]."""
    X, y = _make_synthetic_data()
    preprocessor = build_preprocessor(_BINARY, _MULTI_CAT, _NUMERIC)
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", LGBMClassifier(**_FAST_ESTIMATOR_PARAMS)),
        ]
    )
    calibrated = CalibratedClassifierCV(
        pipeline,
        method="sigmoid",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        ensemble=False,
    )
    calibrated.fit(X, y)
    return calibrated, X


# ---------------------------------------------------------------------------
# unwrap_calibrated_pipeline
# ---------------------------------------------------------------------------


def test_unwrap_calibrated_pipeline_returns_preprocessor_and_booster(
    fitted_calibrated_pipeline: tuple[CalibratedClassifierCV, pd.DataFrame],
) -> None:
    """ensemble=False collapses calibrated_classifiers_ to length 1, whose
    .estimator is the [preprocessor -> model] Pipeline refit on all data."""
    calibrated, _X = fitted_calibrated_pipeline
    preprocessor, booster = unwrap_calibrated_pipeline(calibrated)
    assert hasattr(preprocessor, "transform")
    assert isinstance(booster, LGBMClassifier)


# ---------------------------------------------------------------------------
# compute_shap_values
# ---------------------------------------------------------------------------


def test_compute_shap_values_returns_one_row_per_input_row(
    fitted_calibrated_pipeline: tuple[CalibratedClassifierCV, pd.DataFrame],
) -> None:
    """shap_values and Xt both carry one row per input customer."""
    calibrated, X = fitted_calibrated_pipeline
    preprocessor, booster = unwrap_calibrated_pipeline(calibrated)
    shap_values, feature_names, base_value, Xt = compute_shap_values(
        preprocessor, booster, X
    )
    assert shap_values.shape[0] == len(X)
    assert Xt.shape[0] == len(X)
    assert shap_values.shape[1] == len(feature_names)
    assert np.isfinite(base_value)


def test_compute_shap_values_feature_names_match_transformed_columns(
    fitted_calibrated_pipeline: tuple[CalibratedClassifierCV, pd.DataFrame],
) -> None:
    """feature_names is exactly the preprocessor's own transformed column order."""
    calibrated, X = fitted_calibrated_pipeline
    preprocessor, booster = unwrap_calibrated_pipeline(calibrated)
    _shap_values, feature_names, _base_value, _Xt = compute_shap_values(
        preprocessor, booster, X
    )
    assert feature_names == list(preprocessor.get_feature_names_out())


# ---------------------------------------------------------------------------
# build_tree_explainer / explain_with_explainer
# ---------------------------------------------------------------------------


def test_explain_with_cached_explainer_matches_compute_shap_values(
    fitted_calibrated_pipeline: tuple[CalibratedClassifierCV, pd.DataFrame],
) -> None:
    """A cached explainer, reused across calls, reproduces compute_shap_values'
    one-shot output exactly — the equivalence serving/predict.py's caching
    seam depends on."""
    calibrated, X = fitted_calibrated_pipeline
    preprocessor, booster = unwrap_calibrated_pipeline(calibrated)

    one_shot = compute_shap_values(preprocessor, booster, X)

    explainer = build_tree_explainer(booster)
    cached_first = explain_with_explainer(explainer, preprocessor, X)
    cached_second = explain_with_explainer(explainer, preprocessor, X.iloc[:5])

    np.testing.assert_allclose(one_shot[0], cached_first[0])
    assert one_shot[1] == cached_first[1]
    assert one_shot[2] == pytest.approx(cached_first[2])
    np.testing.assert_allclose(one_shot[3], cached_first[3])

    # The same cached explainer, reused on a smaller slice, is untouched by
    # the earlier call — no per-call mutable state leaking across requests.
    np.testing.assert_allclose(cached_second[0], one_shot[0][:5])


def test_compute_shap_values_shap_values_are_2d() -> None:
    """The (n_rows, n_features, 2) binary-classifier explanation is collapsed
    to 2D by selecting the positive-class slice — never left 3D."""
    X, y = _make_synthetic_data(n=150, seed=7)
    preprocessor = build_preprocessor(_BINARY, _MULTI_CAT, _NUMERIC)
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", LGBMClassifier(**_FAST_ESTIMATOR_PARAMS)),
        ]
    )
    pipeline.fit(X, y)
    shap_values, _feature_names, _base_value, _Xt = compute_shap_values(
        pipeline.named_steps["preprocessor"], pipeline.named_steps["model"], X
    )
    assert shap_values.ndim == 2

"""SHAP computation — the sole `shap`-importing module under models/.

Kept separate from explain.py (pure summary statistics, no shap import) and
from calibrate.py/error_analysis.py (orchestration): exactly one place knows
how to reach into a CalibratedClassifierCV(ensemble=False) and drive
shap.TreeExplainer, so calibrate.py's dev-SHAP logging and error_analysis.py's
test-SHAP diagnostics can never diverge in how they get there.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap
from numpy.typing import NDArray

__all__ = [
    "build_tree_explainer",
    "compute_shap_values",
    "explain_with_explainer",
    "unwrap_calibrated_pipeline",
]


def unwrap_calibrated_pipeline(model: Any) -> tuple[Any, Any]:
    """Return (preprocessor, booster) from a CalibratedClassifierCV(ensemble=False).

    ensemble=False (calibrate.py::build_calibrated_pipeline) collapses
    calibrated_classifiers_ to length 1, whose .estimator is the
    [preprocessor -> model] Pipeline refit on all of development —
    features.select.compute_shap_audit uses the same access path.
    """
    base_pipeline = model.calibrated_classifiers_[0].estimator
    preprocessor = base_pipeline.named_steps["preprocessor"]
    booster = base_pipeline.named_steps["model"]
    return preprocessor, booster


def build_tree_explainer(booster: Any) -> Any:
    """Build a shap.TreeExplainer for `booster`, cacheable across many calls.

    Split out of compute_shap_values so serving/predict.py can build this
    once per loaded champion (an expensive tree-structure parse) and reuse it
    across many per-request explanations, rather than rebuilding it on every
    call the way compute_shap_values' one-shot batch usage does.
    """
    return shap.TreeExplainer(booster)


def explain_with_explainer(
    explainer: Any, preprocessor: Any, X: pd.DataFrame
) -> tuple[NDArray[np.float64], list[str], float, NDArray[np.float64]]:
    """SHAP values for `X` against an already-built `explainer`.

    Returns (shap_values, feature_names, base_value, Xt); Xt (the transformed
    model input matrix) is returned too because explain.py's
    dependence_points/local_explanations both need feature values in the same
    space shap_values was computed in.

    Uses the explainer's `__call__` API rather than the legacy
    `.shap_values(Xt)`, which warns unconditionally for a binary LightGBM
    classifier (shap/explainers/_tree.py). `Explanation.values` comes back
    shape (n_rows, n_features, 2) for a binary classifier — index `[..., 1]`
    selects the positive (churn) class.
    """
    Xt = preprocessor.transform(X)
    feature_names = list(preprocessor.get_feature_names_out())

    shap_values = explainer(Xt).values
    if shap_values.ndim == 3:
        shap_values = shap_values[..., 1]

    base_value_raw = explainer.expected_value
    base_value = (
        float(np.atleast_1d(base_value_raw)[-1])
        if isinstance(base_value_raw, list | np.ndarray)
        else float(base_value_raw)
    )
    return (
        np.asarray(shap_values, dtype=float),
        feature_names,
        base_value,
        np.asarray(Xt),
    )


def compute_shap_values(
    preprocessor: Any, booster: Any, X: pd.DataFrame
) -> tuple[NDArray[np.float64], list[str], float, NDArray[np.float64]]:
    """TreeExplainer SHAP values for `X` against the champion's base LightGBM step.

    One-shot batch usage (build the explainer and use it exactly once) — see
    build_tree_explainer/explain_with_explainer for the cacheable-explainer
    split serving/predict.py needs instead.
    """
    return explain_with_explainer(build_tree_explainer(booster), preprocessor, X)

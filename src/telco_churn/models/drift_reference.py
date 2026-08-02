"""Pure builder for the champion's drift-monitoring baseline.

No I/O, no MLflow: build_reference takes training features, an out-of-sample
prediction-score vector, and labels, and returns the reference distributions
a later drift check compares live traffic against. register.py persists the
result onto the promoted run; a drift-check flow and dashboards consume it.

The score distribution must come from out-of-sample probabilities, never the
champion's own scores on its training rows — in-sample scores are
systematically sharper, so baselining on them would read as permanent drift
that is really just the in-sample/out-of-sample gap. Feature distributions
have no such problem and come straight from the training population.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from telco_churn.features.schema import FEATURE_SCHEMA

__all__ = ["build_reference"]


def _quantile_bin_edges(x: NDArray[np.float64], n_bins: int) -> NDArray[np.float64]:
    """Deduplicated quantile bin edges spanning x's observed range.

    Quantile-based so a skewed feature gets informative bins near its mass
    rather than one dominant bucket. Falls back to a unit interval around the
    single value when x is constant, since np.histogram requires strictly
    increasing edges.
    """
    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 2:
        center = float(edges[0]) if edges.size else 0.0
        edges = np.array([center - 0.5, center + 0.5])
    return edges


def _numeric_reference(values: pd.Series, n_bins: int) -> dict[str, Any]:
    """Quantile bin edges + observed counts for one numeric feature."""
    x = values.to_numpy(dtype=float)
    x = x[~np.isnan(x)]
    edges = _quantile_bin_edges(x, n_bins)
    counts, edges = np.histogram(x, bins=edges)
    return {
        "bin_edges": edges.tolist(),
        "counts": counts.tolist(),
        "n": int(x.size),
    }


def _categorical_reference(values: pd.Series) -> dict[str, Any]:
    """Category frequencies for one categorical feature, summing to 1.

    Cast to str first so a missing value becomes its own "nan" category
    instead of being silently dropped, which would leave frequencies
    summing to less than 1.
    """
    freq = values.astype(str).value_counts(normalize=True)
    return {
        "frequencies": {str(k): float(v) for k, v in freq.items()},
        "n": int(values.shape[0]),
    }


def _score_reference(oos_proba: NDArray[np.float64], n_bins: int) -> dict[str, Any]:
    """Quantile bin edges + counts for the out-of-sample prediction-score distribution."""
    edges = _quantile_bin_edges(oos_proba, n_bins)
    counts, edges = np.histogram(oos_proba, bins=edges)
    return {
        "bin_edges": edges.tolist(),
        "counts": counts.tolist(),
        "n": int(oos_proba.size),
    }


def build_reference(
    features_df: pd.DataFrame,
    oos_proba: NDArray[np.float64],
    y: NDArray[np.int_],
    n_bins: int = 10,
) -> dict[str, Any]:
    """Build the champion's drift baseline: feature, score, and prevalence references.

    `features_df` is the training population (in-sample is fine here — see
    module docstring for why `oos_proba` must instead be out-of-sample).
    `y` is used only for prevalence.

    Numeric columns (FEATURE_SCHEMA.numeric) get quantile bin edges + counts;
    categorical columns (binary + multi_cat) get category frequencies. Only
    columns present in `features_df` are included, so this works for both the
    full feature space and a post-selection subset without a caller-side
    filter. An unseen category at check time is a plain dict-lookup miss for
    the caller to handle, not something this function guards against.
    """
    numeric_cols = [c for c in FEATURE_SCHEMA.numeric if c in features_df.columns]
    categorical_cols = [
        c
        for c in list(FEATURE_SCHEMA.binary) + list(FEATURE_SCHEMA.multi_cat)
        if c in features_df.columns
    ]

    return {
        "features": {
            "numeric": {
                col: _numeric_reference(features_df[col], n_bins)
                for col in numeric_cols
            },
            "categorical": {
                col: _categorical_reference(features_df[col])
                for col in categorical_cols
            },
        },
        "score": _score_reference(np.asarray(oos_proba, dtype=float), n_bins),
        "prevalence": float(np.mean(y)),
    }

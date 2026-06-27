"""Tree-family ColumnTransformer builder for the discovery loop and training pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

__all__ = ["build_preprocessor"]


def _cast_to_str(X: Any) -> Any:
    """Normalise mixed-dtype binary group to str so OHE receives a uniform type.

    Converts both int 0/1 columns (BINARY_INT_COLS) and Yes/No string columns
    (BINARY_STR_COLS) to numpy str before encoding — required because sklearn's
    OHE sees whichever dtype arrives and would treat '0'/'1' and 0/1 as separate
    categories if the two groups are not normalised first.
    """
    return np.array(X, dtype=str)


def build_preprocessor(
    binary: list[str],
    multi_cat: list[str],
    numeric: list[str],
) -> ColumnTransformer:
    """Return an unfitted tree-family ColumnTransformer for binary, multi-cat, and numeric columns.

    Designed for tree-based models (GBDT, RandomForest, XGBoost): trees are invariant
    to monotone feature transformations, so StandardScaler is deliberately omitted —
    scaling adds no information to split decisions and belongs only in a linear-family
    preprocessor. Must be fitted on the training split only so held-out statistics
    never leak into the preprocessing step.

    binary    — union of BINARY_STR_COLS and BINARY_INT_COLS; _cast_to_str normalises
                mixed dtypes to str before OHE so int 0/1 and Yes/No columns encode
                identically.
    multi_cat — multi-level categoricals; OHE without drop='first' — correct for trees
                (immune to the dummy-variable trap) and keeps all levels visible in
                importance/redundancy screens.
    numeric   — numeric columns; SimpleImputer(median) handles the 11 NaN zero-tenure
                rows (fit on X_train only). No scaling — trees don't need it.

    Model training note: if the linear baseline proves a genuine champion candidate, add a
    separate build_linear_preprocessor (drop='first' + StandardScaler) alongside this
    function — do not parameterise this one.

    Model training note: switch binary OHE to drop='if_binary' (remove _cast_to_str + Pipeline).
    Each binary column then produces 1 column instead of 2, yielding cleaner feature names
    in feature_columns.txt and SHAP plots. Deferred from Phase 4a: zero impact on tree
    model quality during discovery, and the test updates are most natural at the point
    the preprocessor is first serialized and logged.
    """
    binary_pipeline = Pipeline(
        steps=[
            (
                "to_str",
                FunctionTransformer(_cast_to_str, feature_names_out="one-to-one"),
            ),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("binary", binary_pipeline, binary),
            (
                "multi_cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                multi_cat,
            ),
            ("numeric", SimpleImputer(strategy="median"), numeric),
        ],
        remainder="drop",
    )

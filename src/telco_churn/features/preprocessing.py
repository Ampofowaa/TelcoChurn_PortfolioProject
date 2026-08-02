"""Tree-family and linear-family ColumnTransformer builders for the training pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

__all__ = [
    "build_preprocessor",
    "build_linear_preprocessor",
    "TENURE_COHORT_EDGES",
    "TENURE_COHORT_LABELS",
]

# Fixed cohort edges from the tenure survival shape documented in ANALYSIS.md §2.
# Five cohorts: 0–12 m (new, highest risk), 13–24 m, 25–48 m, 49–65 m, 65+ m (loyal).
# Edges are domain constants, not data-derived quantiles — no fit step needed and no
# train-set statistics leak through the binning branch.
TENURE_COHORT_EDGES: list[int] = [0, 12, 24, 48, 65, 72]
TENURE_COHORT_LABELS: list[str] = ["0-12m", "13-24m", "25-48m", "49-65m", "65+m"]


def _cast_to_str(X: Any) -> Any:
    """Normalise mixed-dtype binary group to str so OHE receives a uniform type.

    Used only by build_linear_preprocessor's binary branch. The tree path uses
    OHE(drop='if_binary') directly, which handles heterogeneous dtypes per-column.
    """
    return np.array(X, dtype=str)


def _bin_tenure(X: Any) -> Any:
    """Stateless cohort binning for tenure using fixed TENURE_COHORT_EDGES.

    Receives a 2-D input of shape (n, 1) from ColumnTransformer — either a numpy
    array or a DataFrame depending on the caller — and returns a (n, 1) object
    array of string cohort labels for the downstream OHE step.
    Stateless by construction — fixed edges need no fit and leak nothing.
    """
    binned = pd.cut(
        pd.Series(np.asarray(X)[:, 0]),
        bins=TENURE_COHORT_EDGES,
        labels=TENURE_COHORT_LABELS,
        include_lowest=True,
    ).astype(str)
    return np.array(binned).reshape(-1, 1)


def build_preprocessor(
    binary: list[str],
    multi_cat: list[str],
    numeric: list[str],
) -> ColumnTransformer:
    """Return an unfitted tree-family ColumnTransformer for binary, multi-cat, and numeric columns.

    No scaling: trees are invariant to monotone transforms, so StandardScaler
    is reserved for build_linear_preprocessor. binary columns use
    OHE(drop='if_binary') (1 column per feature, regardless of str/int dtype);
    multi_cat columns keep all levels (no drop='first' — irrelevant for trees,
    and needed for importance screens); numeric columns are median-imputed
    to cover the 11 NaN zero-tenure rows. Must be fit on the training split
    only, so held-out statistics never leak into preprocessing.

    Output is pandas, not the sklearn-default ndarray: LightGBM's sklearn
    wrapper auto-generates Column_0..N feature names for unnamed arrays, which
    false-positives its feature-name-consistency check at predict time if fit
    and predict ever pass through a Pipeline with mismatched output types.
    Named pandas output keeps names genuinely consistent and also reads
    better in feature_columns.txt/SHAP.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "binary",
                OneHotEncoder(
                    drop="if_binary", handle_unknown="ignore", sparse_output=False
                ),
                binary,
            ),
            (
                "multi_cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                multi_cat,
            ),
            ("numeric", SimpleImputer(strategy="median"), numeric),
        ],
        remainder="drop",
    )
    preprocessor.set_output(transform="pandas")
    return preprocessor


def build_linear_preprocessor(
    binary: list[str],
    multi_cat: list[str],
    numeric: list[str],
) -> ColumnTransformer:
    """Return an unfitted linear-family ColumnTransformer for binary, multi-cat, and numeric columns.

    For logistic regression and related linear models — differs from
    build_preprocessor (the tree path) in three ways:

    - OHE(drop='first') on all categoricals, avoiding the dummy-variable trap
      that inflates coefficient variance for linear models.
    - StandardScaler on non-tenure numerics, needed for comparable L2
      regularisation across differently-scaled features.
    - tenure is cohort-binned (via TENURE_COHORT_EDGES) rather than passed as
      a continuous numeric, since one linear slope can't represent the
      survival-distribution shape; per-cohort OHE coefficients are directly
      readable in the odds-ratio table. Callers still pass tenure inside
      `numeric` as usual — the split into its own branch happens internally,
      and the resulting `tenure_cohort` column is not a FeatureSchema column.

    Comparison-only: used by models/train/candidates.py for the
    DummyClassifier / LogisticRegressionCV candidates; nothing past model
    comparison imports it. Must be fit on the training split only, so
    held-out statistics never leak.
    """
    non_tenure_numeric = [c for c in numeric if c != "tenure"]

    binary_pipeline = Pipeline(
        steps=[
            (
                "to_str",
                FunctionTransformer(_cast_to_str, feature_names_out="one-to-one"),
            ),
            (
                "ohe",
                OneHotEncoder(
                    drop="first", handle_unknown="ignore", sparse_output=False
                ),
            ),
        ]
    )

    tenure_cohort_pipeline = Pipeline(
        steps=[
            (
                "bin",
                FunctionTransformer(_bin_tenure, feature_names_out="one-to-one"),
            ),
            (
                "ohe",
                OneHotEncoder(
                    drop="first", handle_unknown="ignore", sparse_output=False
                ),
            ),
        ]
    )

    transformers: list[Any] = [
        ("binary", binary_pipeline, binary),
        (
            "multi_cat",
            OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
            multi_cat,
        ),
    ]

    if "tenure" in numeric:
        transformers.append(("tenure_cohort", tenure_cohort_pipeline, ["tenure"]))

    if non_tenure_numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                non_tenure_numeric,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")

"""Development-partition loaders for calibrate.py, threshold.py, and register.py.

Split out of models/artifacts.py deliberately, not folded into it (PROJECT_PLAN.md's
Phase 8 Prerequisites, PR C note): artifacts.py is shared by error_analysis.py too,
and error_analysis.py must never become transitively reachable to data.split —
that reachability is exactly what the stage-9 guard (Phase 8's dvc.yaml work,
gated on this extraction) checks for. Taking the dev half of partition() here is
legitimate (calibrate.py/threshold.py/register.py all do it directly today), but
only because none of error_analysis.py's imports ever resolve to this module.
"""

from __future__ import annotations

import pandas as pd

from telco_churn.data.split import partition
from telco_churn.features.accessor import load_features
from telco_churn.features.build import TARGET_COL

__all__ = [
    "load_dev_partition",
    "load_dev_features",
    "load_dev_customer_ids",
]


def load_dev_partition() -> pd.DataFrame:
    """Return the full development-partition rows (customerid included), pre feature-subsetting.

    Pure and deterministic over the static processed-features file, so calling
    it repeatedly (once for load_dev_features, once for load_dev_customer_ids,
    once for threshold.py's build_dev_oof_screen_frame) reproduces the
    identical row order every time.
    """
    df = load_features()
    dev_df, _test_df = partition(df)
    return dev_df


def load_dev_features(committed_features: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Load the development-partition rows, restricted to the frozen committed feature set."""
    dev_df = load_dev_partition()
    return dev_df[committed_features], dev_df[TARGET_COL]


def load_dev_customer_ids() -> pd.Series:
    """Return the customerid Series for the development partition.

    Row-order-aligned with load_dev_features's (X_dev, y_dev) — both derive
    from the same load_dev_partition() call — so it can be zipped
    positionally with an OOF probability vector computed over (X_dev, y_dev)
    to build the dev_oof_predictions.parquet artifact.
    """
    return load_dev_partition()["customerid"].reset_index(drop=True)

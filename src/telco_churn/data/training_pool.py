"""training_pool's cyclical write path — the reserve mechanism's reshape step.

Write path 2 of two — the other is seed_training_pool() inside
data/ingest.py's own DVC stage. This module lives outside dvc.yaml entirely: prediction_log ⋈ prediction_outcomes is a live,
continuously-growing Postgres read, not a static, content-hashable file, so
DVC's DAG has nothing meaningful to hash it against. Called by
training_cycle.py (Phase 10b) once per matured reserve cohort — nothing in
this repo calls it yet, since training_cycle.py doesn't exist.

Deliberately not a modification of ingest.py: write path 1 is a deterministic
function of the static raw CSV (a real DVC dependency chain); this write path
reshapes genuinely new, request-time data with no such chain.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine

from telco_churn.data.tables import training_pool
from telco_churn.utils.logging import get_logger

__all__ = [
    "RESERVE_COL",
    "RAW_FEATURE_COLS",
    "build_training_pool_cohort",
    "write_training_pool_cohort",
]

logger = get_logger(__name__)

# Duplicated from data/split.py::RESERVE_COL, deliberately not imported — that
# module carries a __main__ block, and CLAUDE.md forbids importing a name out
# of one (test_no_module_imports_from_a_dunder_main_bearing_module). Both are
# just the literal training_pool/reserve_manifest column name; the true
# source of truth is data/tables.py's training_pool.reserve_month Column,
# which RAW_FEATURE_COLS below is derived from directly.
RESERVE_COL = "reserve_month"

# The 19 raw feature columns training_pool shares with customers_raw — derived
# from the table's own SQLAlchemy Core definition (data/tables.py) rather than
# hardcoded a second time, so this can never silently drift from the real
# schema. Excludes the surrogate PK, the pass-through customerid, and the two
# columns this module computes itself (churn, reserve_month).
RAW_FEATURE_COLS: list[str] = [
    c.name
    for c in training_pool.columns
    if c.name not in {"training_pool_id", "customerid", "churn", RESERVE_COL}
]

_COHORT_QUERY = text("""
    SELECT po.customerid, po.churned, pl.feature_snapshot
    FROM prediction_outcomes po
    JOIN LATERAL (
        SELECT pl2.feature_snapshot
        FROM prediction_log pl2
        WHERE pl2.customerid = po.customerid
          AND pl2.predicted_at <= po.observed_at
        ORDER BY pl2.predicted_at DESC
        LIMIT 1
    ) pl ON true
    WHERE po.customerid IN :customerids
    ORDER BY po.customerid
    """).bindparams(bindparam("customerids", expanding=True))


def _parse_feature_snapshot(raw: Any) -> dict[str, Any]:
    """feature_snapshot round-trips as a dict via psycopg2's JSONB adapter in
    the common case, but pandas' read_sql_query path can hand back a JSON
    string instead depending on driver/version — handle both."""
    return raw if isinstance(raw, dict) else json.loads(raw)


def build_training_pool_cohort(
    engine: Engine,
    customerids: pd.Series,
    reserve_manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Reshape a matured reserve cohort's prediction_log ⋈ prediction_outcomes
    rows into training_pool's flat, customers_raw-shaped schema.

    Joined on customerid plus "most recent prediction before observed_at"
    (never bare customerid — §B8's join contract), so a customer scored more
    than once before maturing is not fanned out across multiple rows.
    feature_snapshot's JSONB is flattened into RAW_FEATURE_COLS — raw fields
    only, deliberately excluding every prediction_log field that describes
    *how* a row was previously scored (probability, model_version,
    dual_score_mode, ...) — carrying a prior model's own output into new
    training data would be a leakage/circularity risk, not a convenience.
    `reserve_manifest` (reserve_ids()'s output) supplies the reserve_month
    tag per customerid — a cohort-reshape call may span more than one matured
    month, so this is a per-row lookup, not a single scalar.

    Raises ValueError if any requested customerid has no matured
    prediction_outcomes ⋈ prediction_log row, or is absent from
    reserve_manifest — a silent join miss here would insert an
    unlabeled/unattributed row into training_pool undetected, the same
    "fail loudly on an unmatched customerid" discipline data/split.py::partition()
    already applies to the (dev, test) partition.

    Does not write to training_pool itself, and does not call validate_raw —
    both are training_cycle.py's (Phase 10b) responsibility, sandwiched
    between this call and write_training_pool_cohort (§D4): reshape, then
    validate, then insert.
    """
    ids = pd.Index(customerids).unique().tolist()
    if not ids:
        raise ValueError("build_training_pool_cohort requires at least one customerid.")

    joined = pd.read_sql_query(_COHORT_QUERY, engine, params={"customerids": ids})
    matched = set(joined["customerid"])
    missing = set(ids) - matched
    if missing:
        raise ValueError(
            f"build_training_pool_cohort: {len(missing)} customerid(s) had no "
            "matured prediction_outcomes ⋈ prediction_log row: "
            f"{sorted(missing)[:10]}"
        )

    reserve_lookup = reserve_manifest.set_index("customerid")[RESERVE_COL]
    unmapped = set(ids) - set(reserve_lookup.index)
    if unmapped:
        raise ValueError(
            f"build_training_pool_cohort: {len(unmapped)} customerid(s) absent "
            f"from reserve_manifest: {sorted(unmapped)[:10]}"
        )

    flattened = pd.json_normalize(
        joined["feature_snapshot"].map(_parse_feature_snapshot)
    )
    missing_cols = set(RAW_FEATURE_COLS) - set(flattened.columns)
    if missing_cols:
        raise ValueError(
            f"build_training_pool_cohort: feature_snapshot missing expected "
            f"column(s): {sorted(missing_cols)}"
        )

    result = flattened[RAW_FEATURE_COLS].copy()
    result.insert(0, "customerid", joined["customerid"].to_numpy())
    result["churn"] = joined["churned"].astype(int).to_numpy()
    result[RESERVE_COL] = result["customerid"].map(reserve_lookup).astype("Int16")
    return result.sort_values("customerid").reset_index(drop=True)


def write_training_pool_cohort(df: pd.DataFrame, engine: Engine) -> int:
    """Append a reshaped cohort (build_training_pool_cohort's output) to
    training_pool.

    Pure append — never delete-and-reload: unlike seed_training_pool()'s
    reserve_month IS NULL seed, every row here is genuinely new (a matured
    reserve cohort is appended exactly once by design), so there is nothing
    to overwrite. Returns the number of rows written.
    """
    df.to_sql(
        "training_pool",
        engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=1000,
    )
    logger.info("training_pool_cohort_written", rows_written=len(df))
    return len(df)

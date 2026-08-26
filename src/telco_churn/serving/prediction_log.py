"""Builds and persists `prediction_log` rows — one row per scored customer.

`build_log_rows` is a pure row-builder with no single-vs-batch branching
inside it: `predict.py` already treats "one customer" as "a batch of one"
before scoring ever runs (its own module docstring says as much), so by the
time this function runs, both call sites (`POST /predict`, `POST
/predict/batch`) already hold the same shape of input — a list of feature
dicts, a `ScoredBatch`, and a `resolution_kind` per row — length 1 for a
single call, length N for a batch.

`write_log_rows` is the async I/O boundary: one bulk `INSERT` via
`data/tables.py`'s `prediction_log` Table. It raises normally on failure —
the fail-open, log-and-swallow discipline (plus the
`prediction_log_write_failures_total` counter) belongs to the caller's
`BackgroundTasks` wrapper (`serving/app.py`), not to this module. Same
"pure function does no error handling, the caller at the system boundary
does" split this project's Code Style conventions require.

`runtime` (not just `scored`) is a required input to `build_log_rows`:
`ScoredBatch` carries each row's `served_source` ("champion"/"challenger")
and `served_model_version`, but not `run_id`, nor the challenger's own
`model_version` for a row that was only shadow-evaluated and never actually
routed there. Both live on `ModelBundle`, resolved here from whichever
`ServingRuntime` produced `scored`.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from sqlalchemy import insert
from sqlalchemy.ext.asyncio import AsyncEngine

from telco_churn.data.tables import prediction_log
from telco_churn.serving.predict import ScoredBatch, ServingRuntime

__all__ = ["build_log_rows", "write_log_rows"]

_ResolutionKind = Literal["id_only", "full_inline", "partial_override"]


def _resolve_serving_identity(
    scored: ScoredBatch, runtime: ServingRuntime, i: int
) -> tuple[str, str]:
    """(model_version, run_id) for whichever bundle actually served row i —
    champion, unless canary routed this specific row to the challenger."""
    if scored.served_source[i] == "challenger":
        assert runtime.challenger is not None
        return runtime.challenger.model_version, runtime.challenger.run_id
    return runtime.champion.model_version, runtime.champion.run_id


def _resolve_dual_score_fields(
    scored: ScoredBatch, runtime: ServingRuntime, i: int
) -> tuple[str | None, str | None, float | None, float | None]:
    """Per-row (dual_score_mode, challenger_version, challenger_probability,
    champion_probability) — all four null together, or all four populated
    together, never a partial mix.

    "Evaluated" is read straight off `scored.challenger_proba`'s own NaN
    sentinel — `predict.py::score_request` only ever fills a row's slot when
    the challenger actually scored it — rather than re-derived from
    `cfg.serving.shadow`/`canary`, the same array `score_request` itself used
    to decide what to score.
    """
    challenger_proba = scored.challenger_proba
    if challenger_proba is None or np.isnan(challenger_proba[i]):
        return None, None, None, None

    assert runtime.challenger is not None
    mode = "canary" if scored.served_source[i] == "challenger" else "shadow"
    return (
        mode,
        runtime.challenger.model_version,
        float(challenger_proba[i]),
        float(scored.champion_proba[i]),
    )


def build_log_rows(
    feature_rows: list[dict[str, Any]],
    scored: ScoredBatch,
    runtime: ServingRuntime,
    contact: list[bool] | None,
    route: Literal["single", "batch", "nightly_batch"],
    request_id: str,
    resolution_kinds: list[_ResolutionKind | None],
) -> list[dict[str, Any]]:
    """One `prediction_log` row per scored customer.

    `feature_rows[i]` is logged verbatim as `feature_snapshot` — the exact
    row `score_request` scored, not a re-derivation from `customerid`. It
    cannot be reconstructed later by re-joining `customerid` back to
    `customers_crm`, since that table isn't a time-stamped history.

    `contact` is `None` for a single-customer call (`/predict` never
    computes a capacity-ranked contact decision — see
    `serving/schemas.py::PredictResponse`) and a length-N list of bools for
    batch. `resolution_kinds` is always length-N: `/predict` passes a
    length-1 list holding its own `PredictRequest.resolution_kind`,
    `/predict/batch` passes `resolve_batch_rows`'s per-row classification.
    """
    n = len(feature_rows)
    customerid_missing = scored.customer_ids.isna().to_numpy()
    rows: list[dict[str, Any]] = []
    for i in range(n):
        model_version, run_id = _resolve_serving_identity(scored, runtime, i)
        (
            dual_score_mode,
            challenger_version,
            challenger_probability,
            champion_probability,
        ) = _resolve_dual_score_fields(scored, runtime, i)
        probability = float(scored.served_proba[i])
        threshold = float(scored.served_threshold[i])
        rows.append(
            {
                "request_id": request_id,
                "customerid": (
                    None if customerid_missing[i] else str(scored.customer_ids.iloc[i])
                ),
                "route": route,
                "feature_snapshot": feature_rows[i],
                "model_version": model_version,
                "run_id": run_id,
                "probability": probability,
                "threshold": threshold,
                "decision": probability >= threshold,
                "contact": contact[i] if contact is not None else None,
                "dual_score_mode": dual_score_mode,
                "challenger_version": challenger_version,
                "challenger_probability": challenger_probability,
                "champion_probability": champion_probability,
                "resolution_kind": resolution_kinds[i],
            }
        )
    return rows


async def write_log_rows(rows: list[dict[str, Any]], engine: AsyncEngine) -> int:
    """Bulk-insert already-built `prediction_log` rows in one round trip.

    A no-op returning 0 for an empty list (e.g. a `/predict/batch` call
    where every row failed validation/resolution before scoring ever ran)
    rather than issuing a zero-row `INSERT`. Raises normally on failure —
    the fail-open, log-and-swallow discipline belongs to the caller's
    `BackgroundTasks` wrapper (`serving/app.py`), not this I/O boundary
    function.
    """
    if not rows:
        return 0
    async with engine.begin() as conn:
        await conn.execute(insert(prediction_log), rows)
    return len(rows)

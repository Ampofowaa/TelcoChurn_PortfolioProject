"""Postgres ID-resolution for serving/app.py's GET /customer/{customerid} and
POST /predict/batch — kept separate from HTTP routing so it's importable and
unit-testable with no FastAPI/ASGI machinery involved, the same one-concern-
per-module split this project already applies to predict.py/contact_policy.py/
policy_config.py/schemas.py.

Three item shapes collapse to one rule: a row missing any required field is
resolved from Postgres by customerid, then any fields the caller did supply
override the looked-up base — a partial-override item and an ID-only item
differ only in how much of the override dict is non-empty. A complete inline
row never touches Postgres at all, whether or not it also carries a
customerid (PROJECT_PLAN.md's Phase 9 note).
"""

from __future__ import annotations

from typing import Any, Final, Literal

from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from telco_churn.serving.schemas import (
    BatchPredictionError,
    BatchPredictItem,
    CustomerFeatures,
)

__all__ = [
    "LOOKUP_COLUMNS",
    "is_complete",
    "format_validation_error",
    "lookup_customers",
    "resolve_batch_rows",
]

# The 19 raw feature columns customers_raw holds — CustomerFeatures's field
# set minus the pass-through customerid, verified equal to FEATURE_SCHEMA's
# raw columns by test_schemas.py. Reused both for GET /customer/{id}'s SELECT
# list and for POST /predict/batch's lookup-vs-inline completeness check.
LOOKUP_COLUMNS: Final[list[str]] = [
    name for name in CustomerFeatures.model_fields if name != "customerid"
]
# totalcharges excluded: it is legitimately None for a zero-tenure customer
# (or prospect), so its absence alone must never trigger a Postgres lookup —
# only a genuinely missing *required* field should.
_REQUIRED_FOR_COMPLETE: Final[list[str]] = [
    name for name in LOOKUP_COLUMNS if name != "totalcharges"
]


def is_complete(item: BatchPredictItem) -> bool:
    """True iff every field the model needs (besides customerid/totalcharges) is present."""
    return all(getattr(item, name) is not None for name in _REQUIRED_FOR_COMPLETE)


def format_validation_error(exc: ValidationError) -> str:
    """First error of a Pydantic ValidationError as one short 'field: message' string."""
    first = exc.errors()[0]
    loc = ".".join(str(part) for part in first["loc"])
    return f"{loc}: {first['msg']}" if loc else str(first["msg"])


async def lookup_customers(
    customerids: list[str], engine: AsyncEngine
) -> dict[str, dict[str, Any]]:
    """Batch-resolve customerids against customers_crm in one round trip.

    Backs both GET /customer/{customerid} (a length-1 list) and
    /predict/batch's ID-resolution path — the same query either way, so a
    caller never pays N round trips for an N-row batch. customers_crm, not
    customers_raw: a live lookup must hit a source distinct from the frozen
    training snapshot. Each returned row carries crm_snapshot_at alongside
    the feature columns — resolve_batch_rows
    ignores it (CustomerFeatures has no such field and Pydantic drops unknown
    keys by default); GET /customer/{customerid} surfaces it explicitly via
    CustomerLookupResponse.
    """
    if not customerids:
        return {}
    columns = ", ".join(["customerid", *LOOKUP_COLUMNS, "crm_snapshot_at"])
    stmt = text(f"SELECT {columns} FROM customers_crm WHERE customerid = ANY(:ids)")
    async with engine.connect() as conn:
        result = await conn.execute(stmt, {"ids": customerids})
        rows = result.mappings().all()
    return {str(row["customerid"]): dict(row) for row in rows}


_ResolutionKind = Literal["id_only", "full_inline", "partial_override"]


def resolve_batch_rows(
    validated: dict[int, BatchPredictItem],
    looked_up: dict[str, dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[int],
    list[str | None],
    list[BatchPredictionError],
    list[_ResolutionKind | None],
]:
    """Turn each validated batch item into an engineered-feature-ready row, or an error.

    The fifth return value, resolution_kinds, is this function's own
    classification of how each row was actually built — free, since
    is_complete(item) and the override dict's emptiness are already computed
    below to resolve the row itself, not derived separately. Zipped 1:1 with
    rows/row_indices/row_customer_ids (a row that errors out contributes to
    none of the five).
    Typed `| None` to match `serving/prediction_log.py::build_log_rows`'s
    parameter shape exactly (list is invariant) — every row this function
    actually resolves gets a concrete classification, never `None`.
    """
    rows: list[dict[str, Any]] = []
    row_indices: list[int] = []
    row_customer_ids: list[str | None] = []
    errors: list[BatchPredictionError] = []
    resolution_kinds: list[_ResolutionKind | None] = []
    for idx, item in validated.items():
        resolution_kind: _ResolutionKind
        if is_complete(item):
            resolved = item.model_dump(exclude={"customerid"})
            customerid = item.customerid
            resolution_kind = "full_inline"
        else:
            if item.customerid is None:
                errors.append(
                    BatchPredictionError(
                        index=idx,
                        customerid=None,
                        reason="validation_error",
                        detail=(
                            "incomplete feature set and no customerid to "
                            "resolve the remaining fields against"
                        ),
                    )
                )
                continue
            base = looked_up.get(item.customerid)
            if base is None:
                errors.append(
                    BatchPredictionError(
                        index=idx,
                        customerid=item.customerid,
                        reason="resolution_error",
                        detail=f"customerid '{item.customerid}' not found",
                    )
                )
                continue
            base_features = CustomerFeatures.model_validate(base).model_dump(
                exclude={"customerid"}
            )
            override = {
                key: value
                for key, value in item.model_dump(exclude={"customerid"}).items()
                if value is not None
            }
            resolved = {**base_features, **override}
            customerid = item.customerid
            resolution_kind = "partial_override" if override else "id_only"
        rows.append(resolved)
        row_indices.append(idx)
        row_customer_ids.append(customerid)
        resolution_kinds.append(resolution_kind)
    return rows, row_indices, row_customer_ids, errors, resolution_kinds

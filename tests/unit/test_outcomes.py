"""Unit tests: serving/outcomes.py's pure row-builder, build_outcome_row()."""

from __future__ import annotations

from datetime import UTC, datetime

from telco_churn.serving.outcomes import build_outcome_row

__all__: list[str] = []


def test_build_outcome_row_shape() -> None:
    observed_at = datetime(2026, 8, 22, tzinfo=UTC)

    row = build_outcome_row("cust-0001", True, observed_at, "synthetic_seed")

    assert row == {
        "customerid": "cust-0001",
        "churned": True,
        "observed_at": observed_at,
        "source": "synthetic_seed",
    }


def test_build_outcome_row_does_not_set_server_generated_fields() -> None:
    row = build_outcome_row(
        "cust-0002", False, datetime(2026, 1, 1, tzinfo=UTC), "manual"
    )

    assert "outcome_id" not in row
    assert "recorded_at" not in row


def test_build_outcome_row_accepts_each_source_literal() -> None:
    observed_at = datetime(2026, 1, 1, tzinfo=UTC)
    for source in ("crm_sync", "manual", "synthetic_seed"):
        row = build_outcome_row("cust-0003", True, observed_at, source)  # type: ignore[arg-type]
        assert row["source"] == source

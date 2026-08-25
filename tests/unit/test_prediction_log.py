"""Unit tests: serving/prediction_log.py's pure row-builder, build_log_rows().

ModelBundle/ServingRuntime/ScoredBatch are constructed directly rather than
through a real MLflow-loaded model — build_log_rows only ever reads
model_version/run_id off a bundle and the numeric/label fields off a
ScoredBatch, so a full model/preprocessor/explainer is unnecessary weight
for these cases.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from telco_churn.serving import predict
from telco_churn.serving.prediction_log import build_log_rows, write_log_rows

__all__: list[str] = []


def _bundle(model_version: str, run_id: str) -> predict.ModelBundle:
    return predict.ModelBundle(
        model=None,
        preprocessor=None,
        booster=None,
        explainer=None,
        model_version=model_version,
        run_id=run_id,
        logged_model_id=f"lm-{model_version}",
        committed_features=[],
        scenarios={},
        thresholds={},
        base_scenario_name="base",
        loaded_at=datetime.now(UTC),
    )


def _scored_batch(
    customer_ids: list[str | None],
    served_proba: list[float],
    served_source: list[str],
    served_model_version: list[str],
    served_threshold: list[float],
    champion_proba: list[float],
    challenger_proba: list[float] | None,
) -> predict.ScoredBatch:
    return predict.ScoredBatch(
        customer_ids=pd.Series(customer_ids),
        served_proba=np.array(served_proba, dtype=float),
        served_source=served_source,
        served_model_version=served_model_version,
        served_threshold=np.array(served_threshold, dtype=float),
        champion_proba=np.array(champion_proba, dtype=float),
        challenger_proba=(
            np.array(challenger_proba, dtype=float)
            if challenger_proba is not None
            else None
        ),
        dual_score_rows=[],
    )


def test_single_row_no_challenger_leaves_dual_score_fields_null() -> None:
    champion = _bundle("1", "run-champ")
    runtime = predict.ServingRuntime(champion=champion, challenger=None)
    scored = _scored_batch(
        customer_ids=["cust-A"],
        served_proba=[0.42],
        served_source=["champion"],
        served_model_version=["1"],
        served_threshold=[0.4],
        champion_proba=[0.42],
        challenger_proba=None,
    )

    rows = build_log_rows(
        feature_rows=[{"tenure": 5}],
        scored=scored,
        runtime=runtime,
        contact=None,
        route="single",
        request_id="req-1",
        resolution_kinds=["full_inline"],
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["request_id"] == "req-1"
    assert row["route"] == "single"
    assert row["customerid"] == "cust-A"
    assert row["feature_snapshot"] == {"tenure": 5}
    assert row["model_version"] == "1"
    assert row["run_id"] == "run-champ"
    assert row["probability"] == pytest.approx(0.42)
    assert row["threshold"] == pytest.approx(0.4)
    assert row["decision"] is True
    assert row["contact"] is None
    assert row["dual_score_mode"] is None
    assert row["challenger_version"] is None
    assert row["challenger_probability"] is None
    assert row["champion_probability"] is None
    assert row["resolution_kind"] == "full_inline"


def test_batch_row_missing_customerid_becomes_none() -> None:
    champion = _bundle("3", "run-champ3")
    runtime = predict.ServingRuntime(champion=champion, challenger=None)
    scored = _scored_batch(
        customer_ids=[None, "cust-B"],
        served_proba=[0.1, 0.9],
        served_source=["champion", "champion"],
        served_model_version=["3", "3"],
        served_threshold=[0.4, 0.4],
        champion_proba=[0.1, 0.9],
        challenger_proba=None,
    )

    rows = build_log_rows(
        feature_rows=[{"tenure": 1}, {"tenure": 2}],
        scored=scored,
        runtime=runtime,
        contact=[False, True],
        route="batch",
        request_id="req-2",
        resolution_kinds=["id_only", "partial_override"],
    )

    assert rows[0]["customerid"] is None
    assert rows[1]["customerid"] == "cust-B"
    assert rows[0]["contact"] is False
    assert rows[1]["contact"] is True
    assert rows[0]["decision"] is False
    assert rows[1]["decision"] is True
    assert rows[0]["resolution_kind"] == "id_only"
    assert rows[1]["resolution_kind"] == "partial_override"


def test_shadow_row_serves_champion_but_logs_both_scores() -> None:
    """prediction_logging_plan.md §B2's shadow.enabled=true, canary.enabled=false
    worked example: served_source stays "champion" — shadow never routes."""
    champion = _bundle("1", "run-champ1")
    challenger = _bundle("2", "run-chal2")
    runtime = predict.ServingRuntime(champion=champion, challenger=challenger)
    scored = _scored_batch(
        customer_ids=["CUST-A"],
        served_proba=[0.3813],
        served_source=["champion"],
        served_model_version=["1"],
        served_threshold=[0.4],
        champion_proba=[0.3813],
        challenger_proba=[0.3818],
    )

    rows = build_log_rows(
        feature_rows=[{"tenure": 5}],
        scored=scored,
        runtime=runtime,
        contact=None,
        route="single",
        request_id="req-3",
        resolution_kinds=[None],
    )

    row = rows[0]
    assert row["model_version"] == "1"
    assert row["run_id"] == "run-champ1"
    assert row["probability"] == pytest.approx(0.3813)
    assert row["dual_score_mode"] == "shadow"
    assert row["challenger_version"] == "2"
    assert row["challenger_probability"] == pytest.approx(0.3818)
    assert row["champion_probability"] == pytest.approx(0.3813)
    assert row["resolution_kind"] is None


def test_canary_row_serves_challenger_and_preserves_champion_score() -> None:
    """prediction_logging_plan.md §B2's canary-routed worked example:
    model_version/probability flip to the challenger's, but champion_probability
    still preserves what the champion alone would have said."""
    champion = _bundle("1", "run-champ1")
    challenger = _bundle("2", "run-chal2")
    runtime = predict.ServingRuntime(champion=champion, challenger=challenger)
    scored = _scored_batch(
        customer_ids=["CUST-A"],
        served_proba=[0.5820],
        served_source=["challenger"],
        served_model_version=["2"],
        served_threshold=[0.45],
        champion_proba=[0.5510],
        challenger_proba=[0.5820],
    )

    rows = build_log_rows(
        feature_rows=[{"tenure": 5}],
        scored=scored,
        runtime=runtime,
        contact=[True],
        route="batch",
        request_id="req-4",
        resolution_kinds=["id_only"],
    )

    row = rows[0]
    assert row["model_version"] == "2"
    assert row["run_id"] == "run-chal2"
    assert row["probability"] == pytest.approx(0.5820)
    assert row["dual_score_mode"] == "canary"
    assert row["challenger_version"] == "2"
    assert row["challenger_probability"] == pytest.approx(0.5820)
    assert row["champion_probability"] == pytest.approx(0.5510)


def test_empty_feature_rows_returns_empty_list() -> None:
    champion = _bundle("1", "run-champ")
    runtime = predict.ServingRuntime(champion=champion, challenger=None)
    scored = _scored_batch(
        customer_ids=[],
        served_proba=[],
        served_source=[],
        served_model_version=[],
        served_threshold=[],
        champion_proba=[],
        challenger_proba=None,
    )

    rows = build_log_rows(
        feature_rows=[],
        scored=scored,
        runtime=runtime,
        contact=[],
        route="batch",
        request_id="req-5",
        resolution_kinds=[],
    )

    assert rows == []


def test_write_log_rows_empty_list_short_circuits_without_touching_engine() -> None:
    written = asyncio.run(write_log_rows([], engine=None))  # type: ignore[arg-type]
    assert written == 0

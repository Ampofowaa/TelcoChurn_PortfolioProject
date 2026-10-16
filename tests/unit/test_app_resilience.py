"""Unit tests: serving/app.py stays responsive when its dependencies stall.

Two failure modes, both seen on the deployed box:
- a customers_crm lookup on a silently dead Postgres connection, which used
  to hold the request open until the kernel gave up on the socket;
- the champion hot-reload poll, which used to run blocking MLflow I/O on the
  event loop itself.

No Postgres or MLflow is needed: the lookup and the model-resolution calls
are monkeypatched, so these exercise only the HTTP layer and the lifespan.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

import telco_churn.serving.app as app_module
from telco_churn.serving import customer_lookup, predict


class _FakeChampion:
    """Only the two fields the lifespan logs once the initial load returns."""

    model_version = "1"
    run_id = "fake-run"


class _FakeRuntime:
    """Stands in for predict.ServingRuntime — the lookup fails before any
    route would read a bundle off it."""

    champion = _FakeChampion()


@pytest.fixture
def poll_threads(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Patch the lifespan's model loading and shorten its timeouts.

    Records which thread the initial load (on the event loop) and each
    refresh tick ran on, and signals once the first refresh has happened.
    """
    record: dict[str, Any] = {"refreshed": threading.Event()}
    real_compose_config = app_module.compose_config

    def _compose(overrides: list[str] | None = None) -> Any:
        cfg = real_compose_config(overrides)
        cfg.serving.poll_interval_seconds = 0.01
        cfg.serving.database.lookup_timeout_seconds = 0.05
        cfg.serving.auth.enabled = False
        return cfg

    async def _fake_initial_load(cfg: Any) -> _FakeRuntime:
        record["loop_thread"] = threading.get_ident()
        return _FakeRuntime()

    def _fake_refresh(cfg: Any, current: Any) -> Any:
        record["refresh_thread"] = threading.get_ident()
        record["refreshed"].set()
        return current

    monkeypatch.setattr(app_module, "compose_config", _compose)
    monkeypatch.setattr(predict, "load_serving_state_with_retry", _fake_initial_load)
    monkeypatch.setattr(predict, "resolve_serving_model", _fake_refresh)
    return record


@pytest.fixture
def stalled_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    """A customers_crm lookup that never returns within the test's timeout."""

    async def _hang(customerids: list[str], engine: Any) -> dict[str, Any]:
        await asyncio.sleep(10)
        return {}

    monkeypatch.setattr(customer_lookup, "lookup_customers", _hang)
    monkeypatch.setattr(app_module, "get_async_engine", lambda: None)


@pytest.fixture
def client(poll_threads: dict[str, Any]) -> Iterator[TestClient]:
    """TestClient with the lifespan running (context-manager form)."""
    with TestClient(app_module.app) as test_client:
        yield test_client


def _timeouts(route: str) -> float:
    return (
        REGISTRY.get_sample_value("customer_lookup_timeouts_total", {"route": route})
        or 0.0
    )


def test_customer_lookup_times_out_with_503(
    client: TestClient, stalled_lookup: None
) -> None:
    before = _timeouts("customer")

    response = client.get("/customer/0002-ORFBO")

    assert response.status_code == 503
    assert "timed out" in response.json()["detail"]
    assert _timeouts("customer") == before + 1


def test_batch_id_resolution_times_out_with_503(
    client: TestClient, stalled_lookup: None
) -> None:
    before = _timeouts("batch")

    response = client.post("/predict/batch", json=[{"customerid": "0002-ORFBO"}])

    assert response.status_code == 503
    assert _timeouts("batch") == before + 1


def test_hot_reload_poll_runs_off_the_event_loop(
    client: TestClient, poll_threads: dict[str, Any]
) -> None:
    assert poll_threads["refreshed"].wait(timeout=5)
    assert poll_threads["refresh_thread"] != poll_threads["loop_thread"]


def test_health_answers_while_a_refresh_is_blocked(
    monkeypatch: pytest.MonkeyPatch, poll_threads: dict[str, Any]
) -> None:
    """A refresh stuck on MLflow must not stall unrelated requests."""
    release = threading.Event()
    entered = threading.Event()

    def _blocking_refresh(cfg: Any, current: Any) -> Any:
        entered.set()
        release.wait(timeout=10)
        return current

    monkeypatch.setattr(predict, "resolve_serving_model", _blocking_refresh)
    try:
        with TestClient(app_module.app) as test_client:
            assert entered.wait(timeout=5)
            response = test_client.get("/health")
            assert response.status_code == 200
    finally:
        release.set()

"""Integration tests: serving/app.py as a real uvicorn subprocess.

tests/unit/test_predict.py already covers load_serving_state_with_retry and
resolve_champion_model via direct function calls (asyncio.run, no real
socket). This file is the one place the actual process boundary is under
test: a real `uvicorn telco_churn.serving.app:app` process, over a real TCP
socket, polled with real HTTP requests — the same two outcomes
CLAUDE.md/serving/predict.py's own module docstring name for the cold-start
path:

- the happy path: MLflow has a `champion` alias to resolve, so the process
  becomes ready and actually serves a prediction ("exit-0" — nothing about
  boot fails);
- the degraded path: no `champion` alias exists anywhere in a genuinely
  empty registry, so `load_serving_state_with_retry`'s backoff loop keeps
  retrying forever — `/ready` must stay 503 (never crash the process, never
  flip to serving) for as long as we're willing to wait and check
  ("stays-503").

No Postgres container here: neither `/health`, `/ready`, nor `/predict`
touches Postgres (see serving/app.py's own `/predict` docstring), so this
file only needs the serving_champion fixture (conftest.py), not
serving_postgres_url/serving_env.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import mlflow
import pytest

from telco_churn.utils.paths import get_project_root

pytestmark = pytest.mark.integration

_PROJECT_ROOT = get_project_root()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _poll_status(url: str, timeout: float, interval: float = 0.25) -> int | None:
    """Poll `url` until it responds (any status) or `timeout` elapses.

    Returns the last-seen HTTP status code, or None if the server never
    accepted a connection at all within the timeout — the two failure modes
    ("connection refused" vs. "responded 503") mean different things to the
    two tests below and must not be collapsed into one.
    """
    deadline = time.monotonic() + timeout
    last_status: int | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                last_status = resp.status
        except urllib.error.HTTPError as exc:
            last_status = exc.code
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        else:
            return last_status
        time.sleep(interval)
    return last_status


def _empty_mlflow_store(tmp_path: Path) -> str:
    """A real, reachable MLflow store with no registered model at all — the
    genuine "nothing to resolve" cold-start case, not a mocked failure.
    """
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.create_experiment(
        "test_predict_subprocess_empty",
        artifact_location=(tmp_path / "artifacts").as_uri(),
    )
    return tracking_uri


def _spawn_uvicorn(
    port: int, env: dict[str, str], log_path: Path
) -> subprocess.Popen[bytes]:
    """Launch uvicorn with combined stdout/stderr redirected to `log_path` —
    a real file rather than subprocess.PIPE, so a long-running server can't
    deadlock this test by filling an unread pipe buffer, and so a failing
    assertion below can still show what the server itself logged.
    """
    log_file = log_path.open("wb")
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "telco_churn.serving.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        env=env,
        cwd=str(_PROJECT_ROOT),
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )


def _tail(log_path: Path, n_chars: int = 4000) -> str:
    if not log_path.exists():
        return "<no log file>"
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return text[-n_chars:]


def _terminate(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=15)


def test_uvicorn_subprocess_reaches_ready_and_serves_a_prediction(
    serving_champion: dict[str, Any], tmp_path: Path
) -> None:
    """A real uvicorn process, given a real champion alias to resolve, must
    reach /ready (200) and then actually serve /predict — the happy path a
    plain in-process resolve_serving_model call can't exercise, since it
    never opens a socket."""
    port = _free_port()
    log_path = tmp_path / "uvicorn.log"
    env = {**os.environ, "MLFLOW_TRACKING_URI": str(serving_champion["tracking_uri"])}
    proc = _spawn_uvicorn(port, env, log_path)
    try:
        status = _poll_status(f"http://127.0.0.1:{port}/ready", timeout=60.0)
        assert status == 200, (
            f"server never became ready (last status: {status}); "
            f"champion resolution failed at the real process boundary\n"
            f"--- uvicorn log tail ---\n{_tail(log_path)}"
        )

        request = urllib.request.Request(f"http://127.0.0.1:{port}/ready", method="GET")
        with urllib.request.urlopen(request, timeout=5.0) as resp:
            body = json.loads(resp.read())
        assert body["model_version"] == serving_champion["model_version"]

        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/health", timeout=5.0
        ) as resp:
            assert resp.status == 200

        assert proc.poll() is None, "server process exited unexpectedly"
    finally:
        _terminate(proc)

    assert proc.returncode is not None, "server process failed to exit after terminate"


def test_uvicorn_subprocess_stays_503_when_no_champion_is_registered(
    tmp_path: Path,
) -> None:
    """No `champion` alias exists anywhere in this store — the process must
    never crash and must never report ready; /ready stays 503 for as long as
    we're willing to observe it, exercising load_serving_state_with_retry's
    backoff loop for real rather than asserting it in-process.
    """
    tracking_uri = _empty_mlflow_store(tmp_path)
    port = _free_port()
    log_path = tmp_path / "uvicorn.log"
    env = {**os.environ, "MLFLOW_TRACKING_URI": tracking_uri}
    proc = _spawn_uvicorn(port, env, log_path)
    try:
        # /health is liveness-only and must never be gated on the champion
        # ever resolving — serving/app.py's lifespan runs the initial
        # champion load in a background task specifically so uvicorn's own
        # listening socket (which it only opens once ASGI lifespan startup
        # returns) isn't held hostage by a retry loop with nothing to
        # resolve. A generous timeout here since real MLflow/model-load
        # machinery (even failing) still costs real wall-clock time on the
        # first attempt before the loop's own asyncio.sleep hands control
        # back to uvicorn's startup handshake.
        health_status = _poll_status(f"http://127.0.0.1:{port}/health", timeout=45.0)
        assert health_status == 200, (
            f"server never became reachable at all (last status: "
            f"{health_status})\n--- uvicorn log tail ---\n{_tail(log_path)}"
        )

        deadline = time.monotonic() + 5.0
        observed_statuses: set[int] = set()
        while time.monotonic() < deadline:
            status = _poll_status(f"http://127.0.0.1:{port}/ready", timeout=1.0)
            if status is not None:
                observed_statuses.add(status)
            assert proc.poll() is None, "server crashed while champion never resolved"

        assert observed_statuses == {503}, (
            f"expected /ready to stay 503 with no champion registered, saw: "
            f"{observed_statuses}"
        )
    finally:
        _terminate(proc)

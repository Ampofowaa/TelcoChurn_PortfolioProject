"""Unit tests for src/telco_churn/utils/logging.py."""

from __future__ import annotations

import io
import logging
import sys
from collections.abc import Generator

import pytest
import structlog

from telco_churn.utils.logging import configure_logging, get_logger


@pytest.fixture(autouse=True)
def _restore_root_logger() -> Generator[None]:
    """Save/restore root logger state so configure_logging calls don't leak between tests."""
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_level = root.level
    yield
    root.handlers[:] = original_handlers
    root.setLevel(original_level)


def test_configure_logging_default_level_is_info() -> None:
    configure_logging()
    assert logging.getLogger().level == logging.INFO


def test_configure_logging_respects_requested_level() -> None:
    configure_logging("DEBUG")
    assert logging.getLogger().level == logging.DEBUG


def test_configure_logging_adds_one_stream_handler() -> None:
    root = logging.getLogger()
    count_before = len(root.handlers)
    configure_logging()
    assert len(root.handlers) == count_before + 1
    assert isinstance(root.handlers[-1], logging.StreamHandler)


def test_configure_logging_handler_uses_processor_formatter() -> None:
    configure_logging()
    root = logging.getLogger()
    assert any(
        isinstance(h.formatter, structlog.stdlib.ProcessorFormatter)
        for h in root.handlers
    )


def test_get_logger_exposes_standard_log_methods() -> None:
    logger = get_logger("test.module")
    assert callable(getattr(logger, "info", None))
    assert callable(getattr(logger, "error", None))
    assert callable(getattr(logger, "warning", None))


def test_configure_logging_defaults_to_console_renderer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LOG_FORMAT", raising=False)
    configure_logging()
    handler = logging.getLogger().handlers[-1]
    assert isinstance(handler.formatter.processors[-1], structlog.dev.ConsoleRenderer)


def test_configure_logging_log_format_json_uses_json_renderer() -> None:
    configure_logging(log_format="json")
    handler = logging.getLogger().handlers[-1]
    assert isinstance(
        handler.formatter.processors[-1], structlog.processors.JSONRenderer
    )


def test_configure_logging_reads_log_format_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOG_FORMAT", "json")
    configure_logging()
    handler = logging.getLogger().handlers[-1]
    assert isinstance(
        handler.formatter.processors[-1], structlog.processors.JSONRenderer
    )


def test_configure_logging_explicit_arg_overrides_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOG_FORMAT", "json")
    configure_logging(log_format="console")
    handler = logging.getLogger().handlers[-1]
    assert isinstance(handler.formatter.processors[-1], structlog.dev.ConsoleRenderer)


# ---------------------------------------------------------------------------
# UTF-8 stdout/stderr reconfigure (Windows cp1252-console workaround)
# ---------------------------------------------------------------------------


def test_configure_logging_reconfigures_non_utf8_stdout_to_utf8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Windows' default cp1252 console encoding can't render output some
    dependencies emit unprompted (e.g. MLflow's emoji run-URL messages) --
    configure_logging must reconfigure a non-UTF-8 stdout to UTF-8 in place."""
    fake_stdout = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", fake_stdout)
    assert fake_stdout.encoding.lower() != "utf-8"

    configure_logging()

    assert sys.stdout.encoding.lower() == "utf-8"


def test_configure_logging_reconfigures_non_utf8_stderr_to_utf8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_stderr = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr(sys, "stderr", fake_stderr)
    assert fake_stderr.encoding.lower() != "utf-8"

    configure_logging()

    assert sys.stderr.encoding.lower() == "utf-8"


def test_configure_logging_leaves_already_utf8_streams_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Streams already on UTF-8 (the common case off Windows) must not raise
    when passed through the same guard."""
    fake_stdout = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
    fake_stderr = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
    monkeypatch.setattr(sys, "stdout", fake_stdout)
    monkeypatch.setattr(sys, "stderr", fake_stderr)

    configure_logging()  # must not raise

    assert sys.stdout.encoding.lower() == "utf-8"
    assert sys.stderr.encoding.lower() == "utf-8"


def test_configure_logging_skips_reconfigure_for_non_textiowrapper_stdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pytest's own capture replaces sys.stdout with a non-TextIOWrapper
    object during normal test runs -- the isinstance guard must skip
    reconfigure() entirely rather than raising AttributeError on a stream
    that doesn't support it (e.g. a plain io.StringIO)."""
    monkeypatch.setattr(sys, "stdout", io.StringIO())

    configure_logging()  # must not raise

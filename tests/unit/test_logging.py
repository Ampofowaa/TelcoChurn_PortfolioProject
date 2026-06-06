"""Unit tests for src/telco_churn/utils/logging.py."""

from __future__ import annotations

import logging
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

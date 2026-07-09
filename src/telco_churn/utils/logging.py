"""Structlog configuration for JSON-formatted logging."""

import io
import logging
import sys
from typing import cast

import structlog


def configure_logging(log_level: str = "INFO") -> None:
    """Configure structlog with JSON output for production and console output for development.

    Also reconfigures stdout/stderr to UTF-8: on Windows, a console's default
    cp1252 encoding can't encode output some dependencies emit unprompted (e.g.
    MLflow's emoji run-URL messages), crashing the process. reconfigure() takes
    effect immediately, unlike PYTHONIOENCODING set via .env — that only affects
    stream construction at interpreter startup, which has already happened by the
    time a script's own code runs.
    """
    if (
        isinstance(sys.stdout, io.TextIOWrapper)
        and sys.stdout.encoding.lower() != "utf-8"
    ):
        sys.stdout.reconfigure(encoding="utf-8")
    if (
        isinstance(sys.stderr, io.TextIOWrapper)
        and sys.stderr.encoding.lower() != "utf-8"
    ):
        sys.stderr.reconfigure(encoding="utf-8")

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a named structlog logger."""
    return cast(structlog.stdlib.BoundLogger, structlog.get_logger(name))

"""Structlog configuration for JSON-formatted logging."""

import io
import logging
import os
import sys
from typing import cast

import structlog

__all__ = ["configure_logging", "get_logger"]


def configure_logging(log_level: str = "INFO", log_format: str | None = None) -> None:
    """Configure structlog with JSON output for production and console output for development.

    log_format selects the renderer: 'json' (structlog.processors.JSONRenderer, for
    CloudWatch/Grafana ingestion in Phase 12+) or 'console' (structlog.dev.ConsoleRenderer,
    colorized key=value pairs for a human reading a local terminal). Defaults to the
    LOG_FORMAT env var, itself defaulting to 'console' — every __main__ entry point calls
    configure_logging() with no arguments before Hydra config is composed, so this can't
    be sourced from configs/config.yaml; an env var is the only signal available that early.
    Production/CI environments set LOG_FORMAT=json explicitly (Phase 11+ CI, Phase 12 AWS).

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
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S UTC", utc=True),
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

    resolved_format = log_format or os.environ.get("LOG_FORMAT", "console")
    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer()
        if resolved_format == "json"
        else structlog.dev.ConsoleRenderer()
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
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

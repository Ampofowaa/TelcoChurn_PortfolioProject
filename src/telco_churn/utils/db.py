"""SQLAlchemy engine factory."""

import os

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

__all__ = ["get_engine"]

_engine: Engine | None = None


def get_engine() -> Engine:
    """Return the shared SQLAlchemy engine, creating it on first call."""
    global _engine
    if _engine is None:
        url = os.environ.get("POSTGRES_URL")
        if not url:
            raise OSError("POSTGRES_URL environment variable is not set")
        _engine = create_engine(url, pool_pre_ping=True)
    return _engine

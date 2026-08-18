"""SQLAlchemy engine factories — sync (psycopg2, the ingest/DVC-stage path)
and async (asyncpg, serving/app.py's I/O-bound customer-lookup routes)."""

import os

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

__all__ = ["get_engine", "get_async_engine", "dispose_async_engine"]

_engine: Engine | None = None
_async_engine: AsyncEngine | None = None


def get_engine() -> Engine:
    """Return the shared SQLAlchemy engine, creating it on first call."""
    global _engine
    if _engine is None:
        url = os.environ.get("POSTGRES_URL")
        if not url:
            raise OSError("POSTGRES_URL environment variable is not set")
        _engine = create_engine(url, pool_pre_ping=True)
    return _engine


def _to_asyncpg_url(url: str) -> str:
    """Rewrite a plain/psycopg2 'postgresql://' URL to the asyncpg dialect.

    POSTGRES_URL is authored once (.env / docker-compose) for the sync driver
    get_engine() already uses; an explicit '+asyncpg' URL is passed through
    unchanged so an operator can still override the driver directly.
    """
    if url.startswith("postgresql+asyncpg://"):
        return url
    if url.startswith("postgresql://"):
        return "postgresql+asyncpg://" + url[len("postgresql://") :]
    return url


def get_async_engine() -> AsyncEngine:
    """Return the shared async SQLAlchemy engine, creating it on first call.

    Backs serving/app.py's GET /customer/{customerid} and POST /predict/batch
    ID-resolution lookups — genuinely I/O-bound Postgres reads, the case
    PROJECT_PLAN.md's Phase 9 section calls out as always worth async for.
    """
    global _async_engine
    if _async_engine is None:
        url = os.environ.get("POSTGRES_URL")
        if not url:
            raise OSError("POSTGRES_URL environment variable is not set")
        _async_engine = create_async_engine(_to_asyncpg_url(url), pool_pre_ping=True)
    return _async_engine


async def dispose_async_engine() -> None:
    """Dispose and clear the shared async engine's connection pool.

    Called from serving/app.py's lifespan shutdown so the process doesn't
    hold open asyncpg connections past the point it stops serving requests;
    a no-op if the async engine was never created (e.g. no request ever hit
    a DB-backed route).
    """
    global _async_engine
    if _async_engine is not None:
        await _async_engine.dispose()
        _async_engine = None

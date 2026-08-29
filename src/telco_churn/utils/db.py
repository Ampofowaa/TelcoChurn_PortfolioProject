"""SQLAlchemy engine factories — sync (psycopg2, the ingest/DVC-stage path)
and async (asyncpg, serving/app.py's I/O-bound customer-lookup routes) — plus
apply_migrations(), the Alembic belt-and-braces call every schema-dependent
CLI entry point (data/ingest.py, serving/crm_data.py, serving/outcomes.py)
runs at startup."""

import os

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from telco_churn.utils.paths import get_project_root

__all__ = [
    "get_engine",
    "get_async_engine",
    "dispose_async_engine",
    "apply_migrations",
]

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


def apply_migrations(database_url: str | None = None) -> None:
    """Run `alembic upgrade head` against the app database.

    Belt-and-braces schema creation for a Postgres instance that never saw
    docker-compose.yml's docker-entrypoint-initdb.d mount — CI's
    testcontainers, Phase 12's RDS, or a developer's already-initialized
    volume. Idempotent: a database already at head is a no-op.

    database_url overrides POSTGRES_URL for this call only (used by tests
    against an ephemeral testcontainers Postgres) — alembic/env.py resolves
    POSTGRES_URL itself when this is left unset, the same entry-point
    convention get_engine()/get_async_engine() already follow.
    """
    from alembic import command
    from alembic.config import Config

    # Named alembic_cfg, not cfg: test_architecture.py's config-read scanner
    # treats any variable literally named `cfg` as this project's composed
    # Hydra DictConfig — an alembic.config.Config under that name would be a
    # false-positive "cfg.set_main_option read" in test_params_match_reads.
    alembic_cfg = Config(str(get_project_root() / "alembic.ini"))
    if database_url is not None:
        alembic_cfg.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(alembic_cfg, "head")

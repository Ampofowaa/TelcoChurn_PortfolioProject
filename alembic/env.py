import os
from logging.config import fileConfig

from dotenv import load_dotenv
from sqlalchemy import engine_from_config, pool

from alembic import context
from telco_churn.data.tables import metadata as target_metadata

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Resolved from POSTGRES_URL (this project's app database — POSTGRES_DB,
# 'telco_churn' locally), never hardcoded in alembic.ini — unless a caller
# already set sqlalchemy.url on this Config object (utils/db.py::apply_migrations'
# database_url override, used against an ephemeral testcontainers Postgres in
# tests), which always wins. load_dotenv() is a no-op if a caller (ingest.py's
# __main__, a test fixture) already loaded .env into the process; it's called
# again here so a bare `alembic ...` invocation from a shell works too, the
# same entry-point convention every other __main__ module in this project
# follows.
#
# Scope boundary — deliberate, not an oversight: this project's Postgres
# server also hosts a sibling 'mlflow' database (sql/schema/000_create_mlflow_db.sql)
# and an 'optuna' schema inside this same database
# (sql/schema/002_create_optuna_schema.sql). Neither is managed by this
# Alembic instance. 'mlflow' is a different database entirely — POSTGRES_URL
# never points at it. 'optuna' lives in this database but a different
# schema; Alembic's autogenerate only diffs the default schema (public)
# unless include_schemas=True is passed to context.configure(), which is
# deliberately never set here — turning it on would pull Optuna's
# self-managed RDBStorage tables into this project's migration history.
database_url = config.get_main_option("sqlalchemy.url")
if not database_url:
    load_dotenv()
    database_url = os.environ.get("POSTGRES_URL")
if not database_url:
    raise OSError("POSTGRES_URL environment variable is not set")
config.set_main_option("sqlalchemy.url", database_url)

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

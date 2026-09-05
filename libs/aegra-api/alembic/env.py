"""Alembic environment configuration for Aegra database migrations."""

import asyncio
import threading
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

# Import your SQLAlchemy models here
from aegra_api.core.migrations import migration_advisory_lock
from aegra_api.core.orm import Base
from aegra_api.settings import settings
from alembic import context

# This is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Stash URL on config.attributes (plain dict) instead of set_main_option.
# set_main_option routes through configparser, which treats `%` as interpolation
# syntax (`%(name)s`). Since 0.9.6 settings.db.database_url passes the password
# through quote_plus, so any password char in `<>^&#:%` produces a `%XX`
# sequence that configparser rejects with `ValueError: invalid interpolation
# syntax`. config.attributes has no interpolation. See issue #357.
# setdefault preserves caller-supplied overrides (e.g. tests or alembic.ini
# round-trip) so _get_database_url() can still fall back to get_main_option.
config.attributes.setdefault("sqlalchemy.url", settings.db.database_url)


def _get_database_url() -> str:
    """Read URL from config.attributes (set in env.py) or fall back to ini."""
    url = config.attributes.get("sqlalchemy.url")
    if url is not None:
        return url
    return config.get_main_option("sqlalchemy.url")


# Interpret the config file for Python logging.
# Only reconfigure logging when running from CLI (main thread).
# When invoked programmatically via asyncio.to_thread(), fileConfig()
# causes a cross-thread deadlock with the application's logging.
# See: https://github.com/sqlalchemy/alembic/discussions/1483
if config.config_file_name is not None and threading.current_thread() is threading.main_thread():
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata

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
    context.configure(
        url=_get_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Apply revisions; the session lock is held by ``run_migrations_online``."""
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    configuration = config.get_section(config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = _get_database_url()

    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode under a session advisory lock."""
    with migration_advisory_lock():
        asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

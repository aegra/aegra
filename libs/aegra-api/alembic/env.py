"""Alembic environment configuration for Aegra database migrations."""

import asyncio
import threading
from logging.config import fileConfig

from sqlalchemy import pool, text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

# Import your SQLAlchemy models here
from aegra_api.core.orm import Base
from aegra_api.settings import settings
from alembic import context

# This is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Override the URL from settings — this respects DATABASE_URL, individual
# POSTGRES_* vars, and preserves query params (e.g. ?sslmode=require).
config.set_main_option("sqlalchemy.url", settings.db.database_url)

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
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        version_table_schema=settings.db.POSTGRES_SCHEMA,
    )

    with context.begin_transaction():
        if settings.db.POSTGRES_SCHEMA:
            context.execute(f'CREATE SCHEMA IF NOT EXISTS "{settings.db.POSTGRES_SCHEMA}"')
            context.execute(f'SET search_path TO "{settings.db.POSTGRES_SCHEMA}", public')
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Run migrations with the given connection."""
    if settings.db.POSTGRES_SCHEMA:
        connection.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{settings.db.POSTGRES_SCHEMA}"'))
        connection.execute(text(f'SET search_path TO "{settings.db.POSTGRES_SCHEMA}", public'))
        connection.commit()

    context.configure(
        connection=connection, target_metadata=target_metadata, version_table_schema=settings.db.POSTGRES_SCHEMA
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = config.get_main_option("sqlalchemy.url")

    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

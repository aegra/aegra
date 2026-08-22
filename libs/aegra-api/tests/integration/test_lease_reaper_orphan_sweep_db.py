"""Real-PostgreSQL regression tests for the lease-less orphan sweep."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from unittest.mock import patch
from uuid import uuid4

import asyncpg
import pytest
from sqlalchemy import delete, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import Thread as ThreadORM
from aegra_api.services.lease_reaper import LeaseReaper
from aegra_api.settings import settings

_USER_ID = "orphan-sweep-test-user"


async def _skip_unless_schema_ready(engine: AsyncEngine) -> None:
    """Skip when the test database is unreachable or unmigrated."""
    runs_table = None
    try:
        async with engine.begin() as conn:
            runs_table = await conn.scalar(text("SELECT to_regclass('public.runs')"))
    except (SQLAlchemyError, asyncpg.PostgresError, OSError) as exc:
        pytest.skip(f"PostgreSQL test database is unavailable: {exc}")
    if runs_table is None:
        pytest.skip("runs table is unavailable; run Alembic migrations before this DB regression test")


async def _seed_leaseless_run(maker: async_sessionmaker, *, age: timedelta) -> tuple[str, str]:
    """Insert a busy thread with a running, never-leased run of the given age."""
    thread_id = f"orphan-thread-{uuid4()}"
    run_id = f"orphan-run-{uuid4()}"
    stamped = datetime.now(UTC) - age

    async with maker() as session:
        session.add(ThreadORM(thread_id=thread_id, status="busy", user_id=_USER_ID))
        await session.flush()
        session.add(
            RunORM(
                run_id=run_id,
                thread_id=thread_id,
                status="running",
                user_id=_USER_ID,
                claimed_by=None,
                lease_expires_at=None,
                updated_at=stamped,
            )
        )
        await session.commit()
    return thread_id, run_id


async def _cleanup(maker: async_sessionmaker, thread_id: str) -> None:
    async with maker() as session:
        await session.execute(delete(RunORM).where(RunORM.thread_id == thread_id))
        await session.execute(delete(ThreadORM).where(ThreadORM.thread_id == thread_id))
        await session.commit()


@asynccontextmanager
async def _seeded_orphan(age: timedelta) -> AsyncIterator[tuple[async_sessionmaker, str, str]]:
    """Seed a lease-less orphan of the given age; always clean up and dispose."""
    engine = create_async_engine(settings.db.database_url)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    thread_id: str | None = None
    try:
        await _skip_unless_schema_ready(engine)
        thread_id, run_id = await _seed_leaseless_run(maker, age=age)
        yield maker, thread_id, run_id
    finally:
        if thread_id is not None:
            await _cleanup(maker, thread_id)
        await engine.dispose()


async def _statuses(maker: async_sessionmaker, thread_id: str, run_id: str) -> tuple[str, str]:
    async with maker() as session:
        run_status = await session.scalar(select(RunORM.status).where(RunORM.run_id == run_id))
        thread_status = await session.scalar(select(ThreadORM.status).where(ThreadORM.thread_id == thread_id))
    return run_status, thread_status


@pytest.mark.asyncio
async def test_sweep_fails_stale_leaseless_run_and_frees_its_thread() -> None:
    """A running row with no lease is invisible to the reaper and must be swept."""
    async with _seeded_orphan(timedelta(hours=1)) as (maker, thread_id, run_id):
        with (
            patch("aegra_api.services.lease_reaper._get_session_maker", return_value=maker),
            patch.object(settings.worker, "ORPHAN_SWEEP_ENABLED", True),
            patch.object(settings.worker, "ORPHAN_SWEEP_MIN_AGE_SECONDS", 300),
        ):
            swept = await LeaseReaper.sweep_leaseless_orphans()

        assert swept >= 1
        assert await _statuses(maker, thread_id, run_id) == ("error", "error")


@pytest.mark.asyncio
async def test_sweep_leaves_a_recently_started_leaseless_run_alone() -> None:
    """A local run that only just started must not be mistaken for a crash."""
    async with _seeded_orphan(timedelta(seconds=5)) as (maker, thread_id, run_id):
        with (
            patch("aegra_api.services.lease_reaper._get_session_maker", return_value=maker),
            patch.object(settings.worker, "ORPHAN_SWEEP_ENABLED", True),
            patch.object(settings.worker, "ORPHAN_SWEEP_MIN_AGE_SECONDS", 300),
        ):
            await LeaseReaper.sweep_leaseless_orphans()

        assert await _statuses(maker, thread_id, run_id) == ("running", "busy")

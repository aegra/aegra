"""Real-PostgreSQL regression tests for the lease-less orphan sweep.

The sweep scans the whole runs table, so every test runs in its own throwaway
schema: it must never see, let alone fail, rows in the developer's database.
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime, timedelta
from unittest.mock import patch
from uuid import uuid4

import asyncpg
import pytest
from sqlalchemy import select, text, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from aegra_api.core.orm import Assistant as AssistantORM
from aegra_api.core.orm import Base
from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import Thread as ThreadORM
from aegra_api.services import lease_reaper
from aegra_api.services.lease_reaper import LeaseReaper
from aegra_api.settings import settings

_USER_ID = "orphan-sweep-test-user"
_TABLES = [AssistantORM.__table__, ThreadORM.__table__, RunORM.__table__]


@asynccontextmanager
async def _isolated_db() -> AsyncIterator[async_sessionmaker]:
    """A session maker bound to a fresh schema holding only the tables the sweep touches."""
    schema = f"orphan_sweep_{uuid4().hex}"
    admin = create_async_engine(settings.db.database_url)
    created = False
    try:
        try:
            async with admin.begin() as conn:
                await conn.execute(text(f'CREATE SCHEMA "{schema}"'))
            created = True
        except (SQLAlchemyError, asyncpg.PostgresError, OSError) as exc:
            pytest.skip(f"PostgreSQL test database is unavailable: {exc}")

        engine = create_async_engine(
            settings.db.database_url,
            connect_args={"server_settings": {"search_path": schema}},
        )
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all, tables=_TABLES)
            yield async_sessionmaker(engine, expire_on_commit=False)
        finally:
            await engine.dispose()
    finally:
        if created:
            async with admin.begin() as conn:
                await conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        await admin.dispose()


async def _seed_thread(maker: async_sessionmaker) -> str:
    thread_id = f"orphan-thread-{uuid4()}"
    async with maker() as session:
        session.add(ThreadORM(thread_id=thread_id, status="busy", user_id=_USER_ID))
        await session.commit()
    return thread_id


async def _seed_leaseless_run(maker: async_sessionmaker, thread_id: str, *, age: timedelta) -> str:
    """Insert a running, never-leased run of the given age on ``thread_id``."""
    run_id = f"orphan-run-{uuid4()}"
    async with maker() as session:
        session.add(
            RunORM(
                run_id=run_id,
                thread_id=thread_id,
                status="running",
                user_id=_USER_ID,
                claimed_by=None,
                lease_expires_at=None,
                updated_at=datetime.now(UTC) - age,
            )
        )
        await session.commit()
    return run_id


@asynccontextmanager
async def _seeded_orphan(age: timedelta) -> AsyncIterator[tuple[async_sessionmaker, str, str]]:
    async with _isolated_db() as maker:
        thread_id = await _seed_thread(maker)
        run_id = await _seed_leaseless_run(maker, thread_id, age=age)
        yield maker, thread_id, run_id


async def _run_status(maker: async_sessionmaker, run_id: str) -> str:
    async with maker() as session:
        return await session.scalar(select(RunORM.status).where(RunORM.run_id == run_id))


async def _thread_status(maker: async_sessionmaker, thread_id: str) -> str:
    async with maker() as session:
        return await session.scalar(select(ThreadORM.status).where(ThreadORM.thread_id == thread_id))


async def _statuses(maker: async_sessionmaker, thread_id: str, run_id: str) -> tuple[str, str]:
    return await _run_status(maker, run_id), await _thread_status(maker, thread_id)


async def _wait_until_blocked_by(maker: async_sessionmaker, holder: AsyncSession) -> None:
    """Return once some backend is waiting on a lock that ``holder``'s backend holds."""
    holder_pid = await holder.scalar(text("SELECT pg_backend_pid()"))
    async with maker() as session:
        for _ in range(100):
            waiting = await session.scalar(
                text("SELECT count(*) FROM pg_stat_activity WHERE :pid = ANY(pg_blocking_pids(pid))"),
                {"pid": holder_pid},
            )
            if waiting:
                return
            await asyncio.sleep(0.05)
    pytest.fail("the sweep never blocked on the thread row")


def _sweep_patches(maker: async_sessionmaker) -> tuple[object, ...]:
    return (
        patch("aegra_api.services.lease_reaper._get_session_maker", return_value=maker),
        patch.object(settings.worker, "ORPHAN_SWEEP_ENABLED", True),
        patch.object(settings.worker, "ORPHAN_SWEEP_MIN_AGE_SECONDS", 300),
    )


@pytest.mark.asyncio
async def test_sweep_fails_stale_leaseless_run_and_frees_its_thread() -> None:
    """A running row with no lease is invisible to the reaper and must be swept."""
    async with _seeded_orphan(timedelta(hours=1)) as (maker, thread_id, run_id):
        p_maker, p_enabled, p_age = _sweep_patches(maker)
        with p_maker, p_enabled, p_age:
            swept = await LeaseReaper.sweep_leaseless_orphans()

        assert swept == 1
        assert await _statuses(maker, thread_id, run_id) == ("error", "error")


@pytest.mark.asyncio
async def test_sweep_leaves_a_recently_started_leaseless_run_alone() -> None:
    """A local run that only just started must not be mistaken for a crash."""
    async with _seeded_orphan(timedelta(seconds=5)) as (maker, thread_id, run_id):
        p_maker, p_enabled, p_age = _sweep_patches(maker)
        with p_maker, p_enabled, p_age:
            await LeaseReaper.sweep_leaseless_orphans()

        assert await _statuses(maker, thread_id, run_id) == ("running", "busy")


@pytest.mark.asyncio
async def test_sweep_still_fails_an_orphan_that_was_locked_when_it_first_looked() -> None:
    """Regression: SKIP LOCKED returned no rows while another transaction held the
    orphan, and the one-shot sweep took that as "nothing left" and stopped."""
    async with _seeded_orphan(timedelta(hours=1)) as (maker, thread_id, run_id):
        async with maker() as holder:
            await holder.execute(select(RunORM.run_id).where(RunORM.run_id == run_id).with_for_update())

            async def release_lock_soon() -> None:
                await asyncio.sleep(0.3)
                await holder.rollback()

            p_maker, p_enabled, p_age = _sweep_patches(maker)
            with (
                p_maker,
                p_enabled,
                p_age,
                patch("aegra_api.services.lease_reaper._SWEEP_LOCK_RETRY_DELAY_SECONDS", 0.2),
            ):
                releaser = asyncio.create_task(release_lock_soon())
                await LeaseReaper.sweep_leaseless_orphans()
                await releaser

        assert await _statuses(maker, thread_id, run_id) == ("error", "error")


@pytest.mark.asyncio
async def test_concurrent_sweeps_taking_different_orphans_of_one_thread_still_free_it() -> None:
    """Regression: each sweep saw the other's orphan as still running, so neither
    reconciled the thread and it stayed busy after both committed."""
    async with _isolated_db() as maker:
        thread_id = await _seed_thread(maker)
        run_ids = [await _seed_leaseless_run(maker, thread_id, age=timedelta(hours=1)) for _ in range(2)]

        reconcile = lease_reaper.set_thread_status_if_no_active_runs
        arrived = 0
        both_arrived = asyncio.Event()

        async def meet_then_reconcile(
            session: AsyncSession, thread_ids: set[str], status: str, *, user_id: str
        ) -> None:
            nonlocal arrived
            arrived += 1
            if arrived == 2:
                both_arrived.set()
            # Line both sweeps up on the thread check; a sweep that serializes on it never arrives.
            with suppress(TimeoutError):
                await asyncio.wait_for(both_arrived.wait(), timeout=1.0)
            await reconcile(session, thread_ids, status, user_id=user_id)

        p_maker, p_enabled, p_age = _sweep_patches(maker)
        with (
            p_maker,
            p_enabled,
            p_age,
            patch("aegra_api.services.lease_reaper._SWEEP_BATCH_SIZE", 1),
            patch("aegra_api.services.lease_reaper._SWEEP_LOCK_RETRY_DELAY_SECONDS", 0.05),
            patch(
                "aegra_api.services.lease_reaper.set_thread_status_if_no_active_runs",
                side_effect=meet_then_reconcile,
            ),
        ):
            swept = await asyncio.wait_for(
                asyncio.gather(LeaseReaper.sweep_leaseless_orphans(), LeaseReaper.sweep_leaseless_orphans()),
                timeout=15,
            )

        assert sum(swept) == 2
        assert [await _run_status(maker, run_id) for run_id in run_ids] == ["error", "error"]
        assert await _thread_status(maker, thread_id) == "error"


@pytest.mark.asyncio
async def test_sweep_does_not_fail_a_thread_that_gained_a_pending_run_meanwhile() -> None:
    """Regression: the thread check waited for a run being created on the thread, then
    re-checked against its old snapshot and marked the thread error under the new run."""
    async with _seeded_orphan(timedelta(hours=1)) as (maker, thread_id, run_id):
        async with maker() as creator:
            # What run creation does: mark the thread busy, insert the run, commit.
            await creator.execute(update(ThreadORM).where(ThreadORM.thread_id == thread_id).values(status="busy"))
            creator.add(RunORM(run_id=f"new-run-{uuid4()}", thread_id=thread_id, status="pending", user_id=_USER_ID))
            await creator.flush()

            p_maker, p_enabled, p_age = _sweep_patches(maker)
            with p_maker, p_enabled, p_age:
                sweep = asyncio.create_task(LeaseReaper.sweep_leaseless_orphans())
                await _wait_until_blocked_by(maker, creator)
                await creator.commit()
                await asyncio.wait_for(sweep, timeout=10)

        assert await _statuses(maker, thread_id, run_id) == ("error", "busy")


@pytest.mark.asyncio
async def test_sweep_gives_up_on_a_thread_row_held_by_another_transaction() -> None:
    """Regression: the thread update waited on the row lock without a timeout, and
    startup awaits the sweep, so one stuck transaction kept the instance from serving."""
    async with _seeded_orphan(timedelta(hours=1)) as (maker, thread_id, run_id):
        async with maker() as holder:
            await holder.execute(select(ThreadORM.thread_id).where(ThreadORM.thread_id == thread_id).with_for_update())

            p_maker, p_enabled, p_age = _sweep_patches(maker)
            with (
                p_maker,
                p_enabled,
                p_age,
                patch("aegra_api.services.lease_reaper._SWEEP_LOCK_TIMEOUT", "200ms"),
                pytest.raises(SQLAlchemyError),
            ):
                await asyncio.wait_for(LeaseReaper.sweep_leaseless_orphans(), timeout=5)
            await holder.rollback()

        assert await _statuses(maker, thread_id, run_id) == ("running", "busy")

"""Real-PostgreSQL regression tests for per-acquisition lease fencing (#502).

The fence lives in SQL predicates, so mocked sessions can only prove statement
shape. These drive the pause -> reap -> reclaim sequence against a real
database and assert which attempt is allowed to write. They skip when no
database is reachable.
"""

from collections.abc import AsyncIterator
from contextlib import ExitStack, asynccontextmanager
from datetime import timedelta
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from sqlalchemy import Interval, delete, func, literal, select, text, update
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import Thread as ThreadORM
from aegra_api.services.lease_reaper import LeaseReaper
from aegra_api.services.run_status import finalize_run, start_run
from aegra_api.services.worker_executor import (
    _acquire_and_load,
    _release_lease,
    _renew_lease,
    _requeue_drained_runs,
)
from aegra_api.settings import settings

_USER_ID = "lease-fencing-test-user"
_WORKER = "test-host-1234-worker-0"

_EXECUTION_PARAMS = {
    "graph_id": "test-graph",
    "user": {"identity": _USER_ID, "is_authenticated": True, "permissions": []},
    "execution": {
        "input_data": {},
        "config": {},
        "context": {},
        "stream_mode": None,
        "checkpoint": None,
        "command": None,
        "event_streaming_v2": False,
    },
    "behavior": {
        "interrupt_before": None,
        "interrupt_after": None,
        "multitask_strategy": None,
        "subgraphs": False,
    },
    "run_metadata": {},
}

# The services resolve sessions through the app-wide manager, which only exists
# inside a running server; point each of them at the test engine instead.
_SESSION_MAKER_TARGETS = (
    "aegra_api.services.worker_executor._get_session_maker",
    "aegra_api.services.lease_reaper._get_session_maker",
    "aegra_api.services.run_status._get_session_maker",
)


async def _skip_unless_schema_ready(engine: AsyncEngine) -> None:
    try:
        async with engine.begin() as conn:
            runs_table = await conn.scalar(text("SELECT to_regclass('public.runs')"))
    except Exception as exc:  # noqa: BLE001 - any connect failure means "no database here"
        pytest.skip(f"PostgreSQL test database is unavailable: {exc}")
    if runs_table is None:
        pytest.skip("runs table is missing; run Alembic migrations before this DB regression test")


@asynccontextmanager
async def _pending_run() -> AsyncIterator[tuple[async_sessionmaker, str, str]]:
    """Seed a busy thread with one pending run; always clean up and dispose."""
    engine = create_async_engine(settings.db.database_url)
    try:
        await _skip_unless_schema_ready(engine)
        maker = async_sessionmaker(engine, expire_on_commit=False)
        thread_id = f"fencing-thread-{uuid4()}"
        run_id = str(uuid4())

        async with maker() as session:
            session.add(ThreadORM(thread_id=thread_id, status="busy", user_id=_USER_ID))
            await session.flush()
            session.add(
                RunORM(
                    run_id=run_id,
                    thread_id=thread_id,
                    status="pending",
                    user_id=_USER_ID,
                    execution_params=_EXECUTION_PARAMS,
                )
            )
            await session.commit()

        try:
            with ExitStack() as stack:
                for target in _SESSION_MAKER_TARGETS:
                    stack.enter_context(patch(target, return_value=maker))
                yield maker, thread_id, run_id
        finally:
            async with maker() as session:
                await session.execute(delete(RunORM).where(RunORM.thread_id == thread_id))
                await session.execute(delete(ThreadORM).where(ThreadORM.thread_id == thread_id))
                await session.commit()
    finally:
        await engine.dispose()


async def _expire_lease(maker: async_sessionmaker, run_id: str) -> None:
    """Backdate the lease on the database clock, which is what the reaper compares against."""
    async with maker() as session:
        await session.execute(
            update(RunORM)
            .where(RunORM.run_id == run_id)
            .values(lease_expires_at=func.now() - literal(timedelta(minutes=1), Interval))
        )
        await session.commit()


async def _reap(maker: async_sessionmaker, run_id: str) -> None:
    await _expire_lease(maker, run_id)
    retryable, exhausted = await LeaseReaper._recover_crashed_runs([run_id])
    assert retryable == [run_id]
    assert exhausted == []


async def _row(maker: async_sessionmaker, run_id: str) -> RunORM:
    async with maker() as session:
        run = await session.scalar(select(RunORM).where(RunORM.run_id == run_id))
        assert run is not None
        return run


async def _thread_status(maker: async_sessionmaker, thread_id: str) -> str:
    async with maker() as session:
        status = await session.scalar(select(ThreadORM.status).where(ThreadORM.thread_id == thread_id))
        assert status is not None
        return status


@pytest.mark.asyncio
async def test_same_worker_reclaiming_a_reaped_run_gets_a_new_claim_id() -> None:
    async with _pending_run() as (maker, _thread_id, run_id):
        first = await _acquire_and_load(run_id, _WORKER)
        assert first is not None
        await _reap(maker, run_id)

        second = await _acquire_and_load(run_id, _WORKER)

        assert second is not None
        assert second.claim_id != first.claim_id
        assert first.claim_id.startswith(f"{_WORKER}:")
        assert (await _row(maker, run_id)).claimed_by == second.claim_id


@pytest.mark.asyncio
async def test_stale_finalize_cannot_overwrite_the_run_its_replacement_holds() -> None:
    """Regression: a reaped attempt finishing late wrote its output over a run the
    replacement still held, set the thread idle, and made the replacement's finalize lose."""
    async with _pending_run() as (maker, thread_id, run_id):
        stale = await _acquire_and_load(run_id, _WORKER)
        assert stale is not None
        await _reap(maker, run_id)
        live = await _acquire_and_load(run_id, _WORKER)
        assert live is not None

        stale_won = await finalize_run(
            run_id,
            thread_id,
            user_id=_USER_ID,
            status="success",
            thread_status="idle",
            output={"writer": "stale"},
            claim_id=stale.claim_id,
        )

        assert stale_won is False
        row = await _row(maker, run_id)
        assert row.status == "running"
        assert row.output is None
        assert row.claimed_by == live.claim_id
        assert await _thread_status(maker, thread_id) == "busy"

        live_won = await finalize_run(
            run_id,
            thread_id,
            user_id=_USER_ID,
            status="success",
            thread_status="idle",
            output={"writer": "live"},
            claim_id=live.claim_id,
        )

        assert live_won is True
        assert await _thread_status(maker, thread_id) == "idle"
        # A heartbeat tick racing the job's own teardown must still match, or it
        # would cancel the job mid-teardown as if the lease had been lost.
        assert await _renew_lease(run_id, live.claim_id) == 1

        await _release_lease(run_id, live.claim_id)
        row = await _row(maker, run_id)
        assert row.status == "success"
        assert row.output == {"writer": "live"}
        assert row.claimed_by is None
        assert row.lease_expires_at is None


@pytest.mark.asyncio
async def test_stale_attempt_cannot_renew_or_release_the_replacement_lease() -> None:
    """Regression: releasing on the reusable worker name stripped the lease off the
    replacement, leaving a running row that no reaper query can see."""
    async with _pending_run() as (maker, _thread_id, run_id):
        stale = await _acquire_and_load(run_id, _WORKER)
        assert stale is not None
        await _reap(maker, run_id)
        live = await _acquire_and_load(run_id, _WORKER)
        assert live is not None

        assert await _renew_lease(run_id, stale.claim_id) == 0
        await _release_lease(run_id, stale.claim_id)

        row = await _row(maker, run_id)
        assert row.status == "running"
        assert row.claimed_by == live.claim_id
        assert row.lease_expires_at is not None


@pytest.mark.asyncio
async def test_live_attempt_release_does_not_hide_a_running_row_from_the_reaper() -> None:
    async with _pending_run() as (maker, _thread_id, run_id):
        live = await _acquire_and_load(run_id, _WORKER)
        assert live is not None

        await _release_lease(run_id, live.claim_id)

        row = await _row(maker, run_id)
        assert row.claimed_by == live.claim_id
        assert row.lease_expires_at is not None


@pytest.mark.asyncio
async def test_stale_start_cannot_turn_a_reaped_pending_run_into_an_unowned_running_one() -> None:
    async with _pending_run() as (maker, _thread_id, run_id):
        stale = await _acquire_and_load(run_id, _WORKER)
        assert stale is not None
        await _reap(maker, run_id)

        started = await start_run(run_id, user_id=_USER_ID, claim_id=stale.claim_id)

        assert started is False
        row = await _row(maker, run_id)
        assert row.status == "pending"
        assert row.claimed_by is None


@pytest.mark.asyncio
async def test_drain_requeue_leaves_a_run_another_attempt_holds() -> None:
    async with _pending_run() as (maker, _thread_id, run_id):
        stale = await _acquire_and_load(run_id, _WORKER)
        assert stale is not None
        await _reap(maker, run_id)
        live = await _acquire_and_load(run_id, "other-host-1-worker-0")
        assert live is not None

        with patch("aegra_api.services.worker_executor.redis_manager") as mock_redis:
            await _requeue_drained_runs([run_id], claim_ids=[stale.claim_id])

        mock_redis.get_client.assert_not_called()
        row = await _row(maker, run_id)
        assert row.status == "running"
        assert row.claimed_by == live.claim_id


@pytest.mark.asyncio
async def test_drain_requeue_hands_back_the_run_this_attempt_holds() -> None:
    async with _pending_run() as (maker, _thread_id, run_id):
        live = await _acquire_and_load(run_id, _WORKER)
        assert live is not None

        client = AsyncMock()
        with patch("aegra_api.services.worker_executor.redis_manager") as mock_redis:
            mock_redis.get_client.return_value = client
            await _requeue_drained_runs([run_id], claim_ids=[live.claim_id])

        assert client.rpush.await_count == 1

        row = await _row(maker, run_id)
        assert row.status == "pending"
        assert row.claimed_by is None
        assert row.lease_expires_at is None

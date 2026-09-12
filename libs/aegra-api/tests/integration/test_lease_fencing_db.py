"""Real-PostgreSQL regression tests for per-acquisition lease fencing (#502).

The fence lives entirely in SQL predicates, so mocked sessions can only prove
the statements are shaped right. These exercise the pause/reap/reclaim sequence
against a real database and assert which attempt is allowed to win.
"""

from collections.abc import AsyncIterator
from contextlib import ExitStack, asynccontextmanager
from datetime import timedelta
from unittest.mock import patch
from uuid import uuid4

import asyncpg
import pytest
from sqlalchemy import Interval, delete, func, literal, select, text, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import Thread as ThreadORM
from aegra_api.services.lease_reaper import LeaseReaper
from aegra_api.services.run_status import finalize_run
from aegra_api.services.worker_executor import _acquire_and_load, _release_lease, _renew_lease
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


async def _skip_unless_schema_ready(engine: AsyncEngine) -> None:
    """Skip when the test database is unreachable or unmigrated."""
    claim_token_column = None
    try:
        async with engine.begin() as conn:
            claim_token_column = await conn.scalar(
                text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'runs' AND column_name = 'claim_token'"
                )
            )
    except (SQLAlchemyError, asyncpg.PostgresError, OSError) as exc:
        pytest.skip(f"PostgreSQL test database is unavailable: {exc}")
    if claim_token_column is None:
        pytest.skip("runs.claim_token is missing; run Alembic migrations before this DB regression test")


# The services under test resolve their sessions through the app-wide manager,
# which only exists inside a running server; point all three at the test engine.
_SESSION_MAKER_TARGETS = (
    "aegra_api.services.worker_executor._get_session_maker",
    "aegra_api.services.lease_reaper._get_session_maker",
    "aegra_api.services.run_status._get_session_maker",
)


@asynccontextmanager
async def _pending_run() -> AsyncIterator[tuple[async_sessionmaker, str, str]]:
    """Seed a busy thread with one pending run; always clean up and dispose."""
    engine = create_async_engine(settings.db.database_url)
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
        await engine.dispose()


async def _expire_lease(maker: async_sessionmaker, run_id: str) -> None:
    """Backdate the lease so the reaper treats the holder as crashed.

    Uses the database clock: the reaper compares against ``now()``, so a value
    written from the test runner's clock could still be in Postgres's future.
    """
    async with maker() as session:
        await session.execute(
            update(RunORM)
            .where(RunORM.run_id == run_id)
            .values(lease_expires_at=func.now() - literal(timedelta(minutes=1), Interval))
        )
        await session.commit()


async def _row(maker: async_sessionmaker, run_id: str) -> RunORM:
    async with maker() as session:
        run = await session.scalar(select(RunORM).where(RunORM.run_id == run_id))
        assert run is not None
        return run


@pytest.mark.asyncio
async def test_same_worker_reclaiming_a_reaped_run_gets_a_new_token() -> None:
    async with _pending_run() as (maker, _thread_id, run_id):
        first = await _acquire_and_load(run_id, _WORKER)
        assert first is not None

        await _expire_lease(maker, run_id)
        retryable, exhausted = await LeaseReaper._recover_crashed_runs([run_id])
        assert retryable == [run_id]
        assert exhausted == []
        assert (await _row(maker, run_id)).claim_token is None

        second = await _acquire_and_load(run_id, _WORKER)
        assert second is not None
        assert second.claim_token != first.claim_token


@pytest.mark.asyncio
async def test_reaped_attempt_cannot_renew_or_release_the_replacement_lease() -> None:
    """Regression: releasing on the reusable worker name stripped the lease off
    the replacement, leaving a running row no reaper query can see."""
    async with _pending_run() as (maker, _thread_id, run_id):
        stale = await _acquire_and_load(run_id, _WORKER)
        assert stale is not None
        await _expire_lease(maker, run_id)
        await LeaseReaper._recover_crashed_runs([run_id])
        live = await _acquire_and_load(run_id, _WORKER)
        assert live is not None

        assert await _renew_lease(run_id, stale.claim_token, timeout=10) == 0
        await _release_lease(run_id, stale.claim_token)

        row = await _row(maker, run_id)
        assert row.status == "running"
        assert row.claim_token == live.claim_token
        assert row.lease_expires_at is not None, "the replacement must stay visible to the reaper"


@pytest.mark.asyncio
async def test_only_the_live_attempt_can_terminalize_the_run() -> None:
    """Regression: a late finish from a reaped worker overwrote the replacement's
    output, making the surviving result non-deterministic."""
    async with _pending_run() as (maker, thread_id, run_id):
        stale = await _acquire_and_load(run_id, _WORKER)
        assert stale is not None
        await _expire_lease(maker, run_id)
        await LeaseReaper._recover_crashed_runs([run_id])
        live = await _acquire_and_load(run_id, _WORKER)
        assert live is not None

        stale_won = await finalize_run(
            run_id,
            thread_id,
            user_id=_USER_ID,
            status="success",
            thread_status="idle",
            output={"from": "stale"},
            claim_token=stale.claim_token,
        )
        assert stale_won is False

        live_won = await finalize_run(
            run_id,
            thread_id,
            user_id=_USER_ID,
            status="success",
            thread_status="idle",
            output={"from": "live"},
            claim_token=live.claim_token,
        )
        assert live_won is True

        row = await _row(maker, run_id)
        assert row.output == {"from": "live"}
        assert row.status == "success"
        assert row.claimed_by is None
        assert row.claim_token is None
        assert row.lease_expires_at is None


@pytest.mark.asyncio
async def test_stale_attempt_cannot_reopen_a_run_it_no_longer_owns() -> None:
    async with _pending_run() as (maker, thread_id, run_id):
        stale = await _acquire_and_load(run_id, _WORKER)
        assert stale is not None
        await _expire_lease(maker, run_id)
        await LeaseReaper._recover_crashed_runs([run_id])
        live = await _acquire_and_load(run_id, _WORKER)
        assert live is not None

        await finalize_run(
            run_id,
            thread_id,
            user_id=_USER_ID,
            status="error",
            thread_status="error",
            error="genuine failure",
            claim_token=live.claim_token,
        )

        reopened = await finalize_run(
            run_id,
            thread_id,
            user_id=_USER_ID,
            status="success",
            thread_status="idle",
            output={"from": "stale"},
            claim_token=stale.claim_token,
        )

        assert reopened is False
        row = await _row(maker, run_id)
        assert row.status == "error"
        assert row.error_message == "genuine failure"

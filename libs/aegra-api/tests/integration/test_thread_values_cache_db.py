"""Thread values cache tests that need a real PostgreSQL.

What matters here is what Postgres does: whether JSONB rejects a value, whether a
savepoint really isolates that rejection from the run's terminal status, and how
checkpoint IDs compare in SQL. Session fakes can't show any of it, so these tests
run against the database and skip when none is reachable.
"""

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import patch
from uuid import uuid4

import pytest
from langgraph.checkpoint.base.id import uuid6
from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import Thread as ThreadORM
from aegra_api.services.run_status import finalize_run, set_thread_values
from aegra_api.services.thread_state_service import ThreadValues
from aegra_api.settings import settings

_USER_ID = "thread-values-cache-test-user"


@pytest.fixture
async def maker() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """Hand out a session factory on the test database, or skip without one."""
    engine = create_async_engine(settings.db.database_url)
    try:
        try:
            async with engine.begin() as conn:
                has_column = await conn.scalar(
                    text(
                        "SELECT 1 FROM information_schema.columns "
                        "WHERE table_name = 'thread' AND column_name = 'values_checkpoint_id'"
                    )
                )
        except Exception as exc:  # noqa: BLE001 - any connect failure means "no database here"
            pytest.skip(f"PostgreSQL test database is unavailable: {exc}")

        if has_column is None:
            pytest.skip("thread values columns are unavailable; run Alembic migrations before this DB test")

        session_maker = async_sessionmaker(engine, expire_on_commit=False)
        try:
            yield session_maker
        finally:
            async with session_maker() as cleanup:
                await cleanup.execute(delete(ThreadORM).where(ThreadORM.user_id == _USER_ID))
                await cleanup.commit()
    finally:
        await engine.dispose()


async def _create_thread_with_running_run(maker: async_sessionmaker[AsyncSession]) -> tuple[str, str]:
    thread_id = f"values-cache-{uuid4()}"
    run_id = str(uuid4())
    async with maker() as session:
        session.add(ThreadORM(thread_id=thread_id, status="busy", metadata_json={}, user_id=_USER_ID))
        await session.flush()
        session.add(RunORM(run_id=run_id, thread_id=thread_id, status="running", user_id=_USER_ID))
        await session.commit()
    return thread_id, run_id


async def _write_cache(maker: async_sessionmaker[AsyncSession], thread_id: str, thread_values: ThreadValues) -> None:
    async with maker() as session:
        await set_thread_values(session, thread_id, thread_values, user_id=_USER_ID)
        await session.commit()


async def _read_thread(maker: async_sessionmaker[AsyncSession], thread_id: str) -> ThreadORM:
    async with maker() as session:
        thread = await session.scalar(select(ThreadORM).where(ThreadORM.thread_id == thread_id))
    assert thread is not None
    return thread


async def _read_run_status(maker: async_sessionmaker[AsyncSession], run_id: str) -> str | None:
    async with maker() as session:
        return await session.scalar(select(RunORM.status).where(RunORM.run_id == run_id))


def _snapshot(answer: Any, checkpoint_id: str | None) -> ThreadValues:
    return ThreadValues(values={"answer": answer}, interrupts={}, checkpoint_id=checkpoint_id)


@pytest.mark.asyncio
async def test_older_snapshot_does_not_replace_newer_cache(maker: async_sessionmaker[AsyncSession]) -> None:
    thread_id, _ = await _create_thread_with_running_run(maker)
    older, newer = str(uuid6()), str(uuid6())

    await _write_cache(maker, thread_id, _snapshot("newer", newer))
    await _write_cache(maker, thread_id, _snapshot("older", older))

    thread = await _read_thread(maker, thread_id)
    assert thread.values_json == {"answer": "newer"}
    assert thread.values_checkpoint_id == newer


@pytest.mark.asyncio
async def test_newer_or_same_snapshot_replaces_cache(maker: async_sessionmaker[AsyncSession]) -> None:
    thread_id, _ = await _create_thread_with_running_run(maker)
    first, second = str(uuid6()), str(uuid6())

    await _write_cache(maker, thread_id, _snapshot("first", first))
    await _write_cache(maker, thread_id, _snapshot("second", second))
    await _write_cache(maker, thread_id, _snapshot("second, re-read", second))

    thread = await _read_thread(maker, thread_id)
    assert thread.values_json == {"answer": "second, re-read"}
    assert thread.values_checkpoint_id == second


@pytest.mark.asyncio
async def test_snapshot_without_checkpoint_never_replaces_a_cached_one(
    maker: async_sessionmaker[AsyncSession],
) -> None:
    thread_id, _ = await _create_thread_with_running_run(maker)
    checkpoint = str(uuid6())

    await _write_cache(maker, thread_id, _snapshot("real", checkpoint))
    await _write_cache(maker, thread_id, _snapshot("empty", None))

    thread = await _read_thread(maker, thread_id)
    assert thread.values_json == {"answer": "real"}


@pytest.mark.asyncio
async def test_finalize_after_concurrent_update_state_keeps_newer_cache(
    maker: async_sessionmaker[AsyncSession],
) -> None:
    """The run read its snapshot, then update_state cached a newer checkpoint before the run finalized."""
    thread_id, run_id = await _create_thread_with_running_run(maker)
    run_snapshot_checkpoint = str(uuid6())
    update_state_checkpoint = str(uuid6())
    await _write_cache(maker, thread_id, _snapshot("from update_state", update_state_checkpoint))

    with patch("aegra_api.services.run_status._get_session_maker", return_value=maker):
        finalized = await finalize_run(
            run_id,
            thread_id,
            user_id=_USER_ID,
            status="success",
            thread_status="idle",
            thread_values=_snapshot("from run", run_snapshot_checkpoint),
        )

    assert finalized is True
    thread = await _read_thread(maker, thread_id)
    assert thread.values_json == {"answer": "from update_state"}
    assert thread.status == "idle"
    assert await _read_run_status(maker, run_id) == "success"


@pytest.mark.asyncio
async def test_finalize_keeps_terminal_status_when_postgres_rejects_cached_values(
    maker: async_sessionmaker[AsyncSession],
) -> None:
    """JSONB has no NaN, so this cache write fails; the run must still be recorded as a success."""
    thread_id, run_id = await _create_thread_with_running_run(maker)

    with patch("aegra_api.services.run_status._get_session_maker", return_value=maker):
        finalized = await finalize_run(
            run_id,
            thread_id,
            user_id=_USER_ID,
            status="success",
            thread_status="idle",
            output={"ok": True},
            thread_values=_snapshot(float("nan"), str(uuid6())),
        )

    assert finalized is True
    assert await _read_run_status(maker, run_id) == "success"
    thread = await _read_thread(maker, thread_id)
    assert thread.status == "idle"
    assert thread.values_json is None
    assert thread.values_checkpoint_id is None

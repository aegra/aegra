"""Thread metadata merge tests that need a real PostgreSQL.

The interesting property is what two *concurrent* transactions do to the same
``thread.metadata_json`` row, which no session fake can show: these tests drive
the real handlers on separate sessions and interleave their reads and commits
the way two HTTP requests do. They skip when no database is reachable.
"""

import asyncio
from collections.abc import AsyncIterator, Callable
from typing import Any
from uuid import uuid4

import pytest
from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from aegra_api.api.threads import update_thread
from aegra_api.core.orm import Thread as ThreadORM
from aegra_api.models import ThreadUpdate, User
from aegra_api.services.run_preparation import update_thread_metadata
from aegra_api.settings import settings

_USER_ID = "thread-metadata-merge-test-user"
_WAIT_TIMEOUT = 10.0

SessionFactory = Callable[[], AsyncSession]


def _user() -> User:
    return User(identity=_USER_ID)


class PausedBeforeWrite:
    """Session proxy that holds a request between its read and its write.

    Concurrent writers interleave like this: both read the row, then both write
    it. Every write entry point is gated, not just one, so the pause lands in
    the same place whether the handler writes through a statement or through
    the flush its commit triggers. Everything else goes to the real session.
    """

    def __init__(self, session: AsyncSession, read_done: asyncio.Event, resume: asyncio.Event) -> None:
        self._session = session
        self._read_done = read_done
        self._resume = resume

    def __getattr__(self, name: str) -> Any:
        return getattr(self._session, name)

    async def scalar(self, *args: Any, **kwargs: Any) -> Any:
        result = await self._session.scalar(*args, **kwargs)
        self._read_done.set()
        return result

    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        await self._resume.wait()
        return await self._session.execute(*args, **kwargs)

    async def flush(self, *args: Any, **kwargs: Any) -> Any:
        await self._resume.wait()
        return await self._session.flush(*args, **kwargs)

    async def commit(self) -> None:
        await self._resume.wait()
        await self._session.commit()


@pytest.fixture
async def sessions() -> AsyncIterator[SessionFactory]:
    """Hand out sessions on independent connections, or skip without a database."""
    engine = create_async_engine(settings.db.database_url)
    try:
        try:
            async with engine.begin() as conn:
                thread_table = await conn.scalar(text("SELECT to_regclass('public.thread')"))
        except Exception as exc:  # noqa: BLE001 - any connect failure means "no database here"
            pytest.skip(f"PostgreSQL test database is unavailable: {exc}")

        if thread_table is None:
            pytest.skip("thread table is unavailable; run Alembic migrations before this DB test")

        maker = async_sessionmaker(engine, expire_on_commit=False)
        try:
            yield maker
        finally:
            async with maker() as cleanup:
                await cleanup.execute(delete(ThreadORM).where(ThreadORM.user_id == _USER_ID))
                await cleanup.commit()
    finally:
        await engine.dispose()


async def _create_thread(sessions: SessionFactory, metadata: dict[str, Any]) -> str:
    thread_id = f"metadata-merge-{uuid4()}"
    async with sessions() as session:
        session.add(ThreadORM(thread_id=thread_id, status="idle", metadata_json=metadata, user_id=_USER_ID))
        await session.commit()
    return thread_id


async def _read_metadata(sessions: SessionFactory, thread_id: str) -> Any:
    async with sessions() as session:
        return await session.scalar(select(ThreadORM.metadata_json).where(ThreadORM.thread_id == thread_id))


@pytest.mark.asyncio
async def test_concurrent_patches_keep_both_sets_of_keys(sessions: SessionFactory) -> None:
    """Two PATCHes that read the same row must not discard each other's keys."""
    thread_id = await _create_thread(sessions, {"graph_id": "agent"})
    read_done = asyncio.Event()
    resume = asyncio.Event()

    async with sessions() as slow, sessions() as fast:
        slow_patch = asyncio.create_task(
            update_thread(
                thread_id,
                ThreadUpdate(metadata={"from_slow": "one"}),
                user=_user(),
                session=PausedBeforeWrite(slow, read_done, resume),
            )
        )
        # The slow request has read the row; the fast one now writes and commits
        # underneath it, so the slow request's write is based on a stale read.
        await asyncio.wait_for(read_done.wait(), timeout=_WAIT_TIMEOUT)
        await update_thread(thread_id, ThreadUpdate(metadata={"from_fast": "two"}), user=_user(), session=fast)
        resume.set()
        await asyncio.wait_for(slow_patch, timeout=_WAIT_TIMEOUT)

    metadata = await _read_metadata(sessions, thread_id)
    assert metadata["from_slow"] == "one"
    assert metadata["from_fast"] == "two"
    assert metadata["graph_id"] == "agent"


@pytest.mark.asyncio
async def test_patch_racing_a_starting_run_keeps_graph_id(sessions: SessionFactory) -> None:
    """A PATCH landing on a starting run must not drop the run's own keys.

    Losing ``graph_id`` is not cosmetic: POST /threads/{id}/state refuses a
    thread whose metadata has none.
    """
    thread_id = await _create_thread(sessions, {"owner": _USER_ID})
    read_done = asyncio.Event()
    resume = asyncio.Event()

    async with sessions() as run, sessions() as client:
        # Run preparation writes assistant_id/graph_id but does not commit —
        # the caller owns the transaction, and it stays open for the rest of
        # run creation.
        await update_thread_metadata(run, thread_id, assistant_id="asst-1", graph_id="agent", user_id=_USER_ID)

        client_patch = asyncio.create_task(
            update_thread(
                thread_id,
                ThreadUpdate(metadata={"client_key": "keep me"}),
                user=_user(),
                session=PausedBeforeWrite(client, read_done, resume),
            )
        )
        await asyncio.wait_for(read_done.wait(), timeout=_WAIT_TIMEOUT)
        resume.set()
        # Let the PATCH reach the row lock the run is holding before releasing it.
        await asyncio.sleep(0.1)
        await run.commit()
        await asyncio.wait_for(client_patch, timeout=_WAIT_TIMEOUT)

    metadata = await _read_metadata(sessions, thread_id)
    assert metadata["client_key"] == "keep me"
    assert metadata["graph_id"] == "agent"
    assert metadata["assistant_id"] == "asst-1"


@pytest.mark.asyncio
async def test_patch_merge_is_shallow(sessions: SessionFactory) -> None:
    """A supplied key replaces that key wholesale; nested objects are not merged."""
    thread_id = await _create_thread(sessions, {"nested": {"a": 1}, "keep": True})

    async with sessions() as session:
        await update_thread(thread_id, ThreadUpdate(metadata={"nested": {"b": 2}}), user=_user(), session=session)

    metadata = await _read_metadata(sessions, thread_id)
    assert metadata == {"nested": {"b": 2}, "keep": True}


@pytest.mark.asyncio
async def test_patch_metadata_containing_a_null_byte_is_written(sessions: SessionFactory) -> None:
    """Postgres rejects NUL in jsonb, so the patch binds through JsonbSafe."""
    thread_id = await _create_thread(sessions, {})

    async with sessions() as session:
        await update_thread(thread_id, ThreadUpdate(metadata={"note": "be\x00fore"}), user=_user(), session=session)

    metadata = await _read_metadata(sessions, thread_id)
    assert metadata["note"] == "before"


@pytest.mark.asyncio
async def test_patch_replaces_metadata_that_is_not_an_object(sessions: SessionFactory) -> None:
    """``'[1,2]'::jsonb || '{"a":1}'::jsonb`` appends instead of merging."""
    thread_id = await _create_thread(sessions, {})
    async with sessions() as session:
        await session.execute(
            text("UPDATE thread SET metadata_json = '[1, 2]'::jsonb WHERE thread_id = :thread_id"),
            {"thread_id": thread_id},
        )
        await session.commit()

    async with sessions() as session:
        await update_thread(thread_id, ThreadUpdate(metadata={"a": 1}), user=_user(), session=session)

    metadata = await _read_metadata(sessions, thread_id)
    assert metadata == {"a": 1}


@pytest.mark.asyncio
async def test_run_preparation_names_only_an_unnamed_thread(sessions: SessionFactory) -> None:
    """thread_name is derived from the first run's input and then left alone."""
    thread_id = await _create_thread(sessions, {"thread_name": ""})

    async with sessions() as session:
        await update_thread_metadata(
            session,
            thread_id,
            assistant_id="asst-1",
            graph_id="agent",
            input_data={"messages": [{"role": "human", "content": "First question"}]},
        )
        await session.commit()

    assert (await _read_metadata(sessions, thread_id))["thread_name"] == "First question"

    async with sessions() as session:
        await update_thread_metadata(
            session,
            thread_id,
            assistant_id="asst-2",
            graph_id="other",
            input_data={"messages": [{"role": "human", "content": "Second question"}]},
        )
        await session.commit()

    metadata = await _read_metadata(sessions, thread_id)
    assert metadata["thread_name"] == "First question"
    assert metadata["assistant_id"] == "asst-2"
    assert metadata["graph_id"] == "other"

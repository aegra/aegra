"""input.inject storage against a real PostgreSQL: admission, order, claims, carry-over, expiry.

These pin what a session fake cannot show: the claim predicate that joins
``run_inputs`` to ``runs``, the unique key behind idempotent repeats, and the
foreign key that removes a thread's messages with it. They skip when no
database is reachable.
"""

from collections.abc import AsyncIterator, Callable
from datetime import timedelta
from typing import Any
from uuid import uuid4

import pytest
from sqlalchemy import delete, func, select, text, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import RunInput as RunInputORM
from aegra_api.core.orm import Thread as ThreadORM
from aegra_api.services import injected_input
from aegra_api.services.injected_input import InjectedInput, mark_delivered, store_message, thread_admission
from aegra_api.settings import settings

_USER_ID = "injected-input-test-user"
_OTHER_USER_ID = "injected-input-other-user"

SessionFactory = Callable[[], AsyncSession]


@pytest.fixture
async def sessions(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[SessionFactory]:
    """Sessions on the test database, also used by the reader; skip without a database."""
    engine = create_async_engine(settings.db.database_url)
    try:
        table = None
        try:
            async with engine.begin() as conn:
                table = await conn.scalar(text("SELECT to_regclass('public.run_inputs')"))
        except (OSError, SQLAlchemyError) as exc:
            pytest.skip(f"PostgreSQL test database is unavailable: {exc}")

        if table is None:
            pytest.skip("run_inputs table is unavailable; run Alembic migrations before this DB test")

        maker = async_sessionmaker(engine, expire_on_commit=False)
        monkeypatch.setattr(injected_input, "_get_session_maker", lambda: maker)
        try:
            yield maker
        finally:
            async with maker() as cleanup:
                await cleanup.execute(delete(ThreadORM).where(ThreadORM.user_id.in_((_USER_ID, _OTHER_USER_ID))))
                await cleanup.commit()
    finally:
        await engine.dispose()


async def _thread(sessions: SessionFactory, *, user_id: str = _USER_ID) -> str:
    thread_id = f"injected-input-{uuid4()}"
    async with sessions() as session:
        session.add(ThreadORM(thread_id=thread_id, status="busy", metadata_json={}, user_id=user_id))
        await session.commit()
    return thread_id


async def _run(sessions: SessionFactory, thread_id: str, *, status: str = "running") -> str:
    run_id = str(uuid4())
    async with sessions() as session:
        session.add(RunORM(run_id=run_id, thread_id=thread_id, status=status, user_id=_USER_ID))
        await session.commit()
    return run_id


async def _set_status(sessions: SessionFactory, run_id: str, status: str) -> None:
    async with sessions() as session:
        await session.execute(update(RunORM).where(RunORM.run_id == run_id).values(status=status))
        await session.commit()


def _message(message_id: str, content: str = "hi") -> dict[str, Any]:
    return {"role": "user", "content": content, "id": message_id, "additional_kwargs": {"injected_by": _USER_ID}}


async def _store(sessions: SessionFactory, thread_id: str, message_id: str, content: str = "hi") -> bool:
    async with sessions() as session:
        return await store_message(
            session,
            thread_id=thread_id,
            user_id=_USER_ID,
            message_id=message_id,
            message=_message(message_id, content),
        )


def _reader(thread_id: str, run_id: str) -> InjectedInput:
    return InjectedInput(run_id=run_id, thread_id=thread_id, user_id=_USER_ID)


async def _pending_ids(reader: InjectedInput) -> list[str]:
    return [item.message_id for item in await reader.pending()]


async def _admission(sessions: SessionFactory, thread_id: str, *, user_id: str = _USER_ID) -> str:
    async with sessions() as session:
        return await thread_admission(session, thread_id=thread_id, user_id=user_id)


async def _age(sessions: SessionFactory, thread_id: str, message_id: str) -> None:
    """Move a message's creation past the TTL."""
    past = timedelta(seconds=settings.event_streaming.INPUT_INJECT_TTL_SECONDS + 60)
    async with sessions() as session:
        await session.execute(
            update(RunInputORM)
            .where(RunInputORM.thread_id == thread_id, RunInputORM.message_id == message_id)
            .values(created_at=func.now() - past)
        )
        await session.commit()


async def _ids_in_table(sessions: SessionFactory, thread_id: str) -> list[str]:
    async with sessions() as session:
        rows = await session.scalars(select(RunInputORM.message_id).where(RunInputORM.thread_id == thread_id))
        return sorted(rows.all())


class TestAdmission:
    async def test_a_thread_without_an_active_run_admits_nothing(self, sessions: SessionFactory) -> None:
        thread_id = await _thread(sessions)
        await _run(sessions, thread_id, status="success")

        assert await _admission(sessions, thread_id) == "no_active_run"

    async def test_an_active_run_admits_input_only_after_it_accepts(self, sessions: SessionFactory) -> None:
        thread_id = await _thread(sessions)
        run_id = await _run(sessions, thread_id)

        assert await _admission(sessions, thread_id) == "not_accepting"
        await _reader(thread_id, run_id).accept()
        assert await _admission(sessions, thread_id) == "accepting"

    async def test_a_requeued_run_keeps_its_declaration(self, sessions: SessionFactory) -> None:
        thread_id = await _thread(sessions)
        run_id = await _run(sessions, thread_id)
        await _reader(thread_id, run_id).accept()

        await _set_status(sessions, run_id, "pending")

        assert await _admission(sessions, thread_id) == "accepting"

    async def test_only_the_thread_owner_is_admitted(self, sessions: SessionFactory) -> None:
        thread_id = await _thread(sessions)
        run_id = await _run(sessions, thread_id)
        await _reader(thread_id, run_id).accept()

        assert await _admission(sessions, thread_id, user_id=_OTHER_USER_ID) == "no_active_run"


class TestStorage:
    async def test_messages_come_back_in_send_order(self, sessions: SessionFactory) -> None:
        thread_id = await _thread(sessions)
        run_id = await _run(sessions, thread_id)
        for message_id in ("c", "a", "b"):
            assert await _store(sessions, thread_id, message_id)

        items = await _reader(thread_id, run_id).pending()

        assert [item.message_id for item in items] == ["c", "a", "b"]
        assert items[0].message == _message("c")

    async def test_an_identical_repeat_is_one_message(self, sessions: SessionFactory) -> None:
        thread_id = await _thread(sessions)
        run_id = await _run(sessions, thread_id)

        assert await _store(sessions, thread_id, "m1")
        assert await _store(sessions, thread_id, "m1")

        assert await _pending_ids(_reader(thread_id, run_id)) == ["m1"]

    async def test_a_reused_id_with_a_different_body_is_refused(self, sessions: SessionFactory) -> None:
        thread_id = await _thread(sessions)
        run_id = await _run(sessions, thread_id)
        await _store(sessions, thread_id, "m1", "first")

        assert await _store(sessions, thread_id, "m1", "second") is False

        items = await _reader(thread_id, run_id).pending()
        assert [item.message["content"] for item in items] == ["first"]

    async def test_message_ids_are_per_thread(self, sessions: SessionFactory) -> None:
        first, second = await _thread(sessions), await _thread(sessions)

        assert await _store(sessions, first, "m1", "one")
        assert await _store(sessions, second, "m1", "two")

    async def test_another_users_reader_sees_nothing(self, sessions: SessionFactory) -> None:
        thread_id = await _thread(sessions)
        run_id = await _run(sessions, thread_id)
        await _store(sessions, thread_id, "m1")

        foreign = InjectedInput(run_id=run_id, thread_id=thread_id, user_id=_OTHER_USER_ID)

        assert await foreign.pending() == []
        assert await foreign.take(["m1"]) == []

    async def test_deleting_the_thread_removes_its_messages(self, sessions: SessionFactory) -> None:
        thread_id = await _thread(sessions)
        await _store(sessions, thread_id, "m1")

        async with sessions() as session:
            await session.execute(delete(ThreadORM).where(ThreadORM.thread_id == thread_id))
            await session.commit()

        assert await _ids_in_table(sessions, thread_id) == []


class TestClaims:
    async def test_an_attempt_never_reads_what_it_took(self, sessions: SessionFactory) -> None:
        thread_id = await _thread(sessions)
        reader = _reader(thread_id, await _run(sessions, thread_id))
        await _store(sessions, thread_id, "m1")
        await _store(sessions, thread_id, "m2")

        assert await reader.take(["m1"]) == ["m1"]

        assert await _pending_ids(reader) == ["m2"]
        assert await reader.take(["m1"]) == []

    async def test_a_taken_message_is_hidden_from_another_active_run(self, sessions: SessionFactory) -> None:
        thread_id = await _thread(sessions)
        first = _reader(thread_id, await _run(sessions, thread_id))
        second = _reader(thread_id, await _run(sessions, thread_id))
        await _store(sessions, thread_id, "m1")

        await first.take(["m1"])

        assert await _pending_ids(second) == []
        assert await second.take(["m1"]) == []

    async def test_take_reports_only_what_it_claimed(self, sessions: SessionFactory) -> None:
        thread_id = await _thread(sessions)
        first = _reader(thread_id, await _run(sessions, thread_id))
        second = _reader(thread_id, await _run(sessions, thread_id))
        await _store(sessions, thread_id, "m1")
        await _store(sessions, thread_id, "m2")
        await first.take(["m1"])

        assert await second.take(["m1", "m2", "missing"]) == ["m2"]

    async def test_a_requeued_run_sees_again_what_its_earlier_attempt_took(self, sessions: SessionFactory) -> None:
        thread_id = await _thread(sessions)
        run_id = await _run(sessions, thread_id)
        earlier = _reader(thread_id, run_id)
        await _store(sessions, thread_id, "m1")
        await earlier.take(["m1"])

        await _set_status(sessions, run_id, "pending")
        requeued = _reader(thread_id, run_id)

        assert await _pending_ids(requeued) == ["m1"]
        assert await requeued.take(["m1"]) == ["m1"]
        assert await _pending_ids(requeued) == []

    @pytest.mark.parametrize("ended", ["error", "interrupted", "success"])
    async def test_the_claim_of_a_run_that_ended_undelivered_is_void(
        self, sessions: SessionFactory, ended: str
    ) -> None:
        thread_id = await _thread(sessions)
        first_run = await _run(sessions, thread_id)
        await _store(sessions, thread_id, "m1")
        await _reader(thread_id, first_run).take(["m1"])

        await _set_status(sessions, first_run, ended)
        second = _reader(thread_id, await _run(sessions, thread_id))

        assert await _pending_ids(second) == ["m1"]

    async def test_delivered_messages_never_come_back(self, sessions: SessionFactory) -> None:
        thread_id = await _thread(sessions)
        first_run = await _run(sessions, thread_id)
        await _store(sessions, thread_id, "m1")
        await _reader(thread_id, first_run).take(["m1"])

        await mark_delivered(run_id=first_run, thread_id=thread_id, user_id=_USER_ID)
        await _set_status(sessions, first_run, "success")
        second = _reader(thread_id, await _run(sessions, thread_id))

        assert await _pending_ids(second) == []
        assert await second.take(["m1"]) == []

    async def test_an_untaken_message_carries_to_the_next_run(self, sessions: SessionFactory) -> None:
        thread_id = await _thread(sessions)
        first_run = await _run(sessions, thread_id)
        await _store(sessions, thread_id, "m1")

        await mark_delivered(run_id=first_run, thread_id=thread_id, user_id=_USER_ID)
        await _set_status(sessions, first_run, "success")
        second = _reader(thread_id, await _run(sessions, thread_id))

        assert await _pending_ids(second) == ["m1"]

    async def test_another_users_run_end_delivers_nothing(self, sessions: SessionFactory) -> None:
        thread_id = await _thread(sessions)
        first_run = await _run(sessions, thread_id)
        await _store(sessions, thread_id, "m1")
        await _reader(thread_id, first_run).take(["m1"])

        await mark_delivered(run_id=first_run, thread_id=thread_id, user_id=_OTHER_USER_ID)
        await _set_status(sessions, first_run, "success")
        second = _reader(thread_id, await _run(sessions, thread_id))

        assert await _pending_ids(second) == ["m1"]


class TestExpiry:
    async def test_an_expired_message_is_never_delivered(self, sessions: SessionFactory) -> None:
        thread_id = await _thread(sessions)
        run_id = await _run(sessions, thread_id)
        await _store(sessions, thread_id, "old")
        await _store(sessions, thread_id, "new")
        await _age(sessions, thread_id, "old")

        reader = _reader(thread_id, run_id)
        assert await _pending_ids(reader) == ["new"]
        assert await reader.take(["old"]) == []

    async def test_the_next_inject_removes_expired_messages(self, sessions: SessionFactory) -> None:
        thread_id = await _thread(sessions)
        await _store(sessions, thread_id, "old")
        await _age(sessions, thread_id, "old")

        await _store(sessions, thread_id, "new")

        assert await _ids_in_table(sessions, thread_id) == ["new"]

    async def test_an_expired_id_can_be_sent_again(self, sessions: SessionFactory) -> None:
        thread_id = await _thread(sessions)
        run_id = await _run(sessions, thread_id)
        await _store(sessions, thread_id, "m1", "first")
        await _age(sessions, thread_id, "m1")

        assert await _store(sessions, thread_id, "m1", "second")

        items = await _reader(thread_id, run_id).pending()
        assert [item.message["content"] for item in items] == ["second"]

    async def test_a_run_end_removes_expired_messages(self, sessions: SessionFactory) -> None:
        thread_id = await _thread(sessions)
        run_id = await _run(sessions, thread_id)
        await _store(sessions, thread_id, "old")
        await _store(sessions, thread_id, "new")
        await _age(sessions, thread_id, "old")

        await mark_delivered(run_id=run_id, thread_id=thread_id, user_id=_USER_ID)

        assert await _ids_in_table(sessions, thread_id) == ["new"]

    async def test_a_requeued_run_sees_again_an_aged_message_it_took(self, sessions: SessionFactory) -> None:
        thread_id = await _thread(sessions)
        run_id = await _run(sessions, thread_id)
        await _store(sessions, thread_id, "m1")
        await _reader(thread_id, run_id).take(["m1"])
        await _age(sessions, thread_id, "m1")

        await _set_status(sessions, run_id, "pending")

        assert await _pending_ids(_reader(thread_id, run_id)) == ["m1"]

    async def test_an_aged_message_a_failed_run_took_carries_to_the_next_run(self, sessions: SessionFactory) -> None:
        thread_id = await _thread(sessions)
        first_run = await _run(sessions, thread_id)
        await _store(sessions, thread_id, "m1")
        await _reader(thread_id, first_run).take(["m1"])
        await _age(sessions, thread_id, "m1")

        await _set_status(sessions, first_run, "error")
        second = _reader(thread_id, await _run(sessions, thread_id))

        assert await _pending_ids(second) == ["m1"]

    async def test_the_next_inject_keeps_an_aged_message_a_run_took(self, sessions: SessionFactory) -> None:
        thread_id = await _thread(sessions)
        run_id = await _run(sessions, thread_id)
        await _store(sessions, thread_id, "m1")
        await _reader(thread_id, run_id).take(["m1"])
        await _age(sessions, thread_id, "m1")

        await _store(sessions, thread_id, "m2")

        assert await _ids_in_table(sessions, thread_id) == ["m1", "m2"]

    async def test_a_run_end_keeps_an_aged_message_another_run_took(self, sessions: SessionFactory) -> None:
        thread_id = await _thread(sessions)
        failed_run = await _run(sessions, thread_id)
        await _store(sessions, thread_id, "m1")
        await _reader(thread_id, failed_run).take(["m1"])
        await _age(sessions, thread_id, "m1")
        await _set_status(sessions, failed_run, "error")

        await mark_delivered(run_id=await _run(sessions, thread_id), thread_id=thread_id, user_id=_USER_ID)

        assert await _ids_in_table(sessions, thread_id) == ["m1"]

    async def test_the_next_inject_removes_an_aged_delivered_message(self, sessions: SessionFactory) -> None:
        thread_id = await _thread(sessions)
        run_id = await _run(sessions, thread_id)
        await _store(sessions, thread_id, "m1")
        await _reader(thread_id, run_id).take(["m1"])
        await mark_delivered(run_id=run_id, thread_id=thread_id, user_id=_USER_ID)
        await _age(sessions, thread_id, "m1")

        await _store(sessions, thread_id, "m2")

        assert await _ids_in_table(sessions, thread_id) == ["m2"]

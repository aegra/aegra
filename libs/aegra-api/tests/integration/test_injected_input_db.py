"""input.inject storage against a real PostgreSQL: admission, order, claims, carry-over, expiry.

These pin what a session fake cannot show: the claim predicate that joins
``run_inputs`` to ``runs``, the unique key behind idempotent repeats, the
foreign key that removes a thread's messages with it, and the nodes LangGraph
retries, runs in parallel, or runs again inside a subgraph. They skip when no
database is reachable.
"""

import asyncio
from collections import Counter, defaultdict
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import timedelta
from typing import Any, TypedDict
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy, Send
from sqlalchemy import delete, func, select, text, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import RunInput as RunInputORM
from aegra_api.core.orm import Thread as ThreadORM
from aegra_api.services import injected_input
from aegra_api.services.injected_input import (
    INJECTED_INPUT_KEY,
    InjectedInput,
    mark_delivered,
    pin_injected_input,
    store_message,
    thread_admission,
)
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
    async def test_a_later_node_of_the_attempt_never_reads_what_an_earlier_node_took(
        self, sessions: SessionFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        thread_id = await _thread(sessions)
        reader = _reader(thread_id, await _run(sessions, thread_id))
        await _store(sessions, thread_id, "m1")
        await _store(sessions, thread_id, "m2")
        monkeypatch.setattr(injected_input, "_current_task", lambda: "first_node:1")
        assert await reader.take(["m1"]) == ["m1"]

        monkeypatch.setattr(injected_input, "_current_task", lambda: "second_node:2")

        assert await _pending_ids(reader) == ["m2"]
        assert await reader.take(["m1"]) == []

    async def test_a_retried_node_sees_again_what_its_failed_try_took(
        self, sessions: SessionFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        thread_id = await _thread(sessions)
        reader = _reader(thread_id, await _run(sessions, thread_id))
        await _store(sessions, thread_id, "m1")
        monkeypatch.setattr(injected_input, "_current_task", lambda: "call_model:1")
        assert await reader.take(["m1"]) == ["m1"]

        assert await _pending_ids(reader) == ["m1"]
        assert await reader.take(["m1"]) == ["m1"]

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

    async def test_a_requeued_run_sees_again_what_its_earlier_attempt_took(
        self, sessions: SessionFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        thread_id = await _thread(sessions)
        run_id = await _run(sessions, thread_id)
        earlier = _reader(thread_id, run_id)
        await _store(sessions, thread_id, "m1")
        monkeypatch.setattr(injected_input, "_current_task", lambda: "work:1")
        await earlier.take(["m1"])

        await _set_status(sessions, run_id, "pending")
        requeued = _reader(thread_id, run_id)
        # The requeued attempt may resume past the node that took it, at a later checkpoint.
        monkeypatch.setattr(injected_input, "_current_task", lambda: "work:2")

        assert await _pending_ids(requeued) == ["m1"]
        assert await requeued.take(["m1"]) == ["m1"]
        monkeypatch.setattr(injected_input, "_current_task", lambda: "work:3")
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


async def _take_as(reader: InjectedInput, task: str, message_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(injected_input, "_current_task", lambda: task)
    assert await reader.take([message_id]) == [message_id]


async def _claimant(sessions: SessionFactory, thread_id: str, message_id: str) -> str | None:
    async with sessions() as session:
        return await session.scalar(
            select(RunInputORM.taken_by_task).where(
                RunInputORM.thread_id == thread_id, RunInputORM.message_id == message_id
            )
        )


class TestRelease:
    async def test_a_released_claim_is_pending_again_for_the_retry(
        self, sessions: SessionFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        thread_id = await _thread(sessions)
        reader = _reader(thread_id, await _run(sessions, thread_id))
        await _store(sessions, thread_id, "m1")
        await _take_as(reader, "agent:p|call_model:t1", "m1", monkeypatch)

        await reader.release("agent:p")

        monkeypatch.setattr(injected_input, "_current_task", lambda: "agent:p|1|call_model:t2")
        assert await _pending_ids(reader) == ["m1"]
        assert await reader.take(["m1"]) == ["m1"]

    async def test_release_reaches_the_task_and_the_tasks_below_it_only(
        self, sessions: SessionFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        thread_id = await _thread(sessions)
        reader = _reader(thread_id, await _run(sessions, thread_id))
        claims = {
            "m1": "call_model:t1",
            "m2": "call_model:t1|tools:t2",
            "m3": "call_model:t12|tools:t3",
            "m4": "callXmodel:t1|tools:t2",
        }
        for message_id, task in claims.items():
            await _store(sessions, thread_id, message_id)
            await _take_as(reader, task, message_id, monkeypatch)

        await reader.release("call_model:t1")

        monkeypatch.setattr(injected_input, "_current_task", lambda: "other:t9")
        assert await _pending_ids(reader) == ["m1", "m2"]

    async def test_release_keeps_what_an_earlier_attempt_took(
        self, sessions: SessionFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        thread_id = await _thread(sessions)
        run_id = await _run(sessions, thread_id)
        await _store(sessions, thread_id, "m1")
        await _take_as(_reader(thread_id, run_id), "work:t1", "m1", monkeypatch)

        await _reader(thread_id, run_id).release("work:t1")

        assert await _claimant(sessions, thread_id, "m1") == "work:t1"

    async def test_release_keeps_what_another_run_took(
        self, sessions: SessionFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        thread_id = await _thread(sessions)
        await _store(sessions, thread_id, "m1")
        await _take_as(_reader(thread_id, await _run(sessions, thread_id)), "work:t1", "m1", monkeypatch)

        await _reader(thread_id, await _run(sessions, thread_id)).release("work:t1")

        assert await _claimant(sessions, thread_id, "m1") == "work:t1"


def _model_failing_once(prompts: list[list[str]]) -> Callable[..., Awaitable[dict[str, Any]]]:
    """The streaming guide's snippet as a node, with a model that fails once with a retryable error."""

    async def call_model(state: MessagesState, config: RunnableConfig) -> dict[str, Any]:
        reader = config["configurable"].get(INJECTED_INPUT_KEY)
        injected = []
        if reader is not None:
            await reader.accept()
            pending = await reader.pending()
            taken = set(await reader.take([item.message_id for item in pending]))
            held = {message.id for message in state["messages"]}
            injected = [item.message for item in pending if item.message_id in taken and item.message_id not in held]
        prompts.append([str(message.text) for message in state["messages"]] + [item["content"] for item in injected])
        if len(prompts) == 1:
            raise ConnectionError("model provider unavailable")
        return {"messages": [*injected, AIMessage(content="ok")]}

    return call_model


class TestNodeRetries:
    async def test_a_node_langgraph_retries_reads_again_what_its_failed_try_took(
        self, sessions: SessionFactory
    ) -> None:
        thread_id = await _thread(sessions)
        run_id = await _run(sessions, thread_id)
        await _store(sessions, thread_id, "m1", "also check the tests")
        prompts: list[list[str]] = []
        graph = (
            StateGraph(MessagesState)
            .add_node(
                "call_model",
                _model_failing_once(prompts),
                retry_policy=RetryPolicy(max_attempts=3, initial_interval=0.01),
            )
            .add_edge(START, "call_model")
            .compile(checkpointer=InMemorySaver())
        )
        config = pin_injected_input({"configurable": {"thread_id": thread_id}}, _reader(thread_id, run_id))

        result = await graph.ainvoke({"messages": [HumanMessage(content="start", id="h1")]}, config)
        await mark_delivered(run_id=run_id, thread_id=thread_id, user_id=_USER_ID)
        await _set_status(sessions, run_id, "success")

        assert prompts == [["start", "also check the tests"], ["start", "also check the tests"]]
        assert [message.content for message in result["messages"]] == ["start", "also check the tests", "ok"]
        assert await _pending_ids(_reader(thread_id, await _run(sessions, thread_id))) == []

    async def test_a_subgraph_node_langgraph_retries_reads_again_what_its_failed_try_took(
        self, sessions: SessionFactory
    ) -> None:
        thread_id = await _thread(sessions)
        run_id = await _run(sessions, thread_id)
        await _store(sessions, thread_id, "m1", "also check the tests")
        prompts: list[list[str]] = []
        # LangGraph re-runs a retried subgraph under a new namespace, so its inner node gets new task ids.
        agent = (
            StateGraph(MessagesState)
            .add_node("call_model", _model_failing_once(prompts))
            .add_edge(START, "call_model")
            .compile()
        )
        graph = (
            StateGraph(MessagesState)
            .add_node("agent", agent, retry_policy=RetryPolicy(max_attempts=3, initial_interval=0.01))
            .add_edge(START, "agent")
            .compile(checkpointer=InMemorySaver())
        )
        config = pin_injected_input({"configurable": {"thread_id": thread_id}}, _reader(thread_id, run_id))

        result = await graph.ainvoke({"messages": [HumanMessage(content="start", id="h1")]}, config)
        await mark_delivered(run_id=run_id, thread_id=thread_id, user_id=_USER_ID)
        await _set_status(sessions, run_id, "success")

        assert prompts == [["start", "also check the tests"], ["start", "also check the tests"]]
        assert [message.content for message in result["messages"]] == ["start", "also check the tests", "ok"]
        assert await _pending_ids(_reader(thread_id, await _run(sessions, thread_id))) == []

    async def test_a_later_node_does_not_read_what_an_earlier_node_took(self, sessions: SessionFactory) -> None:
        thread_id = await _thread(sessions)
        run_id = await _run(sessions, thread_id)
        await _store(sessions, thread_id, "m1")
        seen_later: list[str] = []

        async def first(state: MessagesState, config: RunnableConfig) -> dict[str, Any]:
            reader = config["configurable"][INJECTED_INPUT_KEY]
            await reader.take(await _pending_ids(reader))
            await _store(sessions, thread_id, "m2")
            return {}

        async def later(state: MessagesState, config: RunnableConfig) -> dict[str, Any]:
            seen_later.extend(await _pending_ids(config["configurable"][INJECTED_INPUT_KEY]))
            return {}

        graph = (
            StateGraph(MessagesState)
            .add_node("first", first)
            .add_node("later", later)
            .add_edge(START, "first")
            .add_edge("first", "later")
            .compile(checkpointer=InMemorySaver())
        )
        config = pin_injected_input({"configurable": {"thread_id": thread_id}}, _reader(thread_id, run_id))

        await graph.ainvoke({"messages": [HumanMessage(content="start", id="h1")]}, config)

        assert seen_later == ["m2"]


class _Work(TypedDict):
    worker: int


async def _take_injected(config: RunnableConfig, held: set[str]) -> list[dict[str, Any]]:
    """The streaming guide's snippet: take every pending message, keep those the node claimed and the state lacks."""
    reader = config["configurable"][INJECTED_INPUT_KEY]
    pending = await reader.pending()
    taken = set(await reader.take([item.message_id for item in pending]))
    return [item.message for item in pending if item.message_id in taken and item.message_id not in held]


def _fanning_out(worker: Callable[[_Work, RunnableConfig], Awaitable[dict[str, Any]]]) -> CompiledStateGraph:
    """A compiled subgraph that runs ``worker`` as two parallel Send tasks."""

    def fan_out(state: MessagesState) -> list[Send]:
        return [Send("worker", {"worker": 1}), Send("worker", {"worker": 2})]

    return (
        StateGraph(MessagesState).add_node("worker", worker).add_conditional_edges(START, fan_out, ["worker"]).compile()
    )


class TestNodesOfASubgraph:
    async def test_parallel_tasks_of_a_subgraph_take_a_message_once(self, sessions: SessionFactory) -> None:
        thread_id = await _thread(sessions)
        run_id = await _run(sessions, thread_id)
        await _store(sessions, thread_id, "m1", "also check the tests")
        took: dict[int, list[str]] = {}

        async def worker(state: _Work, config: RunnableConfig) -> dict[str, Any]:
            injected = await _take_injected(config, held=set())
            took[state["worker"]] = [item["content"] for item in injected]
            return {"messages": injected}

        graph = (
            StateGraph(MessagesState)
            .add_node("agent", _fanning_out(worker))
            .add_edge(START, "agent")
            .compile(checkpointer=InMemorySaver())
        )
        config = pin_injected_input({"configurable": {"thread_id": thread_id}}, _reader(thread_id, run_id))

        result = await graph.ainvoke({"messages": [HumanMessage(content="start", id="h1")]}, config)

        assert sorted(took.values()) == [[], ["also check the tests"]]
        assert [message.content for message in result["messages"]] == ["start", "also check the tests"]

    async def test_a_retried_subgraph_gives_a_message_to_one_parallel_task_per_try(
        self, sessions: SessionFactory
    ) -> None:
        thread_id = await _thread(sessions)
        run_id = await _run(sessions, thread_id)
        await _store(sessions, thread_id, "m1", "also check the tests")
        took: list[tuple[int, int, list[str]]] = []
        tries: Counter[int] = Counter()
        first_took: defaultdict[int, asyncio.Event] = defaultdict(asyncio.Event)

        async def worker(state: _Work, config: RunnableConfig) -> dict[str, Any]:
            number = state["worker"]
            tries[number] += 1
            attempt = tries[number]
            # Worker 2 reads after worker 1 took, then fails its first try, so LangGraph retries the subgraph.
            if number == 2:
                await first_took[attempt].wait()
            injected = await _take_injected(config, held=set())
            took.append((attempt, number, [item["content"] for item in injected]))
            first_took[attempt].set()
            if number == 2 and attempt == 1:
                raise ConnectionError("model provider unavailable")
            return {"messages": injected}

        graph = (
            StateGraph(MessagesState)
            .add_node("agent", _fanning_out(worker), retry_policy=RetryPolicy(max_attempts=3, initial_interval=0.01))
            .add_edge(START, "agent")
            .compile(checkpointer=InMemorySaver())
        )
        config = pin_injected_input({"configurable": {"thread_id": thread_id}}, _reader(thread_id, run_id))

        result = await graph.ainvoke({"messages": [HumanMessage(content="start", id="h1")]}, config)
        await mark_delivered(run_id=run_id, thread_id=thread_id, user_id=_USER_ID)
        await _set_status(sessions, run_id, "success")

        assert sorted(took) == [
            (1, 1, ["also check the tests"]),
            (1, 2, []),
            (2, 1, ["also check the tests"]),
            (2, 2, []),
        ]
        assert [message.content for message in result["messages"]] == ["start", "also check the tests"]
        assert await _pending_ids(_reader(thread_id, await _run(sessions, thread_id))) == []

    async def test_a_node_a_subgraph_runs_again_is_not_offered_what_its_first_run_took(
        self, sessions: SessionFactory
    ) -> None:
        thread_id = await _thread(sessions)
        run_id = await _run(sessions, thread_id)
        await _store(sessions, thread_id, "m1", "also check the tests")
        offered: list[list[str]] = []

        async def call_model(state: MessagesState, config: RunnableConfig) -> dict[str, Any]:
            reader = config["configurable"][INJECTED_INPUT_KEY]
            pending = await reader.pending()
            offered.append([item.message_id for item in pending])
            taken = set(await reader.take([item.message_id for item in pending]))
            return {"messages": [*(item.message for item in pending if item.message_id in taken), AIMessage("ok")]}

        def tools(state: MessagesState) -> dict[str, Any]:
            return {}

        def route(state: MessagesState) -> str:
            return "tools" if len(offered) == 1 else END

        # A ReAct loop: call_model runs again after tools, inside a compiled subgraph and without the held filter.
        agent = (
            StateGraph(MessagesState)
            .add_node("call_model", call_model)
            .add_node("tools", tools)
            .add_edge(START, "call_model")
            .add_conditional_edges("call_model", route, ["tools", END])
            .add_edge("tools", "call_model")
            .compile()
        )
        graph = (
            StateGraph(MessagesState)
            .add_node("agent", agent)
            .add_edge(START, "agent")
            .compile(checkpointer=InMemorySaver())
        )
        config = pin_injected_input({"configurable": {"thread_id": thread_id}}, _reader(thread_id, run_id))

        result = await graph.ainvoke({"messages": [HumanMessage(content="start", id="h1")]}, config)

        assert offered == [["m1"], []]
        assert [message.content for message in result["messages"]] == ["start", "also check the tests", "ok", "ok"]


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

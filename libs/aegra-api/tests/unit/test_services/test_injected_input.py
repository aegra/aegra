"""The injected-input reader keys a claim by the LangGraph task that made it, and a task that fails releases them."""

import asyncio
from uuid import UUID, uuid4

import pytest
from langchain_core.runnables import RunnableLambda
from langgraph.errors import GraphInterrupt, ParentCommand
from langgraph.types import Command

from aegra_api.services.injected_input import InjectedInput, ReleaseOnFailure, _current_task


def _key_in_node(checkpoint_ns: str) -> str | None:
    """What the reader records for a claim made under this LangGraph checkpoint namespace."""
    runnable: RunnableLambda[None, str | None] = RunnableLambda(lambda _: _current_task())
    return runnable.invoke(None, config={"configurable": {"checkpoint_ns": checkpoint_ns}})


class TestClaimKey:
    def test_outside_a_graph_node_there_is_no_key(self) -> None:
        assert _current_task() is None

    def test_the_key_is_the_tasks_namespace(self) -> None:
        assert _key_in_node("agent:p|1|model:t2") == "agent:p|1|model:t2"

    @pytest.mark.parametrize(
        ("one", "other"),
        [
            pytest.param("first:t1", "later:t2", id="later node"),
            pytest.param("worker:t1", "worker:t2", id="parallel Send tasks"),
            pytest.param("agent:p|first:t1", "agent:p|later:t2", id="later node of a subgraph"),
            pytest.param("agent:p|worker:t1", "agent:p|worker:t2", id="parallel Send tasks of a subgraph"),
            pytest.param("agent:p|call_model:t1", "agent:p|call_model:t2", id="a node a subgraph runs again"),
            pytest.param("agent:p|model:t1", "agent:q|model:t2", id="the subgraph run again as another task"),
        ],
    )
    def test_other_nodes_key_their_claims_apart(self, one: str, other: str) -> None:
        assert _key_in_node(one) != _key_in_node(other)


@pytest.fixture
def released(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The namespaces the reader is asked to release."""
    calls: list[str] = []

    async def release(self: InjectedInput, task: str) -> None:
        calls.append(task)

    monkeypatch.setattr(InjectedInput, "release", release)
    return calls


def _handler() -> ReleaseOnFailure:
    return ReleaseOnFailure(InjectedInput(run_id="run-1", thread_id="thread-1", user_id="user-1"))


async def _start(handler: ReleaseOnFailure, *, namespace: str | None, parent: UUID | None = None) -> UUID:
    run_id = uuid4()
    metadata = {} if namespace is None else {"langgraph_checkpoint_ns": namespace}
    await handler.on_chain_start({}, {}, run_id=run_id, parent_run_id=parent, metadata=metadata)
    return run_id


class TestReleaseOnFailure:
    async def test_a_task_that_fails_releases_its_claims(self, released: list[str]) -> None:
        handler = _handler()
        graph = await _start(handler, namespace=None)
        task = await _start(handler, namespace="call_model:t1", parent=graph)

        await handler.on_chain_error(ConnectionError("model provider unavailable"), run_id=task, parent_run_id=graph)

        assert released == ["call_model:t1"]

    async def test_a_task_that_ends_releases_nothing(self, released: list[str]) -> None:
        handler = _handler()
        graph = await _start(handler, namespace=None)
        task = await _start(handler, namespace="call_model:t1", parent=graph)

        await handler.on_chain_end({}, run_id=task, parent_run_id=graph)
        await handler.on_chain_error(ValueError("late"), run_id=task, parent_run_id=graph)

        assert released == []

    @pytest.mark.parametrize(
        "error",
        [
            pytest.param(GraphInterrupt(), id="interrupt"),
            pytest.param(ParentCommand(Command(goto="next")), id="command to the parent graph"),
            pytest.param(asyncio.CancelledError(), id="cancel"),
        ],
    )
    async def test_an_interrupt_or_a_cancel_releases_nothing(self, released: list[str], error: BaseException) -> None:
        handler = _handler()
        graph = await _start(handler, namespace=None)
        task = await _start(handler, namespace="call_model:t1", parent=graph)

        await handler.on_chain_error(error, run_id=task, parent_run_id=graph)

        assert released == []

    async def test_an_error_of_a_run_outside_a_task_releases_nothing(self, released: list[str]) -> None:
        handler = _handler()
        graph = await _start(handler, namespace=None)

        await handler.on_chain_error(ValueError("boom"), run_id=graph)
        await handler.on_chain_error(ValueError("boom"), run_id=uuid4())

        assert released == []

    async def test_a_failing_run_nested_in_a_task_leaves_the_task_to_recover(self, released: list[str]) -> None:
        handler = _handler()
        graph = await _start(handler, namespace=None)
        task = await _start(handler, namespace="call_model:t1", parent=graph)
        nested = await _start(handler, namespace="call_model:t1", parent=task)

        await handler.on_chain_error(ValueError("parser failed"), run_id=nested, parent_run_id=task)

        assert released == []
        await handler.on_chain_error(ValueError("parser failed"), run_id=task, parent_run_id=graph)
        assert released == ["call_model:t1"]

    async def test_a_failing_subgraph_releases_its_failed_task_and_then_itself(self, released: list[str]) -> None:
        handler = _handler()
        graph = await _start(handler, namespace=None)
        agent = await _start(handler, namespace="agent:p", parent=graph)
        subgraph = await _start(handler, namespace="agent:p", parent=agent)
        model = await _start(handler, namespace="agent:p|model:t1", parent=subgraph)
        error = ConnectionError("model provider unavailable")

        await handler.on_chain_error(error, run_id=model, parent_run_id=subgraph)
        await handler.on_chain_error(error, run_id=subgraph, parent_run_id=agent)
        await handler.on_chain_error(error, run_id=agent, parent_run_id=graph)

        assert released == ["agent:p|model:t1", "agent:p"]

    async def test_a_release_that_fails_fails_the_task(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def release(self: InjectedInput, task: str) -> None:
            raise ConnectionRefusedError(111, "Connect call failed")

        def fail(_: None) -> None:
            raise ValueError("model provider unavailable")

        monkeypatch.setattr(InjectedInput, "release", release)
        runnable: RunnableLambda[None, None] = RunnableLambda(fail)

        with pytest.raises(ConnectionRefusedError):
            await runnable.ainvoke(
                None, config={"callbacks": [_handler()], "metadata": {"langgraph_checkpoint_ns": "call_model:t1"}}
            )

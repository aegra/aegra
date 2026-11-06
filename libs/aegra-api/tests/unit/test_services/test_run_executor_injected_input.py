"""The run executor pins the injected-input reader and settles taken messages when a graph completes."""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.exc import OperationalError

from aegra_api.models.auth import User
from aegra_api.models.run_job import RunExecution, RunIdentity, RunJob
from aegra_api.services import run_executor as run_executor_module
from aegra_api.services.injected_input import INJECTED_INPUT_KEY, InjectedInput
from aegra_api.services.run_executor import _build_run_config, _stream_graph


def _job(config: dict[str, Any] | None = None) -> RunJob:
    return RunJob(
        identity=RunIdentity(run_id="run-1", thread_id="thread-1", graph_id="graph-1"),
        user=User(identity="user-1"),
        execution=RunExecution(input_data={"messages": []}, config=config or {}),
    )


class TestReaderIsPinned:
    def test_execution_config_carries_a_reader_bound_to_the_run(self) -> None:
        config = _build_run_config(_job())

        reader = config["configurable"][INJECTED_INPUT_KEY]
        assert isinstance(reader, InjectedInput)
        assert (reader.run_id, reader.thread_id, reader.user_id) == ("run-1", "thread-1", "user-1")

    def test_a_client_value_under_the_key_is_replaced(self) -> None:
        config = _build_run_config(_job({"configurable": {INJECTED_INPUT_KEY: {"run_id": "someone-elses-run"}}}))

        reader = config["configurable"][INJECTED_INPUT_KEY]
        assert isinstance(reader, InjectedInput)
        assert reader.run_id == "run-1"


@pytest.fixture
def graph_service(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    @asynccontextmanager
    async def get_graph(*_args: Any, **_kwargs: Any) -> AsyncGenerator[MagicMock]:
        yield MagicMock()

    service = MagicMock()
    service.get_graph = get_graph
    monkeypatch.setattr(run_executor_module, "get_langgraph_service", lambda: service)
    return service


@pytest.fixture
def delivered(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    mock = AsyncMock()
    monkeypatch.setattr(run_executor_module, "mark_delivered", mock)
    return mock


class TestTakenMessagesSettle:
    async def test_a_completed_graph_marks_its_taken_messages_delivered(
        self, graph_service: MagicMock, delivered: AsyncMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(run_executor_module, "_stream_legacy", AsyncMock())

        await _stream_graph(_job())

        delivered.assert_awaited_once_with(run_id="run-1", thread_id="thread-1", user_id="user-1")

    async def test_a_graph_paused_at_an_interrupt_leaves_them_to_the_resume(
        self, graph_service: MagicMock, delivered: AsyncMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def pause(*args: Any) -> None:
            args[-1].has_interrupt = True

        monkeypatch.setattr(run_executor_module, "_stream_legacy", pause)

        result = await _stream_graph(_job())

        assert result.has_interrupt is True
        delivered.assert_not_awaited()

    async def test_a_graph_that_raises_leaves_them_for_the_next_run(
        self, graph_service: MagicMock, delivered: AsyncMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(run_executor_module, "_stream_legacy", AsyncMock(side_effect=RuntimeError("boom")))

        with pytest.raises(RuntimeError):
            await _stream_graph(_job())

        delivered.assert_not_awaited()

    async def test_a_cancelled_graph_leaves_them_for_the_next_run(
        self, graph_service: MagicMock, delivered: AsyncMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(run_executor_module, "_stream_legacy", AsyncMock(side_effect=asyncio.CancelledError()))

        with pytest.raises(asyncio.CancelledError):
            await _stream_graph(_job())

        delivered.assert_not_awaited()

    async def test_a_database_failure_while_settling_does_not_fail_the_run(
        self, graph_service: MagicMock, delivered: AsyncMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(run_executor_module, "_stream_legacy", AsyncMock())
        delivered.side_effect = OperationalError("UPDATE run_inputs", {}, Exception("connection lost"))
        warning = MagicMock()
        monkeypatch.setattr(run_executor_module.logger, "warning", warning)

        result = await _stream_graph(_job())

        assert result.has_interrupt is False
        warning.assert_called_once()

    async def test_a_refused_connection_while_settling_does_not_fail_the_run(
        self, graph_service: MagicMock, delivered: AsyncMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(run_executor_module, "_stream_legacy", AsyncMock())
        # asyncpg raises connect-time failures as OSError, which SQLAlchemy does not wrap.
        delivered.side_effect = ConnectionRefusedError(111, "Connect call failed")
        warning = MagicMock()
        monkeypatch.setattr(run_executor_module.logger, "warning", warning)

        result = await _stream_graph(_job())

        assert result.has_interrupt is False
        warning.assert_called_once()

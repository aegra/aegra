import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from aegra_api.api import threads
from aegra_api.models import User


class TrackingRow:
    def __init__(self, session: "TrackingSession", metadata: dict[str, Any]) -> None:
        self.session = session
        self.values = {
            "thread_id": "thread-lifecycle",
            "user_id": "owner-1",
            "status": "idle",
            "metadata_json": metadata,
            "created_at": datetime(2026, 1, 1, tzinfo=UTC),
            "updated_at": datetime(2026, 1, 2, tzinfo=UTC),
        }

    def __getattr__(self, name: str) -> Any:
        if name not in self.values:
            raise AttributeError(name)
        assert not self.session.closed, f"ORM field {name} accessed after session closure"
        return self.values[name]


class TrackingSession:
    def __init__(self, metadata: dict[str, Any], *, found: bool = True) -> None:
        self.closed = False
        self.close_count = 0
        self.row = TrackingRow(self, metadata) if found else None
        self.statement: Any = None

    async def scalar(self, statement: Any) -> TrackingRow | None:
        self.statement = statement
        return self.row

    async def close(self) -> None:
        self.closed = True
        self.close_count += 1


class BlockingAgent:
    def __init__(self, entered: asyncio.Event, release: asyncio.Event, *, failure: bool) -> None:
        self.entered, self.release, self.failure = entered, release, failure

    def with_config(self, config: dict[str, Any]) -> "BlockingAgent":
        assert config["configurable"]["thread_id"] == "thread-lifecycle"
        return self

    async def aget_state(self, config: dict[str, Any], **kwargs: Any) -> None:
        self.entered.set()
        await self.release.wait()
        if self.failure:
            raise ValueError("synthetic checkpoint failure")
        return None


@pytest.mark.parametrize("outcome", ["success", "error", "cancel"])
@pytest.mark.asyncio
async def test_get_thread_closes_session_before_blocked_checkpoint(
    monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    session = TrackingSession({"graph_id": "graph-lifecycle", "tenant": "demo"})
    entered, release = asyncio.Event(), asyncio.Event()
    agent = BlockingAgent(entered, release, failure=outcome == "error")
    cleaned: list[bool] = []
    acquisitions: list[dict[str, Any]] = []

    @asynccontextmanager
    async def get_graph(graph_id: str, **kwargs: Any) -> AsyncIterator[BlockingAgent]:
        assert session.closed, "session must close before graph factory acquisition"
        assert graph_id == "graph-lifecycle"
        acquisitions.append(kwargs)
        try:
            yield agent
        finally:
            cleaned.append(True)

    class Service:
        def list_graphs(self) -> list[str]:
            return ["graph-lifecycle"]

    service = Service()
    monkeypatch.setattr(service, "get_graph", get_graph, raising=False)
    monkeypatch.setattr(threads, "handle_event", AsyncMock(return_value={"tenant": "demo"}))
    monkeypatch.setattr(threads, "get_langgraph_service", lambda: service)
    user = User(identity="owner-1", scopes=[])
    task = asyncio.create_task(threads.get_thread("thread-lifecycle", user=user, session=session))
    waiter = asyncio.create_task(entered.wait())
    try:
        done, _ = await asyncio.wait([task, waiter], timeout=1)
        if task in done:
            await task
        assert entered.is_set(), "checkpoint read must be reached"
        assert session.closed and session.close_count == 1
        params = session.statement.compile().params
        assert "owner-1" in params.values() and "thread-lifecycle" in params.values()
        assert {"tenant": "demo"} in params.values(), params
        assert acquisitions[0]["user"] is user
        assert acquisitions[0]["access_context"] == "threads.read"
        if outcome == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            release.set()
            if outcome == "error":
                with pytest.raises(ValueError, match="synthetic checkpoint failure"):
                    await task
            else:
                response = await task
                assert response.thread_id == "thread-lifecycle" and response.user_id == "owner-1"
                assert response.metadata == {"graph_id": "graph-lifecycle", "tenant": "demo"}
                assert response.values == {} and response.interrupts == {} and response.config == {}
        assert cleaned == [True]
    finally:
        release.set()
        for pending in (task, waiter):
            if not pending.done():
                pending.cancel()
        await asyncio.gather(task, waiter, return_exceptions=True)


@pytest.mark.asyncio
async def test_denied_thread_does_not_acquire_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    session = TrackingSession({}, found=False)
    acquire = AsyncMock()
    monkeypatch.setattr(threads, "handle_event", AsyncMock(return_value={"tenant": "allowed"}))
    monkeypatch.setattr(threads, "get_langgraph_service", acquire)
    with pytest.raises(HTTPException) as error:
        await threads.get_thread("thread-lifecycle", user=User(identity="intruder", scopes=[]), session=session)
    assert error.value.status_code == 404
    assert "intruder" in session.statement.compile().params.values()
    assert {"tenant": "allowed"} in session.statement.compile().params.values()
    acquire.assert_not_called()


@pytest.mark.asyncio
async def test_no_graph_metadata_keeps_empty_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    session = TrackingSession({"tenant": "demo"})
    acquire = AsyncMock()
    monkeypatch.setattr(threads, "handle_event", AsyncMock(return_value={}))
    monkeypatch.setattr(threads, "get_langgraph_service", acquire)
    response = await threads.get_thread("thread-lifecycle", user=User(identity="owner-1", scopes=[]), session=session)
    assert response.metadata == {"tenant": "demo"} and response.state_updated_at is None
    assert response.values == {} and response.interrupts == {} and response.config == {}
    assert session.close_count == 1
    acquire.assert_not_called()

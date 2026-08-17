"""Regression tests for aegra/aegra#517.

Aborting an in-flight thread history/state request could leave its
SQLAlchemy session (and asyncpg connection) checked out, because the
endpoint held the session open across the long-running, cancellable
LangGraph checkpoint call. These endpoints now look up the thread's
graph_id via a short-lived session that is closed *before* the LangGraph
call starts, so a cancellation during that call can no longer leak the
connection. Each test fails loudly if a future change reverses the order.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from aegra_api.api.threads import (
    get_thread_history_post,
    get_thread_state,
    get_thread_state_at_checkpoint,
    update_thread_state,
)
from aegra_api.models import ThreadHistoryRequest, ThreadStateUpdate, ThreadStateUpdateResponse, User


class _TrackingSession:
    """Fake AsyncSession that records when it is closed via __aexit__."""

    def __init__(self, thread_row: Any, closed_marker: list[bool]) -> None:
        self._thread_row = thread_row
        self._closed_marker = closed_marker

    async def scalar(self, _stmt: Any) -> Any:
        return self._thread_row

    async def __aenter__(self) -> "_TrackingSession":
        return self

    async def __aexit__(self, *exc_info: object) -> bool:
        self._closed_marker.append(True)
        return False


class _AssertsSessionClosedAgent:
    """Fake LangGraph agent that fails the test if the session isn't closed yet."""

    def __init__(self, closed_marker: list[bool], snapshots: list[Any] | None = None) -> None:
        self._closed_marker = closed_marker
        self._snapshots = snapshots or []

    def with_config(self, _config: dict[str, Any]) -> "_AssertsSessionClosedAgent":
        return self

    async def aget_state(self, _config: dict[str, Any], **_kwargs: Any) -> None:
        assert self._closed_marker, "session must be released before aget_state starts"
        return None

    async def aget_state_history(self, _config: dict[str, Any], **_kwargs: Any) -> AsyncIterator[Any]:
        assert self._closed_marker, "session must be released before aget_state_history starts"
        for snapshot in self._snapshots:
            yield snapshot

    async def aupdate_state(self, _config: dict[str, Any], _values: Any, as_node: str | None = None) -> dict:
        assert self._closed_marker, "session must be released before aupdate_state starts"
        return {"configurable": {"checkpoint_id": "cp-1", "checkpoint_ns": ""}}


def _patch_session_maker(thread_row: Any, closed_marker: list[bool]) -> Any:
    session = _TrackingSession(thread_row, closed_marker)
    return patch("aegra_api.api.threads._get_session_maker", return_value=MagicMock(return_value=session))


def _patch_langgraph_service(agent: _AssertsSessionClosedAgent) -> Any:
    @asynccontextmanager
    async def _get_graph(*_args: Any, **_kwargs: Any) -> AsyncIterator[_AssertsSessionClosedAgent]:
        yield agent

    mock_service = MagicMock()
    mock_service.get_graph = _get_graph
    return patch("aegra_api.services.langgraph_service.get_langgraph_service", return_value=mock_service)


class TestSessionReleasedBeforeCheckpointRead:
    @pytest.mark.asyncio
    async def test_get_thread_state_releases_session_before_aget_state(self) -> None:
        user = User(identity="user-1", scopes=[])
        thread_row = MagicMock()
        thread_row.metadata_json = {"graph_id": "graph-123"}
        closed_marker: list[bool] = []
        agent = _AssertsSessionClosedAgent(closed_marker)

        with (
            _patch_session_maker(thread_row, closed_marker),
            _patch_langgraph_service(agent),
            pytest.raises(HTTPException) as exc_info,
        ):
            await get_thread_state("thread-123", user=user)

        assert exc_info.value.status_code == 404
        assert closed_marker == [True]

    @pytest.mark.asyncio
    async def test_get_thread_state_at_checkpoint_releases_session_before_aget_state(self) -> None:
        user = User(identity="user-1", scopes=[])
        thread_row = MagicMock()
        thread_row.metadata_json = {"graph_id": "graph-123"}
        closed_marker: list[bool] = []
        agent = _AssertsSessionClosedAgent(closed_marker)

        with (
            _patch_session_maker(thread_row, closed_marker),
            _patch_langgraph_service(agent),
            pytest.raises(HTTPException) as exc_info,
        ):
            await get_thread_state_at_checkpoint("thread-123", "checkpoint-1", user=user)

        assert exc_info.value.status_code == 404
        assert closed_marker == [True]

    @pytest.mark.asyncio
    async def test_update_thread_state_releases_session_before_aupdate_state(self) -> None:
        user = User(identity="user-1", scopes=[])
        thread_row = MagicMock()
        thread_row.metadata_json = {"graph_id": "graph-123"}
        closed_marker: list[bool] = []
        agent = _AssertsSessionClosedAgent(closed_marker)

        with (
            _patch_session_maker(thread_row, closed_marker),
            _patch_langgraph_service(agent),
        ):
            result = await update_thread_state(
                "thread-123",
                ThreadStateUpdate(values={"foo": "bar"}),
                user=user,
            )

        assert isinstance(result, ThreadStateUpdateResponse)
        assert result.checkpoint["checkpoint_id"] == "cp-1"
        assert closed_marker == [True]

    @pytest.mark.asyncio
    async def test_get_thread_history_post_releases_session_before_aget_state_history(self) -> None:
        user = User(identity="user-1", scopes=[])
        thread_row = MagicMock()
        thread_row.metadata_json = {"graph_id": "graph-123"}
        closed_marker: list[bool] = []
        agent = _AssertsSessionClosedAgent(closed_marker)

        with (
            _patch_session_maker(thread_row, closed_marker),
            _patch_langgraph_service(agent),
        ):
            result = await get_thread_history_post(
                "thread-123",
                ThreadHistoryRequest(limit=10),
                user=user,
            )

        assert result == []
        assert closed_marker == [True]

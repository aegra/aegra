"""Unit tests for thread state retrieval endpoint."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from aegra_api.api.threads import get_thread_state
from aegra_api.models import User


def create_get_graph_mock(return_value=None, side_effect=None):
    """Create a mock for get_graph that works as an async context manager."""

    @asynccontextmanager
    async def async_cm(*args, **kwargs):
        if side_effect:
            raise side_effect
        yield return_value

    def get_graph(*args, **kwargs):
        return async_cm(*args, **kwargs)

    return get_graph


def _make_session_maker(session: AsyncMock) -> MagicMock:
    """Build a mock async_sessionmaker that always yields the given session.

    get_thread_state manages its session manually via _get_session_maker
    (not Depends(get_session)) so the pooled connection is released before
    any LangGraph checkpoint read starts.
    """
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return MagicMock(return_value=ctx)


class TestGetThreadState:
    """Exercise edge cases and success path for get_thread_state."""

    @pytest.mark.asyncio
    async def test_thread_not_found(self):
        """404 returned when the thread is missing for the user."""
        user = User(identity="user-1", scopes=[])
        session = AsyncMock()
        session.scalar.return_value = None

        with (
            patch("aegra_api.api.threads._get_session_maker", return_value=_make_session_maker(session)),
            pytest.raises(HTTPException) as exc_info,
        ):
            await get_thread_state("thread-123", user=user)

        assert exc_info.value.status_code == 404
        assert "not found" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_missing_graph_id(self):
        """Empty state returned when thread metadata does not include a graph."""
        user = User(identity="user-1", scopes=[])
        session = AsyncMock()
        thread_row = MagicMock()
        thread_row.metadata_json = {}
        session.scalar.return_value = thread_row

        with patch("aegra_api.api.threads._get_session_maker", return_value=_make_session_maker(session)):
            result = await get_thread_state("thread-123", user=user)

        # get_thread_state returns ThreadState Pydantic model
        assert hasattr(result, "values")
        assert hasattr(result, "checkpoint")
        assert result.checkpoint.checkpoint_id is None

    @pytest.mark.asyncio
    async def test_graph_load_failure(self):
        """500 returned when LangGraph graph loading fails."""
        user = User(identity="user-1", scopes=[])
        session = AsyncMock()
        thread_row = MagicMock()
        thread_row.metadata_json = {"graph_id": "graph-123"}
        session.scalar.return_value = thread_row

        with (
            patch("aegra_api.api.threads._get_session_maker", return_value=_make_session_maker(session)),
            patch("aegra_api.services.langgraph_service.get_langgraph_service") as mock_service,
        ):
            mock_service.return_value.get_graph = create_get_graph_mock(side_effect=Exception("boom"))

            with pytest.raises(HTTPException) as exc_info:
                await get_thread_state("thread-123", user=user)

        assert exc_info.value.status_code == 500
        assert "failed to retrieve thread state" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_no_state_snapshot_found(self):
        """404 returned when LangGraph does not provide a snapshot."""
        user = User(identity="user-1", scopes=[])
        session = AsyncMock()
        thread_row = MagicMock()
        thread_row.metadata_json = {"graph_id": "graph-123"}
        session.scalar.return_value = thread_row

        mock_agent = MagicMock()
        mock_agent.with_config.return_value = mock_agent
        mock_agent.aget_state = AsyncMock(return_value=None)

        with (
            patch("aegra_api.api.threads._get_session_maker", return_value=_make_session_maker(session)),
            patch("aegra_api.services.langgraph_service.get_langgraph_service") as mock_service,
            patch(
                "aegra_api.services.langgraph_service.create_thread_config",
                return_value={"configurable": {}},
            ),
        ):
            mock_service.return_value.get_graph = create_get_graph_mock(return_value=mock_agent)

            with pytest.raises(HTTPException) as exc_info:
                await get_thread_state("thread-123", user=user)

        assert exc_info.value.status_code == 404
        assert "no state found" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_conversion_failure(self):
        """500 returned when snapshot conversion raises."""
        user = User(identity="user-1", scopes=[])
        session = AsyncMock()
        thread_row = MagicMock()
        thread_row.metadata_json = {"graph_id": "graph-123"}
        session.scalar.return_value = thread_row

        mock_agent = MagicMock()
        mock_agent.with_config.return_value = mock_agent
        mock_agent.aget_state = AsyncMock(return_value={"values": {}})

        with (
            patch("aegra_api.api.threads._get_session_maker", return_value=_make_session_maker(session)),
            patch("aegra_api.services.langgraph_service.get_langgraph_service") as mock_service,
            patch(
                "aegra_api.services.langgraph_service.create_thread_config",
                return_value={"configurable": {}},
            ),
            patch(
                "aegra_api.api.threads.thread_state_service.convert_snapshot_to_thread_state",
                side_effect=Exception("convert failed"),
            ) as mock_convert,
        ):
            mock_service.return_value.get_graph = create_get_graph_mock(return_value=mock_agent)

            with pytest.raises(HTTPException) as exc_info:
                await get_thread_state("thread-123", user=user)

        assert exc_info.value.status_code == 500
        assert "failed to retrieve thread state" in exc_info.value.detail.lower()
        mock_convert.assert_called_once()

    @pytest.mark.asyncio
    async def test_success_with_checkpoint_ns_and_subgraphs(self):
        """Successful retrieval includes checkpoint namespace and subgraph flag."""
        user = User(identity="user-1", scopes=[])
        session = AsyncMock()
        thread_row = MagicMock()
        thread_row.metadata_json = {"graph_id": "graph-123"}
        session.scalar.return_value = thread_row

        mock_agent = MagicMock()
        mock_agent.with_config.return_value = mock_agent
        mock_agent.aget_state = AsyncMock(return_value={"values": {"foo": "bar"}})

        mock_thread_state = MagicMock()
        mock_thread_state.checkpoint = MagicMock(checkpoint_id="cp-999")

        config = {"configurable": {}}

        with (
            patch("aegra_api.api.threads._get_session_maker", return_value=_make_session_maker(session)),
            patch("aegra_api.services.langgraph_service.get_langgraph_service") as mock_service,
            patch(
                "aegra_api.services.langgraph_service.create_thread_config",
                return_value=config,
            ),
            patch(
                "aegra_api.api.threads.thread_state_service.convert_snapshot_to_thread_state",
                return_value=mock_thread_state,
            ) as mock_convert,
        ):
            mock_service.return_value.get_graph = create_get_graph_mock(return_value=mock_agent)

            result = await get_thread_state(
                "thread-123",
                subgraphs=True,
                checkpoint_ns="ns-1",
                user=user,
            )

        assert result is mock_thread_state
        assert config["configurable"]["checkpoint_ns"] == "ns-1"
        mock_agent.aget_state.assert_awaited_once_with(config, subgraphs=True)
        mock_convert.assert_called_once_with({"values": {"foo": "bar"}}, "thread-123", subgraphs=True)

    @pytest.mark.asyncio
    async def test_http_exception_passthrough(self):
        """HTTPException raised by LangGraph agent is propagated unchanged."""
        user = User(identity="user-1", scopes=[])
        session = AsyncMock()
        thread_row = MagicMock()
        thread_row.metadata_json = {"graph_id": "graph-123"}
        session.scalar.return_value = thread_row

        mock_agent = MagicMock()
        mock_agent.with_config.return_value = mock_agent
        mock_agent.aget_state = AsyncMock(side_effect=HTTPException(status_code=418, detail="teapot"))

        with (
            patch("aegra_api.api.threads._get_session_maker", return_value=_make_session_maker(session)),
            patch("aegra_api.services.langgraph_service.get_langgraph_service") as mock_service,
            patch(
                "aegra_api.services.langgraph_service.create_thread_config",
                return_value={"configurable": {}},
            ),
        ):
            mock_service.return_value.get_graph = create_get_graph_mock(return_value=mock_agent)

            with pytest.raises(HTTPException) as exc_info:
                await get_thread_state("thread-123", user=user)

        assert exc_info.value.status_code == 418
        assert exc_info.value.detail == "teapot"

    @pytest.mark.asyncio
    async def test_unexpected_error_wrapped(self):
        """Unexpected exceptions become HTTP 500 responses."""
        user = User(identity="user-1", scopes=[])
        session = AsyncMock()
        session.scalar.side_effect = Exception("database down")

        with (
            patch("aegra_api.api.threads._get_session_maker", return_value=_make_session_maker(session)),
            pytest.raises(HTTPException) as exc_info,
        ):
            await get_thread_state("thread-123", user=user)

        assert exc_info.value.status_code == 500
        assert "error retrieving thread state" in exc_info.value.detail.lower()

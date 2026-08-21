"""Unit tests for thread state at checkpoint endpoint."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from aegra_api.api.threads import (
    get_thread_state_at_checkpoint,
    get_thread_state_at_checkpoint_post,
)
from aegra_api.models import ThreadCheckpoint, ThreadCheckpointPostRequest, User


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

    get_thread_state_at_checkpoint manages its session manually via
    _get_session_maker (not Depends(get_session)) so the pooled connection is
    released before any LangGraph checkpoint read starts.
    """
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return MagicMock(return_value=ctx)


class TestGetThreadStateAtCheckpoint:
    """Exercise edge cases and success path for get_thread_state_at_checkpoint."""

    @pytest.mark.asyncio
    async def test_success_with_subgraphs_passed_to_service(self):
        """Verify subgraphs parameter is passed to convert_snapshot_to_thread_state."""
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

            result = await get_thread_state_at_checkpoint(
                "thread-123",
                "checkpoint-1",
                subgraphs=True,
                user=user,
            )

        assert result is mock_thread_state
        assert config["configurable"]["checkpoint_id"] == "checkpoint-1"
        mock_agent.aget_state.assert_awaited_once_with(config, subgraphs=True)
        # Verify subgraphs was passed to conversion service
        mock_convert.assert_called_once_with({"values": {"foo": "bar"}}, "thread-123", subgraphs=True)

    @pytest.mark.asyncio
    async def test_success_with_checkpoint_ns(self):
        """Verify checkpoint_ns parameter is properly set in config."""
        user = User(identity="user-1", scopes=[])
        session = AsyncMock()
        thread_row = MagicMock()
        thread_row.metadata_json = {"graph_id": "graph-123"}
        session.scalar.return_value = thread_row

        mock_agent = MagicMock()
        mock_agent.with_config.return_value = mock_agent
        mock_agent.aget_state = AsyncMock(return_value={"values": {}})

        mock_thread_state = MagicMock()

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
            ),
        ):
            mock_service.return_value.get_graph = create_get_graph_mock(return_value=mock_agent)

            await get_thread_state_at_checkpoint(
                "thread-123",
                "checkpoint-1",
                subgraphs=False,
                checkpoint_ns="my-namespace",
                user=user,
            )

        assert config["configurable"]["checkpoint_id"] == "checkpoint-1"
        assert config["configurable"]["checkpoint_ns"] == "my-namespace"
        mock_agent.aget_state.assert_awaited_once_with(config, subgraphs=False)

    @pytest.mark.asyncio
    async def test_checkpoint_ns_none_not_set(self):
        """Verify checkpoint_ns=None doesn't set the config key."""
        user = User(identity="user-1", scopes=[])
        session = AsyncMock()
        thread_row = MagicMock()
        thread_row.metadata_json = {"graph_id": "graph-123"}
        session.scalar.return_value = thread_row

        mock_agent = MagicMock()
        mock_agent.with_config.return_value = mock_agent
        mock_agent.aget_state = AsyncMock(return_value={"values": {}})

        mock_thread_state = MagicMock()

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
            ),
        ):
            mock_service.return_value.get_graph = create_get_graph_mock(return_value=mock_agent)

            await get_thread_state_at_checkpoint(
                "thread-123",
                "checkpoint-1",
                subgraphs=False,
                checkpoint_ns=None,
                user=user,
            )

        assert config["configurable"]["checkpoint_id"] == "checkpoint-1"
        assert "checkpoint_ns" not in config["configurable"]


class TestGetThreadStateAtCheckpointPost:
    """Exercise edge cases and success path for get_thread_state_at_checkpoint_post."""

    @pytest.mark.asyncio
    async def test_success_with_checkpoint_ns(self):
        """Verify checkpoint_ns from request is passed to GET endpoint."""
        user = User(identity="user-1", scopes=[])

        checkpoint = ThreadCheckpoint(
            checkpoint_id="checkpoint-1",
            checkpoint_ns="my-namespace",
            thread_id="thread-123",
        )
        request = ThreadCheckpointPostRequest(checkpoint=checkpoint, subgraphs=True)

        mock_thread_state = MagicMock()

        with patch(
            "aegra_api.api.threads.get_thread_state_at_checkpoint",
            return_value=mock_thread_state,
        ) as mock_get:
            result = await get_thread_state_at_checkpoint_post("thread-123", request, user=user)

        assert result is mock_thread_state
        # Verify GET endpoint was called with checkpoint_ns
        mock_get.assert_called_once_with(
            "thread-123",
            "checkpoint-1",
            True,  # subgraphs
            "my-namespace",  # checkpoint_ns
            user,
        )

    @pytest.mark.asyncio
    async def test_success_without_checkpoint_ns(self):
        """Verify None checkpoint_ns is handled correctly."""
        user = User(identity="user-1", scopes=[])

        checkpoint = ThreadCheckpoint(
            checkpoint_id="checkpoint-1",
            checkpoint_ns="",  # Empty string should become None
            thread_id="thread-123",
        )
        request = ThreadCheckpointPostRequest(checkpoint=checkpoint, subgraphs=False)

        mock_thread_state = MagicMock()

        with patch(
            "aegra_api.api.threads.get_thread_state_at_checkpoint",
            return_value=mock_thread_state,
        ) as mock_get:
            result = await get_thread_state_at_checkpoint_post("thread-123", request, user=user)

        assert result is mock_thread_state
        # Verify GET endpoint was called with None checkpoint_ns
        mock_get.assert_called_once_with(
            "thread-123",
            "checkpoint-1",
            False,  # subgraphs
            None,  # checkpoint_ns (empty string becomes None)
            user,
        )

    @pytest.mark.asyncio
    async def test_missing_checkpoint_id_raises_error(self):
        """Verify missing checkpoint_id raises 400 error."""
        user = User(identity="user-1", scopes=[])

        checkpoint = ThreadCheckpoint(
            checkpoint_id=None,  # Missing checkpoint_id
            checkpoint_ns="my-namespace",
            thread_id="thread-123",
        )
        request = ThreadCheckpointPostRequest(checkpoint=checkpoint, subgraphs=False)

        with pytest.raises(HTTPException) as exc_info:
            await get_thread_state_at_checkpoint_post("thread-123", request, user=user)

        assert exc_info.value.status_code == 400
        assert "checkpoint_id is required" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_subgraphs_passed_correctly(self):
        """Verify subgraphs parameter from request is passed through."""
        user = User(identity="user-1", scopes=[])

        checkpoint = ThreadCheckpoint(
            checkpoint_id="checkpoint-1",
            checkpoint_ns=None,
            thread_id="thread-123",
        )
        request = ThreadCheckpointPostRequest(checkpoint=checkpoint, subgraphs=True)

        mock_thread_state = MagicMock()

        with patch(
            "aegra_api.api.threads.get_thread_state_at_checkpoint",
            return_value=mock_thread_state,
        ) as mock_get:
            await get_thread_state_at_checkpoint_post("thread-123", request, user=user)

        # Verify subgraphs=True was passed
        call_args = mock_get.call_args[0]
        assert call_args[2] is True  # subgraphs parameter

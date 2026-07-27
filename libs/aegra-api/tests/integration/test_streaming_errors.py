"""Integration tests for streaming error handling.

Tests verify that errors during graph execution are properly caught,
sent to frontend via SSE, and stored for replay.
"""

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from aegra_api.models import User
from aegra_api.models.run_job import RunExecution, RunIdentity, RunJob
from aegra_api.services import streaming_service as streaming_service_module
from aegra_api.services.broker import BrokerManager, RunBroker
from aegra_api.services.run_executor import execute_run as execute_run_async


@pytest.mark.asyncio
class TestStreamingErrorHandling:
    """Test error handling during streaming execution"""

    @pytest.fixture
    def mock_user(self) -> User:
        return User(identity="test-user")

    @pytest.fixture
    def run_id(self) -> str:
        return str(uuid4())

    @pytest.fixture
    def thread_id(self) -> str:
        return str(uuid4())

    @pytest.fixture
    def local_broker_manager(self, monkeypatch: pytest.MonkeyPatch) -> BrokerManager:
        """Provide a fresh BrokerManager for each test, patched into the modules under test."""
        manager = BrokerManager()
        monkeypatch.setattr(streaming_service_module, "broker_manager", manager)
        return manager

    async def test_error_during_event_processing_sent_to_frontend(
        self, mock_user: User, run_id: str, thread_id: str, local_broker_manager: BrokerManager
    ) -> None:
        """Test that errors during event processing are caught and sent immediately"""
        graph_id = "test-graph"

        # Create a broker to capture events
        broker = RunBroker(run_id)
        local_broker_manager._brokers[run_id] = broker

        # Mock graph that yields events, then fails during processing
        mock_graph = MagicMock()
        mock_graph.__aenter__ = AsyncMock(return_value=mock_graph)
        mock_graph.__aexit__ = AsyncMock(return_value=None)

        async def failing_stream() -> AsyncGenerator[tuple[str, Any], None]:
            """Stream that yields one event then raises error"""
            yield ("values", {"step": 1})
            # Simulate error during event processing
            raise ValueError("Graph execution failed")

        with (
            patch("aegra_api.services.run_executor.get_langgraph_service") as mock_lg_service,
            patch(
                "aegra_api.services.run_executor.stream_graph_events",
                return_value=failing_stream(),
            ),
            patch("aegra_api.services.run_executor.try_mark_run_running", new_callable=AsyncMock, return_value=True),
            patch("aegra_api.services.run_executor.finalize_run", new_callable=AsyncMock),
        ):
            mock_lg_service.return_value.get_graph.return_value.__aenter__ = AsyncMock(return_value=mock_graph)
            mock_lg_service.return_value.get_graph.return_value.__aexit__ = AsyncMock(return_value=None)

            job = RunJob(
                identity=RunIdentity(run_id=run_id, thread_id=thread_id, graph_id=graph_id),
                user=mock_user,
                execution=RunExecution(
                    input_data={},
                    config={},
                    context={},
                    stream_mode=["values"],
                ),
            )

            # Execute run - error is handled internally (no re-raise from background tasks)
            await execute_run_async(job)

            # Verify error event was sent to broker
            events_received = []
            try:
                async for event_id, raw_event in broker.aiter():
                    events_received.append((event_id, raw_event))
                    # Stop after a few events
                    if len(events_received) >= 5:
                        break
            except Exception:
                pass

            # Should have received:
            # 1. The values event
            # 2. An error event
            # 3. An end event
            assert len(events_received) >= 1, "Should receive at least one event"

            # Check for error event
            error_events = [evt for _, evt in events_received if isinstance(evt, tuple) and evt[0] == "error"]
            assert len(error_events) > 0, "Error event should be sent to broker"

            error_event = error_events[0]
            assert error_event[0] == "error"
            assert "error" in error_event[1]
            assert "message" in error_event[1]
            assert error_event[1]["message"] == "ValueError: execution failed"

    async def test_error_stored_for_replay(
        self, mock_user: User, run_id: str, thread_id: str, local_broker_manager: BrokerManager
    ) -> None:
        """Test that error events are stored in the broker's replay buffer"""
        graph_id = "test-graph"

        async def failing_stream() -> AsyncGenerator[tuple[str, Any], None]:
            yield ("values", {"step": 1})
            raise RuntimeError("Storage test error")

        broker = RunBroker(run_id)
        local_broker_manager._brokers[run_id] = broker

        with (
            patch("aegra_api.services.run_executor.get_langgraph_service") as mock_lg_service,
            patch(
                "aegra_api.services.run_executor.stream_graph_events",
                return_value=failing_stream(),
            ),
            patch("aegra_api.services.run_executor.try_mark_run_running", new_callable=AsyncMock, return_value=True),
            patch("aegra_api.services.run_executor.finalize_run", new_callable=AsyncMock),
        ):
            mock_graph = MagicMock()
            mock_lg_service.return_value.get_graph.return_value.__aenter__ = AsyncMock(return_value=mock_graph)
            mock_lg_service.return_value.get_graph.return_value.__aexit__ = AsyncMock(return_value=None)

            job = RunJob(
                identity=RunIdentity(run_id=run_id, thread_id=thread_id, graph_id=graph_id),
                user=mock_user,
                execution=RunExecution(
                    input_data={},
                    config={},
                    context={},
                    stream_mode=["values"],
                ),
            )

            await execute_run_async(job)

            # Verify error was stored in replay buffer
            replay_events = await broker.replay(None)
            error_events = [
                (eid, payload) for eid, payload in replay_events if isinstance(payload, tuple) and payload[0] == "error"
            ]
            assert len(error_events) > 0, "Error event should be stored in replay buffer"
            error_payload = error_events[0][1]
            assert error_payload[1]["error"] == "RuntimeError"
            assert error_payload[1]["message"] == "RuntimeError: execution failed"

    async def test_error_type_preserved(
        self, mock_user: User, run_id: str, thread_id: str, local_broker_manager: BrokerManager
    ) -> None:
        """Test that error type is correctly preserved and sent"""
        graph_id = "test-graph"

        async def failing_stream() -> AsyncGenerator[tuple[str, Any], None]:
            """Async generator that raises ValueError"""
            raise ValueError("Type preservation test")
            yield  # This will never be reached but makes it an async generator

        broker = RunBroker(run_id)
        local_broker_manager._brokers[run_id] = broker

        with (
            patch("aegra_api.services.run_executor.get_langgraph_service") as mock_lg_service,
            patch(
                "aegra_api.services.run_executor.stream_graph_events",
            ) as mock_stream_graph,
            patch("aegra_api.services.run_executor.try_mark_run_running", new_callable=AsyncMock, return_value=True),
            patch("aegra_api.services.run_executor.finalize_run", new_callable=AsyncMock),
        ):
            # Set up the mock to return the async generator
            mock_stream_graph.return_value = failing_stream()

            mock_graph = MagicMock()
            mock_lg_service.return_value.get_graph.return_value.__aenter__ = AsyncMock(return_value=mock_graph)
            mock_lg_service.return_value.get_graph.return_value.__aexit__ = AsyncMock(return_value=None)

            job = RunJob(
                identity=RunIdentity(run_id=run_id, thread_id=thread_id, graph_id=graph_id),
                user=mock_user,
                execution=RunExecution(
                    input_data={},
                    config={},
                    context={},
                    stream_mode=["values"],
                ),
            )

            # Error is handled internally (no re-raise from background tasks)
            await execute_run_async(job)

            # Check error event has correct type
            events_received = []
            try:
                async for event_id, raw_event in broker.aiter():
                    events_received.append((event_id, raw_event))
                    if len(events_received) >= 3:
                        break
            except Exception:
                pass

            error_events = [evt for _, evt in events_received if isinstance(evt, tuple) and evt[0] == "error"]
            assert len(error_events) > 0, "Should receive error event"

            error_event = error_events[0]
            assert error_event[1]["error"] == "ValueError"
            assert error_event[1]["message"] == "ValueError: execution failed"

    async def test_multiple_errors_only_send_once(
        self, mock_user: User, run_id: str, thread_id: str, local_broker_manager: BrokerManager
    ) -> None:
        """Test that if error is caught in inner handler, outer handler doesn't duplicate"""
        graph_id = "test-graph"

        async def failing_stream() -> AsyncGenerator[tuple[str, Any], None]:
            yield ("values", {"step": 1})
            raise KeyError("Single error test")

        broker = RunBroker(run_id)
        local_broker_manager._brokers[run_id] = broker

        with (
            patch("aegra_api.services.run_executor.get_langgraph_service") as mock_lg_service,
            patch(
                "aegra_api.services.run_executor.stream_graph_events",
                return_value=failing_stream(),
            ),
            patch("aegra_api.services.run_executor.try_mark_run_running", new_callable=AsyncMock, return_value=True),
            patch("aegra_api.services.run_executor.finalize_run", new_callable=AsyncMock),
        ):
            mock_graph = MagicMock()
            mock_lg_service.return_value.get_graph.return_value.__aenter__ = AsyncMock(return_value=mock_graph)
            mock_lg_service.return_value.get_graph.return_value.__aexit__ = AsyncMock(return_value=None)

            job = RunJob(
                identity=RunIdentity(run_id=run_id, thread_id=thread_id, graph_id=graph_id),
                user=mock_user,
                execution=RunExecution(
                    input_data={},
                    config={},
                    context={},
                    stream_mode=["values"],
                ),
            )

            # Error is handled internally (no re-raise from background tasks)
            await execute_run_async(job)

            # Count error events - should only be one
            events_received = []
            try:
                async for event_id, raw_event in broker.aiter():
                    events_received.append((event_id, raw_event))
                    if len(events_received) >= 5:
                        break
            except Exception:
                pass

            error_events = [evt for _, evt in events_received if isinstance(evt, tuple) and evt[0] == "error"]
            # Should have exactly one error event (not duplicated)
            assert len(error_events) == 1, f"Expected 1 error event, got {len(error_events)}"

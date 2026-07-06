"""Test stream_mode='events' functionality.

Tests for issue #99: stream_mode="events" does not work
https://github.com/aegra/aegra/issues/99
"""

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock

import pytest

from aegra_api.services.graph_streaming import stream_graph_events


class _RecordingGraph:
    output_channels: list[str] | None = None

    def __init__(self) -> None:
        self.astream_kwargs: dict[str, Any] | None = None
        self.astream_events_kwargs: dict[str, Any] | None = None

    async def astream(self, _input_data: Any, _config: dict[str, Any], **kwargs: Any) -> AsyncIterator[tuple[str, Any]]:
        self.astream_kwargs = kwargs
        yield "values", {"done": True}

    async def astream_events(
        self, _input_data: Any, _config: dict[str, Any], **kwargs: Any
    ) -> AsyncIterator[dict[str, Any]]:
        self.astream_events_kwargs = kwargs
        yield {
            "event": "on_chain_stream",
            "run_id": "run-123",
            "data": {"chunk": ("values", {"done": True})},
        }


class TestEventsMode:
    """Test events mode streaming."""

    @pytest.mark.asyncio
    async def test_events_mode_yields_raw_events(self):
        """Test that stream_mode='events' yields raw events."""
        # Create a mock graph that yields astream_events
        mock_graph = MagicMock()

        # Mock astream_events to yield some events
        async def mock_astream_events(*args, **kwargs):
            # Yield a few different event types
            yield {
                "event": "on_chain_start",
                "name": "test_chain",
                "run_id": "run-123",
                "data": {"input": "test"},
            }
            yield {
                "event": "on_chain_stream",
                "name": "test_chain",
                "run_id": "run-123",
                "data": {"chunk": ("values", {"key": "value"})},
            }
            yield {
                "event": "on_chain_end",
                "name": "test_chain",
                "run_id": "run-123",
                "data": {"output": "result"},
            }

        # Return the async generator directly, not a coroutine
        mock_graph.astream_events = mock_astream_events

        # Mock get_context_jsonschema if it exists
        if hasattr(mock_graph, "get_context_jsonschema"):
            mock_graph.get_context_jsonschema = MagicMock(return_value={})

        config = {"run_id": "run-123", "metadata": {"run_attempt": 1}}
        input_data = {"messages": [{"role": "user", "content": "test"}]}

        # Stream with events mode
        events_yielded = []
        async for mode, payload in stream_graph_events(
            mock_graph,
            input_data,
            config,
            stream_mode=["events"],
        ):
            if mode == "events":
                events_yielded.append((mode, payload))

        # Should yield raw events
        assert len(events_yielded) > 0, "Expected at least one 'events' event to be yielded"

        # Check that we got the raw events
        event_types = [payload.get("event") for _, payload in events_yielded]
        assert "on_chain_start" in event_types or "on_chain_stream" in event_types or "on_chain_end" in event_types

    @pytest.mark.asyncio
    async def test_events_mode_with_on_chain_stream_events(self):
        """Test that on_chain_stream events are also yielded as raw events.

        This test reproduces issue #99: on_chain_stream events are processed
        but not yielded as raw events when stream_mode='events'.
        """
        mock_graph = MagicMock()

        async def mock_astream_events(*args, **kwargs):
            # Yield on_chain_stream events (these are the problematic ones)
            # These get processed but should ALSO be yielded as raw events
            yield {
                "event": "on_chain_stream",
                "name": "test_chain",
                "run_id": "run-123",
                "data": {"chunk": ("values", {"key": "value"})},
            }
            yield {
                "event": "on_chain_stream",
                "name": "test_chain",
                "run_id": "run-123",
                "data": {"chunk": ("debug", {"type": "checkpoint", "payload": {"tasks": []}})},
            }

        # Return the async generator directly, not a coroutine
        mock_graph.astream_events = mock_astream_events

        if hasattr(mock_graph, "get_context_jsonschema"):
            mock_graph.get_context_jsonschema = MagicMock(return_value={})

        config = {"run_id": "run-123", "metadata": {"run_attempt": 1}}
        input_data = {"messages": [{"role": "user", "content": "test"}]}

        # Stream with events mode ONLY (no other modes)
        events_yielded = []
        all_yielded = []
        async for mode, payload in stream_graph_events(
            mock_graph,
            input_data,
            config,
            stream_mode=["events"],  # Only events mode
        ):
            all_yielded.append((mode, payload))
            if mode == "events":
                events_yielded.append((mode, payload))

        # Debug: print what we got
        print(f"\nDEBUG: Total events yielded: {len(all_yielded)}")
        print(f"DEBUG: Event modes: {[m for m, _ in all_yielded]}")
        print(f"DEBUG: Raw 'events' yielded: {len(events_yielded)}")
        print(f"DEBUG: Events payloads event types: {[p.get('event') for _, p in events_yielded]}")

        # Check if we got on_chain_stream events as raw events
        on_chain_stream_events = [p for _, p in events_yielded if p.get("event") == "on_chain_stream"]

        # This test should FAIL with current implementation
        # on_chain_stream events are processed but not yielded as raw events
        # We should get at least 2 raw events (one for each on_chain_stream)
        assert len(on_chain_stream_events) >= 2, (
            f"BUG #99: Expected at least 2 raw 'events' with event='on_chain_stream', "
            f"but got {len(on_chain_stream_events)}. "
            f"Total raw events: {len(events_yielded)}, "
            f"Event types in raw events: {[p.get('event') for _, p in events_yielded]}. "
            f"on_chain_stream events are processed but not yielded as raw events when stream_mode='events'."
        )

    @pytest.mark.asyncio
    async def test_forwards_interrupts_to_astream(self) -> None:
        graph = _RecordingGraph()
        config = {
            "configurable": {"run_id": "run-123"},
            "metadata": {"run_attempt": 1},
            "interrupt_before": ["agent"],
            "interrupt_after": ["tools"],
        }

        async for _mode, _payload in stream_graph_events(
            graph,
            {"messages": []},
            config,
            stream_mode=["values"],
        ):
            pass

        assert graph.astream_kwargs is not None
        assert graph.astream_kwargs["interrupt_before"] == ["agent"]
        assert graph.astream_kwargs["interrupt_after"] == ["tools"]

    @pytest.mark.asyncio
    async def test_forwards_interrupts_to_astream_events(self) -> None:
        graph = _RecordingGraph()
        config = {
            "configurable": {"run_id": "run-123"},
            "metadata": {"run_attempt": 1},
            "interrupt_before": ["agent"],
            "interrupt_after": ["tools"],
        }

        async for _mode, _payload in stream_graph_events(
            graph,
            {"messages": []},
            config,
            stream_mode=["events"],
        ):
            pass

        assert graph.astream_events_kwargs is not None
        assert graph.astream_events_kwargs["interrupt_before"] == ["agent"]
        assert graph.astream_events_kwargs["interrupt_after"] == ["tools"]

    @pytest.mark.asyncio
    async def test_preserves_all_nodes_interrupt_sentinel(self) -> None:
        graph = _RecordingGraph()
        config = {
            "configurable": {"run_id": "run-123"},
            "metadata": {"run_attempt": 1},
            "interrupt_before": ["*"],
            "interrupt_after": ["*"],
        }

        async for _mode, _payload in stream_graph_events(
            graph,
            {"messages": []},
            config,
            stream_mode=["values"],
        ):
            pass

        assert graph.astream_kwargs is not None
        assert graph.astream_kwargs["interrupt_before"] == "*"
        assert graph.astream_kwargs["interrupt_after"] == "*"

    @pytest.mark.asyncio
    async def test_preserves_all_nodes_interrupt_sentinel_astream_events(self) -> None:
        graph = _RecordingGraph()
        config = {
            "configurable": {"run_id": "run-123"},
            "metadata": {"run_attempt": 1},
            "interrupt_before": ["*"],
            "interrupt_after": ["*"],
        }

        async for _mode, _payload in stream_graph_events(
            graph,
            {"messages": []},
            config,
            stream_mode=["events"],
        ):
            pass

        assert graph.astream_events_kwargs is not None
        assert graph.astream_events_kwargs["interrupt_before"] == "*"
        assert graph.astream_events_kwargs["interrupt_after"] == "*"

"""Test stream_mode='events' functionality.

Tests for issue #99: stream_mode="events" does not work
https://github.com/aegra/aegra/issues/99
"""

import asyncio
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from copy import deepcopy
from types import ModuleType
from typing import Any, TypedDict
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from aegra_api.models.auth import User
from aegra_api.models.run_job import RunBehavior, RunExecution, RunIdentity, RunJob
from aegra_api.services import run_executor
from aegra_api.services.graph_streaming import stream_graph_events


class _BreakpointState(TypedDict):
    x: int


def _breakpoint_graph(executed: list[str], *, interrupt_before: list[str] | None = None) -> CompiledStateGraph:
    def a(state: _BreakpointState) -> _BreakpointState:
        executed.append("a")
        return {"x": state["x"] + 1}

    def b(state: _BreakpointState) -> _BreakpointState:
        executed.append("b")
        return {"x": state["x"] + 1}

    builder = StateGraph(_BreakpointState)
    builder.add_node("a", a)
    builder.add_node("b", b)
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    return builder.compile(checkpointer=InMemorySaver(), interrupt_before=interrupt_before)


@pytest.mark.parametrize("stream_mode", [["events"], ["events", "values"], ["events", "updates"], ["updates"]])
@pytest.mark.parametrize("subgraphs", [False, True])
@pytest.mark.parametrize(
    ("interrupt_key", "nodes", "expected_executed", "expected_x", "expected_next"),
    [
        ("interrupt_before", ["b"], ["a"], 1, ("b",)),
        ("interrupt_after", ["a"], ["a"], 1, ("b",)),
        ("interrupt_before", ["*"], [], 0, ("a",)),
        ("interrupt_after", ["*"], ["a"], 1, ("b",)),
    ],
)
async def test_static_breakpoints_stop_execution_and_resume(
    *,
    stream_mode: list[str],
    subgraphs: bool,
    interrupt_key: str,
    nodes: list[str],
    expected_executed: list[str],
    expected_x: int,
    expected_next: tuple[str, ...],
) -> None:
    executed: list[str] = []
    graph = _breakpoint_graph(executed)
    compiled_before = list(graph.interrupt_before_nodes)
    compiled_after = list(graph.interrupt_after_nodes)
    compiled_channels = dict(graph.channels)
    run_id = str(uuid4())
    config: dict[str, Any] = {
        "run_id": run_id,
        "configurable": {"thread_id": str(uuid4()), "run_id": run_id},
        interrupt_key: nodes,
    }

    events = [
        event
        async for event in stream_graph_events(graph, {"x": 0}, config, stream_mode=stream_mode, subgraphs=subgraphs)
    ]
    state = await graph.aget_state(config)

    assert executed == expected_executed
    assert state.values == {"x": expected_x}
    assert state.next == expected_next
    assert graph.interrupt_before_nodes == compiled_before
    assert graph.interrupt_after_nodes == compiled_after
    assert all(graph.channels[key] is channel for key, channel in compiled_channels.items())
    assert config[interrupt_key] == nodes
    if "events" in stream_mode:
        raw_events = [payload for mode, payload in events if mode == "events"]
        assert any(event["event"] == "on_chain_start" for event in raw_events)
        assert any(event["event"] == "on_chain_stream" and event["run_id"] == run_id for event in raw_events)
        assert any(event["event"] == "on_chain_end" and event["run_id"] == run_id for event in raw_events)

    resume_run_id = str(uuid4())
    resume_config = {
        "run_id": resume_run_id,
        "configurable": {**config["configurable"], "run_id": resume_run_id},
    }
    async for _event in stream_graph_events(graph, None, resume_config, stream_mode=stream_mode):
        pass
    resumed_state = await graph.aget_state(resume_config)

    assert executed == ["a", "b"]
    assert resumed_state.values == {"x": 2}
    assert resumed_state.next == ()


@pytest.mark.parametrize("interrupt_override", [None, []])
async def test_events_mode_retains_compiled_breakpoints(interrupt_override: list[str] | None) -> None:
    executed: list[str] = []
    graph = _breakpoint_graph(executed, interrupt_before=["b"])
    run_id = str(uuid4())
    config: dict[str, Any] = {
        "run_id": run_id,
        "configurable": {"thread_id": str(uuid4()), "run_id": run_id},
    }
    if interrupt_override is not None:
        config["interrupt_before"] = interrupt_override

    async for _event in stream_graph_events(graph, {"x": 0}, config, stream_mode=["events"]):
        pass

    assert executed == ["a"]
    assert (await graph.aget_state(config)).next == ("b",)
    assert graph.interrupt_before_nodes == ["b"]


async def test_concurrent_events_runs_isolate_breakpoints_on_shared_graph() -> None:
    executed: list[str] = []
    graph = _breakpoint_graph(executed)
    graph.config = {"metadata": {"owner": "fixture"}, "tags": ["shared"]}
    compiled_config = deepcopy(graph.config)
    compiled_channels = dict(graph.channels)

    async def stream(config: dict[str, Any]) -> None:
        async for _event in stream_graph_events(graph, {"x": 0}, config, stream_mode=["events"]):
            pass

    paused_id = str(uuid4())
    completed_id = str(uuid4())
    paused_config = {
        "run_id": paused_id,
        "configurable": {"thread_id": str(uuid4()), "run_id": paused_id},
        "interrupt_before": ["b"],
    }
    completed_config = {
        "run_id": completed_id,
        "configurable": {"thread_id": str(uuid4()), "run_id": completed_id},
    }

    await asyncio.gather(stream(paused_config), stream(completed_config))
    paused_state = await graph.aget_state(paused_config)
    completed_state = await graph.aget_state(completed_config)

    assert paused_state.values == {"x": 1}
    assert paused_state.next == ("b",)
    assert completed_state.values == {"x": 2}
    assert completed_state.next == ()
    assert graph.interrupt_before_nodes == []
    assert graph.config == compiled_config
    assert all(graph.channels[key] is channel for key, channel in compiled_channels.items())


@pytest.mark.parametrize("stream_mode", [["events", "updates"], ["events", "values", "updates"]])
@pytest.mark.parametrize(("interrupt_key", "node"), [("interrupt_before", "b"), ("interrupt_after", "a")])
async def test_events_run_finalizes_interrupted_with_pending_node(
    monkeypatch: pytest.MonkeyPatch, stream_mode: list[str], interrupt_key: str, node: str
) -> None:
    executed: list[str] = []
    graph = _breakpoint_graph(executed)
    run_config: dict[str, Any] = {}

    @asynccontextmanager
    async def get_graph(_graph_id: str, *, config: dict[str, Any], **_kwargs: Any) -> AsyncIterator[CompiledStateGraph]:
        run_config.update(config)
        yield graph

    finalize = AsyncMock(return_value=True)
    signal_end = AsyncMock()
    put_to_broker = AsyncMock()
    monkeypatch.setattr(run_executor, "get_langgraph_service", lambda: MagicMock(get_graph=get_graph))
    monkeypatch.setattr(run_executor, "start_run", AsyncMock(return_value=True))
    monkeypatch.setattr(run_executor, "finalize_run", finalize)
    monkeypatch.setattr(run_executor.broker_manager, "allocate_event_id", AsyncMock(return_value="event-1"))
    monkeypatch.setattr(run_executor.streaming_service, "put_to_broker", put_to_broker)
    monkeypatch.setattr(run_executor.streaming_service, "cleanup_run", AsyncMock())
    monkeypatch.setattr(run_executor, "_signal_end_event", signal_end)
    monkeypatch.setattr(run_executor, "_signal_run_done", AsyncMock())
    job = RunJob(
        identity=RunIdentity(run_id=str(uuid4()), thread_id=str(uuid4()), graph_id="breakpoints"),
        user=User(identity="test-user"),
        execution=RunExecution(input_data={"x": 0}, stream_mode=stream_mode),
        behavior=RunBehavior(**{interrupt_key: [node]}),
    )

    await run_executor.execute_run(job)

    assert executed == ["a"]
    assert (await graph.aget_state(run_config)).next == ("b",)
    finalize.assert_awaited_once()
    assert finalize.await_args.kwargs["status"] == "interrupted"
    assert finalize.await_args.kwargs["thread_status"] == "interrupted"
    if "values" in stream_mode:
        assert finalize.await_args.kwargs["output"] == {"x": 1}
    signal_end.assert_awaited_once_with(job.identity.run_id, "interrupted")
    assert any(call.args[2][0] == "events" for call in put_to_broker.await_args_list)


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


async def test_js_graph_keeps_remote_events_interface(monkeypatch: pytest.MonkeyPatch) -> None:
    class RemoteGraph(_RecordingGraph):
        pass

    js_base = ModuleType("langgraph_api.js.base")
    js_base.BaseRemotePregel = RemoteGraph
    monkeypatch.setitem(sys.modules, "langgraph_api", ModuleType("langgraph_api"))
    monkeypatch.setitem(sys.modules, "langgraph_api.js", ModuleType("langgraph_api.js"))
    monkeypatch.setitem(sys.modules, "langgraph_api.js.base", js_base)
    graph = RemoteGraph()
    config = {
        "configurable": {"run_id": "run-123"},
        "interrupt_before": ["*"],
        "interrupt_after": ["b"],
    }

    events = [
        event async for event in stream_graph_events(graph, {"x": 0}, config, stream_mode=["values", "messages-tuple"])
    ]

    assert graph.astream_kwargs is None
    assert graph.astream_events_kwargs is not None
    assert graph.astream_events_kwargs["interrupt_before"] == "*"
    assert graph.astream_events_kwargs["interrupt_after"] == ["b"]
    assert "messages-tuple" in graph.astream_events_kwargs["stream_mode"]
    assert ("values", {"done": True}) in events


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

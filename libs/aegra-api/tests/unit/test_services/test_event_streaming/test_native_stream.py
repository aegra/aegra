"""Tests for the native v3 stream producer (forwarding + message-tuple unwrap)."""

from __future__ import annotations

from typing import Any

import pytest

from aegra_api.services.event_streaming.native_stream import (
    stream_native_v3_events,
    unwrap_message_event,
)


class _FakeRunStream:
    """Mimics langgraph's AsyncGraphRunStream: async-iterable + async context."""

    def __init__(self, events: list[Any]) -> None:
        self._events = events

    async def __aenter__(self) -> _FakeRunStream:
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        return None

    def __aiter__(self) -> _FakeRunStream:
        self._it = iter(self._events)
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration from None


class _FakeGraph:
    def __init__(self, events: list[Any]) -> None:
        self._events = events
        self.calls: list[dict[str, Any]] = []

    async def astream_events(self, input_data: Any, config: Any, **kwargs: Any) -> _FakeRunStream:
        self.calls.append({"input": input_data, "config": config, **kwargs})
        return _FakeRunStream(self._events)


def _event(method: str, data: Any, *, namespace: list[str] | None = None) -> dict[str, Any]:
    return {"type": "event", "method": method, "params": {"namespace": namespace or [], "data": data}}


class TestUnwrapMessageEvent:
    def test_unwraps_event_meta_tuple(self) -> None:
        """v3 message data arrives as [event_dict, metadata]; element 0 is the event."""
        data = [{"event": "content-block-delta", "index": 0, "delta": {"type": "text-delta", "text": "hi"}}, {"m": 1}]
        assert unwrap_message_event(data) == {
            "event": "content-block-delta",
            "index": 0,
            "delta": {"type": "text-delta", "text": "hi"},
        }

    def test_bare_event_dict_passes_through(self) -> None:
        data = {"event": "message-start", "id": "m1"}
        assert unwrap_message_event(data) == data

    def test_non_event_tuple_returns_none(self) -> None:
        assert unwrap_message_event([{"no_event_key": 1}, {}]) is None

    def test_non_message_shape_returns_none(self) -> None:
        assert unwrap_message_event("nonsense") is None


@pytest.mark.asyncio
class TestStreamNativeV3Events:
    async def test_drives_v3_and_forwards_method_event_pairs(self) -> None:
        events = [
            _event("values", {"messages": []}),
            _event("messages", [{"event": "message-start", "id": "m1"}, {}]),
        ]
        graph = _FakeGraph(events)

        out: list[tuple[str, Any]] = []
        async for method, payload in stream_native_v3_events(graph=graph, input_data={"x": 1}, config={"c": 2}):
            out.append((method, payload))

        # v3 is requested with version="v3"
        assert graph.calls[0]["version"] == "v3"
        # values forwarded as-is
        assert out[0][0] == "values"
        assert out[0][1]["params"]["data"] == {"messages": []}
        # messages: tuple unwrapped so params.data is the event dict
        assert out[1][0] == "messages"
        assert out[1][1]["params"]["data"] == {"event": "message-start", "id": "m1"}

    async def test_skips_non_event_dicts(self) -> None:
        graph = _FakeGraph([{"not": "an event"}, _event("values", {"a": 1})])
        out = [pair async for pair in stream_native_v3_events(graph=graph, input_data={}, config={})]
        assert len(out) == 1
        assert out[0][0] == "values"

    async def test_drops_unreconstructable_message_events(self) -> None:
        """A messages event whose data isn't an event tuple/dict is dropped, not forwarded raw."""
        graph = _FakeGraph([_event("messages", [{"no_event": 1}, {}])])
        out = [pair async for pair in stream_native_v3_events(graph=graph, input_data={}, config={})]
        assert out == []

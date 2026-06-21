"""Tests for ThreadEventSession: seq, channel filter, since-replay, lifecycle."""

from collections.abc import Iterator

import pytest
from langchain_core.messages import AIMessageChunk

from aegra_api.services.broker import BrokerManager
from aegra_api.services.event_streaming import session as session_module
from aegra_api.services.event_streaming.session import (
    ThreadEventSession,
    validate_channels,
)


@pytest.fixture
def manager(monkeypatch: pytest.MonkeyPatch) -> Iterator[BrokerManager]:
    """Swap in a fresh in-memory broker manager for the session under test."""
    mgr = BrokerManager()
    monkeypatch.setattr(session_module, "broker_manager", mgr)
    yield mgr


async def _seed(mgr: BrokerManager, run_id: str, events: list[tuple[str, object]]) -> None:
    broker = mgr.get_or_create_broker(run_id)
    for i, raw in enumerate(events, start=1):
        await broker.put(f"{run_id}_event_{i}", raw)


def _chunk(text: str, *, msg_id: str = "m1", last: bool = False) -> AIMessageChunk:
    chunk = AIMessageChunk(content=text, id=msg_id)
    if last:
        chunk.chunk_position = "last"
    return chunk


async def _collect(session: ThreadEventSession) -> list[dict]:
    return [evt async for evt in session.stream()]


class TestSessionStreaming:
    async def test_message_stream_projects_protocol_events(self, manager: BrokerManager) -> None:
        await _seed(
            manager,
            "run-1",
            [
                ("messages", (_chunk("hello"), {})),
                ("messages", (_chunk(" world", last=True), {})),
                ("end", {"status": "success"}),
            ],
        )
        session = ThreadEventSession("run-1", channels={"messages", "lifecycle"})
        events = await _collect(session)

        methods = [(e["method"], e["params"].get("event")) for e in events]
        assert methods == [
            ("messages", "message-start"),
            ("messages", "content-block-delta"),
            ("messages", "content-block-delta"),
            ("messages", "message-finish"),
            ("lifecycle", "completed"),
        ]

    async def test_seq_is_monotonic_from_one(self, manager: BrokerManager) -> None:
        await _seed(manager, "run-1", [("values", {"a": 1}), ("end", {"status": "success"})])
        events = await _collect(ThreadEventSession("run-1", channels={"values", "lifecycle"}))
        assert [e["seq"] for e in events] == [1, 2]

    async def test_channel_filter_drops_unsubscribed(self, manager: BrokerManager) -> None:
        await _seed(
            manager,
            "run-1",
            [("values", {"a": 1}), ("updates", {"n": {"b": 2}}), ("end", {"status": "success"})],
        )
        events = await _collect(ThreadEventSession("run-1", channels={"values"}))
        assert {e["method"] for e in events} == {"values"}

    async def test_since_skips_already_seen(self, manager: BrokerManager) -> None:
        await _seed(
            manager,
            "run-1",
            [("values", {"a": 1}), ("values", {"a": 2}), ("end", {"status": "success"})],
        )
        # since=1 means the client already saw seq 1; expect seq 2 and 3 only.
        events = await _collect(ThreadEventSession("run-1", channels={"values", "lifecycle"}, since=1))
        assert [e["seq"] for e in events] == [2, 3]

    async def test_seq_is_absolute_not_filter_relative(self, manager: BrokerManager) -> None:
        """seq counts every translated event, so a filtered channel still advances it."""
        await _seed(
            manager,
            "run-1",
            [("values", {"a": 1}), ("updates", {"n": {"b": 2}}), ("end", {"status": "success"})],
        )
        # Subscribe only to lifecycle: values(seq1) + updates(seq2) are dropped,
        # lifecycle lands at the absolute seq 3.
        events = await _collect(ThreadEventSession("run-1", channels={"lifecycle"}))
        assert [(e["method"], e["seq"]) for e in events] == [("lifecycle", 3)]

    async def test_reconnect_with_narrower_channels_keeps_terminal_event(self, manager: BrokerManager) -> None:
        """A resume on a narrower channel set still delivers later events.

        Absolute seq means the lifecycle event keeps its run-stream position, so
        a since cursor from the wider first session does not skip it.
        """
        await _seed(
            manager,
            "run-1",
            [("messages", (_chunk("hi", last=True), {})), ("end", {"status": "success"})],
        )
        # First session saw through seq 3 (start, delta, finish). Reconnect for
        # lifecycle only with since=3; the terminal event is seq 4, still delivered.
        events = await _collect(ThreadEventSession("run-1", channels={"lifecycle"}, since=3))
        assert [(e["method"], e["seq"]) for e in events] == [("lifecycle", 4)]

    async def test_lifecycle_interrupted(self, manager: BrokerManager) -> None:
        await _seed(manager, "run-1", [("end", {"status": "interrupted"})])
        events = await _collect(ThreadEventSession("run-1", channels={"lifecycle"}))
        assert events[0]["params"] == {"event": "interrupted"}

    async def test_lifecycle_failed_carries_error(self, manager: BrokerManager) -> None:
        await _seed(manager, "run-1", [("error", {"status": "error", "message": "boom"})])
        events = await _collect(ThreadEventSession("run-1", channels={"lifecycle"}))
        assert events[0]["params"] == {"event": "failed", "error": "boom"}

    async def test_wire_event_id_is_unique_per_event(self, manager: BrokerManager) -> None:
        await _seed(manager, "run-1", [("messages", (_chunk("hi"), {})), ("end", {"status": "success"})])
        events = await _collect(ThreadEventSession("run-1", channels={"messages", "lifecycle"}))
        ids = [e["event_id"] for e in events]
        assert len(ids) == len(set(ids))

    async def test_namespaced_custom_subscription_receives_custom_events(self, manager: BrokerManager) -> None:
        """Subscribing to custom:<name> still receives the base custom events."""
        await _seed(manager, "run-1", [("custom", {"hello": "world"}), ("end", {"status": "success"})])
        events = await _collect(ThreadEventSession("run-1", channels={"custom:my_event", "lifecycle"}))
        assert any(e["method"] == "custom" for e in events)

    async def test_plain_custom_subscription_receives_custom_events(self, manager: BrokerManager) -> None:
        await _seed(manager, "run-1", [("custom", {"hello": "world"}), ("end", {"status": "success"})])
        events = await _collect(ThreadEventSession("run-1", channels={"custom", "lifecycle"}))
        assert any(e["method"] == "custom" for e in events)


class TestValidateChannels:
    def test_valid_channels(self) -> None:
        valid, invalid = validate_channels(["messages", "values", "custom:foo"])
        assert valid == {"messages", "values", "custom:foo"}
        assert invalid == []

    def test_invalid_channels_collected(self) -> None:
        valid, invalid = validate_channels(["messages", "bogus"])
        assert valid == {"messages"}
        assert invalid == ["bogus"]

    def test_empty_list_is_error(self) -> None:
        valid, invalid = validate_channels([])
        assert valid == set()
        assert invalid

    def test_non_list_is_error(self) -> None:
        valid, invalid = validate_channels("messages")
        assert invalid

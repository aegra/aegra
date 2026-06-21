"""Tests for ThreadEventSession: native-event forwarding, seq, filter, since, HITL, lifecycle."""

from collections.abc import Iterator
from typing import Any

import pytest

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


def _lister(*run_ids: str):
    async def list_run_ids() -> list[str]:
        return list(run_ids)

    return list_run_ids


def _protocol_event(method: str, data: Any, *, namespace: list[str] | None = None, **params: Any) -> dict[str, Any]:
    """A native v3 ProtocolEvent as it sits in the broker for a v2 run."""
    return {
        "type": "event",
        "method": method,
        "params": {"namespace": namespace or [], "data": data, **params},
    }


async def _seed(manager: BrokerManager, run_id: str, events: list[tuple[str, Any]]) -> None:
    broker = manager.get_or_create_broker(run_id)
    for i, raw in enumerate(events, start=1):
        await broker.put(f"{run_id}_event_{i}", raw)


def _msg_event(event_kind: str, **extra: Any) -> tuple[str, dict[str, Any]]:
    return ("messages", _protocol_event("messages", {"event": event_kind, **extra}))


def _make_session(
    thread_id: str, *, channels: set[str], run_ids: tuple[str, ...], since: int | None = None
) -> ThreadEventSession:
    return ThreadEventSession(
        thread_id,
        channels=channels,
        list_run_ids=_lister(*run_ids),
        since=since,
        idle_grace_seconds=0.0,
    )


async def _collect(session: ThreadEventSession) -> list[dict[str, Any]]:
    return [evt async for evt in session.stream()]


class TestForwarding:
    async def test_message_lifecycle_forwarded_verbatim(self, manager: BrokerManager) -> None:
        await _seed(
            manager,
            "run-1",
            [
                _msg_event("message-start", role="ai", id="m1"),
                _msg_event("content-block-delta", index=0, delta={"type": "text-delta", "text": "hi"}),
                _msg_event("message-finish", usage={"total_tokens": 5}),
                ("end", {"status": "success"}),
            ],
        )
        events = await _collect(_make_session("t1", channels={"messages", "lifecycle"}, run_ids=("run-1",)))
        kinds = [(e["method"], e["params"]["data"].get("event")) for e in events]
        assert kinds == [
            ("messages", "message-start"),
            ("messages", "content-block-delta"),
            ("messages", "message-finish"),
            ("lifecycle", "completed"),
        ]

    async def test_values_payload_in_params_data(self, manager: BrokerManager) -> None:
        await _seed(manager, "run-1", [("values", _protocol_event("values", {"a": 1})), ("end", {"status": "success"})])
        events = await _collect(_make_session("t1", channels={"values"}, run_ids=("run-1",)))
        assert events[0]["params"]["data"] == {"a": 1}
        assert events[0]["params"]["namespace"] == []

    async def test_namespace_preserved(self, manager: BrokerManager) -> None:
        await _seed(
            manager,
            "run-1",
            [("values", _protocol_event("values", {"a": 1}, namespace=["sub:1"])), ("end", {"status": "success"})],
        )
        events = await _collect(_make_session("t1", channels={"values"}, run_ids=("run-1",)))
        assert events[0]["params"]["namespace"] == ["sub:1"]

    async def test_tools_channel_forwarded(self, manager: BrokerManager) -> None:
        await _seed(
            manager,
            "run-1",
            [
                ("tools", _protocol_event("tools", {"event": "tool-started", "tool_call_id": "c1"})),
                ("end", {"status": "success"}),
            ],
        )
        events = await _collect(_make_session("t1", channels={"tools"}, run_ids=("run-1",)))
        assert events[0]["params"]["data"] == {"event": "tool-started", "tool_call_id": "c1"}


class TestSeqAndFilter:
    async def test_seq_monotonic_from_one(self, manager: BrokerManager) -> None:
        await _seed(manager, "run-1", [("values", _protocol_event("values", {"a": 1})), ("end", {"status": "success"})])
        events = await _collect(_make_session("t1", channels={"values", "lifecycle"}, run_ids=("run-1",)))
        assert [e["seq"] for e in events] == [1, 2]

    async def test_seq_spans_multiple_runs(self, manager: BrokerManager) -> None:
        await _seed(manager, "run-1", [("values", _protocol_event("values", {"a": 1})), ("end", {"status": "success"})])
        await _seed(manager, "run-2", [("values", _protocol_event("values", {"a": 2})), ("end", {"status": "success"})])
        events = await _collect(_make_session("t1", channels={"values", "lifecycle"}, run_ids=("run-1", "run-2")))
        assert [e["seq"] for e in events] == [1, 2, 3, 4]

    async def test_channel_filter_drops_unsubscribed(self, manager: BrokerManager) -> None:
        await _seed(
            manager,
            "run-1",
            [
                ("values", _protocol_event("values", {"a": 1})),
                ("updates", _protocol_event("updates", {"node": "n", "values": {"b": 2}})),
                ("end", {"status": "success"}),
            ],
        )
        events = await _collect(_make_session("t1", channels={"values"}, run_ids=("run-1",)))
        assert {e["method"] for e in events} == {"values"}

    async def test_seq_absolute_not_filter_relative(self, manager: BrokerManager) -> None:
        await _seed(
            manager,
            "run-1",
            [
                ("values", _protocol_event("values", {"a": 1})),
                ("updates", _protocol_event("updates", {"node": "n", "values": {"b": 2}})),
                ("end", {"status": "success"}),
            ],
        )
        events = await _collect(_make_session("t1", channels={"lifecycle"}, run_ids=("run-1",)))
        assert [(e["method"], e["seq"]) for e in events] == [("lifecycle", 3)]

    async def test_since_skips_already_seen(self, manager: BrokerManager) -> None:
        await _seed(
            manager,
            "run-1",
            [
                ("values", _protocol_event("values", {"a": 1})),
                ("values", _protocol_event("values", {"a": 2})),
                ("end", {"status": "success"}),
            ],
        )
        events = await _collect(_make_session("t1", channels={"values", "lifecycle"}, run_ids=("run-1",), since=1))
        assert [e["seq"] for e in events] == [2, 3]

    async def test_applied_through_seq_tracks_max(self, manager: BrokerManager) -> None:
        await _seed(manager, "run-1", [("values", _protocol_event("values", {"a": 1})), ("end", {"status": "success"})])
        session = _make_session("t1", channels={"values", "lifecycle"}, run_ids=("run-1",))
        await _collect(session)
        assert session.applied_through_seq == 2


class TestInterruptsToInputChannel:
    async def test_interrupt_on_values_becomes_input_requested(self, manager: BrokerManager) -> None:
        """An interrupt riding on a values event's params.interrupts surfaces on input."""
        interrupt_payload = {"question": "Approve?"}
        await _seed(
            manager,
            "run-1",
            [
                (
                    "values",
                    _protocol_event(
                        "values",
                        {"messages": []},
                        interrupts=[{"id": "int-1", "value": interrupt_payload}],
                    ),
                ),
                ("end", {"status": "interrupted"}),
            ],
        )
        events = await _collect(_make_session("t1", channels={"input", "values"}, run_ids=("run-1",)))
        input_events = [e for e in events if e["method"] == "input.requested"]
        assert input_events
        assert input_events[0]["params"]["data"] == {"interrupt_id": "int-1", "payload": interrupt_payload}

    async def test_interrupt_stripped_from_values_payload(self, manager: BrokerManager) -> None:
        """__interrupt__ never leaks into the forwarded values data."""
        await _seed(
            manager,
            "run-1",
            [
                ("values", _protocol_event("values", {"messages": [], "__interrupt__": [{"id": "int-1", "value": 1}]})),
                ("end", {"status": "interrupted"}),
            ],
        )
        events = await _collect(_make_session("t1", channels={"values"}, run_ids=("run-1",)))
        values_events = [e for e in events if e["method"] == "values"]
        assert "__interrupt__" not in values_events[0]["params"]["data"]


class TestLifecycle:
    async def test_completed(self, manager: BrokerManager) -> None:
        await _seed(manager, "run-1", [("end", {"status": "success"})])
        events = await _collect(_make_session("t1", channels={"lifecycle"}, run_ids=("run-1",)))
        assert events[0]["params"]["data"] == {"event": "completed"}

    async def test_interrupted(self, manager: BrokerManager) -> None:
        await _seed(manager, "run-1", [("end", {"status": "interrupted"})])
        events = await _collect(_make_session("t1", channels={"lifecycle"}, run_ids=("run-1",)))
        assert events[0]["params"]["data"] == {"event": "interrupted"}

    async def test_failed_carries_error(self, manager: BrokerManager) -> None:
        await _seed(manager, "run-1", [("error", {"status": "error", "message": "boom"})])
        events = await _collect(_make_session("t1", channels={"lifecycle"}, run_ids=("run-1",)))
        assert events[0]["params"]["data"] == {"event": "failed", "error": "boom"}


class TestMisc:
    async def test_namespaced_custom_subscription_receives_custom(self, manager: BrokerManager) -> None:
        await _seed(
            manager,
            "run-1",
            [("custom", _protocol_event("custom", {"payload": {"hello": "world"}})), ("end", {"status": "success"})],
        )
        events = await _collect(_make_session("t1", channels={"custom:my_event", "lifecycle"}, run_ids=("run-1",)))
        assert any(e["method"] == "custom" for e in events)

    async def test_empty_thread_closes_after_idle(self, manager: BrokerManager) -> None:
        events = await _collect(_make_session("t1", channels={"lifecycle"}, run_ids=()))
        assert events == []


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

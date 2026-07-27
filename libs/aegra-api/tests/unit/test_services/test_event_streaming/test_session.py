"""Tests for ThreadEventSession: native-event forwarding, seq, filter, since, HITL, lifecycle."""

import asyncio
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


def _lister(*run_ids: str, statuses: dict[str, str] | None = None, graph: str | None = None):
    """Run lister returning (run_id, status, graph_name) rows; status defaults to running."""

    async def list_run_ids() -> list[tuple[str, str | None, str | None]]:
        return [(run_id, (statuses or {}).get(run_id, "running"), graph) for run_id in run_ids]

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
    thread_id: str,
    *,
    channels: set[str],
    run_ids: tuple[str, ...],
    since: int | None = None,
    namespaces: list[list[str]] | None = None,
    depth: int | None = None,
    statuses: dict[str, str] | None = None,
    graph: str | None = None,
) -> ThreadEventSession:
    return ThreadEventSession(
        thread_id,
        channels=channels,
        list_run_ids=_lister(*run_ids, statuses=statuses, graph=graph),
        since=since,
        namespaces=namespaces,
        depth=depth,
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
            ("lifecycle", "running"),
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

    async def test_raw_v3_updates_normalized_to_node_values(self, manager: BrokerManager) -> None:
        """v3 emits updates as raw {node: values}; the wire form is {node, values}."""
        await _seed(
            manager,
            "run-1",
            [("updates", _protocol_event("updates", {"worker": {"count": 1}})), ("end", {"status": "success"})],
        )
        events = await _collect(_make_session("t1", channels={"updates"}, run_ids=("run-1",)))
        assert events[0]["params"]["data"] == {"node": "worker", "values": {"count": 1}}

    async def test_interrupt_node_in_updates_routes_to_input(self, manager: BrokerManager) -> None:
        """An __interrupt__ update surfaces on the input channel, not as an updates event."""
        await _seed(
            manager,
            "run-1",
            [
                ("updates", _protocol_event("updates", {"__interrupt__": [{"id": "int-9", "value": {"q": "ok?"}}]})),
                ("end", {"status": "interrupted"}),
            ],
        )
        events = await _collect(_make_session("t1", channels={"input", "updates"}, run_ids=("run-1",)))
        input_events = [e for e in events if e["method"] == "input.requested"]
        assert input_events[0]["params"]["data"] == {"interrupt_id": "int-9", "value": {"q": "ok?"}}
        assert not [e for e in events if e["method"] == "updates"]


class TestSeqAndFilter:
    async def test_seq_monotonic_from_one(self, manager: BrokerManager) -> None:
        await _seed(manager, "run-1", [("values", _protocol_event("values", {"a": 1})), ("end", {"status": "success"})])
        events = await _collect(_make_session("t1", channels={"values", "lifecycle"}, run_ids=("run-1",)))
        # seq 1 = the run's root lifecycle running seed.
        assert [e["seq"] for e in events] == [1, 2, 3]

    async def test_seq_spans_multiple_runs(self, manager: BrokerManager) -> None:
        await _seed(manager, "run-1", [("values", _protocol_event("values", {"a": 1})), ("end", {"status": "success"})])
        await _seed(manager, "run-2", [("values", _protocol_event("values", {"a": 2})), ("end", {"status": "success"})])
        events = await _collect(_make_session("t1", channels={"values", "lifecycle"}, run_ids=("run-1", "run-2")))
        assert [e["seq"] for e in events] == [1, 2, 3, 4, 5, 6]

    async def test_channel_filter_drops_unsubscribed(self, manager: BrokerManager) -> None:
        await _seed(
            manager,
            "run-1",
            [
                ("values", _protocol_event("values", {"a": 1})),
                ("updates", _protocol_event("updates", {"n": {"b": 2}})),
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
                ("updates", _protocol_event("updates", {"n": {"b": 2}})),
                ("end", {"status": "success"}),
            ],
        )
        events = await _collect(_make_session("t1", channels={"lifecycle"}, run_ids=("run-1",)))
        # running seed = seq 1; values/updates burn 2-3 unsubscribed; terminal = 4.
        assert [(e["method"], e["seq"]) for e in events] == [("lifecycle", 1), ("lifecycle", 4)]

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
        assert [e["seq"] for e in events] == [2, 3, 4]

    async def test_applied_through_seq_tracks_max(self, manager: BrokerManager) -> None:
        await _seed(manager, "run-1", [("values", _protocol_event("values", {"a": 1})), ("end", {"status": "success"})])
        session = _make_session("t1", channels={"values", "lifecycle"}, run_ids=("run-1",))
        await _collect(session)
        assert session.applied_through_seq == 3


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
        assert input_events[0]["params"]["data"] == {"interrupt_id": "int-1", "value": interrupt_payload}

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
    async def test_running_seed_opens_each_run(self, manager: BrokerManager) -> None:
        await _seed(manager, "run-1", [("end", {"status": "success"})])
        events = await _collect(_make_session("t1", channels={"lifecycle"}, run_ids=("run-1",)))
        assert events[0]["params"]["data"] == {"event": "running"}
        assert events[0]["params"]["namespace"] == []

    async def test_root_lifecycle_carries_graph_name_when_known(self, manager: BrokerManager) -> None:
        await _seed(manager, "run-1", [("end", {"status": "success"})])
        events = await _collect(_make_session("t1", channels={"lifecycle"}, run_ids=("run-1",), graph="my_graph"))
        assert events[0]["params"]["data"] == {"event": "running", "graph_name": "my_graph"}
        assert events[-1]["params"]["data"] == {"event": "completed", "graph_name": "my_graph"}

    async def test_completed(self, manager: BrokerManager) -> None:
        await _seed(manager, "run-1", [("end", {"status": "success"})])
        events = await _collect(_make_session("t1", channels={"lifecycle"}, run_ids=("run-1",)))
        assert events[-1]["params"]["data"] == {"event": "completed"}

    async def test_interrupted(self, manager: BrokerManager) -> None:
        await _seed(manager, "run-1", [("end", {"status": "interrupted"})])
        events = await _collect(_make_session("t1", channels={"lifecycle"}, run_ids=("run-1",)))
        assert events[-1]["params"]["data"] == {"event": "interrupted"}

    async def test_failed_carries_error(self, manager: BrokerManager) -> None:
        # The real broker error event is {error, message} with NO status; the
        # failed status must come from the method, not a status key.
        await _seed(manager, "run-1", [("error", {"error": "RuntimeError", "message": "boom"})])
        events = await _collect(_make_session("t1", channels={"lifecycle"}, run_ids=("run-1",)))
        assert events[-1]["params"]["data"] == {"event": "failed", "error": "boom"}


class TestMisc:
    async def test_custom_payload_wrapped_in_wire_shape(self, manager: BrokerManager) -> None:
        await _seed(
            manager,
            "run-1",
            [("custom", _protocol_event("custom", {"hello": "world"})), ("end", {"status": "success"})],
        )
        events = await _collect(_make_session("t1", channels={"custom"}, run_ids=("run-1",)))
        custom = [e for e in events if e["method"] == "custom"]
        assert custom and custom[0]["params"]["data"] == {"payload": {"hello": "world"}}

    async def test_named_custom_source_becomes_custom_with_name(self, manager: BrokerManager) -> None:
        await _seed(
            manager,
            "run-1",
            [("custom:my_event", _protocol_event("custom:my_event", {"x": 1})), ("end", {"status": "success"})],
        )
        events = await _collect(_make_session("t1", channels={"custom:my_event"}, run_ids=("run-1",)))
        custom = [e for e in events if e["method"] == "custom"]
        assert custom and custom[0]["params"]["data"] == {"name": "my_event", "payload": {"x": 1}}

    async def test_named_subscription_filters_other_custom_events(self, manager: BrokerManager) -> None:
        """custom:foo subscribers get only name==foo; unnamed events need a plain custom subscription."""
        await _seed(
            manager,
            "run-1",
            [
                ("custom:other", _protocol_event("custom:other", {"x": 1})),
                ("custom", _protocol_event("custom", {"y": 2})),
                ("end", {"status": "success"}),
            ],
        )
        events = await _collect(_make_session("t1", channels={"custom:my_event"}, run_ids=("run-1",)))
        assert not [e for e in events if e["method"] == "custom"]

    async def test_plain_custom_subscription_receives_named_events(self, manager: BrokerManager) -> None:
        await _seed(
            manager,
            "run-1",
            [("custom:my_event", _protocol_event("custom:my_event", {"x": 1})), ("end", {"status": "success"})],
        )
        events = await _collect(_make_session("t1", channels={"custom"}, run_ids=("run-1",)))
        assert [e for e in events if e["method"] == "custom"]

    async def test_empty_thread_closes_after_idle(self, manager: BrokerManager) -> None:
        events = await _collect(_make_session("t1", channels={"lifecycle"}, run_ids=()))
        assert events == []


class TestResumeAcrossRuns:
    """A HITL resume starts a fresh run on the thread; its events must reach the
    same open stream, not be dropped when the interrupted run drains first."""

    async def test_followup_run_after_interrupt_streams_on_same_session(self, manager: BrokerManager) -> None:
        await _seed(
            manager,
            "run-a",
            [
                (
                    "values",
                    _protocol_event("values", {"messages": []}, interrupts=[{"id": "int-1", "value": {"q": "ok?"}}]),
                ),
                ("end", {"status": "interrupted"}),
            ],
        )

        revealed = False

        async def lister() -> list[tuple[str, str | None, str | None]]:
            nonlocal revealed
            if not revealed:
                revealed = True
                return [("run-a", "interrupted", None)]
            return [("run-a", "interrupted", None), ("run-b", "running", None)]

        async def reveal_run_b() -> None:
            # run-b appears a beat after run-a drains, as the resume round-trip lands.
            await asyncio.sleep(0.05)
            await _seed(
                manager,
                "run-b",
                [("values", _protocol_event("values", {"messages": [{"type": "ai", "content": "done"}]}))],
            )
            await manager.get_or_create_broker("run-b").put("run-b_end", ("end", {"status": "success"}))

        session = ThreadEventSession(
            "t1",
            channels={"input", "values", "lifecycle"},
            list_run_ids=lister,
            idle_grace_seconds=5.0,
        )
        seeder = asyncio.create_task(reveal_run_b())
        events = [e async for e in session.stream()]
        await asyncio.wait_for(seeder, timeout=1.0)

        methods = [(e["method"], e["params"]["data"]) for e in events]
        assert ("input.requested", {"interrupt_id": "int-1", "value": {"q": "ok?"}}) in methods
        # run-b's completion proves the stream stayed open across the run gap.
        assert any(e["method"] == "lifecycle" and e["params"]["data"] == {"event": "completed"} for e in events)

    async def test_terminal_run_still_closes_after_grace(self, manager: BrokerManager) -> None:
        """A completed run with no follow-up closes after one grace window."""
        await _seed(manager, "run-1", [("values", _protocol_event("values", {"a": 1})), ("end", {"status": "success"})])
        session = _make_session("t1", channels={"values", "lifecycle"}, run_ids=("run-1",))
        events = await _collect(session)
        assert [e["method"] for e in events] == ["lifecycle", "values", "lifecycle"]


class TestNamespaceFilter:
    """namespaces (prefix include-list) and depth (nesting cap) filter subgraph events."""

    async def test_namespaces_include_only_matching_prefix(self, manager: BrokerManager) -> None:
        await _seed(
            manager,
            "run-1",
            [
                ("values", _protocol_event("values", {"a": 1}, namespace=["sub_a"])),
                ("values", _protocol_event("values", {"b": 2}, namespace=["sub_b"])),
                ("end", {"status": "success"}),
            ],
        )
        events = await _collect(_make_session("t1", channels={"values"}, run_ids=("run-1",), namespaces=[["sub_a"]]))
        assert [e["params"]["data"] for e in events] == [{"a": 1}]

    async def test_depth_caps_nesting(self, manager: BrokerManager) -> None:
        await _seed(
            manager,
            "run-1",
            [
                ("values", _protocol_event("values", {"top": 1}, namespace=["sub_a"])),
                ("values", _protocol_event("values", {"deep": 2}, namespace=["sub_a", "sub_b"])),
                ("end", {"status": "success"}),
            ],
        )
        events = await _collect(_make_session("t1", channels={"values"}, run_ids=("run-1",), depth=1))
        assert [e["params"]["data"] for e in events] == [{"top": 1}]

    async def test_thread_level_events_always_pass_the_filter(self, manager: BrokerManager) -> None:
        """Lifecycle (empty namespace) is not subgraph-scoped; a namespace filter must not drop it."""
        await _seed(manager, "run-1", [("end", {"status": "success"})])
        events = await _collect(_make_session("t1", channels={"lifecycle"}, run_ids=("run-1",), namespaces=[["sub_a"]]))
        assert events[-1]["params"]["data"] == {"event": "completed"}

    async def test_namespace_filter_matches_dynamic_task_id_suffix(self, manager: BrokerManager) -> None:
        """Real subgraph namespaces are node:<task_id>; a clean-name prefix must match."""
        await _seed(
            manager,
            "run-1",
            [
                ("values", _protocol_event("values", {"a": 1}, namespace=["sub_a:9f3c-1"])),
                ("values", _protocol_event("values", {"b": 2}, namespace=["sub_b:7e1a-2"])),
                ("end", {"status": "success"}),
            ],
        )
        events = await _collect(_make_session("t1", channels={"values"}, run_ids=("run-1",), namespaces=[["sub_a"]]))
        assert [e["params"]["data"] for e in events] == [{"a": 1}]

    async def test_seq_stays_absolute_under_namespace_filter(self, manager: BrokerManager) -> None:
        """Filtered-out events still advance seq, so reconnect cursors stay stable."""
        await _seed(
            manager,
            "run-1",
            [
                ("values", _protocol_event("values", {"a": 1}, namespace=["sub_b"])),  # filtered
                ("values", _protocol_event("values", {"b": 2}, namespace=["sub_a"])),  # kept, seq=2
                ("end", {"status": "success"}),
            ],
        )
        events = await _collect(_make_session("t1", channels={"values"}, run_ids=("run-1",), namespaces=[["sub_a"]]))
        # seq 1 = running seed (lifecycle channel, unsubscribed), 2 = filtered sub_b.
        assert [(e["params"]["data"], e["seq"]) for e in events] == [({"b": 2}, 3)]


class TestUpdatesInterruptHandling:
    async def test_sibling_node_update_survives_interrupt_in_same_chunk(self, manager: BrokerManager) -> None:
        """A parallel branch's update arriving alongside __interrupt__ must not vanish."""
        await _seed(
            manager,
            "run-1",
            [
                (
                    "updates",
                    _protocol_event(
                        "updates",
                        {"__interrupt__": [{"id": "int-1", "value": {"q": "ok?"}}], "worker": {"count": 2}},
                    ),
                ),
                ("end", {"status": "interrupted"}),
            ],
        )
        events = await _collect(_make_session("t1", channels={"input", "updates"}, run_ids=("run-1",)))
        assert [e for e in events if e["method"] == "input.requested"]
        updates = [e for e in events if e["method"] == "updates"]
        assert updates and updates[0]["params"]["data"] == {"node": "worker", "values": {"count": 2}}

    async def test_interrupt_nested_in_node_values_surfaces_on_input(self, manager: BrokerManager) -> None:
        """An interrupt riding inside a node's values must emit input.requested, not be silently stripped."""
        await _seed(
            manager,
            "run-1",
            [
                (
                    "updates",
                    _protocol_event(
                        "updates",
                        {"gate": {"messages": [], "__interrupt__": [{"id": "int-2", "value": {"q": "sure?"}}]}},
                    ),
                ),
                ("end", {"status": "interrupted"}),
            ],
        )
        events = await _collect(_make_session("t1", channels={"input", "updates"}, run_ids=("run-1",)))
        input_events = [e for e in events if e["method"] == "input.requested"]
        assert input_events and input_events[0]["params"]["data"]["interrupt_id"] == "int-2"
        updates = [e for e in events if e["method"] == "updates"]
        assert "__interrupt__" not in updates[0]["params"]["data"]["values"]

    async def test_same_interrupt_via_updates_then_values_emits_once(self, manager: BrokerManager) -> None:
        """The interrupt often rides an updates chunk AND the following values snapshot."""
        await _seed(
            manager,
            "run-1",
            [
                ("updates", _protocol_event("updates", {"__interrupt__": [{"id": "int-3", "value": {"q": "go?"}}]})),
                (
                    "values",
                    _protocol_event("values", {"messages": []}, interrupts=[{"id": "int-3", "value": {"q": "go?"}}]),
                ),
                ("end", {"status": "interrupted"}),
            ],
        )
        events = await _collect(_make_session("t1", channels={"input", "values"}, run_ids=("run-1",)))
        input_events = [e for e in events if e["method"] == "input.requested"]
        assert len(input_events) == 1
        assert input_events[0]["params"]["data"]["interrupt_id"] == "int-3"


class TestExpiredBrokerBackstop:
    """A historical run whose broker events expired must not wedge the stream:
    the persisted run status drains it instead of tailing an empty broker forever."""

    async def test_terminal_run_with_empty_broker_drains_silently(self, manager: BrokerManager) -> None:
        # run-old: success in DB, broker events long gone (empty recreated broker).
        # run-new: live events. Without the status backstop, run-old wedges forever.
        await _seed(
            manager, "run-new", [("values", _protocol_event("values", {"a": 1})), ("end", {"status": "success"})]
        )
        events = await asyncio.wait_for(
            _collect(
                _make_session(
                    "t1",
                    channels={"values", "lifecycle"},
                    run_ids=("run-old", "run-new"),
                    statuses={"run-old": "success", "run-new": "success"},
                )
            ),
            timeout=2.0,
        )
        assert [e["method"] for e in events] == ["lifecycle", "values", "lifecycle"]

    async def test_terminal_run_with_lost_end_frame_gets_synthesized_terminal(self, manager: BrokerManager) -> None:
        # Events survived but the end frame was lost — client still needs closure.
        broker = manager.get_or_create_broker("run-old")
        await broker.put("run-old_event_1", ("values", _protocol_event("values", {"a": 1})))
        events = await asyncio.wait_for(
            _collect(
                _make_session(
                    "t1",
                    channels={"values", "lifecycle"},
                    run_ids=("run-old",),
                    statuses={"run-old": "success"},
                )
            ),
            timeout=2.0,
        )
        assert [(e["method"], e["params"]["data"].get("event")) for e in events] == [
            ("lifecycle", "running"),
            ("values", None),
            ("lifecycle", "completed"),
        ]

    async def test_error_status_synthesizes_failed_lifecycle(self, manager: BrokerManager) -> None:
        broker = manager.get_or_create_broker("run-old")
        await broker.put("run-old_event_1", ("values", _protocol_event("values", {"a": 1})))
        events = await asyncio.wait_for(
            _collect(
                _make_session(
                    "t1",
                    channels={"lifecycle"},
                    run_ids=("run-old",),
                    statuses={"run-old": "error"},
                )
            ),
            timeout=2.0,
        )
        assert events[-1]["params"]["data"]["event"] == "failed"


class TestSubgraphLifecycle:
    """Native producer emits per-subgraph lifecycle at root scope with the child
    namespace in data.namespace; the session promotes it onto the wire so nested
    agents get started/completed/failed frames on their own namespace."""

    async def test_subgraph_started_completed_promoted_to_child_namespace(self, manager: BrokerManager) -> None:
        ns = ["subgraph_agent:abc-123"]
        await _seed(
            manager,
            "run-1",
            [
                (
                    "lifecycle",
                    _protocol_event(
                        "lifecycle",
                        {
                            "event": "started",
                            "namespace": ns,
                            "graph_name": "subgraph_agent",
                            "trigger_call_id": "abc-123",
                        },
                    ),
                ),
                ("lifecycle", _protocol_event("lifecycle", {"event": "completed", "namespace": ns})),
                ("end", {"status": "success"}),
            ],
        )
        events = await _collect(_make_session("t1", channels={"lifecycle"}, run_ids=("run-1",)))
        lc = [(e["params"]["data"].get("event"), e["params"]["namespace"]) for e in events]
        assert lc == [("running", []), ("started", ns), ("completed", ns), ("completed", [])]
        started = next(e for e in events if e["params"]["data"].get("event") == "started")
        assert started["params"]["data"]["graph_name"] == "subgraph_agent"

    async def test_subgraph_failed_carries_error_on_child_namespace(self, manager: BrokerManager) -> None:
        ns = ["child_node:xyz"]
        await _seed(
            manager,
            "run-1",
            [
                (
                    "lifecycle",
                    _protocol_event(
                        "lifecycle",
                        {"event": "started", "namespace": ns, "graph_name": "child_graph", "trigger_call_id": "xyz"},
                    ),
                ),
                ("lifecycle", _protocol_event("lifecycle", {"event": "failed", "namespace": ns, "error": "boom"})),
                ("error", {"error": "RuntimeError", "message": "boom"}),
            ],
        )
        events = await _collect(_make_session("t1", channels={"lifecycle"}, run_ids=("run-1",)))
        failed = next(e for e in events if e["params"]["data"].get("event") == "failed")
        assert failed["params"]["namespace"] == ns
        assert failed["params"]["data"]["error"] == "boom"

    async def test_root_scoped_native_lifecycle_dropped(self, manager: BrokerManager) -> None:
        """A native lifecycle whose data.namespace is empty is root-scoped; the
        terminal _lifecycle owns root, so the native one must not double-emit."""
        await _seed(
            manager,
            "run-1",
            [
                ("lifecycle", _protocol_event("lifecycle", {"event": "running", "namespace": []})),
                ("end", {"status": "success"}),
            ],
        )
        events = await _collect(_make_session("t1", channels={"lifecycle"}, run_ids=("run-1",)))
        assert [e["params"]["data"] for e in events] == [{"event": "running"}, {"event": "completed"}]

    async def test_terminal_cascades_still_open_subgraph_namespaces(self, manager: BrokerManager) -> None:
        """A cancel mid-subgraph never gets the producer's completed — the terminal
        must close the open namespace so clients don't see it started forever."""
        ns = ["worker:abc"]
        await _seed(
            manager,
            "run-1",
            [
                (
                    "lifecycle",
                    _protocol_event("lifecycle", {"event": "started", "namespace": ns, "graph_name": "worker"}),
                ),
                ("end", {"status": "interrupted"}),
            ],
        )
        events = await _collect(_make_session("t1", channels={"lifecycle"}, run_ids=("run-1",)))
        lc = [(e["params"]["data"].get("event"), e["params"]["namespace"]) for e in events]
        assert lc == [("running", []), ("started", ns), ("completed", ns), ("interrupted", [])]

    async def test_subgraph_lifecycle_respects_namespace_filter(self, manager: BrokerManager) -> None:
        await _seed(
            manager,
            "run-1",
            [
                (
                    "lifecycle",
                    _protocol_event("lifecycle", {"event": "started", "namespace": ["sub_a:1"], "graph_name": "a"}),
                ),
                (
                    "lifecycle",
                    _protocol_event("lifecycle", {"event": "started", "namespace": ["sub_b:1"], "graph_name": "b"}),
                ),
                ("end", {"status": "success"}),
            ],
        )
        events = await _collect(_make_session("t1", channels={"lifecycle"}, run_ids=("run-1",), namespaces=[["sub_a"]]))
        started = [e["params"]["namespace"] for e in events if e["params"]["data"].get("event") == "started"]
        assert started == [["sub_a:1"]]


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


class TestQueuedRunHandling:
    """A parked (queued) double-text publishes no broker events and has no task:
    draining it would emit a false 'running' lifecycle and wedge on an empty broker."""

    async def test_queued_run_is_deferred_not_drained(self, manager: BrokerManager) -> None:
        session = _make_session("t1", channels={"lifecycle"}, run_ids=("q1",), statuses={"q1": "queued"})

        async with asyncio.timeout(5):  # pre-fix behavior blocks forever in broker.aiter()
            events = await _collect(session)

        assert events == []  # no false lifecycle seed, session ends via idle grace

    async def test_dropped_queued_run_drains_terminal_on_a_later_tick(self, manager: BrokerManager) -> None:
        # Tick 1 sees the run queued (skipped); a multitask gate then drops it to
        # 'interrupted' — tick 2 must treat it as terminal instead of tailing it.
        statuses = {"q1": "queued"}
        calls = 0

        async def list_run_ids() -> list[tuple[str, str | None, str | None]]:
            nonlocal calls
            calls += 1
            if calls > 1:
                statuses["q1"] = "interrupted"
            return [("q1", statuses["q1"], None)]

        session = ThreadEventSession(
            "t1",
            channels={"lifecycle"},
            list_run_ids=list_run_ids,
            idle_grace_seconds=0.0,
        )

        async with asyncio.timeout(5):
            events = await _collect(session)

        assert events == []  # empty replay + terminal status → silent drain, no wedge
        assert "q1" in session._drained

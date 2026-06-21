"""Tests for v2 payload normalizers (interrupts, updates, state messages, lifecycle)."""

from aegra_api.services.event_streaming.normalizers import (
    lifecycle_status,
    normalize_input_requested,
    normalize_state_payload,
    normalize_updates,
    strip_interrupts,
)


class TestLifecycleStatus:
    def test_known_statuses_map(self) -> None:
        assert lifecycle_status("success") == "completed"
        assert lifecycle_status("error") == "failed"
        assert lifecycle_status("interrupted") == "interrupted"

    def test_unknown_status_defaults_running(self) -> None:
        """An in-progress/unknown status maps to running, never completed."""
        assert lifecycle_status("pending") == "running"
        assert lifecycle_status("anything") == "running"


class TestNormalizeUpdates:
    def test_single_node_extracts_node_and_values(self) -> None:
        assert normalize_updates({"agent": {"x": 1}}) == {"node": "agent", "values": {"x": 1}}

    def test_multi_node_keeps_whole_dict_as_values(self) -> None:
        out = normalize_updates({"a": {"x": 1}, "b": {"y": 2}})
        assert out == {"values": {"a": {"x": 1}, "b": {"y": 2}}}

    def test_non_dict_node_values_wrapped(self) -> None:
        assert normalize_updates({"agent": "done"}) == {"node": "agent", "values": {"value": "done"}}

    def test_non_dict_payload_wrapped(self) -> None:
        assert normalize_updates("x") == {"values": {"value": "x"}}


class TestNormalizeInputRequested:
    def test_interrupt_entry_to_request(self) -> None:
        entries = [{"id": "int-1", "value": {"question": "ok?"}}]
        assert normalize_input_requested(entries) == [{"interrupt_id": "int-1", "payload": {"question": "ok?"}}]

    def test_entry_without_value_omits_payload(self) -> None:
        assert normalize_input_requested([{"id": "int-2"}]) == [{"interrupt_id": "int-2"}]

    def test_entry_without_string_id_skipped(self) -> None:
        assert normalize_input_requested([{"value": "x"}, {"id": 5}]) == []

    def test_dunder_interrupt_wrapper_accepted(self) -> None:
        wrapped = {"__interrupt__": [{"id": "int-3", "value": "v"}]}
        assert normalize_input_requested(wrapped) == [{"interrupt_id": "int-3", "payload": "v"}]

    def test_non_interrupt_value_yields_nothing(self) -> None:
        assert normalize_input_requested({"messages": []}) == []


class TestStripInterrupts:
    def test_separates_interrupt_from_values(self) -> None:
        payload = {"messages": [], "__interrupt__": [{"id": "int-1", "value": "v"}]}
        requests, cleaned = strip_interrupts(payload)
        assert requests == [{"interrupt_id": "int-1", "payload": "v"}]
        assert cleaned == {"messages": []}
        assert "__interrupt__" not in cleaned

    def test_no_interrupt_passes_through(self) -> None:
        requests, cleaned = strip_interrupts({"messages": []})
        assert requests == []
        assert cleaned == {"messages": []}


class TestNormalizeStatePayload:
    def test_strips_interrupt_key_recursively(self) -> None:
        out = normalize_state_payload({"a": 1, "__interrupt__": ["x"]})
        assert out == {"a": 1}

    def test_assistant_role_normalized_to_ai(self) -> None:
        out = normalize_state_payload({"messages": [{"type": "assistant", "content": "hi", "id": "m1"}]})
        assert out["messages"][0]["type"] == "ai"

    def test_user_role_normalized_to_human(self) -> None:
        out = normalize_state_payload({"messages": [{"type": "user", "content": "hi", "id": "m1"}]})
        assert out["messages"][0]["type"] == "human"

    def test_tool_calls_split_valid(self) -> None:
        msg = {
            "type": "ai",
            "content": "",
            "id": "m1",
            "tool_calls": [{"id": "c1", "name": "get_weather", "args": {"city": "Paris"}}],
        }
        out = normalize_state_payload({"messages": [msg]})
        tcs = out["messages"][0]["tool_calls"]
        assert tcs == [{"type": "tool_call", "name": "get_weather", "args": {"city": "Paris"}, "id": "c1"}]

    def test_malformed_tool_call_args_marked_invalid(self) -> None:
        msg = {
            "type": "ai",
            "content": "",
            "id": "m1",
            "tool_calls": [{"id": "c1", "name": "f", "args": "{not json"}],
        }
        out = normalize_state_payload({"messages": [msg]})
        invalid = out["messages"][0]["invalid_tool_calls"]
        assert invalid[0]["type"] == "invalid_tool_call"
        assert invalid[0]["error"] == "Malformed args."

    def test_non_message_values_untouched(self) -> None:
        assert normalize_state_payload({"count": 3, "flag": True}) == {"count": 3, "flag": True}

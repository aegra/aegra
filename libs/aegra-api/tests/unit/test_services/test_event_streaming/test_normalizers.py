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

    def test_timeout_is_terminal_failed(self) -> None:
        """timeout is a terminal RunStatus, not an in-progress one."""
        assert lifecycle_status("timeout") == "failed"

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
        assert normalize_input_requested(entries) == [{"interrupt_id": "int-1", "value": {"question": "ok?"}}]

    def test_entry_without_value_omits_value(self) -> None:
        assert normalize_input_requested([{"id": "int-2"}]) == [{"interrupt_id": "int-2"}]

    def test_entry_without_string_id_skipped(self) -> None:
        assert normalize_input_requested([{"value": "x"}, {"id": 5}]) == []

    def test_dunder_interrupt_wrapper_accepted(self) -> None:
        wrapped = {"__interrupt__": [{"id": "int-3", "value": "v"}]}
        assert normalize_input_requested(wrapped) == [{"interrupt_id": "int-3", "value": "v"}]

    def test_non_interrupt_value_yields_nothing(self) -> None:
        assert normalize_input_requested({"messages": []}) == []


class TestStripInterrupts:
    def test_separates_interrupt_from_values(self) -> None:
        payload = {"messages": [], "__interrupt__": [{"id": "int-1", "value": "v"}]}
        requests, cleaned = strip_interrupts(payload)
        assert requests == [{"interrupt_id": "int-1", "value": "v"}]
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


class TestContentBlockCoverage:
    def _content(self, content: object) -> object:
        out = normalize_state_payload({"messages": [{"type": "ai", "content": content, "id": "m1"}]})
        return out["messages"][0]["content"]

    def test_server_tool_call_blocks_pass_through(self) -> None:
        blocks = [
            {"type": "server_tool_call", "name": "search", "args": {}},
            {"type": "server_tool_call_result", "output": "x"},
        ]
        assert self._content(blocks) == blocks

    def test_image_url_string_converts_to_image(self) -> None:
        assert self._content([{"type": "image_url", "image_url": "https://x/img.png"}]) == [
            {"type": "image", "url": "https://x/img.png"}
        ]

    def test_image_url_object_converts_to_image(self) -> None:
        assert self._content([{"type": "image_url", "image_url": {"url": "https://x/img.png"}}]) == [
            {"type": "image", "url": "https://x/img.png"}
        ]

    def test_input_audio_converts_to_audio(self) -> None:
        blocks = self._content([{"type": "input_audio", "input_audio": {"data": "b64", "mime_type": "audio/wav"}}])
        assert blocks == [{"type": "audio", "data": "b64", "mime_type": "audio/wav"}]

    def test_unknown_typed_block_wraps_as_non_standard(self) -> None:
        blocks = self._content([{"type": "provider_widget", "payload": 1}])
        assert blocks == [{"type": "non_standard", "value": {"type": "provider_widget", "payload": 1}}]

    def test_audio_synthesized_from_additional_kwargs_on_ai(self) -> None:
        msg = {
            "type": "ai",
            "content": "listen",
            "id": "m1",
            "additional_kwargs": {"audio": {"data": "b64", "format": "mp3", "transcript": "hi"}},
        }
        out = normalize_state_payload({"messages": [msg]})
        content = out["messages"][0]["content"]
        assert content == [
            {"type": "text", "text": "listen"},
            {"type": "audio", "data": "b64", "mime_type": "audio/mpeg", "transcript": "hi"},
        ]


class TestToolCallShapes:
    def test_openai_nested_function_shape_extracted(self) -> None:
        msg = {
            "type": "ai",
            "content": "",
            "id": "m1",
            "tool_calls": [{"id": "c1", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}],
        }
        out = normalize_state_payload({"messages": [msg]})
        assert out["messages"][0]["tool_calls"] == [
            {"type": "tool_call", "name": "get_weather", "args": {"city": "Paris"}, "id": "c1"}
        ]

    def test_tool_calls_fall_back_to_additional_kwargs(self) -> None:
        msg = {
            "type": "ai",
            "content": "",
            "id": "m1",
            "additional_kwargs": {"tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": "{}"}}]},
        }
        out = normalize_state_payload({"messages": [msg]})
        assert out["messages"][0]["tool_calls"][0]["name"] == "f"

    def test_explicit_invalid_tool_calls_preserved_with_error(self) -> None:
        msg = {
            "type": "ai",
            "content": "",
            "id": "m1",
            "invalid_tool_calls": [{"id": "c9", "name": "f", "args": "{broken", "error": "model hiccup"}],
        }
        out = normalize_state_payload({"messages": [msg]})
        invalid = out["messages"][0]["invalid_tool_calls"]
        assert invalid == [
            {"type": "invalid_tool_call", "id": "c9", "name": "f", "args": "{broken", "error": "model hiccup"}
        ]


class TestMessageFieldPreservation:
    def test_tool_artifact_preserved(self) -> None:
        msg = {"type": "tool", "content": "ok", "id": "m1", "tool_call_id": "c1", "artifact": {"rows": [1, 2]}}
        out = normalize_state_payload({"messages": [msg]})
        assert out["messages"][0]["artifact"] == {"rows": [1, 2]}

    def test_example_flag_preserved_on_ai_and_human(self) -> None:
        msgs = [
            {"type": "human", "content": "q", "id": "m1", "example": True},
            {"type": "ai", "content": "a", "id": "m2", "example": True},
        ]
        out = normalize_state_payload({"messages": msgs})
        assert all(m["example"] is True for m in out["messages"])

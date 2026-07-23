"""Unit tests for SSE utilities"""

import json
from datetime import datetime

from langchain_core.messages import ToolMessage
from langgraph.types import Command

from aegra_api.core.sse import (
    SSEEvent,
    create_debug_event,
    create_end_event,
    create_error_event,
    create_messages_event,
    create_metadata_event,
    format_sse_message,
    get_sse_headers,
)


class TestGetSSEHeaders:
    """Test get_sse_headers function"""

    def test_get_sse_headers(self):
        """Test SSE headers are correct"""
        headers = get_sse_headers()

        assert headers["Cache-Control"] == "no-cache"
        assert headers["Connection"] == "keep-alive"
        assert headers["Content-Type"] == "text/event-stream"
        assert headers["Access-Control-Allow-Origin"] == "*"
        assert headers["Access-Control-Allow-Headers"] == "Last-Event-ID"


class TestFormatSSEMessage:
    """Test format_sse_message function"""

    def test_format_basic_message(self):
        """Test basic SSE message formatting"""
        result = format_sse_message("test_event", {"key": "value"})

        assert "event: test_event\n" in result
        assert "data: " in result
        assert result.endswith("\n\n")

    def test_format_message_with_event_id(self):
        """Test SSE message with event ID"""
        result = format_sse_message("test_event", {"key": "value"}, event_id="evt-123")

        assert "event: test_event\n" in result
        assert "id: evt-123\n" in result
        assert "data: " in result

    def test_format_message_with_none_data(self):
        """Test SSE message with None data"""
        result = format_sse_message("test_event", None)

        assert "event: test_event\n" in result
        assert "data: \n" in result

    def test_format_message_with_nested_data(self):
        """Test SSE message with nested data"""
        data = {"outer": {"inner": {"deep": "value"}}}
        result = format_sse_message("test_event", data)

        assert "event: test_event\n" in result
        data_line = [line for line in result.split("\n") if line.startswith("data: ")][0]
        parsed_data = json.loads(data_line.replace("data: ", ""))
        assert parsed_data == data

    def test_format_message_decodes_literal_unicode_escapes(self):
        """Regression: literal \\uXXXX sequences from LLM streaming are decoded in SSE output.

        Some LLMs stream tool_call_chunks.args as raw JSON text where non-ASCII
        characters are literal \\uXXXX sequences instead of the actual characters.
        Verify these are decoded so the client sees Cyrillic, not escape sequences.
        """
        data = {"tool_call_chunks": [{"args": '{"thought": "\\u041f\\u0440\\u0438\\u0432\\u0435\\u0442"}'}]}
        result = format_sse_message("messages", data)
        data_line = next(line for line in result.split("\n") if line.startswith("data: "))
        assert "Привет" in data_line
        assert "\\u041f" not in data_line

    def test_format_message_preserves_ascii_control_escapes(self):
        """ASCII control-character escapes (\\u0000–\\u007f) are not decoded to keep JSON valid."""
        # \u0022 is a double-quote — if decoded inside a nested JSON string it would break structure.
        # Simulate an LLM streaming a nested JSON with a literal ASCII escape sequence.
        data = {"tool_call_chunks": [{"args": '{"key": "hello\\u0022world"}'}]}
        result = format_sse_message("test_event", data)
        data_line = next(line for line in result.split("\n") if line.startswith("data: "))
        # The \u0022 escape must NOT be decoded to a literal quote
        assert "\\u0022" in data_line
        # Overall JSON structure must remain valid
        parsed_data = json.loads(data_line.replace("data: ", ""))
        assert parsed_data == data

    def test_format_message_decodes_surrogate_pairs(self):
        """Surrogate pairs from LLM streaming are decoded to the actual character.

        Some LLMs encode emoji as surrogate pairs (\\uD83D\\uDE00 → 😀).
        Decoding each half independently produces a lone surrogate that cannot
        be encoded to UTF-8 and would crash the stream. Verify they are combined.
        """
        data = {"tool_call_chunks": [{"args": '{"emoji": "\\uD83D\\uDE00"}'}]}
        result = format_sse_message("messages", data)
        data_line = next(line for line in result.split("\n") if line.startswith("data: "))
        assert "😀" in data_line
        assert "\\uD83D" not in data_line

    def test_format_message_preserves_lone_surrogates(self):
        """A lone surrogate without its pair is left intact rather than decoded."""
        data = {"tool_call_chunks": [{"args": '{"x": "\\uD83D"}'}]}
        result = format_sse_message("messages", data)
        data_line = next(line for line in result.split("\n") if line.startswith("data: "))
        assert "\\uD83D" in data_line

    def test_format_message_with_custom_serializer(self):
        """Test SSE message with custom serializer"""

        def custom_serializer(obj):
            if isinstance(obj, datetime):
                return "custom_date"
            return str(obj)

        data = {"date": datetime.now()}
        result = format_sse_message("test_event", data, serializer=custom_serializer)

        assert "custom_date" in result

    def test_format_message_serializes_langgraph_command_structurally(self):
        """A Command embedded in an events payload (astream_events surfaces a
        tool-returned Command as on_tool_end.data.output) must reach the wire as a
        structural dict, not a repr string, so consumers can read
        output.update.messages instead of an undefined tool_call_id off a string."""
        event = {
            "event": "on_tool_end",
            "data": {
                "output": Command(
                    update={"messages": [ToolMessage("done", tool_call_id="call_abc", name="write_todos")]}
                )
            },
        }

        result = format_sse_message("events", event)

        data_line = next(line[len("data: ") :] for line in result.splitlines() if line.startswith("data: "))
        output = json.loads(data_line)["data"]["output"]
        assert isinstance(output, dict), "Command must not reach the wire as a repr string"
        message = output["update"]["messages"][0]
        assert message["type"] == "tool"
        assert message["tool_call_id"] == "call_abc"


class TestCreateMetadataEvent:
    """Test create_metadata_event function"""

    def test_create_metadata_event(self):
        """Test metadata event creation"""
        result = create_metadata_event("run-123")

        assert "event: metadata\n" in result
        assert "run-123" in result
        assert '"attempt":1' in result

    def test_create_metadata_event_with_event_id(self):
        """Test metadata event with event ID"""
        result = create_metadata_event("run-123", event_id="evt-1")

        assert "event: metadata\n" in result
        assert "id: evt-1\n" in result

    def test_create_metadata_event_with_custom_attempt(self):
        """Test metadata event with custom attempt"""
        result = create_metadata_event("run-123", attempt=3)

        assert '"attempt":3' in result


class TestCreateDebugEvent:
    """Test create_debug_event function"""

    def test_create_debug_event_basic(self):
        """Test basic debug event"""
        data = {"type": "task_result", "payload": {"result": "success"}}
        result = create_debug_event(data)

        assert "event: debug\n" in result
        assert "task_result" in result

    def test_create_debug_event_with_checkpoint_extraction(self):
        """Test debug event with checkpoint extraction"""
        data = {
            "type": "task_result",
            "payload": {
                "config": {
                    "configurable": {
                        "thread_id": "thread-123",
                        "checkpoint_id": "cp-456",
                        "checkpoint_ns": "ns",
                    }
                }
            },
        }
        result = create_debug_event(data)

        assert "thread-123" in result
        assert "cp-456" in result
        assert "checkpoint" in result

    def test_create_debug_event_with_parent_checkpoint_extraction(self):
        """Test debug event with parent checkpoint extraction"""
        data = {
            "type": "task_result",
            "payload": {
                "parent_config": {
                    "configurable": {
                        "thread_id": "thread-123",
                        "checkpoint_id": "cp-parent",
                    }
                }
            },
        }
        result = create_debug_event(data)

        assert "thread-123" in result
        assert "cp-parent" in result
        assert "parent_checkpoint" in result

    def test_create_debug_event_with_null_parent_config(self):
        """Test debug event with null parent config"""
        data = {"type": "task_result", "payload": {"parent_config": None}}
        result = create_debug_event(data)

        assert "event: debug\n" in result


class TestCreateEndEvent:
    """Test create_end_event function"""

    def test_create_end_event(self):
        """Test end event creation"""
        result = create_end_event()

        assert "event: end\n" in result
        assert "success" in result


class TestCreateErrorEvent:
    """Test create_error_event function"""

    def test_create_error_event_string(self):
        """Test error event creation with string format"""
        result = create_error_event("Something went wrong")

        assert "event: error\n" in result
        assert "Something went wrong" in result

        # Parse and verify structure matches standard error format
        data_line = [line for line in result.split("\n") if line.startswith("data: ")][0]
        parsed_data = json.loads(data_line.replace("data: ", ""))
        assert parsed_data["error"] == "Error"
        assert parsed_data["message"] == "Something went wrong"
        # Verify format is standard: only 'error' and 'message' fields
        assert len(parsed_data) == 2, "Should only have 'error' and 'message' fields"

    def test_create_error_event_dict_format(self):
        """Test error event creation with structured dict format"""
        error_dict = {"error": "ValueError", "message": "Invalid input provided"}
        result = create_error_event(error_dict)

        assert "event: error\n" in result

        # Parse and verify structure matches standard error format
        data_line = [line for line in result.split("\n") if line.startswith("data: ")][0]
        parsed_data = json.loads(data_line.replace("data: ", ""))
        assert parsed_data["error"] == "ValueError"
        assert parsed_data["message"] == "Invalid input provided"
        # Verify format is standard: only 'error' and 'message' fields
        assert len(parsed_data) == 2, "Should only have 'error' and 'message' fields"

    def test_create_error_event_dict_partial(self):
        """Test error event with partial dict (missing fields)"""
        error_dict = {"error": "GraphRecursionError"}
        result = create_error_event(error_dict)

        data_line = [line for line in result.split("\n") if line.startswith("data: ")][0]
        parsed_data = json.loads(data_line.replace("data: ", ""))
        assert parsed_data["error"] == "GraphRecursionError"
        assert "message" in parsed_data  # Should have default message
        # Verify format is standard: only 'error' and 'message' fields
        assert len(parsed_data) == 2, "Should only have 'error' and 'message' fields"

    def test_create_error_event_with_event_id(self):
        """Test error event with event ID for reconnection support"""
        result = create_error_event("Test error", event_id="evt-error-123")

        assert "event: error\n" in result
        assert "id: evt-error-123\n" in result
        assert "Test error" in result


class TestCreateMessagesEvent:
    """Test create_messages_event function"""

    def test_create_messages_event_with_list(self):
        """Test messages event with list data"""
        messages = [{"role": "user", "content": "hello"}]
        result = create_messages_event(messages)

        assert "event: messages\n" in result
        assert "hello" in result

    def test_create_messages_event_with_tuple(self):
        """Test messages event with tuple (streaming format)"""
        message_chunk = {"content": "hello"}
        metadata = {"model": "gpt-4"}
        messages_data = (message_chunk, metadata)

        result = create_messages_event(messages_data)

        assert "event: messages\n" in result
        assert "hello" in result
        assert "gpt-4" in result

    def test_create_messages_event_with_custom_event_type(self):
        """Test messages event with custom event type"""
        messages = [{"role": "assistant", "content": "hi"}]
        result = create_messages_event(messages, event_type="messages/partial")

        assert "event: messages/partial\n" in result


class TestSSEEvent:
    """Test SSEEvent dataclass"""

    def test_sse_event_creation(self):
        """Test SSEEvent creation"""
        event = SSEEvent(id="evt-1", event="test", data={"key": "value"})

        assert event.id == "evt-1"
        assert event.event == "test"
        assert event.data == {"key": "value"}
        assert event.timestamp is not None

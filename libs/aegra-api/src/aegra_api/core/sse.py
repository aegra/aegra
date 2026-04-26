"""Server-Sent Events utilities and formatting"""

import json
import re
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sse_starlette import ServerSentEvent

from aegra_api.core.serializers import GeneralSerializer

# Global serializer instance
_serializer = GeneralSerializer()

# Cached SSE keepalive payload: ``: heartbeat\r\n\r\n`` (15 bytes).
# Matches langgraph-api's wire-format so tcpdump/logs line up with LangGraph
# Platform, and avoids per-tick datetime formatting that sse-starlette's
# default ping does.
_HEARTBEAT_EVENT = ServerSentEvent(comment="heartbeat")


def heartbeat_factory() -> ServerSentEvent:
    """Ping factory for ``EventSourceResponse(ping_message_factory=...)``.

    Returns the same cached ``ServerSentEvent`` on every call — the encoded
    payload is identical on every tick, so there's no reason to allocate.
    """
    return _HEARTBEAT_EVENT


# Some LLMs stream tool_call_chunks.args with literal \uXXXX sequences
# instead of actual Unicode characters. After json.dumps these become \\uXXXX (double-escaped).
# We decode them back in two passes: surrogate pairs first to avoid lone surrogates that
# cannot be encoded to UTF-8, then remaining non-ASCII, non-surrogate code points.
# ASCII control characters (< 0x80) are left intact to preserve JSON validity.
_SURROGATE_PAIR_RE = re.compile(
    r"\\\\u(D[89AB][0-9a-fA-F]{2})\\\\u(D[C-F][0-9a-fA-F]{2})",
    re.IGNORECASE,
)
_NONASCII_ESCAPE_RE = re.compile(r"\\\\u([0-9a-fA-F]{4})")


def _decode_literal_unicode_escapes(data_str: str) -> str:
    """Decode double-escaped \\uXXXX sequences in an already-JSON-encoded string."""
    if "\\u" not in data_str:
        return data_str
    # First pass: combine surrogate pairs (e.g. \\uD83D\\uDE00 → 😀)
    data_str = _SURROGATE_PAIR_RE.sub(
        lambda m: chr(((int(m.group(1), 16) - 0xD800) << 10) + (int(m.group(2), 16) - 0xDC00) + 0x10000),
        data_str,
    )
    # Second pass: decode remaining non-ASCII, non-surrogate escapes
    return _NONASCII_ESCAPE_RE.sub(
        lambda m: chr(cp) if (cp := int(m.group(1), 16)) >= 0x80 and not (0xD800 <= cp <= 0xDFFF) else m.group(0),
        data_str,
    )


def get_sse_headers() -> dict[str, str]:
    """Get standard SSE headers"""
    return {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Content-Type": "text/event-stream",
        "X-Accel-Buffering": "no",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Last-Event-ID",
    }


async def sse_to_bytes(inner: AsyncIterator[str]) -> AsyncIterator[bytes]:
    """Adapt an ``AsyncIterator[str]`` of pre-formatted SSE messages to bytes.

    ``sse_starlette.EventSourceResponse`` wraps plain strings as *new* SSE
    events (double-encoding them). Pass bytes through its iterator so our
    already-formatted messages reach the wire untouched.
    """
    async for chunk in inner:
        yield chunk.encode("utf-8")


def format_sse_message(
    event: str,
    data: Any,
    event_id: str | None = None,
    serializer: Callable[[Any], Any] | None = None,
) -> str:
    """Format a message as Server-Sent Event following SSE standard

    Args:
        event: SSE event type
        data: Data to serialize and send
        event_id: Optional event ID
        serializer: Optional custom serializer function
    """
    lines = []

    lines.append(f"event: {event}")

    # Convert data to JSON string
    if data is None:
        data_str = ""
    else:
        # Use our general serializer by default to handle complex objects
        default_serializer = serializer or _serializer.serialize
        data_str = json.dumps(data, default=default_serializer, separators=(",", ":"), ensure_ascii=False)
        data_str = _decode_literal_unicode_escapes(data_str)

    lines.append(f"data: {data_str}")

    if event_id:
        lines.append(f"id: {event_id}")

    lines.append("")  # Empty line to end the event

    return "\n".join(lines) + "\n"


def create_metadata_event(run_id: str, event_id: str | None = None, attempt: int = 1) -> str:
    """Create metadata event for LangSmith Studio compatibility"""
    data = {"run_id": run_id, "attempt": attempt}
    return format_sse_message("metadata", data, event_id)


def create_debug_event(debug_data: dict[str, Any], event_id: str | None = None) -> str:
    """Create debug event with checkpoint fields for LangSmith Studio compatibility"""

    # Add checkpoint and parent_checkpoint fields if not present
    if "payload" in debug_data and isinstance(debug_data["payload"], dict):
        payload = debug_data["payload"]

        # Extract checkpoint from config.configurable
        if "checkpoint" not in payload and "config" in payload:
            config = payload.get("config", {})
            if isinstance(config, dict) and "configurable" in config:
                configurable = config["configurable"]
                if isinstance(configurable, dict):
                    payload["checkpoint"] = {
                        "thread_id": configurable.get("thread_id"),
                        "checkpoint_id": configurable.get("checkpoint_id"),
                        "checkpoint_ns": configurable.get("checkpoint_ns", ""),
                    }

        # Extract parent_checkpoint from parent_config.configurable
        if "parent_checkpoint" not in payload and "parent_config" in payload:
            parent_config = payload.get("parent_config")
            if isinstance(parent_config, dict) and "configurable" in parent_config:
                configurable = parent_config["configurable"]
                if isinstance(configurable, dict):
                    payload["parent_checkpoint"] = {
                        "thread_id": configurable.get("thread_id"),
                        "checkpoint_id": configurable.get("checkpoint_id"),
                        "checkpoint_ns": configurable.get("checkpoint_ns", ""),
                    }
            elif parent_config is None:
                payload["parent_checkpoint"] = None

    return format_sse_message("debug", debug_data, event_id)


def create_end_event(event_id: str | None = None, status: str = "success") -> str:
    """Create end event — signals completion of stream."""
    return format_sse_message("end", {"status": status}, event_id)


def create_error_event(error: str | dict[str, Any], event_id: str | None = None) -> str:
    """Create error event with structured error information.

    Error format: {"error": str, "message": str}
    This format ensures compatibility with standard SSE error event consumers.

    Args:
        error: Either a simple error string, or a dict with structured error info.
               Dict format: {"error": "ErrorType", "message": "detailed message"}
        event_id: Optional SSE event ID for reconnection support.

    Returns:
        SSE-formatted error event string with standard error format.
    """
    if isinstance(error, dict):
        # Structured error format - standard format: {error: str, message: str}
        data = {
            "error": error.get("error", "Error"),
            "message": error.get("message", str(error)),
        }
    else:
        # Simple string format - wrap it to standard format
        data = {
            "error": "Error",
            "message": str(error),
        }
    return format_sse_message("error", data, event_id)


def create_messages_event(messages_data: Any, event_type: str = "messages", event_id: str | None = None) -> str:
    """Create messages event (messages, messages/partial, messages/complete, messages/metadata)"""
    # Handle tuple format for token streaming: (message_chunk, metadata)
    if isinstance(messages_data, tuple) and len(messages_data) == 2:
        message_chunk, metadata = messages_data
        # Format as expected by LangGraph SDK client
        data = [message_chunk, metadata]
        return format_sse_message(event_type, data, event_id)
    else:
        # Handle list of messages format
        return format_sse_message(event_type, messages_data, event_id)


@dataclass
class SSEEvent:
    """SSE Event data structure for event storage"""

    id: str
    event: str
    data: dict[str, Any]
    timestamp: datetime | None = None

    def __post_init__(self) -> None:
        """Set timestamp to current UTC time if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now(UTC)

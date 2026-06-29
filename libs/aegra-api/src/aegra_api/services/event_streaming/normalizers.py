"""Normalize raw v3 stream payloads into Agent Protocol v2 shapes.

The native v3 stream hands messages already content-block-shaped, but
``values`` / ``updates`` payloads still carry raw langchain message dicts
and ``__interrupt__`` markers. These helpers project those into the
protocol's state-message shape, split interrupts onto the input channel,
and map run status to a lifecycle status.
"""

from __future__ import annotations

import json
from typing import Any

_STATE_MESSAGE_TYPES = frozenset({"human", "user", "ai", "assistant", "system", "tool", "function", "remove"})

_CONTENT_BLOCK_TYPES = frozenset(
    {
        "text",
        "reasoning",
        "tool_call",
        "tool_call_chunk",
        "invalid_tool_call",
        "image",
        "audio",
        "video",
        "file",
        "non_standard",
    }
)


def lifecycle_status(run_status: str) -> str:
    """Map a persisted run status to a protocol lifecycle status."""
    if run_status == "success":
        return "completed"
    if run_status in ("error", "timeout"):
        return "failed"
    if run_status == "interrupted":
        return "interrupted"
    return "running"


def normalize_updates(payload: Any) -> dict[str, Any]:
    """Extract ``node`` + ``values`` from an updates payload (``{node: values}``)."""
    if isinstance(payload, dict) and len(payload) == 1:
        node, values = next(iter(payload.items()))
        return {"node": node, "values": _as_values(values)}
    return {"values": _as_values(payload)}


def _as_values(value: Any) -> Any:
    return value if isinstance(value, dict) else {"value": value}


def normalize_input_requested(payload: Any) -> list[dict[str, Any]]:
    """Project interrupt entries into ``{interrupt_id, value?}`` requests.

    ``value`` (not ``payload``) is the SDK's ``InterruptPayload`` field — it is
    what ``thread.interrupts[].value`` surfaces to the client.
    """
    requests: list[dict[str, Any]] = []
    for entry in _interrupt_array(payload):
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            continue
        request: dict[str, Any] = {"interrupt_id": entry["id"]}
        if "value" in entry:
            request["value"] = entry["value"]
        requests.append(request)
    return requests


def _interrupt_array(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("__interrupt__"), list):
        return payload["__interrupt__"]
    return []


def strip_interrupts(payload: Any) -> tuple[list[dict[str, Any]], Any]:
    """Split ``__interrupt__`` off a values payload: ``(input_requests, cleaned)``."""
    requests = normalize_input_requested(payload)
    if not isinstance(payload, dict) or "__interrupt__" not in payload:
        return requests, payload
    cleaned = {key: value for key, value in payload.items() if key != "__interrupt__"}
    return requests, cleaned


def normalize_state_payload(value: Any) -> Any:
    """Recursively normalize a state payload: message shapes in, ``__interrupt__`` out."""
    if isinstance(value, list):
        return [
            _normalize_message(item) if _is_state_message(item) else normalize_state_payload(item) for item in value
        ]
    if not isinstance(value, dict):
        return value
    out: dict[str, Any] = {}
    for key, entry in value.items():
        if key == "__interrupt__":
            continue
        if key == "messages" and isinstance(entry, list):
            out[key] = [_normalize_message(item) if _is_state_message(item) else item for item in entry]
            continue
        out[key] = normalize_state_payload(entry)
    return out


def _message_type(value: Any) -> str | None:
    if value == "assistant":
        return "ai"
    if value == "user":
        return "human"
    return value if isinstance(value, str) else None


def _is_state_message(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    msg_type = _message_type(value.get("type"))
    return msg_type is not None and msg_type in _STATE_MESSAGE_TYPES


def _normalize_message(value: dict[str, Any]) -> dict[str, Any]:
    msg_type = _message_type(value.get("type"))
    if msg_type is None:
        return value

    message: dict[str, Any] = {"type": msg_type, "content": _normalize_content(value.get("content", ""))}
    if isinstance(value.get("id"), str):
        message["id"] = value["id"]
    if isinstance(value.get("name"), str):
        message["name"] = value["name"]

    if msg_type == "tool":
        if isinstance(value.get("tool_call_id"), str):
            message["tool_call_id"] = value["tool_call_id"]
        if value.get("status") in ("success", "error"):
            message["status"] = value["status"]

    if msg_type == "ai":
        tool_calls, invalid = _split_tool_calls(value.get("tool_calls"))
        if tool_calls:
            message["tool_calls"] = tool_calls
        if invalid:
            message["invalid_tool_calls"] = invalid

    return message


def _normalize_content(content: Any) -> Any:
    if isinstance(content, str) or not isinstance(content, list):
        return content
    blocks: list[Any] = []
    for entry in content:
        if isinstance(entry, str):
            blocks.append({"type": "text", "text": entry})
        elif isinstance(entry, dict) and entry.get("type") in _CONTENT_BLOCK_TYPES:
            blocks.append(entry)
    return blocks or content


def _split_tool_calls(raw: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(raw, list):
        return [], []
    valid: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name") if isinstance(entry.get("name"), str) else None
        args = _coerce_args(entry.get("args"))
        if name is None:
            invalid.append(_invalid_call(entry, "Incomplete tool call."))
            continue
        if not args["valid"]:
            invalid.append(_invalid_call(entry, "Malformed args.", name=name))
            continue
        call: dict[str, Any] = {"type": "tool_call", "name": name, "args": args["args"]}
        if isinstance(entry.get("id"), str):
            call["id"] = entry["id"]
        valid.append(call)
    return valid, invalid


def _coerce_args(value: Any) -> dict[str, Any]:
    if isinstance(value, dict) or value is None:
        return {"valid": True, "args": value or {}}
    if isinstance(value, str):
        if not value:
            return {"valid": True, "args": {}}
        try:
            return {"valid": True, "args": json.loads(value)}
        except (json.JSONDecodeError, ValueError):
            return {"valid": False, "args": value}
    return {"valid": True, "args": value}


def _invalid_call(entry: dict[str, Any], error: str, *, name: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"type": "invalid_tool_call", "error": error}
    if name is not None:
        out["name"] = name
    if isinstance(entry.get("id"), str):
        out["id"] = entry["id"]
    if isinstance(entry.get("args"), str):
        out["args"] = entry["args"]
    return out

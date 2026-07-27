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
        "server_tool_call",
        "server_tool_call_chunk",
        "server_tool_call_result",
        "image",
        "audio",
        "video",
        "file",
        "non_standard",
    }
)

_MIME_TYPE_BY_AUDIO_FORMAT: dict[str, str] = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "pcm16": "audio/wav",
    "pcm": "audio/wav",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
}


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

    additional_kwargs = value["additional_kwargs"] if isinstance(value.get("additional_kwargs"), dict) else None

    message: dict[str, Any] = {
        "type": msg_type,
        # Audio rides additional_kwargs on ai messages only (voice model outputs).
        "content": _normalize_content(
            value.get("content", ""), additional_kwargs=additional_kwargs if msg_type == "ai" else None
        ),
    }
    if isinstance(value.get("id"), str):
        message["id"] = value["id"]
    if isinstance(value.get("name"), str):
        message["name"] = value["name"]
    if msg_type in ("ai", "human") and isinstance(value.get("example"), bool):
        message["example"] = value["example"]

    if msg_type == "tool":
        if isinstance(value.get("tool_call_id"), str):
            message["tool_call_id"] = value["tool_call_id"]
        if value.get("status") in ("success", "error"):
            message["status"] = value["status"]
        if "artifact" in value:
            message["artifact"] = value["artifact"]

    if msg_type == "ai":
        # OpenAI-shaped payloads may carry tool calls only in additional_kwargs.
        raw_tool_calls = value.get("tool_calls") if isinstance(value.get("tool_calls"), list) else None
        if not raw_tool_calls and additional_kwargs and isinstance(additional_kwargs.get("tool_calls"), list):
            raw_tool_calls = additional_kwargs["tool_calls"]
        tool_calls, invalid_from_valid = _split_tool_calls(raw_tool_calls)
        # An explicit invalid_tool_calls field wins over ones derived from tool_calls.
        raw_invalid = value.get("invalid_tool_calls")
        invalid = (
            _normalize_invalid_tool_calls(raw_invalid)
            if isinstance(raw_invalid, list) and raw_invalid
            else invalid_from_valid
        )
        if tool_calls:
            message["tool_calls"] = tool_calls
        if invalid:
            message["invalid_tool_calls"] = invalid

    return message


def _normalize_content(content: Any, additional_kwargs: dict[str, Any] | None = None) -> Any:
    audio_block = _audio_block_from_additional_kwargs(additional_kwargs)
    if isinstance(content, str):
        if audio_block is None:
            return content
        blocks: list[Any] = [{"type": "text", "text": content}] if content else []
        blocks.append(audio_block)
        return blocks
    if not isinstance(content, list):
        return [audio_block] if audio_block is not None else content

    blocks = []
    for entry in content:
        if isinstance(entry, str):
            blocks.append({"type": "text", "text": entry})
            continue
        normalized = _normalize_block(entry)
        if normalized is not None:
            blocks.append(normalized)
    if audio_block is not None and not any(b.get("type") == "audio" for b in blocks):
        blocks.append(audio_block)
    return blocks if blocks else content


def _normalize_block(value: Any) -> dict[str, Any] | None:
    """Normalize one raw content block into a protocol block.

    Known types pass through; provider shapes (``image_url``, ``input_audio``)
    convert; anything else typed wraps as ``non_standard`` rather than dropping.
    """
    if not isinstance(value, dict) or not isinstance(value.get("type"), str):
        return None
    if value["type"] in _CONTENT_BLOCK_TYPES:
        return value

    if value["type"] == "image_url":
        raw_image = value.get("image_url")
        if isinstance(raw_image, str):
            return {"type": "image", "url": raw_image}
        if isinstance(raw_image, dict) and isinstance(raw_image.get("url"), str):
            return {"type": "image", "url": raw_image["url"]}
        return None

    if value["type"] == "input_audio":
        raw_audio = value.get("input_audio") if isinstance(value.get("input_audio"), dict) else None
        if raw_audio is None:
            return None
        block: dict[str, Any] = {"type": "audio"}
        if isinstance(raw_audio.get("data"), str):
            block["data"] = raw_audio["data"]
        if isinstance(raw_audio.get("mime_type"), str):
            block["mime_type"] = raw_audio["mime_type"]
        return block

    return {"type": "non_standard", "value": {**value}}


def _audio_block_from_additional_kwargs(additional_kwargs: dict[str, Any] | None) -> dict[str, Any] | None:
    if additional_kwargs is None:
        return None
    audio = additional_kwargs.get("audio")
    if not isinstance(audio, dict):
        return None
    data = audio.get("data") if isinstance(audio.get("data"), str) else None
    url = audio.get("url") if isinstance(audio.get("url"), str) else None
    if data is None and url is None:
        return None

    fmt = (
        audio["format"].lower()
        if isinstance(audio.get("format"), str)
        else (None if isinstance(audio.get("mime_type"), str) else "wav")
    )
    block: dict[str, Any] = {"type": "audio"}
    if isinstance(audio.get("id"), str):
        block["id"] = audio["id"]
    if url is not None:
        block["url"] = url
    if data is not None:
        block["data"] = data
    if isinstance(audio.get("mime_type"), str):
        block["mime_type"] = audio["mime_type"]
    elif fmt is not None and fmt in _MIME_TYPE_BY_AUDIO_FORMAT:
        block["mime_type"] = _MIME_TYPE_BY_AUDIO_FORMAT[fmt]
    if isinstance(audio.get("transcript"), str):
        block["transcript"] = audio["transcript"]
    return block


def _tool_call_identity(value: dict[str, Any]) -> tuple[str | None, str | None]:
    """(id, name) for a tool call, reading the OpenAI nested function shape too."""
    nested = value.get("function") if isinstance(value.get("function"), dict) else None
    call_id = value["id"] if isinstance(value.get("id"), str) else None
    name = value.get("name") if isinstance(value.get("name"), str) else None
    if name is None and nested is not None and isinstance(nested.get("name"), str):
        name = nested["name"]
    return call_id, name


def _tool_call_args(value: dict[str, Any]) -> Any:
    if "args" in value:
        return value["args"]
    nested = value.get("function") if isinstance(value.get("function"), dict) else None
    return nested.get("arguments") if nested else None


def _split_tool_calls(raw: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(raw, list):
        return [], []
    valid: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        call_id, name = _tool_call_identity(entry)
        raw_args = _tool_call_args(entry)
        args = _coerce_args(raw_args)
        if name is None:
            item: dict[str, Any] = {"type": "invalid_tool_call", "error": "Incomplete tool call."}
            if call_id is not None:
                item["id"] = call_id
            if isinstance(raw_args, str):
                item["args"] = raw_args
            invalid.append(item)
            continue
        if not args["valid"]:
            item = {"type": "invalid_tool_call", "name": name, "error": "Malformed args."}
            if call_id is not None:
                item["id"] = call_id
            if isinstance(args["args"], str):
                item["args"] = args["args"]
            invalid.append(item)
            continue
        call: dict[str, Any] = {"type": "tool_call", "name": name, "args": args["args"]}
        if call_id is not None:
            call["id"] = call_id
        valid.append(call)
    return valid, invalid


def _normalize_invalid_tool_calls(raw: list[Any]) -> list[dict[str, Any]]:
    """Normalize a message's own invalid_tool_calls list, preserving its error strings."""
    result: list[dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        call_id, name = _tool_call_identity(entry)
        item: dict[str, Any] = {"type": "invalid_tool_call"}
        if call_id is not None:
            item["id"] = call_id
        if name is not None:
            item["name"] = name
        if isinstance(entry.get("args"), str):
            item["args"] = entry["args"]
        item["error"] = entry["error"] if isinstance(entry.get("error"), str) else "Malformed args."
        result.append(item)
    return result


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

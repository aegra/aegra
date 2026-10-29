"""Pure mapping between Open Responses payloads and LangGraph runs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from aegra_api.models.responses import ResponseCreate, ResponsesError

_ROLE_BY_TYPE = {
    "human": "user",
    "user": "user",
    "ai": "assistant",
    "assistant": "assistant",
    "system": "system",
    "developer": "system",
    "tool": "tool",
    "function": "tool",
}

_REJECTED_PART_TYPES = frozenset(
    {
        "input_image",
        "input_file",
        "image_url",
        "input_audio",
        "audio",
        "file",
    }
)

_TEXT_PART_TYPES = frozenset({"input_text", "output_text", "text"})


@dataclass(frozen=True)
class PreparedTurn:
    """What a Responses request should do to the graph."""

    messages: dict[str, Any] | None
    resume: Any
    rejected: bool
    reject_reason: str | None
    interrupt_id: str | None = None


def conversation_id_of(value: str | dict[str, Any] | None) -> str | None:
    """Read a conversation id from a string or ``{"id": ...}``."""
    if value is None:
        return None
    if isinstance(value, str):
        if not value:
            raise ResponsesError(422, "conversation id must not be empty")
        return value
    if isinstance(value, dict) and isinstance(value.get("id"), str) and value["id"]:
        return value["id"]
    raise ResponsesError(422, "conversation must be an id string or an object with an id")


def is_ephemeral(*, store: bool, conversation_id: str | None, previous_response_id: str | None) -> bool:
    """Stateless responses are deleted after the call unless the client stores them."""
    return not store and conversation_id is None and previous_response_id is None


def threads_conflict(conversation_id: str | None, previous_thread_id: str | None) -> bool:
    """True when both ids are set and they are not the same thread."""
    return bool(conversation_id and previous_thread_id and conversation_id != previous_thread_id)


def bind_resume(interrupt_id: str, resume: Any, pending_ids: list[str]) -> Any:
    """Attach a resume value to the interrupt the client named.

    One pending interrupt receives the value directly, which is what
    ``interrupt()`` yields. Several pending interrupts need LangGraph's
    id-keyed map so the value cannot land on a different interrupt.
    """
    if interrupt_id not in pending_ids:
        raise ResponsesError(409, f"Interrupt '{interrupt_id}' is not pending")
    if len(pending_ids) == 1:
        return resume
    return {interrupt_id: resume}


def response_status(run_status: str, output: Mapping[str, Any] | None) -> str:
    """Map a persisted run status onto a Responses status."""
    if run_status in ("pending", "running"):
        return "in_progress"
    if run_status == "success":
        return "completed"
    if run_status in ("error", "timeout"):
        return "failed"
    if run_status == "interrupted":
        if _interrupt_entries(output):
            return "incomplete"
        return "cancelled"
    return "failed"


def prepare_turn(body: ResponseCreate) -> PreparedTurn:
    """Turn Responses input into graph messages or a resume command."""
    if isinstance(body.input, str):
        return PreparedTurn(
            messages=_message_input(body.instructions, body.input), resume=None, rejected=False, reject_reason=None
        )
    if not isinstance(body.input, list) or not body.input:
        raise ResponsesError(422, "input must be a non-empty string or a list of items")

    approvals: list[dict[str, Any]] = []
    call_outputs: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = []
    for item in body.input:
        if not isinstance(item, dict):
            raise ResponsesError(422, "each input item must be an object")
        item_type = item.get("type")
        if item_type == "mcp_approval_response":
            approvals.append(item)
        elif item_type == "function_call_output":
            call_outputs.append(item)
        elif item_type in (None, "message") or "role" in item:
            messages.append(item)
        else:
            raise ResponsesError(422, f"input item type {item_type!r} is not supported")

    if approvals and call_outputs:
        raise ResponsesError(422, "send either an approval or a function_call_output, not both")
    if len(approvals) > 1 or len(call_outputs) > 1:
        raise ResponsesError(422, "only one approval or function_call_output is supported")
    if (approvals or call_outputs) and messages:
        raise ResponsesError(422, "approval input cannot be combined with messages")
    needs_thread = bool(approvals or call_outputs)
    if needs_thread and conversation_id_of(body.conversation) is None and not body.previous_response_id:
        raise ResponsesError(422, "resuming an interrupt requires a conversation or previous_response_id")

    if approvals:
        interrupt_id = _require_id(approvals[0].get("approval_request_id"), "mcp_approval_response.approval_request_id")
        return _approval_turn(approvals[0], interrupt_id)
    if call_outputs:
        interrupt_id = _require_id(call_outputs[0].get("call_id"), "function_call_output.call_id")
        return PreparedTurn(
            messages=None,
            resume=decode_resume(call_outputs[0].get("output")),
            rejected=False,
            reject_reason=None,
            interrupt_id=interrupt_id,
        )

    built = _items_to_messages(body.instructions, messages)
    return PreparedTurn(messages={"messages": built}, resume=None, rejected=False, reject_reason=None)


def decode_resume(output: Any) -> Any:
    """JSON-decode a function_call_output into a ``Command(resume=...)`` value."""
    if isinstance(output, (dict, list, int, float, bool)) or output is None:
        return output
    if isinstance(output, str):
        try:
            return json.loads(output)
        except json.JSONDecodeError as exc:
            raise ResponsesError(422, "function_call_output.output must be JSON") from exc
    raise ResponsesError(422, "function_call_output.output must be JSON")


def project_response(
    *,
    response_id: str,
    run_status: str,
    output: dict[str, Any] | None,
    error_message: str | None,
    created_at: datetime,
    assistant_id: str | None,
    meta: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a Response object from a run row and its facade metadata."""
    status = response_status(run_status, output)
    raw_offset = meta.get("message_offset")
    offset = raw_offset if isinstance(raw_offset, int) and raw_offset > 0 else 0
    items: list[dict[str, Any]] = []
    if status not in ("failed", "in_progress"):
        items = project_response_output(output, offset=offset)
    error: dict[str, str] | None = None
    incomplete: dict[str, str] | None = None
    if status == "failed":
        code = meta.get("error_code")
        error = {
            "code": code if isinstance(code, str) else "server_error",
            "message": error_message or "Run failed",
        }
    if status == "incomplete":
        incomplete = {"reason": "approval_required"}
    conversation_id = meta.get("conversation_id")
    raw_metadata = meta.get("metadata")
    metadata: dict[Any, Any] = raw_metadata if isinstance(raw_metadata, dict) else {}
    model = meta.get("model")
    body: dict[str, Any] = {
        "id": response_id,
        "object": "response",
        "created_at": int(created_at.timestamp()),
        "status": status,
        "error": error,
        "incomplete_details": incomplete,
        "instructions": meta.get("instructions") if isinstance(meta.get("instructions"), str) else None,
        "model": model if isinstance(model, str) and model else (assistant_id or ""),
        "output": items,
        "previous_response_id": meta.get("previous_response_id")
        if isinstance(meta.get("previous_response_id"), str)
        else None,
        "conversation": {"id": conversation_id} if isinstance(conversation_id, str) else None,
        "metadata": {key: value for key, value in metadata.items() if isinstance(key, str) and isinstance(value, str)},
        "background": bool(meta.get("background")),
    }
    user = meta.get("user")
    if isinstance(user, str):
        body["user"] = user
    return body


def retains_stateless_thread(run_status: str, output: Mapping[str, Any] | None) -> bool:
    """Keep a store=false thread when it is paused on an interrupt the client must answer."""
    return run_status == "interrupted" and bool(_interrupt_entries(output))


def project_response_output(output: Mapping[str, Any] | None, *, offset: int = 0) -> list[dict[str, Any]]:
    """Assistant-facing output items: messages, tool calls, and approval requests.

    ``offset`` drops messages that were already on the thread before this run.
    """
    messages = _messages(output)
    if offset > 0:
        messages = messages[offset:]
    items: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        mapped = _as_mapping(message)
        if mapped is None:
            continue
        role = _message_role(mapped)
        if role == "assistant":
            items.extend(_assistant_items(mapped, index))
        elif role == "tool":
            tool_item = _tool_output_item(mapped, index)
            if tool_item is not None:
                items.append(tool_item)
    items.extend(interrupt_items(output))
    return items


def project_conversation_items(messages: list[Any]) -> list[dict[str, Any]]:
    """Full thread history as conversation items, including the caller's turns."""
    items: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        mapped = _as_mapping(message)
        if mapped is None:
            continue
        role = _message_role(mapped)
        if role in ("user", "system", "assistant"):
            text = _text_from_content(mapped.get("content"))
            if role != "assistant" or text:
                items.append(_message_item(mapped, index, role=role, text=text, output=role == "assistant"))
            if role == "assistant":
                items.extend(_function_calls(mapped, index))
        elif role == "tool":
            tool_item = _tool_output_item(mapped, index)
            if tool_item is not None:
                items.append(tool_item)
    return items


def interrupt_items(output: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Pending LangGraph interrupts as ``mcp_approval_request`` items."""
    items: list[dict[str, Any]] = []
    for entry in _interrupt_entries(output):
        interrupt_id = entry.get("id")
        if not isinstance(interrupt_id, str):
            continue
        value = entry.get("value")
        items.append(
            {
                "id": interrupt_id,
                "type": "mcp_approval_request",
                "server_label": "aegra",
                "name": _interrupt_name(value),
                "arguments": json.dumps(value, default=str),
            }
        )
    return items


def public_thread_metadata(metadata: Mapping[str, Any] | None) -> dict[str, str]:
    """Caller metadata stored on the thread, without Aegra's internal keys."""
    if not isinstance(metadata, Mapping):
        return {}
    stored = metadata.get("responses_metadata")
    if not isinstance(stored, dict):
        return {}
    return {key: value for key, value in stored.items() if isinstance(key, str) and isinstance(value, str)}


def meta_from_params(execution_params: Mapping[str, Any] | None, assistant_id: str | None) -> dict[str, Any]:
    """Read facade bookkeeping, falling back to the assistant id for the model name."""
    raw = (execution_params or {}).get("responses_meta") if execution_params else None
    if isinstance(raw, dict):
        return raw
    return {"model": assistant_id or ""}


def build_responses_meta(
    body: ResponseCreate, *, conversation_id: str | None, error_code: str | None = None
) -> dict[str, Any]:
    """Bookkeeping persisted on the run, separate from trace metadata."""
    meta: dict[str, Any] = {
        "model": body.model,
        "previous_response_id": body.previous_response_id,
        "conversation_id": conversation_id,
        "instructions": body.instructions,
        "metadata": body.metadata or {},
        "background": body.background,
        "user": body.user,
    }
    if error_code is not None:
        meta["error_code"] = error_code
    return meta


def _require_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ResponsesError(422, f"{field} is required")
    return value


def _approval_turn(item: dict[str, Any], interrupt_id: str) -> PreparedTurn:
    approve = item.get("approve")
    if not isinstance(approve, bool):
        raise ResponsesError(422, "mcp_approval_response.approve must be a boolean")
    if approve:
        return PreparedTurn(
            messages=None,
            resume={"decisions": [{"type": "approve"}]},
            rejected=False,
            reject_reason=None,
            interrupt_id=interrupt_id,
        )
    reason = item.get("reason")
    if reason is not None and not isinstance(reason, str):
        raise ResponsesError(422, "mcp_approval_response.reason must be a string")
    return PreparedTurn(
        messages=None,
        resume=None,
        rejected=True,
        reject_reason=reason or "interrupt rejected",
        interrupt_id=interrupt_id,
    )


def _message_input(instructions: str | None, text: str) -> dict[str, Any]:
    messages: list[dict[str, str]] = []
    if instructions:
        messages.append({"role": "system", "content": instructions})
    messages.append({"role": "user", "content": text})
    return {"messages": messages}


def _items_to_messages(instructions: str | None, items: list[dict[str, Any]]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if instructions:
        messages.append({"role": "system", "content": instructions})
    if not items:
        raise ResponsesError(422, "input must include a message")
    for item in items:
        role = _message_role(item)
        if role not in ("user", "assistant", "system"):
            raise ResponsesError(422, "message role must be user, assistant, or system")
        messages.append({"role": role, "content": _input_text(item.get("content"))})
    return messages


def _input_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if not isinstance(content, list):
        raise ResponsesError(422, "input content must be a string or a list of text parts")
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
            continue
        if not isinstance(block, dict):
            raise ResponsesError(422, "input content parts must be objects")
        block_type = block.get("type")
        if block_type in _REJECTED_PART_TYPES or (block_type is not None and block_type not in _TEXT_PART_TYPES):
            raise ResponsesError(422, f"input content type {block_type!r} is not supported")
        text = block.get("text")
        if not isinstance(text, str):
            raise ResponsesError(422, "text content parts require a string text field")
        parts.append(text)
    return "".join(parts)


def _messages(output: Mapping[str, Any] | None) -> list[Any]:
    if not isinstance(output, Mapping):
        return []
    messages = output.get("messages")
    if isinstance(messages, list):
        return messages
    return []


def _interrupt_entries(output: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(output, Mapping):
        return []
    raw = output.get("__interrupt__")
    if not isinstance(raw, list):
        return []
    return [entry for entry in raw if isinstance(entry, dict)]


def _interrupt_name(value: Any) -> str:
    if isinstance(value, dict):
        action = value.get("action_request")
        if isinstance(action, dict) and isinstance(action.get("action"), str):
            return action["action"]
        if isinstance(value.get("name"), str):
            return value["name"]
    return "interrupt"


def _message_role(message: Mapping[str, Any]) -> str:
    raw = message.get("role") or message.get("type") or ""
    if not isinstance(raw, str):
        return ""
    return _ROLE_BY_TYPE.get(raw.lower(), "")


def _text_from_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
                continue
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str) and block.get("type") in (None, "text", "output_text", "input_text"):
                    parts.append(text)
        return "".join(parts)
    return ""


def _as_mapping(message: Any) -> dict[str, Any] | None:
    if isinstance(message, dict):
        return message
    if hasattr(message, "model_dump"):
        try:
            dumped = message.model_dump()
        except (TypeError, ValueError):
            dumped = None
        if isinstance(dumped, dict):
            return dumped
    if not hasattr(message, "content") and not hasattr(message, "type"):
        return None
    return {
        "id": getattr(message, "id", None),
        "type": getattr(message, "type", None),
        "role": getattr(message, "role", None),
        "content": getattr(message, "content", None),
        "tool_calls": getattr(message, "tool_calls", None),
        "tool_call_id": getattr(message, "tool_call_id", None),
        "name": getattr(message, "name", None),
    }


def _assistant_items(message: Mapping[str, Any], index: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    text = _text_from_content(message.get("content"))
    if text:
        items.append(_message_item(message, index, role="assistant", text=text, output=True))
    items.extend(_function_calls(message, index))
    return items


def _message_item(
    message: Mapping[str, Any],
    index: int,
    *,
    role: str,
    text: str,
    output: bool,
) -> dict[str, Any]:
    item_id = message.get("id") if isinstance(message.get("id"), str) else f"msg_{index}"
    content_type = "output_text" if output else "input_text"
    content: dict[str, Any] = {"type": content_type, "text": text}
    if output:
        content["annotations"] = []
    return {
        "id": item_id,
        "type": "message",
        "role": role,
        "status": "completed",
        "content": [content],
    }


def _function_calls(message: Mapping[str, Any], index: int) -> list[dict[str, Any]]:
    raw_calls = message.get("tool_calls")
    if not isinstance(raw_calls, list):
        return []
    items: list[dict[str, Any]] = []
    for call_index, call in enumerate(raw_calls):
        item = _function_call_item(call, f"{index}_{call_index}")
        if item is not None:
            items.append(item)
    return items


def _function_call_item(call: Any, suffix: str) -> dict[str, Any] | None:
    if not isinstance(call, dict):
        return None
    name = call.get("name")
    if not isinstance(name, str):
        return None
    call_id = call.get("id") if isinstance(call.get("id"), str) else f"call_{suffix}"
    args = call.get("args", call.get("arguments", {}))
    arguments = args if isinstance(args, str) else json.dumps(args, default=str)
    return {
        "id": call_id,
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
        "status": "completed",
    }


def _tool_output_item(message: Mapping[str, Any], index: int) -> dict[str, Any] | None:
    call_id = message.get("tool_call_id") or message.get("id")
    if not isinstance(call_id, str):
        call_id = f"call_{index}"
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": _text_from_content(message.get("content")),
    }

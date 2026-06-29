"""Wire builders for the Agent Protocol v2 envelope.

The protocol types in ``langchain_protocol`` are plain ``TypedDict``s with
no serialization behaviour — the wire form is a plain dict serialized with
``json.dumps``. These helpers build those dicts so field names live in one
place and stay exact (``event_id``, ``seq``, ``applied_through_seq``).
"""

from __future__ import annotations

from typing import Any, Literal

# Server event channels (the envelope ``method``).
EventMethod = Literal[
    "lifecycle",
    "messages",
    "tools",
    "input.requested",
    "values",
    "updates",
    "checkpoints",
    "custom",
    "tasks",
]

# Error codes a command response may carry.
ErrorCode = Literal[
    "invalid_argument",
    "unknown_command",
    "unknown_error",
    "no_such_run",
    "no_such_interrupt",
    "permission_denied",
    "not_supported",
]


def build_event(
    method: EventMethod | str,
    data: dict[str, Any],
    *,
    namespace: list[str] | None = None,
    seq: int,
    event_id: str | None = None,
) -> dict[str, Any]:
    """Build a server-push event envelope.

    ``params`` wraps the per-channel payload as ``{data, namespace}`` — the
    shape the LangGraph SDK reads (``params.data`` is the payload,
    ``params.namespace`` the subgraph path, ``[]`` at the root).
    """
    params: dict[str, Any] = {"data": data, "namespace": namespace or []}
    event: dict[str, Any] = {"type": "event", "seq": seq, "method": method, "params": params}
    if event_id is not None:
        event["event_id"] = event_id
    return event


def build_success(
    command_id: int,
    result: dict[str, Any],
    *,
    applied_through_seq: int | None = None,
) -> dict[str, Any]:
    """Build a success command response."""
    response: dict[str, Any] = {"type": "success", "id": command_id, "result": result}
    if applied_through_seq is not None:
        response["meta"] = {"applied_through_seq": applied_through_seq}
    return response


def build_error(
    command_id: int | None,
    error: ErrorCode,
    message: str,
) -> dict[str, Any]:
    """Build an error command response."""
    return {"type": "error", "id": command_id, "error": error, "message": message}

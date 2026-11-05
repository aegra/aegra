"""The Agent Protocol v2 ``input.inject`` command: a user or system message for a running run.

The message is stored for the thread and reaches the graph through its injected-input
reader (``aegra_api.services.injected_input``); no run is started. The route's
``threads.create_run`` handler result is honored as on v1 run creation: a returned
dict wins, else the value it may have mutated, and its ``params.message`` fields
merge over the client's message.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from aegra_api.models import User
from aegra_api.services.event_streaming.protocol import ErrorCode, build_error, build_success
from aegra_api.services.injected_input import store_message, thread_admission

_ROLES = frozenset({"user", "system"})
_MESSAGE_FIELDS = frozenset({"role", "content", "name", "id"})


async def input_inject(
    command_id: int,
    params: dict[str, Any],
    *,
    session: AsyncSession,
    thread_id: str,
    user: User,
    authorized: dict[str, Any] | None,
) -> dict[str, Any]:
    """Store a message for the thread's active run; returns the response envelope."""
    namespace_error = _namespace_error(params.get("namespace", []))
    if namespace_error is not None:
        return build_error(command_id, *namespace_error)

    raw = params.get("message")
    if not isinstance(raw, dict):
        return build_error(command_id, "invalid_argument", "input.inject requires a message object.")
    raw = {**raw, **_handler_message_fields(authorized)}
    invalid = _message_error(raw)
    if invalid is not None:
        return build_error(command_id, "invalid_argument", invalid)

    admission = await thread_admission(session, thread_id=thread_id, user_id=user.identity)
    if admission == "no_active_run":
        return build_error(command_id, "no_such_run", "No active run on this thread; use run.start to send input.")
    if admission == "not_accepting":
        return build_error(
            command_id, "not_supported", "The active run does not take input.inject; its graph never accepted it."
        )

    message_id = raw.get("id") or str(uuid4())
    message = _graph_message(raw, message_id=message_id, injected_by=user.identity)
    if not await store_message(
        session, thread_id=thread_id, user_id=user.identity, message_id=message_id, message=message
    ):
        return build_error(
            command_id, "invalid_argument", f"Message id {message_id!r} was already sent with a different message."
        )
    return build_success(command_id, {}, applied_through_seq=0)


def _namespace_error(namespace: Any) -> tuple[ErrorCode, str] | None:
    if not isinstance(namespace, list) or not all(isinstance(part, str) for part in namespace):
        return "invalid_argument", "namespace must be an array of strings."
    if namespace:
        return "not_supported", "input.inject targets the root graph only; namespace must be []."
    return None


def _handler_message_fields(authorized: dict[str, Any] | None) -> dict[str, Any]:
    params = authorized.get("params") if isinstance(authorized, dict) else None
    message = params.get("message") if isinstance(params, dict) else None
    return message if isinstance(message, dict) else {}


def _message_error(raw: dict[str, Any]) -> str | None:
    if raw.get("role") not in _ROLES:
        return "message.role must be 'user' or 'system'."
    content = raw.get("content")
    if not isinstance(content, str) or not content:
        return "message.content must be a non-empty string."
    if "name" in raw and not isinstance(raw["name"], str):
        return "message.name must be a string."
    if "id" in raw and (not isinstance(raw["id"], str) or not raw["id"]):
        return "message.id must be a non-empty string."
    return None


def _graph_message(raw: dict[str, Any], *, message_id: str, injected_by: str) -> dict[str, Any]:
    """The message as the graph receives it: extension fields in additional_kwargs, the sender stamped."""
    extra = {key: value for key, value in raw.items() if key not in _MESSAGE_FIELDS}
    message: dict[str, Any] = {
        "role": raw["role"],
        "content": raw["content"],
        "id": message_id,
        "additional_kwargs": {**extra, "injected_by": injected_by},
    }
    if "name" in raw:
        message["name"] = raw["name"]
    return message

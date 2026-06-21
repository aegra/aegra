"""Dispatch Agent Protocol v2 thread commands.

Commands are JSON-RPC-style: ``{id, method, params}`` in, a success or
error envelope out. They re-front the existing run machinery — ``run.start``
and ``input.respond`` both build a ``RunCreate`` and go through the same
``_prepare_run`` path the legacy endpoints use, so execution semantics are
identical; only the transport differs.
"""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from aegra_api.models import User
from aegra_api.models.runs import RunCreate
from aegra_api.services.event_streaming.protocol import ErrorCode, build_error, build_success
from aegra_api.services.run_preparation import _prepare_run

logger = structlog.getLogger(__name__)


def _status_to_error_code(status_code: int) -> ErrorCode:
    """Map an HTTP status from run preparation to a protocol error code."""
    if status_code == 404:
        return "no_such_run"
    if status_code == 403:
        return "permission_denied"
    return "invalid_argument"


async def handle_command(
    payload: dict[str, Any],
    *,
    session: AsyncSession,
    thread_id: str,
    user: User,
) -> tuple[dict[str, Any], str | None]:
    """Dispatch one command. Returns ``(response_envelope, started_run_id)``.

    ``started_run_id`` is the run a ``run.start`` / ``input.respond`` created,
    so the caller can open a stream for it; ``None`` for other commands.
    """
    command_id = payload.get("id")
    method = payload.get("method")
    params = payload.get("params")

    if not isinstance(command_id, int) or not isinstance(method, str):
        return build_error(
            command_id if isinstance(command_id, int) else None,
            "invalid_argument",
            "Commands must include an integer id and string method.",
        ), None

    if not isinstance(params, dict):
        return build_error(command_id, "invalid_argument", "params must be an object."), None

    # Run preparation raises HTTPException (unknown assistant/graph, bad resume)
    # and RunCreate raises ValidationError on malformed params. Map both to a
    # protocol error envelope so the client never sees FastAPI's {detail: ...}.
    try:
        if method == "run.start":
            return await _run_start(command_id, params, session=session, thread_id=thread_id, user=user)
        if method == "input.respond":
            return await _input_respond(command_id, params, session=session, thread_id=thread_id, user=user)
    except HTTPException as exc:
        return build_error(command_id, _status_to_error_code(exc.status_code), str(exc.detail)), None
    except ValidationError as exc:
        return build_error(command_id, "invalid_argument", str(exc.errors()[0].get("msg", "invalid params"))), None

    return build_error(command_id, "not_supported", f"Command {method!r} is not supported."), None


async def _run_start(
    command_id: int,
    params: dict[str, Any],
    *,
    session: AsyncSession,
    thread_id: str,
    user: User,
) -> tuple[dict[str, Any], str | None]:
    """Start a run on the thread from ``RunStartParams``."""
    assistant_id = params.get("assistant_id")
    if not isinstance(assistant_id, str) or not assistant_id:
        return build_error(command_id, "invalid_argument", "run.start requires a string assistant_id."), None

    # No stream_mode: v2 runs stream via the native v3 path, which selects
    # channels through transformers, not stream_mode.
    request = RunCreate(
        assistant_id=assistant_id,
        input=params.get("input"),
        config=params.get("config") or {},
        metadata=params.get("metadata"),
    )
    run_id = await _start(session, thread_id, request, user)
    return build_success(command_id, {"run_id": run_id}), run_id


async def _input_respond(
    command_id: int,
    params: dict[str, Any],
    *,
    session: AsyncSession,
    thread_id: str,
    user: User,
) -> tuple[dict[str, Any], str | None]:
    """Resume an interrupted run by replaying a HITL response as a command."""
    assistant_id = params.get("assistant_id")
    if not isinstance(assistant_id, str) or not assistant_id:
        return build_error(command_id, "invalid_argument", "input.respond requires a string assistant_id."), None
    if "response" not in params:
        return build_error(command_id, "invalid_argument", "input.respond requires a response value."), None

    request = RunCreate(
        assistant_id=assistant_id,
        config=params.get("config") or {},
        command={"resume": params["response"]},
    )
    run_id = await _start(session, thread_id, request, user)
    return build_success(command_id, {"run_id": run_id}), run_id


async def _start(session: AsyncSession, thread_id: str, request: RunCreate, user: User) -> str:
    """Persist + enqueue a run via the shared preparation path (native v3 stream)."""
    run_id, _run, _job = await _prepare_run(
        session, thread_id, request, user, initial_status="pending", event_streaming_v2=True
    )
    return run_id

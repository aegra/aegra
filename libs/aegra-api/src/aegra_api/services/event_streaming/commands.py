"""Dispatch Agent Protocol v2 thread commands.

Commands are JSON-RPC-style: ``{id, method, params}`` in, a success or
error envelope out. They re-front the existing run machinery — ``run.start``
and ``input.respond`` both build a ``RunCreate`` and go through the same
``_prepare_run`` path the legacy endpoints use, so execution semantics are
identical; only the transport differs.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

import structlog
from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import Thread as ThreadORM
from aegra_api.core.orm import _get_session_maker
from aegra_api.models import User
from aegra_api.models.runs import RunCreate
from aegra_api.services.event_streaming.protocol import ErrorCode, build_error, build_success
from aegra_api.services.run_preparation import (
    _RESUME_SETTLE_ATTEMPTS,
    _RESUME_SETTLE_INTERVAL_SECONDS,
    _prepare_run,
)

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
    # Anything else becomes unknown_error — an RPC response must stay an envelope.
    try:
        if method == "run.start":
            return await _run_start(command_id, params, session=session, thread_id=thread_id, user=user)
        if method == "input.respond":
            return await _input_respond(command_id, params, session=session, thread_id=thread_id, user=user)
    except HTTPException as exc:
        return build_error(command_id, _status_to_error_code(exc.status_code), str(exc.detail)), None
    except ValidationError as exc:
        return build_error(command_id, "invalid_argument", str(exc.errors()[0].get("msg", "invalid params"))), None
    except Exception:
        logger.exception("Unhandled error dispatching v2 command", method=method, thread_id=thread_id)
        return build_error(command_id, "unknown_error", f"Command {method!r} failed unexpectedly."), None

    return build_error(command_id, "unknown_command", f"Unknown command {method!r}."), None


_MULTITASK_STRATEGIES = frozenset({"reject", "rollback", "interrupt", "enqueue"})


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

    multitask = params.get("multitaskStrategy", params.get("multitask_strategy"))
    if multitask is not None and multitask not in _MULTITASK_STRATEGIES:
        return build_error(command_id, "invalid_argument", f"Unknown multitaskStrategy {multitask!r}."), None

    # run.start with input on an interrupted thread means "answer the pending
    # interrupt" — resume with the input instead of starting a fresh turn that
    # would discard the pending tasks.
    input_data = params.get("input")
    command: dict[str, Any] | None = None
    if input_data is not None and await _thread_interrupted_with_settle(session, thread_id, user):
        command = {"resume": input_data}
        input_data = None

    # No stream_mode: v2 runs stream via the native v3 path, which selects
    # channels through transformers, not stream_mode. interrupt_before/after are
    # forwarded so v2 clients can set node-level HITL breakpoints like v1.
    # applied_through_seq is 0 on this stateless POST transport (no stream binding).
    request = RunCreate(
        assistant_id=assistant_id,
        input=input_data,
        command=command,
        config=params.get("config") or {},
        metadata=params.get("metadata"),
        interrupt_before=params.get("interrupt_before"),
        interrupt_after=params.get("interrupt_after"),
        multitask_strategy=multitask,
    )
    run_id = await _start(session, thread_id, request, user)
    return build_success(command_id, {"run_id": run_id}, applied_through_seq=0), run_id


async def _thread_is_interrupted(session: AsyncSession, thread_id: str, user: User) -> bool:
    status = await session.scalar(
        select(ThreadORM.status).where(ThreadORM.thread_id == thread_id, ThreadORM.user_id == user.identity)
    )
    return status == "interrupted"


async def _thread_interrupted_with_settle(session: AsyncSession, thread_id: str, user: User) -> bool:
    """Interrupt check for run.start input classification, settle-aware.

    The interrupt event reaches the client (via the broker) before finalize_run
    commits thread_status='interrupted' (see run_preparation's settle note). A client
    answering the instant it sees the interrupt would be misclassified as fresh input
    and parked behind the pause forever — so when a run is still in flight, poll fresh
    sessions for the commit before deciding.
    """
    if await _thread_is_interrupted(session, thread_id, user):
        return True
    in_flight = await session.scalar(
        select(RunORM.run_id)
        .where(
            RunORM.thread_id == thread_id,
            RunORM.user_id == user.identity,
            RunORM.status.in_(("running", "pending")),
        )
        .limit(1)
    )
    if in_flight is None:
        return False
    maker = _get_session_maker()
    thread_stmt = select(ThreadORM.status).where(ThreadORM.thread_id == thread_id, ThreadORM.user_id == user.identity)
    for _ in range(_RESUME_SETTLE_ATTEMPTS):
        await asyncio.sleep(_RESUME_SETTLE_INTERVAL_SECONDS)
        async with maker() as fresh:
            status = await fresh.scalar(thread_stmt)
        if status == "interrupted":
            return True
        if status != "busy":
            # Settled to idle/error: the in-flight run finished without pausing,
            # so this input is a genuine fresh turn.
            return False
    return False


# langgraph interrupt ids are xxh3-128 hexdigests; a resume map is only
# recognized as id-targeted when every key matches this shape.
_INTERRUPT_ID_PATTERN = re.compile(r"[0-9a-f]{32}")


async def _input_respond(
    command_id: int,
    params: dict[str, Any],
    *,
    session: AsyncSession,
    thread_id: str,
    user: User,
) -> tuple[dict[str, Any], str | None]:
    """Resume an interrupted run by replaying a HITL response as a command.

    The stock SDK's ``input.respond`` sends only ``{interrupt_id, namespace,
    response}`` — no assistant. Recover the assistant from the thread's most
    recent run so a resume works without the client re-supplying it. A supplied
    ``interrupt_id`` targets that interrupt via an id-keyed resume map (the only
    form that works with multiple pending interrupts); the batch ``responses``
    form merges several targets into one resume.
    """
    resume, error = _build_resume(params)
    if error is not None:
        code, message = error
        return build_error(command_id, code, message), None

    assistant_id = params.get("assistant_id")
    if not isinstance(assistant_id, str) or not assistant_id:
        assistant_id = await _thread_assistant_id(session, thread_id, user)
    if not assistant_id:
        return build_error(command_id, "no_such_run", "No run on this thread to resume."), None

    request = RunCreate(
        assistant_id=assistant_id,
        config=params.get("config") or {},
        metadata=params.get("metadata"),
        command={"resume": resume},
    )
    run_id = await _start(session, thread_id, request, user)
    return build_success(command_id, {"run_id": run_id}, applied_through_seq=0), run_id


def _build_resume(params: dict[str, Any]) -> tuple[Any, tuple[ErrorCode, str] | None]:
    """Build the resume value from single or batch input.respond params.

    Returns ``(resume, None)`` or ``(None, (error_code, message))``.
    """
    responses = params.get("responses")
    if isinstance(responses, list):
        resume_map: dict[str, Any] = {}
        for entry in responses:
            if not isinstance(entry, dict) or "response" not in entry:
                return None, ("invalid_argument", "Each responses entry requires interrupt_id and response.")
            interrupt_id = entry.get("interrupt_id")
            if not isinstance(interrupt_id, str) or not _INTERRUPT_ID_PATTERN.fullmatch(interrupt_id):
                return None, ("no_such_interrupt", f"Unknown interrupt id {interrupt_id!r}.")
            resume_map[interrupt_id] = entry["response"]
        if not resume_map:
            return None, ("invalid_argument", "responses must be a non-empty array.")
        return resume_map, None

    if "response" not in params:
        return None, ("invalid_argument", "input.respond requires a response value.")

    interrupt_id = params.get("interrupt_id")
    if interrupt_id is None:
        # Untargeted resume: valid only while a single interrupt is pending.
        return params["response"], None
    if not isinstance(interrupt_id, str) or not _INTERRUPT_ID_PATTERN.fullmatch(interrupt_id):
        return None, ("no_such_interrupt", f"Unknown interrupt id {interrupt_id!r}.")
    return {interrupt_id: params["response"]}, None


async def _thread_assistant_id(session: AsyncSession, thread_id: str, user: User) -> str | None:
    """The assistant bound to the thread's most recent run, user-scoped."""
    return await session.scalar(
        select(RunORM.assistant_id)
        .where(RunORM.thread_id == thread_id, RunORM.user_id == user.identity)
        .order_by(RunORM.created_at.desc())
        .limit(1)
    )


async def _start(session: AsyncSession, thread_id: str, request: RunCreate, user: User) -> str:
    """Persist + enqueue a run via the shared preparation path (native v3 stream)."""
    run_id, _run, _job = await _prepare_run(
        session, thread_id, request, user, initial_status="pending", event_streaming_v2=True
    )
    return run_id

"""Orchestrate Open Responses calls on top of threads and runs."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

import structlog
from fastapi import HTTPException
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from aegra_api.api.runs import _apply_create_run_auth
from aegra_api.api.threads import _resolve_ttl_row, get_thread_state
from aegra_api.core.auth_filters import build_metadata_filter
from aegra_api.core.auth_handlers import build_auth_context, handle_event
from aegra_api.core.orm import Assistant as AssistantORM
from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import Thread as ThreadORM
from aegra_api.core.orm import _get_session_maker
from aegra_api.models import Run, RunCreate, User
from aegra_api.models.responses import ConversationCreate, ResponseCreate, ResponsesError
from aegra_api.models.run_job import RunBehavior, RunExecution
from aegra_api.services.broker import broker_manager
from aegra_api.services.executor import executor
from aegra_api.services.langgraph_service import get_langgraph_service
from aegra_api.services.responses_mapping import (
    PreparedTurn,
    bind_resume,
    build_responses_meta,
    conversation_id_of,
    is_ephemeral,
    meta_from_params,
    prepare_turn,
    project_conversation_items,
    project_response,
    public_thread_metadata,
    retains_stateless_thread,
    threads_conflict,
)
from aegra_api.services.responses_stream import iter_responses_sse
from aegra_api.services.run_cleanup import delete_thread_by_id
from aegra_api.services.run_preparation import _prepare_run
from aegra_api.services.run_status import ACTIVE_RUN_STATES
from aegra_api.services.streaming_service import parse_broker_event, streaming_service
from aegra_api.settings import settings
from aegra_api.utils.assistants import resolve_assistant_id

logger = structlog.getLogger(__name__)

_RESPONSE_STREAM_MODES = ["messages", "values"]
# Same window as run resume validation: the interrupt event is published
# before the executor commits thread and run status.
_INTERRUPT_SETTLE_ATTEMPTS = 10
_INTERRUPT_SETTLE_INTERVAL_SECONDS = 0.1


@dataclass(frozen=True)
class ResolvedThread:
    """Thread a response will run on."""

    thread_id: str
    ephemeral: bool
    thread: ThreadORM | None
    minted: bool


@dataclass(frozen=True)
class StartedRun:
    """A run that the Responses stream still has to follow."""

    run: Run
    payload: dict[str, Any]
    ephemeral: bool
    thread_id: str
    user_id: str
    cancel_on_disconnect: bool
    message_offset: int


async def load_owned_run(session: AsyncSession, run_id: str, user_id: str) -> RunORM | None:
    """Load a run only when it belongs to ``user_id``."""
    return await session.scalar(
        select(RunORM).where(
            RunORM.run_id == run_id,
            RunORM.user_id == user_id,
        )
    )


async def load_owned_thread(
    session: AsyncSession,
    thread_id: str,
    user_id: str,
    filters: dict[str, Any] | None = None,
) -> ThreadORM | None:
    """Load a thread only when it belongs to ``user_id`` and passes handler filters."""
    stmt = select(ThreadORM).where(
        ThreadORM.thread_id == thread_id,
        ThreadORM.user_id == user_id,
    )
    auth_filter = build_metadata_filter(ThreadORM.metadata_json, filters)
    if auth_filter is not None:
        stmt = stmt.where(auth_filter)
    return await session.scalar(stmt)


async def load_visible_response(
    session: AsyncSession,
    user: User,
    response_id: str,
    *,
    action: Literal["read", "update"],
) -> RunORM | None:
    """Load an owned run, then its thread under the handler filters for ``action``."""
    run = await load_owned_run(session, response_id, user.identity)
    if run is None:
        return None
    filters = await handle_event(
        build_auth_context(user, "threads", action),
        {"thread_id": run.thread_id},
    )
    thread = await load_owned_thread(
        session,
        run.thread_id,
        user.identity,
        filters if isinstance(filters, dict) else None,
    )
    if thread is None:
        return None
    return run


def project_run(run: RunORM) -> dict[str, Any]:
    """Project a run row into a Response object."""
    output = run.output if isinstance(run.output, dict) else None
    return project_response(
        response_id=run.run_id,
        run_status=run.status,
        output=output,
        error_message=run.error_message,
        created_at=run.created_at,
        assistant_id=run.assistant_id,
        meta=meta_from_params(run.execution_params, run.assistant_id),
    )


async def execute_response(user: User, body: ResponseCreate) -> dict[str, Any] | StartedRun:
    """Create a response, wait for it, or hand back a run the caller will stream.

    Preparation uses a short-lived session. The wait happens after that session
    closes so a synchronous response does not hold a pool connection.
    """
    maker = _get_session_maker()
    async with maker() as session:
        turn = prepare_turn(body)
        resolved = await resolve_thread(session, user, body)
        await _authorize_thread(user, resolved)

        if turn.interrupt_id is not None:
            pending = await _await_pending_interrupts(resolved.thread_id, user.identity, turn.interrupt_id)
        else:
            pending = []

        if turn.rejected:
            if resolved.thread is not None:
                await session.refresh(resolved.thread)
            if resolved.thread is None or resolved.thread.status != "interrupted":
                raise HTTPException(409, "Cannot reject an approval when the conversation is not interrupted")
            await _authorize_run(user, resolved.thread_id, body.model)
            return await _persist_rejection(
                session, user, body, resolved.thread, reason=turn.reject_reason or "interrupt rejected"
            )

        resume = bind_resume(turn.interrupt_id, turn.resume, pending) if turn.interrupt_id else turn.resume
        request = _run_create(body, turn, resume=resume)
        await _apply_create_run_auth(user, resolved.thread_id, request, session)
        message_offset = 0 if resolved.minted else await _message_offset(resolved.thread_id, user)
        meta = build_responses_meta(body, conversation_id=conversation_id_of(body.conversation))
        meta["message_offset"] = message_offset
        run_id, run_model, _job = await _prepare_run(
            session,
            resolved.thread_id,
            request,
            user,
            initial_status="pending",
            responses_meta=meta,
        )
        orm = await load_owned_run(session, run_id, user.identity)
        if orm is None:
            raise HTTPException(500, f"Response '{run_id}' was not persisted")
        payload = project_run(orm)
        ephemeral = resolved.ephemeral
        thread_id = resolved.thread_id

    if body.stream:
        return StartedRun(
            run=run_model,
            payload=payload,
            ephemeral=ephemeral,
            thread_id=thread_id,
            user_id=user.identity,
            cancel_on_disconnect=not body.background,
            message_offset=message_offset,
        )
    if body.background:
        return payload

    fresh: RunORM | None = None
    try:
        try:
            await executor.wait_for_completion(run_id, timeout=float(settings.worker.BG_JOB_TIMEOUT_SECS))
        except TimeoutError:
            logger.warning("responses wait timed out", run_id=run_id)
            await streaming_service.cancel_run(run_id)
            fresh = await _reload_run(run_id, user.identity)
            return await _timeout_payload(run_id, user.identity)
        fresh = await _reload_run(run_id, user.identity)
        if fresh is None:
            raise HTTPException(404, f"Response '{run_id}' not found")
        return project_run(fresh)
    finally:
        if ephemeral and not _retains_run(fresh):
            await asyncio.shield(delete_thread_by_id(thread_id, user.identity))


async def iter_response_sse(started: StartedRun, *, starting_after: str | None) -> AsyncGenerator[str, None]:
    """Follow a started run and delete a stateless thread when the stream ends."""
    try:
        async for frame in iter_responses_sse(
            started.run,
            started.payload,
            starting_after=starting_after,
            message_offset=started.message_offset,
        ):
            yield frame
    finally:
        if started.ephemeral:
            fresh = await _reload_run(started.run.run_id, started.user_id)
            if not _retains_run(fresh):
                await asyncio.shield(delete_thread_by_id(started.thread_id, started.user_id))


async def resolve_thread(session: AsyncSession, user: User, body: ResponseCreate) -> ResolvedThread:
    """Pick the thread from a conversation, a previous response, or a new id."""
    conversation_id = conversation_id_of(body.conversation)
    previous_thread_id: str | None = None
    if body.previous_response_id:
        previous = await load_owned_run(session, body.previous_response_id, user.identity)
        if previous is None:
            raise HTTPException(404, f"Response '{body.previous_response_id}' not found")
        previous_thread_id = previous.thread_id
    if threads_conflict(conversation_id, previous_thread_id):
        raise HTTPException(422, "conversation and previous_response_id refer to different threads")

    thread_id = conversation_id or previous_thread_id
    if thread_id is None:
        return ResolvedThread(
            thread_id=str(uuid4()),
            ephemeral=is_ephemeral(store=body.store, conversation_id=None, previous_response_id=None),
            thread=None,
            minted=True,
        )

    thread = await _thread_visible_to(session, user, thread_id)
    if thread is None:
        if conversation_id:
            raise HTTPException(404, f"Conversation '{thread_id}' not found")
        raise HTTPException(404, f"Thread for response '{body.previous_response_id}' not found")
    return ResolvedThread(thread_id=thread.thread_id, ephemeral=False, thread=thread, minted=False)


async def create_conversation(session: AsyncSession, user: User, body: ConversationCreate) -> dict[str, Any]:
    """Create an empty thread and return it as a conversation."""
    # Snapshot first. Handlers mutate this dict in place, and injected keys must
    # stay out of the client metadata that is copied under responses_metadata.
    client_metadata = dict(body.metadata) if body.metadata else None
    value = body.model_dump()
    filters = await handle_event(build_auth_context(user, "threads", "create"), value)
    now = datetime.now(UTC)
    thread_id = str(uuid4())
    metadata = _conversation_metadata(client_metadata, filters, value, owner=user.identity)
    thread = ThreadORM(
        thread_id=thread_id,
        status="idle",
        metadata_json=metadata,
        user_id=user.identity,
        created_at=now,
        updated_at=now,
    )
    session.add(thread)
    ttl_row = _resolve_ttl_row(thread_id, None)
    if ttl_row is not None:
        session.add(ttl_row)
    await session.commit()
    return _conversation_body(thread_id, now, public_thread_metadata(metadata))


async def read_conversation(session: AsyncSession, user: User, conversation_id: str) -> dict[str, Any]:
    """Return one conversation owned by the caller."""
    thread = await _thread_visible_to(session, user, conversation_id)
    if thread is None:
        raise HTTPException(404, f"Conversation '{conversation_id}' not found")
    return _conversation_body(thread.thread_id, thread.created_at, public_thread_metadata(thread.metadata_json))


async def list_conversation_items(session: AsyncSession, user: User, conversation_id: str) -> list[dict[str, Any]]:
    """Read thread checkpoint messages as conversation items."""
    thread = await _thread_visible_to(session, user, conversation_id)
    if thread is None:
        raise HTTPException(404, f"Conversation '{conversation_id}' not found")
    try:
        state = await get_thread_state(conversation_id, False, None, user)
    except HTTPException as exc:
        if exc.status_code == 404:
            return []
        raise
    values = state.values if isinstance(state.values, dict) else {}
    messages = values.get("messages")
    if not isinstance(messages, list):
        return []
    return project_conversation_items(messages)


async def load_assistant(session: AsyncSession, user: User, model: str) -> AssistantORM:
    """Resolve ``model`` to an assistant the caller can run."""
    graphs = get_langgraph_service().list_graphs()
    assistant_id = resolve_assistant_id(model, graphs)
    assistant = await session.scalar(
        select(AssistantORM).where(
            AssistantORM.assistant_id == assistant_id,
            or_(AssistantORM.user_id == user.identity, AssistantORM.user_id == "system"),
        )
    )
    if assistant is None:
        raise HTTPException(404, f"Assistant '{model}' not found")
    return assistant


async def _authorize_thread(user: User, resolved: ResolvedThread) -> None:
    if not resolved.minted:
        return
    await handle_event(build_auth_context(user, "threads", "create"), {"thread_id": resolved.thread_id})


async def _authorize_run(user: User, thread_id: str, assistant_id: str) -> None:
    await handle_event(
        build_auth_context(user, "threads", "create_run"),
        {"thread_id": thread_id, "assistant_id": assistant_id},
    )
    await handle_event(build_auth_context(user, "assistants", "read"), {"assistant_id": assistant_id})


async def _persist_rejection(
    session: AsyncSession,
    user: User,
    body: ResponseCreate,
    thread: ThreadORM,
    *,
    reason: str,
) -> dict[str, Any]:
    assistant = await load_assistant(session, user, body.model)
    now = datetime.now(UTC)
    run_id = str(uuid4())
    meta = build_responses_meta(
        body,
        conversation_id=conversation_id_of(body.conversation),
        error_code="interrupt_rejected",
    )
    run = RunORM(
        run_id=run_id,
        thread_id=thread.thread_id,
        assistant_id=assistant.assistant_id,
        status="error",
        input=None,
        output={},
        error_message=reason,
        config={},
        context={},
        user_id=user.identity,
        created_at=now,
        updated_at=now,
        execution_params={
            "graph_id": assistant.graph_id,
            "user": user.model_dump(),
            "execution": RunExecution().model_dump(),
            "behavior": RunBehavior().model_dump(),
            "run_metadata": {},
            "responses_meta": meta,
        },
    )
    session.add(run)
    await session.commit()
    return project_run(run)


def _run_create(body: ResponseCreate, turn: PreparedTurn, *, resume: Any) -> RunCreate:
    if turn.interrupt_id is not None:
        return RunCreate(
            assistant_id=body.model,
            input=None,
            command={"resume": resume},
            stream_mode=_RESPONSE_STREAM_MODES,
        )
    return RunCreate(
        assistant_id=body.model,
        input=turn.messages,
        stream_mode=_RESPONSE_STREAM_MODES,
    )


async def _thread_visible_to(session: AsyncSession, user: User, thread_id: str) -> ThreadORM | None:
    """Load a thread only when ``threads.read`` allows it, including metadata filters."""
    filters = await handle_event(build_auth_context(user, "threads", "read"), {"thread_id": thread_id})
    handler_filters = filters if isinstance(filters, dict) else None
    return await load_owned_thread(session, thread_id, user.identity, handler_filters)


def _conversation_metadata(
    requested: dict[str, Any] | None,
    filters: dict[str, Any] | None,
    value: dict[str, Any],
    *,
    owner: str,
) -> dict[str, Any]:
    """Thread metadata, keeping handler-injected keys beside the public conversation metadata."""
    public = {key: item for key, item in (requested or {}).items() if isinstance(key, str) and isinstance(item, str)}
    handler_meta: dict[str, Any] = {}
    if isinstance(filters, dict) and isinstance(filters.get("metadata"), dict):
        handler_meta = filters["metadata"]
    elif isinstance(value.get("metadata"), dict):
        handler_meta = value["metadata"]
    metadata: dict[str, Any] = {
        "assistant_id": None,
        "graph_id": None,
        "thread_name": "",
        "responses_metadata": dict(public),
    }
    for key, item in handler_meta.items():
        if key in public or key == "owner":
            continue
        metadata[key] = item
    metadata["owner"] = owner
    return metadata


def _retains_run(run: RunORM | None) -> bool:
    if run is None:
        return False
    output = run.output if isinstance(run.output, dict) else None
    return retains_stateless_thread(run.status, output)


async def _message_offset(thread_id: str, user: User) -> int:
    """How many checkpoint messages already exist, so this run's output can skip them."""
    try:
        state = await get_thread_state(thread_id, False, None, user)
    except HTTPException as exc:
        if exc.status_code == 404:
            return 0
        raise
    values = state.values if isinstance(state.values, dict) else {}
    messages = values.get("messages")
    if not isinstance(messages, list):
        return 0
    return len(messages)


async def _await_pending_interrupts(thread_id: str, user_id: str, interrupt_id: str) -> list[str]:
    """Read pending interrupts, retrying while the executor's commit is still in flight."""
    pending = await _pending_on_fresh_session(thread_id, user_id)
    if interrupt_id in pending:
        return pending
    for _ in range(_INTERRUPT_SETTLE_ATTEMPTS):
        await asyncio.sleep(_INTERRUPT_SETTLE_INTERVAL_SECONDS)
        pending = await _pending_on_fresh_session(thread_id, user_id)
        if interrupt_id in pending:
            return pending
    raise ResponsesError(409, f"Interrupt '{interrupt_id}' is not pending")


async def _pending_on_fresh_session(thread_id: str, user_id: str) -> list[str]:
    maker = _get_session_maker()
    async with maker() as session:
        return await pending_interrupt_ids(session, thread_id, user_id)


async def pending_interrupt_ids(session: AsyncSession, thread_id: str, user_id: str) -> list[str]:
    """Ids the caller can resume. The broker has them before the run row commits."""
    interrupted = await session.scalar(
        select(RunORM)
        .where(
            RunORM.thread_id == thread_id,
            RunORM.user_id == user_id,
            RunORM.status == "interrupted",
        )
        .order_by(RunORM.created_at.desc())
        .limit(1)
    )
    if interrupted is not None:
        ids = _interrupt_ids_from_output(interrupted.output)
        if not ids:
            ids = await _interrupt_ids_from_broker(interrupted.run_id)
        if ids:
            return ids

    active = await session.scalar(
        select(RunORM)
        .where(
            RunORM.thread_id == thread_id,
            RunORM.user_id == user_id,
            RunORM.status.in_(ACTIVE_RUN_STATES),
        )
        .order_by(RunORM.created_at.desc())
        .limit(1)
    )
    if active is None:
        return []
    return await _interrupt_ids_from_broker(active.run_id)


def _interrupt_ids_from_output(output: Any) -> list[str]:
    if not isinstance(output, dict):
        return []
    raw = output.get("__interrupt__")
    if not isinstance(raw, list):
        return []
    return [entry["id"] for entry in raw if isinstance(entry, dict) and isinstance(entry.get("id"), str)]


async def _interrupt_ids_from_broker(run_id: str) -> list[str]:
    """Interrupt ids streamed for this run, including another instance's Redis history."""
    broker = broker_manager.get_or_create_broker(run_id)
    found: list[str] = []
    for _event_id, raw_event in await broker.replay(None):
        parsed = parse_broker_event(raw_event)
        if parsed is None:
            continue
        event_type, payload = parsed
        if event_type != "values" and not event_type.startswith("values"):
            continue
        for interrupt_id in _interrupt_ids_from_output(payload):
            if interrupt_id not in found:
                found.append(interrupt_id)
    return found


async def _reload_run(run_id: str, user_id: str) -> RunORM | None:
    maker = _get_session_maker()
    async with maker() as session:
        return await load_owned_run(session, run_id, user_id)


async def _timeout_payload(run_id: str, user_id: str) -> dict[str, Any]:
    fresh = await _reload_run(run_id, user_id)
    if fresh is None:
        raise HTTPException(404, f"Response '{run_id}' not found")
    payload = project_run(fresh)
    payload["status"] = "failed"
    payload["error"] = {
        "code": "timeout",
        "message": f"Response '{run_id}' did not finish within the wait timeout",
    }
    return payload


def _conversation_body(thread_id: str, created_at: datetime, metadata: dict[str, str]) -> dict[str, Any]:
    return {
        "id": thread_id,
        "object": "conversation",
        "created_at": int(created_at.timestamp()),
        "metadata": metadata,
    }

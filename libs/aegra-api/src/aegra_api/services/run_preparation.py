"""Run preparation logic extracted from api/runs.py.

Contains the shared run-creation helper, thread metadata updates,
resume-command validation, and config/context merging logic.
"""

import asyncio
import hashlib
import json
from datetime import UTC, datetime
from typing import Any
from uuid import NAMESPACE_URL, uuid4, uuid5

import structlog
from asgi_correlation_id import correlation_id
from fastapi import HTTPException
from sqlalchemy import or_, select, text, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from aegra_api.core.orm import Assistant as AssistantORM
from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import Thread as ThreadORM
from aegra_api.core.orm import _get_session_maker
from aegra_api.models import Run, RunCreate, User
from aegra_api.models.run_job import RunBehavior, RunExecution, RunIdentity, RunJob
from aegra_api.services.executor import executor
from aegra_api.services.langgraph_service import get_langgraph_service
from aegra_api.services.run_status import set_thread_status
from aegra_api.services.streaming_service import streaming_service
from aegra_api.utils.assistants import resolve_assistant_id
from aegra_api.utils.run_utils import _merge_jsonb

logger = structlog.getLogger(__name__)


# The interrupt reaches the client (via the broker/SSE) before the run executor
# commits thread_status="interrupted" in finalize_run. A client that resumes the
# instant it sees the interrupt can beat that commit, so after the first read poll
# a few times on FRESH sessions (each a new snapshot that sees the commit) before
# rejecting — otherwise a valid resume races to a spurious 400.
_RESUME_SETTLE_ATTEMPTS = 10
_RESUME_SETTLE_INTERVAL_SECONDS = 0.1
_INTERRUPT_SETTLE_ATTEMPTS = 100
_INTERRUPT_SETTLE_INTERVAL_SECONDS = 0.1
_TERMINAL_RUN_STATUSES = frozenset({"success", "error", "timeout", "interrupted"})


def _idempotent_run_id(user_id: str, thread_id: str, key: str) -> str:
    return str(uuid5(NAMESPACE_URL, json.dumps([user_id, thread_id, key], separators=(",", ":"))))


def _request_fingerprint(request: RunCreate) -> str:
    payload = json.dumps(request.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


async def _validate_resume_command(session: AsyncSession, thread_id: str, command: dict[str, Any] | None) -> None:
    """Validate resume command requirements.

    Guard on the presence of the ``resume`` key, not a non-None value: ``None``
    is a valid resume payload, and a ``{"resume": None}`` command must still be
    rejected against a thread that is not interrupted.
    """
    if not command or "resume" not in command:
        return

    thread_stmt = select(ThreadORM).where(ThreadORM.thread_id == thread_id)
    thread = await session.scalar(thread_stmt)
    if not thread:
        raise HTTPException(404, f"Thread '{thread_id}' not found")
    if thread.status == "interrupted":
        return

    # Not interrupted on the request session's snapshot — poll fresh sessions in
    # case finalize_run's commit is still in flight.
    maker = _get_session_maker()
    for _ in range(_RESUME_SETTLE_ATTEMPTS):
        await asyncio.sleep(_RESUME_SETTLE_INTERVAL_SECONDS)
        async with maker() as fresh:
            fresh_thread = await fresh.scalar(thread_stmt)
        if fresh_thread is not None and fresh_thread.status == "interrupted":
            return

    raise HTTPException(400, "Cannot resume: thread is not in interrupted state")


_THREAD_NAME_MAX_LENGTH = 100


def _resolve_content_text(content: Any) -> str:
    """Extract plain text from a message content field.

    Handles both plain strings and list-of-blocks format used by some SDKs::

        [{"type": "text", "text": "Hello world"}]
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if isinstance(text, str):
                    parts.append(text)
        return " ".join(parts)
    return ""


def _extract_thread_name(input_data: dict[str, Any]) -> str:
    """Derive a thread name from the first human message in the run input.

    Supports the common ``{"messages": [{"role": "human", "content": "..."}]}``
    shape emitted by agent-chat-ui, LangGraph Studio, and the JS/Python SDKs.
    Also handles list-of-blocks content from OpenAI-compatible APIs.
    Returns an empty string when no suitable message is found.
    """
    messages = input_data.get("messages")
    if not isinstance(messages, list) or not messages:
        return ""
    _HUMAN_ROLES = {"human", "user"}
    for msg in messages:
        raw_content: Any = None
        role: str | None = None
        if isinstance(msg, dict):
            role_val = msg.get("role")
            role = role_val if isinstance(role_val, str) else None
            if role is None:
                type_val = msg.get("type")
                role = type_val if isinstance(type_val, str) else None
            raw_content = msg.get("content")
        elif hasattr(msg, "content"):
            raw_content = getattr(msg, "content", None)
            msg_type = getattr(msg, "type", None)
            role = msg_type if isinstance(msg_type, str) else None
        if role not in _HUMAN_ROLES:
            continue
        text = _resolve_content_text(raw_content)
        if text.strip():
            name = text.strip()
            if len(name) > _THREAD_NAME_MAX_LENGTH:
                return name[:_THREAD_NAME_MAX_LENGTH].rsplit(" ", 1)[0] + "..."
            return name
    return ""


async def update_thread_metadata(
    session: AsyncSession,
    thread_id: str,
    assistant_id: str,
    graph_id: str,
    *,
    user_id: str | None = None,
    input_data: dict[str, Any] | None = None,
) -> None:
    """Update thread metadata with assistant and graph information (dialect agnostic).

    If thread doesn't exist, auto-creates it.
    When *input_data* is provided and the thread has no name yet, the first
    human message content is used as ``thread_name``.
    Does NOT commit — the caller controls the transaction boundary.
    """
    # Read-modify-write to avoid DB-specific JSON concat operators
    thread = await session.scalar(select(ThreadORM).where(ThreadORM.thread_id == thread_id))

    thread_name = _extract_thread_name(input_data or {})

    if not thread:
        # Auto-create thread if it doesn't exist
        if not user_id:
            raise HTTPException(400, "Cannot auto-create thread: user_id is required")

        metadata = {
            "owner": user_id,
            "assistant_id": str(assistant_id),
            "graph_id": graph_id,
            "thread_name": thread_name,
        }

        thread_orm = ThreadORM(
            thread_id=thread_id,
            status="idle",
            metadata_json=metadata,
            user_id=user_id,
        )
        session.add(thread_orm)
        return

    md = dict(getattr(thread, "metadata_json", {}) or {})
    md.update(
        {
            "assistant_id": str(assistant_id),
            "graph_id": graph_id,
        }
    )
    # Only set thread_name if empty and we have a name from the input
    if thread_name and not md.get("thread_name"):
        md["thread_name"] = thread_name
    await session.execute(
        update(ThreadORM).where(ThreadORM.thread_id == thread_id).values(metadata_json=md, updated_at=datetime.now(UTC))
    )


async def _prepare_run(
    session: AsyncSession,
    thread_id: str,
    request: RunCreate,
    user: User,
    *,
    initial_status: str,
    event_streaming_v2: bool = False,
    idempotency_key: str | None = None,
    openswe_background_admission: bool = False,
) -> tuple[str, Run, RunJob]:
    """Shared run-creation logic used by create, stream, and wait endpoints.

    Validates inputs, resolves the assistant, persists the RunORM record,
    builds a RunJob, submits it to the executor, and returns the triple
    ``(run_id, run_model, job)``.
    """
    request_fingerprint = _request_fingerprint(request) if idempotency_key is not None else None
    run_id = (
        _idempotent_run_id(user.identity, thread_id, idempotency_key) if idempotency_key is not None else str(uuid4())
    )
    if openswe_background_admission and (idempotency_key is not None or request.multitask_strategy == "interrupt"):
        await session.execute(
            text("SELECT pg_advisory_xact_lock(hashtextextended(:scope, 0))"),
            {"scope": f"{len(user.identity)}:{user.identity}{thread_id}"},
        )
    if request_fingerprint is not None:
        existing = await session.scalar(
            select(RunORM).where(
                RunORM.run_id == run_id,
                RunORM.thread_id == thread_id,
                RunORM.user_id == user.identity,
            )
        )
        if existing is not None:
            if (existing.execution_params or {}).get("idempotency_fingerprint") != request_fingerprint:
                raise HTTPException(409, "Idempotency-Key was already used with a different request")
            return run_id, Run.model_validate(existing), RunJob.from_run_orm(existing)

    await _validate_resume_command(session, thread_id, request.command)
    langgraph_service = get_langgraph_service()
    logger.info(
        "Scheduling run",
        run_id=run_id,
        thread_id=thread_id,
        user_id=user.identity,
        status=initial_status,
    )

    # Resolve assistant / graph
    requested_id = str(request.assistant_id)
    available_graphs = langgraph_service.list_graphs()
    resolved_assistant_id = resolve_assistant_id(requested_id, available_graphs)

    # Config / context merging
    config = request.config or {}
    context = request.context or {}
    configurable = config.get("configurable", {})
    if not isinstance(configurable, dict):
        raise HTTPException(status_code=422, detail="`config.configurable` must be a mapping")

    if not context:
        context = configurable.copy()

    assistant_stmt = select(AssistantORM).where(
        AssistantORM.assistant_id == resolved_assistant_id,
        or_(AssistantORM.user_id == user.identity, AssistantORM.user_id == "system"),
    )
    assistant = await session.scalar(assistant_stmt)
    if not assistant:
        raise HTTPException(404, f"Assistant '{request.assistant_id}' not found")

    config = _merge_jsonb(assistant.config, config)
    context = _merge_jsonb(assistant.context, context)

    # Validate the assistant's graph exists
    available_graphs = langgraph_service.list_graphs()
    if assistant.graph_id not in available_graphs:
        raise HTTPException(404, f"Graph '{assistant.graph_id}' not found for assistant")

    if openswe_background_admission and request.multitask_strategy == "interrupt":
        predecessors = (
            await session.scalars(
                select(RunORM).where(
                    RunORM.thread_id == thread_id,
                    RunORM.user_id == user.identity,
                    RunORM.status.in_(("pending", "running")),
                )
            )
        ).all()
        running_ids: list[str] = []
        for predecessor in predecessors:
            status = predecessor.status
            if status == "pending":
                claimed = await session.scalar(
                    update(RunORM)
                    .where(
                        RunORM.run_id == predecessor.run_id,
                        RunORM.thread_id == thread_id,
                        RunORM.user_id == user.identity,
                        RunORM.status == "pending",
                    )
                    .values(status="interrupted", updated_at=datetime.now(UTC))
                    .returning(RunORM.run_id)
                )
                if claimed is not None:
                    continue
                status = await session.scalar(
                    select(RunORM.status).where(
                        RunORM.run_id == predecessor.run_id,
                        RunORM.thread_id == thread_id,
                        RunORM.user_id == user.identity,
                    )
                )
                if status not in ("running", "interrupted"):
                    raise HTTPException(409, f"Active run '{predecessor.run_id}' completed without interruption")
            if status == "running":
                if not await streaming_service.interrupt_run(predecessor.run_id):
                    raise HTTPException(409, f"Could not interrupt active run '{predecessor.run_id}'")
                running_ids.append(predecessor.run_id)
            elif status not in _TERMINAL_RUN_STATUSES:
                raise HTTPException(409, f"Active run '{predecessor.run_id}' changed unexpectedly")
        for _ in range(_INTERRUPT_SETTLE_ATTEMPTS):
            if not running_ids:
                break
            states: dict[str, str] = {}
            rows = await session.execute(
                select(RunORM.run_id, RunORM.status).where(
                    RunORM.run_id.in_(running_ids),
                    RunORM.thread_id == thread_id,
                    RunORM.user_id == user.identity,
                )
            )
            for active_run_id, active_status in rows.all():
                states[active_run_id] = active_status
            if all(states.get(run_id) in _TERMINAL_RUN_STATUSES for run_id in running_ids):
                if any(states.get(run_id) != "interrupted" for run_id in running_ids):
                    raise HTTPException(409, "Active run completed without acknowledging interruption")
                break
            await asyncio.sleep(_INTERRUPT_SETTLE_INTERVAL_SECONDS)
        else:
            raise HTTPException(409, "Timed out waiting for active runs to acknowledge interruption")

    # Mark thread as busy and update metadata
    await update_thread_metadata(
        session, thread_id, assistant.assistant_id, assistant.graph_id, user_id=user.identity, input_data=request.input
    )
    await set_thread_status(session, thread_id, "busy")

    # Build the RunJob before persisting so we can store execution_params
    job = RunJob(
        identity=RunIdentity(run_id=run_id, thread_id=thread_id, graph_id=assistant.graph_id),
        user=user,
        execution=RunExecution(
            input_data=request.input,  # preserve None so LangGraph resumes from checkpoint
            config=config,
            context=context,
            stream_mode=request.stream_mode,
            checkpoint=request.checkpoint,
            command=request.command,
            event_streaming_v2=event_streaming_v2,
        ),
        behavior=RunBehavior(
            interrupt_before=request.interrupt_before,
            interrupt_after=request.interrupt_after,
            multitask_strategy=request.multitask_strategy,
            subgraphs=request.stream_subgraphs or False,
        ),
        run_metadata=request.metadata or {},
    )

    # Persist run record with trace metadata for worker observability.
    # The correlation_id from the HTTP request is stored so workers can
    # link their logs and spans back to the original request.
    exec_params = job.to_execution_params()
    exec_params["trace"] = {
        "correlation_id": correlation_id.get(""),
        "user_id": user.identity,
        "thread_id": thread_id,
        "graph_id": assistant.graph_id,
    }
    if request_fingerprint is not None:
        exec_params["idempotency_fingerprint"] = request_fingerprint

    now = datetime.now(UTC)
    run_orm = RunORM(
        run_id=run_id,
        thread_id=thread_id,
        assistant_id=resolved_assistant_id,
        status=initial_status,
        input=request.input,  # preserve None for checkpoint-only resume; matches RunExecution.input_data
        config=config,
        context=context,
        user_id=user.identity,
        created_at=now,
        updated_at=now,
        output=None,
        error_message=None,
        execution_params=exec_params,
    )
    if request_fingerprint is None:
        session.add(run_orm)
        await session.commit()
    else:
        inserted = await session.scalar(
            insert(RunORM)
            .values(
                run_id=run_id,
                thread_id=thread_id,
                assistant_id=resolved_assistant_id,
                status=initial_status,
                input=request.input,
                config=config,
                context=context,
                user_id=user.identity,
                created_at=now,
                updated_at=now,
                output=None,
                error_message=None,
                execution_params=exec_params,
            )
            .on_conflict_do_nothing(index_elements=[RunORM.run_id])
            .returning(RunORM.run_id)
        )
        if inserted is None:
            await session.rollback()
            existing = await session.scalar(
                select(RunORM).where(
                    RunORM.run_id == run_id,
                    RunORM.thread_id == thread_id,
                    RunORM.user_id == user.identity,
                )
            )
            if existing is None:
                raise RuntimeError("idempotent run disappeared after insertion conflict")
            if (existing.execution_params or {}).get("idempotency_fingerprint") != request_fingerprint:
                raise HTTPException(409, "Idempotency-Key was already used with a different request")
            return run_id, Run.model_validate(existing), RunJob.from_run_orm(existing)
        await session.commit()

    run = Run.model_validate(run_orm)

    # Submit to executor
    await executor.submit(job)
    logger.info("Submitted run to executor", run_id=run_id)

    return run_id, run, job

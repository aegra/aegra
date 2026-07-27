"""Run preparation logic extracted from api/runs.py.

Contains the shared run-creation helper, thread metadata updates,
resume-command validation, and config/context merging logic.
"""

import asyncio
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import structlog
from asgi_correlation_id import correlation_id
from fastapi import HTTPException
from sqlalchemy import or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from aegra_api.core.orm import Assistant as AssistantORM
from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import Thread as ThreadORM
from aegra_api.core.orm import _get_session_maker
from aegra_api.models import Run, RunCreate, User
from aegra_api.models.enums import MULTITASK_DEFAULT
from aegra_api.models.run_job import RunBehavior, RunExecution, RunIdentity, RunJob
from aegra_api.services.executor import executor
from aegra_api.services.langgraph_service import get_langgraph_service
from aegra_api.services.run_status import set_thread_status
from aegra_api.services.streaming_service import streaming_service
from aegra_api.utils.assistants import resolve_assistant_id
from aegra_api.utils.run_utils import _merge_jsonb, map_command_to_langgraph

logger = structlog.getLogger(__name__)


# The interrupt reaches the client (via the broker/SSE) before the run executor
# commits thread_status="interrupted" in finalize_run. A client that resumes the
# instant it sees the interrupt can beat that commit, so after the first read poll
# a few times on FRESH sessions (each a new snapshot that sees the commit) before
# rejecting — otherwise a valid resume races to a spurious 400.
_RESUME_SETTLE_ATTEMPTS = 10
_RESUME_SETTLE_INTERVAL_SECONDS = 0.1


async def _validate_resume_command(
    session: AsyncSession, thread_id: str, command: dict[str, Any] | None, user: User
) -> None:
    """Validate a run's input mode against the thread's interrupt state.

    A command bearing a ``resume`` key requires the thread to be paused, and a
    ``None`` resume payload is rejected outright: LangGraph's ``map_command``
    drops it, so the run would produce no writes and crash the pause to 'error'.
    Symmetrically, a plain fresh-input run (no command) must not land on a thread
    paused at a human-in-the-loop ``interrupt()``: running plain input there
    silently consumes the pending interrupt, so reject and direct the caller to
    resume with a command.
    """
    if command is not None:
        # Reject a malformed command shape up front (e.g. {'goto': [0]}) so it can't bypass the
        # gate and crash to 'error' mid-run — which on a paused thread would corrupt the HITL pause.
        try:
            map_command_to_langgraph(command)
        except (TypeError, KeyError, ValueError, AttributeError) as exc:
            raise HTTPException(status_code=422, detail=f"Invalid command: {exc}") from exc
    # Key presence, not a non-None value: {'resume': None} must be gated as a resume
    # attempt too, or it slips through and errors out mid-run on whatever thread it hits.
    resume_shaped = command is not None and "resume" in command
    # Truthiness matches LangGraph's map_command (`if cmd.update:` / `if cmd.goto:`): an
    # empty container ({'update': {}}, {'goto': []}) produces no writes and would crash a
    # paused thread to 'error', so it must NOT early-return — it falls through to the gate.
    state_op = bool(command and (command.get("update") or command.get("goto")))
    if command is not None and not resume_shaped and state_op:
        # A deliberate update/goto command manipulates graph state directly and is allowed even
        # on a paused thread. Safety here leans on two LangGraph behaviours: empty containers are
        # falsy (so they fall through to the 409 gate, not here), and an unknown goto/Send target
        # is silently ignored rather than raising. If a future LangGraph version raised on unknown
        # targets, such a command could crash mid-run and clobber a HITL pause — gate it here then.
        return

    thread_stmt = select(ThreadORM).where(ThreadORM.thread_id == thread_id, ThreadORM.user_id == user.identity)
    thread = await session.scalar(thread_stmt)
    if resume_shaped:
        if not thread:
            raise HTTPException(404, f"Thread '{thread_id}' not found")
        status = thread.status
        if status != "interrupted":
            # Not interrupted on the request session's snapshot — poll fresh sessions in
            # case finalize_run's commit is still in flight.
            maker = _get_session_maker()
            for _ in range(_RESUME_SETTLE_ATTEMPTS):
                await asyncio.sleep(_RESUME_SETTLE_INTERVAL_SECONDS)
                async with maker() as fresh:
                    fresh_thread = await fresh.scalar(thread_stmt)
                if fresh_thread is not None and fresh_thread.status == "interrupted":
                    status = "interrupted"
                    break
        if status != "interrupted":
            raise HTTPException(400, "Cannot resume: thread is not in interrupted state")
        if command is not None and command.get("resume") is None:
            # map_command drops a None resume; the run would crash the pause to 'error'.
            raise HTTPException(
                409,
                "Thread is paused on a human-in-the-loop interrupt; resume it with "
                "a non-null command={'resume': ...} payload",
            )
        return
    if thread is not None and thread.status == "interrupted":
        raise HTTPException(
            409,
            "Thread is paused on a human-in-the-loop interrupt; resume it with "
            "command={'resume': ...} instead of starting a new run",
        )


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


def _is_resume_run(run: RunORM) -> bool:
    """Whether a run row was created to resume a HITL interrupt (command bears a resume key)."""
    params = run.execution_params or {}
    command = (params.get("execution") or {}).get("command")
    return isinstance(command, dict) and "resume" in command


# A run holds (or is queued for) its thread while in one of these states.
_ACTIVE_RUN_STATUSES = ("running", "pending", "queued")
# Only an actually-dispatched run has a task/worker to cancel.
_CANCELLABLE_RUN_STATUSES = ("running", "pending")
# Terminal states a rollback may target when no run is currently active.
_TERMINAL_RUN_STATUSES = ("interrupted", "error", "success")


async def _apply_multitask_strategy(
    session: AsyncSession, thread_id: str, strategy: str, user: User, *, is_resume: bool = False
) -> tuple[bool, list[str], str | None]:
    """Resolve a new run against the thread's in-flight runs per ``strategy``.

    Locks the thread row ``FOR UPDATE`` (thread-then-run order, matching
    finalize_run) so concurrent creates serialize. Returns ``(should_run,
    cancel_ids, rollback_target_run_id)``: should_run is True to execute now,
    False to park as ``queued``; cancel_ids are runs the caller cancels AFTER
    commit (so the lock is released before the cancel triggers the run's own
    finalize); rollback_target_run_id is the prior run whose checkpoints the
    worker will revert by forking. Raises 409 for reject. interrupt/rollback
    mark the active run interrupted (no rows are deleted — rollback reverts via
    a checkpoint fork, leaving the old branch as harmless siblings).
    """
    await session.execute(
        select(ThreadORM.thread_id)
        .where(ThreadORM.thread_id == thread_id, ThreadORM.user_id == user.identity)
        .with_for_update()
    )
    active = (
        await session.scalars(
            select(RunORM)
            .where(
                RunORM.thread_id == thread_id,
                RunORM.user_id == user.identity,
                RunORM.status.in_(_ACTIVE_RUN_STATUSES),
            )
            .order_by(RunORM.created_at.asc())
        )
    ).all()
    if is_resume:
        # A resume jumps ahead of merely-*queued* runs (it alone can clear a HITL pause), but
        # must still serialize behind a running/pending run. Checked under the FOR UPDATE lock,
        # this rejects a second concurrent resume (which unblocks after the first commits its
        # pending run) so two resumes cannot double-execute on one paused checkpoint.
        if any(run.status in _CANCELLABLE_RUN_STATUSES for run in active):
            raise HTTPException(status_code=409, detail=f"Thread '{thread_id}' already has an active run")
        return True, [], None
    if not active:
        # Idle thread: rollback repairs a broken last turn (an interrupted/errored run can
        # leave an orphaned tool call) but never silently reverts a cleanly completed one.
        if strategy == "rollback":
            last = await session.scalar(
                select(RunORM)
                .where(
                    RunORM.thread_id == thread_id,
                    RunORM.user_id == user.identity,
                    RunORM.status.in_(_TERMINAL_RUN_STATUSES),
                )
                .order_by(RunORM.created_at.desc())
                .limit(1)
            )
            target = last.run_id if last is not None and last.status != "success" else None
            return True, [], target
        return True, [], None

    if strategy == "reject":
        raise HTTPException(status_code=409, detail=f"Thread '{thread_id}' already has an active run")
    if strategy == "enqueue":
        return False, [], None

    # interrupt / rollback: abandon all in-flight work (the active run AND any runs the
    # user double-texted earlier that are still queued), then run the new one now.
    cancel_ids: list[str] = []
    rollback_target: str | None = None
    for run in active:
        if run.status == "queued":
            # Parked behind the active run with no task/broker to cancel: drop it from
            # the queue so it does not execute after the new run (stale double-text).
            run.status = "interrupted"
            continue
        if run.status not in _CANCELLABLE_RUN_STATUSES:
            continue
        if _is_resume_run(run):
            # Cancelling an in-flight resume and running fresh input would land the new
            # run on the pending-interrupt checkpoint, silently consuming the HITL pause.
            raise HTTPException(
                status_code=409,
                detail="Thread has a resume in flight for a pending interrupt; "
                "wait for it to settle instead of pre-empting it",
            )
        cancel_ids.append(run.run_id)
        run.status = "interrupted"
        # Release the lease with the pre-emption so a prod worker whose pub/sub cancel
        # is lost detects lease loss on its next heartbeat and self-cancels the job.
        run.claimed_by = None
        run.lease_expires_at = None
        # The rollback target is the run that actually executed, never a queued one.
        if strategy == "rollback" and rollback_target is None:
            rollback_target = run.run_id
    return True, cancel_ids, rollback_target


async def _prepare_run(
    session: AsyncSession,
    thread_id: str,
    request: RunCreate,
    user: User,
    *,
    initial_status: str,
    event_streaming_v2: bool = False,
) -> tuple[str, Run, RunJob]:
    """Shared run-creation logic used by create, stream, and wait endpoints.

    Validates inputs, resolves the assistant, persists the RunORM record,
    builds a RunJob, submits it to the executor, and returns the triple
    ``(run_id, run_model, job)``.
    """
    await _validate_resume_command(session, thread_id, request.command, user)

    run_id = str(uuid4())
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

    # Mark thread as busy and update metadata
    await update_thread_metadata(
        session, thread_id, assistant.assistant_id, assistant.graph_id, user_id=user.identity, input_data=request.input
    )
    await set_thread_status(session, thread_id, "busy")

    # Resolve double-texting: run now, queue behind the active run, reject, or
    # interrupt/rollback the active run. None defaults to enqueue.
    strategy = request.multitask_strategy or MULTITASK_DEFAULT
    is_resume = bool(request.command and request.command.get("resume") is not None)
    should_run, cancel_ids, rollback_target = await _apply_multitask_strategy(
        session, thread_id, strategy, user, is_resume=is_resume
    )
    run_status = initial_status if should_run else "queued"

    # rollback only makes sense for a fresh dict-input run (not a resume/command
    # or an explicit client checkpoint, which target state themselves).
    rollback_target_run_id = (
        rollback_target
        if strategy == "rollback"
        and rollback_target is not None
        and isinstance(request.input, dict)
        and request.command is None
        and request.checkpoint is None
        else None
    )

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
            rollback_target_run_id=rollback_target_run_id,
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

    now = datetime.now(UTC)
    run_orm = RunORM(
        run_id=run_id,
        thread_id=thread_id,
        assistant_id=resolved_assistant_id,
        status=run_status,
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
    session.add(run_orm)
    await session.commit()

    run = Run.model_validate(run_orm)

    # Cancel pre-empted runs after commit so the thread lock is released before
    # the cancel triggers their finalize (which also locks the thread row). The
    # cancel is fire-and-forget, so a rolled-back active run may write one late
    # checkpoint that briefly out-orders the new run's head in GET state until it
    # stops — harmless: the rollback fork resolves its base by run_id, not by head.
    for cancelled_id in cancel_ids:
        await streaming_service.cancel_run(cancelled_id)

    if should_run:
        await executor.submit(job)
        logger.info("Submitted run to executor", run_id=run_id)
    else:
        logger.info("Run queued behind active run", run_id=run_id, thread_id=thread_id)

    return run_id, run, job

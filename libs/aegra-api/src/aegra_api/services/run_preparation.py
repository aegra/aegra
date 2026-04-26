"""Run preparation logic extracted from api/runs.py.

Contains the shared run-creation helper, thread metadata updates,
resume-command validation, and config/context merging logic.
"""

import time
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import structlog
from asgi_correlation_id import correlation_id
from fastapi import HTTPException
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from aegra_api.core.orm import Assistant as AssistantORM
from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import Thread as ThreadORM
from aegra_api.models import Run, RunCreate, User
from aegra_api.models.run_job import RunBehavior, RunExecution, RunIdentity, RunJob
from aegra_api.services.executor import executor
from aegra_api.services.langgraph_service import get_langgraph_service
from aegra_api.services.run_status import set_thread_status
from aegra_api.utils.assistants import resolve_assistant_id
from aegra_api.utils.run_utils import _merge_jsonb

logger = structlog.getLogger(__name__)


async def _validate_resume_command(session: AsyncSession, thread_id: str, command: dict[str, Any] | None) -> None:
    """Validate resume command requirements."""
    if command and command.get("resume") is not None:
        # Check if thread exists and is in interrupted state
        thread_stmt = select(ThreadORM).where(ThreadORM.thread_id == thread_id)
        thread = await session.scalar(thread_stmt)
        if not thread:
            raise HTTPException(404, f"Thread '{thread_id}' not found")
        if thread.status != "interrupted":
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
) -> tuple[str, Run, RunJob]:
    """Shared run-creation logic used by create, stream, and wait endpoints.

    Validates inputs, resolves the assistant, persists the RunORM record,
    builds a RunJob, submits it to the executor, and returns the triple
    ``(run_id, run_model, job)``.
    """
    await _validate_resume_command(session, thread_id, request.command)

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

    # Build the RunJob before persisting so we can store execution_params
    job = RunJob(
        identity=RunIdentity(run_id=run_id, thread_id=thread_id, graph_id=assistant.graph_id),
        user=user,
        execution=RunExecution(
            input_data=request.input or {},
            config=config,
            context=context,
            stream_mode=request.stream_mode,
            checkpoint=request.checkpoint,
            command=request.command,
        ),
        behavior=RunBehavior(
            interrupt_before=request.interrupt_before,
            interrupt_after=request.interrupt_after,
            multitask_strategy=request.multitask_strategy,
            subgraphs=request.stream_subgraphs or False,
        ),
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
    exec_params["_enqueued_at"] = time.time()

    now = datetime.now(UTC)
    run_orm = RunORM(
        run_id=run_id,
        thread_id=thread_id,
        assistant_id=resolved_assistant_id,
        status=initial_status,
        input=request.input or {},
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

    # Submit to executor
    await executor.submit(job)
    logger.info("Submitted run to executor", run_id=run_id)

    return run_id, run, job

"""Run endpoints for Agent Protocol"""

import asyncio
import contextlib
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import structlog
from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from aegra_api.core.active_runs import active_runs
from aegra_api.core.auth_deps import auth_dependency, get_current_user
from aegra_api.core.auth_handlers import build_auth_context, handle_event
from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import _get_session_maker, get_session
from aegra_api.core.sse import create_end_event, get_sse_headers
from aegra_api.models import Run, RunCreate, RunStatus, User
from aegra_api.models.errors import CONFLICT, NOT_FOUND, SSE_RESPONSE
from aegra_api.observability.metrics import get_worker_metrics
from aegra_api.services.run_executor import mark_user_cancellation, unmark_user_cancellation
from aegra_api.services.run_preparation import _prepare_run
from aegra_api.services.run_waiters import TERMINAL_STATES, encode_output, heartbeat_wait_body
from aegra_api.services.streaming_service import streaming_service
from aegra_api.settings import settings
from aegra_api.utils.status_compat import validate_run_status

router = APIRouter(tags=["Thread Runs"], dependencies=auth_dependency)

logger = structlog.getLogger(__name__)


# active_runs is imported from aegra_api.core.active_runs (dependency-free module)

# Default stream modes for background run execution
DEFAULT_STREAM_MODES = ["values"]


@router.post("/threads/{thread_id}/runs", response_model=Run, responses={**NOT_FOUND, **CONFLICT})
async def create_run(
    thread_id: str,
    request: RunCreate,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> Run:
    """Create and execute a new run.

    Starts graph execution asynchronously and returns the run record
    immediately with status `pending`. Poll the run or use the stream
    endpoint to follow progress. Provide either `input` or `command` (for
    human-in-the-loop resumption) but not both.
    """
    # Authorization check (create_run action on threads resource)
    ctx = build_auth_context(user, "threads", "create_run")
    value = {**request.model_dump(), "thread_id": thread_id}
    filters = await handle_event(ctx, value)

    # If handler modified config/context, update request
    if filters:
        if "config" in filters and isinstance(filters["config"], dict):
            request.config = {**(request.config or {}), **filters["config"]}
        if "context" in filters and isinstance(filters["context"], dict):
            request.context = {**(request.context or {}), **filters["context"]}
    else:
        value_config = value.get("config")
        if isinstance(value_config, dict):
            request.config = {**(request.config or {}), **value_config}

        value_context = value.get("context")
        if isinstance(value_context, dict):
            request.context = {**(request.context or {}), **value_context}

    _run_id, run, _job = await _prepare_run(session, thread_id, request, user, initial_status="pending")

    return run


@router.post("/threads/{thread_id}/runs/stream", responses={**SSE_RESPONSE, **NOT_FOUND, **CONFLICT})
async def create_and_stream_run(
    thread_id: str,
    request: RunCreate,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> StreamingResponse:
    """Create a new run and stream its execution via SSE.

    Returns a `text/event-stream` response with Server-Sent Events. Each
    event has a `type` field (e.g. `values`, `updates`, `messages`,
    `metadata`, `end`) and a JSON `data` payload.

    Set `on_disconnect` to `"continue"` if the run should keep executing
    after the client disconnects (default is `"cancel"`). Use `stream_mode`
    to control which event types are emitted.
    """
    run_id, run, _job = await _prepare_run(session, thread_id, request, user, initial_status="pending")

    # Default to cancel on disconnect - this matches user expectation that clicking
    # "Cancel" in the frontend will stop the backend task. Users can explicitly
    # set on_disconnect="continue" if they want the task to continue.
    cancel_on_disconnect = (request.on_disconnect or "cancel").lower() == "cancel"

    return StreamingResponse(
        streaming_service.stream_run_execution(
            run,
            None,
            cancel_on_disconnect=cancel_on_disconnect,
        ),
        media_type="text/event-stream",
        headers={
            **get_sse_headers(),
            "Location": f"/threads/{thread_id}/runs/{run_id}/stream",
            "Content-Location": f"/threads/{thread_id}/runs/{run_id}",
        },
    )


@router.get("/threads/{thread_id}/runs/{run_id}", response_model=Run, responses={**NOT_FOUND})
async def get_run(
    thread_id: str,
    run_id: str,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> Run:
    """Get a run by its ID.

    Returns the current state of the run including its status, input, output,
    and error information.
    """
    # Authorization check (read action on runs resource)
    ctx = build_auth_context(user, "runs", "read")
    value = {"run_id": run_id, "thread_id": thread_id}
    await handle_event(ctx, value)

    stmt = select(RunORM).where(
        RunORM.run_id == str(run_id),
        RunORM.thread_id == thread_id,
        RunORM.user_id == user.identity,
    )
    logger.info(f"[get_run] querying DB run_id={run_id} thread_id={thread_id} user={user.identity}")
    run_orm = await session.scalar(stmt)
    if not run_orm:
        raise HTTPException(404, f"Run '{run_id}' not found")

    # Refresh to ensure we have the latest data (in case background task updated it)
    await session.refresh(run_orm)

    logger.info(
        f"[get_run] found run status={run_orm.status} user={user.identity} thread_id={thread_id} run_id={run_id}"
    )
    # Convert to Pydantic
    return Run.model_validate(run_orm)


@router.get("/threads/{thread_id}/runs", response_model=list[Run])
async def list_runs(
    thread_id: str,
    limit: int = Query(10, ge=1, description="Maximum number of runs to return"),
    offset: int = Query(0, ge=0, description="Number of runs to skip for pagination"),
    status: str | None = Query(
        None, description="Filter by run status (e.g. pending, running, success, error, interrupted)"
    ),
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> list[Run]:
    """List runs for a thread.

    Returns runs ordered by creation time (newest first). Use `status` to
    filter and `limit`/`offset` to paginate.
    """
    stmt = (
        select(RunORM)
        .where(
            RunORM.thread_id == thread_id,
            RunORM.user_id == user.identity,
            *([RunORM.status == status] if status else []),
        )
        .limit(limit)
        .offset(offset)
        .order_by(RunORM.created_at.desc())
    )
    logger.info(f"[list_runs] querying DB thread_id={thread_id} user={user.identity}")
    result = await session.scalars(stmt)
    rows = result.all()
    runs = [Run.model_validate(r) for r in rows]
    logger.info(f"[list_runs] total={len(runs)} user={user.identity} thread_id={thread_id}")
    return runs


@router.patch("/threads/{thread_id}/runs/{run_id}", response_model=Run, responses={**NOT_FOUND})
async def update_run(
    thread_id: str,
    run_id: str,
    request: RunStatus,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> Run:
    """Update a run's status.

    Primarily used to interrupt a running execution. Set `status` to
    `"interrupted"` to cooperatively stop the run.
    """
    logger.info(f"[update_run] fetch for update run_id={run_id} thread_id={thread_id} user={user.identity}")
    run_orm = await session.scalar(
        select(RunORM).where(
            RunORM.run_id == str(run_id),
            RunORM.thread_id == thread_id,
            RunORM.user_id == user.identity,
        )
    )
    if not run_orm:
        raise HTTPException(404, f"Run '{run_id}' not found")

    # Handle interruption/cancellation
    # Validate status conforms to API specification
    validated_status = validate_run_status(request.status)

    if validated_status == "interrupted":
        logger.info(f"[update_run] cancelling/interrupting run_id={run_id} user={user.identity} thread_id={thread_id}")
        # Handle interruption - use interrupt_run for cooperative interruption
        await streaming_service.interrupt_run(run_id)
        logger.info(f"[update_run] set DB status=interrupted run_id={run_id}")
        await session.execute(
            update(RunORM)
            .where(RunORM.run_id == str(run_id))
            .values(status="interrupted", updated_at=datetime.now(UTC))
        )
        await session.commit()
        logger.info(f"[update_run] commit done (interrupted) run_id={run_id}")

    # Return final run state
    run_orm = await session.scalar(select(RunORM).where(RunORM.run_id == run_id))
    if not run_orm:
        raise HTTPException(404, f"Run '{run_id}' not found")
    # Refresh to ensure we have the latest data after our own update
    await session.refresh(run_orm)
    return Run.model_validate(run_orm)


@router.get("/threads/{thread_id}/runs/{run_id}/join", responses={**NOT_FOUND})
async def join_run(
    thread_id: str,
    run_id: str,
    user: User = Depends(get_current_user),
) -> StreamingResponse:
    """Wait for a run to complete and return its output.

    Returns a chunked ``application/json`` response. While the run is still
    executing, the server sends periodic ``\\n`` heartbeat bytes to keep the
    connection alive through proxies and load balancers (AWS ALB, Cloudflare,
    etc.). The final chunk is the JSON result. Leading whitespace is ignored
    by JSON parsers, so clients can parse the concatenated body normally.

    If the run is already in a terminal state, the output is returned
    immediately with no heartbeat overhead.

    Sessions are managed manually (not via ``Depends``) to avoid holding a
    pool connection during the long wait.
    """
    maker = _get_session_maker()

    # Short-lived session: validate run exists and check terminal state
    async with maker() as session:
        run_orm = await session.scalar(
            select(RunORM).where(
                RunORM.run_id == str(run_id),
                RunORM.thread_id == thread_id,
                RunORM.user_id == user.identity,
            )
        )
        if not run_orm:
            raise HTTPException(404, f"Run '{run_id}' not found")

        if run_orm.status in TERMINAL_STATES:
            return StreamingResponse(
                iter([encode_output(run_orm.output or {})]),
                media_type="application/json",
            )

    return StreamingResponse(
        heartbeat_wait_body(
            run_id,
            thread_id,
            user.identity,
            timeout=settings.worker.BG_JOB_TIMEOUT_SECS,
        ),
        media_type="application/json",
        headers={
            "Location": f"/threads/{thread_id}/runs/{run_id}/join",
            "Content-Location": f"/threads/{thread_id}/runs/{run_id}",
        },
    )


@router.post("/threads/{thread_id}/runs/wait", responses={**NOT_FOUND, **CONFLICT})
async def wait_for_run(
    thread_id: str,
    request: RunCreate,
    user: User = Depends(get_current_user),
) -> StreamingResponse:
    """Create a run, execute it, and wait for completion.

    Returns a chunked ``application/json`` response with periodic ``\\n``
    heartbeat bytes to keep the connection alive. The final chunk is the
    JSON result. Uses ``BG_JOB_TIMEOUT_SECS`` (default 1 hour) as the
    safety-net timeout.

    Sessions are managed manually (not via ``Depends``) to avoid holding a
    pool connection during the long wait.
    """
    maker = _get_session_maker()

    # Session block: all pre-execution DB work (validate, create run, submit)
    async with maker() as session:
        run_id, _run, _job = await _prepare_run(session, thread_id, request, user, initial_status="pending")

    # No pool connection held from here — safe for long waits
    return StreamingResponse(
        heartbeat_wait_body(
            run_id,
            thread_id,
            user.identity,
            timeout=settings.worker.BG_JOB_TIMEOUT_SECS,
        ),
        media_type="application/json",
        headers={
            "Location": f"/threads/{thread_id}/runs/{run_id}/join",
            "Content-Location": f"/threads/{thread_id}/runs/{run_id}",
        },
    )


@router.get("/threads/{thread_id}/runs/{run_id}/stream", responses={**SSE_RESPONSE, **NOT_FOUND})
async def stream_run(
    thread_id: str,
    run_id: str,
    last_event_id: str | None = Header(None, alias="Last-Event-ID"),
    _stream_mode: str | None = Query(None, description="Override the stream mode for this connection."),
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> StreamingResponse:
    """Stream an existing run's execution via SSE.

    Attach to a run that was created without streaming (e.g. via the create
    endpoint) to receive its events in real time. If the run has already
    finished, a single `end` event is emitted. Use the `Last-Event-ID`
    header to resume from a specific event after a disconnect.
    """
    logger.info(f"[stream_run] fetch for stream run_id={run_id} thread_id={thread_id} user={user.identity}")
    run_orm = await session.scalar(
        select(RunORM).where(
            RunORM.run_id == str(run_id),
            RunORM.thread_id == thread_id,
            RunORM.user_id == user.identity,
        )
    )
    if not run_orm:
        raise HTTPException(404, f"Run '{run_id}' not found")

    logger.info(f"[stream_run] status={run_orm.status} user={user.identity} thread_id={thread_id} run_id={run_id}")
    # If already terminal and no Last-Event-ID, just emit end.
    # If Last-Event-ID is present, fall through to stream_run_execution
    # which will replay missed events from the buffer before ending.
    if run_orm.status in TERMINAL_STATES and not last_event_id:
        final_status = "error" if run_orm.status == "error" else run_orm.status

        async def generate_final() -> AsyncIterator[str]:
            yield create_end_event(status=final_status)

        logger.info(f"[stream_run] starting terminal stream run_id={run_id} status={run_orm.status}")
        return StreamingResponse(
            generate_final(),
            media_type="text/event-stream",
            headers={
                **get_sse_headers(),
                "Location": f"/threads/{thread_id}/runs/{run_id}/stream",
                "Content-Location": f"/threads/{thread_id}/runs/{run_id}",
            },
        )

    # Stream active or pending runs via broker

    # Build a lightweight Pydantic Run from ORM for streaming context (IDs already strings)
    run_model = Run.model_validate(run_orm)

    return StreamingResponse(
        streaming_service.stream_run_execution(run_model, last_event_id, cancel_on_disconnect=False),
        media_type="text/event-stream",
        headers={
            **get_sse_headers(),
            "Location": f"/threads/{thread_id}/runs/{run_id}/stream",
            "Content-Location": f"/threads/{thread_id}/runs/{run_id}",
        },
    )


@router.post(
    "/threads/{thread_id}/runs/{run_id}/cancel",
    response_model=Run,
    responses={**NOT_FOUND},
)
async def cancel_run_endpoint(
    thread_id: str,
    run_id: str,
    wait: int = Query(0, ge=0, le=1, description="Set to 1 to wait for the run task to settle before returning."),
    action: str = Query(
        "cancel",
        pattern="^(cancel|interrupt)$",
        description="Cancellation strategy: 'cancel' for hard cancel, 'interrupt' for cooperative interrupt.",
    ),
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> Run:
    """Cancel or interrupt a running execution.

    Use `action=cancel` to hard-cancel the run immediately, or
    `action=interrupt` to cooperatively interrupt (the graph can handle the
    interrupt and save partial state). Set `wait=1` to block until the
    background task has fully settled before returning the updated run.
    """
    logger.info(f"[cancel_run] fetch run run_id={run_id} thread_id={thread_id} user={user.identity}")
    run_orm = await session.scalar(
        select(RunORM).where(
            RunORM.run_id == run_id,
            RunORM.thread_id == thread_id,
            RunORM.user_id == user.identity,
        )
    )
    if not run_orm:
        raise HTTPException(404, f"Run '{run_id}' not found")

    if run_orm.status == "pending":
        # Atomic CAS: only update if still pending (prevents TOCTOU with worker pickup)
        cas_result = await session.execute(
            update(RunORM)
            .where(RunORM.run_id == run_id, RunORM.status == "pending")
            .values(status="interrupted", updated_at=datetime.now(UTC))
        )
        if cas_result.rowcount > 0:  # type: ignore[union-attr]
            await session.commit()
            logger.info(f"[cancel_run] cancelled pending run_id={run_id}")
            # Run never went through execute_run, so increment completed here
            metrics = get_worker_metrics()
            if metrics is not None:
                graph_id = (run_orm.execution_params or {}).get("graph_id", "unknown")
                metrics.runs_completed.labels(graph_id=graph_id, status="interrupted").inc()
        else:
            # Worker picked it up between SELECT and CAS — now running.
            # Fall through to the running-cancel path.
            await session.rollback()
            run_orm = await session.scalar(select(RunORM).where(RunORM.run_id == run_id))
            if run_orm is None:
                raise HTTPException(404, f"Run '{run_id}' not found")
            if run_orm.status in TERMINAL_STATES:
                # Run already completed between CAS and re-fetch
                return Run.model_validate(run_orm)
            mark_user_cancellation(run_id)
            try:
                if action == "interrupt":
                    logger.info(f"[cancel_run] interrupt (was pending) run_id={run_id}")
                    await streaming_service.interrupt_run(run_id)
                else:
                    logger.info(f"[cancel_run] cancel (was pending) run_id={run_id}")
                    await streaming_service.cancel_run(run_id)
            except Exception:
                unmark_user_cancellation(run_id)
                raise
            # Safety-net UPDATE: only if execute_run hasn't finalized yet.
            # execute_run handles finalization via _user_cancellations path,
            # but the cancel is async — this catches the gap.
            await session.execute(
                update(RunORM)
                .where(RunORM.run_id == run_id, RunORM.status.notin_(TERMINAL_STATES))
                .values(status="interrupted", updated_at=datetime.now(UTC))
            )
            await session.commit()
    elif action == "interrupt":
        logger.info(f"[cancel_run] interrupt run_id={run_id} user={user.identity} thread_id={thread_id}")
        mark_user_cancellation(run_id)
        try:
            await streaming_service.interrupt_run(run_id)
        except Exception:
            unmark_user_cancellation(run_id)
            raise
        await session.execute(
            update(RunORM)
            .where(RunORM.run_id == str(run_id), RunORM.status.notin_(TERMINAL_STATES))
            .values(status="interrupted", updated_at=datetime.now(UTC))
        )
        await session.commit()
    else:
        logger.info(f"[cancel_run] cancel run_id={run_id} user={user.identity} thread_id={thread_id}")
        mark_user_cancellation(run_id)
        try:
            await streaming_service.cancel_run(run_id)
        except Exception:
            unmark_user_cancellation(run_id)
            raise
        await session.execute(
            update(RunORM)
            .where(RunORM.run_id == str(run_id), RunORM.status.notin_(TERMINAL_STATES))
            .values(status="interrupted", updated_at=datetime.now(UTC))
        )
        await session.commit()

    # Optionally wait for the run to settle
    if wait:
        # Poll DB until the run reaches a terminal state (or 10s timeout).
        # This is simpler and more reliable than pub/sub for cancel-with-wait
        # since the cancel has already been issued and the status update committed.
        for _ in range(20):
            await asyncio.sleep(0.5)
            session.expire_all()  # sync method, clears cache
            fresh = await session.scalar(select(RunORM).where(RunORM.run_id == run_id))
            if fresh and fresh.status in TERMINAL_STATES:
                break

    # Reload and return updated Run (do NOT delete here; deletion is a separate endpoint)
    run_orm = await session.scalar(
        select(RunORM).where(
            RunORM.run_id == run_id,
            RunORM.thread_id == thread_id,
            RunORM.user_id == user.identity,
        )
    )
    if not run_orm:
        raise HTTPException(404, f"Run '{run_id}' not found after cancellation")
    return Run.model_validate(run_orm)


@router.delete(
    "/threads/{thread_id}/runs/{run_id}",
    status_code=204,
    responses={**NOT_FOUND, **CONFLICT},
)
async def delete_run(
    thread_id: str,
    run_id: str,
    force: int = Query(0, ge=0, le=1, description="Set to 1 to cancel an active run before deleting it."),
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> None:
    """Delete a run record.

    If the run is active (pending or running) and `force=0`, returns 409
    Conflict. Set `force=1` to cancel the run first (best-effort) and then
    delete it. Returns 204 No Content on success.
    """
    # Authorization check (delete action on runs resource)
    ctx = build_auth_context(user, "runs", "delete")
    value = {"run_id": run_id, "thread_id": thread_id}
    await handle_event(ctx, value)
    logger.info(f"[delete_run] fetch run run_id={run_id} thread_id={thread_id} user={user.identity}")
    run_orm = await session.scalar(
        select(RunORM).where(
            RunORM.run_id == str(run_id),
            RunORM.thread_id == thread_id,
            RunORM.user_id == user.identity,
        )
    )
    if not run_orm:
        raise HTTPException(404, f"Run '{run_id}' not found")

    # If active and not forcing, reject deletion
    if run_orm.status in ["pending", "running"] and not force:
        raise HTTPException(
            status_code=409,
            detail="Run is active. Retry with force=1 to cancel and delete.",
        )

    # If forcing and active, cancel first
    if force and run_orm.status in ["pending", "running"]:
        logger.info(f"[delete_run] force-cancelling active run run_id={run_id}")
        await streaming_service.cancel_run(run_id)
        # Best-effort: wait for bg task to settle
        task = active_runs.get(run_id)
        if task:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    # Delete the record
    await session.execute(
        delete(RunORM).where(
            RunORM.run_id == str(run_id),
            RunORM.thread_id == thread_id,
            RunORM.user_id == user.identity,
        )
    )
    await session.commit()

    # Clean up active task if exists
    task = active_runs.pop(run_id, None)
    if task and not task.done():
        task.cancel()

    # 204 No Content
    return

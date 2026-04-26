"""Stateless (thread-free) run endpoints.

These endpoints accept POST /runs/stream, /runs/wait, and /runs without a
thread_id. They generate an ephemeral thread, delegate to the existing threaded
endpoint functions, and clean up the thread afterward (unless the caller
explicitly sets ``on_completion="keep"``).
"""

import asyncio
from collections.abc import AsyncIterator
from uuid import uuid4

import structlog
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette import EventSourceResponse

from aegra_api.api.runs import (
    create_and_stream_run,
    create_run,
    wait_for_run,
)
from aegra_api.core.active_runs import active_runs
from aegra_api.core.auth_deps import auth_dependency, get_current_user
from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import Thread as ThreadORM
from aegra_api.core.orm import _get_session_maker, get_session
from aegra_api.core.sse import heartbeat_factory
from aegra_api.models import Run, RunCreate, User
from aegra_api.models.errors import CONFLICT, NOT_FOUND, SSE_RESPONSE
from aegra_api.services.executor import executor
from aegra_api.services.streaming_service import streaming_service
from aegra_api.settings import settings

router = APIRouter(tags=["Stateless Runs"], dependencies=auth_dependency)
logger = structlog.getLogger(__name__)

# Strong references to fire-and-forget cleanup tasks to prevent GC
_background_cleanup_tasks: set[asyncio.Task[None]] = set()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _delete_thread_by_id(thread_id: str, user_id: str) -> None:
    """Delete an ephemeral thread and cascade-delete its runs.

    Opens its own DB session so it can be called after the request session has
    been closed (e.g. in a ``finally`` block or background task).
    """
    maker = _get_session_maker()
    async with maker() as session:
        # Cancel any still-active runs on this thread
        active_runs_stmt = select(RunORM).where(
            RunORM.thread_id == thread_id,
            RunORM.user_id == user_id,
            RunORM.status.in_(["pending", "running"]),
        )
        active_runs_list = (await session.scalars(active_runs_stmt)).all()

        for run in active_runs_list:
            run_id = run.run_id
            await streaming_service.cancel_run(run_id)
            task = active_runs.pop(run_id, None)
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    logger.exception("Error awaiting cancelled task during thread cleanup", run_id=run_id)

        # Delete thread (cascade deletes runs via FK)
        thread = await session.scalar(
            select(ThreadORM).where(
                ThreadORM.thread_id == thread_id,
                ThreadORM.user_id == user_id,
            )
        )
        if thread:
            await session.delete(thread)
            await session.commit()


async def _cleanup_after_background_run(run_id: str, thread_id: str, user_id: str) -> None:
    """Wait for a background run to finish, then delete the ephemeral thread.

    Uses executor.wait_for_completion which works both in-process (dev)
    and cross-instance (prod with Redis workers).
    """
    try:
        await executor.wait_for_completion(run_id, timeout=3600.0)
    except (asyncio.CancelledError, TimeoutError):
        pass
    except Exception:
        logger.exception("Error waiting for background run", run_id=run_id)

    try:
        await _delete_thread_by_id(thread_id, user_id)
    except Exception:
        logger.exception("Failed to delete ephemeral thread", thread_id=thread_id, run_id=run_id)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/runs/wait", responses={**NOT_FOUND, **CONFLICT})
async def stateless_wait_for_run(
    request: RunCreate,
    user: User = Depends(get_current_user),
) -> StreamingResponse:
    """Create a stateless run and wait for completion.

    Generates an ephemeral thread, delegates to the threaded ``wait_for_run``
    endpoint, and deletes the thread after the response finishes streaming
    (unless ``on_completion="keep"``).
    """
    thread_id = str(uuid4())
    should_delete = request.on_completion != "keep"

    try:
        response = await wait_for_run(thread_id, request, user)
    except Exception:
        if should_delete:
            try:
                await _delete_thread_by_id(thread_id, user.identity)
            except Exception:
                logger.exception(
                    "Failed to delete ephemeral thread after wait error",
                    thread_id=thread_id,
                )
        raise

    if not should_delete:
        return response

    # Wrap the body_iterator so cleanup happens after the stream ends
    original_iterator = response.body_iterator

    async def _wrapped_iterator() -> AsyncIterator[bytes]:
        completed = False
        try:
            async for chunk in original_iterator:
                yield chunk
            completed = True
        finally:
            aclose = getattr(original_iterator, "aclose", None)
            if aclose is not None:
                await aclose()
            if completed:
                try:
                    await _delete_thread_by_id(thread_id, user.identity)
                except Exception:
                    logger.exception(
                        "Failed to delete ephemeral thread after wait",
                        thread_id=thread_id,
                    )
            else:
                logger.info(
                    "Client disconnected before stream completed, keeping ephemeral thread",
                    thread_id=thread_id,
                )

    return StreamingResponse(
        _wrapped_iterator(),
        status_code=response.status_code,
        media_type=response.media_type,
        headers=dict(response.headers),
    )


@router.post("/runs/stream", responses={**SSE_RESPONSE, **NOT_FOUND, **CONFLICT})
async def stateless_stream_run(
    request: RunCreate,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> EventSourceResponse:
    """Create a stateless run and stream its execution.

    Generates an ephemeral thread, delegates to the threaded
    ``create_and_stream_run`` endpoint, and deletes the thread after the
    stream finishes (unless ``on_completion="keep"``).
    """
    thread_id = str(uuid4())
    should_delete = request.on_completion != "keep"

    try:
        response = await create_and_stream_run(thread_id, request, user, session)
    except Exception:
        # create_and_stream_run may have auto-created the thread via
        # update_thread_metadata before raising; clean up to avoid orphans.
        if should_delete:
            try:
                await _delete_thread_by_id(thread_id, user.identity)
            except Exception:
                logger.exception(
                    "Failed to delete ephemeral thread after stream setup error",
                    thread_id=thread_id,
                )
        raise

    if not should_delete:
        return response

    # Wrap the body_iterator so cleanup happens after the stream ends.
    # The inner EventSourceResponse is never served directly — we only
    # steal its iterator and close-handler, then re-wrap in a new
    # EventSourceResponse so the outer ping/listen-for-disconnect tasks
    # run against the cleaned-up iterator.
    original_iterator = response.body_iterator
    inner_close_handler = response.client_close_handler_callable

    async def _wrapped_iterator() -> AsyncIterator[bytes]:
        completed = False
        try:
            async for chunk in original_iterator:
                yield chunk
            completed = True
        finally:
            aclose = getattr(original_iterator, "aclose", None)
            if aclose is not None:
                await aclose()
            if completed:
                try:
                    await _delete_thread_by_id(thread_id, user.identity)
                except Exception:
                    logger.exception(
                        "Failed to delete ephemeral thread after stream",
                        thread_id=thread_id,
                    )
            else:
                # Early client disconnect: deleting the thread here would
                # cancel the still-running background execution (via
                # _delete_thread_by_id) and break the on_disconnect="continue"
                # contract. Mirror stateless_wait_for_run and keep the thread.
                logger.info(
                    "Client disconnected before stream completed, keeping ephemeral thread",
                    thread_id=thread_id,
                )

    return EventSourceResponse(
        _wrapped_iterator(),
        status_code=response.status_code,
        ping=settings.app.sse_ping_interval_secs,
        ping_message_factory=heartbeat_factory,
        client_close_handler_callable=inner_close_handler,
        headers=dict(response.headers),
    )


@router.post("/runs", response_model=Run, responses={**NOT_FOUND, **CONFLICT})
async def stateless_create_run(
    request: RunCreate,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> Run:
    """Create a stateless background run.

    Generates an ephemeral thread, delegates to the threaded ``create_run``
    endpoint, and schedules cleanup as a background task (unless
    ``on_completion="keep"``).
    """
    thread_id = str(uuid4())
    should_delete = request.on_completion != "keep"

    try:
        result = await create_run(thread_id, request, user, session)
    except Exception:
        # create_run may have auto-created the thread via
        # update_thread_metadata before raising; clean up to avoid orphans.
        if should_delete:
            try:
                await _delete_thread_by_id(thread_id, user.identity)
            except Exception:
                logger.exception(
                    "Failed to delete ephemeral thread after create error",
                    thread_id=thread_id,
                )
        raise

    if should_delete:
        task = asyncio.create_task(_cleanup_after_background_run(result.run_id, thread_id, user.identity))
        _background_cleanup_tasks.add(task)
        task.add_done_callback(_background_cleanup_tasks.discard)

    return result

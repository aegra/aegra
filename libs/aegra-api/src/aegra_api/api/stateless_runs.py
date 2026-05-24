"""Stateless (thread-free) run endpoints.

These endpoints accept POST /runs/stream, /runs/wait, and /runs without a
thread_id. They generate an ephemeral thread, delegate to the existing threaded
endpoint functions, and clean up the thread afterward (unless the caller
explicitly sets ``on_completion="keep"``).
"""

import asyncio
from collections.abc import AsyncIterator, Mapping
from uuid import uuid4

import structlog
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette import EventSourceResponse

from aegra_api.api.runs import (
    create_and_stream_run,
    create_run,
    wait_for_run,
)
from aegra_api.core.auth_deps import auth_dependency, get_current_user
from aegra_api.core.orm import get_session
from aegra_api.core.sse import make_sse_response
from aegra_api.models import Run, RunCreate, User
from aegra_api.models.errors import CONFLICT, NOT_FOUND, SSE_RESPONSE
from aegra_api.services.broker import broker_manager
from aegra_api.services.run_cleanup import (
    _CLEANUP_ERRORS,
    _background_cleanup_tasks,
    delete_thread_by_id,
    schedule_background_cleanup,
)

router = APIRouter(tags=["Stateless Runs"], dependencies=auth_dependency)
logger = structlog.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stateless-specific helpers (slow-client detection, header parsing)
# ---------------------------------------------------------------------------


def _run_finished(run_id: str) -> bool:
    """Best-effort check: did the run already publish its end event locally?

    Both broker backends set ``_finished`` whenever ``put(end)``
    (producer-side) or ``aiter()`` reading the end event (consumer-side)
    runs in this process. From the calling instance's perspective:

      - In-memory (dev): producer and consumer share this process, so
        any ``end`` event reliably flips the flag.
      - Redis (prod): if the run was picked up by a worker on another
        instance, this process never called ``put()`` — the flag flips
        only after our local pub/sub subscriber drains the end event.
        A slow-client abort that races ahead of the drain leaves the
        flag False even though the run itself terminated.

    Returning False errs on keeping the thread. The broker cleanup task
    removes the broker dict entry after an hour, but it does NOT delete
    the ephemeral thread row from Postgres — leaving an unmatched abort
    here leaks an empty ephemeral thread.

    TODO(orphan-thread-sweeper): file as a follow-up issue. The sweeper
    should periodically delete ephemeral threads (flagged via
    ``is_ephemeral`` column or metadata) where no run is in
    pending/running status and ``updated_at`` is older than a configured
    retention window. Triggering condition for accumulation: prod-mode
    Redis broker + slow-client / dead-proxy aborts that race ahead of
    the local pub/sub subscriber draining the run's end event.

    A missing broker (None) also returns False — the safe answer for
    "run not started" and "broker dict entry already swept".
    """
    broker = broker_manager.get_broker(run_id)
    return broker is not None and broker.is_finished()


def _extract_run_id_from_headers(headers: Mapping[str, str]) -> str | None:
    """Pull ``run_id`` from a streaming response's ``Content-Location``.

    Both ``wait_for_run`` and ``create_and_stream_run`` set
    ``Content-Location: /threads/{thread_id}/runs/{run_id}`` with the same
    format; we currently call this only from the stream endpoint, where
    slow-client cleanup needs the run_id to consult the broker.

    Starlette's ``Headers``/``MutableHeaders`` is case-insensitive on
    ``.get`` and stores keys lowercase per the ASGI spec, so the
    lowercase lookup is the canonical path. The capitalized fallback
    covers plain ``dict`` callers (e.g. unit tests constructing headers
    directly) — cheap defense vs the silent "always returns None" bug
    if someone ever bypasses Starlette normalization. Failing to parse
    disables the slow-client cleanup branch — not fatal.
    """
    location = headers.get("content-location") or headers.get("Content-Location") or ""
    if not location:
        return None
    parts = location.strip("/").split("/")
    if len(parts) >= 4 and parts[0] == "threads" and parts[2] == "runs":
        return parts[3]
    return None


async def _delete_thread_with_log(thread_id: str, user_id: str, *, reason: str) -> None:
    """Delete an ephemeral thread, logging infra failures."""
    try:
        await delete_thread_by_id(thread_id, user_id)
    except _CLEANUP_ERRORS:
        logger.exception(reason, thread_id=thread_id)


def _schedule_thread_cleanup(thread_id: str, user_id: str, *, reason: str) -> None:
    """Fire-and-forget delete keyed off ``_background_cleanup_tasks``."""
    task = asyncio.create_task(_delete_thread_with_log(thread_id, user_id, reason=reason))
    _background_cleanup_tasks.add(task)
    task.add_done_callback(_background_cleanup_tasks.discard)


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
                await delete_thread_by_id(thread_id, user.identity)
            except _CLEANUP_ERRORS:
                logger.exception(
                    "Failed to delete ephemeral thread after wait error",
                    thread_id=thread_id,
                )
        raise

    if not should_delete:
        return response

    # Wrap the body_iterator so cleanup happens after the stream ends.
    # The slow-client cleanup branch is intentionally omitted here: in
    # prod-mode the wait endpoint never iterates the broker locally
    # (heartbeat_wait_body polls executor state, not events), so the
    # consumer-instance broker stays out of the local cache and
    # ``_run_finished`` would always return False cross-instance.
    # Stream endpoint keeps the slow-client branch where the consumer
    # actually subscribes to the broker.
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
                await _delete_thread_with_log(
                    thread_id, user.identity, reason="Failed to delete ephemeral thread after wait"
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
                await delete_thread_by_id(thread_id, user.identity)
            except _CLEANUP_ERRORS:
                logger.exception(
                    "Failed to delete ephemeral thread after stream setup error",
                    thread_id=thread_id,
                )
        raise

    if not should_delete:
        return response

    # The inner EventSourceResponse is never ASGI-served — its background
    # ping/listen-for-disconnect tasks are only spawned inside __call__,
    # which we never invoke. We borrow its iterator + close-handler here
    # and build a fresh outer response so those background tasks run
    # against the cleanup-wrapped iterator instead.
    original_iterator = response.body_iterator
    inner_close_handler = response.client_close_handler_callable
    run_id = _extract_run_id_from_headers(response.headers)

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
                await _delete_thread_with_log(
                    thread_id, user.identity, reason="Failed to delete ephemeral thread after stream"
                )
            elif run_id is not None and _run_finished(run_id):
                # Slow-client / dead-proxy abort after the run already
                # finished: there's nothing left to resume, so schedule a
                # deferred delete instead of leaking the ephemeral thread.
                _schedule_thread_cleanup(
                    thread_id,
                    user.identity,
                    reason="Failed to delete ephemeral thread after slow-client abort",
                )
            else:
                # Early client disconnect with the run still active:
                # delete_thread_by_id would cancel the background execution
                # and break the on_disconnect="continue" contract.
                logger.info(
                    "Client disconnected before stream completed, keeping ephemeral thread",
                    thread_id=thread_id,
                )

    return make_sse_response(
        _wrapped_iterator(),
        status_code=response.status_code,
        close_handler=inner_close_handler,
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
                await delete_thread_by_id(thread_id, user.identity)
            except _CLEANUP_ERRORS:
                logger.exception(
                    "Failed to delete ephemeral thread after create error",
                    thread_id=thread_id,
                )
        raise

    if should_delete:
        schedule_background_cleanup(result.run_id, thread_id, user.identity)

    return result

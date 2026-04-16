"""Stateless (thread-free) run endpoints.

These endpoints accept POST /runs/stream, /runs/wait, and /runs without a
thread_id. They generate an ephemeral thread, delegate to the existing threaded
endpoint functions, and clean up the thread afterward (unless the caller
explicitly sets ``on_completion="keep"``).
"""

from collections.abc import AsyncIterator
from uuid import uuid4

import structlog
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from aegra_api.api.runs import (
    create_and_stream_run,
    create_run,
    wait_for_run,
)
from aegra_api.core.auth_deps import auth_dependency, get_current_user
from aegra_api.core.orm import get_session
from aegra_api.models import Run, RunCreate, User
from aegra_api.models.errors import CONFLICT, NOT_FOUND, SSE_RESPONSE
from aegra_api.services.run_cleanup import (
    cleanup_after_background_run,
    delete_thread_by_id,
    schedule_background_cleanup,
)

router = APIRouter(tags=["Stateless Runs"], dependencies=auth_dependency)
logger = structlog.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _delete_thread_by_id(thread_id: str, user_id: str) -> None:
    """Delete an ephemeral thread and cascade-delete its runs."""
    await delete_thread_by_id(thread_id, user_id)


async def _cleanup_after_background_run(run_id: str, thread_id: str, user_id: str) -> None:
    """Wait for a background run to finish, then delete the ephemeral thread."""
    await cleanup_after_background_run(run_id, thread_id, user_id)


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
) -> StreamingResponse:
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

    # Wrap the body_iterator so cleanup happens after the stream ends
    original_iterator = response.body_iterator

    async def _wrapped_iterator() -> AsyncIterator[str | bytes]:
        try:
            async for chunk in original_iterator:
                yield chunk
        finally:
            # Close the underlying iterator if it supports aclose()
            aclose = getattr(original_iterator, "aclose", None)
            if aclose is not None:
                await aclose()
            try:
                await _delete_thread_by_id(thread_id, user.identity)
            except Exception:
                logger.exception(
                    "Failed to delete ephemeral thread after stream",
                    thread_id=thread_id,
                )

    return StreamingResponse(
        _wrapped_iterator(),
        status_code=response.status_code,
        media_type=response.media_type,
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
        schedule_background_cleanup(result.run_id, thread_id, user.identity)

    return result

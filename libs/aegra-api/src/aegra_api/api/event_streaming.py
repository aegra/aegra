"""Agent Protocol v2 event streaming endpoints.

* ``POST /threads/{thread_id}/stream/events`` — SSE stream of a run's
  events, filtered by channel. Body is an ``EventStreamRequest``.
* ``POST /threads/{thread_id}/commands`` — run a thread command
  (``run.start``, ``input.respond``) and get a JSON response envelope.

Both gate on the ``FF_V2_EVENT_STREAMING`` flag + runtime capability, and
verify thread (and run) ownership before doing anything.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import structlog
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette import EventSourceResponse

from aegra_api.core.auth_deps import auth_dependency, get_current_user
from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import Thread as ThreadORM
from aegra_api.core.orm import get_session
from aegra_api.core.sse import format_sse_message, get_sse_headers, make_sse_response, sse_to_bytes
from aegra_api.models import User
from aegra_api.models.event_streaming import EventStreamRequest, ThreadCommand
from aegra_api.services.event_streaming.capabilities import get_v2_capabilities
from aegra_api.services.event_streaming.commands import handle_command
from aegra_api.services.event_streaming.session import RunLister, ThreadEventSession, validate_channels

logger = structlog.getLogger(__name__)

router = APIRouter(tags=["Event Streaming"], dependencies=auth_dependency)


async def _verify_thread_owned_or_new(session: AsyncSession, thread_id: str, user: User) -> None:
    """Allow a not-yet-existing thread; block one owned by someone else.

    The SDK mints the thread id client-side and expects ``run.start`` to
    create it (run preparation does, owned by the caller). So a missing
    thread is fine here; an existing thread owned by another user is 404.
    """
    existing_owner = await session.scalar(select(ThreadORM.user_id).where(ThreadORM.thread_id == thread_id))
    if existing_owner is not None and existing_owner != user.identity:
        raise HTTPException(404, f"Thread '{thread_id}' not found")


def _thread_run_lister(session: AsyncSession, thread_id: str, user: User) -> RunLister:
    """Async callable returning the thread's run ids (oldest first), user-scoped.

    Called repeatedly while a stream is live so a run started after the
    stream opened is picked up. Only the thread owner's runs are listed.
    """

    async def list_run_ids() -> list[str]:
        rows = await session.scalars(
            select(RunORM.run_id)
            .where(RunORM.thread_id == thread_id, RunORM.user_id == user.identity)
            .order_by(RunORM.created_at.asc())
        )
        return list(rows.all())

    return list_run_ids


def _require_v2_enabled() -> None:
    """503 with a clear reason when v2 is off or the runtime can't serve it."""
    caps = get_v2_capabilities()
    if not caps.ok:
        raise HTTPException(503, caps.error_message)


@router.post("/threads/{thread_id}/stream/events")
async def stream_thread_events(
    thread_id: str,
    body: EventStreamRequest,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> EventSourceResponse:
    """Open a channel-filtered SSE stream of the thread's run events.

    Thread-scoped: events for any run on the thread flow through, so a
    client can open the stream then issue ``run.start``. Each SSE frame's
    ``data:`` is a protocol event envelope; ``id:`` is the ``seq`` a client
    echoes back as ``since`` on resume.
    """
    _require_v2_enabled()
    # The SDK opens the stream (lifecycle watcher) before run.start, against a
    # thread it minted client-side — so a not-yet-existing thread is allowed;
    # a thread owned by another user is not.
    await _verify_thread_owned_or_new(session, thread_id, user)

    channels, invalid = validate_channels(body.channels)
    if invalid:
        raise HTTPException(400, f"Unsupported channels: {', '.join(invalid)}")

    session_stream = ThreadEventSession(
        thread_id,
        channels=channels,
        list_run_ids=_thread_run_lister(session, thread_id, user),
        since=body.since,
    )
    return make_sse_response(sse_to_bytes(_frame_events(session_stream)), headers=get_sse_headers())


@router.post("/threads/{thread_id}/commands")
async def post_thread_command(
    thread_id: str,
    body: ThreadCommand,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> JSONResponse:
    """Run a single v2 command on the thread and return its response envelope."""
    _require_v2_enabled()
    # run.start may target a not-yet-created thread (the SDK mints the id and
    # expects run preparation to create it); other ownership is enforced there.
    await _verify_thread_owned_or_new(session, thread_id, user)

    response, _run_id = await handle_command(body.model_dump(), session=session, thread_id=thread_id, user=user)
    status_code = 200 if response.get("type") == "success" else 400
    return JSONResponse(response, status_code=status_code)


async def _frame_events(session_stream: ThreadEventSession) -> AsyncGenerator[str, None]:
    """Frame v2 event envelopes as SSE messages (event=method, data=envelope, id=seq)."""
    async for envelope in session_stream.stream():
        yield format_sse_message(envelope["method"], envelope, str(envelope["seq"]))

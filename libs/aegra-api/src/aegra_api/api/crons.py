"""Cron job endpoints for Agent Protocol.

Implements the six endpoints consumed by the LangGraph SDK ``CronsClient``:

* ``POST  /runs/crons``                  → create (stateless, returns Cron)
* ``POST  /threads/{thread_id}/runs/crons`` → create for thread (returns Cron)
* ``PATCH /runs/crons/{cron_id}``         → update (returns Cron)
* ``DELETE /runs/crons/{cron_id}``        → delete (204)
* ``POST  /runs/crons/search``            → search (returns list[Cron])
* ``POST  /runs/crons/count``             → count (returns int)
"""

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from aegra_api.core.auth_deps import auth_dependency, get_current_user
from aegra_api.core.auth_handlers import build_auth_context, handle_event
from aegra_api.core.orm import Thread as ThreadORM
from aegra_api.core.orm import get_session
from aegra_api.models import User
from aegra_api.models.crons import (
    CronCountRequest,
    CronCreate,
    CronResponse,
    CronSearchRequest,
    CronUpdate,
)
from aegra_api.models.errors import NOT_FOUND
from aegra_api.services.cron_service import CronService, _cron_to_response, get_cron_service

router = APIRouter(tags=["Crons"], dependencies=auth_dependency)


async def _authorize_cron_create(
    user: User,
    request: CronCreate,
    *,
    thread_id: str | None,
) -> None:
    """Fire the multi-resource auth chain expected by cron creation.

    Spec contract: a cron create touches three resources, so a handler can deny
    at any layer (caller may have crons access but not the underlying assistant
    or thread). Chain mirrors the LangGraph SDK reference.

    * thread-scoped create: ``crons.create`` → ``assistants.read`` → ``threads.read``
    * stateless create:     ``crons.create`` → ``assistants.read`` → ``threads.search``
    """
    cron_value: dict[str, object] = request.model_dump()
    if thread_id is not None:
        cron_value["thread_id"] = thread_id
    await handle_event(build_auth_context(user, "crons", "create"), cron_value)

    await handle_event(
        build_auth_context(user, "assistants", "read"),
        {"assistant_id": request.assistant_id},
    )

    if thread_id is not None:
        await handle_event(
            build_auth_context(user, "threads", "read"),
            {"thread_id": thread_id},
        )
    else:
        await handle_event(build_auth_context(user, "threads", "search"), {})


# ---------------------------------------------------------------------------
# Create (stateless) – POST /runs/crons → returns Cron
# ---------------------------------------------------------------------------


@router.post("/runs/crons", response_model=CronResponse)
async def create_cron(
    request: CronCreate,
    user: User = Depends(get_current_user),
    service: CronService = Depends(get_cron_service),
) -> CronResponse:
    """Create a cron job that fires on a schedule (stateless).

    Returns the persisted ``Cron``, matching LangSmith Deployments. No run
    starts at creation; the first run fires at ``next_run_date``.
    """
    await _authorize_cron_create(user, request, thread_id=None)
    cron = await service.create_cron(request, user.identity)
    return _cron_to_response(cron)


# ---------------------------------------------------------------------------
# Create for thread – POST /threads/{thread_id}/runs/crons → returns Cron
# ---------------------------------------------------------------------------


@router.post("/threads/{thread_id}/runs/crons", response_model=CronResponse)
async def create_cron_for_thread(
    thread_id: str,
    request: CronCreate,
    user: User = Depends(get_current_user),
    service: CronService = Depends(get_cron_service),
    session: AsyncSession = Depends(get_session),
) -> CronResponse:
    """Create a cron job bound to an existing thread.

    The thread is reused for every scheduled run. Returns the persisted
    ``Cron``; the first run fires at ``next_run_date``.
    """
    # Ownership gate at entry: binding a cron onto a thread the caller doesn't
    # own would run every future firing against it. Unlike create_run (which
    # auto-creates a missing thread it then owns), a thread-bound cron names an
    # existing thread, so a missing row is also a 404 — otherwise _prepare_run
    # would silently create a thread the user never intended to bind to.
    existing_thread = await session.scalar(select(ThreadORM).where(ThreadORM.thread_id == thread_id))
    if existing_thread is None or existing_thread.user_id != user.identity:
        raise HTTPException(404, f"Thread '{thread_id}' not found")

    await _authorize_cron_create(user, request, thread_id=thread_id)
    cron = await service.create_cron(request, user.identity, thread_id=thread_id)
    return _cron_to_response(cron)


# ---------------------------------------------------------------------------
# Update – PATCH /runs/crons/{cron_id} → returns Cron
# ---------------------------------------------------------------------------


@router.patch("/runs/crons/{cron_id}", response_model=CronResponse, responses={**NOT_FOUND})
async def update_cron(
    cron_id: str,
    request: CronUpdate,
    user: User = Depends(get_current_user),
    service: CronService = Depends(get_cron_service),
) -> CronResponse:
    """Update an existing cron job.

    Only provided fields are updated (partial patch). Returns the full
    ``Cron`` object after update.
    """
    ctx = build_auth_context(user, "crons", "update")
    value = {"cron_id": cron_id, **request.model_dump(exclude_none=True)}
    await handle_event(ctx, value)

    return await service.update_cron(cron_id, request, user.identity)


# ---------------------------------------------------------------------------
# Delete – DELETE /runs/crons/{cron_id} → 204
# ---------------------------------------------------------------------------


@router.delete("/runs/crons/{cron_id}", status_code=204, responses={**NOT_FOUND})
async def delete_cron(
    cron_id: str,
    user: User = Depends(get_current_user),
    service: CronService = Depends(get_cron_service),
) -> Response:
    """Delete a cron job."""
    ctx = build_auth_context(user, "crons", "delete")
    value = {"cron_id": cron_id}
    await handle_event(ctx, value)

    await service.delete_cron(cron_id, user.identity)
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Search – POST /runs/crons/search → list[Cron]
# ---------------------------------------------------------------------------


@router.post("/runs/crons/search", response_model=list[CronResponse])
async def search_crons(
    request: CronSearchRequest,
    user: User = Depends(get_current_user),
    service: CronService = Depends(get_cron_service),
) -> list[CronResponse]:
    """Search cron jobs with filters and pagination."""
    ctx = build_auth_context(user, "crons", "search")
    value = request.model_dump(exclude_none=True)
    await handle_event(ctx, value)

    return await service.search_crons(request, user.identity)


# ---------------------------------------------------------------------------
# Count – POST /runs/crons/count → int
# ---------------------------------------------------------------------------


@router.post("/runs/crons/count")
async def count_crons(
    request: CronCountRequest,
    user: User = Depends(get_current_user),
    service: CronService = Depends(get_cron_service),
) -> int:
    """Count cron jobs matching filters."""
    ctx = build_auth_context(user, "crons", "search")
    value = request.model_dump(exclude_none=True)
    await handle_event(ctx, value)

    return await service.count_crons(request, user.identity)

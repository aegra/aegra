"""Cron job endpoints for Agent Protocol.

Implements the six endpoints consumed by the LangGraph SDK ``CronsClient``:

* ``POST  /runs/crons``                  → create (stateless, returns Run)
* ``POST  /threads/{thread_id}/runs/crons`` → create for thread (returns Run)
* ``PATCH /runs/crons/{cron_id}``         → update (returns Cron)
* ``DELETE /runs/crons/{cron_id}``        → delete (204)
* ``POST  /runs/crons/search``            → search (returns list[Cron])
* ``POST  /runs/crons/count``             → count (returns int)
"""

from uuid import uuid4

import structlog
from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from aegra_api.core.auth_deps import auth_dependency, get_current_user
from aegra_api.core.auth_handlers import build_auth_context, handle_event
from aegra_api.core.orm import Cron as CronORM
from aegra_api.core.orm import Thread as ThreadORM
from aegra_api.core.orm import get_session
from aegra_api.models import Run, User
from aegra_api.models.crons import (
    CronCountRequest,
    CronCreate,
    CronResponse,
    CronSearchRequest,
    CronUpdate,
)
from aegra_api.models.errors import NOT_FOUND
from aegra_api.services.cron_scheduler import _build_run_create
from aegra_api.services.cron_service import (
    CronService,
    _cron_to_response,
    get_cron_service,
    should_delete_stateless_thread,
)
from aegra_api.services.run_cleanup import delete_thread_by_id, schedule_background_cleanup
from aegra_api.services.run_preparation import _prepare_run

router = APIRouter(tags=["Crons"], dependencies=auth_dependency)
logger = structlog.getLogger(__name__)


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
# Create (stateless) – POST /runs/crons → returns Run
# ---------------------------------------------------------------------------


@router.post("/runs/crons", response_model=Run | CronResponse)
async def create_cron(
    request: CronCreate,
    user: User = Depends(get_current_user),
    service: CronService = Depends(get_cron_service),
    session: AsyncSession = Depends(get_session),
) -> Run | CronResponse:
    """Create a cron job that fires on a schedule (stateless).

    Persists the cron record, then triggers the first run immediately and
    returns the ``Run`` object (matching LangGraph SDK ``create()`` contract).
    When the caller passes ``enabled=False`` the first run is suppressed and
    the response is the persisted ``Cron`` instead.
    """
    await _authorize_cron_create(user, request, thread_id=None)
    return await _create_cron_atomic(request, user, service, session)


# ---------------------------------------------------------------------------
# Create for thread – POST /threads/{thread_id}/runs/crons → returns Run
# ---------------------------------------------------------------------------


@router.post("/threads/{thread_id}/runs/crons", response_model=Run | CronResponse)
async def create_cron_for_thread(
    thread_id: str,
    request: CronCreate,
    user: User = Depends(get_current_user),
    service: CronService = Depends(get_cron_service),
    session: AsyncSession = Depends(get_session),
) -> Run | CronResponse:
    """Create a cron job bound to an existing thread.

    The thread is reused for every scheduled run. Triggers the first run
    immediately and returns the ``Run`` object. When ``enabled=False`` is
    passed the first run is suppressed and the persisted cron is returned.
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
    return await _create_cron_atomic(request, user, service, session, thread_id=thread_id)


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _create_cron_atomic(
    request: CronCreate,
    user: User,
    service: CronService,
    session: AsyncSession,
    *,
    thread_id: str | None = None,
) -> Run | CronResponse:
    """Insert the cron and (when enabled) trigger the first run.

    A failed first run deletes the cron so the operator does not end up with
    an orphan record that the scheduler will keep firing on the next tick.
    When ``request.enabled`` is False the first run is suppressed entirely
    and the persisted ``Cron`` is returned instead of a ``Run``.
    """
    cron = await service.create_cron(request, user.identity, thread_id=thread_id)

    if request.enabled is False:
        return _cron_to_response(cron)

    try:
        return await _trigger_first_run(session, cron, user, thread_id=thread_id)
    except Exception:
        try:
            await service.delete_cron(cron.cron_id, user.identity)
        except Exception:
            logger.exception(
                "Failed to roll back cron after first-run setup error",
                cron_id=cron.cron_id,
            )
        raise


async def _trigger_first_run(
    session: AsyncSession,
    cron: CronORM,
    user: User,
    *,
    thread_id: str | None = None,
) -> Run:
    """Create the initial run for a newly created cron job."""
    effective_thread_id = thread_id or cron.thread_id or str(uuid4())
    should_delete_thread = thread_id is None and should_delete_stateless_thread(cron)

    run_request = _build_run_create(cron)

    try:
        _run_id, run, _job = await _prepare_run(
            session,
            effective_thread_id,
            run_request,
            user,
            initial_status="pending",
        )
    except Exception:
        if should_delete_thread:
            try:
                await delete_thread_by_id(effective_thread_id, user.identity)
            except Exception:
                logger.exception(
                    "Failed to delete stateless cron thread after initial run setup error",
                    thread_id=effective_thread_id,
                    cron_id=cron.cron_id,
                )
        raise

    if should_delete_thread:
        schedule_background_cleanup(_run_id, effective_thread_id, user.identity)

    return run

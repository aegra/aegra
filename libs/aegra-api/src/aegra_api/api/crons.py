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
from fastapi import APIRouter, Depends, Response
from sqlalchemy.ext.asyncio import AsyncSession

from aegra_api.core.auth_deps import auth_dependency, get_current_user
from aegra_api.core.auth_handlers import build_auth_context, handle_event
from aegra_api.core.orm import Cron as CronORM
from aegra_api.core.orm import get_session
from aegra_api.models import Run, RunCreate, User
from aegra_api.models.crons import (
    CronCountRequest,
    CronCreate,
    CronResponse,
    CronSearchRequest,
    CronUpdate,
)
from aegra_api.models.errors import NOT_FOUND
from aegra_api.services.cron_service import CronService, get_cron_service
from aegra_api.services.run_cleanup import delete_thread_by_id, schedule_background_cleanup
from aegra_api.services.run_preparation import _prepare_run

router = APIRouter(tags=["Crons"], dependencies=auth_dependency)
logger = structlog.getLogger(__name__)


# ---------------------------------------------------------------------------
# Create (stateless) – POST /runs/crons → returns Run
# ---------------------------------------------------------------------------


@router.post("/runs/crons", response_model=Run)
async def create_cron(
    request: CronCreate,
    user: User = Depends(get_current_user),
    service: CronService = Depends(get_cron_service),
    session: AsyncSession = Depends(get_session),
) -> Run:
    """Create a cron job that fires on a schedule (stateless).

    Persists the cron record, then triggers the first run immediately and
    returns the ``Run`` object (matching LangGraph SDK ``create()`` contract).
    """
    ctx = build_auth_context(user, "crons", "create")
    value = request.model_dump()
    await handle_event(ctx, value)

    cron = await service.create_cron(request, user.identity)
    return await _trigger_first_run(session, cron, user)


# ---------------------------------------------------------------------------
# Create for thread – POST /threads/{thread_id}/runs/crons → returns Run
# ---------------------------------------------------------------------------


@router.post("/threads/{thread_id}/runs/crons", response_model=Run)
async def create_cron_for_thread(
    thread_id: str,
    request: CronCreate,
    user: User = Depends(get_current_user),
    service: CronService = Depends(get_cron_service),
    session: AsyncSession = Depends(get_session),
) -> Run:
    """Create a cron job bound to an existing thread.

    The thread is reused for every scheduled run. Triggers the first run
    immediately and returns the ``Run`` object.
    """
    ctx = build_auth_context(user, "crons", "create")
    value = {**request.model_dump(), "thread_id": thread_id}
    await handle_event(ctx, value)

    cron = await service.create_cron(request, user.identity, thread_id=thread_id)
    return await _trigger_first_run(session, cron, user, thread_id=thread_id)


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


async def _trigger_first_run(
    session: AsyncSession,
    cron: CronORM,
    user: User,
    *,
    thread_id: str | None = None,
) -> Run:
    """Create the initial run for a newly created cron job."""
    payload = cron.payload or {}
    effective_thread_id = thread_id or cron.thread_id or str(uuid4())
    should_delete_thread = thread_id is None and cron.thread_id is None and cron.on_run_completed != "keep"

    run_request = RunCreate(
        assistant_id=cron.assistant_id,
        input=payload.get("input"),
        config=payload.get("config"),
        context=payload.get("context"),
        checkpoint=payload.get("checkpoint"),
        interrupt_before=payload.get("interrupt_before"),
        interrupt_after=payload.get("interrupt_after"),
        stream_subgraphs=payload.get("stream_subgraphs"),
        stream_mode=payload.get("stream_mode"),
        multitask_strategy=payload.get("multitask_strategy"),
        metadata=None,
    )

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

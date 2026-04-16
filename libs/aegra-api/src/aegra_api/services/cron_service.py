"""Service layer for cron job business logic.

Handles CRUD operations on cron records and delegates run creation
to the existing run preparation pipeline.
"""

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import structlog
from croniter import croniter
from fastapi import Depends, HTTPException
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from aegra_api.core.orm import Assistant as AssistantORM
from aegra_api.core.orm import Cron as CronORM
from aegra_api.core.orm import get_session
from aegra_api.models.crons import (
    CronCountRequest,
    CronCreate,
    CronResponse,
    CronSearchRequest,
    CronUpdate,
)
from aegra_api.services.langgraph_service import LangGraphService, get_langgraph_service
from aegra_api.settings import settings
from aegra_api.utils.assistants import resolve_assistant_id

logger = structlog.getLogger(__name__)


def _build_payload(request: CronCreate | CronUpdate) -> dict[str, Any]:
    """Extract run-related fields into the payload JSONB blob."""
    payload: dict[str, Any] = {}
    for field in (
        "input",
        "config",
        "context",
        "checkpoint_during",
        "interrupt_before",
        "interrupt_after",
        "webhook",
        "multitask_strategy",
        "stream_mode",
        "stream_subgraphs",
        "stream_resumable",
        "durability",
        "timezone",
    ):
        value = getattr(request, field, None)
        if value is not None:
            payload[field] = value
    return payload


def _is_seconds_cron(schedule: str) -> bool:
    """Return True when schedule uses 6 fields (seconds as first field)."""
    return len(schedule.split()) == 6


def _is_valid_schedule(schedule: str) -> bool:
    """Validate a cron expression (5-field standard or 6-field seconds-first)."""
    if _is_seconds_cron(schedule):
        try:
            croniter(schedule, datetime.now(UTC), second_at_beginning=True)
            return True
        except (ValueError, KeyError):
            return False
    return croniter.is_valid(schedule)


def _compute_next_run(
    schedule: str,
    *,
    now: datetime | None = None,
    timezone: str | None = None,
) -> datetime:
    """Compute the next run date from a cron schedule expression, returned in UTC.

    When *timezone* is an IANA timezone string (e.g. ``"America/New_York"``),
    the schedule is evaluated in that timezone so wall-clock expressions like
    ``0 9 * * *`` fire at 09:00 local time instead of 09:00 UTC.

    Supports both standard 5-field expressions and 6-field seconds-first
    expressions (e.g. ``*/30 * * * * *`` = every 30 seconds).
    """
    base = now or datetime.now(UTC)
    if timezone:
        tz = ZoneInfo(timezone)
        base = base.astimezone(tz)
    second_at_beginning = _is_seconds_cron(schedule)
    result = croniter(schedule, base, second_at_beginning=second_at_beginning).get_next(datetime)
    # Normalise to UTC for storage regardless of what timezone croniter used.
    if result.tzinfo is None:
        return result.replace(tzinfo=UTC)
    return result.astimezone(UTC)


def _cron_to_response(row: CronORM) -> CronResponse:
    """Convert ORM row to Pydantic response, mapping ``metadata_dict`` → ``metadata``."""
    return CronResponse(
        cron_id=str(row.cron_id),
        assistant_id=str(row.assistant_id),
        thread_id=row.thread_id,
        on_run_completed=row.on_run_completed,
        end_time=row.end_time,
        schedule=row.schedule,
        created_at=row.created_at,
        updated_at=row.updated_at,
        payload=row.payload or {},
        user_id=row.user_id,
        next_run_date=row.next_run_date,
        metadata=row.metadata_dict or {},
        enabled=row.enabled,
    )


class CronService:
    """CRUD service for cron jobs."""

    def __init__(self, session: AsyncSession, langgraph_service: LangGraphService | None = None) -> None:
        """Store the database session and optional LangGraph dependency."""
        self.session = session
        self.langgraph_service = langgraph_service

    def _get_available_graphs(self) -> dict[str, Any]:
        """Return the configured graphs, requiring LangGraphService when needed."""
        if self.langgraph_service is None:
            raise RuntimeError("LangGraphService is required for graph-aware cron operations")
        return self.langgraph_service.list_graphs()

    def _resolve_assistant_identifier(self, requested_id: str) -> str:
        """Resolve graph IDs to their canonical assistant IDs for cron operations."""
        available_graphs = self._get_available_graphs()
        return resolve_assistant_id(requested_id, available_graphs)

    async def create_cron(
        self,
        request: CronCreate,
        user_identity: str,
        *,
        thread_id: str | None = None,
    ) -> CronORM:
        """Create a new cron job record.

        Returns the ORM row so the caller (API layer) can also trigger
        the first run and return the ``Run`` response.
        """
        # Validate schedule expression
        if not _is_valid_schedule(request.schedule):
            raise HTTPException(422, f"Invalid cron schedule: {request.schedule}")

        # Validate timezone when provided
        if request.timezone is not None:
            try:
                ZoneInfo(request.timezone)
            except (ZoneInfoNotFoundError, KeyError):
                raise HTTPException(422, f"Invalid timezone: {request.timezone}")

        available_graphs = self._get_available_graphs()
        requested_assistant_id = str(request.assistant_id)
        resolved_assistant_id = resolve_assistant_id(requested_assistant_id, available_graphs)

        # Validate assistant exists
        assistant = await self.session.scalar(
            select(AssistantORM).where(AssistantORM.assistant_id == resolved_assistant_id)
        )
        if not assistant:
            raise HTTPException(404, f"Assistant '{request.assistant_id}' not found")

        # Validate the assistant's graph exists
        if assistant.graph_id not in available_graphs:
            raise HTTPException(404, f"Graph '{assistant.graph_id}' not found for assistant")

        payload = _build_payload(request)
        now = datetime.now(UTC)
        # Advance past the immediate first occurrence since _trigger_first_run
        # fires a run right away. We skip to the second occurrence so the
        # scheduler does not fire a duplicate run seconds after creation.
        first_occ = _compute_next_run(request.schedule, now=now, timezone=request.timezone)
        next_run = _compute_next_run(request.schedule, now=first_occ, timezone=request.timezone)

        cron_orm = CronORM(
            cron_id=str(uuid4()),
            assistant_id=resolved_assistant_id,
            thread_id=thread_id,
            user_id=user_identity,
            schedule=request.schedule,
            payload=payload,
            metadata_dict=request.metadata or {},
            on_run_completed=request.on_run_completed,
            enabled=request.enabled if request.enabled is not None else True,
            end_time=request.end_time,
            next_run_date=next_run,
            created_at=now,
            updated_at=now,
        )
        self.session.add(cron_orm)
        await self.session.commit()
        await self.session.refresh(cron_orm)

        logger.info("Created cron job", cron_id=cron_orm.cron_id, schedule=request.schedule)
        return cron_orm

    async def update_cron(
        self,
        cron_id: str,
        request: CronUpdate,
        user_identity: str,
    ) -> CronResponse:
        """Update an existing cron job and return the updated ``CronResponse``."""
        cron = await self._get_cron_or_404(cron_id, user_identity)

        values: dict[str, Any] = {"updated_at": datetime.now(UTC)}
        existing_payload = dict(cron.payload or {})

        if request.timezone is not None:
            try:
                ZoneInfo(request.timezone)
            except (ZoneInfoNotFoundError, KeyError):
                raise HTTPException(422, f"Invalid timezone: {request.timezone}")

        # Schedule
        if request.schedule is not None:
            if not _is_valid_schedule(request.schedule):
                raise HTTPException(422, f"Invalid cron schedule: {request.schedule}")
            values["schedule"] = request.schedule

        # Simple scalar fields
        if request.end_time is not None:
            values["end_time"] = request.end_time
        if request.on_run_completed is not None:
            values["on_run_completed"] = request.on_run_completed
        if request.enabled is not None:
            values["enabled"] = request.enabled

        # Metadata (full replace, matching SDK behavior)
        if request.metadata is not None:
            values["metadata_dict"] = request.metadata

        # Merge payload fields into existing payload
        new_payload = _build_payload(request)
        if new_payload:
            existing_payload.update(new_payload)
            values["payload"] = existing_payload

        if request.schedule is not None or request.timezone is not None:
            schedule = request.schedule if request.schedule is not None else cron.schedule
            timezone = existing_payload.get("timezone")
            values["next_run_date"] = _compute_next_run(schedule, timezone=timezone)

        await self.session.execute(update(CronORM).where(CronORM.cron_id == cron_id).values(**values))
        await self.session.commit()

        updated = await self.session.scalar(select(CronORM).where(CronORM.cron_id == cron_id))
        if not updated:
            raise HTTPException(404, f"Cron '{cron_id}' not found")

        logger.info("Updated cron job", cron_id=cron_id)
        return _cron_to_response(updated)

    async def delete_cron(self, cron_id: str, user_identity: str) -> None:
        """Delete a cron job."""
        cron = await self._get_cron_or_404(cron_id, user_identity)
        await self.session.delete(cron)
        await self.session.commit()
        logger.info("Deleted cron job", cron_id=cron_id)

    async def search_crons(
        self,
        request: CronSearchRequest,
        user_identity: str,
    ) -> list[CronResponse]:
        """Search cron jobs with filters, pagination, and sorting."""
        stmt = select(CronORM).where(CronORM.user_id == user_identity)

        if request.assistant_id is not None:
            resolved_assistant_id = self._resolve_assistant_identifier(request.assistant_id)
            stmt = stmt.where(CronORM.assistant_id == resolved_assistant_id)
        if request.thread_id is not None:
            stmt = stmt.where(CronORM.thread_id == request.thread_id)
        if request.enabled is not None:
            stmt = stmt.where(CronORM.enabled == request.enabled)

        # Sorting
        sort_column = CronORM.created_at
        if request.sort_by == "next_run_date":
            sort_column = CronORM.next_run_date
        elif request.sort_by == "updated_at":
            sort_column = CronORM.updated_at

        if request.sort_order == "desc":
            stmt = stmt.order_by(sort_column.desc())
        else:
            stmt = stmt.order_by(sort_column.asc())

        stmt = stmt.offset(request.offset).limit(request.limit)

        result = await self.session.scalars(stmt)
        return [_cron_to_response(row) for row in result.all()]

    async def count_crons(
        self,
        request: CronCountRequest,
        user_identity: str,
    ) -> int:
        """Count cron jobs matching filters."""
        stmt = select(func.count()).select_from(CronORM).where(CronORM.user_id == user_identity)

        if request.assistant_id is not None:
            resolved_assistant_id = self._resolve_assistant_identifier(request.assistant_id)
            stmt = stmt.where(CronORM.assistant_id == resolved_assistant_id)
        if request.thread_id is not None:
            stmt = stmt.where(CronORM.thread_id == request.thread_id)

        total = await self.session.scalar(stmt)
        return total or 0

    async def get_due_crons(self, now: datetime | None = None) -> list[CronORM]:
        """Atomically claim enabled cron jobs whose ``next_run_date`` is in the past.

        The claim temporarily moves ``next_run_date`` forward by one poll interval
        so concurrent scheduler instances do not pick the same rows before they are
        processed. Successful execution overwrites this temporary value with the real
        next occurrence; failed execution leaves the cron claimable again on a later
        poll instead of skipping the schedule permanently.
        """
        now = now or datetime.now(UTC)
        claimed_until = now + timedelta(seconds=settings.cron.CRON_POLL_INTERVAL_SECONDS)
        stmt = (
            update(CronORM)
            .where(
                CronORM.enabled.is_(True),
                CronORM.next_run_date <= now,
            )
            .values(next_run_date=claimed_until, updated_at=now)
            .returning(CronORM)
        )
        result = await self.session.execute(stmt)
        await self.session.commit()
        return list(result.scalars().all())

    async def advance_next_run(self, cron: CronORM) -> None:
        """Advance ``next_run_date`` to the next occurrence after *now*.

        If the cron has an ``end_time`` that has passed, disable it instead.
        """
        now = datetime.now(UTC)
        if cron.end_time and now >= cron.end_time:
            await self.session.execute(
                update(CronORM).where(CronORM.cron_id == cron.cron_id).values(enabled=False, updated_at=now)
            )
        else:
            timezone = (cron.payload or {}).get("timezone")
            next_run = _compute_next_run(cron.schedule, now=now, timezone=timezone)
            await self.session.execute(
                update(CronORM).where(CronORM.cron_id == cron.cron_id).values(next_run_date=next_run, updated_at=now)
            )
        await self.session.commit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get_cron_or_404(self, cron_id: str, user_identity: str) -> CronORM:
        """Return the user's cron row or raise a 404 when it does not exist."""
        cron = await self.session.scalar(
            select(CronORM).where(CronORM.cron_id == cron_id, CronORM.user_id == user_identity)
        )
        if not cron:
            raise HTTPException(404, f"Cron '{cron_id}' not found")
        return cron


def get_cron_service(
    session: AsyncSession = Depends(get_session),
    langgraph_service: LangGraphService = Depends(get_langgraph_service),
) -> CronService:
    """FastAPI dependency injection for CronService."""
    return CronService(session, langgraph_service)

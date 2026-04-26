"""Background scheduler that fires due cron jobs.

Wakes up every ``CRON_POLL_INTERVAL_SECONDS`` (default 60 s), queries for
enabled crons whose ``next_run_date`` has passed, triggers a run for each
one via the existing run-preparation pipeline, then advances the cron to
its next occurrence.

Follows the same ``start()/stop()`` lifecycle pattern used by
:class:`aegra_api.services.lease_reaper.LeaseReaper`.
"""

import asyncio
import contextlib
from datetime import UTC, datetime
from uuid import uuid4

import structlog
from fastapi import HTTPException
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from aegra_api.core.orm import Cron as CronORM
from aegra_api.core.orm import _get_session_maker
from aegra_api.models import RunCreate, User
from aegra_api.services.cron_service import CronService, _compute_next_run, _is_seconds_cron
from aegra_api.services.run_cleanup import delete_thread_by_id, schedule_background_cleanup
from aegra_api.services.run_preparation import _prepare_run
from aegra_api.settings import settings

logger = structlog.getLogger(__name__)


def _should_delete_stateless_thread(cron: CronORM) -> bool:
    """Return True when the cron should delete its ephemeral thread after completion."""
    return cron.thread_id is None and cron.on_run_completed != "keep"


class CronScheduler:
    """Periodically fires due cron jobs by creating runs."""

    def __init__(self) -> None:
        """Initialize the scheduler state for the background polling loop."""
        self._task: asyncio.Task[None] | None = None
        self._running = False

    async def start(self) -> None:
        """Start the background polling task if the scheduler is enabled."""
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info(
            "Cron scheduler started",
            interval_seconds=settings.cron.CRON_POLL_INTERVAL_SECONDS,
        )

    async def stop(self) -> None:
        """Stop the background polling task and wait for cancellation to finish."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        logger.info("Cron scheduler stopped")

    async def _loop(self) -> None:
        """Sleep for the configured interval and trigger cron polling until stopped."""
        interval = settings.cron.CRON_POLL_INTERVAL_SECONDS
        while self._running:
            try:
                await asyncio.sleep(interval)
                await self._tick()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in cron scheduler tick")

    async def _tick(self) -> None:
        """Find all due crons and fire a run for each one."""
        now = datetime.now(UTC)
        maker = _get_session_maker()

        async with maker() as session:
            due_crons = await self._find_due_crons(session, now)
            if not due_crons:
                logger.debug("Cron tick: no jobs due")
                return

            logger.info("Cron tick: found due jobs", count=len(due_crons))

            for cron in due_crons:
                try:
                    await self._fire_cron(session, cron)
                except Exception:
                    logger.exception("Failed to fire cron job", cron_id=cron.cron_id)

    @staticmethod
    async def _find_due_crons(session: AsyncSession, now: datetime) -> list[CronORM]:
        """Return enabled crons whose next_run_date has passed."""
        return await CronService(session).get_due_crons(now)

    @staticmethod
    async def _fire_cron(session: AsyncSession, cron: CronORM) -> None:
        """Create a run from the cron's payload and advance next_run_date."""
        payload = cron.payload or {}
        now = datetime.now(UTC)
        should_delete_thread = _should_delete_stateless_thread(cron)
        run_created = False

        # Build a RunCreate from the stored payload
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

        # Determine thread_id: use cron.thread_id if bound, else create ephemeral
        thread_id = cron.thread_id or str(uuid4())

        user = User(
            identity=cron.user_id,
            display_name="cron-scheduler",
            is_authenticated=True,
        )

        try:
            _run_id, _run, _job = await _prepare_run(
                session,
                thread_id,
                run_request,
                user,
                initial_status="pending",
            )
            run_created = True
            logger.info("Cron fired run", cron_id=cron.cron_id, run_id=_run_id, thread_id=thread_id)
            if should_delete_thread:
                schedule_background_cleanup(_run_id, thread_id, cron.user_id)
        except HTTPException as exc:
            logger.error(
                "Cron run creation failed",
                cron_id=cron.cron_id,
                status_code=exc.status_code,
                detail=exc.detail,
            )
            if should_delete_thread:
                await CronScheduler._cleanup_failed_stateless_thread(thread_id, cron)
        except Exception:
            logger.exception("Cron run creation failed unexpectedly", cron_id=cron.cron_id)
            if should_delete_thread:
                await CronScheduler._cleanup_failed_stateless_thread(thread_id, cron)

        # Advance to next occurrence (or disable if past end_time)
        if run_created:
            if cron.end_time and now >= cron.end_time:
                await session.execute(
                    update(CronORM).where(CronORM.cron_id == cron.cron_id).values(enabled=False, updated_at=now)
                )
            else:
                timezone = (cron.payload or {}).get("timezone")
                next_run = _compute_next_run(cron.schedule, now=now, timezone=timezone)
                logger.debug(
                    "Advancing cron schedule",
                    cron_id=cron.cron_id,
                    schedule=repr(cron.schedule),
                    field_count=len(cron.schedule.split()),
                    is_seconds_cron=_is_seconds_cron(cron.schedule),
                    now=now.isoformat(),
                    next_run=next_run.isoformat(),
                )
                await session.execute(
                    update(CronORM)
                    .where(CronORM.cron_id == cron.cron_id)
                    .values(next_run_date=next_run, updated_at=now)
                )
        await session.commit()

    @staticmethod
    async def _cleanup_failed_stateless_thread(thread_id: str, cron: CronORM) -> None:
        """Delete a stateless cron thread when run preparation fails mid-flight."""
        try:
            await delete_thread_by_id(thread_id, cron.user_id)
        except Exception:
            logger.exception(
                "Failed to delete stateless cron thread after run setup error",
                thread_id=thread_id,
                cron_id=cron.cron_id,
            )


# Module-level singleton (matches executor / lease_reaper pattern)
cron_scheduler = CronScheduler()

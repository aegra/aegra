"""Abstract interface for run execution dispatch.

Follows the same strategy pattern as the broker abstraction:
one interface, two backends (local asyncio tasks vs Redis workers).
"""

from abc import ABC, abstractmethod
from datetime import UTC, datetime

import structlog
from sqlalchemy import select, update

from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import Thread as ThreadORM
from aegra_api.core.orm import _get_session_maker
from aegra_api.models.run_job import RunJob

logger = structlog.getLogger(__name__)

_OCCUPYING_RUN_STATUSES = ("running", "pending")


class BaseExecutor(ABC):
    """Dispatches RunJobs for execution and tracks their lifecycle."""

    # Cleared in stop() so a run finalizing during shutdown does not promote a
    # queued run that would be orphaned when the process exits. Queued runs are
    # picked up on the next start by recovery instead.
    _accepting: bool = True

    @abstractmethod
    async def submit(self, job: RunJob) -> None:
        """Enqueue a job for execution. Returns immediately."""

    @abstractmethod
    async def wait_for_completion(self, run_id: str, *, timeout: float = 300.0) -> None:
        """Block until the run reaches a terminal state or timeout expires."""

    @abstractmethod
    async def start(self) -> None:
        """Initialize resources (called during app startup)."""

    @abstractmethod
    async def stop(self) -> None:
        """Drain in-flight work and release resources (called during shutdown)."""

    async def dispatch_next_for_thread(self, thread_id: str) -> None:
        """Promote the oldest queued run on a thread and submit it.

        Called after a run finalizes to start the next double-texted run.
        Locks the thread row so concurrent finalizes/reapers can't double
        dispatch, and no-ops if another run is already occupying the thread.
        """
        if not self._accepting:
            return
        maker = _get_session_maker()
        async with maker() as session:
            thread = await session.scalar(select(ThreadORM).where(ThreadORM.thread_id == thread_id).with_for_update())
            if thread is not None and thread.status == "interrupted":
                # Thread is paused on a human-in-the-loop interrupt(). Promoting a queued
                # fresh-input run would run it against the paused checkpoint and consume the
                # pending interrupt — the admission 409 guard only covers run creation, so the
                # guard must also hold here. Leave queued runs parked until a resume clears it.
                return
            occupying = await session.scalar(
                select(RunORM.run_id)
                .where(RunORM.thread_id == thread_id, RunORM.status.in_(_OCCUPYING_RUN_STATUSES))
                .limit(1)
            )
            if occupying is not None:
                # Heal a stale 'idle' a pre-empted run's finalize may have written.
                await session.execute(
                    update(ThreadORM)
                    .where(ThreadORM.thread_id == thread_id)
                    .values(status="busy", updated_at=datetime.now(UTC))
                )
                await session.commit()
                return

            run_orm = await session.scalar(
                select(RunORM)
                .where(RunORM.thread_id == thread_id, RunORM.status == "queued")
                .order_by(RunORM.created_at.asc())
                .limit(1)
                .with_for_update(skip_locked=True)
            )
            if run_orm is None:
                return

            run_orm.status = "pending"
            # Stamp updated_at so the stuck-pending reaper measures time since promotion,
            # not since the run was first queued (created_at), and skips fresh promotions.
            run_orm.updated_at = datetime.now(UTC)
            await session.execute(
                update(ThreadORM)
                .where(ThreadORM.thread_id == thread_id)
                .values(status="busy", updated_at=datetime.now(UTC))
            )
            job = RunJob.from_run_orm(run_orm)
            await session.commit()

        logger.info("Dispatched queued run", run_id=job.identity.run_id, thread_id=thread_id)
        await self.submit(job)

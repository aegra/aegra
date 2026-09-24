"""Abstract interface for run execution dispatch.

Follows the same strategy pattern as the broker abstraction:
one interface, two backends (local asyncio tasks vs Redis workers).
"""

import asyncio
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any

import structlog
from sqlalchemy import select, update

from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import Thread as ThreadORM
from aegra_api.core.orm import _get_session_maker
from aegra_api.models.run_job import RunJob
from aegra_api.settings import settings

logger = structlog.getLogger(__name__)

_OCCUPYING_RUN_STATUSES = ("running", "pending")
# Sentinel: a promotion attempt failed a corrupt row and the caller should try the next one.
_RETRY = object()


class BaseExecutor(ABC):
    """Dispatches RunJobs for execution and tracks their lifecycle."""

    # Cleared by _begin_shutdown() so a run finalizing during shutdown does not
    # promote a queued run that would be orphaned when the process exits. Queued
    # runs are picked up by recovery (startup sweep / reaper) instead.
    _accepting: bool = True
    # Dispatches past the _accepting check; _begin_shutdown() waits for them so a
    # promotion cannot commit queued->pending and submit() behind the drain.
    _inflight_dispatches: int = 0
    _dispatches_idle: asyncio.Event | None = None

    @abstractmethod
    async def submit(self, job: RunJob) -> None:
        """Enqueue a job for execution. Returns immediately."""

    @abstractmethod
    async def wait_for_completion(self, run_id: str, *, timeout: float = 300.0) -> None:
        """Block until the run reaches a terminal state.

        Raises:
            TimeoutError: The run was still running after *timeout* seconds.
        """

    @abstractmethod
    async def start(self) -> None:
        """Initialize resources (called during app startup)."""

    @abstractmethod
    async def stop(self) -> None:
        """Drain in-flight work and release resources (called during shutdown)."""

    async def _begin_shutdown(self) -> None:
        """Stop accepting queued-run dispatches and wait for the in-flight ones.

        Subclasses call this first in ``stop()``. A dispatch that passed the
        ``_accepting`` check has already counted itself in (no await between the
        two), so once this returns no dispatch can commit or submit behind the
        drain that follows.
        """
        self._accepting = False
        while self._inflight_dispatches > 0:
            if self._dispatches_idle is None:
                self._dispatches_idle = asyncio.Event()
            self._dispatches_idle.clear()
            await self._dispatches_idle.wait()

    async def dispatch_next_for_thread(self, thread_id: str) -> None:
        """Promote the oldest queued run on a thread and submit it.

        Called after a run finalizes (or is cancelled / pre-empted) to start the
        next double-texted run. Locks the thread row so concurrent finalizes and
        reapers can't double dispatch, and no-ops if another run is already
        occupying the thread or a human-in-the-loop pause holds it.
        """
        if not self._accepting:
            return
        self._inflight_dispatches += 1
        try:
            await self._dispatch_next_for_thread(thread_id)
        finally:
            self._inflight_dispatches -= 1
            if self._inflight_dispatches == 0 and self._dispatches_idle is not None:
                self._dispatches_idle.set()

    async def _dispatch_next_for_thread(self, thread_id: str) -> None:
        maker = _get_session_maker()
        while True:
            job = await self._promote_oldest_queued(maker, thread_id)
            if job is None:
                return
            if job is _RETRY:
                continue  # a corrupt row was failed; the thread is still free, try the next one
            logger.info("Dispatched queued run", run_id=job.identity.run_id, thread_id=thread_id)
            await self.submit(job)
            return

    async def _promote_oldest_queued(self, maker: Any, thread_id: str) -> RunJob | object | None:
        """One promotion attempt: the promoted RunJob, ``_RETRY`` after failing a corrupt row, else None."""
        async with maker() as session:
            thread = await session.scalar(select(ThreadORM).where(ThreadORM.thread_id == thread_id).with_for_update())
            if thread is None:
                # Deleted underneath us (its runs cascade with it): nothing to promote onto.
                return None
            if thread.status == "interrupted" and settings.multitask.MULTITASK_PAUSED_THREAD_POLICY == "reject":
                # Thread is paused on a human-in-the-loop interrupt(). Promoting a queued
                # fresh-input run would run it against the paused checkpoint and consume the
                # pending interrupt — the admission 409 guard only covers run creation, so the
                # guard must also hold here. Leave queued runs parked until a resume clears it.
                return None
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
                return None

            run_orm = await session.scalar(
                select(RunORM)
                .where(RunORM.thread_id == thread_id, RunORM.status == "queued")
                .order_by(RunORM.created_at.asc())
                .limit(1)
                .with_for_update(skip_locked=True)
            )
            if run_orm is None:
                return None

            try:
                job = RunJob.from_run_orm(run_orm)
            except (ValueError, KeyError, TypeError) as exc:
                # Corrupt/legacy row (missing or malformed execution_params): it can never run,
                # so fail it instead of letting every recovery sweep trip over it, and let the
                # caller try the run parked behind it right away.
                run_orm.status = "error"
                run_orm.error_message = f"Queued run cannot be dispatched: {exc!r}"
                run_orm.updated_at = datetime.now(UTC)
                await session.commit()
                logger.error("Queued run cannot be dispatched, marked error", run_id=run_orm.run_id, error=repr(exc))
                return _RETRY

            run_orm.status = "pending"
            # Stamp updated_at so the stuck-pending reaper measures time since promotion,
            # not since the run was first queued (created_at), and skips fresh promotions.
            run_orm.updated_at = datetime.now(UTC)
            await session.execute(
                update(ThreadORM)
                .where(ThreadORM.thread_id == thread_id)
                .values(status="busy", updated_at=datetime.now(UTC))
            )
            await session.commit()
        return job

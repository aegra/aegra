"""Background task that recovers runs with expired worker leases.

Periodically scans the runs table for rows where
``status='running' AND lease_expires_at < now()``. It atomically either
returns them to ``pending`` or marks their retry budget exhausted, then
re-enqueues only retryable run IDs. It also restarts threads whose queued
(double-texted) runs lost their dispatch wakeup.
"""

import asyncio
import contextlib
from datetime import UTC, datetime, timedelta

import structlog
from redis import RedisError
from sqlalchemy import select, update

from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import _get_session_maker
from aegra_api.core.redis_manager import redis_manager
from aegra_api.observability.metrics import REAPER_RECOVERED_RUNS
from aegra_api.services.run_status import ACTIVE_RUN_STATES, QUEUED_RUN_STATE, set_thread_status_if_no_active_runs
from aegra_api.settings import settings

logger = structlog.getLogger(__name__)


class LeaseReaper:
    """Recovers runs whose worker leases have expired."""

    def __init__(self) -> None:
        self._task: asyncio.Task[None] | None = None
        self._running = False

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info(
            "Lease reaper started",
            interval_seconds=settings.worker.REAPER_INTERVAL_SECONDS,
        )

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        logger.info("Lease reaper stopped")

    async def _loop(self) -> None:
        interval = settings.worker.REAPER_INTERVAL_SECONDS
        while self._running:
            await asyncio.sleep(interval)
            try:
                await self._reap()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in lease reaper")

    async def _reap(self) -> None:
        """Find crashed workers, stuck pending runs and stranded queues, recover them."""
        crashed, stuck_pending = await self._find_recoverable()

        if crashed:
            logger.warning("Reaping crashed worker runs", count=len(crashed), run_ids=crashed)
            retryable, exhausted = await self._recover_crashed_runs(crashed)
            if exhausted:
                REAPER_RECOVERED_RUNS.labels(outcome="crashed_exhausted").inc(len(exhausted))
            if retryable:
                pushed = await self._reenqueue(retryable)
                REAPER_RECOVERED_RUNS.labels(outcome="crashed_retried").inc(len(pushed))

        # Stuck pending: just re-enqueue (never executed, no retry budget)
        if stuck_pending:
            logger.warning("Re-enqueueing stuck pending runs", count=len(stuck_pending), run_ids=stuck_pending)
            pushed = await self._reenqueue(stuck_pending)
            REAPER_RECOVERED_RUNS.labels(outcome="stuck_pending").inc(len(pushed))

        # Stranded queued: a prior run's dispatch wakeup was lost (process died, or its
        # head run was just permanently failed above). Detected AFTER the crashed/stuck
        # handling so a freshly-failed head's successor is caught this cycle, not the next.
        stranded_queued = await self._find_stranded_queued_threads()
        if stranded_queued:
            logger.warning("Dispatching stranded queued runs", thread_count=len(stranded_queued))
            await self._dispatch_stranded_queued(stranded_queued)

        if not crashed and not stuck_pending and not stranded_queued:
            return

        logger.info(
            "Lease recovery complete",
            crashed_recovered=len(crashed),
            stuck_reenqueued=len(stuck_pending),
            stranded_dispatched=len(stranded_queued),
        )

    @staticmethod
    async def _find_stranded_queued_threads() -> list[str]:
        """Threads with a queued run but no running/pending run holding them."""
        maker = _get_session_maker()
        async with maker() as session:
            queued = {
                row[0]
                for row in (
                    await session.execute(select(RunORM.thread_id).where(RunORM.status == QUEUED_RUN_STATE).distinct())
                ).all()
            }
            if not queued:
                return []
            active = {
                row[0]
                for row in (
                    await session.execute(
                        select(RunORM.thread_id).where(RunORM.status.in_(ACTIVE_RUN_STATES)).distinct()
                    )
                ).all()
            }
            return list(queued - active)

    @staticmethod
    async def _dispatch_stranded_queued(thread_ids: list[str]) -> None:
        # Deferred import: executor -> run_executor -> run_status, none of which
        # may import lease_reaper at module load.
        from aegra_api.services.executor import executor

        for thread_id in thread_ids:
            try:
                await executor.dispatch_next_for_thread(thread_id)
            except Exception:  # per-thread isolation: one bad row must not stall the rest of the cycle
                logger.exception("Failed to dispatch stranded queued run", thread_id=thread_id)

    @staticmethod
    async def _find_recoverable() -> tuple[list[str], list[str]]:
        """Find two categories: crashed workers (expired lease) and stuck pending runs.

        Returns (crashed_run_ids, stuck_pending_run_ids) separately so retry
        budget is only charged to crashed runs, not stuck pending ones.
        """
        now = datetime.now(UTC)
        maker = _get_session_maker()
        async with maker() as session:
            crashed_result = await session.execute(
                select(RunORM.run_id).where(
                    RunORM.status == "running",
                    RunORM.lease_expires_at.isnot(None),
                    RunORM.lease_expires_at < now,
                )
            )
            crashed = [row[0] for row in crashed_result.fetchall()]

            stuck_result = await session.execute(
                select(RunORM.run_id).where(
                    RunORM.status == "pending",
                    RunORM.claimed_by.is_(None),
                    # updated_at, not created_at: a queued run promoted to pending keeps its
                    # original created_at, so created_at would flag fresh promotions as stuck.
                    RunORM.updated_at < now - timedelta(seconds=settings.worker.STUCK_PENDING_THRESHOLD_SECONDS),
                )
            )
            stuck_pending = [row[0] for row in stuck_result.fetchall()]

        return crashed, stuck_pending

    @staticmethod
    async def _recover_crashed_runs(run_ids: list[str]) -> tuple[list[str], list[str]]:
        """Atomically classify expired runs and apply their next state."""
        now = datetime.now(UTC)
        max_retries = settings.worker.BG_JOB_MAX_RETRIES
        retryable: list[str] = []
        exhausted: list[str] = []
        exhausted_threads_by_user: dict[str, set[str]] = {}

        maker = _get_session_maker()
        async with maker() as session:
            locked_result = await session.execute(
                select(
                    RunORM.run_id,
                    RunORM.thread_id,
                    RunORM.user_id,
                    RunORM.execution_params,
                )
                .where(
                    RunORM.run_id.in_(run_ids),
                    RunORM.status == "running",
                    RunORM.lease_expires_at < now,
                )
                .with_for_update(skip_locked=True)
            )

            for run_id, thread_id, user_id, execution_params in locked_result.fetchall():
                params = dict(execution_params or {})
                retry_count = params.get("_retry_count", 0) + 1
                params["_retry_count"] = retry_count

                values: dict[str, object] = {
                    "execution_params": params,
                    "claimed_by": None,
                    "lease_expires_at": None,
                    "updated_at": now,
                }
                is_exhausted = retry_count > max_retries
                if is_exhausted:
                    values.update(
                        status="error",
                        error_message="Max retries exceeded after repeated worker failures",
                    )
                else:
                    values["status"] = "pending"

                update_result = await session.execute(
                    update(RunORM)
                    .where(
                        RunORM.run_id == run_id,
                        RunORM.user_id == user_id,
                        RunORM.status == "running",
                        RunORM.lease_expires_at < now,
                    )
                    .values(**values)
                    .returning(RunORM.run_id)
                )
                if update_result.scalar_one_or_none() is None:
                    continue

                if is_exhausted:
                    exhausted.append(run_id)
                    exhausted_threads_by_user.setdefault(user_id, set()).add(thread_id)
                    logger.error(
                        "Run exceeded max retries, marking as permanently failed",
                        run_id=run_id,
                        retries=retry_count,
                        max_retries=max_retries,
                    )
                else:
                    retryable.append(run_id)
                    logger.info(
                        "Incrementing retry count",
                        run_id=run_id,
                        retry_count=retry_count,
                        max_retries=max_retries,
                    )

            for user_id, thread_ids in exhausted_threads_by_user.items():
                await set_thread_status_if_no_active_runs(
                    session,
                    thread_ids,
                    "error",
                    user_id=user_id,
                )
            await session.commit()

        return retryable, exhausted

    @staticmethod
    async def _reenqueue(run_ids: list[str]) -> list[str]:
        """Push run_ids to the worker queue. Returns only the IDs confirmed pushed."""
        queue_key = settings.worker.WORKER_QUEUE_KEY
        pushed: list[str] = []
        try:
            client = redis_manager.get_client()
            for run_id in run_ids:
                await client.rpush(queue_key, run_id)  # type: ignore[arg-type]
                pushed.append(run_id)
                logger.info("Re-enqueued recovered run", run_id=run_id)
        except RedisError:
            logger.warning(
                "Redis unavailable during re-enqueue; workers will pick up via Postgres poll",
                run_ids=run_ids[len(pushed) :],
            )
        return pushed


lease_reaper = LeaseReaper()

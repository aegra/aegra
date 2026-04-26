"""Background task that recovers runs with expired worker leases.

Periodically scans the runs table for rows where
``status='running' AND lease_expires_at < now()``, resets them to
``pending`` (clearing the lease), and re-enqueues their run_ids to the
Redis job queue so another worker can pick them up.
"""

import asyncio
import contextlib
import time
from datetime import UTC, datetime, timedelta

import structlog
from redis import RedisError
from sqlalchemy import select, update

from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import _get_session_maker
from aegra_api.core.redis_manager import redis_manager
from aegra_api.observability.metrics import get_reaper_metrics, get_worker_metrics
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
        """Find crashed workers and stuck pending runs, recover them."""
        metrics = get_reaper_metrics()
        start = time.monotonic()

        # Queue depth: every cycle, including no-op
        if metrics is not None:
            depth = await self._get_queue_depth()
            metrics.queue_depth.set(depth)

        crashed, stuck_pending = await self._find_recoverable()

        if not crashed and not stuck_pending:
            if metrics is not None:
                metrics.cycle_seconds.observe(time.monotonic() - start)
            return

        # Crashed workers: reset first (atomic claim), then check retries
        if crashed:
            logger.warning("Reaping crashed worker runs", count=len(crashed), run_ids=crashed)
            actually_reset = await self._reset_to_pending(crashed)
            if actually_reset:
                if metrics is not None:
                    metrics.crashed_recovered.inc(len(actually_reset))
                retryable, exhausted = await self._check_retry_limits(actually_reset)
                if exhausted:
                    await self._mark_permanently_failed(exhausted)
                    if metrics is not None:
                        metrics.permanently_failed.inc(len(exhausted))
                if retryable:
                    await self._reenqueue(retryable)

        # Stuck pending: just re-enqueue (never executed, no retry budget)
        if stuck_pending:
            logger.warning("Re-enqueueing stuck pending runs", count=len(stuck_pending), run_ids=stuck_pending)
            await self._reenqueue(stuck_pending)
            if metrics is not None:
                metrics.stuck_reenqueued.inc(len(stuck_pending))

        if metrics is not None:
            metrics.cycle_seconds.observe(time.monotonic() - start)

        logger.info(
            "Lease recovery complete",
            crashed_recovered=len(crashed),
            stuck_reenqueued=len(stuck_pending),
        )

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
                    RunORM.created_at < now - timedelta(seconds=settings.worker.STUCK_PENDING_THRESHOLD_SECONDS),
                )
            )
            stuck_pending = [row[0] for row in stuck_result.fetchall()]

        return crashed, stuck_pending

    @staticmethod
    async def _reset_to_pending(run_ids: list[str]) -> list[str]:
        """Reset crashed runs to pending. Re-checks lease expiry atomically."""
        maker = _get_session_maker()
        async with maker() as session:
            result = await session.execute(
                update(RunORM)
                .where(
                    RunORM.run_id.in_(run_ids),
                    RunORM.status == "running",
                    RunORM.lease_expires_at < datetime.now(UTC),
                )
                .values(status="pending", claimed_by=None, lease_expires_at=None)
                .returning(RunORM.run_id)
            )
            reset_ids = [row[0] for row in result.fetchall()]
            await session.commit()
            return reset_ids

    @staticmethod
    async def _get_queue_depth() -> int:
        """Get the number of run_ids in the Redis job queue."""
        try:
            client = redis_manager.get_client()
            return await client.llen(settings.worker.WORKER_QUEUE_KEY)  # type: ignore[misc]
        except RedisError:
            return 0

    @staticmethod
    async def _reenqueue(run_ids: list[str]) -> None:
        """Re-enqueue run_ids to Redis, updating ``_enqueued_at`` for queue wait measurement."""
        # Update _enqueued_at via read-modify-write (consistent with _check_retry_limits pattern)
        maker = _get_session_maker()
        async with maker() as session:
            now = time.time()
            for run_id in run_ids:
                run_orm = await session.scalar(select(RunORM).where(RunORM.run_id == run_id).with_for_update())
                if run_orm is not None and run_orm.execution_params is not None:
                    params = run_orm.execution_params
                    params["_enqueued_at"] = now
                    await session.execute(update(RunORM).where(RunORM.run_id == run_id).values(execution_params=params))
            await session.commit()

        queue_key = settings.worker.WORKER_QUEUE_KEY
        try:
            client = redis_manager.get_client()
            for run_id in run_ids:
                await client.rpush(queue_key, run_id)  # type: ignore[arg-type]
                logger.info("Re-enqueued recovered run", run_id=run_id)
        except RedisError:
            logger.warning(
                "Redis unavailable during re-enqueue; workers will pick up via Postgres poll",
                run_ids=run_ids,
            )

    @staticmethod
    async def _check_retry_limits(run_ids: list[str]) -> tuple[list[str], list[str]]:
        """Split runs into retryable vs exhausted based on retry count.

        Increments ``_retry_count`` and resets ``_enqueued_at`` in
        execution_params for each retryable run.
        Returns (retryable_ids, exhausted_ids).
        """
        max_retries = settings.worker.BG_JOB_MAX_RETRIES
        retryable: list[str] = []
        exhausted: list[str] = []

        maker = _get_session_maker()
        async with maker() as session:
            for run_id in run_ids:
                run_orm = await session.scalar(select(RunORM).where(RunORM.run_id == run_id).with_for_update())
                if run_orm is None:
                    continue

                params = run_orm.execution_params or {}
                retry_count = params.get("_retry_count", 0) + 1

                if retry_count > max_retries:
                    exhausted.append(run_id)
                    logger.error(
                        "Run exceeded max retries, marking as permanently failed",
                        run_id=run_id,
                        retries=retry_count,
                        max_retries=max_retries,
                    )
                else:
                    params["_retry_count"] = retry_count
                    params["_enqueued_at"] = time.time()
                    await session.execute(update(RunORM).where(RunORM.run_id == run_id).values(execution_params=params))
                    retryable.append(run_id)

                    metrics = get_worker_metrics()
                    if metrics is not None:
                        graph_id = params.get("graph_id", "unknown")
                        metrics.run_retries.labels(graph_id=graph_id, retry_number=str(retry_count)).inc()

                    logger.info(
                        "Incrementing retry count",
                        run_id=run_id,
                        retry_count=retry_count,
                        max_retries=max_retries,
                    )

            await session.commit()

        return retryable, exhausted

    @staticmethod
    async def _mark_permanently_failed(run_ids: list[str]) -> None:
        """Mark runs as error with max retries exceeded message."""
        maker = _get_session_maker()
        async with maker() as session:
            await session.execute(
                update(RunORM)
                .where(RunORM.run_id.in_(run_ids))
                .values(
                    status="error",
                    error_message="Max retries exceeded after repeated worker failures",
                    claimed_by=None,
                    lease_expires_at=None,
                )
            )
            await session.commit()


lease_reaper = LeaseReaper()

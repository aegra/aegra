"""Background task that recovers runs with expired worker leases.

Periodically scans the runs table for rows where
``status='running' AND lease_expires_at < now()``, resets them to
``pending`` (clearing the lease), and re-enqueues their run_ids to the
Redis job queue so another worker can pick them up.
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
        crashed, stuck_pending = await self._find_recoverable()

        if not crashed and not stuck_pending:
            return

        # Crashed workers: reset first (atomic claim), then check retries
        if crashed:
            logger.warning("Reaping crashed worker runs", count=len(crashed), run_ids=crashed)
            actually_reset = await self._reset_to_pending(crashed)
            if actually_reset:
                retryable, exhausted = await self._check_retry_limits(actually_reset)
                if exhausted:
                    await self._mark_permanently_failed(exhausted)
                if retryable:
                    await self._reenqueue(retryable)

        # Stuck pending: just re-enqueue (never executed, no retry budget)
        if stuck_pending:
            logger.warning("Re-enqueueing stuck pending runs", count=len(stuck_pending), run_ids=stuck_pending)
            await self._reenqueue(stuck_pending)

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
        assistant_id = settings.worker.ASSISTANT_ID
        now = datetime.now(UTC)
        maker = _get_session_maker()
        async with maker() as session:
            crashed_result = await session.execute(
                select(RunORM.run_id).where(
                    RunORM.status == "running",
                    RunORM.lease_expires_at.isnot(None),
                    RunORM.lease_expires_at < now,
                    or_(
                        assistant_id is None,
                        RunORM.assistant_id == assistant_id
                    )
                )
            )
            crashed = [row[0] for row in crashed_result.fetchall()]

            stuck_result = await session.execute(
                select(RunORM.run_id).where(
                    RunORM.status == "pending",
                    RunORM.claimed_by.is_(None),
                    RunORM.created_at < now - timedelta(seconds=settings.worker.STUCK_PENDING_THRESHOLD_SECONDS),
                    or_(
                        assistant_id is None,
                        RunORM.assistant_id == assistant_id
                    )
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
    async def _reenqueue(run_ids: list[str]) -> None:
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

        Increments _retry_count in execution_params for each run.
        Returns (retryable_ids, exhausted_ids).
        """
        max_retries = settings.worker.BG_JOB_MAX_RETRIES
        retryable: list[str] = []
        exhausted: list[str] = []

        maker = _get_session_maker()
        async with maker() as session:
            for run_id in run_ids:
                run_orm = await session.scalar(select(RunORM).where(RunORM.run_id == run_id))
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
                    await session.execute(update(RunORM).where(RunORM.run_id == run_id).values(execution_params=params))
                    retryable.append(run_id)
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

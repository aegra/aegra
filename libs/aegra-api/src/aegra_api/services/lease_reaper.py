"""Background task that recovers runs with expired worker leases.

Periodically scans the runs table for rows where
``status='running' AND lease_expires_at < now()``. It atomically either
returns them to ``pending`` or marks their retry budget exhausted, then
re-enqueues only retryable run IDs.

Runs that never took a lease cannot match that predicate; a startup sweep
reconciles those separately.
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
from aegra_api.services.run_status import set_thread_status_if_no_active_runs
from aegra_api.settings import settings

logger = structlog.getLogger(__name__)

# Bound the startup sweep so a large backlog cannot open one huge transaction.
_SWEEP_BATCH_SIZE = 500
_SWEEP_MAX_BATCHES = 100
_MAX_LOGGED_RUN_IDS = 20


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
    async def sweep_leaseless_orphans() -> int:
        """Fail stale ``running`` rows that never took a lease. Returns the count swept.

        Assumes every replica agrees on ``REDIS_BROKER_ENABLED``; a replica still
        running the local executor would have its active runs swept.
        """
        if not settings.worker.ORPHAN_SWEEP_ENABLED:
            logger.info("Orphan sweep disabled, skipping")
            return 0

        total = 0
        sample: list[str] = []

        for _ in range(_SWEEP_MAX_BATCHES):
            selected, swept = await LeaseReaper._sweep_one_batch()
            if swept:
                total += len(swept)
                sample.extend(swept[: _MAX_LOGGED_RUN_IDS - len(sample)])
                REAPER_RECOVERED_RUNS.labels(outcome="local_orphan").inc(len(swept))
            if selected == 0:
                break
        else:
            logger.warning("Orphan sweep hit its batch limit; the rest waits for the next startup")

        if total:
            logger.warning("Swept runs orphaned without a lease", count=total, run_ids=sample)

        return total

    @staticmethod
    async def _sweep_one_batch() -> tuple[int, list[str]]:
        """Sweep one batch in a single transaction. Returns (rows seen, run IDs swept)."""
        now = datetime.now(UTC)
        cutoff = now - timedelta(seconds=settings.worker.ORPHAN_SWEEP_MIN_AGE_SECONDS)
        swept: list[str] = []
        threads_by_user: dict[str, set[str]] = {}

        maker = _get_session_maker()
        async with maker() as session:
            locked_result = await session.execute(
                select(RunORM.run_id, RunORM.thread_id, RunORM.user_id)
                .where(
                    RunORM.status == "running",
                    RunORM.claimed_by.is_(None),
                    RunORM.lease_expires_at.is_(None),
                    RunORM.updated_at < cutoff,
                )
                .limit(_SWEEP_BATCH_SIZE)
                .with_for_update(skip_locked=True)
            )
            rows = locked_result.fetchall()

            for run_id, thread_id, user_id in rows:
                update_result = await session.execute(
                    update(RunORM)
                    .where(
                        RunORM.run_id == run_id,
                        RunORM.user_id == user_id,
                        RunORM.status == "running",
                        RunORM.claimed_by.is_(None),
                        RunORM.lease_expires_at.is_(None),
                    )
                    .values(
                        status="error",
                        error_message="Run orphaned without a lease and could not be recovered",
                        updated_at=now,
                    )
                    .returning(RunORM.run_id)
                )
                if update_result.scalar_one_or_none() is None:
                    continue

                swept.append(run_id)
                threads_by_user.setdefault(user_id, set()).add(thread_id)

            for user_id, thread_ids in threads_by_user.items():
                await set_thread_status_if_no_active_runs(
                    session,
                    thread_ids,
                    "error",
                    user_id=user_id,
                )
            await session.commit()

        return len(rows), swept

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

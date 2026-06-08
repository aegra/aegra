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
from typing import Any

import structlog
from redis import RedisError
from sqlalchemy import select, update
from sqlalchemy.exc import SQLAlchemyError

from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import _get_session_maker
from aegra_api.core.redis_manager import redis_manager
from aegra_api.observability.metrics import extract_graph_id, get_reaper_metrics, get_worker_metrics
from aegra_api.settings import settings

logger = structlog.getLogger(__name__)

# Truncate run_id lists in log payloads so storms (e.g. 100 crashed runs)
# don't flood the log shipper with kilobyte messages. Failure counts are
# always logged in full alongside the truncated sample.
_LOG_RUN_IDS_SAMPLE = 10


def _truncate_ids(run_ids: list[str]) -> list[str] | str:
    """Return run_ids verbatim if small, otherwise sample + count for logs."""
    if len(run_ids) <= _LOG_RUN_IDS_SAMPLE:
        return run_ids
    sample = run_ids[:_LOG_RUN_IDS_SAMPLE]
    return f"{sample}... (+{len(run_ids) - _LOG_RUN_IDS_SAMPLE} more)"


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

        # Queue depth: every cycle, including no-op.
        # Skip the gauge update when Redis is unreachable — keeping the last
        # value avoids dashboards showing "queue empty" when Redis is down.
        if metrics is not None:
            depth = await self._get_queue_depth()
            if depth is not None:
                metrics.queue_depth.set(depth)

        crashed, stuck_pending = await self._find_recoverable()

        if not crashed and not stuck_pending:
            if metrics is not None:
                metrics.cycle_seconds.observe(time.monotonic() - start)
            return

        # Crashed workers: reset first (atomic claim), then check retries
        if crashed:
            logger.warning("Reaping crashed worker runs", count=len(crashed), run_ids=_truncate_ids(crashed))
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
                    # bump_dispatched=True: a retry is a fresh dispatch
                    # event for the queue. The original submit + every
                    # retry both increment ``runs_dispatched``.
                    await self._reenqueue(retryable, bump_dispatched=True)

        # Stuck pending: just re-enqueue (never executed, no retry budget)
        if stuck_pending:
            logger.warning(
                "Re-enqueueing stuck pending runs",
                count=len(stuck_pending),
                run_ids=_truncate_ids(stuck_pending),
            )
            # bump_dispatched=False: stuck-pending runs were already
            # counted by their original submit. Incrementing here would
            # double-count the same run and break the identity invariant
            # ``dispatched == completed + permanently_failed + in_flight + discarded``.
            await self._reenqueue(stuck_pending, bump_dispatched=False)
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
    async def _get_queue_depth() -> int | None:
        """Get the number of run_ids in the Redis job queue.

        Returns ``None`` when Redis is unreachable so callers can choose
        to keep the last gauge value rather than reporting a misleading 0.
        """
        try:
            client = redis_manager.get_client()
            return await client.llen(settings.worker.WORKER_QUEUE_KEY)  # type: ignore[misc]
        except RedisError:
            return None

    @staticmethod
    async def _update_enqueued_at(run_id: str, now: float) -> dict[str, Any] | None:
        """Refresh ``_enqueued_at`` on the row's ``execution_params`` JSON.

        Returns the updated params dict on success, ``None`` when the row
        is missing, has no params (data corruption), or the UPDATE fails.
        Each call gets its own session so a failure on one row does not
        poison the others.
        """
        maker = _get_session_maker()
        try:
            async with maker() as session:
                run_orm = await session.scalar(select(RunORM).where(RunORM.run_id == run_id).with_for_update())
                if run_orm is None or run_orm.execution_params is None:
                    logger.warning(
                        "Run missing or has no execution_params, skipping rpush",
                        run_id=run_id,
                    )
                    return None
                params: dict[str, Any] = run_orm.execution_params
                params["_enqueued_at"] = now
                await session.execute(update(RunORM).where(RunORM.run_id == run_id).values(execution_params=params))
                await session.commit()
                return params
        except SQLAlchemyError as exc:
            logger.warning(
                "Failed to update _enqueued_at, skipping rpush",
                run_id=run_id,
                error=str(exc),
                exc_info=True,
            )
            return None

    @staticmethod
    async def _reenqueue(run_ids: list[str], *, bump_dispatched: bool = True) -> None:
        """Re-enqueue run_ids to Redis, updating ``_enqueued_at`` for queue wait measurement.

        Only run_ids whose ``_enqueued_at`` update committed are rpush'ed
        — otherwise the queue-wait metric would be seeded with a stale
        timestamp for a run that silently failed to update.

        ``bump_dispatched`` controls whether ``runs_dispatched`` is
        incremented per successful rpush. ``True`` for crashed-retry
        paths (each retry is a fresh dispatch event), ``False`` for
        stuck-pending re-enqueue (the original submit already counted).
        """
        updated: list[tuple[str, dict[str, Any]]] = []
        now = time.time()
        for run_id in run_ids:
            params = await LeaseReaper._update_enqueued_at(run_id, now)
            if params is not None:
                updated.append((run_id, params))

        if not updated:
            return

        queue_key = settings.worker.WORKER_QUEUE_KEY
        worker_metrics = get_worker_metrics()
        try:
            client = redis_manager.get_client()
            for run_id, params in updated:
                await client.rpush(queue_key, run_id)  # type: ignore[arg-type]
                # Increment AFTER a successful rpush so a mid-loop Redis
                # failure does not over-count: the failed rpush raises,
                # we exit to the except block, and runs not yet pushed
                # remain uncounted (they'll be re-tried on the next cycle).
                if bump_dispatched and worker_metrics is not None:
                    worker_metrics.runs_dispatched.labels(graph_id=extract_graph_id(params)).inc()
                logger.info("Re-enqueued recovered run", run_id=run_id)
        except RedisError:
            logger.warning(
                "Redis unavailable during re-enqueue; workers will pick up via Postgres poll",
                run_ids=_truncate_ids([run_id for run_id, _ in updated]),
            )

    @staticmethod
    async def _bump_retry_count(run_id: str, max_retries: int) -> tuple[int, dict[str, Any]] | None:
        """Increment ``_retry_count`` for one run.

        Returns ``(new_count, params)`` on success, ``None`` when the row
        is missing, the row has no ``execution_params`` (data corruption
        — synthesizing one would revive a broken run with no graph/input
        metadata), or the UPDATE fails. Caller decides whether the new
        count exceeds the retry limit.
        """
        maker = _get_session_maker()
        try:
            async with maker() as session:
                run_orm = await session.scalar(select(RunORM).where(RunORM.run_id == run_id).with_for_update())
                if run_orm is None or run_orm.execution_params is None:
                    logger.warning(
                        "Run missing or has no execution_params, skipping retry bump",
                        run_id=run_id,
                    )
                    return None
                params: dict[str, Any] = run_orm.execution_params
                retry_count = params.get("_retry_count", 0) + 1
                if retry_count > max_retries:
                    return retry_count, params
                params["_retry_count"] = retry_count
                # ``_reenqueue`` overwrites ``_enqueued_at`` via
                # ``_update_enqueued_at`` next — setting it here is dead.
                await session.execute(update(RunORM).where(RunORM.run_id == run_id).values(execution_params=params))
                await session.commit()
                return retry_count, params
        except SQLAlchemyError as exc:
            logger.warning(
                "Failed to update retry count, skipping run for this cycle",
                run_id=run_id,
                error=str(exc),
                exc_info=True,
            )
            return None

    @staticmethod
    async def _check_retry_limits(run_ids: list[str]) -> tuple[list[str], list[str]]:
        """Split runs into retryable vs exhausted based on retry count.

        Increments ``_retry_count`` and resets ``_enqueued_at`` in
        execution_params for each retryable run.
        Returns (retryable_ids, exhausted_ids).

        Each run_id gets its own session for the same reason as
        ``_reenqueue``: a failure on one row (deadlock, FK violation, etc.)
        must not poison the others. Failures are logged and the run is
        skipped — it will be re-evaluated on the next reaper cycle.
        """
        max_retries = settings.worker.BG_JOB_MAX_RETRIES
        retryable: list[str] = []
        exhausted: list[str] = []

        for run_id in run_ids:
            outcome = await LeaseReaper._bump_retry_count(run_id, max_retries)
            if outcome is None:
                continue
            retry_count, params = outcome
            if retry_count > max_retries:
                exhausted.append(run_id)
                logger.error(
                    "Run exceeded max retries, marking as permanently failed",
                    run_id=run_id,
                    retries=retry_count,
                    max_retries=max_retries,
                )
                continue
            retryable.append(run_id)
            metrics = get_worker_metrics()
            if metrics is not None:
                graph_id = extract_graph_id(params)
                metrics.run_retries.labels(graph_id=graph_id, retry_number=str(retry_count)).inc()
            logger.info(
                "Incrementing retry count",
                run_id=run_id,
                retry_count=retry_count,
                max_retries=max_retries,
            )

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

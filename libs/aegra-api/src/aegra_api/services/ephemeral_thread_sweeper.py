"""Reclaim stateless-run threads left behind by disconnected clients.

Stateless endpoints normally delete their generated thread when the response
finishes. A disconnect can race with terminal-event delivery, especially when
Redis workers and API instances are separate. This sweeper handles only rows
explicitly marked as ephemeral and old enough to be safely reclaimed.
"""

import asyncio
from datetime import UTC, datetime, timedelta

import structlog
from psycopg import Error as PsycopgError
from sqlalchemy import Select, select
from sqlalchemy.exc import SQLAlchemyError

from aegra_api.core.database import db_manager
from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import Thread as ThreadORM
from aegra_api.core.orm import _get_session_maker
from aegra_api.observability.metrics import EPHEMERAL_THREAD_SWEPT
from aegra_api.settings import settings

logger = structlog.getLogger(__name__)

_SWEEP_ERRORS: tuple[type[BaseException], ...] = (PsycopgError, SQLAlchemyError, OSError)


def _orphaned_threads_stmt(*, cutoff: datetime, limit: int) -> Select[tuple[ThreadORM]]:
    """Claim old ephemeral threads that have no active run.

    Locking the thread row prevents a concurrent run insert from racing the
    checkpoint/thread deletion. The run foreign key takes a key-share lock and
    therefore waits for this claim transaction to finish.
    """
    conditions = [
        ThreadORM.is_ephemeral.is_(True),
        ThreadORM.updated_at <= cutoff,
        ~select(RunORM.run_id)
        .where(
            RunORM.thread_id == ThreadORM.thread_id,
            RunORM.status.in_(("pending", "running")),
        )
        .exists(),
    ]
    return (
        select(ThreadORM)
        .where(*conditions)
        .order_by(ThreadORM.updated_at.asc())
        .limit(limit)
        .with_for_update(skip_locked=True, of=ThreadORM)
    )


async def sweep_orphaned_threads() -> tuple[int, int, int]:
    """Delete one bounded batch and return ``(claimed, deleted, errors)``."""
    cutoff = datetime.now(UTC) - timedelta(seconds=settings.ephemeral_thread.EPHEMERAL_THREAD_RETENTION_SECONDS)
    maker = _get_session_maker()
    async with maker() as session:
        deleted = 0
        errors = 0
        deleted_threads: list[ThreadORM] = []
        rows = (
            await session.scalars(
                _orphaned_threads_stmt(
                    cutoff=cutoff,
                    limit=settings.ephemeral_thread.EPHEMERAL_THREAD_SWEEP_LIMIT,
                )
            )
        ).all()
        claimed = len(rows)
        for thread in rows:
            # Recheck after locking because the claim query's snapshot may predate
            # a run committed while this transaction waited for the thread lock.
            active_run = await session.scalar(
                select(RunORM.run_id)
                .where(
                    RunORM.thread_id == thread.thread_id,
                    RunORM.status.in_(("pending", "running")),
                )
                .limit(1)
            )
            if active_run is not None:
                continue
            try:
                # Checkpoints are in the LangGraph pool, so delete them first.
                # A failure leaves the thread row available for a later retry.
                await db_manager.get_checkpointer().adelete_thread(thread.thread_id)
                await session.delete(thread)
                deleted_threads.append(thread)
            except _SWEEP_ERRORS:
                errors += 1
                EPHEMERAL_THREAD_SWEPT.labels(outcome="error").inc()
                logger.exception("Failed to reclaim orphaned ephemeral thread", thread_id=thread.thread_id)
        try:
            await session.commit()
        except _SWEEP_ERRORS:
            await session.rollback()
            errors += len(deleted_threads)
            for _ in deleted_threads:
                EPHEMERAL_THREAD_SWEPT.labels(outcome="error").inc()
            logger.exception("Failed to commit orphaned ephemeral thread sweep")
        else:
            deleted = len(deleted_threads)
            for _ in deleted_threads:
                EPHEMERAL_THREAD_SWEPT.labels(outcome="deleted").inc()
    return claimed, deleted, errors


class EphemeralThreadSweeper:
    """Periodically reclaims stale ephemeral threads."""

    def __init__(self) -> None:
        """Initialize the background task state."""
        self._task: asyncio.Task[None] | None = None
        self._running = False

    async def start(self) -> None:
        """Start periodic orphan cleanup."""
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info(
            "Ephemeral thread sweeper started",
            retention_seconds=settings.ephemeral_thread.EPHEMERAL_THREAD_RETENTION_SECONDS,
            interval_seconds=settings.ephemeral_thread.EPHEMERAL_THREAD_SWEEP_INTERVAL_SECONDS,
            sweep_limit=settings.ephemeral_thread.EPHEMERAL_THREAD_SWEEP_LIMIT,
        )

    async def stop(self) -> None:
        """Stop the background task."""
        self._running = False
        if self._task is not None:
            task = self._task
            self._task = None
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        logger.info("Ephemeral thread sweeper stopped")

    async def _loop(self) -> None:
        """Sweep immediately, then sleep between bounded batches."""
        interval = settings.ephemeral_thread.EPHEMERAL_THREAD_SWEEP_INTERVAL_SECONDS
        while self._running:
            try:
                claimed, deleted, errors = await sweep_orphaned_threads()
                if claimed:
                    logger.info(
                        "Ephemeral thread sweep completed",
                        claimed=claimed,
                        deleted=deleted,
                        errors=errors,
                    )
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in ephemeral thread sweep")
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break


ephemeral_thread_sweeper = EphemeralThreadSweeper()

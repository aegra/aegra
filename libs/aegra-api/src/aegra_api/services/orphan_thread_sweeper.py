"""Background task that deletes leaked ephemeral threads.

Stateless run endpoints (POST /runs, /runs/wait, /runs/stream) create an
ephemeral thread and normally delete it once the run finishes. In prod mode,
a slow-client / dead-proxy abort can race ahead of the broker's end-event
drain, and the fast-path delete is skipped to avoid cancelling a still-active
run (see stateless_runs.py::_run_finished). When that happens the thread is
never revisited, and its row (and checkpoints) leak in Postgres forever.

Unlike the thread_ttl sweeper, which is opt-in and a deliberate no-op when
the operator hasn't configured a TTL policy, this sweeper always runs: it
only ever touches threads the stateless endpoints marked ``is_ephemeral``,
never threads created via POST /threads, so there is no data-retention
policy for an operator to opt into.
"""

import asyncio
import contextlib
from datetime import UTC, datetime, timedelta

import structlog
from psycopg import Error as PsycopgError
from sqlalchemy import Select, delete, select

from aegra_api.core.database import db_manager
from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import Thread as ThreadORM
from aegra_api.core.orm import _get_session_maker
from aegra_api.observability.metrics import ORPHAN_THREADS_SWEPT
from aegra_api.settings import settings

logger = structlog.getLogger(__name__)


def _claim_stmt(*, cutoff: datetime, limit: int) -> Select[tuple[str]]:
    """Claim query for idle ephemeral threads, locking the thread row.

    ``skip_locked`` partitions work across instances; excluding threads with
    an active run avoids racing a run's own thread-status update, and closes
    the check-then-act race the same way thread_ttl's claim does: a run
    INSERT takes FOR KEY SHARE on the thread (FK), so it waits for this
    transaction instead of slipping in after the check.
    """
    active_runs_exist = (
        select(RunORM.run_id)
        .where(
            RunORM.thread_id == ThreadORM.thread_id,
            RunORM.status.in_(("pending", "running")),
        )
        .exists()
    )
    return (
        select(ThreadORM.thread_id)
        .where(
            ThreadORM.is_ephemeral.is_(True),
            ThreadORM.updated_at < cutoff,
            ~active_runs_exist,
        )
        .order_by(ThreadORM.updated_at.asc())
        .limit(limit)
        .with_for_update(skip_locked=True, of=ThreadORM)
    )


class OrphanThreadSweeper:
    """Periodically deletes ephemeral threads left behind by missed cleanup."""

    def __init__(self) -> None:
        """Initialize the sweeper state for the background polling loop."""
        self._task: asyncio.Task[None] | None = None
        self._running = False

    async def start(self) -> None:
        """Start the background sweep task."""
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info(
            "Orphan thread sweeper started",
            interval_seconds=settings.orphan_thread.ORPHAN_THREAD_SWEEP_INTERVAL_SECONDS,
            retention_minutes=settings.orphan_thread.ORPHAN_THREAD_RETENTION_MINUTES,
        )

    async def stop(self) -> None:
        """Stop the background sweep task and wait for cancellation to finish."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        logger.info("Orphan thread sweeper stopped")

    async def _loop(self) -> None:
        interval = settings.orphan_thread.ORPHAN_THREAD_SWEEP_INTERVAL_SECONDS
        while self._running:
            try:
                await self._sweep()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in orphan thread sweep")
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break

    async def _sweep(self) -> None:
        """Claim and delete one batch of idle ephemeral threads."""
        now = datetime.now(UTC)
        cutoff = now - timedelta(minutes=settings.orphan_thread.ORPHAN_THREAD_RETENTION_MINUTES)
        limit = settings.orphan_thread.ORPHAN_THREAD_SWEEP_BATCH_SIZE
        claim_stmt = _claim_stmt(cutoff=cutoff, limit=limit)

        maker = _get_session_maker()
        deleted = 0
        failed: list[str] = []
        async with maker() as session:
            thread_ids = (await session.execute(claim_stmt)).scalars().all()
            if not thread_ids:
                return

            for thread_id in thread_ids:
                try:
                    # Checkpoints live on the psycopg lg_pool, a separate
                    # connection from this SQLAlchemy session: a failure here
                    # cannot poison the session's transaction.
                    await db_manager.get_checkpointer().adelete_thread(thread_id)
                except (PsycopgError, OSError):
                    ORPHAN_THREADS_SWEPT.labels(outcome="error").inc()
                    logger.exception("Failed to delete orphaned thread checkpoints", thread_id=thread_id)
                    failed.append(thread_id)
                    continue
                await session.execute(delete(ThreadORM).where(ThreadORM.thread_id == thread_id))
                deleted += 1
                ORPHAN_THREADS_SWEPT.labels(outcome="deleted").inc()

            await session.commit()

        logger.info(
            "Orphan thread sweep completed",
            claimed=len(thread_ids),
            deleted=deleted,
            failed=len(failed),
        )


orphan_thread_sweeper = OrphanThreadSweeper()

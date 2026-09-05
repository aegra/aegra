"""Thread TTL: config resolution and expiry sweep (issue #288 phase 2).

Threads opt into a TTL via a ``thread_ttl`` row (server default on creation or
per-thread override). A background sweep claims expired rows with
``FOR UPDATE SKIP LOCKED`` and applies the row's strategy:

- ``delete``  — remove checkpoints then the thread row (cascades runs/crons).
- ``keep_latest`` — prune checkpoint history, keep the latest state, re-arm.

The sweep never cancels runs: threads with pending/running runs are skipped
and picked up on a later tick once their runs settle.
"""

import asyncio
import contextlib
import json
from collections.abc import Collection
from datetime import UTC, datetime, timedelta
from functools import cache
from typing import Literal

import structlog
from psycopg import Error as PsycopgError
from pydantic import BaseModel, Field
from sqlalchemy import ColumnElement, Select, delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from aegra_api.config import load_checkpointer_config
from aegra_api.core.database import db_manager
from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import Thread as ThreadORM
from aegra_api.core.orm import ThreadTTL as ThreadTTLORM
from aegra_api.core.orm import _get_session_maker
from aegra_api.models.threads import MAX_TTL_MINUTES
from aegra_api.observability.metrics import THREAD_TTL_SWEPT
from aegra_api.settings import settings

logger = structlog.getLogger(__name__)

# Rows claimed per transaction. Bounds lock-hold time and the window a user
# DELETE on a claimed thread blocks on the thread_ttl cascade.
_CLAIM_BATCH = 100

# --- keep_latest pruning SQL -------------------------------------------------
# Raw SQL over the psycopg lg_pool against checkpointer-owned tables. Pinned to
# the schema of langgraph-checkpoint-postgres 3.x (base.py MIGRATIONS); the
# schema-drift canary in tests/unit/test_services/test_thread_ttl.py fails
# loudly if a package bump changes the assumptions below.

# Cheap probe so idle keep_latest cycles skip the three DELETEs: history is
# prunable only when some namespace holds more than one checkpoint (PK-indexed).
_HAS_PRUNABLE_HISTORY_SQL = """
SELECT EXISTS (
    SELECT 1 FROM checkpoints
    WHERE thread_id = %(tid)s
    GROUP BY checkpoint_ns
    HAVING count(*) > 1
) AS has_prunable_history
"""

# Latest checkpoint per (thread_id, checkpoint_ns) = max(checkpoint_id):
# checkpoint_ids are monotonic and langgraph's own "latest" is
# ORDER BY checkpoint_id DESC.
_PRUNE_CHECKPOINTS_SQL = """
DELETE FROM checkpoints c
USING (
    SELECT checkpoint_ns, max(checkpoint_id) AS latest_id
    FROM checkpoints
    WHERE thread_id = %(tid)s
    GROUP BY checkpoint_ns
) keep
WHERE c.thread_id = %(tid)s
  AND c.checkpoint_ns = keep.checkpoint_ns
  AND c.checkpoint_id <> keep.latest_id
"""

# Writes of surviving checkpoints are pending writes needed on resume — keep
# them; drop writes whose checkpoint is gone.
_PRUNE_WRITES_SQL = """
DELETE FROM checkpoint_writes w
WHERE w.thread_id = %(tid)s
  AND NOT EXISTS (
      SELECT 1 FROM checkpoints c
      WHERE c.thread_id = w.thread_id
        AND c.checkpoint_ns = w.checkpoint_ns
        AND c.checkpoint_id = w.checkpoint_id
  )
"""

# A kept checkpoint may reference OLD blob versions for unchanged channels, so
# keeping max(version) would corrupt state. Keep exactly the (channel, version)
# pairs the surviving checkpoints reference — the same join langgraph's own
# SELECT_SQL performs to load channel_values.
_PRUNE_BLOBS_SQL = """
DELETE FROM checkpoint_blobs b
WHERE b.thread_id = %(tid)s
  AND NOT EXISTS (
      SELECT 1
      FROM checkpoints c,
           jsonb_each_text(c.checkpoint -> 'channel_versions') AS cv(channel, version)
      WHERE c.thread_id = b.thread_id
        AND c.checkpoint_ns = b.checkpoint_ns
        AND cv.channel = b.channel
        AND cv.version = b.version
  )
"""


class ThreadTTLConfig(BaseModel):
    """Validated TTL configuration merged from env and aegra.json."""

    strategy: Literal["delete", "keep_latest"] = "delete"
    default_ttl: float = Field(43200, gt=0, le=MAX_TTL_MINUTES)  # minutes; 30 days
    sweep_interval_minutes: float = Field(5, gt=0, le=MAX_TTL_MINUTES)
    # 10000 matches the documented LangGraph Platform default for this block.
    sweep_limit: int = Field(10000, gt=0)


@cache
def get_thread_ttl_config() -> ThreadTTLConfig | None:
    """Resolve TTL config: AEGRA_THREAD_TTL env var wins over aegra.json.

    The env var is either a bare number (default_ttl in minutes) or a JSON
    object; when set it replaces the checkpointer.ttl block entirely (same
    whole-source precedence as DATABASE_URL over POSTGRES_*). Returns None
    when neither source is configured — the feature is off. Invalid config
    raises so a misconfigured retention policy fails at startup instead of
    silently deleting (or retaining) the wrong data.
    """
    # LANGGRAPH_THREAD_TTL is a migration alias; the AEGRA_ var wins when both set.
    raw = settings.thread_ttl.AEGRA_THREAD_TTL
    if raw is None or not raw.strip():
        raw = settings.thread_ttl.LANGGRAPH_THREAD_TTL
    if raw is not None and raw.strip():
        try:
            data: dict[str, object] = {"default_ttl": float(raw)}
        except ValueError:
            data = json.loads(raw)
        return ThreadTTLConfig.model_validate(data)

    checkpointer_config = load_checkpointer_config()
    ttl_config = checkpointer_config.get("ttl") if checkpointer_config else None
    if ttl_config is not None:
        return ThreadTTLConfig.model_validate(ttl_config)

    return None


async def _prune_checkpoint_history(thread_id: str) -> None:
    """Delete all checkpoint history for a thread except the latest state.

    One explicit transaction (the lg_pool is autocommit): a concurrent reader
    sees pre- or post-prune state, never a torn middle. Idempotent — a re-run
    deletes nothing further.
    """
    pool = db_manager.lg_pool
    if pool is None:
        raise RuntimeError("Database not initialized")
    async with pool.connection() as conn:
        cursor = await conn.execute(_HAS_PRUNABLE_HISTORY_SQL, {"tid": thread_id})
        row = await cursor.fetchone()
        # The lg_pool is configured with row_factory=dict_row — access by name.
        if row is None or not row["has_prunable_history"]:
            return
        async with conn.transaction():
            await conn.execute(_PRUNE_CHECKPOINTS_SQL, {"tid": thread_id})
            await conn.execute(_PRUNE_WRITES_SQL, {"tid": thread_id})
            await conn.execute(_PRUNE_BLOBS_SQL, {"tid": thread_id})


async def _apply_strategy(
    session: AsyncSession, *, thread_id: str, strategy: str, ttl_minutes: float, now: datetime
) -> str:
    """Apply one expired row's strategy inside the claim transaction; return the outcome label."""
    if strategy == "keep_latest":
        await _prune_checkpoint_history(thread_id)
        # Re-arm: keep_latest is periodic compaction, not a one-shot.
        await session.execute(
            update(ThreadTTLORM)
            .where(ThreadTTLORM.thread_id == thread_id)
            .values(expires_at=now + timedelta(minutes=ttl_minutes))
        )
        return "pruned"

    # Checkpoints first — ordering rationale from the delete_thread route: a
    # failure here leaves the thread row intact and retryable, never orphans.
    await db_manager.get_checkpointer().adelete_thread(thread_id)
    await session.execute(delete(ThreadORM).where(ThreadORM.thread_id == thread_id))
    return "deleted"


def _expired_claim_stmt(
    *,
    now: datetime,
    limit: int,
    user_id: str | None = None,
    auth_filter: ColumnElement[bool] | None = None,
    exclude_ids: Collection[str] = (),
) -> Select[tuple[str, str, float]]:
    """Claim query for expired thread_ttl rows, locking thread_ttl AND thread.

    ``skip_locked`` partitions work across instances (and between the sweeper
    and /threads/prune). Threads with active runs are skipped, not cancelled.
    Locking the thread row too closes the check-then-act race with run
    creation: a run INSERT takes FOR KEY SHARE on the thread (FK), so it waits
    for the claim transaction instead of slipping in after the active-run
    check and being cascade-deleted mid-flight.
    """
    active_runs_exist = (
        select(RunORM.run_id)
        .where(
            RunORM.thread_id == ThreadTTLORM.thread_id,
            RunORM.status.in_(("pending", "running")),
        )
        .exists()
    )
    stmt = (
        select(ThreadTTLORM.thread_id, ThreadTTLORM.strategy, ThreadTTLORM.ttl_minutes)
        .join(ThreadORM, ThreadORM.thread_id == ThreadTTLORM.thread_id)
        .where(
            ThreadTTLORM.expires_at <= now,
            ~active_runs_exist,
        )
    )
    if user_id is not None:
        stmt = stmt.where(ThreadORM.user_id == user_id)
        if auth_filter is not None:
            stmt = stmt.where(auth_filter)
    if exclude_ids:
        # Rows that already failed this pass sit first in expiry order; without
        # the exclusion they'd be re-claimed every batch and starve later rows.
        stmt = stmt.where(ThreadTTLORM.thread_id.not_in(exclude_ids))
    return (
        stmt.order_by(ThreadTTLORM.expires_at.asc())
        .limit(limit)
        .with_for_update(skip_locked=True, of=(ThreadTTLORM, ThreadORM))
    )


async def _process_expired_batch(
    session: AsyncSession, stmt: Select[tuple[str, str, float]], now: datetime
) -> tuple[int, int, int, list[str]]:
    """Claim and process one batch in a single transaction.

    Returns (claimed, deleted, pruned, failed_ids). The thread-row DELETE must
    run in the claim transaction: it cascades into the locked thread_ttl row,
    and a separate session would wait forever on a lock this transaction holds.
    """
    rows = (await session.execute(stmt)).all()
    if not rows:
        return 0, 0, 0, []

    deleted = 0
    pruned = 0
    failed_ids: list[str] = []
    for thread_id, strategy, ttl_minutes in rows:
        try:
            outcome = await _apply_strategy(
                session, thread_id=thread_id, strategy=strategy, ttl_minutes=ttl_minutes, now=now
            )
        except (PsycopgError, OSError):
            # Checkpointer-side failure on the other pool: this transaction is
            # untouched — skip the item, retry it on a later claim.
            THREAD_TTL_SWEPT.labels(outcome="error").inc()
            logger.exception("Thread TTL item failed", thread_id=thread_id, strategy=strategy)
            failed_ids.append(thread_id)
            continue
        if outcome == "deleted":
            deleted += 1
        else:
            pruned += 1
        THREAD_TTL_SWEPT.labels(outcome=outcome).inc()

    await session.commit()
    return len(rows), deleted, pruned, failed_ids


async def prune_expired_threads_for_user(
    session: AsyncSession,
    *,
    user_id: str,
    auth_filter: ColumnElement[bool] | None = None,
) -> tuple[int, int]:
    """Immediately apply TTL strategies to the caller's expired threads.

    Backs POST /threads/prune; works without server-side TTL config (rows may
    exist from per-thread opt-ins). Returns (deleted, pruned).
    """
    config = get_thread_ttl_config() or ThreadTTLConfig()
    total_deleted = 0
    total_pruned = 0
    claimed_total = 0
    failed: set[str] = set()
    while claimed_total < config.sweep_limit:
        now = datetime.now(UTC)
        limit = min(_CLAIM_BATCH, config.sweep_limit - claimed_total)
        stmt = _expired_claim_stmt(now=now, limit=limit, user_id=user_id, auth_filter=auth_filter, exclude_ids=failed)
        claimed, deleted, pruned, failed_ids = await _process_expired_batch(session, stmt, now)
        if claimed == 0:
            break
        claimed_total += claimed
        total_deleted += deleted
        total_pruned += pruned
        failed.update(failed_ids)
    return total_deleted, total_pruned


class ThreadTTLSweeper:
    """Periodically deletes or compacts expired threads."""

    def __init__(self) -> None:
        """Initialize the sweeper state for the background polling loop."""
        self._task: asyncio.Task[None] | None = None
        self._running = False

    async def start(self) -> None:
        """Start the background sweep task."""
        self._running = True
        self._task = asyncio.create_task(self._loop())
        config = get_thread_ttl_config()
        logger.info(
            "Thread TTL sweeper started",
            interval_minutes=config.sweep_interval_minutes if config else None,
            sweep_limit=config.sweep_limit if config else None,
        )

    async def stop(self) -> None:
        """Stop the background sweep task and wait for cancellation to finish."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        logger.info("Thread TTL sweeper stopped")

    async def _loop(self) -> None:
        """Tick then sleep so overdue threads are swept right after startup.

        The sleep sits in its own try so a persistent _tick failure (DB down)
        cannot spin the loop at CPU speed.
        """
        config = get_thread_ttl_config()
        interval_seconds = (config.sweep_interval_minutes if config else 5.0) * 60
        while self._running:
            try:
                await self._tick()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in thread TTL sweep tick")
            try:
                await asyncio.sleep(interval_seconds)
            except asyncio.CancelledError:
                break

    async def _tick(self) -> None:
        """Sweep up to sweep_limit expired threads in sub-batched transactions."""
        config = get_thread_ttl_config()
        if config is None:
            return

        maker = _get_session_maker()
        claimed_total = 0
        deleted_total = 0
        pruned_total = 0
        failed: set[str] = set()
        while claimed_total < config.sweep_limit:
            now = datetime.now(UTC)
            limit = min(_CLAIM_BATCH, config.sweep_limit - claimed_total)
            stmt = _expired_claim_stmt(now=now, limit=limit, exclude_ids=failed)
            async with maker() as session:
                claimed, deleted, pruned, failed_ids = await _process_expired_batch(session, stmt, now)
            if claimed == 0:
                break
            claimed_total += claimed
            deleted_total += deleted
            pruned_total += pruned
            failed.update(failed_ids)

        if claimed_total:
            logger.info(
                "Thread TTL sweep completed",
                claimed=claimed_total,
                deleted=deleted_total,
                pruned=pruned_total,
            )


thread_ttl_sweeper = ThreadTTLSweeper()

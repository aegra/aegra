"""Run and thread status management.

Provides the database-level status update operations used by both the
API layer (cancel, interrupt) and the execution layer (run_executor,
worker_executor). Extracted from api/runs.py to eliminate the circular
dependency where service code imported from the API module.
"""

from collections.abc import Collection
from datetime import UTC, datetime
from typing import Any, cast

import structlog
from sqlalchemy import CursorResult, exists, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import Thread as ThreadORM
from aegra_api.core.orm import _get_session_maker
from aegra_api.core.serializers import GeneralSerializer
from aegra_api.utils.status_compat import validate_run_status, validate_thread_status

logger = structlog.getLogger(__name__)
_serializer = GeneralSerializer()
# A run in one of these states occupies its thread: it has (or is about to get) a task.
ACTIVE_RUN_STATES = ("pending", "running")
# Internal double-texting park state: the run waits behind an active run and has no
# task or worker. Reported as ``pending`` over the API (Run.validate_status).
QUEUED_RUN_STATE = "queued"


async def start_run(run_id: str, *, user_id: str) -> bool:
    """Move an active run to running via CAS, so a run cancelled or pre-empted between dispatch
    and start cannot resurrect itself."""
    maker = _get_session_maker()
    async with maker() as session:
        result = await session.execute(
            update(RunORM)
            .where(
                RunORM.run_id == run_id,
                RunORM.user_id == user_id,
                RunORM.status.in_(ACTIVE_RUN_STATES),
            )
            .values(status="running", updated_at=datetime.now(UTC))
            .returning(RunORM.run_id)
        )
        started = result.scalar_one_or_none() is not None
        if started:
            await session.commit()
        else:
            await session.rollback()
        return started


async def get_run_status(run_id: str) -> str | None:
    """Return a run's current status, or None if the run no longer exists."""
    maker = _get_session_maker()
    async with maker() as session:
        return await session.scalar(select(RunORM.status).where(RunORM.run_id == run_id))


async def set_thread_status(session: AsyncSession, thread_id: str, status: str) -> None:
    """Update a thread's status column.

    Does NOT commit — the caller controls the transaction boundary.
    This allows thread status and run updates to share a single commit.
    """
    validated = validate_thread_status(status)
    result = cast(
        CursorResult,
        await session.execute(
            update(ThreadORM)
            .where(ThreadORM.thread_id == thread_id)
            .values(status=validated, updated_at=datetime.now(UTC))
        ),
    )
    if result.rowcount == 0:
        raise ValueError(f"Thread '{thread_id}' not found")


async def set_thread_status_if_no_active_runs(
    session: AsyncSession,
    thread_ids: Collection[str],
    status: str,
    *,
    user_id: str,
) -> None:
    """Update threads that no longer have a pending or running run.

    Does not commit so callers can keep the run and thread transitions in
    one transaction. Queued runs hold no task, so they do not count.
    """
    if not thread_ids:
        return

    validated = validate_thread_status(status)
    active_run_exists = exists(
        select(RunORM.run_id)
        .where(
            RunORM.thread_id == ThreadORM.thread_id,
            RunORM.user_id == user_id,
            RunORM.status.in_(ACTIVE_RUN_STATES),
        )
        .correlate(ThreadORM)
    )
    await session.execute(
        update(ThreadORM)
        .where(
            ThreadORM.thread_id.in_(thread_ids),
            ThreadORM.user_id == user_id,
            ~active_run_exists,
        )
        .values(status=validated, updated_at=datetime.now(UTC))
    )


async def _lock_thread_row(session: AsyncSession, thread_id: str) -> None:
    """Take the thread row lock before any run row, matching the admission gate; the reverse
    order deadlocks (40P01) against a concurrent interrupt/rollback create."""
    await session.execute(select(ThreadORM.thread_id).where(ThreadORM.thread_id == thread_id).with_for_update())


async def interrupt_unowned_run(
    session: AsyncSession,
    run_id: str,
    thread_id: str,
    *,
    user_id: str,
) -> bool:
    """Interrupt an active run only when it has no live database owner.

    The ownership predicate is checked in the UPDATE so a worker that renews
    or claims the run concurrently cannot be overwritten by the API process.
    If a live worker merely missed its lease, the caller asks it to stop through
    the broker, and guarded finalization rejects any late worker write.
    Does not dispatch queued runs: only the caller knows whether a task still executes this one.
    """
    now = datetime.now(UTC)
    # Lock-then-CAS in a SAVEPOINT: on a miss, rolling it back releases the thread lock at once
    # (the owner's finalize needs it) without expiring rows a bulk cancel is still iterating.
    savepoint = await session.begin_nested()
    await _lock_thread_row(session, thread_id)
    result = cast(
        CursorResult,
        await session.execute(
            update(RunORM)
            .where(
                RunORM.run_id == run_id,
                RunORM.thread_id == thread_id,
                RunORM.user_id == user_id,
                RunORM.status.in_(ACTIVE_RUN_STATES),
                or_(
                    RunORM.claimed_by.is_(None),
                    RunORM.lease_expires_at < now,
                ),
            )
            .values(
                status="interrupted",
                claimed_by=None,
                lease_expires_at=None,
                updated_at=now,
            )
            .returning(RunORM.run_id)
        ),
    )
    if result.scalar_one_or_none() is None:
        await savepoint.rollback()
        return False
    await savepoint.commit()

    await set_thread_status_if_no_active_runs(session, [thread_id], "idle", user_id=user_id)
    await session.commit()
    logger.info("Interrupted unowned run", run_id=run_id, thread_id=thread_id)
    return True


async def cancel_queued_run(
    session: AsyncSession,
    run_id: str,
    thread_id: str,
    *,
    user_id: str,
) -> bool:
    """Drop a parked (``queued``) run with a guarded status flip; the thread row is left alone.
    False means it was promoted or finished meanwhile and must be cancelled as an active run."""
    result = await session.execute(
        update(RunORM)
        .where(
            RunORM.run_id == run_id,
            RunORM.thread_id == thread_id,
            RunORM.user_id == user_id,
            RunORM.status == QUEUED_RUN_STATE,
        )
        .values(status="interrupted", updated_at=datetime.now(UTC))
        .returning(RunORM.run_id)
    )
    if result.scalar_one_or_none() is None:
        return False
    await session.commit()
    logger.info("Cancelled queued run", run_id=run_id, thread_id=thread_id)
    await dispatch_next_queued_run(thread_id)
    return True


async def cancel_queued_run_by_id(run_id: str, thread_id: str, *, user_id: str) -> bool:
    """``cancel_queued_run`` on a fresh session, for callers without a request session."""
    maker = _get_session_maker()
    async with maker() as session:
        return await cancel_queued_run(session, run_id, thread_id, user_id=user_id)


async def finalize_run(
    run_id: str,
    thread_id: str,
    *,
    user_id: str,
    status: str,
    thread_status: str,
    output: Any = None,
    error: str | None = None,
) -> bool:
    """Conditionally update run and thread status in one transaction.

    Returns false when another actor has already made the run terminal (a reconciled
    cancellation or a multitask pre-emption). Dispatches the next queued run either way.
    """
    validated_run = validate_run_status(status)
    validated_thread = validate_thread_status(thread_status)
    maker = _get_session_maker()

    run_values: dict[str, Any] = {
        "status": validated_run,
        "updated_at": datetime.now(UTC),
    }
    if output is not None:
        run_values["output"] = _safe_serialize(output, run_id)
    if error is not None:
        run_values["error_message"] = error

    async with maker() as session:
        await _lock_thread_row(session, thread_id)
        result = await session.execute(
            update(RunORM)
            .where(
                RunORM.run_id == run_id,
                RunORM.user_id == user_id,
                RunORM.status.in_(ACTIVE_RUN_STATES),
            )
            .values(**run_values)
            .returning(RunORM.run_id)
        )
        if result.scalar_one_or_none() is None:
            await session.rollback()
            logger.info("Skipped finalizing terminal run", run_id=run_id, status=validated_run)
            finalized = False
        else:
            await set_thread_status_if_no_active_runs(
                session,
                [thread_id],
                validated_thread,
                user_id=user_id,
            )
            await session.commit()
            finalized = True

    if finalized:
        logger.info("Finalized run", run_id=run_id, status=validated_run, thread_status=validated_thread)
    await dispatch_next_queued_run(thread_id)
    return finalized


async def dispatch_next_queued_run(thread_id: str) -> None:
    """Best-effort promotion of the oldest queued run; idempotent while the thread is occupied or
    paused. Failures are logged, not raised: recovery sweeps re-dispatch stranded queues."""
    # Deferred: executor -> run_executor -> run_status would cycle at module load.
    from aegra_api.services.executor import executor

    try:
        await executor.dispatch_next_for_thread(thread_id)
    except Exception:  # intentionally broad — ANY failure here must not clobber the committed status
        logger.exception("Failed to dispatch next queued run", thread_id=thread_id)


def _safe_serialize(output: Any, run_id: str) -> Any:
    """Serialize output with a fallback for non-JSON-compatible objects."""
    try:
        return _serializer.serialize(output)
    except Exception as exc:
        logger.warning("Output serialization failed", run_id=run_id, error=str(exc))
        return {
            "error": "Output serialization failed",
            "original_type": str(type(output)),
        }

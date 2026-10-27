"""Run and thread status management.

Provides the database-level status update operations used by both the
API layer (cancel, interrupt) and the execution layer (run_executor,
worker_executor). Extracted from api/runs.py to eliminate the circular
dependency where service code imported from the API module.
"""

from collections.abc import Collection, Iterable
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import structlog
from sqlalchemy import TIMESTAMP, ColumnElement, CursorResult, Select, exists, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from aegra_api.core.active_runs import active_runs
from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import Thread as ThreadORM
from aegra_api.core.orm import _get_session_maker
from aegra_api.core.serializers import GeneralSerializer
from aegra_api.settings import settings
from aegra_api.utils.status_compat import validate_run_status, validate_thread_status

logger = structlog.getLogger(__name__)
_serializer = GeneralSerializer()
# A run in one of these states occupies its thread: it has (or is about to get) a task.
ACTIVE_RUN_STATES = ("pending", "running")
# Internal double-texting park state: the run waits behind an active run and has no
# task or worker. Reported as ``pending`` over the API (Run.validate_status).
QUEUED_RUN_STATE = "queued"
# Stamped into execution_params when a multitask gate pre-empts a live run, so the heartbeat,
# the reaper and the drain path end it as ``interrupted`` even if the pub/sub cancel was lost.
INTERRUPT_REQUESTED_KEY = "_interrupt_requested_at"


def interrupt_requested_expr() -> ColumnElement[str]:
    """The pre-emption stamp as SQL text (``->>``); NULL for a run that was never pre-empted."""
    return RunORM.execution_params[INTERRUPT_REQUESTED_KEY].astext


def mark_interrupt_requested(run: RunORM, *, now: datetime) -> None:
    """Record the pre-emption on the row (a fresh dict, so the JSONB change is detected). The first
    stamp stays: a repeated double-text must not restart the pre-emption timeout."""
    params = run.execution_params or {}
    if INTERRUPT_REQUESTED_KEY in params:
        return
    run.execution_params = {**params, INTERRUPT_REQUESTED_KEY: now.isoformat()}


def interrupt_requested(execution_params: dict[str, Any] | None) -> bool:
    return bool((execution_params or {}).get(INTERRUPT_REQUESTED_KEY))


async def settle_threads_of(session: AsyncSession, ended: Iterable[tuple[str, str, str, str]]) -> None:
    """Set the thread of each ended ``(run_id, thread_id, user_id, thread_status)`` row where nothing is
    active any more; one statement per user and status, since the predicate is tenant-scoped. Does not commit."""
    threads_by_target: dict[tuple[str, str], set[str]] = {}
    for _run_id, thread_id, user_id, status in ended:
        threads_by_target.setdefault((user_id, status), set()).add(thread_id)
    if not threads_by_target:
        return
    # Thread rows are locked first (the gate's order), so an admission committing meanwhile is seen
    # by the no-active-run check; one sorted batch per call, so two sweeps cannot deadlock on each other.
    thread_ids_sorted = sorted({thread_id for ids in threads_by_target.values() for thread_id in ids})
    await session.execute(
        select(ThreadORM.thread_id)
        .where(ThreadORM.thread_id.in_(thread_ids_sorted))
        .order_by(ThreadORM.thread_id)
        .with_for_update()
    )
    for (user_id, status), thread_ids in threads_by_target.items():
        # Only a thread still ``busy`` from the ended run: a newer run may have finished or paused it
        # since the run rows committed, and that later status must not be overwritten.
        await set_thread_status_if_no_active_runs(session, thread_ids, status, user_id=user_id, only_if_status="busy")


def with_thread_status(rows: Iterable[tuple[str, str, str]], status: str) -> list[tuple[str, str, str, str]]:
    """``end_preempted_runs`` rows tagged with the thread status ``settle_threads_of`` should apply."""
    return [(run_id, thread_id, user_id, status) for run_id, thread_id, user_id in rows]


async def end_preempted_runs(session: AsyncSession, *where: ColumnElement[bool]) -> list[tuple[str, str, str]]:
    """End the pre-empted (stamped) runs matching ``where`` as ``interrupted`` with their lease cleared.
    Returns their ``(run_id, thread_id, user_id)``; the caller commits, then settles the threads."""
    result = await session.execute(
        update(RunORM)
        .where(interrupt_requested_expr().isnot(None), *where)
        .values(status="interrupted", claimed_by=None, lease_expires_at=None, updated_at=datetime.now(UTC))
        .returning(RunORM.run_id, RunORM.thread_id, RunORM.user_id)
    )
    return [(run_id, thread_id, user_id) for run_id, thread_id, user_id in result.all()]


async def finalize_overdue_preemptions(session: AsyncSession) -> list[tuple[str, str]]:
    """End pre-empted runs still ``running`` past MULTITASK_PREEMPT_TIMEOUT_SECONDS as ``interrupted``,
    so a task that ignores cancellation cannot hold its thread forever. Returns (run_id, thread_id)."""
    cutoff = datetime.now(UTC) - timedelta(seconds=settings.multitask.MULTITASK_PREEMPT_TIMEOUT_SECONDS)
    rows = await end_preempted_runs(
        session,
        RunORM.status == "running",
        interrupt_requested_expr().cast(TIMESTAMP(timezone=True)) < cutoff,
    )
    if not rows:
        return []
    # Run rows first, thread rows in a second transaction: the gate and finalize lock thread-then-run.
    await session.commit()
    await settle_threads_of(session, with_thread_status(rows, "idle"))
    await session.commit()
    logger.warning("Pre-empted runs ignored cancellation past the timeout", run_ids=[row[0] for row in rows])
    return [(run_id, thread_id) for run_id, thread_id, _user_id in rows]


def queued_threads_stmt() -> Select[tuple[str]]:
    """Threads holding a parked run that a recovery sweep may promote; under the ``reject`` policy a
    thread paused on a human-in-the-loop interrupt is left out, since promotion is refused there anyway."""
    stmt = select(RunORM.thread_id).where(RunORM.status == QUEUED_RUN_STATE).distinct()
    if settings.multitask.MULTITASK_PAUSED_THREAD_POLICY == "reject":
        stmt = stmt.join(ThreadORM, ThreadORM.thread_id == RunORM.thread_id).where(ThreadORM.status != "interrupted")
    return stmt


def has_live_executor(run: RunORM, *, now: datetime | None = None) -> bool:
    """Whether a task may still be executing ``run``: a local task in this process, or a worker
    claim whose lease has not expired (the ownership predicate ``interrupt_unowned_run`` uses)."""
    if active_runs.get(run.run_id) is not None:
        return True
    if run.claimed_by is None:
        return False
    return run.lease_expires_at is None or run.lease_expires_at >= (now or datetime.now(UTC))


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
    only_if_status: str | None = None,
) -> None:
    """Update threads that no longer have a pending or running run.

    Does not commit so callers can keep the run and thread transitions in
    one transaction. Queued runs hold no task, so they do not count. With
    ``only_if_status``, a thread whose status has moved on since is left alone.
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
    conditions = [ThreadORM.thread_id.in_(thread_ids), ThreadORM.user_id == user_id, ~active_run_exists]
    if only_if_status is not None:
        conditions.append(ThreadORM.status == validate_thread_status(only_if_status))
    await session.execute(update(ThreadORM).where(*conditions).values(status=validated, updated_at=datetime.now(UTC)))


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


async def drop_queued_runs(session: AsyncSession, thread_id: str, *, user_id: str) -> list[str]:
    """Interrupt every parked run on a thread in place (guarded, committed) and return their ids so
    the caller can close their streams; the thread row is left alone."""
    result = await session.execute(
        update(RunORM)
        .where(
            RunORM.thread_id == thread_id,
            RunORM.user_id == user_id,
            RunORM.status == QUEUED_RUN_STATE,
        )
        .values(status="interrupted", updated_at=datetime.now(UTC))
        .returning(RunORM.run_id)
    )
    dropped = [row[0] for row in result.all()]
    if dropped:
        await session.commit()
        logger.info("Dropped queued runs", thread_id=thread_id, run_ids=dropped)
    return dropped


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

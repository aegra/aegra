"""Run and thread status management.

Provides the database-level status update operations used by both the
API layer (cancel, interrupt) and the execution layer (run_executor,
worker_executor). Extracted from api/runs.py to eliminate the circular
dependency where service code imported from the API module.
"""

from datetime import UTC, datetime
from typing import Any, cast

import structlog
from sqlalchemy import CursorResult, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import Thread as ThreadORM
from aegra_api.core.orm import _get_session_maker
from aegra_api.core.serializers import GeneralSerializer
from aegra_api.utils.status_compat import validate_run_status, validate_thread_status

logger = structlog.getLogger(__name__)
_serializer = GeneralSerializer()


async def get_run_status(run_id: str) -> str | None:
    """Return a run's current status, or None if the run no longer exists.

    Used at rollback-fork time to re-read the target's status, so a run that
    raced to success after the admission gate is not reverted.
    """
    maker = _get_session_maker()
    async with maker() as session:
        return await session.scalar(select(RunORM.status).where(RunORM.run_id == run_id))


async def update_run_status(
    run_id: str,
    status: str,
    *,
    output: Any = None,
    error: str | None = None,
) -> None:
    """Persist a run's status to the database.

    Opens a short-lived session to avoid holding a connection during
    long-running graph execution.
    """
    validated = validate_run_status(status)
    maker = _get_session_maker()
    async with maker() as session:
        values: dict[str, Any] = {
            "status": validated,
            "updated_at": datetime.now(UTC),
        }
        if output is not None:
            values["output"] = _safe_serialize(output, run_id)
        if error is not None:
            values["error_message"] = error

        logger.info("Updating run status", run_id=run_id, status=validated)
        await session.execute(update(RunORM).where(RunORM.run_id == run_id).values(**values))
        await session.commit()


async def try_mark_run_running(run_id: str) -> bool:
    """CAS a run to ``running``; False when it is no longer pending/running.

    A run the multitask gate pre-empted between dispatch and start must not
    resurrect itself — the gate owns its terminal status, and a resurrected run's
    finalize would stomp thread state the newer run now owns.
    """
    maker = _get_session_maker()
    async with maker() as session:
        result = cast(
            "CursorResult[Any]",
            await session.execute(
                update(RunORM)
                .where(RunORM.run_id == run_id, RunORM.status.in_(("pending", "running")))
                .values(status="running", updated_at=datetime.now(UTC))
            ),
        )
        await session.commit()
    return result.rowcount > 0


async def terminalize_user_cancel(run_id: str, thread_id: str) -> None:
    """Converge a user-initiated cancel: unpark a queued run, else finalize an active one.

    A queued run holds no task and does not occupy the thread, so a guarded status
    flip suffices — but if it was the head of a stranded queue (its active predecessor
    already gone), the runs parked behind it must not wait for the recovery sweep, so
    dispatch follows the unpark. An active run needs the full finalize (thread reset +
    next-queued dispatch); the CAS inside finalize_run makes this race benignly with
    the run task's own finalize — exactly one caller wins and performs the duties.
    """
    maker = _get_session_maker()
    async with maker() as session:
        result = cast(
            "CursorResult[Any]",
            await session.execute(
                update(RunORM)
                .where(RunORM.run_id == run_id, RunORM.status == "queued")
                .values(status="interrupted", updated_at=datetime.now(UTC))
            ),
        )
        await session.commit()
    if result.rowcount > 0:
        logger.info("Cancelled queued run", run_id=run_id)
        # Idempotent: no-ops when a run occupies the thread or a HITL pause holds it.
        # Deferred import breaks the run_status <- executor <- run_executor cycle.
        from aegra_api.services.executor import executor

        try:
            await executor.dispatch_next_for_thread(thread_id)
        except Exception:  # intentionally broad — the cancel itself is already committed
            logger.exception("Failed to dispatch next queued run", thread_id=thread_id)
        return
    await finalize_run(
        run_id,
        thread_id,
        status="interrupted",
        thread_status="idle",
        output={},
        clear_lease=True,
    )


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


async def finalize_run(
    run_id: str,
    thread_id: str,
    *,
    status: str,
    thread_status: str,
    output: Any = None,
    error: str | None = None,
    allow_terminal_override: bool = False,
    claimed_by: str | None = None,
    clear_lease: bool = False,
) -> None:
    """Update run status + thread status in a single transaction.

    Batches two UPDATE statements into one DB round-trip instead of
    opening separate sessions for update_run_status and set_thread_status.

    The run UPDATE only matches a still-active run (``running``/``pending``) so a
    run a multitask gate already moved to ``interrupted`` cannot be resurrected by
    its own late finalize. If it matches nothing, this finalize does not own the run,
    so the thread row and queue dispatch are left untouched (another run owns them).
    ``allow_terminal_override`` lifts the active-run filter for the authoritative late
    corrector (the worker timeout handler), which must overwrite a terminal status.
    ``claimed_by`` narrows the override to runs still leased by that worker, so the
    corrector cannot stomp an 'interrupted' a multitask gate wrote (gate cancels
    clear the lease). ``clear_lease`` releases the lease with the terminal write so a
    worker whose pub/sub cancel was lost self-cancels on its next heartbeat.
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
    if clear_lease:
        run_values["claimed_by"] = None
        run_values["lease_expires_at"] = None

    async with maker() as session:
        # Lock the thread row FIRST so finalize and the create-time multitask gate
        # (which also locks thread-then-run) share one lock order — without this
        # an interrupt/rollback create racing this finalize deadlocks (40P01).
        await session.execute(select(ThreadORM.thread_id).where(ThreadORM.thread_id == thread_id).with_for_update())
        # The corrector (timeout) may overwrite the cancel handler's 'interrupted', but never a
        # committed 'success'/'error' — so even override only widens the set, never drops the filter.
        overwritable = ("running", "pending", "interrupted") if allow_terminal_override else ("running", "pending")
        conditions = [RunORM.run_id == run_id, RunORM.status.in_(overwritable)]
        if claimed_by is not None:
            conditions.append(RunORM.claimed_by == claimed_by)
        # DML execute() returns a CursorResult at runtime; cast so ``.rowcount`` is reachable.
        run_result = cast(
            "CursorResult[Any]",
            await session.execute(update(RunORM).where(*conditions).values(**run_values)),
        )
        if run_result.rowcount == 0:
            # This finalize does not own the run (a multitask gate already terminalized it):
            # leave the thread row and the queue alone — another run owns them now. Critically,
            # this prevents a late pre-empted finalize from stomping a HITL pause back to 'idle'.
            await session.commit()
            logger.info("Finalize skipped non-owned run", run_id=run_id, attempted_status=validated_run)
            return
        await session.execute(
            update(ThreadORM)
            .where(ThreadORM.thread_id == thread_id)
            .values(status=validated_thread, updated_at=datetime.now(UTC))
        )
        await session.commit()

    logger.info("Finalized run", run_id=run_id, status=validated_run, thread_status=validated_thread)

    # Start the next double-texted (queued) run, if any. Best-effort: a dispatch
    # failure must not bubble into execute_run's error path and clobber the status
    # just committed — the queued run is durable and recovered by the reaper / boot
    # sweep. Deferred import breaks run_status <- executor <- run_executor cycle.
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

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
ACTIVE_RUN_STATES = ("pending", "running")


async def update_run_status(
    run_id: str,
    status: str,
    *,
    user_id: str,
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
        await session.execute(
            update(RunORM)
            .where(
                RunORM.run_id == run_id,
                RunORM.user_id == user_id,
            )
            .values(**values)
        )
        await session.commit()


async def start_run(run_id: str, *, user_id: str) -> bool:
    """Move an active run to running without reviving a terminal run."""
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
    one transaction.
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
    """
    now = datetime.now(UTC)
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
        return False

    await set_thread_status_if_no_active_runs(session, [thread_id], "idle", user_id=user_id)
    await session.commit()
    logger.info("Interrupted unowned run", run_id=run_id, thread_id=thread_id)
    return True


async def finalize_run(
    run_id: str,
    thread_id: str,
    *,
    user_id: str,
    status: str,
    thread_status: str,
    output: Any = None,
    error: str | None = None,
    allowed_current_statuses: Collection[str] = ACTIVE_RUN_STATES,
) -> bool:
    """Conditionally update run and thread status in one transaction.

    Returns false when another actor has already made the run terminal. This
    prevents an expired worker from overwriting a reconciled cancellation.
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
        result = await session.execute(
            update(RunORM)
            .where(
                RunORM.run_id == run_id,
                RunORM.user_id == user_id,
                RunORM.status.in_(allowed_current_statuses),
            )
            .values(**run_values)
            .returning(RunORM.run_id)
        )
        if result.scalar_one_or_none() is None:
            await session.rollback()
            logger.info("Skipped finalizing terminal run", run_id=run_id, status=validated_run)
            return False

        await set_thread_status_if_no_active_runs(
            session,
            [thread_id],
            validated_thread,
            user_id=user_id,
        )
        await session.commit()

    logger.info("Finalized run", run_id=run_id, status=validated_run, thread_status=validated_thread)
    return True


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

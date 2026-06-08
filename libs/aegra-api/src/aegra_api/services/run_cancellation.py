"""User-driven run cancellation use case.

Symmetric to ``run_preparation`` (which orchestrates run creation),
this module orchestrates the cancel-by-user side: tag the in-process
``CancellationRegistry`` so the executor classifies the resulting
CancelledError correctly, broadcast the cancel/interrupt via the
streaming layer, and provide the safety-net DB update the HTTP cancel
endpoint relies on.

Endpoint code (``api/runs.py``) imports from here and stays free of
registry / streaming / DB-update plumbing. ``run_status.py`` stays
strictly DB-only, ``streaming_service.py`` stays focused on streaming.
"""

from datetime import UTC, datetime
from typing import Literal

import structlog
from redis import RedisError
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from aegra_api.core.cancellation_state import cancellations
from aegra_api.core.orm import Run as RunORM
from aegra_api.observability.metrics import extract_graph_id, get_worker_metrics
from aegra_api.services.streaming_service import streaming_service

logger = structlog.getLogger(__name__)

CancelAction = Literal["cancel", "interrupt"]

# Terminal run states. Duplicated here intentionally to avoid an import
# cycle with ``run_waiters`` and to keep this module self-contained.
TERMINAL_RUN_STATES = frozenset({"success", "error", "interrupted"})


async def signal_user_cancel(run_id: str, action: CancelAction) -> None:
    """Tag the run as user-cancelled and broadcast the cancel/interrupt signal.

    Tagging happens **before** the broadcast so any worker that picks up
    the cancel (locally or via Redis pub/sub) classifies the resulting
    CancelledError as a user cancel. If the broadcast itself fails, the
    user tag is rolled back via ``clear(only="user")`` so a lease-loss
    tag that arrived in the meantime survives.

    Catches the narrow set of failure modes the streaming layer can
    legitimately raise (Redis down, broker missing, broadcast races) —
    deeper exceptions still propagate.
    """
    cancellations.mark(run_id, "user")
    fn = streaming_service.interrupt_run if action == "interrupt" else streaming_service.cancel_run
    try:
        await fn(run_id)
    except (RedisError, RuntimeError):
        cancellations.clear(run_id, only="user")
        raise


async def mark_interrupted_unless_terminal(session: AsyncSession, run_id: str) -> None:
    """Set ``status='interrupted'`` only if the run is not already in a terminal state.

    Safety net for the cancel endpoint: ``execute_run`` finalizes user
    cancels itself, but the cancel is async — this catches the gap
    where the broadcast has been issued but the executing coroutine has
    not yet observed CancelledError or committed its own status update.
    Commits within the passed session.
    """
    await session.execute(
        update(RunORM)
        .where(RunORM.run_id == str(run_id), RunORM.status.notin_(TERMINAL_RUN_STATES))
        .values(status="interrupted", updated_at=datetime.now(UTC))
    )
    await session.commit()


async def try_cancel_pending(session: AsyncSession, run_orm: RunORM) -> bool:
    """Atomically transition a pending run to ``interrupted`` via CAS.

    Returns ``True`` when the CAS hit (run was still pending and we
    finalized it). Returns ``False`` when a worker claimed the run
    between the caller's SELECT and our UPDATE — caller should re-fetch
    and fall through to the running-cancel path.

    On a successful CAS the run never enters ``execute_run``, so this
    is the only place that increments ``runs_completed{status="interrupted"}``
    for the pending-cancel branch and the only place that emits the
    terminal ``end`` event for clients already streaming via
    ``/threads/{thread_id}/runs/stream`` or
    ``/threads/{thread_id}/runs/{run_id}/stream`` — without that signal
    those clients keep waiting on the broker even though the DB row is
    already terminal.
    """
    cas_result = await session.execute(
        update(RunORM)
        .where(RunORM.run_id == run_orm.run_id, RunORM.status == "pending")
        .values(status="interrupted", updated_at=datetime.now(UTC))
    )
    if cas_result.rowcount == 0:  # type: ignore[union-attr]
        await session.rollback()
        return False
    await session.commit()
    logger.info("Cancelled pending run", run_id=run_orm.run_id)
    metrics = get_worker_metrics()
    if metrics is not None:
        graph_id = extract_graph_id(run_orm.execution_params)
        metrics.runs_completed.labels(graph_id=graph_id, status="interrupted").inc()
    # Best-effort terminal signal so attached streaming clients see the
    # end event. ``signal_run_cancelled`` is idempotent (no-op if the
    # broker is already finished) and goes through the Redis-backed
    # broker, so it works cross-instance. A Redis outage here must not
    # fail the cancel — the DB row is already ``interrupted`` and the
    # client will eventually time out instead of finalizing successfully.
    try:
        await streaming_service.signal_run_cancelled(run_orm.run_id)
    except (RedisError, RuntimeError) as exc:
        logger.warning(
            "Failed to broadcast pending-cancel end event",
            run_id=run_orm.run_id,
            error=str(exc),
        )
    return True


async def cancel_running_run(session: AsyncSession, run_id: str, action: CancelAction) -> None:
    """Cancel a running run: tag, broadcast, then write the safety-net status.

    Pairs ``signal_user_cancel`` (registry tag + streaming broadcast)
    with ``mark_interrupted_unless_terminal`` (DB safety net) so the
    cancel endpoint stays a thin glue layer.
    """
    await signal_user_cancel(run_id, action)
    await mark_interrupted_unless_terminal(session, run_id)

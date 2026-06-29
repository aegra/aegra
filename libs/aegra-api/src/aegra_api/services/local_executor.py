"""In-process executor using asyncio tasks.

Used in development mode (REDIS_BROKER_ENABLED=false). Runs execute
as background coroutines in the same event loop as the API server.
"""

import asyncio
import contextlib
from datetime import UTC, datetime
from typing import Any, cast

import structlog
from sqlalchemy import CursorResult, select, update
from sqlalchemy.exc import SQLAlchemyError

from aegra_api.core.active_runs import active_runs
from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import Thread as ThreadORM
from aegra_api.core.orm import _get_session_maker
from aegra_api.models.run_job import RunJob
from aegra_api.observability.span_enrichment import make_run_trace_context
from aegra_api.services.base_executor import BaseExecutor

logger = structlog.getLogger(__name__)

_TERMINAL_STATUSES = frozenset({"success", "error", "interrupted"})
_STRANDED_SWEEP_INTERVAL_SECONDS = 30


async def _is_run_terminal(run_id: str) -> bool:
    """True if the run reached a terminal state (or no longer exists)."""
    maker = _get_session_maker()
    async with maker() as session:
        status = await session.scalar(select(RunORM.status).where(RunORM.run_id == run_id))
    return status is None or status in _TERMINAL_STATUSES


class LocalExecutor(BaseExecutor):
    """Runs graphs as local asyncio tasks (single-instance dev mode)."""

    async def submit(self, job: RunJob) -> None:
        # Deferred import: run_executor imports services that reference
        # the executor singleton, creating a circular chain at module level.
        from aegra_api.services.run_executor import execute_run

        trace_ctx = make_run_trace_context(
            job.identity.run_id,
            job.identity.thread_id,
            job.identity.graph_id,
            job.user.identity,
            extra_metadata=job.run_metadata,
        )
        task = asyncio.create_task(execute_run(job), context=trace_ctx)
        active_runs[job.identity.run_id] = task
        logger.info(
            "Submitted run to local executor",
            run_id=job.identity.run_id,
            task_id=id(task),
        )

    async def wait_for_completion(self, run_id: str, *, timeout: float = 300.0) -> None:
        # A run may still be 'queued' (no task yet) — poll until it is dispatched
        # (a task appears) or reaches a terminal state, so join/wait don't return
        # an empty result for a double-texted run that hasn't started.
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        poll_count = 0
        while loop.time() < deadline:
            task = active_runs.get(run_id)
            if task is not None:
                with contextlib.suppress(TimeoutError, asyncio.CancelledError):
                    await asyncio.wait_for(asyncio.shield(task), timeout=max(0.0, deadline - loop.time()))
                return
            # In-memory active_runs is checked every tick; the DB probe (for a
            # still-queued or already-terminal run) only every other tick.
            poll_count += 1
            if poll_count % 2 == 0 and await _is_run_terminal(run_id):
                return
            await asyncio.sleep(0.5)

    async def start(self) -> None:
        self._accepting = True
        logger.info("Local executor started (in-process asyncio tasks)")
        await self._recover_orphaned_queue()
        self._sweep_task = asyncio.create_task(self._stranded_queue_loop())

    async def _stranded_queue_loop(self) -> None:
        """Periodically dispatch threads stranded with a queued run but nothing running.

        Mirrors the prod lease reaper: in dev a swallowed finalize-time dispatch failure
        would otherwise wedge a queued run until the next restart.
        """
        while self._accepting:
            try:
                await asyncio.sleep(_STRANDED_SWEEP_INTERVAL_SECONDS)
            except asyncio.CancelledError:
                break
            if not self._accepting:
                break
            # Catch-all (like the prod reaper's loop) so a corrupt-params row or any other
            # non-SQLAlchemyError cannot permanently kill the periodic recovery task.
            try:
                await self._sweep_stranded_queues()
            except Exception:
                logger.exception("Stranded-queue sweep failed")

    async def _sweep_stranded_queues(self) -> None:
        """Re-dispatch every thread holding a queued run (a no-op when one is occupying)."""
        maker = _get_session_maker()
        async with maker() as session:
            threads = {
                row[0]
                for row in (
                    await session.execute(select(RunORM.thread_id).where(RunORM.status == "queued").distinct())
                ).all()
            }
        for thread_id in threads:
            # Per-thread isolation: one bad row must not abort recovery for the rest.
            try:
                await self.dispatch_next_for_thread(thread_id)
            except Exception:
                logger.exception("Stranded-queue dispatch failed", thread_id=thread_id)

    async def _recover_orphaned_queue(self) -> None:
        """Recover threads with in-flight or queued runs after a restart.

        A fresh dev process has no live tasks, so any running/pending run is an
        orphan from the dead process. For each affected thread, fail the orphaned
        run (so the thread isn't wedged forever, esp. under reject) and dispatch
        the next queued run, if any.
        """
        maker = _get_session_maker()
        async with maker() as session:
            threads = {
                row[0]
                for row in (
                    await session.execute(
                        select(RunORM.thread_id).where(RunORM.status.in_(("queued", "running", "pending"))).distinct()
                    )
                ).all()
            }
        if not threads:
            return

        logger.warning("Recovering threads after restart", thread_count=len(threads))
        for thread_id in threads:
            try:
                async with maker() as session:
                    result = cast(
                        "CursorResult[Any]",
                        await session.execute(
                            update(RunORM)
                            .where(RunORM.thread_id == thread_id, RunORM.status.in_(("running", "pending")))
                            .values(
                                status="error",
                                error_message="Orphaned by server restart",
                                updated_at=datetime.now(UTC),
                            )
                        ),
                    )
                    if result.rowcount > 0:
                        # The orphans' finalizes never ran: reset the thread the way an error
                        # finalize would have, so it is not left 'busy' with no active run.
                        # Queued-only threads match nothing here, preserving a HITL pause.
                        await session.execute(
                            update(ThreadORM)
                            .where(ThreadORM.thread_id == thread_id)
                            .values(status="error", updated_at=datetime.now(UTC))
                        )
                    await session.commit()
                await self.dispatch_next_for_thread(thread_id)
            except SQLAlchemyError:
                logger.exception("Failed to recover thread after restart", thread_id=thread_id)

    async def stop(self) -> None:
        self._accepting = False
        sweep_task = getattr(self, "_sweep_task", None)
        if sweep_task is not None:
            sweep_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sweep_task
        tasks_to_cancel = [task for task in active_runs.values() if not task.done()]
        for task in tasks_to_cancel:
            task.cancel()
        if tasks_to_cancel:
            logger.info("Draining cancelled tasks", count=len(tasks_to_cancel))
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
        logger.info("Local executor stopped")

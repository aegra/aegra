"""In-process executor using asyncio tasks.

Used in development mode (REDIS_BROKER_ENABLED=false). Runs execute
as background coroutines in the same event loop as the API server.
"""

import asyncio
import contextlib

import structlog

from aegra_api.core.active_runs import active_runs
from aegra_api.core.cancellation_state import cancellations
from aegra_api.models.run_job import RunJob
from aegra_api.observability.span_enrichment import make_run_trace_context
from aegra_api.services.base_executor import BaseExecutor

logger = structlog.getLogger(__name__)


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
        # Drop the cancel tag once the run is done — execute_run leaves it
        # alive so the executor can read it (here: via done-callback).
        run_id = job.identity.run_id
        task.add_done_callback(lambda _: cancellations.clear(run_id))
        logger.info(
            "Submitted run to local executor",
            run_id=job.identity.run_id,
            task_id=id(task),
        )

    async def wait_for_completion(self, run_id: str, *, timeout: float = 300.0) -> None:
        task = active_runs.get(run_id)
        if task is None:
            return
        with contextlib.suppress(TimeoutError, asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(task), timeout=timeout)

    async def start(self) -> None:
        logger.info("Local executor started (in-process asyncio tasks)")

    async def stop(self) -> None:
        tasks_to_cancel = [task for task in active_runs.values() if not task.done()]
        for task in tasks_to_cancel:
            task.cancel()
        if tasks_to_cancel:
            logger.info("Draining cancelled tasks", count=len(tasks_to_cancel))
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
        logger.info("Local executor stopped")

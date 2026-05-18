"""Redis-backed executor with concurrent async execution and lease-based crash recovery.

Production mode (REDIS_BROKER_ENABLED=true). Each worker loop dequeues
run_ids from Redis via BLPOP and spawns up to N_JOBS_PER_WORKER
concurrent asyncio tasks. Each task acquires a lease, executes the
graph with periodic heartbeats, and releases the lease on completion.
If a worker crashes, the lease expires and a background reaper
re-enqueues the run.
"""

import asyncio
import contextlib
import contextvars
import os
import re
import socket
from datetime import UTC, datetime, timedelta

import structlog
from asgi_correlation_id import correlation_id
from redis import RedisError
from sqlalchemy import select, update

from aegra_api.core.active_runs import active_runs
from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import _get_session_maker
from aegra_api.core.redis_manager import redis_manager
from aegra_api.models.run_job import RunJob
from aegra_api.observability.span_enrichment import merge_run_metadata, seed_otel_trace_id, set_trace_context
from aegra_api.services.base_executor import BaseExecutor
from aegra_api.services.run_executor import _lease_loss_cancellations, execute_run
from aegra_api.services.run_status import finalize_run, update_run_status
from aegra_api.settings import settings

logger = structlog.getLogger(__name__)

# Terminal run states (kept local to avoid circular import with run_waiters -> executor)
_TERMINAL_STATUSES = frozenset({"success", "error", "interrupted"})
_RUN_ID_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def _is_valid_run_id(value: str) -> bool:
    """Check if a string is a valid UUID v4 hex format."""
    return bool(_RUN_ID_PATTERN.match(value))


class WorkerExecutor(BaseExecutor):
    """Dispatches runs via Redis List; workers consume with BLPOP + semaphore."""

    def __init__(self) -> None:
        self._worker_tasks: list[asyncio.Task[None]] = []
        self._job_tasks: set[asyncio.Task[None]] = set()
        self._running = False
        self._instance_id = f"{socket.gethostname()}-{os.getpid()}"

    # ------------------------------------------------------------------
    # Submit (API side)
    # ------------------------------------------------------------------

    async def submit(self, job: RunJob) -> None:
        client = redis_manager.get_client()
        await client.rpush(settings.worker.WORKER_QUEUE_KEY, job.identity.run_id)  # type: ignore[arg-type]
        logger.info(
            "Enqueued run_id to job queue",
            run_id=job.identity.run_id,
            queue=settings.worker.WORKER_QUEUE_KEY,
        )

    # ------------------------------------------------------------------
    # Wait for completion (API side)
    # ------------------------------------------------------------------

    async def wait_for_completion(self, run_id: str, *, timeout: float = 300.0) -> None:
        """Wait for a run to finish by polling a Redis done-key with DB fallback."""
        done_key = f"{settings.redis.REDIS_CHANNEL_PREFIX}done:{run_id}"
        client = redis_manager.get_client()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        poll_count = 0

        while loop.time() < deadline:
            try:
                if await client.exists(done_key):
                    return
            except RedisError:
                pass

            poll_count += 1
            if poll_count % 2 == 0 and await _is_run_terminal(run_id):
                return

            await asyncio.sleep(2.0)

        raise TimeoutError(f"Run {run_id} did not complete within {timeout}s")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._running = True
        count = settings.worker.WORKER_COUNT
        if count == 0:
            logger.warning(
                "WORKER_COUNT=0: no workers on this instance, runs will queue until another instance picks them up"
            )
        for idx in range(count):
            name = f"{self._instance_id}-worker-{idx}"
            task = asyncio.create_task(self._worker_loop(name))
            self._worker_tasks.append(task)

        max_concurrent = count * settings.worker.N_JOBS_PER_WORKER
        logger.info(
            "Worker executor started",
            worker_count=count,
            jobs_per_worker=settings.worker.N_JOBS_PER_WORKER,
            max_concurrent=max_concurrent,
            instance=self._instance_id,
        )

    async def stop(self) -> None:
        self._running = False
        drain_timeout = settings.worker.WORKER_DRAIN_TIMEOUT

        # Wait for in-flight job tasks to finish
        if self._job_tasks:
            logger.info("Draining in-flight jobs", count=len(self._job_tasks))
            _, pending = await asyncio.wait(self._job_tasks, timeout=drain_timeout)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        # Cancel worker loops
        for task in self._worker_tasks:
            task.cancel()
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)

        self._worker_tasks.clear()
        self._job_tasks.clear()
        logger.info("Worker executor stopped", instance=self._instance_id)

    # ------------------------------------------------------------------
    # Worker loop (dequeue + spawn concurrent tasks)
    # ------------------------------------------------------------------

    async def _worker_loop(self, worker_name: str) -> None:
        """Dequeue run_ids and spawn concurrent execution tasks.

        Each worker loop manages a semaphore that limits concurrent runs
        to N_JOBS_PER_WORKER. When all slots are busy, the loop blocks
        on semaphore.acquire until a slot frees up.
        """
        n_jobs = settings.worker.N_JOBS_PER_WORKER
        if n_jobs <= 0:
            raise ValueError(f"N_JOBS_PER_WORKER must be >= 1, got {n_jobs}")
        semaphore = asyncio.Semaphore(n_jobs)
        logger.info(
            "Worker started",
            worker=worker_name,
            max_concurrent=settings.worker.N_JOBS_PER_WORKER,
        )

        while self._running:
            try:
                await semaphore.acquire()

                if not self._running:
                    semaphore.release()
                    break

                run_id = await self._dequeue()
                if run_id is None:
                    semaphore.release()
                    continue

                if not _is_valid_run_id(run_id):
                    logger.warning("Invalid run_id dequeued, discarding", value=run_id[:64])
                    semaphore.release()
                    continue

                task = asyncio.create_task(self._execute_and_release(run_id, worker_name, semaphore))
                self._job_tasks.add(task)
                task.add_done_callback(self._job_tasks.discard)

            except asyncio.CancelledError:
                break
            except Exception:
                semaphore.release()
                logger.exception("Unexpected error in worker loop", worker=worker_name)
                await asyncio.sleep(1.0)

        logger.info("Worker stopped", worker=worker_name)

    async def _execute_and_release(
        self,
        run_id: str,
        worker_name: str,
        semaphore: asyncio.Semaphore,
    ) -> None:
        """Execute a run with lease + timeout, then release the semaphore slot."""
        # Register in active_runs so cancel-on-disconnect and explicit
        # cancel can find and cancel this specific job task.
        current_task = asyncio.current_task()
        if current_task is not None:
            active_runs[run_id] = current_task
        try:
            await asyncio.wait_for(
                self._execute_with_lease(run_id, worker_name),
                timeout=settings.worker.BG_JOB_TIMEOUT_SECS,
            )
        except TimeoutError:
            logger.error(
                "Job exceeded timeout, killing",
                worker=worker_name,
                run_id=run_id,
                timeout_secs=settings.worker.BG_JOB_TIMEOUT_SECS,
            )
            # Look up thread_id so we can set thread status to "error" too.
            # When wait_for fires, execute_run's CancelledError handler runs first
            # and sets thread_status="idle" — we must correct that to "error".
            thread_id = await _get_thread_id_for_run(run_id)
            if thread_id is not None:
                await finalize_run(
                    run_id,
                    thread_id,
                    status="error",
                    thread_status="error",
                    error="Job exceeded maximum execution time",
                )
            else:
                # Fallback: update run status only (thread_id lookup failed)
                await update_run_status(run_id, "error", error="Job exceeded maximum execution time")
            await _release_lease(run_id, worker_name)
        except asyncio.CancelledError:
            logger.info("Job task cancelled", worker=worker_name, run_id=run_id)
            raise
        except Exception:
            logger.exception("Unexpected error in job execution", run_id=run_id)
        finally:
            active_runs.pop(run_id, None)
            semaphore.release()

    # ------------------------------------------------------------------
    # Job execution (lease + heartbeat)
    # ------------------------------------------------------------------

    async def _dequeue(self) -> str | None:
        """BLPOP with 5s timeout. Falls back to Postgres polling if Redis is down."""
        try:
            client = redis_manager.get_client()
            result = await client.blpop(settings.worker.WORKER_QUEUE_KEY, timeout=5)  # type: ignore[arg-type]
            if result is None:
                return None
            return result[1]  # type: ignore[return-value]
        except RedisError as exc:
            logger.warning("Redis BLPOP failed, falling back to Postgres poll", error=str(exc))
            await asyncio.sleep(settings.worker.POSTGRES_POLL_INTERVAL_SECONDS)
            return await self._poll_postgres()

    async def _execute_with_lease(self, run_id: str, worker_name: str) -> None:
        """Acquire lease, load job from DB, execute with heartbeat."""
        lease_acquired_at = datetime.now(UTC)
        loaded = await _acquire_and_load(run_id, worker_name)
        if loaded is None:
            logger.debug("Lease not acquired or job missing, skipping", run_id=run_id, worker=worker_name)
            return

        _restore_trace_context(run_id, loaded.job, loaded.trace)
        logger.info(
            "Worker picked up run",
            worker=worker_name,
            run_id=run_id,
            graph_id=loaded.job.identity.graph_id,
        )
        # Wrap execute_run in a task so the heartbeat can cancel it on
        # lease loss, preventing double execution by a second worker.
        job_task = asyncio.create_task(execute_run(loaded.job))
        heartbeat_task = asyncio.create_task(
            _heartbeat_loop(run_id, worker_name, job_task=job_task),
            context=contextvars.copy_context(),
        )

        try:
            await job_task
        except asyncio.CancelledError:
            logger.info("Worker job cancelled", worker=worker_name, run_id=run_id)
        except Exception:
            logger.exception("Worker job failed", worker=worker_name, run_id=run_id)
        finally:
            # Cancel both child tasks — job_task may still be running if
            # this coroutine was cancelled by wait_for timeout (CancelledError
            # is delivered to `await job_task`, but the Task itself is not
            # cancelled automatically).
            if not job_task.done():
                job_task.cancel()
            heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.gather(job_task, heartbeat_task, return_exceptions=True)
            await _release_lease(run_id, worker_name)

            elapsed = (datetime.now(UTC) - lease_acquired_at).total_seconds()
            logger.info(
                "Worker finished run",
                worker=worker_name,
                run_id=run_id,
                execution_seconds=round(elapsed, 2),
            )

    @staticmethod
    async def _poll_postgres() -> str | None:
        """Pick the oldest pending, unclaimed run from Postgres."""
        maker = _get_session_maker()
        async with maker() as session:
            run_id = await session.scalar(
                select(RunORM.run_id)
                .where(RunORM.status == "pending", RunORM.claimed_by.is_(None))
                .order_by(RunORM.created_at.asc())
                .limit(1)
            )
            return run_id


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


async def _get_thread_id_for_run(run_id: str) -> str | None:
    """Look up the thread_id for a run. Returns None if the row is missing."""
    maker = _get_session_maker()
    async with maker() as session:
        return await session.scalar(select(RunORM.thread_id).where(RunORM.run_id == run_id))


# ------------------------------------------------------------------
# Lease operations (module-level for reuse by LeaseReaper)
# ------------------------------------------------------------------


class _LoadedRun:
    """RunJob plus raw trace metadata from execution_params."""

    __slots__ = ("job", "trace")

    def __init__(self, job: RunJob, trace: dict[str, str]) -> None:
        self.job = job
        self.trace = trace


async def _acquire_and_load(run_id: str, worker_name: str) -> _LoadedRun | None:
    """Acquire lease and load job in a single DB session.

    Combines the lease UPDATE + job SELECT into one session. If the row
    is missing execution_params (data corruption / pre-migration row),
    releases the claim and marks the run as errored.
    """
    lease_until = datetime.now(UTC) + timedelta(seconds=settings.worker.LEASE_DURATION_SECONDS)
    maker = _get_session_maker()
    async with maker() as session:
        result = await session.execute(
            update(RunORM)
            .where(
                RunORM.run_id == run_id,
                RunORM.status == "pending",
                RunORM.claimed_by.is_(None),
            )
            .values(claimed_by=worker_name, lease_expires_at=lease_until, status="running")
        )
        if result.rowcount == 0:  # type: ignore[union-attr]
            await session.rollback()
            return None

        run_orm = await session.scalar(select(RunORM).where(RunORM.run_id == run_id))
        await session.commit()

        if run_orm is None or run_orm.execution_params is None:
            logger.warning(
                "Run not found or missing execution_params after lease, releasing claim",
                run_id=run_id,
                worker=worker_name,
            )
            await session.execute(
                update(RunORM)
                .where(RunORM.run_id == run_id, RunORM.claimed_by == worker_name)
                .values(
                    claimed_by=None,
                    lease_expires_at=None,
                    status="error",
                    error_message="Run missing execution_params (data corruption or pre-migration row)",
                )
            )
            await session.commit()
            return None

        job = RunJob.from_run_orm(run_orm)
        trace = run_orm.execution_params.get("trace", {})
        return _LoadedRun(job=job, trace=trace)


async def _release_lease(run_id: str, worker_name: str) -> None:
    """Clear lease fields after job completion, only if this worker still owns the lease."""
    maker = _get_session_maker()
    async with maker() as session:
        await session.execute(
            update(RunORM)
            .where(RunORM.run_id == run_id, RunORM.claimed_by == worker_name)
            .values(claimed_by=None, lease_expires_at=None)
        )
        await session.commit()


async def _heartbeat_loop(
    run_id: str,
    worker_name: str,
    *,
    job_task: asyncio.Task[None] | None = None,
) -> None:
    """Extend lease periodically while the job is running.

    If the lease is lost (another worker claimed the run), cancels
    ``job_task`` to prevent double execution.
    """
    interval = settings.worker.HEARTBEAT_INTERVAL_SECONDS
    duration = settings.worker.LEASE_DURATION_SECONDS
    maker = _get_session_maker()

    while True:
        await asyncio.sleep(interval)
        try:
            new_expiry = datetime.now(UTC) + timedelta(seconds=duration)
            async with maker() as session:
                result = await session.execute(
                    update(RunORM)
                    .where(RunORM.run_id == run_id, RunORM.claimed_by == worker_name)
                    .values(lease_expires_at=new_expiry)
                )
                await session.commit()
            if result.rowcount == 0:  # type: ignore[union-attr]
                logger.warning(
                    "Lease lost, cancelling job to prevent double execution",
                    run_id=run_id,
                    worker=worker_name,
                )
                if job_task is not None and not job_task.done():
                    _lease_loss_cancellations.add(run_id)
                    job_task.cancel()
                return
            logger.debug("Lease extended", run_id=run_id, worker=worker_name)
        except Exception:
            logger.warning("Heartbeat lease extension failed", run_id=run_id, worker=worker_name)


async def _is_run_terminal(run_id: str) -> bool:
    """Check if a run has reached a terminal state in the DB."""
    maker = _get_session_maker()
    async with maker() as session:
        run_orm = await session.scalar(select(RunORM).where(RunORM.run_id == run_id))
        if run_orm is None:
            return True
        return run_orm.status in _TERMINAL_STATUSES


def _restore_trace_context(run_id: str, job: RunJob, trace: dict[str, str]) -> None:
    """Restore OTEL and structlog trace context for a worker-executed run.

    Clears previous context first to prevent bleed between concurrent
    jobs processed by the same worker.  User-supplied ``run_metadata`` is
    merged with the system runtime keys; system keys win on collision —
    see :func:`merge_run_metadata`.
    """
    structlog.contextvars.clear_contextvars()

    original_request_id = trace.get("correlation_id", "")
    if original_request_id:
        correlation_id.set(original_request_id)

    seed_otel_trace_id(run_id)
    system_metadata: dict[str, str | int | float | bool] = {
        "run_id": run_id,
        "thread_id": job.identity.thread_id,
        "graph_id": job.identity.graph_id,
    }
    # Gate on non-empty: requests without an upstream correlation-id header
    # leave ``original_request_id`` as ``""`` — including the empty string
    # would emit a noisy ``langfuse.trace.metadata.original_request_id=""``
    # attribute on every such trace.
    if original_request_id:
        system_metadata["original_request_id"] = original_request_id
    set_trace_context(
        user_id=job.user.identity,
        session_id=job.identity.thread_id,
        trace_name=job.identity.graph_id,
        metadata=merge_run_metadata(job.run_metadata, system_metadata),
    )

    structlog.contextvars.bind_contextvars(
        run_id=run_id,
        thread_id=job.identity.thread_id,
        graph_id=job.identity.graph_id,
        user_id=job.user.identity,
        original_request_id=original_request_id,
    )

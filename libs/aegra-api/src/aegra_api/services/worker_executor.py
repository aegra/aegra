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
import uuid
from collections.abc import Collection
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from asgi_correlation_id import correlation_id
from redis import RedisError
from redis import TimeoutError as RedisTimeoutError
from sqlalchemy import Interval, and_, func, literal, or_, select, update

from aegra_api.core.active_runs import active_runs, explicit_run_cancellations
from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import _get_session_maker
from aegra_api.core.redis_manager import redis_manager
from aegra_api.models.run_job import RunJob
from aegra_api.observability.span_enrichment import merge_run_metadata, set_trace_context
from aegra_api.services.base_executor import BaseExecutor
from aegra_api.services.run_executor import (
    _lease_loss_cancellations,
    _shutdown_cancellations,
    _timeout_cancellations,
    execute_run,
)
from aegra_api.services.run_status import ACTIVE_RUN_STATES, finalize_run
from aegra_api.settings import settings

logger = structlog.getLogger(__name__)

# Terminal run states (kept local to avoid circular import with run_waiters -> executor)
_TERMINAL_STATUSES = frozenset({"success", "error", "interrupted"})
_RUN_ID_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def _is_valid_run_id(value: str) -> bool:
    """Check if a string is a valid UUID v4 hex format."""
    return bool(_RUN_ID_PATTERN.match(value))


class _ClaimSlot:
    """Hands the claim id minted inside ``_execute_with_lease`` to its supervisor.

    The timeout, explicit-cancel and drain paths run one frame up but must still
    fence their writes on the acquisition that frame holds (#502).
    """

    __slots__ = ("claim_id",)

    def __init__(self) -> None:
        self.claim_id: str | None = None


def _lease_deadline() -> Any:
    """Lease expiry on the database clock; the reaper on another host compares against ``now()``."""
    return func.now() + literal(timedelta(seconds=settings.worker.LEASE_DURATION_SECONDS), Interval)


async def _cleanup_cancelled_execution(run_id: str, execution_task: asyncio.Task[None], claim: _ClaimSlot) -> None:
    """Stop execution and persist an explicit cancellation when applicable."""
    if not execution_task.done():
        execution_task.cancel()
    await asyncio.gather(execution_task, return_exceptions=True)

    if run_id not in explicit_run_cancellations:
        return

    if claim.claim_id is None:
        logger.info("Explicit cancel before this worker claimed the run, nothing to finalize", run_id=run_id)
        return

    try:
        identity = await _get_run_identity(run_id)
        if identity is None:
            return

        thread_id, user_id = identity
        await finalize_run(
            run_id,
            thread_id,
            user_id=user_id,
            status="interrupted",
            thread_status="idle",
            output={},
            claim_id=claim.claim_id,
        )
        await _release_lease(run_id, claim.claim_id)
    except Exception:
        logger.exception("Failed to persist explicit run cancellation", run_id=run_id)


async def _await_cancellation_cleanup(cleanup_task: asyncio.Task[None]) -> None:
    """Let cleanup finish even if the owning task is cancelled repeatedly."""
    while not cleanup_task.done():
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError:
            if cleanup_task.cancelled():
                raise
    await cleanup_task


class WorkerExecutor(BaseExecutor):
    """Dispatches runs via Redis List; workers consume with BLPOP + semaphore."""

    def __init__(self) -> None:
        self._worker_tasks: list[asyncio.Task[None]] = []
        # Task -> run_id, mapped at creation: a task cancelled before it ever
        # runs has no active_runs entry, yet its run still needs the drain requeue.
        self._job_tasks: dict[asyncio.Task[None], str] = {}
        self._job_claims: dict[asyncio.Task[None], _ClaimSlot] = {}
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
        drained: list[str] = []
        drained_claims: list[str] = []
        if self._job_tasks:
            logger.info("Draining in-flight jobs", count=len(self._job_tasks))
            _, pending = await asyncio.wait(set(self._job_tasks), timeout=drain_timeout)
            if pending:
                # Requeue drain survivors instead of finalizing work that a
                # plain crash would recover (#474).
                drained_tasks = [
                    task
                    for task, run_id in self._job_tasks.items()
                    if task in pending and run_id not in explicit_run_cancellations
                ]
                drained = [self._job_tasks[task] for task in drained_tasks]
                drained_claims = [
                    claim.claim_id
                    for task in drained_tasks
                    if (claim := self._job_claims.get(task)) is not None and claim.claim_id is not None
                ]
                _shutdown_cancellations.update(drained)
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)

        # Cancel worker loops before the requeue push: a loop still blocked in
        # BLPOP would steal the handed-off jobs back onto this dying instance.
        for task in self._worker_tasks:
            task.cancel()
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)

        if drained:
            await _requeue_drained_runs(drained, claim_ids=drained_claims)

        self._worker_tasks.clear()
        self._job_tasks.clear()
        self._job_claims.clear()
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

                if not self._running:
                    # Dequeued while shutdown was already underway: hand it back
                    # rather than start untracked work on a terminating instance.
                    await _push_back(run_id)
                    semaphore.release()
                    break

                claim = _ClaimSlot()
                task = asyncio.create_task(self._execute_and_release(run_id, worker_name, semaphore, claim))
                self._job_tasks[task] = run_id
                self._job_claims[task] = claim
                task.add_done_callback(self._forget_job_task)

            except asyncio.CancelledError:
                break
            except Exception:
                semaphore.release()
                logger.exception("Unexpected error in worker loop", worker=worker_name)
                await asyncio.sleep(1.0)

        logger.info("Worker stopped", worker=worker_name)

    def _forget_job_task(self, task: asyncio.Task[None]) -> None:
        self._job_tasks.pop(task, None)
        self._job_claims.pop(task, None)

    async def _execute_and_release(
        self,
        run_id: str,
        worker_name: str,
        semaphore: asyncio.Semaphore,
        claim: _ClaimSlot | None = None,
    ) -> None:
        """Execute a run with lease + timeout, then release the semaphore slot."""
        claim = claim if claim is not None else _ClaimSlot()
        # Register in active_runs so cancel-on-disconnect and explicit
        # cancel can find and cancel this specific job task.
        current_task = asyncio.current_task()
        if current_task is not None:
            active_runs[run_id] = current_task
        execution_task = asyncio.create_task(self._execute_with_lease(run_id, worker_name, claim))
        try:
            done, _ = await asyncio.wait(
                {execution_task},
                timeout=settings.worker.BG_JOB_TIMEOUT_SECS,
            )
            if execution_task in done:
                await execution_task
            else:
                logger.error(
                    "Job exceeded timeout, killing",
                    worker=worker_name,
                    run_id=run_id,
                    timeout_secs=settings.worker.BG_JOB_TIMEOUT_SECS,
                )
                _timeout_cancellations.add(run_id)
                execution_task.cancel()
                await asyncio.gather(execution_task, return_exceptions=True)

                await _finalize_timed_out_run(run_id, claim)
        except asyncio.CancelledError:
            logger.info("Job task cancelled", worker=worker_name, run_id=run_id)
            cleanup_task = asyncio.create_task(_cleanup_cancelled_execution(run_id, execution_task, claim))
            await _await_cancellation_cleanup(cleanup_task)
            raise
        except Exception:
            logger.exception("Unexpected error in job execution", run_id=run_id)
        finally:
            _timeout_cancellations.discard(run_id)
            _shutdown_cancellations.discard(run_id)
            # A newer attempt of this run may have registered here meanwhile; its entry and flag are not ours.
            registered = active_runs.get(run_id)
            if registered is None or registered is current_task:
                explicit_run_cancellations.discard(run_id)
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
            return result[1]
        except RedisTimeoutError:
            # Idle expiry: a blocking BLPOP hit the socket timeout with no jobs.
            # Normal when the queue is empty, not a connectivity failure — re-loop.
            return None
        except RedisError as exc:
            logger.warning("Redis BLPOP failed, falling back to Postgres poll", error=str(exc))
            await asyncio.sleep(settings.worker.POSTGRES_POLL_INTERVAL_SECONDS)
            return await self._poll_postgres()

    async def _execute_with_lease(self, run_id: str, worker_name: str, claim: _ClaimSlot) -> None:
        """Acquire lease, load job from DB, execute with heartbeat."""
        lease_acquired_at = datetime.now(UTC)
        # Taken before the claim is sent, so the heartbeat's view of the lease
        # never outlives the deadline Postgres wrote.
        lease_requested_at = asyncio.get_running_loop().time()
        # Published before the claim is sent: a cancel or drain landing while the
        # acquisition is in flight must still fence on it. An unused id matches nothing.
        claim.claim_id = _new_claim_id(worker_name)
        loaded = await _acquire_and_load(run_id, worker_name, claim_id=claim.claim_id)
        if loaded is None:
            claim.claim_id = None
            logger.debug("Lease not acquired or job missing, skipping", run_id=run_id, worker=worker_name)
            return
        if asyncio.get_running_loop().time() >= lease_requested_at + settings.worker.LEASE_DURATION_SECONDS:
            # The reaper may already be re-dispatching the run; it recovers this row once the lease lapses.
            logger.warning("Lease ran out while it was being acquired, not starting the run", run_id=run_id)
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
        job_task = asyncio.create_task(execute_run(loaded.job, claim_id=loaded.claim_id))
        heartbeat_task = asyncio.create_task(
            _heartbeat_loop(run_id, loaded.claim_id, job_task=job_task, leased_at=lease_requested_at),
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
            await _release_lease(run_id, loaded.claim_id)

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


async def _get_run_identity(run_id: str) -> tuple[str, str] | None:
    """Look up the thread and tenant identity for a run."""
    maker = _get_session_maker()
    async with maker() as session:
        result = await session.execute(select(RunORM.thread_id, RunORM.user_id).where(RunORM.run_id == run_id))
        row = result.one_or_none()
        if row is None:
            return None
        return row.thread_id, row.user_id


async def _finalize_timed_out_run(run_id: str, claim: _ClaimSlot) -> None:
    """Backstop for a job killed at BG_JOB_TIMEOUT_SECS before execute_run could finalize it.

    Without a held claim there is nothing to fence the write on, so the reaper owns the row.
    """
    if claim.claim_id is None:
        return
    identity = await _get_run_identity(run_id)
    if identity is not None:
        thread_id, user_id = identity
        await finalize_run(
            run_id,
            thread_id,
            user_id=user_id,
            status="error",
            thread_status="error",
            error="Job exceeded maximum execution time",
            claim_id=claim.claim_id,
        )
    await _release_lease(run_id, claim.claim_id)


# ------------------------------------------------------------------
# Lease operations (module-level for reuse by LeaseReaper)
# ------------------------------------------------------------------


class _LoadedRun:
    """RunJob, the claim id it was acquired under, and raw trace metadata."""

    __slots__ = ("claim_id", "job", "trace")

    def __init__(self, job: RunJob, trace: dict[str, str], claim_id: str) -> None:
        self.job = job
        self.trace = trace
        self.claim_id = claim_id


def _new_claim_id(worker_name: str) -> str:
    """Unique per acquisition; the worker-name prefix keeps ``claimed_by`` readable in logs and SQL."""
    return f"{worker_name}:{uuid.uuid4().hex}"


async def _acquire_and_load(run_id: str, worker_name: str, *, claim_id: str | None = None) -> _LoadedRun | None:
    """Acquire lease and load job in a single DB session.

    Combines the lease UPDATE + job SELECT into one session. ``claimed_by`` is
    a fresh claim id rather than the reusable worker name, so a reaped attempt
    stops matching its own ownership predicates once anyone re-acquires the run,
    including the same worker (#502). If the row is missing execution_params
    (data corruption / pre-migration row), releases the claim and errors the run.
    """
    claim_id = claim_id if claim_id is not None else _new_claim_id(worker_name)
    maker = _get_session_maker()
    async with maker() as session:
        result = await session.execute(
            update(RunORM)
            .where(
                RunORM.run_id == run_id,
                RunORM.status == "pending",
                RunORM.claimed_by.is_(None),
            )
            .values(claimed_by=claim_id, lease_expires_at=_lease_deadline(), status="running")
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
                .where(RunORM.run_id == run_id, RunORM.claimed_by == claim_id)
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
        return _LoadedRun(job=job, trace=trace, claim_id=claim_id)


async def _push_back(run_id: str) -> None:
    """Best-effort return of a dequeued-but-unstarted run to the queue."""
    try:
        client = redis_manager.get_client()
        await client.rpush(settings.worker.WORKER_QUEUE_KEY, run_id)  # type: ignore[arg-type]
    except RedisError:
        # Row is still pending/unclaimed; the stuck-pending reaper recovers it.
        logger.warning("Could not push back dequeued run at shutdown", run_id=run_id)


async def _requeue_drained_runs(run_ids: list[str], *, claim_ids: Collection[str] = ()) -> None:
    """Hand runs cancelled at the drain deadline back to the queue.

    Rows still 'running' under one of this instance's ``claim_ids`` were
    mid-execution with finalize skipped; rows still 'pending' and unclaimed were
    dequeued but never claimed. A running row under any other claim belongs to
    an attempt that replaced ours and must be left alone.
    """
    maker = _get_session_maker()
    async with maker() as session:
        result = await session.execute(
            update(RunORM)
            .where(
                RunORM.run_id.in_(run_ids),
                or_(
                    and_(RunORM.status == "running", RunORM.claimed_by.in_(list(claim_ids))),
                    and_(RunORM.status == "pending", RunORM.claimed_by.is_(None)),
                ),
            )
            .values(status="pending", claimed_by=None, lease_expires_at=None)
            .returning(RunORM.run_id)
        )
        reset_ids = [row[0] for row in result.fetchall()]
        await session.commit()

    if not reset_ids:
        return

    logger.info("Requeueing drained runs for another instance", count=len(reset_ids), run_ids=reset_ids)
    pushed = 0
    try:
        client = redis_manager.get_client()
        for run_id in reset_ids:
            await client.rpush(settings.worker.WORKER_QUEUE_KEY, run_id)  # type: ignore[arg-type]
            pushed += 1
    except RedisError:
        # Rows are already pending + unclaimed: the stuck-pending reaper or the
        # Postgres poll fallback on a surviving instance picks them up.
        logger.warning(
            "Redis unavailable during drain requeue, reaper will recover",
            run_ids=reset_ids[pushed:],
        )


async def _release_lease(run_id: str, claim_id: str) -> None:
    """Clear this attempt's lease once its run is terminal.

    Matches the claim id, not the worker name, so a reaped attempt cannot strip
    its replacement's lease. Running rows are never touched: without an owner or
    an expiry no reaper query could find them again.
    """
    maker = _get_session_maker()
    async with maker() as session:
        await session.execute(
            update(RunORM)
            .where(
                RunORM.run_id == run_id,
                RunORM.claimed_by == claim_id,
                RunORM.status.not_in(ACTIVE_RUN_STATES),
            )
            .values(claimed_by=None, lease_expires_at=None)
        )
        await session.commit()


async def _renew_lease(run_id: str, claim_id: str) -> int:
    """Extend one attempt's lease; returns the number of rows still held under ``claim_id``."""
    maker = _get_session_maker()
    async with maker() as session:
        result = await session.execute(
            update(RunORM)
            .where(RunORM.run_id == run_id, RunORM.claimed_by == claim_id)
            .values(lease_expires_at=_lease_deadline())
        )
        await session.commit()
    return result.rowcount  # type: ignore[union-attr]


def _abandon_run(run_id: str, job_task: asyncio.Task[None] | None) -> None:
    """Stop executing without finalizing: the run is no longer ours to write."""
    if job_task is None or job_task.done():
        return
    _lease_loss_cancellations.add(run_id)
    job_task.cancel()


def _discard_renewal(renewal: asyncio.Task[int]) -> None:
    """Mark the renewal's outcome retrieved so asyncio does not report it as lost."""
    if not renewal.cancelled():
        renewal.exception()


async def _heartbeat_loop(
    run_id: str,
    claim_id: str,
    *,
    job_task: asyncio.Task[None] | None = None,
    leased_at: float | None = None,
) -> None:
    """Extend the lease periodically while the job is running.

    Cancels ``job_task`` when the run has been handed to another attempt, and
    also when renewal cannot be confirmed before the lease runs out: past that
    point the reaper may already be dispatching a replacement (#502).
    """
    interval = settings.worker.HEARTBEAT_INTERVAL_SECONDS
    duration = settings.worker.LEASE_DURATION_SECONDS
    loop = asyncio.get_running_loop()
    # Monotonic lower bound on the lease Postgres holds: every deadline is
    # measured from before the statement that set it was sent.
    expires_at = (leased_at if leased_at is not None else loop.time()) + duration

    while True:
        # Wake while a full interval of lease is still left for the renewal to land in.
        await asyncio.sleep(min(interval, max(0.0, expires_at - loop.time() - interval)))
        sent_at = loop.time()
        budget = expires_at - sent_at
        if budget <= 0:
            logger.warning("Lease ran out before the heartbeat woke, abandoning run", run_id=run_id, claim=claim_id)
            _abandon_run(run_id, job_task)
            return

        # A wait, not wait_for: cancelling a stuck renewal must not delay the
        # abandon by however long the driver takes to unwind it.
        renewal = asyncio.create_task(_renew_lease(run_id, claim_id))
        try:
            await asyncio.wait({renewal}, timeout=budget)
        finally:
            if not renewal.done():
                renewal.cancel()
                renewal.add_done_callback(_discard_renewal)
            else:
                _discard_renewal(renewal)

        if not renewal.done() or renewal.cancelled():
            logger.warning("Lease renewal outlived the lease, abandoning run", run_id=run_id, claim=claim_id)
            _abandon_run(run_id, job_task)
            return

        error = renewal.exception()
        if error is not None:
            if loop.time() + interval < expires_at:
                logger.warning(
                    "Heartbeat lease extension failed, retrying", run_id=run_id, claim=claim_id, error=str(error)
                )
                continue
            logger.warning(
                "Heartbeat cannot renew before the lease expires, abandoning run",
                run_id=run_id,
                claim=claim_id,
                error=str(error),
            )
            _abandon_run(run_id, job_task)
            return

        if renewal.result() == 0:
            logger.warning("Lease lost, cancelling job to prevent double execution", run_id=run_id, claim=claim_id)
            _abandon_run(run_id, job_task)
            return

        expires_at = sent_at + duration
        logger.debug("Lease extended", run_id=run_id, claim=claim_id)


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

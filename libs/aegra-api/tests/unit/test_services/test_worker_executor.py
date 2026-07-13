"""Unit tests for worker_executor service."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis import ConnectionError as RedisConnectionError
from redis import TimeoutError as RedisTimeoutError

from aegra_api.core.active_runs import active_runs
from aegra_api.models.auth import User
from aegra_api.models.run_job import RunBehavior, RunExecution, RunIdentity, RunJob
from aegra_api.services.worker_executor import (
    WorkerExecutor,
    _acquire_and_load,
    _heartbeat_loop,
    _is_run_terminal,
    _is_valid_run_id,
    _LoadedRun,
    _release_lease,
    _restore_trace_context,
)

MODULE = "aegra_api.services.worker_executor"


def _make_session_maker(session: AsyncMock) -> MagicMock:
    """Wrap a mock session in a context-manager-returning maker."""
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    maker = MagicMock(return_value=ctx)
    return maker


def _make_run_job(
    *,
    run_id: str = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    thread_id: str = "11111111-2222-3333-4444-555555555555",
    graph_id: str = "test-graph",
) -> RunJob:
    """Create a minimal RunJob for testing."""
    return RunJob(
        identity=RunIdentity(run_id=run_id, thread_id=thread_id, graph_id=graph_id),
        user=User(identity="test-user"),
        execution=RunExecution(),
        behavior=RunBehavior(),
    )


def _make_run_orm(
    *,
    run_id: str = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    thread_id: str = "11111111-2222-3333-4444-555555555555",
    status: str = "pending",
    execution_params: dict | None = None,
) -> MagicMock:
    """Create a mock RunORM row."""
    orm = MagicMock()
    orm.run_id = run_id
    orm.thread_id = thread_id
    orm.status = status
    orm.assistant_id = "test-assistant"
    orm.execution_params = execution_params or {
        "graph_id": "test-graph",
        "user": {"identity": "test-user", "is_authenticated": True, "permissions": []},
        "execution": {
            "input_data": {},
            "config": {},
            "context": {},
            "stream_mode": None,
            "checkpoint": None,
            "command": None,
        },
        "behavior": {
            "interrupt_before": None,
            "interrupt_after": None,
            "multitask_strategy": None,
            "subgraphs": False,
        },
        "trace": {"correlation_id": "req-123"},
    }
    return orm


# ------------------------------------------------------------------
# _is_valid_run_id
# ------------------------------------------------------------------


class TestIsValidRunId:
    def test_returns_true_for_valid_uuid(self) -> None:
        assert _is_valid_run_id("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee") is True

    def test_returns_false_for_empty_string(self) -> None:
        assert _is_valid_run_id("") is False

    def test_returns_false_for_non_uuid_string(self) -> None:
        assert _is_valid_run_id("not-a-uuid") is False

    def test_returns_false_for_uuid_with_wrong_format(self) -> None:
        # Too short in last segment
        assert _is_valid_run_id("aaaaaaaa-bbbb-cccc-dddd-eeeeeeee") is False
        # Uppercase (pattern is lowercase hex only)
        assert _is_valid_run_id("AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE") is False


# ------------------------------------------------------------------
# _acquire_and_load
# ------------------------------------------------------------------


class TestAcquireAndLoad:
    @pytest.mark.asyncio
    async def test_returns_loaded_run_when_lease_acquired(self) -> None:
        run_orm = _make_run_orm()
        session = AsyncMock()

        # First execute: UPDATE (lease acquisition)
        update_result = MagicMock()
        update_result.rowcount = 1
        # Second call: scalar (SELECT run)
        session.execute = AsyncMock(return_value=update_result)
        session.scalar = AsyncMock(return_value=run_orm)
        session.commit = AsyncMock()
        maker = _make_session_maker(session)

        with patch(f"{MODULE}._get_session_maker", return_value=maker):
            result = await _acquire_and_load("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", "worker-0")

        assert result is not None
        assert isinstance(result, _LoadedRun)
        assert result.job.identity.run_id == "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        assert result.trace == {"correlation_id": "req-123"}
        session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_returns_none_when_lease_already_taken(self) -> None:
        session = AsyncMock()
        update_result = MagicMock()
        update_result.rowcount = 0
        session.execute = AsyncMock(return_value=update_result)
        session.rollback = AsyncMock()
        maker = _make_session_maker(session)

        with patch(f"{MODULE}._get_session_maker", return_value=maker):
            result = await _acquire_and_load("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", "worker-0")

        assert result is None
        session.rollback.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_returns_none_when_execution_params_is_none(self) -> None:
        run_orm = _make_run_orm()
        run_orm.execution_params = None

        session = AsyncMock()
        update_result = MagicMock()
        update_result.rowcount = 1
        session.execute = AsyncMock(return_value=update_result)
        session.scalar = AsyncMock(return_value=run_orm)
        session.commit = AsyncMock()
        maker = _make_session_maker(session)

        with patch(f"{MODULE}._get_session_maker", return_value=maker):
            result = await _acquire_and_load("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", "worker-0")

        assert result is None


# ------------------------------------------------------------------
# _release_lease
# ------------------------------------------------------------------


class TestReleaseLease:
    @pytest.mark.asyncio
    async def test_clears_claimed_by_and_lease_expires_at(self) -> None:
        session = AsyncMock()
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        maker = _make_session_maker(session)

        with patch(f"{MODULE}._get_session_maker", return_value=maker):
            await _release_lease("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", "test-worker")

        session.execute.assert_awaited_once()
        session.commit.assert_awaited_once()


# ------------------------------------------------------------------
# _heartbeat_loop
# ------------------------------------------------------------------


class TestHeartbeatLoop:
    @pytest.mark.asyncio
    async def test_extends_lease_on_each_iteration(self) -> None:
        session = AsyncMock()
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        maker = _make_session_maker(session)

        call_count = 0

        async def counting_sleep(delay: float) -> None:
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise asyncio.CancelledError
            # Don't actually sleep

        with (
            patch(f"{MODULE}._get_session_maker", return_value=maker),
            patch(f"{MODULE}.settings") as mock_settings,
            patch(f"{MODULE}.asyncio.sleep", side_effect=counting_sleep),
        ):
            mock_settings.worker.HEARTBEAT_INTERVAL_SECONDS = 1
            mock_settings.worker.LEASE_DURATION_SECONDS = 30

            with pytest.raises(asyncio.CancelledError):
                await _heartbeat_loop("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", "worker-0")

        # One iteration completed before cancellation on second sleep
        assert session.execute.await_count == 1
        assert session.commit.await_count == 1

    @pytest.mark.asyncio
    async def test_continues_loop_on_db_error(self) -> None:
        session = AsyncMock()
        session.execute = AsyncMock(side_effect=Exception("DB connection lost"))
        maker = _make_session_maker(session)

        call_count = 0

        async def counting_sleep(delay: float) -> None:
            nonlocal call_count
            call_count += 1
            if call_count >= 3:
                raise asyncio.CancelledError

        with (
            patch(f"{MODULE}._get_session_maker", return_value=maker),
            patch(f"{MODULE}.settings") as mock_settings,
            patch(f"{MODULE}.asyncio.sleep", side_effect=counting_sleep),
        ):
            mock_settings.worker.HEARTBEAT_INTERVAL_SECONDS = 1
            mock_settings.worker.LEASE_DURATION_SECONDS = 30

            with pytest.raises(asyncio.CancelledError):
                await _heartbeat_loop("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", "worker-0")

        # Loop continued despite DB errors (2 iterations before cancel on 3rd sleep)
        assert session.execute.await_count == 2


# ------------------------------------------------------------------
# _is_run_terminal
# ------------------------------------------------------------------


class TestIsRunTerminal:
    @pytest.mark.asyncio
    async def test_returns_true_for_success(self) -> None:
        run_orm = MagicMock()
        run_orm.status = "success"
        session = AsyncMock()
        session.scalar = AsyncMock(return_value=run_orm)
        maker = _make_session_maker(session)

        with patch(f"{MODULE}._get_session_maker", return_value=maker):
            result = await _is_run_terminal("run-1")

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_true_for_error(self) -> None:
        run_orm = MagicMock()
        run_orm.status = "error"
        session = AsyncMock()
        session.scalar = AsyncMock(return_value=run_orm)
        maker = _make_session_maker(session)

        with patch(f"{MODULE}._get_session_maker", return_value=maker):
            result = await _is_run_terminal("run-1")

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_true_for_interrupted(self) -> None:
        run_orm = MagicMock()
        run_orm.status = "interrupted"
        session = AsyncMock()
        session.scalar = AsyncMock(return_value=run_orm)
        maker = _make_session_maker(session)

        with patch(f"{MODULE}._get_session_maker", return_value=maker):
            result = await _is_run_terminal("run-1")

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_true_when_run_not_found(self) -> None:
        session = AsyncMock()
        session.scalar = AsyncMock(return_value=None)
        maker = _make_session_maker(session)

        with patch(f"{MODULE}._get_session_maker", return_value=maker):
            result = await _is_run_terminal("run-1")

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_for_pending(self) -> None:
        run_orm = MagicMock()
        run_orm.status = "pending"
        session = AsyncMock()
        session.scalar = AsyncMock(return_value=run_orm)
        maker = _make_session_maker(session)

        with patch(f"{MODULE}._get_session_maker", return_value=maker):
            result = await _is_run_terminal("run-1")

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_for_running(self) -> None:
        run_orm = MagicMock()
        run_orm.status = "running"
        session = AsyncMock()
        session.scalar = AsyncMock(return_value=run_orm)
        maker = _make_session_maker(session)

        with patch(f"{MODULE}._get_session_maker", return_value=maker):
            result = await _is_run_terminal("run-1")

        assert result is False


# ------------------------------------------------------------------
# _restore_trace_context
# ------------------------------------------------------------------


class TestRestoreTraceContext:
    def test_sets_structlog_context_vars(self) -> None:
        job = _make_run_job()
        trace = {"correlation_id": "req-abc"}

        with patch(f"{MODULE}.set_trace_context") as mock_set_trace:
            _restore_trace_context("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", job, trace)

        mock_set_trace.assert_called_once()
        call_kwargs = mock_set_trace.call_args.kwargs
        assert call_kwargs["user_id"] == "test-user"
        assert call_kwargs["session_id"] == "11111111-2222-3333-4444-555555555555"
        assert call_kwargs["trace_name"] == "test-graph"

    def test_clears_previous_context_before_setting_new(self) -> None:
        job = _make_run_job()
        trace = {"correlation_id": "req-abc"}
        call_order: list[str] = []

        with (
            patch(f"{MODULE}.structlog.contextvars.clear_contextvars", side_effect=lambda: call_order.append("clear")),
            patch(f"{MODULE}.set_trace_context", side_effect=lambda **kw: call_order.append("set_trace")),
            patch(
                f"{MODULE}.structlog.contextvars.bind_contextvars", side_effect=lambda **kw: call_order.append("bind")
            ),
            patch(f"{MODULE}.correlation_id"),
        ):
            _restore_trace_context("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", job, trace)

        assert call_order == ["clear", "set_trace", "bind"]

    def test_user_metadata_merged_with_system_keys(self) -> None:
        """job.run_metadata is merged into the trace context metadata."""
        job = RunJob(
            identity=RunIdentity(
                run_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                thread_id="11111111-2222-3333-4444-555555555555",
                graph_id="test-graph",
            ),
            user=User(identity="test-user"),
            run_metadata={"tenant": "acme", "feature_flag": True},
        )
        trace = {"correlation_id": "req-abc"}

        with patch(f"{MODULE}.set_trace_context") as mock_set_trace:
            _restore_trace_context("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", job, trace)

        metadata = mock_set_trace.call_args.kwargs["metadata"]
        assert metadata["tenant"] == "acme"
        assert metadata["feature_flag"] is True
        # System keys still present
        assert metadata["run_id"] == "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        assert metadata["thread_id"] == "11111111-2222-3333-4444-555555555555"
        assert metadata["graph_id"] == "test-graph"
        assert metadata["original_request_id"] == "req-abc"

    def test_user_metadata_cannot_override_system_keys(self) -> None:
        """Reserved system keys win on collision; user spoof is dropped."""
        job = RunJob(
            identity=RunIdentity(
                run_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                thread_id="11111111-2222-3333-4444-555555555555",
                graph_id="test-graph",
            ),
            user=User(identity="test-user"),
            run_metadata={"run_id": "spoofed", "tenant": "acme"},
        )
        trace = {"correlation_id": "req-abc"}

        with patch(f"{MODULE}.set_trace_context") as mock_set_trace:
            _restore_trace_context("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", job, trace)

        metadata = mock_set_trace.call_args.kwargs["metadata"]
        assert metadata["run_id"] == "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        assert metadata["tenant"] == "acme"

    def test_empty_run_metadata_with_correlation_id_keeps_four_system_keys(self) -> None:
        """When a correlation-id is present, ``original_request_id`` is
        included in the metadata alongside the three runtime keys."""
        job = _make_run_job()  # run_metadata defaults to {}
        trace = {"correlation_id": "req-abc"}

        with patch(f"{MODULE}.set_trace_context") as mock_set_trace:
            _restore_trace_context("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", job, trace)

        metadata = mock_set_trace.call_args.kwargs["metadata"]
        assert set(metadata.keys()) == {"run_id", "thread_id", "graph_id", "original_request_id"}

    def test_missing_correlation_id_omits_original_request_id(self) -> None:
        """Requests without an upstream correlation-id header should not produce
        a ``langfuse.trace.metadata.original_request_id=""`` empty attribute."""
        job = _make_run_job()
        trace: dict[str, str] = {}  # no correlation_id

        with patch(f"{MODULE}.set_trace_context") as mock_set_trace:
            _restore_trace_context("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", job, trace)

        metadata = mock_set_trace.call_args.kwargs["metadata"]
        assert "original_request_id" not in metadata
        assert set(metadata.keys()) == {"run_id", "thread_id", "graph_id"}


# ------------------------------------------------------------------
# WorkerExecutor.submit
# ------------------------------------------------------------------


class TestWorkerExecutorSubmit:
    @pytest.mark.asyncio
    async def test_pushes_run_id_to_redis(self) -> None:
        mock_client = AsyncMock()
        mock_client.rpush = AsyncMock()

        job = _make_run_job()

        with (
            patch(f"{MODULE}.redis_manager") as mock_redis,
            patch(f"{MODULE}.settings") as mock_settings,
        ):
            mock_redis.get_client.return_value = mock_client
            mock_settings.worker.WORKER_QUEUE_KEY = "aegra:jobs"

            executor = WorkerExecutor()
            await executor.submit(job)

        mock_client.rpush.assert_awaited_once_with("aegra:jobs", "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")


# ------------------------------------------------------------------
# WorkerExecutor.wait_for_completion
# ------------------------------------------------------------------


class TestWorkerExecutorWaitForCompletion:
    @pytest.mark.asyncio
    async def test_done_key_uses_configured_channel_prefix(self) -> None:
        """Regression: done-key must derive from REDIS_CHANNEL_PREFIX, not a hardcoded string."""
        mock_client = AsyncMock()
        mock_client.exists = AsyncMock(return_value=True)

        with (
            patch(f"{MODULE}.redis_manager") as mock_redis,
            patch(f"{MODULE}.settings") as mock_settings,
        ):
            mock_redis.get_client.return_value = mock_client
            mock_settings.redis.REDIS_CHANNEL_PREFIX = "aegra:agent-foo:run:"

            executor = WorkerExecutor()
            await executor.wait_for_completion("run-1")

        mock_client.exists.assert_awaited_once_with("aegra:agent-foo:run:done:run-1")


# ------------------------------------------------------------------
# WorkerExecutor.start / stop
# ------------------------------------------------------------------


class TestWorkerExecutorStart:
    @pytest.mark.asyncio
    async def test_creates_worker_tasks(self) -> None:
        with patch(f"{MODULE}.settings") as mock_settings:
            mock_settings.worker.WORKER_COUNT = 2
            mock_settings.worker.N_JOBS_PER_WORKER = 5

            executor = WorkerExecutor()
            # Patch _worker_loop to be a no-op coroutine
            executor._worker_loop = AsyncMock()  # type: ignore[method-assign]
            await executor.start()

        assert len(executor._worker_tasks) == 2
        # Clean up tasks
        for t in executor._worker_tasks:
            t.cancel()
        await asyncio.gather(*executor._worker_tasks, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_warns_when_worker_count_zero(self) -> None:
        with (
            patch(f"{MODULE}.settings") as mock_settings,
            patch(f"{MODULE}.logger") as mock_logger,
        ):
            mock_settings.worker.WORKER_COUNT = 0
            mock_settings.worker.N_JOBS_PER_WORKER = 5

            executor = WorkerExecutor()
            await executor.start()

        mock_logger.warning.assert_called_once()
        assert "WORKER_COUNT=0" in mock_logger.warning.call_args[0][0]
        assert len(executor._worker_tasks) == 0


class TestWorkerExecutorStop:
    @pytest.mark.asyncio
    async def test_cancels_worker_tasks(self) -> None:
        with patch(f"{MODULE}.settings") as mock_settings:
            mock_settings.worker.WORKER_DRAIN_TIMEOUT = 1.0

            executor = WorkerExecutor()

            # Create some fake tasks
            async def hang_forever() -> None:
                await asyncio.sleep(9999)

            task1 = asyncio.create_task(hang_forever())
            task2 = asyncio.create_task(hang_forever())
            executor._worker_tasks = [task1, task2]

            await executor.stop()

        assert task1.cancelled()
        assert task2.cancelled()
        assert len(executor._worker_tasks) == 0


# ------------------------------------------------------------------
# _execute_and_release
# ------------------------------------------------------------------


class TestExecuteAndRelease:
    @pytest.mark.asyncio
    async def test_registers_in_active_runs_and_cleans_up(self) -> None:
        run_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        semaphore = asyncio.Semaphore(1)
        await semaphore.acquire()  # Pre-acquire so we can verify release

        executor = WorkerExecutor()

        registered_in_active: bool = False

        async def mock_execute_with_lease(rid: str, wn: str) -> None:
            nonlocal registered_in_active
            registered_in_active = run_id in active_runs

        executor._execute_with_lease = AsyncMock(side_effect=mock_execute_with_lease)  # type: ignore[method-assign]

        with patch(f"{MODULE}.settings") as mock_settings:
            mock_settings.worker.BG_JOB_TIMEOUT_SECS = 60

            await executor._execute_and_release(run_id, "worker-0", semaphore)

        # Task was registered during execution
        assert registered_in_active is True
        # Cleaned up after execution
        assert run_id not in active_runs
        # Semaphore was released
        assert not semaphore.locked()

    @pytest.mark.asyncio
    async def test_handles_timeout_error(self) -> None:
        run_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        semaphore = asyncio.Semaphore(1)
        await semaphore.acquire()

        executor = WorkerExecutor()

        async def slow_execute(rid: str, wn: str) -> None:
            await asyncio.sleep(9999)

        executor._execute_with_lease = AsyncMock(side_effect=slow_execute)  # type: ignore[method-assign]

        thread_id = "tttttttt-tttt-tttt-tttt-tttttttttttt"

        with (
            patch(f"{MODULE}.settings") as mock_settings,
            patch(f"{MODULE}._get_thread_id_for_run", new_callable=AsyncMock, return_value=thread_id),
            patch(f"{MODULE}.finalize_run", new_callable=AsyncMock) as mock_finalize,
            patch(f"{MODULE}._release_lease") as mock_release,
        ):
            mock_settings.worker.BG_JOB_TIMEOUT_SECS = 0.01  # Very short timeout
            mock_release.return_value = None

            await executor._execute_and_release(run_id, "worker-0", semaphore)

        mock_finalize.assert_awaited_once_with(
            run_id,
            thread_id,
            status="error",
            thread_status="error",
            error="Job exceeded maximum execution time",
        )
        mock_release.assert_awaited_once_with(run_id, "worker-0")
        # Semaphore released even on timeout
        assert not semaphore.locked()
        # Cleaned up
        assert run_id not in active_runs


class TestExecuteWithLease:
    @pytest.mark.asyncio
    async def test_cancels_job_task_in_finally(self) -> None:
        """Regression: when _execute_with_lease is cancelled (e.g. by wait_for
        timeout), the inner job_task must also be cancelled to prevent orphaned
        execution that corrupts run state."""
        run_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        executor = WorkerExecutor()

        job_task_was_cancelled = False

        async def long_running_job(job: object) -> None:
            nonlocal job_task_was_cancelled
            try:
                await asyncio.sleep(9999)
            except asyncio.CancelledError:
                job_task_was_cancelled = True
                raise

        mock_loaded = MagicMock(spec=_LoadedRun)
        mock_loaded.job = _make_run_job()
        mock_loaded.trace = {}

        with (
            patch(f"{MODULE}._acquire_and_load", new_callable=AsyncMock, return_value=mock_loaded),
            patch(f"{MODULE}._restore_trace_context"),
            patch(f"{MODULE}.execute_run", side_effect=long_running_job),
            patch(f"{MODULE}._heartbeat_loop", new_callable=AsyncMock),
            patch(f"{MODULE}._release_lease", new_callable=AsyncMock),
        ):
            # Run _execute_with_lease in a task and cancel it (simulating wait_for timeout).
            # The CancelledError is caught internally by _execute_with_lease's
            # except block, so the task completes normally — but the inner
            # job_task must still have been cancelled.
            task = asyncio.create_task(executor._execute_with_lease(run_id, "worker-0"))
            await asyncio.sleep(0.05)  # Let it start
            task.cancel()
            await task  # Completes normally (CancelledError is handled internally)

        assert job_task_was_cancelled, "job_task must be cancelled when _execute_with_lease is cancelled"


class TestDequeue:
    """Tests for WorkerExecutor._dequeue BLPOP handling."""

    def _make_executor_with_blpop(self, blpop: AsyncMock) -> WorkerExecutor:
        executor = WorkerExecutor()
        executor._poll_postgres = AsyncMock(return_value="from-postgres")  # type: ignore[method-assign]
        self._client = MagicMock()
        self._client.blpop = blpop
        return executor

    @pytest.mark.asyncio
    async def test_returns_run_id_on_queue_hit(self) -> None:
        blpop = AsyncMock(return_value=("aegra:worker:queue", "run-123"))
        executor = self._make_executor_with_blpop(blpop)

        with patch(f"{MODULE}.redis_manager.get_client", return_value=self._client):
            result = await executor._dequeue()

        assert result == "run-123"
        executor._poll_postgres.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_returns_none_when_blpop_returns_none(self) -> None:
        blpop = AsyncMock(return_value=None)
        executor = self._make_executor_with_blpop(blpop)

        with patch(f"{MODULE}.redis_manager.get_client", return_value=self._client):
            result = await executor._dequeue()

        assert result is None
        executor._poll_postgres.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_idle_socket_timeout_returns_none_without_fallback(self) -> None:
        """A blocking BLPOP that hits the socket timeout raises redis TimeoutError
        (a RedisError subclass). That is a normal idle expiry, not a connectivity
        failure: it must return None silently, never poll Postgres (GH #bug)."""
        blpop = AsyncMock(side_effect=RedisTimeoutError("Timeout reading from redis:6379"))
        executor = self._make_executor_with_blpop(blpop)

        with (
            patch(f"{MODULE}.redis_manager.get_client", return_value=self._client),
            patch(f"{MODULE}.logger.warning") as mock_warning,
        ):
            result = await executor._dequeue()

        assert result is None
        executor._poll_postgres.assert_not_awaited()
        mock_warning.assert_not_called()

    @pytest.mark.asyncio
    async def test_connection_error_falls_back_to_postgres(self) -> None:
        """A genuine Redis failure (connection lost) must still warn and fall
        back to the Postgres poll so jobs are not stranded."""
        blpop = AsyncMock(side_effect=RedisConnectionError("Connection refused"))
        executor = self._make_executor_with_blpop(blpop)

        with (
            patch(f"{MODULE}.redis_manager.get_client", return_value=self._client),
            patch(f"{MODULE}.asyncio.sleep", new_callable=AsyncMock),
            patch(f"{MODULE}.logger.warning") as mock_warning,
        ):
            result = await executor._dequeue()

        assert result == "from-postgres"
        executor._poll_postgres.assert_awaited_once()
        mock_warning.assert_called_once()

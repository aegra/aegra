"""Unit tests for worker_executor service."""

import asyncio
from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis import ConnectionError as RedisConnectionError
from redis import TimeoutError as RedisTimeoutError

from aegra_api.core.active_runs import active_runs, explicit_run_cancellations
from aegra_api.models.auth import User
from aegra_api.models.run_job import RunBehavior, RunExecution, RunIdentity, RunJob
from aegra_api.services.run_executor import _lease_loss_cancellations, _shutdown_cancellations, _timeout_cancellations
from aegra_api.services.worker_executor import (
    WorkerExecutor,
    _acquire_and_load,
    _ClaimSlot,
    _heartbeat_loop,
    _is_run_terminal,
    _is_valid_run_id,
    _LoadedRun,
    _release_lease,
    _requeue_drained_runs,
    _restore_trace_context,
)

MODULE = "aegra_api.services.worker_executor"
CLAIM_ID = "worker-0:0123456789abcdef"
RUN_ID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


async def _claim_then_cancel(rid: str, wn: str, claim: _ClaimSlot) -> None:
    claim.claim_id = CLAIM_ID
    raise asyncio.CancelledError


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


class TestHeartbeatFencing:
    """The heartbeat must stop the graph before its lease can be handed to another attempt (#502)."""

    @pytest.fixture(autouse=True)
    def _clear_lease_loss(self: "TestHeartbeatFencing") -> Iterator[None]:
        _lease_loss_cancellations.clear()
        yield
        _lease_loss_cancellations.clear()

    @staticmethod
    async def _idle_job() -> None:
        await asyncio.Event().wait()

    async def _run_heartbeat(
        self,
        renew: object,
        *,
        interval: float,
        duration: float,
        leased_ago: float = 0.0,
    ) -> tuple[asyncio.Task[None], float]:
        job_task = asyncio.create_task(self._idle_job())
        loop = asyncio.get_running_loop()
        started = loop.time()
        with (
            patch(f"{MODULE}._renew_lease", side_effect=renew),
            patch(f"{MODULE}.settings") as mock_settings,
        ):
            mock_settings.worker.HEARTBEAT_INTERVAL_SECONDS = interval
            mock_settings.worker.LEASE_DURATION_SECONDS = duration
            await asyncio.wait_for(
                _heartbeat_loop(RUN_ID, CLAIM_ID, job_task=job_task, leased_at=started - leased_ago),
                timeout=5,
            )
        elapsed = loop.time() - started
        await asyncio.gather(job_task, return_exceptions=True)
        return job_task, elapsed

    @pytest.mark.asyncio
    async def test_abandons_by_the_deadline_even_when_the_renewal_will_not_unwind(self: "TestHeartbeatFencing") -> None:
        """Regression: wait_for waited for the cancelled renewal to unwind, so
        the graph kept running past the lease while the reaper re-dispatched it."""

        async def stuck_renewal(run_id: str, claim_id: str) -> int:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await asyncio.sleep(0.5)
                raise
            return 1

        job_task, elapsed = await self._run_heartbeat(stuck_renewal, interval=0.05, duration=0.3)
        await asyncio.sleep(0.6)  # let the abandoned renewal finish unwinding before the loop closes

        assert job_task.cancelled()
        assert RUN_ID in _lease_loss_cancellations
        assert elapsed < 0.6

    @pytest.mark.asyncio
    async def test_next_deadline_is_measured_from_before_the_renewal_was_sent(self: "TestHeartbeatFencing") -> None:
        calls = 0

        async def slow_then_stuck(run_id: str, claim_id: str) -> int:
            nonlocal calls
            calls += 1
            if calls == 1:
                await asyncio.sleep(0.3)
                return 1
            await asyncio.Event().wait()
            return 1

        job_task, elapsed = await self._run_heartbeat(slow_then_stuck, interval=0.1, duration=0.5)

        # The first renewal was sent at ~0.1s, so the lease it bought ends at ~0.6s.
        # Counting from its reply (~0.4s) would let the graph run until ~0.9s.
        assert job_task.cancelled()
        assert elapsed < 0.8

    @pytest.mark.asyncio
    async def test_abandons_when_the_claim_was_handed_to_another_attempt(self: "TestHeartbeatFencing") -> None:
        job_task, _ = await self._run_heartbeat(AsyncMock(return_value=0), interval=0.01, duration=1)

        assert job_task.cancelled()
        assert RUN_ID in _lease_loss_cancellations

    @pytest.mark.asyncio
    async def test_abandons_when_renewal_keeps_failing_until_the_lease_runs_out(self: "TestHeartbeatFencing") -> None:
        renew = AsyncMock(side_effect=OSError("database unreachable"))

        job_task, elapsed = await self._run_heartbeat(renew, interval=0.05, duration=0.3)

        assert job_task.cancelled()
        assert renew.await_count >= 2, "a failure with runway left is retried first"
        assert elapsed < 0.6

    @pytest.mark.asyncio
    async def test_keeps_running_while_renewals_succeed(self: "TestHeartbeatFencing") -> None:
        renew = AsyncMock(return_value=1)
        job_task = asyncio.create_task(self._idle_job())
        with (
            patch(f"{MODULE}._renew_lease", renew),
            patch(f"{MODULE}.settings") as mock_settings,
        ):
            mock_settings.worker.HEARTBEAT_INTERVAL_SECONDS = 0.02
            mock_settings.worker.LEASE_DURATION_SECONDS = 0.1
            heartbeat = asyncio.create_task(_heartbeat_loop(RUN_ID, CLAIM_ID, job_task=job_task))
            await asyncio.sleep(0.3)
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)

        assert not job_task.done()
        assert renew.await_count >= 5
        renew.assert_awaited_with(RUN_ID, CLAIM_ID)
        job_task.cancel()
        await asyncio.gather(job_task, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_abandons_by_the_deadline_when_acquisition_used_up_most_of_the_lease(
        self: "TestHeartbeatFencing",
    ) -> None:
        """Regression: the first wake-up came a full interval after the graph started,
        past a lease that a slow acquisition had nearly used up."""

        async def stuck_renewal(run_id: str, claim_id: str) -> int:
            await asyncio.Event().wait()
            return 1

        job_task, elapsed = await self._run_heartbeat(stuck_renewal, interval=1.0, duration=2.5, leased_ago=2.4)

        assert job_task.cancelled()
        assert elapsed < 0.6, "the lease ended ~0.1s in; waiting a 1s interval overran it"

    @pytest.mark.asyncio
    async def test_renews_at_once_when_less_than_an_interval_of_the_lease_is_left(
        self: "TestHeartbeatFencing",
    ) -> None:
        renew = AsyncMock(return_value=1)
        job_task = asyncio.create_task(self._idle_job())
        leased_at = asyncio.get_running_loop().time() - 2.3
        with (
            patch(f"{MODULE}._renew_lease", renew),
            patch(f"{MODULE}.settings") as mock_settings,
        ):
            mock_settings.worker.HEARTBEAT_INTERVAL_SECONDS = 1.0
            mock_settings.worker.LEASE_DURATION_SECONDS = 2.5
            heartbeat = asyncio.create_task(_heartbeat_loop(RUN_ID, CLAIM_ID, job_task=job_task, leased_at=leased_at))
            await asyncio.sleep(0.3)
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)

        renew.assert_awaited_with(RUN_ID, CLAIM_ID)
        assert not job_task.done()
        job_task.cancel()
        await asyncio.gather(job_task, return_exceptions=True)


class TestClaimIds:
    @pytest.mark.asyncio
    async def test_each_acquisition_writes_a_fresh_claim_id_instead_of_the_worker_name(self: "TestClaimIds") -> None:
        session = AsyncMock()
        update_result = MagicMock()
        update_result.rowcount = 1
        session.execute = AsyncMock(return_value=update_result)
        session.scalar = AsyncMock(return_value=_make_run_orm())
        maker = _make_session_maker(session)

        with patch(f"{MODULE}._get_session_maker", return_value=maker):
            first = await _acquire_and_load(RUN_ID, "worker-0")
            second = await _acquire_and_load(RUN_ID, "worker-0")

        assert first is not None and second is not None
        assert first.claim_id != second.claim_id
        assert first.claim_id.startswith("worker-0:")
        written = [call.args[0].compile().params for call in session.execute.await_args_list]
        assert written[0]["claimed_by"] == first.claim_id
        assert written[1]["claimed_by"] == second.claim_id

    @pytest.mark.asyncio
    async def test_release_only_clears_a_terminal_run_held_by_this_claim(self: "TestClaimIds") -> None:
        session = AsyncMock()
        maker = _make_session_maker(session)

        with patch(f"{MODULE}._get_session_maker", return_value=maker):
            await _release_lease(RUN_ID, CLAIM_ID)

        statement = session.execute.await_args.args[0]
        sql = str(statement.compile())
        assert "runs.claimed_by = " in sql
        assert "runs.status NOT IN" in sql
        assert CLAIM_ID in statement.compile().params.values()


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
    @pytest.fixture(autouse=True)
    def _clear_cancellation_state(self) -> Iterator[None]:
        active_runs.clear()
        explicit_run_cancellations.clear()
        _timeout_cancellations.clear()
        yield
        for task in active_runs.values():
            if not task.done():
                task.cancel()
        active_runs.clear()
        explicit_run_cancellations.clear()
        _timeout_cancellations.clear()

    @pytest.mark.asyncio
    async def test_registers_in_active_runs_and_cleans_up(self) -> None:
        run_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        semaphore = asyncio.Semaphore(1)
        await semaphore.acquire()  # Pre-acquire so we can verify release

        executor = WorkerExecutor()

        registered_in_active: bool = False

        async def mock_execute_with_lease(rid: str, wn: str, claim: _ClaimSlot) -> None:
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
        timeout_marker_seen = False

        async def slow_execute(rid: str, wn: str, claim: _ClaimSlot) -> None:
            nonlocal timeout_marker_seen
            claim.claim_id = CLAIM_ID
            try:
                await asyncio.sleep(9999)
            except asyncio.CancelledError:
                timeout_marker_seen = run_id in _timeout_cancellations
                raise

        executor._execute_with_lease = AsyncMock(side_effect=slow_execute)  # type: ignore[method-assign]

        thread_id = "tttttttt-tttt-tttt-tttt-tttttttttttt"

        with (
            patch(f"{MODULE}.settings") as mock_settings,
            patch(
                f"{MODULE}._get_run_identity",
                new_callable=AsyncMock,
                return_value=(thread_id, "user-1"),
            ),
            patch(f"{MODULE}.finalize_run", new_callable=AsyncMock) as mock_finalize,
            patch(f"{MODULE}._release_lease") as mock_release,
        ):
            mock_settings.worker.BG_JOB_TIMEOUT_SECS = 0.01  # Very short timeout
            mock_release.return_value = None

            await executor._execute_and_release(run_id, "worker-0", semaphore)

        mock_finalize.assert_awaited_once_with(
            run_id,
            thread_id,
            user_id="user-1",
            status="error",
            thread_status="error",
            error="Job exceeded maximum execution time",
            claim_id=CLAIM_ID,
        )
        mock_release.assert_awaited_once_with(run_id, CLAIM_ID)
        assert timeout_marker_seen is True
        # Semaphore released even on timeout
        assert not semaphore.locked()
        # Cleaned up
        assert run_id not in active_runs
        assert run_id not in _timeout_cancellations

    @pytest.mark.asyncio
    async def test_explicit_cancel_after_claim_reconciles_database(self: "TestExecuteAndRelease") -> None:
        run_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        thread_id = "tttttttt-tttt-tttt-tttt-tttttttttttt"
        semaphore = asyncio.Semaphore(1)
        await semaphore.acquire()
        executor = WorkerExecutor()
        executor._execute_with_lease = AsyncMock(side_effect=_claim_then_cancel)  # type: ignore[method-assign]
        explicit_run_cancellations.add(run_id)

        with (
            patch(f"{MODULE}.settings") as mock_settings,
            patch(
                f"{MODULE}._get_run_identity",
                new_callable=AsyncMock,
                return_value=(thread_id, "user-1"),
            ),
            patch(f"{MODULE}.finalize_run", new_callable=AsyncMock, return_value=True) as mock_finalize,
            patch(f"{MODULE}._release_lease", new_callable=AsyncMock) as mock_release,
        ):
            mock_settings.worker.BG_JOB_TIMEOUT_SECS = 60
            with pytest.raises(asyncio.CancelledError):
                await executor._execute_and_release(run_id, "worker-0", semaphore)

        mock_release.assert_awaited_once_with(run_id, CLAIM_ID)

        mock_finalize.assert_awaited_once_with(
            run_id,
            thread_id,
            user_id="user-1",
            status="interrupted",
            thread_status="idle",
            output={},
            claim_id=CLAIM_ID,
        )
        assert run_id not in explicit_run_cancellations
        assert run_id not in active_runs
        assert not semaphore.locked()

    @pytest.mark.asyncio
    async def test_explicit_cancel_before_execution_starts_leaves_finalization_to_the_api(
        self: "TestExecuteAndRelease",
    ) -> None:
        """Cancelled before a claim id was minted, so no claim can have committed;
        the API's unowned-run reconciliation owns that row."""
        run_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        semaphore = asyncio.Semaphore(1)
        await semaphore.acquire()
        executor = WorkerExecutor()
        executor._execute_with_lease = AsyncMock(side_effect=asyncio.CancelledError)  # type: ignore[method-assign]
        explicit_run_cancellations.add(run_id)

        with (
            patch(f"{MODULE}.settings") as mock_settings,
            patch(f"{MODULE}.finalize_run", new_callable=AsyncMock) as mock_finalize,
        ):
            mock_settings.worker.BG_JOB_TIMEOUT_SECS = 60
            with pytest.raises(asyncio.CancelledError):
                await executor._execute_and_release(run_id, "worker-0", semaphore)

        mock_finalize.assert_not_awaited()
        assert not semaphore.locked()

    @pytest.mark.asyncio
    async def test_explicit_cancel_during_acquisition_still_finalizes_under_the_claim(
        self: "TestExecuteAndRelease",
    ) -> None:
        """Regression: the claim committed before the worker learned its id, so a
        cancel landing in between skipped finalize and the reaper later re-ran the run."""
        run_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        thread_id = "tttttttt-tttt-tttt-tttt-tttttttttttt"
        semaphore = asyncio.Semaphore(1)
        await semaphore.acquire()
        executor = WorkerExecutor()
        acquiring = asyncio.Event()
        sent_claims: list[str | None] = []

        async def committed_then_stalled(rid: str, wn: str, *, claim_id: str | None = None) -> None:
            sent_claims.append(claim_id)
            acquiring.set()
            await asyncio.Event().wait()

        explicit_run_cancellations.add(run_id)
        with (
            patch(f"{MODULE}.settings") as mock_settings,
            patch(f"{MODULE}._acquire_and_load", side_effect=committed_then_stalled),
            patch(f"{MODULE}._get_run_identity", new_callable=AsyncMock, return_value=(thread_id, "user-1")),
            patch(f"{MODULE}.finalize_run", new_callable=AsyncMock, return_value=True) as mock_finalize,
            patch(f"{MODULE}._release_lease", new_callable=AsyncMock) as mock_release,
        ):
            mock_settings.worker.BG_JOB_TIMEOUT_SECS = 60
            task = asyncio.create_task(executor._execute_and_release(run_id, "worker-0", semaphore))
            await asyncio.wait_for(acquiring.wait(), timeout=1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        assert sent_claims[0] is not None and sent_claims[0].startswith("worker-0:")
        assert mock_finalize.await_args.kwargs["claim_id"] == sent_claims[0]
        mock_release.assert_awaited_once_with(run_id, sent_claims[0])
        assert not semaphore.locked()

    @pytest.mark.asyncio
    async def test_timeout_before_claim_does_not_finalize(self: "TestExecuteAndRelease") -> None:
        run_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        semaphore = asyncio.Semaphore(1)
        await semaphore.acquire()
        executor = WorkerExecutor()

        async def never_claims(rid: str, wn: str, claim: _ClaimSlot) -> None:
            await asyncio.sleep(9999)

        executor._execute_with_lease = AsyncMock(side_effect=never_claims)  # type: ignore[method-assign]

        with (
            patch(f"{MODULE}.settings") as mock_settings,
            patch(f"{MODULE}.finalize_run", new_callable=AsyncMock) as mock_finalize,
            patch(f"{MODULE}._release_lease", new_callable=AsyncMock) as mock_release,
        ):
            mock_settings.worker.BG_JOB_TIMEOUT_SECS = 0.01
            await executor._execute_and_release(run_id, "worker-0", semaphore)

        mock_finalize.assert_not_awaited()
        mock_release.assert_not_awaited()
        assert not semaphore.locked()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("failure_source", ["identity", "finalize"])
    async def test_cleanup_failure_preserves_cancellation(self, failure_source: str) -> None:
        run_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        thread_id = "tttttttt-tttt-tttt-tttt-tttttttttttt"
        semaphore = asyncio.Semaphore(1)
        await semaphore.acquire()
        executor = WorkerExecutor()
        executor._execute_with_lease = AsyncMock(side_effect=_claim_then_cancel)  # type: ignore[method-assign]
        explicit_run_cancellations.add(run_id)

        with (
            patch(f"{MODULE}.settings") as mock_settings,
            patch(
                f"{MODULE}._get_run_identity",
                new_callable=AsyncMock,
                return_value=(thread_id, "user-1"),
                side_effect=RuntimeError("database unavailable") if failure_source == "identity" else None,
            ),
            patch(
                f"{MODULE}.finalize_run",
                new_callable=AsyncMock,
                side_effect=RuntimeError("database unavailable") if failure_source == "finalize" else None,
            ),
            patch(f"{MODULE}._release_lease", new_callable=AsyncMock),
            patch(f"{MODULE}.logger.exception") as mock_log_exception,
        ):
            mock_settings.worker.BG_JOB_TIMEOUT_SECS = 60
            with pytest.raises(asyncio.CancelledError):
                await executor._execute_and_release(run_id, "worker-0", semaphore)

        mock_log_exception.assert_called_once_with(
            "Failed to persist explicit run cancellation",
            run_id=run_id,
        )
        assert run_id not in explicit_run_cancellations
        assert run_id not in active_runs
        assert not semaphore.locked()

    @pytest.mark.asyncio
    async def test_repeated_external_cancel_waits_for_database_reconciliation(self) -> None:
        run_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        thread_id = "tttttttt-tttt-tttt-tttt-tttttttttttt"
        semaphore = asyncio.Semaphore(1)
        await semaphore.acquire()
        executor = WorkerExecutor()
        execution_started = asyncio.Event()
        finalize_started = asyncio.Event()
        allow_finalize = asyncio.Event()
        inner_cancelled = False

        async def long_running(rid: str, worker_name: str, claim: _ClaimSlot) -> None:
            nonlocal inner_cancelled
            claim.claim_id = CLAIM_ID
            execution_started.set()
            try:
                await asyncio.sleep(9999)
            except asyncio.CancelledError:
                inner_cancelled = True
                raise

        async def delayed_finalize(*args: object, **kwargs: object) -> bool:
            finalize_started.set()
            await allow_finalize.wait()
            return True

        executor._execute_with_lease = AsyncMock(side_effect=long_running)  # type: ignore[method-assign]
        explicit_run_cancellations.add(run_id)

        with (
            patch(f"{MODULE}.settings") as mock_settings,
            patch(
                f"{MODULE}._get_run_identity",
                new_callable=AsyncMock,
                return_value=(thread_id, "user-1"),
            ),
            patch(
                f"{MODULE}.finalize_run",
                new_callable=AsyncMock,
                side_effect=delayed_finalize,
            ) as mock_finalize,
            patch(f"{MODULE}._release_lease", new_callable=AsyncMock),
        ):
            mock_settings.worker.BG_JOB_TIMEOUT_SECS = 60
            task = asyncio.create_task(executor._execute_and_release(run_id, "worker-0", semaphore))
            await asyncio.wait_for(execution_started.wait(), timeout=1)

            task.cancel()
            await asyncio.wait_for(finalize_started.wait(), timeout=1)
            task.cancel()
            await asyncio.sleep(0)

            assert not task.done()
            allow_finalize.set()
            with pytest.raises(asyncio.CancelledError):
                await task

        assert inner_cancelled is True
        mock_finalize.assert_awaited_once_with(
            run_id,
            thread_id,
            user_id="user-1",
            status="interrupted",
            thread_status="idle",
            output={},
            claim_id=CLAIM_ID,
        )
        assert run_id not in explicit_run_cancellations
        assert run_id not in active_runs
        assert not semaphore.locked()

    @pytest.mark.asyncio
    async def test_cleanup_leaves_a_newer_attempts_registration_alone(self: "TestExecuteAndRelease") -> None:
        """Regression: a reaped attempt finishing late popped the entry and cancel flag of
        the attempt that replaced it on this instance, so a cancel could no longer reach it."""
        semaphore = asyncio.Semaphore(1)
        await semaphore.acquire()
        executor = WorkerExecutor()
        newer_attempt = asyncio.create_task(asyncio.Event().wait())

        async def replaced_while_running(rid: str, wn: str, claim: _ClaimSlot) -> None:
            active_runs[rid] = newer_attempt
            explicit_run_cancellations.add(rid)

        executor._execute_with_lease = AsyncMock(side_effect=replaced_while_running)  # type: ignore[method-assign]
        with patch(f"{MODULE}.settings") as mock_settings:
            mock_settings.worker.BG_JOB_TIMEOUT_SECS = 60
            await executor._execute_and_release(RUN_ID, "worker-0", semaphore)

        assert active_runs.get(RUN_ID) is newer_attempt
        assert RUN_ID in explicit_run_cancellations
        assert not semaphore.locked()

    @pytest.mark.asyncio
    async def test_cleanup_clears_the_cancel_flag_when_someone_else_removed_the_entry(
        self: "TestExecuteAndRelease",
    ) -> None:
        """Regression: a thread delete popping the entry mid-unwind left the cancel flag set forever."""
        semaphore = asyncio.Semaphore(1)
        await semaphore.acquire()
        executor = WorkerExecutor()

        async def entry_removed_elsewhere(rid: str, wn: str, claim: _ClaimSlot) -> None:
            explicit_run_cancellations.add(rid)
            active_runs.pop(rid, None)

        executor._execute_with_lease = AsyncMock(side_effect=entry_removed_elsewhere)  # type: ignore[method-assign]
        with patch(f"{MODULE}.settings") as mock_settings:
            mock_settings.worker.BG_JOB_TIMEOUT_SECS = 60
            await executor._execute_and_release(RUN_ID, "worker-0", semaphore)

        assert RUN_ID not in explicit_run_cancellations
        assert RUN_ID not in active_runs
        assert not semaphore.locked()


class TestExecuteWithLease:
    @pytest.mark.asyncio
    async def test_cancels_job_task_in_finally(self) -> None:
        """Regression: when _execute_with_lease is cancelled (e.g. by wait_for
        timeout), the inner job_task must also be cancelled to prevent orphaned
        execution that corrupts run state."""
        run_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        executor = WorkerExecutor()

        job_task_was_cancelled = False

        async def long_running_job(job: object, *, claim_id: str | None = None) -> None:
            nonlocal job_task_was_cancelled
            try:
                await asyncio.sleep(9999)
            except asyncio.CancelledError:
                job_task_was_cancelled = True
                raise

        mock_loaded = MagicMock(spec=_LoadedRun)
        mock_loaded.job = _make_run_job()
        mock_loaded.trace = {}
        mock_loaded.claim_id = CLAIM_ID

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
            task = asyncio.create_task(executor._execute_with_lease(run_id, "worker-0", _ClaimSlot()))
            await asyncio.sleep(0.05)  # Let it start
            task.cancel()
            await task  # Completes normally (CancelledError is handled internally)

        assert job_task_was_cancelled, "job_task must be cancelled when _execute_with_lease is cancelled"

    @pytest.mark.asyncio
    async def test_does_not_start_the_graph_when_the_lease_ran_out_during_acquisition(
        self: "TestExecuteWithLease",
    ) -> None:
        """Regression: a slow claim round-trip handed the graph a lease that had already
        expired, so it ran while the reaper was free to dispatch the run elsewhere."""
        loaded = MagicMock(spec=_LoadedRun)
        loaded.job = _make_run_job()
        loaded.trace = {}
        loaded.claim_id = CLAIM_ID

        async def slow_acquire(rid: str, wn: str, *, claim_id: str | None = None) -> _LoadedRun:
            await asyncio.sleep(0.2)
            return loaded

        with (
            patch(f"{MODULE}.settings") as mock_settings,
            patch(f"{MODULE}._acquire_and_load", side_effect=slow_acquire),
            patch(f"{MODULE}._restore_trace_context"),
            patch(f"{MODULE}.execute_run", new_callable=AsyncMock) as mock_execute,
            patch(f"{MODULE}._heartbeat_loop", new_callable=AsyncMock),
            patch(f"{MODULE}._release_lease", new_callable=AsyncMock),
        ):
            mock_settings.worker.LEASE_DURATION_SECONDS = 0.1
            mock_settings.worker.HEARTBEAT_INTERVAL_SECONDS = 0.03
            await WorkerExecutor()._execute_with_lease(RUN_ID, "worker-0", _ClaimSlot())

        mock_execute.assert_not_called()


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


# ------------------------------------------------------------------
# Drain requeue (#474)
# ------------------------------------------------------------------


class TestRequeueDrainedRuns:
    @pytest.mark.asyncio
    async def test_resets_rows_and_pushes_to_queue(self) -> None:
        session = AsyncMock()
        result = MagicMock()
        result.fetchall.return_value = [("run-1",), ("run-2",)]
        session.execute = AsyncMock(return_value=result)
        mock_client = AsyncMock()

        with (
            patch(f"{MODULE}._get_session_maker", return_value=_make_session_maker(session)),
            patch(f"{MODULE}.redis_manager") as mock_redis,
            patch(f"{MODULE}.settings") as mock_settings,
        ):
            mock_redis.get_client.return_value = mock_client
            mock_settings.worker.WORKER_QUEUE_KEY = "aegra:jobs"

            await _requeue_drained_runs(["run-1", "run-2"], claim_ids=["c-1", "c-2"])

        session.commit.assert_awaited_once()
        stmt = session.execute.await_args.args[0]
        sql = str(stmt.compile())
        assert "UPDATE runs SET" in sql
        assert "runs.claimed_by IN" in sql
        assert "RETURNING runs.run_id" in sql
        assert mock_client.rpush.await_count == 2
        pushed = [call.args for call in mock_client.rpush.await_args_list]
        assert pushed == [("aegra:jobs", "run-1"), ("aegra:jobs", "run-2")]

    @pytest.mark.asyncio
    async def test_redis_outage_does_not_raise_after_rows_reset(self) -> None:
        """The DB reset commits first, so the stuck-pending reaper can recover
        even when the queue push fails."""
        session = AsyncMock()
        result = MagicMock()
        result.fetchall.return_value = [("run-1",)]
        session.execute = AsyncMock(return_value=result)
        mock_client = AsyncMock()
        mock_client.rpush = AsyncMock(side_effect=RedisConnectionError("down"))

        with (
            patch(f"{MODULE}._get_session_maker", return_value=_make_session_maker(session)),
            patch(f"{MODULE}.redis_manager") as mock_redis,
            patch(f"{MODULE}.settings") as mock_settings,
            patch(f"{MODULE}.logger.warning") as mock_warning,
        ):
            mock_redis.get_client.return_value = mock_client
            mock_settings.worker.WORKER_QUEUE_KEY = "aegra:jobs"

            await _requeue_drained_runs(["run-1"])

        session.commit.assert_awaited_once()
        mock_warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_rows_reset_skips_queue_push(self) -> None:
        """Runs that finalized during the drain (user cancel, completion) are
        terminal, so nothing is reset and nothing is pushed."""
        session = AsyncMock()
        result = MagicMock()
        result.fetchall.return_value = []
        session.execute = AsyncMock(return_value=result)

        with (
            patch(f"{MODULE}._get_session_maker", return_value=_make_session_maker(session)),
            patch(f"{MODULE}.redis_manager") as mock_redis,
        ):
            await _requeue_drained_runs(["run-1"])

        mock_redis.get_client.assert_not_called()


class TestStopRequeuesDrainedRuns:
    @pytest.mark.asyncio
    async def test_stop_requeues_jobs_alive_at_drain_deadline(self) -> None:
        """Jobs still executing when the drain window closes are cancelled with
        shutdown provenance and handed to the requeue path, not finalized (#474)."""

        async def _hang() -> None:
            await asyncio.Event().wait()

        hung_task = asyncio.create_task(_hang())
        await asyncio.sleep(0)  # let it start

        executor = WorkerExecutor()
        # Mapped at creation, deliberately absent from active_runs: a task
        # cancelled before it runs must still reach the requeue.
        executor._job_tasks = {hung_task: "run-hung"}

        seen_in_flag_set: list[bool] = []

        async def _fake_requeue(run_ids: list[str], *, claim_ids: list[str]) -> None:
            seen_in_flag_set.append("run-hung" in _shutdown_cancellations)
            assert run_ids == ["run-hung"]

        try:
            with (
                patch(f"{MODULE}._requeue_drained_runs", side_effect=_fake_requeue) as mock_requeue,
                patch(f"{MODULE}.settings") as mock_settings,
            ):
                mock_settings.worker.WORKER_DRAIN_TIMEOUT = 0.05
                await executor.stop()

            mock_requeue.assert_called_once()
            assert hung_task.cancelled()
            # The flag was set before the cancel reached the task
            assert seen_in_flag_set == [True]
        finally:
            _shutdown_cancellations.discard("run-hung")

    @pytest.mark.asyncio
    async def test_stop_excludes_explicitly_cancelled_runs_from_requeue(self) -> None:
        """A user cancel that races the shutdown stays a cancel: the run must
        not be resurrected by the drain requeue."""

        async def _hang() -> None:
            await asyncio.Event().wait()

        hung_task = asyncio.create_task(_hang())
        cancelled_task = asyncio.create_task(_hang())
        await asyncio.sleep(0)

        executor = WorkerExecutor()
        executor._job_tasks = {hung_task: "run-hung", cancelled_task: "run-user-cancel"}
        explicit_run_cancellations.add("run-user-cancel")

        try:
            with (
                patch(f"{MODULE}._requeue_drained_runs", new_callable=AsyncMock) as mock_requeue,
                patch(f"{MODULE}.settings") as mock_settings,
            ):
                mock_settings.worker.WORKER_DRAIN_TIMEOUT = 0.05
                await executor.stop()

            mock_requeue.assert_awaited_once_with(["run-hung"], claim_ids=[])
            assert "run-user-cancel" not in _shutdown_cancellations
        finally:
            explicit_run_cancellations.discard("run-user-cancel")
            _shutdown_cancellations.discard("run-hung")

    @pytest.mark.asyncio
    async def test_stop_does_not_requeue_jobs_that_finish_in_time(self) -> None:
        """Jobs completing inside the drain window finalize normally."""

        async def _quick() -> None:
            await asyncio.sleep(0)

        quick_task = asyncio.create_task(_quick())
        executor = WorkerExecutor()
        executor._job_tasks = {quick_task: "run-quick"}

        with (
            patch(f"{MODULE}._requeue_drained_runs", new_callable=AsyncMock) as mock_requeue,
            patch(f"{MODULE}.settings") as mock_settings,
        ):
            mock_settings.worker.WORKER_DRAIN_TIMEOUT = 1.0
            await executor.stop()

        mock_requeue.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_stop_cancels_worker_loops_before_requeue_push(self) -> None:
        """A loop still blocked in BLPOP would steal the requeue push back onto
        this dying instance, so loops must be gone before the push happens."""

        async def _hang() -> None:
            await asyncio.Event().wait()

        job_task = asyncio.create_task(_hang())
        loop_task = asyncio.create_task(_hang())
        await asyncio.sleep(0)

        events: list[str] = []
        loop_task.add_done_callback(lambda _t: events.append("loops-stopped"))

        async def _fake_requeue(run_ids: list[str], *, claim_ids: list[str]) -> None:
            events.append("requeue")

        executor = WorkerExecutor()
        executor._job_tasks = {job_task: "run-hung"}
        executor._worker_tasks = [loop_task]

        try:
            with (
                patch(f"{MODULE}._requeue_drained_runs", side_effect=_fake_requeue),
                patch(f"{MODULE}.settings") as mock_settings,
            ):
                mock_settings.worker.WORKER_DRAIN_TIMEOUT = 0.05
                await executor.stop()

            assert events == ["loops-stopped", "requeue"]
        finally:
            _shutdown_cancellations.discard("run-hung")

    @pytest.mark.asyncio
    async def test_worker_loop_pushes_back_run_dequeued_during_shutdown(self) -> None:
        """A loop that comes out of BLPOP after shutdown began must hand the
        run back instead of starting untracked work."""
        run_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        executor = WorkerExecutor()
        executor._running = True

        async def _dequeue_then_shutdown() -> str:
            executor._running = False
            return run_id

        executor._dequeue = _dequeue_then_shutdown  # type: ignore[method-assign]

        with (
            patch(f"{MODULE}._push_back", new_callable=AsyncMock) as mock_push_back,
            patch(f"{MODULE}.settings") as mock_settings,
        ):
            mock_settings.worker.N_JOBS_PER_WORKER = 1
            await executor._worker_loop("test-worker")

        mock_push_back.assert_awaited_once_with(run_id)
        assert not executor._job_tasks

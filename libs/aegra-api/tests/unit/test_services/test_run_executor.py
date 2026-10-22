"""Unit tests for run_executor service."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aegra_api.core.active_runs import active_runs
from aegra_api.models.auth import User
from aegra_api.models.run_job import RunExecution, RunIdentity, RunJob
from aegra_api.models.runs import Durability
from aegra_api.services import run_executor as run_executor_module
from aegra_api.services.run_executor import (
    _GraphResult,
    _lease_loss_cancellations,
    _shutdown_cancellations,
    _signal_end_event,
    _signal_run_done,
    _stream_legacy,
    _stream_native_v2,
    _timeout_cancellations,
    execute_run,
)


async def _empty_async_gen():  # type: ignore[no-untyped-def]
    return
    yield  # noqa: RET504 — makes this an async generator


def _make_job(run_id: str = "run-1") -> RunJob:
    return RunJob(
        identity=RunIdentity(run_id=run_id, thread_id="thread-1", graph_id="graph-1"),
        user=User(identity="user-1"),
        execution=RunExecution(input_data={"msg": "hello"}),
    )


def _patch_execute_run_deps() -> dict[str, MagicMock | AsyncMock]:
    """Return a dict of patch targets and their mocks for execute_run tests."""
    return {}


class TestExecuteRunSuccess:
    @pytest.mark.asyncio
    async def test_success_path_updates_status_and_signals(self) -> None:
        """execute_run sets running -> success and signals end event."""
        mock_graph = MagicMock()
        mock_graph.__aenter__ = AsyncMock(return_value=mock_graph)
        mock_graph.__aexit__ = AsyncMock(return_value=False)

        mock_service = MagicMock()
        mock_service.get_graph = MagicMock(return_value=mock_graph)

        mock_start = AsyncMock(return_value=True)
        mock_finalize = AsyncMock(return_value=True)

        with (
            patch("aegra_api.services.run_executor.get_langgraph_service", return_value=mock_service),
            patch("aegra_api.services.run_executor.start_run", mock_start),
            patch("aegra_api.services.run_executor.finalize_run", mock_finalize),
            patch("aegra_api.services.run_executor.streaming_service") as mock_streaming,
            patch("aegra_api.services.run_executor.stream_graph_events", return_value=_empty_async_gen()),
            patch("aegra_api.services.run_executor._signal_end_event", new_callable=AsyncMock) as mock_signal_end,
            patch("aegra_api.services.run_executor._signal_run_done", new_callable=AsyncMock),
            patch("aegra_api.services.run_executor.with_auth_ctx") as mock_auth,
        ):
            mock_auth_ctx = AsyncMock()
            mock_auth_ctx.__aenter__ = AsyncMock(return_value=None)
            mock_auth_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_auth.return_value = mock_auth_ctx
            mock_streaming.cleanup_run = AsyncMock()

            await execute_run(_make_job())

        mock_start.assert_awaited_once_with("run-1", user_id="user-1", claim_id=None)

        # finalize_run called once for success
        mock_finalize.assert_awaited_once()
        assert mock_finalize.await_args.kwargs["status"] == "success"

        mock_signal_end.assert_awaited_once_with("run-1", "success")


class TestExecuteRunCancelledError:
    @pytest.mark.asyncio
    async def test_cancelled_error_sets_interrupted_and_signals(self) -> None:
        mock_start = AsyncMock(return_value=True)
        mock_finalize = AsyncMock(return_value=True)

        with (
            patch("aegra_api.services.run_executor.start_run", mock_start),
            patch("aegra_api.services.run_executor.finalize_run", mock_finalize),
            patch("aegra_api.services.run_executor.streaming_service") as mock_streaming,
            patch("aegra_api.services.run_executor._signal_run_done", new_callable=AsyncMock),
            patch(
                "aegra_api.services.run_executor._stream_graph",
                new_callable=AsyncMock,
                side_effect=asyncio.CancelledError,
            ),
        ):
            mock_streaming.signal_run_cancelled = AsyncMock()
            mock_streaming.cleanup_run = AsyncMock()

            with pytest.raises(asyncio.CancelledError):
                await execute_run(_make_job())

        mock_start.assert_awaited_once_with("run-1", user_id="user-1", claim_id=None)
        # finalize_run called for "interrupted"
        mock_finalize.assert_awaited_once()
        assert mock_finalize.await_args.kwargs["status"] == "interrupted"
        assert mock_finalize.await_args.kwargs["user_id"] == "user-1"
        mock_streaming.signal_run_cancelled.assert_awaited_once_with("run-1")

    @pytest.mark.asyncio
    async def test_timeout_cancel_sets_error_and_signals(self) -> None:
        mock_finalize = AsyncMock(return_value=True)

        with (
            patch("aegra_api.services.run_executor.start_run", new_callable=AsyncMock, return_value=True),
            patch("aegra_api.services.run_executor.finalize_run", mock_finalize),
            patch("aegra_api.services.run_executor.streaming_service") as mock_streaming,
            patch("aegra_api.services.run_executor._signal_run_done", new_callable=AsyncMock),
            patch(
                "aegra_api.services.run_executor._stream_graph",
                new_callable=AsyncMock,
                side_effect=asyncio.CancelledError,
            ),
        ):
            mock_streaming.signal_run_error = AsyncMock()
            mock_streaming.cleanup_run = AsyncMock()
            _timeout_cancellations.add("run-1")
            try:
                with pytest.raises(asyncio.CancelledError):
                    await execute_run(_make_job())
            finally:
                _timeout_cancellations.discard("run-1")

        mock_finalize.assert_awaited_once_with(
            "run-1",
            "thread-1",
            user_id="user-1",
            claim_id=None,
            status="error",
            thread_status="error",
            output={},
            error="Job exceeded maximum execution time",
        )
        mock_streaming.signal_run_error.assert_awaited_once_with(
            "run-1",
            "TimeoutError: execution failed",
            "TimeoutError",
        )


class TestExecuteRunException:
    @pytest.mark.asyncio
    async def test_exception_sets_error_and_signals(self) -> None:
        mock_start = AsyncMock(return_value=True)
        mock_finalize = AsyncMock(return_value=True)

        with (
            patch("aegra_api.services.run_executor.start_run", mock_start),
            patch("aegra_api.services.run_executor.finalize_run", mock_finalize),
            patch("aegra_api.services.run_executor.streaming_service") as mock_streaming,
            patch("aegra_api.services.run_executor._signal_run_done", new_callable=AsyncMock),
            patch(
                "aegra_api.services.run_executor._stream_graph",
                new_callable=AsyncMock,
                side_effect=RuntimeError("graph exploded"),
            ),
        ):
            mock_streaming.signal_run_error = AsyncMock()
            mock_streaming.cleanup_run = AsyncMock()

            await execute_run(_make_job())

        mock_start.assert_awaited_once_with("run-1", user_id="user-1", claim_id=None)
        # finalize_run called for "error"
        mock_finalize.assert_awaited_once()
        assert mock_finalize.await_args.kwargs["status"] == "error"
        assert mock_finalize.await_args.kwargs["thread_status"] == "error"
        assert mock_finalize.await_args.kwargs["user_id"] == "user-1"
        mock_streaming.signal_run_error.assert_awaited_once()
        # Verify sanitized message used (not raw exception)
        error_args = mock_streaming.signal_run_error.await_args
        assert "RuntimeError" in error_args.args[1]
        assert "execution failed" in error_args.args[1]


def _v3_stream(*events: tuple[str, dict]):  # type: ignore[no-untyped-def]
    async def gen(**_kwargs):  # type: ignore[no-untyped-def]
        for method, event in events:
            yield method, event

    return gen


class TestStreamNativeV2InterruptDetection:
    """_stream_native_v2 must flag has_interrupt for every interrupt shape, else
    the run finalizes 'success' and the client gets input.requested + completed."""

    async def _run(self, *events: tuple[str, dict]) -> bool:
        result = _GraphResult()
        with (
            patch.object(run_executor_module, "stream_native_v3_events", _v3_stream(*events)),
            patch.object(run_executor_module, "broker_manager") as bm,
            patch.object(run_executor_module, "streaming_service") as ss,
        ):
            bm.allocate_event_id = AsyncMock(return_value="run-1_event_1")
            ss.put_to_broker = AsyncMock()
            await _stream_native_v2(_make_job(), MagicMock(), {"msg": "x"}, {}, result)
        return result.has_interrupt

    @pytest.mark.asyncio
    async def test_interrupt_via_values_params_interrupts(self) -> None:
        event = {"params": {"data": {"messages": []}, "interrupts": [{"id": "i1", "value": 1}]}}
        assert await self._run(("values", event)) is True

    @pytest.mark.asyncio
    async def test_interrupt_via_updates_dunder_interrupt(self) -> None:
        # The path session.py routes to input.requested but the executor missed.
        event = {"params": {"data": {"__interrupt__": [{"id": "i1", "value": 1}]}}}
        assert await self._run(("updates", event)) is True

    @pytest.mark.asyncio
    async def test_no_interrupt_when_absent(self) -> None:
        event = {"params": {"data": {"messages": []}}}
        assert await self._run(("values", event)) is False


class TestDurabilityForwarding:
    """The job's resolved durability reaches both stream producers."""

    @staticmethod
    def _job(durability: Durability | None) -> RunJob:
        return RunJob(
            identity=RunIdentity(run_id="run-1", thread_id="thread-1", graph_id="graph-1"),
            user=User(identity="user-1"),
            execution=RunExecution(input_data={"msg": "hello"}, durability=durability),
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("durability", ["exit", None])
    async def test_legacy_stream_receives_job_durability(self, durability: Durability | None) -> None:
        stream = MagicMock(return_value=_empty_async_gen())
        with patch.object(run_executor_module, "stream_graph_events", stream):
            await _stream_legacy(self._job(durability), MagicMock(), {"msg": "x"}, {}, ["values"], _GraphResult())

        assert stream.call_args.kwargs["durability"] == durability

    @pytest.mark.asyncio
    @pytest.mark.parametrize("durability", ["exit", None])
    async def test_native_v2_stream_receives_job_durability(self, durability: Durability | None) -> None:
        stream = MagicMock(return_value=_empty_async_gen())
        with patch.object(run_executor_module, "stream_native_v3_events", stream):
            await _stream_native_v2(self._job(durability), MagicMock(), {"msg": "x"}, {}, _GraphResult())

        assert stream.call_args.kwargs["durability"] == durability


class TestSignalEndEvent:
    @pytest.mark.asyncio
    async def test_publishes_end_event(self) -> None:
        mock_broker = MagicMock()
        mock_broker.is_finished.return_value = False
        mock_broker.put = AsyncMock()

        with patch("aegra_api.services.run_executor.broker_manager") as mock_bm:
            mock_bm.get_broker.return_value = mock_broker
            mock_bm.allocate_event_id = AsyncMock(return_value="run-1_event_5")

            await _signal_end_event("run-1", "success")

        mock_broker.put.assert_awaited_once_with("run-1_event_5", ("end", {"status": "success"}))

    @pytest.mark.asyncio
    async def test_noop_when_broker_is_none(self) -> None:
        with patch("aegra_api.services.run_executor.broker_manager") as mock_bm:
            mock_bm.get_broker.return_value = None

            await _signal_end_event("run-1", "success")
            # No error, no put call

    @pytest.mark.asyncio
    async def test_noop_when_broker_is_finished(self) -> None:
        mock_broker = MagicMock()
        mock_broker.is_finished.return_value = True

        with patch("aegra_api.services.run_executor.broker_manager") as mock_bm:
            mock_bm.get_broker.return_value = mock_broker

            await _signal_end_event("run-1", "success")


class TestSignalRunDone:
    @pytest.mark.asyncio
    async def test_sets_redis_key(self) -> None:
        mock_client = AsyncMock()

        with patch("aegra_api.services.run_executor.redis_manager") as mock_rm:
            mock_rm.get_client.return_value = mock_client

            await _signal_run_done("run-1")

        mock_client.set.assert_awaited_once()
        call_args = mock_client.set.await_args
        assert "run-1" in call_args.args[0]
        assert call_args.args[1] == "1"

    @pytest.mark.asyncio
    async def test_uses_configured_channel_prefix(self) -> None:
        """Regression: done-key must derive from REDIS_CHANNEL_PREFIX, not a hardcoded string."""
        mock_client = AsyncMock()

        with (
            patch("aegra_api.services.run_executor.redis_manager") as mock_rm,
            patch("aegra_api.services.run_executor.settings") as mock_settings,
        ):
            mock_rm.get_client.return_value = mock_client
            mock_settings.redis.REDIS_CHANNEL_PREFIX = "aegra:agent-foo:run:"

            await _signal_run_done("run-1")

        key = mock_client.set.await_args.args[0]
        assert key == "aegra:agent-foo:run:done:run-1"

    @pytest.mark.asyncio
    async def test_logs_debug_on_redis_failure(self) -> None:
        with patch("aegra_api.services.run_executor.redis_manager") as mock_rm:
            mock_rm.get_client.side_effect = Exception("connection refused")

            # Should not raise
            await _signal_run_done("run-1")


class TestLeaseLossCancellation:
    @pytest.mark.asyncio
    async def test_lease_loss_cancel_skips_finalize_and_signal(self) -> None:
        """Regression: when cancellation is due to lease loss (not user action),
        execute_run must NOT finalize the run, send SSE events, signal done,
        or clean up the broker - another worker will re-execute it."""
        mock_start = AsyncMock(return_value=True)
        mock_finalize = AsyncMock()
        mock_signal_done = AsyncMock()

        with (
            patch("aegra_api.services.run_executor.start_run", mock_start),
            patch("aegra_api.services.run_executor.finalize_run", mock_finalize),
            patch("aegra_api.services.run_executor.streaming_service") as mock_streaming,
            patch("aegra_api.services.run_executor._signal_run_done", mock_signal_done),
            patch(
                "aegra_api.services.run_executor._stream_graph",
                new_callable=AsyncMock,
                side_effect=asyncio.CancelledError,
            ),
        ):
            mock_streaming.signal_run_cancelled = AsyncMock()
            mock_streaming.cleanup_run = AsyncMock()

            # Simulate heartbeat marking this as a lease-loss cancel
            _lease_loss_cancellations.add("run-1")
            try:
                with pytest.raises(asyncio.CancelledError):
                    await execute_run(_make_job())
            finally:
                _lease_loss_cancellations.discard("run-1")

        # finalize_run must NOT be called — the new worker owns this run
        mock_finalize.assert_not_awaited()
        # SSE cancel signal must NOT be sent — clients should stay connected
        mock_streaming.signal_run_cancelled.assert_not_awaited()
        # Done-key must NOT be set — would cause wait_for_completion to return early
        mock_signal_done.assert_not_awaited()
        # Broker must NOT be cleaned up — new worker needs it
        mock_streaming.cleanup_run.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_user_cancel_still_finalizes(self) -> None:
        """Normal (user-initiated) cancellation must still finalize and signal."""
        mock_start = AsyncMock(return_value=True)
        mock_finalize = AsyncMock(return_value=True)
        mock_signal_done = AsyncMock()

        with (
            patch("aegra_api.services.run_executor.start_run", mock_start),
            patch("aegra_api.services.run_executor.finalize_run", mock_finalize),
            patch("aegra_api.services.run_executor.streaming_service") as mock_streaming,
            patch("aegra_api.services.run_executor._signal_run_done", mock_signal_done),
            patch(
                "aegra_api.services.run_executor._stream_graph",
                new_callable=AsyncMock,
                side_effect=asyncio.CancelledError,
            ),
        ):
            mock_streaming.signal_run_cancelled = AsyncMock()
            mock_streaming.cleanup_run = AsyncMock()

            with pytest.raises(asyncio.CancelledError):
                await execute_run(_make_job())

        # Normal cancel: finalize and signal MUST happen
        mock_finalize.assert_awaited_once()
        assert mock_finalize.await_args.kwargs["status"] == "interrupted"
        mock_streaming.signal_run_cancelled.assert_awaited_once_with("run-1")
        # Done-key and cleanup MUST happen on normal cancel
        mock_signal_done.assert_awaited_once_with("run-1")
        mock_streaming.cleanup_run.assert_awaited_once_with("run-1")


class TestShutdownDrainCancellation:
    @pytest.mark.asyncio
    async def test_shutdown_cancel_skips_finalize_and_signal(self) -> None:
        """A drain cancel goes back to the queue: finalizing it as interrupted
        would make graceful shutdown lose runs a plain crash recovers (#474)."""
        mock_start = AsyncMock(return_value=True)
        mock_finalize = AsyncMock()
        mock_signal_done = AsyncMock()

        with (
            patch("aegra_api.services.run_executor.start_run", mock_start),
            patch("aegra_api.services.run_executor.finalize_run", mock_finalize),
            patch("aegra_api.services.run_executor.streaming_service") as mock_streaming,
            patch("aegra_api.services.run_executor._signal_run_done", mock_signal_done),
            patch(
                "aegra_api.services.run_executor._stream_graph",
                new_callable=AsyncMock,
                side_effect=asyncio.CancelledError,
            ),
        ):
            mock_streaming.signal_run_cancelled = AsyncMock()
            mock_streaming.cleanup_run = AsyncMock()

            _shutdown_cancellations.add("run-1")
            try:
                with pytest.raises(asyncio.CancelledError):
                    await execute_run(_make_job())
            finally:
                _shutdown_cancellations.discard("run-1")

        # finalize_run must NOT be called — the run is requeued for another instance
        mock_finalize.assert_not_awaited()
        # SSE cancel signal must NOT be sent — clients should stay connected
        mock_streaming.signal_run_cancelled.assert_not_awaited()
        # Done-key must NOT be set — wait_for_completion would return early
        mock_signal_done.assert_not_awaited()
        # Broker must NOT be cleaned up — the resuming worker needs it
        mock_streaming.cleanup_run.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_shutdown_flag_is_cleared_after_cancel(self) -> None:
        """The provenance flag must not leak into a later run with the same id."""
        with (
            patch("aegra_api.services.run_executor.start_run", new_callable=AsyncMock, return_value=True),
            patch("aegra_api.services.run_executor.finalize_run", new_callable=AsyncMock),
            patch("aegra_api.services.run_executor.streaming_service") as mock_streaming,
            patch("aegra_api.services.run_executor._signal_run_done", new_callable=AsyncMock),
            patch(
                "aegra_api.services.run_executor._stream_graph",
                new_callable=AsyncMock,
                side_effect=asyncio.CancelledError,
            ),
        ):
            mock_streaming.signal_run_cancelled = AsyncMock()
            mock_streaming.cleanup_run = AsyncMock()

            _shutdown_cancellations.add("run-1")
            with pytest.raises(asyncio.CancelledError):
                await execute_run(_make_job())

        assert "run-1" not in _shutdown_cancellations


class TestTerminalStateRaces:
    @pytest.mark.asyncio
    async def test_terminal_run_does_not_start_graph_execution(self) -> None:
        with (
            patch("aegra_api.services.run_executor.start_run", new_callable=AsyncMock, return_value=False),
            patch("aegra_api.services.run_executor._stream_graph", new_callable=AsyncMock) as mock_stream,
            patch("aegra_api.services.run_executor.finalize_run", new_callable=AsyncMock) as mock_finalize,
            patch("aegra_api.services.run_executor.streaming_service") as mock_streaming,
            patch("aegra_api.services.run_executor._signal_run_done", new_callable=AsyncMock),
        ):
            mock_streaming.cleanup_run = AsyncMock()

            await execute_run(_make_job())

        mock_stream.assert_not_awaited()
        mock_finalize.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_lost_finalization_race_does_not_signal_success(self) -> None:
        graph_result = MagicMock(has_interrupt=False, data={"late": True})
        with (
            patch("aegra_api.services.run_executor.start_run", new_callable=AsyncMock, return_value=True),
            patch("aegra_api.services.run_executor._stream_graph", new_callable=AsyncMock, return_value=graph_result),
            patch("aegra_api.services.run_executor.finalize_run", new_callable=AsyncMock, return_value=False),
            patch("aegra_api.services.run_executor._signal_end_event", new_callable=AsyncMock) as mock_signal_end,
            patch("aegra_api.services.run_executor.streaming_service") as mock_streaming,
            patch("aegra_api.services.run_executor._signal_run_done", new_callable=AsyncMock),
        ):
            mock_streaming.cleanup_run = AsyncMock()

            await execute_run(_make_job())

        mock_signal_end.assert_not_awaited()


class TestClaimFencing:
    """A worker's claim id fences every run write, and stream teardown (#502)."""

    @staticmethod
    def _patches(*, finalized: bool, run_active: bool) -> tuple[AsyncMock, AsyncMock, AsyncMock, MagicMock, AsyncMock]:
        graph_result = MagicMock(has_interrupt=False, data={"answer": 42})
        return (
            AsyncMock(return_value=True),
            AsyncMock(return_value=finalized),
            AsyncMock(return_value=graph_result),
            MagicMock(),
            AsyncMock(return_value=run_active),
        )

    async def _run(
        self: "TestClaimFencing", *, claim_id: str | None, finalized: bool, run_active: bool
    ) -> dict[str, AsyncMock]:
        start, finalize, stream, streaming, is_active = self._patches(finalized=finalized, run_active=run_active)
        streaming.cleanup_run = AsyncMock()
        with (
            patch("aegra_api.services.run_executor.start_run", start),
            patch("aegra_api.services.run_executor.finalize_run", finalize),
            patch("aegra_api.services.run_executor._stream_graph", stream),
            patch("aegra_api.services.run_executor.streaming_service", streaming),
            patch("aegra_api.services.run_executor.is_run_active", is_active),
            patch("aegra_api.services.run_executor._signal_end_event", new_callable=AsyncMock),
            patch("aegra_api.services.run_executor._signal_run_done", new_callable=AsyncMock) as done,
        ):
            await execute_run(_make_job(), claim_id=claim_id)
        return {
            "start": start,
            "finalize": finalize,
            "cleanup": streaming.cleanup_run,
            "done": done,
            "active": is_active,
        }

    @pytest.mark.asyncio
    async def test_passes_claim_id_to_start_and_finalize(self: "TestClaimFencing") -> None:
        mocks = await self._run(claim_id="host-1-worker-0:abc", finalized=True, run_active=False)

        mocks["start"].assert_awaited_once_with("run-1", user_id="user-1", claim_id="host-1-worker-0:abc")
        assert mocks["finalize"].await_args.kwargs["claim_id"] == "host-1-worker-0:abc"
        mocks["cleanup"].assert_awaited_once_with("run-1")
        mocks["active"].assert_not_awaited()

    @pytest.mark.asyncio
    async def test_rejected_finalize_leaves_stream_to_the_attempt_that_replaced_it(self: "TestClaimFencing") -> None:
        mocks = await self._run(claim_id="host-1-worker-0:stale", finalized=False, run_active=True)

        mocks["active"].assert_awaited_once_with("run-1", user_id="user-1")
        mocks["cleanup"].assert_not_awaited()
        mocks["done"].assert_not_awaited()

    @pytest.mark.asyncio
    async def test_rejected_finalize_on_a_terminal_run_still_tears_down(self: "TestClaimFencing") -> None:
        mocks = await self._run(claim_id="host-1-worker-0:stale", finalized=False, run_active=False)

        mocks["cleanup"].assert_awaited_once_with("run-1")
        mocks["done"].assert_awaited_once_with("run-1")

    @pytest.mark.asyncio
    async def test_unleased_executor_tears_down_without_an_ownership_check(self: "TestClaimFencing") -> None:
        mocks = await self._run(claim_id=None, finalized=False, run_active=True)

        mocks["active"].assert_not_awaited()
        mocks["cleanup"].assert_awaited_once_with("run-1")

    @pytest.mark.asyncio
    async def test_ownership_check_failure_skips_teardown(self: "TestClaimFencing") -> None:
        start, finalize, stream, streaming, _ = self._patches(finalized=False, run_active=True)
        streaming.cleanup_run = AsyncMock()
        with (
            patch("aegra_api.services.run_executor.start_run", start),
            patch("aegra_api.services.run_executor.finalize_run", finalize),
            patch("aegra_api.services.run_executor._stream_graph", stream),
            patch("aegra_api.services.run_executor.streaming_service", streaming),
            patch(
                "aegra_api.services.run_executor.is_run_active",
                new_callable=AsyncMock,
                side_effect=OSError("database unreachable"),
            ),
            patch("aegra_api.services.run_executor._signal_run_done", new_callable=AsyncMock) as done,
        ):
            await execute_run(_make_job(), claim_id="host-1-worker-0:stale")

        streaming.cleanup_run.assert_not_awaited()
        done.assert_not_awaited()


class TestActiveRunRegistration:
    @pytest.mark.asyncio
    async def test_finishing_leaves_another_tasks_registration_alone(self: "TestActiveRunRegistration") -> None:
        """Regression: a reaped worker attempt unwinding late popped the entry of the
        attempt that replaced it, so cancels could no longer reach the live one."""
        newer_attempt = asyncio.create_task(asyncio.Event().wait())
        active_runs["run-1"] = newer_attempt
        try:
            with (
                patch("aegra_api.services.run_executor.start_run", AsyncMock(return_value=True)),
                patch("aegra_api.services.run_executor.finalize_run", AsyncMock(return_value=True)),
                patch(
                    "aegra_api.services.run_executor._stream_graph",
                    AsyncMock(return_value=MagicMock(has_interrupt=False, data={})),
                ),
                patch("aegra_api.services.run_executor.streaming_service") as streaming,
                patch("aegra_api.services.run_executor._signal_end_event", new_callable=AsyncMock),
                patch("aegra_api.services.run_executor._signal_run_done", new_callable=AsyncMock),
            ):
                streaming.cleanup_run = AsyncMock()
                await execute_run(_make_job(), claim_id="host-1-worker-0:stale")

            assert active_runs.get("run-1") is newer_attempt
        finally:
            active_runs.pop("run-1", None)
            newer_attempt.cancel()
            await asyncio.gather(newer_attempt, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_finishing_removes_its_own_registration(self: "TestActiveRunRegistration") -> None:
        async def run_registered() -> None:
            with (
                patch("aegra_api.services.run_executor.start_run", AsyncMock(return_value=True)),
                patch("aegra_api.services.run_executor.finalize_run", AsyncMock(return_value=True)),
                patch(
                    "aegra_api.services.run_executor._stream_graph",
                    AsyncMock(return_value=MagicMock(has_interrupt=False, data={})),
                ),
                patch("aegra_api.services.run_executor.streaming_service") as streaming,
                patch("aegra_api.services.run_executor._signal_end_event", new_callable=AsyncMock),
                patch("aegra_api.services.run_executor._signal_run_done", new_callable=AsyncMock),
            ):
                streaming.cleanup_run = AsyncMock()
                await execute_run(_make_job())

        task = asyncio.create_task(run_registered())
        active_runs["run-1"] = task
        await asyncio.wait_for(task, timeout=5)

        assert "run-1" not in active_runs

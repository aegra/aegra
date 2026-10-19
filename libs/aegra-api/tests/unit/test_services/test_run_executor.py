"""Unit tests for run_executor service."""

import asyncio
import contextlib
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aegra_api.models.auth import User
from aegra_api.models.run_job import RunBehavior, RunExecution, RunIdentity, RunJob
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
from aegra_api.services.webhook_delivery import RunOutcome


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

        mock_start.assert_awaited_once_with("run-1", user_id="user-1")

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

        mock_start.assert_awaited_once_with("run-1", user_id="user-1")
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

        mock_start.assert_awaited_once_with("run-1", user_id="user-1")
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


def _stream_filling(
    data: dict[str, Any], *, interrupt: bool = False, error: BaseException | None = None
) -> Callable[[RunJob, _GraphResult], Awaitable[_GraphResult]]:
    """A ``_stream_graph`` stand-in that streams *data* into the caller's result, then optionally fails."""

    async def _fake(_job: RunJob, result: _GraphResult) -> _GraphResult:
        result.data = data
        result.state = data
        result.has_interrupt = interrupt
        if error is not None:
            raise error
        return result

    return _fake


class TestExecuteRunWebhook:
    """Every terminal run hands its outcome to the webhook dispatcher; runs that resume elsewhere do not."""

    @staticmethod
    async def _execute(
        stream: Callable[[RunJob, _GraphResult], Awaitable[_GraphResult]],
        *,
        finalized: bool = True,
        started: bool = True,
        finalize: AsyncMock | None = None,
    ) -> MagicMock:
        with (
            patch("aegra_api.services.run_executor.start_run", new_callable=AsyncMock, return_value=started),
            patch("aegra_api.services.run_executor._stream_graph", stream),
            patch("aegra_api.services.run_executor.finalize_run", finalize or AsyncMock(return_value=finalized)),
            patch("aegra_api.services.run_executor._signal_end_event", new_callable=AsyncMock),
            patch("aegra_api.services.run_executor.streaming_service") as mock_streaming,
            patch("aegra_api.services.run_executor._signal_run_done", new_callable=AsyncMock),
            patch("aegra_api.services.run_executor.notify_run_finished") as mock_notify,
        ):
            mock_streaming.cleanup_run = AsyncMock()
            mock_streaming.signal_run_error = AsyncMock()
            mock_streaming.signal_run_cancelled = AsyncMock()
            with contextlib.suppress(asyncio.CancelledError):
                await execute_run(_make_job())
        return mock_notify

    @staticmethod
    def _outcome(mock_notify: MagicMock) -> RunOutcome:
        mock_notify.assert_called_once()
        job, outcome = mock_notify.call_args.args
        assert job.identity.run_id == "run-1"
        return outcome

    async def test_success_reports_final_values(self) -> None:
        mock_notify = await self._execute(_stream_filling({"count": 2}))

        outcome = self._outcome(mock_notify)
        assert outcome.status == "success"
        assert outcome.values == {"count": 2}
        assert outcome.error is None
        assert outcome.run_started_at is not None
        assert outcome.run_started_at <= outcome.run_ended_at

    async def test_graph_interrupt_reports_interrupted(self) -> None:
        mock_notify = await self._execute(_stream_filling({"count": 1}, interrupt=True))

        assert self._outcome(mock_notify).status == "interrupted"

    async def test_failure_reports_error_and_partial_values(self) -> None:
        mock_notify = await self._execute(_stream_filling({"count": 1}, error=RuntimeError("graph exploded")))

        outcome = self._outcome(mock_notify)
        assert outcome.status == "error"
        assert outcome.error == {"error": "RuntimeError", "message": "graph exploded"}
        assert outcome.values == {"count": 1}

    async def test_timeout_reports_timeout_error(self) -> None:
        _timeout_cancellations.add("run-1")
        try:
            mock_notify = await self._execute(_stream_filling({}, error=asyncio.CancelledError()))
        finally:
            _timeout_cancellations.discard("run-1")

        outcome = self._outcome(mock_notify)
        assert outcome.status == "error"
        assert outcome.error == {"error": "TimeoutError", "message": "Job exceeded maximum execution time"}

    async def test_user_cancel_reports_interrupted(self) -> None:
        mock_notify = await self._execute(_stream_filling({"count": 1}, error=asyncio.CancelledError()))

        outcome = self._outcome(mock_notify)
        assert outcome.status == "interrupted"
        assert outcome.error is None

    @pytest.mark.parametrize("cancellations", [_lease_loss_cancellations, _shutdown_cancellations])
    async def test_run_that_resumes_elsewhere_is_not_reported(self, cancellations: set[str]) -> None:
        cancellations.add("run-1")
        try:
            mock_notify = await self._execute(_stream_filling({}, error=asyncio.CancelledError()))
        finally:
            cancellations.discard("run-1")

        mock_notify.assert_not_called()

    async def test_lost_finalization_race_is_not_reported(self) -> None:
        mock_notify = await self._execute(_stream_filling({"count": 2}), finalized=False)

        mock_notify.assert_not_called()

    async def test_failed_success_finalization_reports_the_error_status_it_stored(self) -> None:
        # The success write fails, the error path then records the run as an error.
        finalize = AsyncMock(side_effect=[RuntimeError("db write failed"), True])

        mock_notify = await self._execute(_stream_filling({"count": 2}), finalize=finalize)

        outcome = self._outcome(mock_notify)
        assert finalize.await_args_list[1].kwargs["status"] == "error"
        assert outcome.status == "error"
        assert outcome.error == {"error": "RuntimeError", "message": "db write failed"}

    async def test_run_that_never_started_is_not_reported(self) -> None:
        mock_notify = await self._execute(_stream_filling({}), started=False)

        mock_notify.assert_not_called()


def _webhook_job() -> RunJob:
    return RunJob(
        identity=RunIdentity(run_id="run-1", thread_id="thread-1", graph_id="graph-1"),
        user=User(identity="user-1"),
        execution=RunExecution(input_data={"msg": "hello"}, stream_mode=["updates"]),
        behavior=RunBehavior(webhook="https://hooks.example.com/runs"),
    )


class TestStreamGraphResult:
    """What ``_stream_graph`` keeps for the completion webhook."""

    @staticmethod
    async def _stream(
        job: RunJob,
        *events: tuple[str, Any],
        error: Exception | None = None,
        state: Any = None,
        state_error: Exception | None = None,
    ) -> tuple[_GraphResult, MagicMock]:
        result = _GraphResult()

        async def _events(**_kwargs: Any) -> AsyncIterator[tuple[str, Any]]:
            for event in events:
                yield event
            if error is not None:
                raise error

        mock_graph = MagicMock()
        mock_graph.__aenter__ = AsyncMock(return_value=mock_graph)
        mock_graph.__aexit__ = AsyncMock(return_value=False)
        mock_graph.aget_state = AsyncMock(return_value=state, side_effect=state_error)
        mock_service = MagicMock()
        mock_service.get_graph = MagicMock(return_value=mock_graph)

        with (
            patch.object(run_executor_module, "get_langgraph_service", return_value=mock_service),
            patch.object(run_executor_module, "stream_graph_events", _events),
            patch.object(run_executor_module, "with_auth_ctx") as mock_auth,
            patch.object(run_executor_module, "broker_manager") as bm,
            patch.object(run_executor_module, "streaming_service") as ss,
        ):
            mock_auth.return_value.__aenter__ = AsyncMock(return_value=None)
            mock_auth.return_value.__aexit__ = AsyncMock(return_value=False)
            bm.allocate_event_id = AsyncMock(return_value="run-1_event_1")
            ss.put_to_broker = AsyncMock()
            if error is None:
                await run_executor_module._stream_graph(job, result)
            else:
                with pytest.raises(type(error)):
                    await run_executor_module._stream_graph(job, result)
        return result, mock_graph

    async def test_keeps_values_streamed_before_a_failure_in_the_callers_result(self) -> None:
        result, _graph = await self._stream(
            _make_job(), ("values", {"count": 1}), error=RuntimeError("mid-run failure")
        )

        assert result.data == {"count": 1}
        assert result.state == {"count": 1}

    async def test_webhook_run_reports_checkpointed_state_without_values_events(self) -> None:
        # The client only streamed updates; the webhook still gets the final state.
        result, graph = await self._stream(
            _webhook_job(), ("updates", {"step": {"count": 3}}), state=MagicMock(values={"count": 3})
        )

        assert result.state == {"count": 3}
        config = graph.aget_state.await_args.args[0]
        assert config["configurable"]["thread_id"] == "thread-1"
        assert "checkpoint_id" not in config["configurable"]

    async def test_run_without_webhook_skips_the_state_read(self) -> None:
        result, graph = await self._stream(_make_job(), ("values", {"count": 1}))

        graph.aget_state.assert_not_awaited()
        assert result.state == {"count": 1}

    async def test_interrupt_notice_is_not_taken_for_state(self) -> None:
        interrupt = {"__interrupt__": [{"value": "approve?", "id": "i1"}]}

        result, _graph = await self._stream(_make_job(), ("values", {"count": 1}), ("values", interrupt))

        assert result.has_interrupt is True
        assert result.data == interrupt
        assert result.state == {"count": 1}

    async def test_failed_state_read_keeps_the_streamed_values_and_the_run(self) -> None:
        # The read is best effort: it must neither fail the run nor drop what was streamed.
        result, graph = await self._stream(
            _webhook_job(), ("values", {"count": 2}), state_error=RuntimeError("checkpointer down")
        )

        graph.aget_state.assert_awaited_once()
        assert result.state == {"count": 2}

"""Tests for executor abstraction (local and worker)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aegra_api.core.active_runs import active_runs
from aegra_api.core.cancellation_state import cancellations
from aegra_api.models.auth import User
from aegra_api.models.run_job import RunExecution, RunIdentity, RunJob
from aegra_api.services.local_executor import LocalExecutor


async def _empty_async_gen():
    return
    yield  # noqa: RET504 — makes this an async generator


def _make_job(run_id: str = "run-1") -> RunJob:
    return RunJob(
        identity=RunIdentity(run_id=run_id, thread_id="thread-1", graph_id="graph-1"),
        user=User(identity="user-1"),
        execution=RunExecution(input_data={"msg": "hello"}),
    )


class TestLocalExecutor:
    @pytest.mark.asyncio
    async def test_submit_creates_task(self) -> None:
        executor = LocalExecutor()
        mock_execute = AsyncMock()

        with (
            patch("aegra_api.services.run_executor.execute_run", mock_execute),
            patch("aegra_api.services.local_executor.make_run_trace_context", return_value=None),
        ):
            job = _make_job()
            await executor.submit(job)

            # Task should be registered in active_runs
            from aegra_api.core.active_runs import active_runs

            assert "run-1" in active_runs
            task = active_runs.pop("run-1")
            task.cancel()

    @pytest.mark.asyncio
    async def test_submit_done_callback_clears_cancel_tag(self) -> None:
        """LocalExecutor owns the cancel-tag clear (since execute_run no longer
        does it). Without the done-callback, the registry would leak one entry
        per user-cancelled run in dev mode."""
        executor = LocalExecutor()

        async def fake_execute(_job: RunJob) -> None:
            await asyncio.sleep(0)

        with (
            patch("aegra_api.services.run_executor.execute_run", side_effect=fake_execute),
            patch("aegra_api.services.local_executor.make_run_trace_context", return_value=None),
        ):
            job = _make_job(run_id="run-cb")
            cancellations.mark("run-cb", "user")
            await executor.submit(job)
            task = active_runs["run-cb"]
            # fake_execute returns None — capture so done_callback fires
            # before the assertion below.
            result = await task
            assert result is None

        # Done-callback fires synchronously; the tag must be gone now.
        assert cancellations.reason_of("run-cb") is None
        active_runs.pop("run-cb", None)

    @pytest.mark.asyncio
    async def test_wait_for_completion_returns_on_done(self) -> None:
        executor = LocalExecutor()

        # Create a task that completes immediately
        async def quick() -> None:
            pass

        from aegra_api.core.active_runs import active_runs

        task = asyncio.create_task(quick())
        active_runs["run-done"] = task
        await asyncio.sleep(0.01)

        await executor.wait_for_completion("run-done", timeout=1.0)
        active_runs.pop("run-done", None)

    @pytest.mark.asyncio
    async def test_wait_for_completion_returns_on_missing_run(self) -> None:
        executor = LocalExecutor()
        # Should return immediately, not raise
        await executor.wait_for_completion("nonexistent", timeout=1.0)

    @pytest.mark.asyncio
    async def test_stop_cancels_active_tasks(self) -> None:
        executor = LocalExecutor()

        from aegra_api.core.active_runs import active_runs

        async def hang_forever() -> None:
            await asyncio.sleep(9999)

        task = asyncio.create_task(hang_forever())
        active_runs["run-hang"] = task

        await executor.stop()
        # Give event loop a tick to process cancellation
        await asyncio.sleep(0.01)
        assert task.done()
        active_runs.pop("run-hang", None)


class TestRunExecutorBoundaryConditions:
    """Boundary condition tests for run_executor edge cases."""

    @pytest.mark.asyncio
    async def test_empty_context_passed_as_dict_not_none(self) -> None:
        """Empty context {} must reach get_graph as {}, not None.

        Regression test: `context or None` evaluates to None because
        empty dict is falsy. Factory graphs that read context.model
        crash with AttributeError if context is None.
        """
        job = RunJob(
            identity=RunIdentity(run_id="r1", thread_id="t1", graph_id="g1"),
            user=User(identity="u1"),
            execution=RunExecution(context={}),
        )

        mock_graph = MagicMock()
        mock_graph.__aenter__ = AsyncMock(return_value=mock_graph)
        mock_graph.__aexit__ = AsyncMock(return_value=False)

        mock_service = MagicMock()
        mock_service.get_graph = MagicMock(return_value=mock_graph)

        with (
            patch("aegra_api.services.run_executor.get_langgraph_service", return_value=mock_service),
            patch("aegra_api.services.run_executor.update_run_status", new_callable=AsyncMock),
            patch("aegra_api.services.run_executor.finalize_run", new_callable=AsyncMock),
            patch("aegra_api.services.run_executor.streaming_service") as mock_streaming,
            patch("aegra_api.services.run_executor.stream_graph_events", return_value=_empty_async_gen()),
        ):
            mock_streaming.cleanup_run = AsyncMock()
            mock_streaming.signal_run_error = AsyncMock()

            from aegra_api.services.run_executor import execute_run

            await execute_run(job)

            # Verify context was passed as {} not None
            call_kwargs = mock_service.get_graph.call_args
            assert call_kwargs.kwargs["context"] == {}, (
                f"Expected context={{}}, got context={call_kwargs.kwargs['context']}"
            )


class TestExecutorFactory:
    def test_creates_local_when_redis_disabled(self) -> None:
        with patch("aegra_api.services.executor.settings") as mock_settings:
            mock_settings.redis.REDIS_BROKER_ENABLED = False
            from aegra_api.services.executor import _create_executor

            result = _create_executor()
            assert isinstance(result, LocalExecutor)

    def test_creates_worker_when_redis_enabled(self) -> None:
        with patch("aegra_api.services.executor.settings") as mock_settings:
            mock_settings.redis.REDIS_BROKER_ENABLED = True
            from aegra_api.services.executor import _create_executor
            from aegra_api.services.worker_executor import WorkerExecutor

            result = _create_executor()
            assert isinstance(result, WorkerExecutor)

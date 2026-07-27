"""Tests for executor abstraction (local and worker)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
        # Absent from active_runs and terminal/absent in DB → returns, not hang/raise.
        with patch("aegra_api.services.local_executor._is_run_terminal", new_callable=AsyncMock, return_value=True):
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
            patch("aegra_api.services.run_executor.try_mark_run_running", new_callable=AsyncMock, return_value=True),
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


class TestRecoverOrphanedQueue:
    """LocalExecutor restart recovery for queued/orphaned runs (dev mode)."""

    @staticmethod
    def _maker(session: AsyncMock) -> MagicMock:
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=session)
        ctx.__aexit__ = AsyncMock(return_value=False)
        return MagicMock(return_value=ctx)

    @pytest.mark.asyncio
    async def test_errors_orphans_and_dispatches_per_affected_thread(self) -> None:
        executor = LocalExecutor()
        session = AsyncMock()
        affected = MagicMock()
        affected.all.return_value = [("t1",)]
        affected.rowcount = 1  # the orphan-error UPDATE reports a matched run
        session.execute = AsyncMock(return_value=affected)
        session.commit = AsyncMock()

        with (
            patch(
                "aegra_api.services.local_executor._get_session_maker",
                return_value=TestRecoverOrphanedQueue._maker(session),
            ),
            patch.object(executor, "dispatch_next_for_thread", new_callable=AsyncMock) as mock_dispatch,
        ):
            await executor._recover_orphaned_queue()

        mock_dispatch.assert_awaited_once_with("t1")
        assert session.execute.await_count >= 2  # distinct SELECT + orphan-error UPDATE
        session.commit.assert_awaited()

    @pytest.mark.asyncio
    async def test_orphaned_thread_status_reset_to_error(self) -> None:
        # The orphans' finalizes never ran — recovery must reset the thread the way an
        # error finalize would, so it is not left 'busy' forever with no active run.
        executor = LocalExecutor()
        session = AsyncMock()
        statements: list[str] = []

        async def _exec(stmt: object, *a: object, **k: object) -> MagicMock:
            statements.append(str(stmt))
            result = MagicMock()
            result.all.return_value = [("t1",)]
            result.rowcount = 1  # the orphan-error UPDATE matched a run
            return result

        session.execute = _exec
        session.commit = AsyncMock()

        with (
            patch(
                "aegra_api.services.local_executor._get_session_maker",
                return_value=TestRecoverOrphanedQueue._maker(session),
            ),
            patch.object(executor, "dispatch_next_for_thread", new_callable=AsyncMock),
        ):
            await executor._recover_orphaned_queue()

        thread_updates = [s for s in statements if s.startswith("UPDATE thread ")]
        assert len(thread_updates) == 1
        assert "status" in thread_updates[0]

    @pytest.mark.asyncio
    async def test_queued_only_thread_status_untouched(self) -> None:
        # A thread with only parked (queued) runs has no orphan to error; its status —
        # possibly a HITL 'interrupted' pause — must not be overwritten by recovery.
        executor = LocalExecutor()
        session = AsyncMock()
        statements: list[str] = []

        async def _exec(stmt: object, *a: object, **k: object) -> MagicMock:
            statements.append(str(stmt))
            result = MagicMock()
            result.all.return_value = [("t1",)]
            result.rowcount = 0  # no running/pending orphan matched
            return result

        session.execute = _exec
        session.commit = AsyncMock()

        with (
            patch(
                "aegra_api.services.local_executor._get_session_maker",
                return_value=TestRecoverOrphanedQueue._maker(session),
            ),
            patch.object(executor, "dispatch_next_for_thread", new_callable=AsyncMock) as mock_dispatch,
        ):
            await executor._recover_orphaned_queue()

        assert not any(s.startswith("UPDATE thread ") for s in statements)
        mock_dispatch.assert_awaited_once_with("t1")  # queued run still gets promoted

    @pytest.mark.asyncio
    async def test_noop_when_no_affected_threads(self) -> None:
        executor = LocalExecutor()
        session = AsyncMock()
        empty = MagicMock()
        empty.all.return_value = []
        session.execute = AsyncMock(return_value=empty)

        with (
            patch(
                "aegra_api.services.local_executor._get_session_maker",
                return_value=TestRecoverOrphanedQueue._maker(session),
            ),
            patch.object(executor, "dispatch_next_for_thread", new_callable=AsyncMock) as mock_dispatch,
        ):
            await executor._recover_orphaned_queue()

        mock_dispatch.assert_not_awaited()


class TestStrandedQueueSweep:
    """Dev periodic stranded-queue recovery must survive a bad row (mirrors the prod reaper)."""

    @pytest.mark.asyncio
    async def test_sweep_survives_non_sqlalchemy_error(self) -> None:
        # A corrupt-params queued row makes dispatch raise ValueError (not SQLAlchemyError);
        # per-thread isolation must keep the sweep alive and still attempt the other threads.
        executor = LocalExecutor()
        session = AsyncMock()
        threads = MagicMock()
        threads.all.return_value = [("t1",), ("t2",)]
        session.execute = AsyncMock(return_value=threads)

        with (
            patch(
                "aegra_api.services.local_executor._get_session_maker",
                return_value=TestRecoverOrphanedQueue._maker(session),
            ),
            patch.object(
                executor,
                "dispatch_next_for_thread",
                new_callable=AsyncMock,
                side_effect=ValueError("corrupt execution_params"),
            ) as mock_dispatch,
        ):
            await executor._sweep_stranded_queues()  # must NOT raise

        assert mock_dispatch.await_count == 2  # both threads attempted despite the first raising

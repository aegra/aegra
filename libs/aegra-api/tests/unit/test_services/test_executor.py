"""Tests for executor abstraction (local and worker)."""

import asyncio
import contextlib
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.exc import SQLAlchemyError

from aegra_api.core.active_runs import active_runs
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
    async def test_wait_for_completion_raises_timeout_on_slow_run(self) -> None:
        """Callers distinguish a slow run from a finished one, as WorkerExecutor already allows."""
        from aegra_api.core.active_runs import active_runs

        executor = LocalExecutor()
        task = asyncio.create_task(asyncio.sleep(9999))
        active_runs["run-slow"] = task
        try:
            with pytest.raises(TimeoutError):
                await executor.wait_for_completion("run-slow", timeout=0.01)
        finally:
            task.cancel()
            active_runs.pop("run-slow", None)

    @pytest.mark.asyncio
    async def test_wait_for_completion_leaves_a_timed_out_run_running(self) -> None:
        """The wait is shielded: giving up on a run must not cancel it."""
        from aegra_api.core.active_runs import active_runs

        executor = LocalExecutor()
        task = asyncio.create_task(asyncio.sleep(9999))
        active_runs["run-shielded"] = task
        try:
            with pytest.raises(TimeoutError):
                await executor.wait_for_completion("run-shielded", timeout=0.01)
            assert not task.done()
        finally:
            task.cancel()
            active_runs.pop("run-shielded", None)

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
            patch("aegra_api.services.run_executor.start_run", new_callable=AsyncMock, return_value=True),
            patch("aegra_api.services.run_executor.finalize_run", new_callable=AsyncMock, return_value=True),
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

    @staticmethod
    def _recording_execute(
        statements: list[Any], *, preempted: list[tuple[str, str, str]], orphan_rowcount: int
    ) -> Any:
        """session.execute stand-in: the scan lists thread t1, the stamped-orphan UPDATE returns
        ``preempted``, and the orphan-error UPDATE reports ``orphan_rowcount`` matched rows."""

        async def _exec(stmt: object, *a: object, **k: object) -> MagicMock:
            statements.append(stmt)
            result = MagicMock()
            ends_preempted = "'interrupted'" in str(stmt.compile(compile_kwargs={"literal_binds": True}))  # type: ignore[attr-defined]
            result.all.return_value = preempted if ends_preempted else [("t1",)]
            result.rowcount = orphan_rowcount
            return result

        return _exec

    async def _recover(
        self, *, preempted: list[tuple[str, str, str]], orphan_rowcount: int
    ) -> tuple[list[Any], AsyncMock]:
        executor = LocalExecutor()
        session = AsyncMock()
        statements: list[Any] = []
        session.execute = self._recording_execute(statements, preempted=preempted, orphan_rowcount=orphan_rowcount)
        session.commit = AsyncMock()

        with (
            patch("aegra_api.services.local_executor._get_session_maker", return_value=self._maker(session)),
            patch.object(executor, "dispatch_next_for_thread", new_callable=AsyncMock) as mock_dispatch,
        ):
            await executor._recover_orphaned_queue()
        return statements, mock_dispatch

    @staticmethod
    def _thread_updates(statements: list[Any]) -> list[Any]:
        return [s for s in statements if str(s).startswith("UPDATE thread ")]

    @pytest.mark.asyncio
    async def test_errors_orphans_and_dispatches_per_affected_thread(self) -> None:
        statements, mock_dispatch = await self._recover(preempted=[], orphan_rowcount=1)

        mock_dispatch.assert_awaited_once_with("t1")
        assert len(statements) >= 3  # distinct SELECT + the two orphan UPDATEs

    @pytest.mark.asyncio
    async def test_orphaned_thread_status_reset_to_error(self) -> None:
        # The orphans' finalizes never ran — recovery must reset the thread the way an
        # error finalize would, so it is not left 'busy' forever with no active run.
        statements, _ = await self._recover(preempted=[], orphan_rowcount=1)

        thread_updates = self._thread_updates(statements)
        assert len(thread_updates) == 1
        assert thread_updates[0].compile().params["status"] == "error"

    @pytest.mark.asyncio
    async def test_preempted_orphan_ends_interrupted_and_thread_idle(self) -> None:
        # Worker mode (reaper, drain) ends a pre-empted run interrupted; a dev restart must agree
        # instead of reporting it as an error with the thread in error.
        statements, _ = await self._recover(preempted=[("run-1", "t1", "user-1")], orphan_rowcount=0)

        run_updates = [s for s in statements if str(s).startswith("UPDATE runs ")]
        assert len(run_updates) == 2
        assert run_updates[0].compile().params["status"] == "interrupted"
        assert "_interrupt_requested_at" in str(run_updates[0].compile(compile_kwargs={"literal_binds": True}))
        thread_updates = self._thread_updates(statements)
        assert len(thread_updates) == 1
        assert thread_updates[0].compile().params["status"] == "idle"

    @pytest.mark.asyncio
    async def test_start_records_the_start_time_and_the_recovery_outcome(self) -> None:
        executor = LocalExecutor()
        with patch.object(executor, "_recover_orphaned_queue", new_callable=AsyncMock, return_value=False) as recover:
            await executor.start()
            try:
                assert executor._started_at is not None and executor._started_at.tzinfo is UTC
                assert executor._orphans_recovered is False  # the sweep will retry
                recover.assert_awaited_once()
                assert not executor._sweep_task.done()
            finally:
                executor._sweep_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await executor._sweep_task

    @pytest.mark.asyncio
    async def test_thread_failure_reports_unrecovered(self) -> None:
        # The scan succeeded but a thread's recovery failed: the pass does not count as done.
        executor = LocalExecutor()
        session = AsyncMock()
        calls = {"n": 0}

        async def _exec(stmt: object, *a: object, **k: object) -> MagicMock:
            calls["n"] += 1
            if calls["n"] == 1:
                scan = MagicMock()
                scan.all.return_value = [("t1",)]
                return scan
            raise SQLAlchemyError("lock timeout")

        session.execute = _exec
        session.commit = AsyncMock()

        with (
            patch("aegra_api.services.local_executor._get_session_maker", return_value=self._maker(session)),
            patch.object(executor, "dispatch_next_for_thread", new_callable=AsyncMock) as mock_dispatch,
        ):
            assert await executor._recover_orphaned_queue() is False

        mock_dispatch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_scan_failure_reports_unrecovered(self) -> None:
        # start() records the result so the periodic sweep knows to retry.
        executor = LocalExecutor()
        session = AsyncMock()
        session.execute = AsyncMock(side_effect=SQLAlchemyError("db down"))

        with patch(
            "aegra_api.services.local_executor._get_session_maker",
            return_value=TestRecoverOrphanedQueue._maker(session),
        ):
            assert await executor._recover_orphaned_queue() is False

    @pytest.mark.asyncio
    async def test_successful_pass_reports_recovered(self) -> None:
        executor = LocalExecutor()
        session = AsyncMock()
        empty = MagicMock()
        empty.all.return_value = []
        session.execute = AsyncMock(return_value=empty)

        with patch(
            "aegra_api.services.local_executor._get_session_maker",
            return_value=TestRecoverOrphanedQueue._maker(session),
        ):
            assert await executor._recover_orphaned_queue() is True

    @pytest.mark.asyncio
    async def test_retry_spares_runs_of_this_process(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A retry runs while this process has its own runs: only rows last written before it
        # started, and with no task here, count as orphans.
        executor = LocalExecutor()
        executor._started_at = datetime(2026, 10, 27, tzinfo=UTC)
        live = MagicMock()
        live.done.return_value = False
        monkeypatch.setitem(active_runs, "run-live", live)
        session = AsyncMock()
        statements: list[Any] = []
        session.execute = self._recording_execute(statements, preempted=[], orphan_rowcount=1)
        session.commit = AsyncMock()

        with (
            patch("aegra_api.services.local_executor._get_session_maker", return_value=self._maker(session)),
            patch.object(executor, "dispatch_next_for_thread", new_callable=AsyncMock),
        ):
            assert await executor._recover_orphaned_queue() is True

        errored = [
            s
            for s in statements
            if str(s).startswith("UPDATE runs ") and "'error'" in str(s.compile(compile_kwargs={"literal_binds": True}))
        ]
        assert len(errored) == 1
        sql = str(errored[0].compile(compile_kwargs={"literal_binds": True}))
        assert "runs.updated_at < '2026-10-27 00:00:00+00:00'" in sql
        assert "runs.run_id NOT IN ('run-live')" in sql
        # The same filters guard the scan (plus parked runs) and the stamped-orphan UPDATE.
        scan = str(statements[0].compile(compile_kwargs={"literal_binds": True}))
        assert "runs.updated_at < '2026-10-27 00:00:00+00:00'" in scan and "NOT IN ('run-live')" in scan
        assert "runs.status = 'queued'" in scan
        stamped = [
            s
            for s in statements
            if str(s).startswith("UPDATE runs ")
            and "'interrupted'" in str(s.compile(compile_kwargs={"literal_binds": True}))
        ]
        assert len(stamped) == 1 and "NOT IN ('run-live')" in str(
            stamped[0].compile(compile_kwargs={"literal_binds": True})
        )
        # The thread row is locked before any run row, the order the admission gate uses.
        first_lock = next(i for i, s in enumerate(statements) if "FOR UPDATE" in str(s))
        first_run_update = next(i for i, s in enumerate(statements) if str(s).startswith("UPDATE runs "))
        assert "FROM thread" in str(statements[first_lock]) and first_lock < first_run_update

    @pytest.mark.asyncio
    async def test_queued_only_thread_status_untouched(self) -> None:
        # A thread with only parked (queued) runs has no orphan to error; its status —
        # possibly a HITL 'interrupted' pause — must not be overwritten by recovery.
        statements, mock_dispatch = await self._recover(preempted=[], orphan_rowcount=0)

        assert not self._thread_updates(statements)
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

    @pytest.fixture(autouse=True)
    def _no_overdue_preemptions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The sweep first ends pre-empted runs past their timeout (tested in test_run_status).
        monkeypatch.setattr(
            "aegra_api.services.local_executor.finalize_overdue_preemptions", AsyncMock(return_value=[])
        )

    @pytest.mark.asyncio
    async def test_failed_startup_recovery_is_retried_until_it_succeeds(self) -> None:
        executor = LocalExecutor()
        executor._orphans_recovered = False

        with (
            patch.object(
                executor, "_recover_orphaned_queue", new_callable=AsyncMock, side_effect=[False, True]
            ) as recover,
            patch.object(executor, "_sweep_stranded_queues", new_callable=AsyncMock) as sweep,
        ):
            await executor._sweep_once()
            assert executor._orphans_recovered is False
            await executor._sweep_once()
            assert executor._orphans_recovered is True
            await executor._sweep_once()  # recovered: not attempted again

        assert recover.await_count == 2
        assert sweep.await_count == 3

    @pytest.mark.asyncio
    async def test_overdue_local_task_is_cancelled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The sweep ended the run's row; the in-process task that ignored its cancel is told again.
        executor = LocalExecutor()
        session = AsyncMock()
        threads = MagicMock()
        threads.all.return_value = []
        session.execute = AsyncMock(return_value=threads)
        task = MagicMock()
        task.done.return_value = False
        monkeypatch.setitem(active_runs, "run-overdue", task)
        monkeypatch.setattr(
            "aegra_api.services.local_executor.finalize_overdue_preemptions",
            AsyncMock(return_value=[("run-overdue", "t1")]),
        )

        with patch(
            "aegra_api.services.local_executor._get_session_maker",
            return_value=TestRecoverOrphanedQueue._maker(session),
        ):
            await executor._sweep_stranded_queues()

        task.cancel.assert_called_once()

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

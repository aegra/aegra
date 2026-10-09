"""Unit tests for double-texting (multitask) strategy handling.

Covers the admission gate, queued-run dispatch and the run_status guards the gate relies on.
"""

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from sqlalchemy.dialects import postgresql

from aegra_api.core.active_runs import active_runs
from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import Thread as ThreadORM
from aegra_api.models.auth import User
from aegra_api.services import run_preparation as run_preparation_mod
from aegra_api.services.base_executor import BaseExecutor
from aegra_api.services.run_executor import _resolve_rollback_base, _rollback_fork_base
from aegra_api.services.run_preparation import (
    _apply_multitask_strategy,
    _validate_resume_command,
    update_thread_metadata,
)
from aegra_api.services.run_status import cancel_queued_run, finalize_run, interrupt_unowned_run, start_run
from aegra_api.settings import settings

_USER = User(identity="test-user")


def _fake_run(run_id: str = "run-1", status: str = "running", *, live: bool = False) -> MagicMock:
    """A run row as dev mode stores it (never claimed); ``live`` models a worker holding a valid lease."""
    run = MagicMock()
    run.run_id = run_id
    run.status = status
    run.claimed_by = "worker-1" if live else None
    run.lease_expires_at = datetime.now(UTC) + timedelta(minutes=5) if live else None
    return run


def _session_with_active(active: list[MagicMock]) -> AsyncMock:
    """Mock session whose active-run query returns ``active``."""
    session = AsyncMock()
    result = MagicMock()
    result.all.return_value = active
    session.scalars = AsyncMock(return_value=result)
    session.scalar = AsyncMock(return_value=None)
    session.execute = AsyncMock()
    session.delete = AsyncMock()
    return session


def _make_session_maker(session: AsyncMock) -> MagicMock:
    maker = MagicMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    maker.return_value = ctx
    return maker


class TestApplyMultitaskStrategy:
    @pytest.mark.asyncio
    async def test_no_active_run_runs_immediately(self) -> None:
        session = _session_with_active([])

        should_run, cancel_ids, target = await _apply_multitask_strategy(session, "thread-1", "enqueue", _USER)

        assert should_run is True
        assert cancel_ids == []
        assert target is None
        session.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_reject_with_active_raises_409(self) -> None:
        session = _session_with_active([_fake_run()])

        with pytest.raises(HTTPException) as exc:
            await _apply_multitask_strategy(session, "thread-1", "reject", _USER)

        assert exc.value.status_code == 409

    @pytest.mark.asyncio
    async def test_enqueue_with_active_queues(self) -> None:
        session = _session_with_active([_fake_run()])

        should_run, cancel_ids, target = await _apply_multitask_strategy(session, "thread-1", "enqueue", _USER)

        assert should_run is False
        assert cancel_ids == []
        assert target is None
        session.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_interrupt_of_live_run_parks_behind_it(self) -> None:
        active = _fake_run(status="running", live=True)
        session = _session_with_active([active])

        should_run, cancel_ids, target = await _apply_multitask_strategy(session, "thread-1", "interrupt", _USER)

        assert should_run is False  # parked until the executing run's task has exited
        assert cancel_ids == ["run-1"]  # caller cancels post-commit to avoid deadlock
        assert active.status == "running"  # its own finalize writes interrupted once it has stopped
        assert active.claimed_by == "worker-1"  # the lease stays with the worker still executing it
        assert target is None  # interrupt does not revert checkpoints
        session.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_interrupt_of_running_run_without_executor_starts_immediately(self) -> None:
        # Dev mode with no task in this process: nothing can still write for the run, so it
        # is interrupted here and the new run need not wait.
        active = _fake_run(status="running")
        session = _session_with_active([active])

        should_run, cancel_ids, target = await _apply_multitask_strategy(session, "thread-1", "interrupt", _USER)

        assert should_run is True
        assert cancel_ids == ["run-1"]  # still signalled, to close any attached stream
        assert active.status == "interrupted"
        assert target is None

    @pytest.mark.asyncio
    async def test_rollback_marks_target_and_parks_without_delete(self) -> None:
        active = _fake_run(status="running", live=True)
        session = _session_with_active([active])

        should_run, cancel_ids, target = await _apply_multitask_strategy(session, "thread-1", "rollback", _USER)

        assert should_run is False  # parked: the fork waits for the target's task to stop writing
        assert cancel_ids == ["run-1"]
        assert target == "run-1"  # worker forks the new run from before this run
        assert active.status == "running"
        session.delete.assert_not_called()  # rollback reverts via fork, never deletes

    @pytest.mark.asyncio
    async def test_rollback_on_idle_thread_runs_fresh_without_target(self) -> None:
        # Like LangGraph Platform, rollback only reverts an in-flight run: an idle thread's last
        # turn is left alone even when it ended interrupted or errored.
        session = _session_with_active([])

        should_run, cancel_ids, target = await _apply_multitask_strategy(session, "thread-1", "rollback", _USER)

        assert should_run is True
        assert cancel_ids == []
        assert target is None
        session.scalar.assert_not_called()  # no lookup of prior terminal runs

    @pytest.mark.asyncio
    async def test_queued_run_is_dropped_on_interrupt(self) -> None:
        # interrupt/rollback abandons in-flight work: a parked 'queued' double-text is dropped
        # from the queue (marked interrupted); it has no task, but its stream still gets closed.
        queued = _fake_run(run_id="queued-1", status="queued")
        session = _session_with_active([queued])

        should_run, cancel_ids, target = await _apply_multitask_strategy(session, "thread-1", "interrupt", _USER)

        assert should_run is True
        assert cancel_ids == ["queued-1"]
        assert queued.status == "interrupted"

    @pytest.mark.asyncio
    async def test_interrupt_drops_queued_behind_active(self) -> None:
        running = _fake_run(run_id="r1", status="running", live=True)
        queued = _fake_run(run_id="q1", status="queued")
        session = _session_with_active([running, queued])

        should_run, cancel_ids, target = await _apply_multitask_strategy(session, "thread-1", "interrupt", _USER)

        assert should_run is False  # r1 is executing: wait for it
        assert cancel_ids == ["r1", "q1"]  # r1's task is stopped; q1 only has a stream to close
        assert running.status == "running"  # still occupies the thread until its task exits
        assert queued.status == "interrupted"  # the stale double-text is dropped, not run later

    @pytest.mark.asyncio
    async def test_rollback_targets_running_not_queued(self) -> None:
        running = _fake_run(run_id="r1", status="running", live=True)
        queued = _fake_run(run_id="q1", status="queued")
        session = _session_with_active([running, queued])

        _should_run, cancel_ids, target = await _apply_multitask_strategy(session, "thread-1", "rollback", _USER)

        assert cancel_ids == ["r1", "q1"]
        assert target == "r1"  # revert the run that executed, never a queued one
        assert queued.status == "interrupted"

    @pytest.mark.asyncio
    async def test_interrupt_of_pending_run_starts_immediately(self) -> None:
        # A pre-empted run that has not started never will (its start CAS fails once this
        # commits), so there is nothing to wait for and no task to cancel.
        pending = _fake_run(run_id="p1", status="pending")
        session = _session_with_active([pending])

        should_run, cancel_ids, target = await _apply_multitask_strategy(session, "thread-1", "interrupt", _USER)

        assert should_run is True
        assert cancel_ids == ["p1"]  # nothing to stop, but a client streaming it must be released
        assert target is None
        assert pending.status == "interrupted"
        assert pending.claimed_by is None

    @pytest.mark.asyncio
    async def test_rollback_waits_when_any_preempted_run_is_executing(self) -> None:
        running = _fake_run(run_id="r1", status="running", live=True)
        pending = _fake_run(run_id="p1", status="pending")
        session = _session_with_active([running, pending])

        should_run, cancel_ids, target = await _apply_multitask_strategy(session, "thread-1", "rollback", _USER)

        assert should_run is False
        assert cancel_ids == ["r1", "p1"]  # p1 cannot start (its start CAS fails); its stream is closed
        assert target == "r1"
        assert pending.status == "interrupted"

    @pytest.mark.asyncio
    async def test_resume_runs_immediately_jumping_queue(self) -> None:
        # A resume must run NOW even with a run queued ahead — it is the only run that can
        # clear a HITL pause. The queued run is left parked, promoted after the resume ends.
        queued = _fake_run(run_id="q1", status="queued")
        session = _session_with_active([queued])

        should_run, cancel_ids, target = await _apply_multitask_strategy(
            session, "thread-1", "enqueue", _USER, is_resume=True
        )

        assert should_run is True
        assert cancel_ids == []
        assert target is None
        assert queued.status == "queued"  # untouched — stays parked, not dropped

    @pytest.mark.asyncio
    async def test_resume_rejected_when_a_run_is_active(self) -> None:
        # Two concurrent resumes must not double-execute: the second, unblocking after the first
        # commits its pending run, sees that running/pending run under the lock and is 409'd.
        running = _fake_run(run_id="r1", status="running")
        session = _session_with_active([running])

        with pytest.raises(HTTPException) as exc:
            await _apply_multitask_strategy(session, "thread-1", "enqueue", _USER, is_resume=True)

        assert exc.value.status_code == 409


class TestUpdateThreadMetadataOwnership:
    """The route checks ownership on an earlier snapshot; the read that decides create-vs-merge re-checks it."""

    @pytest.mark.asyncio
    async def test_existing_thread_of_another_user_is_404(self) -> None:
        # TOCTOU: the thread did not exist when the route looked, someone else created it since.
        session = AsyncMock()
        session.scalar = AsyncMock(return_value=MagicMock(user_id="someone-else"))

        with pytest.raises(HTTPException) as exc:
            await update_thread_metadata(session, "thread-1", "asst", "graph", user_id="me")

        assert exc.value.status_code == 404
        session.execute.assert_not_awaited()  # no metadata merged into a thread we do not own

    @pytest.mark.asyncio
    async def test_own_existing_thread_is_merged(self) -> None:
        session = AsyncMock()
        session.scalar = AsyncMock(return_value=MagicMock(user_id="me"))

        await update_thread_metadata(session, "thread-1", "asst", "graph", user_id="me")

        session.execute.assert_awaited_once()  # the jsonb merge UPDATE


class _RecordingExecutor(BaseExecutor):
    """Concrete executor that records submitted jobs."""

    def __init__(self) -> None:
        self.submitted: list[object] = []

    async def submit(self, job: object) -> None:
        self.submitted.append(job)

    async def wait_for_completion(self, run_id: str, *, timeout: float = 300.0) -> None:
        return None

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None


def _session_with_mutable_runs(runs: list[MagicMock]) -> AsyncMock:
    """Apply the query's status filter to rows mutated by earlier admission calls."""
    session = AsyncMock()

    def matching_runs(stmt: Any) -> list[MagicMock]:
        params = stmt.compile(dialect=postgresql.dialect()).params
        statuses = params["status_1"]
        if isinstance(statuses, str):
            statuses = [statuses]
        return [run for run in runs if run.status in statuses]

    async def scalars(stmt: Any) -> MagicMock:
        result = MagicMock()
        result.all.return_value = matching_runs(stmt)
        return result

    async def scalar(stmt: Any) -> object:
        column = stmt.column_descriptions[0]
        if column["entity"] is ThreadORM:
            return MagicMock(status="busy")
        assert column["entity"] is RunORM
        matching = matching_runs(stmt)
        if not matching:
            return None
        return matching[0].run_id if column["name"] == "run_id" else matching[0]

    session.scalars.side_effect = scalars
    session.scalar.side_effect = scalar
    return session


@pytest.fixture
async def running_task_with_delayed_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[asyncio.Task[None], asyncio.Event]]:
    started = asyncio.Event()
    cancelling = asyncio.Event()
    release = asyncio.Event()

    async def execute() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelling.set()
            await release.wait()
            raise

    task = asyncio.create_task(execute())
    monkeypatch.setitem(active_runs, "run-a", task)
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        yield task, cancelling
    finally:
        release.set()
        if not task.done() and not task.cancelling():
            task.cancel()
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=1)


class TestPreemptionWhileCancellationInProgress:
    @pytest.mark.parametrize("strategy", ["interrupt", "rollback"])
    @pytest.mark.asyncio
    async def test_third_request_waits_until_preempted_task_exits(
        self,
        strategy: str,
        running_task_with_delayed_cancellation: tuple[asyncio.Task[None], asyncio.Event],
    ) -> None:
        task, cancelling = running_task_with_delayed_cancellation
        original = _fake_run("run-a", "running")
        original.execution_params = {}
        runs = [original]
        session = _session_with_mutable_runs(runs)

        should_run, cancel_ids, _ = await _apply_multitask_strategy(session, "thread-1", strategy, _USER)
        assert should_run is False
        assert cancel_ids == ["run-a"]
        runs.append(_fake_run("run-b", "queued"))
        await session.commit()
        task.cancel()
        await asyncio.wait_for(cancelling.wait(), timeout=1)

        should_run, _, _ = await _apply_multitask_strategy(session, "thread-1", strategy, _USER)

        assert not task.done()
        assert should_run is False, "Run C must wait while pre-empted run A is still exiting"

    @pytest.mark.parametrize("strategy", ["interrupt", "rollback"])
    @pytest.mark.asyncio
    async def test_queue_promotion_waits_until_preempted_task_exits(
        self,
        strategy: str,
        running_task_with_delayed_cancellation: tuple[asyncio.Task[None], asyncio.Event],
    ) -> None:
        task, cancelling = running_task_with_delayed_cancellation
        original = _fake_run("run-a", "running")
        original.execution_params = {}
        runs = [original]
        session = _session_with_mutable_runs(runs)
        executor = _RecordingExecutor()
        replacement = _fake_run("run-b", "queued")
        fake_job = MagicMock()
        fake_job.identity.run_id = "run-b"

        should_run, cancel_ids, _ = await _apply_multitask_strategy(session, "thread-1", strategy, _USER)
        assert should_run is False
        assert cancel_ids == ["run-a"]
        runs.append(replacement)
        await session.commit()
        task.cancel()
        await asyncio.wait_for(cancelling.wait(), timeout=1)

        with (
            patch("aegra_api.services.base_executor._get_session_maker", return_value=_make_session_maker(session)),
            patch("aegra_api.services.base_executor.RunJob.from_run_orm", return_value=fake_job),
        ):
            await executor.dispatch_next_for_thread("thread-1")

        assert not task.done()
        assert executor.submitted == [], "Recovery must not dispatch run B while pre-empted run A is still exiting"
        assert replacement.status == "queued"


class TestDispatchNextForThread:
    @pytest.mark.asyncio
    async def test_noop_when_not_accepting(self) -> None:
        ex = _RecordingExecutor()
        ex._accepting = False

        await ex.dispatch_next_for_thread("thread-1")

        assert ex.submitted == []

    @pytest.mark.asyncio
    async def test_noop_when_thread_interrupted_under_reject(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # side_effect is padded with the occupying + queued lookups so that removing the HITL
        # guard fails on `submitted == []`, not on StopAsyncIteration.
        monkeypatch.setattr(settings.multitask, "MULTITASK_PAUSED_THREAD_POLICY", "reject")
        ex = _RecordingExecutor()
        queued = _fake_run(run_id="queued-1", status="queued")
        session = AsyncMock()
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        session.scalar = AsyncMock(side_effect=[MagicMock(status="interrupted"), None, queued])

        fake_job = MagicMock()
        fake_job.identity.run_id = "queued-1"
        with (
            patch("aegra_api.services.base_executor._get_session_maker", return_value=_make_session_maker(session)),
            patch("aegra_api.services.base_executor.RunJob.from_run_orm", return_value=fake_job),
        ):
            await ex.dispatch_next_for_thread("thread-1")

        assert ex.submitted == []

    @pytest.mark.asyncio
    async def test_noop_when_thread_deleted(self) -> None:
        # The thread (and, by cascade, its runs) is gone: nothing to promote onto. Must stop
        # at the thread read — a delete racing a finalize relies on this.
        ex = _RecordingExecutor()
        session = AsyncMock()
        session.scalar = AsyncMock(side_effect=[None])

        with patch("aegra_api.services.base_executor._get_session_maker", return_value=_make_session_maker(session)):
            await ex.dispatch_next_for_thread("thread-1")

        assert ex.submitted == []
        assert session.scalar.await_count == 1

    @pytest.mark.asyncio
    async def test_corrupt_queued_row_is_failed_and_the_next_one_promoted(self) -> None:
        # A row whose execution_params cannot be rebuilt (KeyError, not just ValueError) is
        # marked error and the run parked behind it is promoted in the same dispatch.
        ex = _RecordingExecutor()
        corrupt = _fake_run(run_id="bad", status="queued")
        good = _fake_run(run_id="good", status="queued")
        session = AsyncMock()
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        # attempt 1: thread, no occupying run, corrupt row; attempt 2: thread, no occupying, good row
        session.scalar = AsyncMock(
            side_effect=[MagicMock(status="busy"), None, corrupt, MagicMock(status="busy"), None, good]
        )

        fake_job = MagicMock()
        fake_job.identity.run_id = "good"
        with (
            patch("aegra_api.services.base_executor._get_session_maker", return_value=_make_session_maker(session)),
            patch("aegra_api.services.base_executor.RunJob.from_run_orm", side_effect=[KeyError("graph_id"), fake_job]),
        ):
            await ex.dispatch_next_for_thread("thread-1")

        assert corrupt.status == "error"
        assert "graph_id" in corrupt.error_message
        assert good.status == "pending"
        assert ex.submitted == [fake_job]

    @pytest.mark.asyncio
    async def test_noop_when_thread_occupied(self) -> None:
        ex = _RecordingExecutor()
        session = AsyncMock()
        session.execute = AsyncMock()
        # thread (not interrupted), then the occupying check hits.
        session.scalar = AsyncMock(side_effect=[MagicMock(status="busy"), "some-active-run-id"])

        with patch("aegra_api.services.base_executor._get_session_maker", return_value=_make_session_maker(session)):
            await ex.dispatch_next_for_thread("thread-1")

        assert ex.submitted == []

    @pytest.mark.asyncio
    async def test_noop_when_no_queued(self) -> None:
        ex = _RecordingExecutor()
        session = AsyncMock()
        session.execute = AsyncMock()
        # thread (not interrupted), not occupied, nothing queued.
        session.scalar = AsyncMock(side_effect=[MagicMock(status="busy"), None, None])

        with patch("aegra_api.services.base_executor._get_session_maker", return_value=_make_session_maker(session)):
            await ex.dispatch_next_for_thread("thread-1")

        assert ex.submitted == []

    @pytest.mark.asyncio
    async def test_promotes_and_submits_oldest_queued(self) -> None:
        ex = _RecordingExecutor()
        queued = _fake_run(run_id="queued-1", status="queued")
        session = AsyncMock()
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        # thread (not interrupted), not occupied, one queued.
        session.scalar = AsyncMock(side_effect=[MagicMock(status="busy"), None, queued])

        fake_job = MagicMock()
        fake_job.identity.run_id = "queued-1"
        with (
            patch("aegra_api.services.base_executor._get_session_maker", return_value=_make_session_maker(session)),
            patch("aegra_api.services.base_executor.RunJob.from_run_orm", return_value=fake_job),
        ):
            await ex.dispatch_next_for_thread("thread-1")

        assert queued.status == "pending"
        assert isinstance(queued.updated_at, datetime)  # stamped at promotion (stuck-pending reaper)
        assert ex.submitted == [fake_job]


class TestDispatchPausedThreadPolicy:
    @pytest.mark.asyncio
    async def test_default_admit_policy_promotes_onto_paused_thread(self) -> None:
        # Under the default MULTITASK_PAUSED_THREAD_POLICY=admit fresh input is admitted onto a
        # HITL pause at creation, so promotion must not hold parked runs back either.
        assert settings.multitask.MULTITASK_PAUSED_THREAD_POLICY == "admit"
        ex = _RecordingExecutor()
        queued = _fake_run(run_id="queued-1", status="queued")
        session = AsyncMock()
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        session.scalar = AsyncMock(side_effect=[MagicMock(status="interrupted"), None, queued])

        fake_job = MagicMock()
        fake_job.identity.run_id = "queued-1"
        with (
            patch("aegra_api.services.base_executor._get_session_maker", return_value=_make_session_maker(session)),
            patch("aegra_api.services.base_executor.RunJob.from_run_orm", return_value=fake_job),
        ):
            await ex.dispatch_next_for_thread("thread-1")

        assert ex.submitted == [fake_job]


class TestShutdownBarrier:
    """stop() must not begin draining while a promotion is between its commit and submit()."""

    @pytest.mark.asyncio
    async def test_begin_shutdown_waits_for_inflight_dispatch(self) -> None:
        ex = _RecordingExecutor()
        release = asyncio.Event()

        async def _slow_dispatch(thread_id: str) -> None:
            await release.wait()
            ex.submitted.append(thread_id)

        ex._dispatch_next_for_thread = _slow_dispatch  # type: ignore[method-assign]
        dispatch = asyncio.create_task(ex.dispatch_next_for_thread("thread-1"))
        await asyncio.sleep(0)  # past the _accepting check, blocked in the DB work
        shutdown = asyncio.create_task(ex._begin_shutdown())
        await asyncio.sleep(0)

        assert ex._accepting is False
        assert not shutdown.done()  # waits for the in-flight promotion

        release.set()
        await asyncio.wait_for(shutdown, timeout=1)
        await dispatch
        assert ex.submitted == ["thread-1"]  # the promotion completed before the drain

    @pytest.mark.asyncio
    async def test_dispatch_after_shutdown_is_a_noop(self) -> None:
        ex = _RecordingExecutor()
        await ex._begin_shutdown()
        ex._dispatch_next_for_thread = AsyncMock()  # type: ignore[method-assign]

        await ex.dispatch_next_for_thread("thread-1")

        ex._dispatch_next_for_thread.assert_not_awaited()


class _FakeSnap:
    """Minimal CheckpointTuple stand-in for rollback-base resolution."""

    def __init__(self, run_id: str | None, checkpoint_id: str, parent_checkpoint_id: str | None = None) -> None:
        self.metadata = {"run_id": run_id} if run_id is not None else {}
        self.config = {"configurable": {"checkpoint_id": checkpoint_id}}
        self.parent_config = (
            {"configurable": {"checkpoint_id": parent_checkpoint_id}} if parent_checkpoint_id is not None else None
        )


def _history(*snaps: _FakeSnap) -> MagicMock:
    """A graph whose checkpointer lists ``snaps`` (newest first) honouring the metadata ``filter``."""

    async def _alist(config: object, *, filter: dict[str, Any] | None = None) -> AsyncIterator[_FakeSnap]:
        for s in snaps:
            if filter is None or all(s.metadata.get(k) == v for k, v in filter.items()):
                yield s

    graph = MagicMock()
    graph.checkpointer.alist = _alist
    return graph


class TestResolveRollbackBase:
    """Worker-side resolution of the pre-target checkpoint to fork from (by lineage)."""

    @staticmethod
    async def _resolve(graph: MagicMock, target: str) -> str | None:
        with patch("aegra_api.services.run_executor.create_thread_config", return_value={"configurable": {}}):
            return await _resolve_rollback_base(graph, "t", User(identity="u"), target)

    @pytest.mark.asyncio
    async def test_returns_parent_of_oldest_target_checkpoint(self) -> None:
        # target wrote cp-3 (parent cp-2) and cp-2 (input, parent cp-1=base); cp-1 is a prior run.
        graph = _history(
            _FakeSnap("target", "cp-3", parent_checkpoint_id="cp-2"),
            _FakeSnap("target", "cp-2", parent_checkpoint_id="cp-1"),
            _FakeSnap("prev", "cp-1", parent_checkpoint_id="cp-0"),
        )
        assert await self._resolve(graph, "target") == "cp-1"

    @pytest.mark.asyncio
    async def test_filters_by_target_run_id_without_a_window(self) -> None:
        # A long thread: the target's rows sit behind thousands of newer checkpoints. The
        # checkpointer filter finds them directly, so no scan cap can make the rollback fail.
        newer = [_FakeSnap("later", f"cp-n{i}", parent_checkpoint_id=f"cp-n{i - 1}") for i in range(2000, 0, -1)]
        graph = _history(
            *newer,
            _FakeSnap("target", "cp-2", parent_checkpoint_id="cp-1"),
            _FakeSnap("prev", "cp-1", parent_checkpoint_id="cp-0"),
        )
        assert await self._resolve(graph, "target") == "cp-1"

    @pytest.mark.asyncio
    async def test_second_rollback_anchors_to_lineage_not_sibling(self) -> None:
        # P -> A(rolled back) -> B(forked from cp-P): history DESC interleaves the abandoned A
        # branch, and a second rollback targeting B must fork from cp-P (B's parent), not cp-A.
        graph = _history(
            _FakeSnap("B", "cp-B", parent_checkpoint_id="cp-P"),
            _FakeSnap("A", "cp-A", parent_checkpoint_id="cp-P"),
            _FakeSnap("P", "cp-P", parent_checkpoint_id=None),
        )
        assert await self._resolve(graph, "B") == "cp-P"

    @pytest.mark.asyncio
    async def test_retry_skips_own_partial_checkpoints(self) -> None:
        # Retry of crashed run B (target still A): B's own newer partial must be skipped;
        # base is A's oldest checkpoint's parent (cp-P), never B's partial.
        graph = _history(
            _FakeSnap("B", "cp-Bpart", parent_checkpoint_id="cp-P"),  # newer, current run's own partial
            _FakeSnap("A", "cp-A2", parent_checkpoint_id="cp-A1"),
            _FakeSnap("A", "cp-A1", parent_checkpoint_id="cp-P"),
            _FakeSnap("P", "cp-P", parent_checkpoint_id=None),
        )
        assert await self._resolve(graph, "A") == "cp-P"

    @pytest.mark.asyncio
    async def test_first_run_target_has_no_parent(self) -> None:
        # Target was the thread's first run: its oldest checkpoint has no parent -> None.
        graph = _history(
            _FakeSnap("target", "cp-1", parent_checkpoint_id="cp-0"),
            _FakeSnap("target", "cp-0", parent_checkpoint_id=None),
        )
        assert await self._resolve(graph, "target") is None

    @pytest.mark.asyncio
    async def test_returns_none_when_target_has_no_checkpoints(self) -> None:
        graph = _history(_FakeSnap("other", "cp-1", parent_checkpoint_id=None))
        assert await self._resolve(graph, "target") is None

    @pytest.mark.asyncio
    async def test_returns_none_on_empty_history(self) -> None:
        assert await self._resolve(_history(), "target") is None

    @pytest.mark.asyncio
    async def test_ignores_straggler_from_cancelled_target(self) -> None:
        # The cancelled target writes a late 'straggler' checkpoint after the new run forked;
        # the base must still be the target's original oldest parent (cp-P), not the straggler's.
        graph = _history(
            _FakeSnap("A", "cp-straggler", parent_checkpoint_id="cp-Bnew"),  # newest: A's late write
            _FakeSnap("B", "cp-Bnew", parent_checkpoint_id="cp-P"),  # the new run's own checkpoint
            _FakeSnap("A", "cp-A1", parent_checkpoint_id="cp-P"),  # A's original (oldest) checkpoint
            _FakeSnap("P", "cp-P", parent_checkpoint_id=None),
        )
        assert await self._resolve(graph, "A") == "cp-P"


class TestRollbackForkBase:
    """Worker re-reads the target's status so a run that raced to success is never reverted."""

    @staticmethod
    def _job() -> MagicMock:
        job = MagicMock()
        job.execution.rollback_target_run_id = "target"
        job.identity.run_id = "new"
        job.identity.thread_id = "thread"
        job.user = User(identity="u")
        return job

    @pytest.mark.asyncio
    async def test_skips_revert_when_target_succeeded(self) -> None:
        # A non-empty history would resolve to cp-0; base is None only because we skipped.
        graph = _history(_FakeSnap("target", "cp-1", parent_checkpoint_id="cp-0"))
        with patch("aegra_api.services.run_executor.get_run_status", new=AsyncMock(return_value="success")):
            assert await _rollback_fork_base(graph, self._job()) is None

    @pytest.mark.asyncio
    async def test_resolves_base_when_target_not_success(self) -> None:
        graph = _history(
            _FakeSnap("target", "cp-1", parent_checkpoint_id="cp-0"),
            _FakeSnap("target", "cp-0", parent_checkpoint_id="cp-prev"),
        )
        with (
            patch("aegra_api.services.run_executor.get_run_status", new=AsyncMock(return_value="interrupted")),
            patch("aegra_api.services.run_executor.create_thread_config", return_value={"configurable": {}}),
        ):
            assert await _rollback_fork_base(graph, self._job()) == "cp-prev"


def _session_with_thread(status: str | None) -> AsyncMock:
    """Mock session whose thread lookup returns a thread row with ``status`` (None = no thread)."""
    session = AsyncMock()
    thread = MagicMock(status=status) if status is not None else None
    session.scalar = AsyncMock(return_value=thread)
    return session


def _collapse_settle(monkeypatch: pytest.MonkeyPatch, status: str | None) -> None:
    """Zero the resume-settle poll interval; every fresh session sees ``status``."""
    monkeypatch.setattr(run_preparation_mod, "_RESUME_SETTLE_INTERVAL_SECONDS", 0)
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=_session_with_thread(status))
    ctx.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(run_preparation_mod, "_get_session_maker", lambda: MagicMock(return_value=ctx))


class TestValidateResumeCommand:
    """Admission-time guard tying a run's input mode to the thread's interrupt state."""

    @pytest.mark.asyncio
    async def test_fresh_input_on_paused_thread_admitted_by_default(self) -> None:
        # LangGraph Platform admits fresh input on a paused thread (the graph restarts from
        # __start__ and the pending interrupt is discarded); so does the default policy.
        session = _session_with_thread("interrupted")
        await _validate_resume_command(session, "thread-1", None, _USER)  # no raise

    @pytest.mark.asyncio
    async def test_fresh_input_on_paused_thread_rejected_409_under_reject(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Opt-in for approval flows: a stray message must not consume the pending interrupt.
        monkeypatch.setattr(settings.multitask, "MULTITASK_PAUSED_THREAD_POLICY", "reject")
        session = _session_with_thread("interrupted")

        with pytest.raises(HTTPException) as exc:
            await _validate_resume_command(session, "thread-1", None, _USER)

        assert exc.value.status_code == 409

    @pytest.mark.asyncio
    async def test_fresh_input_on_idle_thread_allowed(self) -> None:
        session = _session_with_thread("idle")
        await _validate_resume_command(session, "thread-1", None, _USER)  # no raise

    @pytest.mark.asyncio
    async def test_fresh_input_on_new_thread_allowed(self) -> None:
        session = _session_with_thread(None)  # thread does not exist yet
        await _validate_resume_command(session, "thread-1", None, _USER)  # no raise

    @pytest.mark.asyncio
    async def test_resume_requires_interrupted_thread(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The settle poll re-reads fresh sessions before rejecting; keep them idle here.
        _collapse_settle(monkeypatch, "idle")
        session = _session_with_thread("idle")
        with pytest.raises(HTTPException) as exc:
            await _validate_resume_command(session, "thread-1", {"resume": "go"}, _USER)
        assert exc.value.status_code == 400

    @pytest.mark.asyncio
    async def test_resume_on_paused_thread_allowed(self) -> None:
        session = _session_with_thread("interrupted")
        await _validate_resume_command(session, "thread-1", {"resume": "go"}, _USER)  # no raise

    @pytest.mark.asyncio
    async def test_non_resume_command_skips_interrupt_check(self) -> None:
        # An update/goto command on a paused thread is a deliberate state edit, not a
        # naive fresh run — it is allowed and must not even query the thread.
        session = _session_with_thread("interrupted")
        await _validate_resume_command(session, "thread-1", {"update": {"k": "v"}}, _USER)
        session.scalar.assert_not_called()

    @pytest.mark.asyncio
    async def test_null_resume_command_on_paused_thread_passes_validation(self) -> None:
        # ``None`` is a valid resume value at the API boundary, as on main; the thread must
        # still be paused for it, like any other resume.
        session = _session_with_thread("interrupted")
        await _validate_resume_command(session, "thread-1", {"resume": None}, _USER)  # no raise

    @pytest.mark.parametrize("command", [{"update": {}}, {"goto": []}])
    @pytest.mark.asyncio
    async def test_empty_container_command_on_paused_thread_rejected_under_reject(
        self, command: dict[str, object], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Empty update/goto produce no LangGraph writes and would crash a paused thread to
        # 'error'; truthiness classification routes them to the 409 gate, keeping the pause.
        monkeypatch.setattr(settings.multitask, "MULTITASK_PAUSED_THREAD_POLICY", "reject")
        session = _session_with_thread("interrupted")
        with pytest.raises(HTTPException) as exc:
            await _validate_resume_command(session, "thread-1", command, _USER)
        assert exc.value.status_code == 409

    @pytest.mark.asyncio
    async def test_truthy_update_command_still_allowed(self) -> None:
        session = _session_with_thread("interrupted")
        await _validate_resume_command(session, "thread-1", {"update": {"k": "v"}}, _USER)
        session.scalar.assert_not_called()

    @pytest.mark.asyncio
    async def test_malformed_command_rejected_422(self) -> None:
        # {'goto': [0]} can't be mapped to a LangGraph Command (0['node'] -> TypeError); reject it
        # at admission so it can't bypass the gate and crash a paused thread to 'error'.
        session = _session_with_thread("interrupted")
        with pytest.raises(HTTPException) as exc:
            await _validate_resume_command(session, "thread-1", {"goto": [0]}, _USER)
        assert exc.value.status_code == 422


class TestFinalizeRunGuard:
    """finalize must not resurrect a run a multitask pre-emption already terminated."""

    @pytest.mark.asyncio
    async def test_run_update_is_guarded_by_active_status(self) -> None:
        # The run UPDATE filters status IN (running, pending), so a target the gate moved to
        # 'interrupted' cannot be flipped to 'success' by its own late finalize.
        captured: list[str] = []
        session = AsyncMock()

        async def _exec(stmt: object, *a: object, **k: object) -> MagicMock:
            captured.append(str(stmt))
            return MagicMock()

        session.execute = _exec
        session.commit = AsyncMock()
        with (
            patch("aegra_api.services.run_status._get_session_maker", return_value=_make_session_maker(session)),
            patch("aegra_api.services.run_status.dispatch_next_queued_run", new_callable=AsyncMock),
        ):
            assert await finalize_run("run-1", "thread-1", user_id="u", status="success", thread_status="idle")

        run_update = next(s for s in captured if s.startswith("UPDATE runs"))
        assert "runs.status IN" in run_update

    @pytest.mark.asyncio
    async def test_non_owned_finalize_leaves_thread_alone_but_still_dispatches(self) -> None:
        # No row returned => a pre-emption gate already terminalized the run; finalize must not
        # touch the thread but still dispatches, since this exit is what a parked run waits for.
        executed: list[str] = []
        session = AsyncMock()

        async def _exec(stmt: object, *a: object, **k: object) -> MagicMock:
            text = str(stmt)
            executed.append(text)
            result = MagicMock()
            result.scalar_one_or_none.return_value = None if text.startswith("UPDATE runs") else "row"
            return result

        session.execute = _exec
        session.commit = AsyncMock()
        with (
            patch("aegra_api.services.run_status._get_session_maker", return_value=_make_session_maker(session)),
            patch("aegra_api.services.run_status.dispatch_next_queued_run", new_callable=AsyncMock) as dispatched,
        ):
            finalized = await finalize_run("run-1", "thread-1", user_id="u", status="success", thread_status="idle")

        assert finalized is False
        assert not any(s.startswith("UPDATE thread") for s in executed)  # thread not clobbered (table is singular)
        session.rollback.assert_awaited_once()
        session.commit.assert_not_awaited()
        dispatched.assert_awaited_once_with("thread-1")


def _capturing_session(*, rowcount: int = 1, returning: object = "row") -> tuple[AsyncMock, list[str]]:
    """Mock session whose execute() records compiled statements; ``returning`` feeds
    ``.scalar_one_or_none()`` for the ``UPDATE ... RETURNING`` guards (None = no match)."""
    captured: list[str] = []
    session = AsyncMock()

    async def _exec(stmt: object, *a: object, **k: object) -> MagicMock:
        captured.append(str(stmt.compile(compile_kwargs={"literal_binds": True})))
        result = MagicMock()
        result.rowcount = rowcount
        result.scalar_one_or_none.return_value = returning
        return result

    session.execute = _exec
    session.commit = AsyncMock()
    session.begin_nested = AsyncMock(return_value=AsyncMock())  # interrupt_unowned_run's SAVEPOINT
    return session, captured


class TestStartRunGuard:
    """The pending→running transition is a CAS so a gate-pre-empted run cannot resurrect itself."""

    @pytest.mark.asyncio
    async def test_returns_true_when_run_still_active(self) -> None:
        session, captured = _capturing_session(returning="run-1")
        with patch("aegra_api.services.run_status._get_session_maker", return_value=_make_session_maker(session)):
            assert await start_run("run-1", user_id="u") is True
        # The UPDATE is guarded on the run still being pending/running.
        assert "'pending'" in captured[0] and "'running'" in captured[0]
        session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_returns_false_when_gate_terminalized_the_run(self) -> None:
        session, _ = _capturing_session(returning=None)
        with patch("aegra_api.services.run_status._get_session_maker", return_value=_make_session_maker(session)):
            assert await start_run("run-1", user_id="u") is False
        session.rollback.assert_awaited_once()


class TestCancelQueuedRun:
    """Cancelling a parked run is a guarded flip; a run promoted meanwhile is left to the active path."""

    @pytest.mark.asyncio
    async def test_queued_run_is_dropped_and_queue_dispatched(self) -> None:
        session, captured = _capturing_session(returning="run-1")
        with patch("aegra_api.services.run_status.dispatch_next_queued_run", new_callable=AsyncMock) as dispatched:
            assert await cancel_queued_run(session, "run-1", "thread-1", user_id="u") is True

        assert "'queued'" in captured[0]  # guarded flip, only a still-queued run matches
        assert "user_id = 'u'" in captured[0]  # tenant-scoped like every run write
        assert not any(s.startswith("UPDATE thread") for s in captured)  # a parked run never held the thread
        session.commit.assert_awaited_once()
        # If the dropped run headed a stranded queue, the runs behind it must not
        # wait for the recovery sweep — dispatch follows the drop (idempotent).
        dispatched.assert_awaited_once_with("thread-1")

    @pytest.mark.asyncio
    async def test_run_promoted_meanwhile_is_left_to_the_active_path(self) -> None:
        # The caller read 'queued', but the dispatcher promoted the run in between: nothing
        # matches, and the caller must go on to cancel it as the active run it now is.
        session, _ = _capturing_session(returning=None)
        with patch("aegra_api.services.run_status.dispatch_next_queued_run", new_callable=AsyncMock) as dispatched:
            assert await cancel_queued_run(session, "run-1", "thread-1", user_id="u") is False

        session.commit.assert_not_awaited()
        dispatched.assert_not_awaited()


class TestInterruptUnownedRunDispatch:
    @pytest.mark.asyncio
    async def test_reconciled_cancel_locks_thread_first_and_leaves_dispatch_to_caller(self) -> None:
        # Same lock order as the admission gate (thread, then run), and no promotion here: only
        # the caller knows whether a local task still executes the run (dev never sets claimed_by).
        session, captured = _capturing_session(returning="run-1")
        with patch("aegra_api.services.run_status.dispatch_next_queued_run", new_callable=AsyncMock) as dispatched:
            assert await interrupt_unowned_run(session, "run-1", "thread-1", user_id="u") is True

        assert captured[0].startswith("SELECT thread") and "FOR UPDATE" in captured[0]
        session.commit.assert_awaited_once()
        dispatched.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_live_owned_run_releases_the_thread_lock(self) -> None:
        # With a live worker owning the run nothing is written, so the caller's session must drop
        # the thread lock at once: the worker's finalize takes that lock first.
        session, captured = _capturing_session(returning=None)
        with patch("aegra_api.services.run_status.dispatch_next_queued_run", new_callable=AsyncMock) as dispatched:
            assert await interrupt_unowned_run(session, "run-1", "thread-1", user_id="u") is False

        assert "FOR UPDATE" in captured[0]
        # Released via the savepoint, never session.rollback(): that would expire the caller's
        # loaded rows and a bulk cancel iterating them would 500 on an implicit refresh.
        session.begin_nested.return_value.rollback.assert_awaited_once()
        session.rollback.assert_not_awaited()
        session.commit.assert_not_awaited()
        dispatched.assert_not_awaited()


class TestGatePausedThreadUnderLock:
    """The pre-lock pause check in _validate_resume_command can race the active run pausing;
    under the admission lock the policy must hold again."""

    @staticmethod
    def _paused_session() -> AsyncMock:
        session = _session_with_active([])
        session.scalar = AsyncMock(return_value="interrupted")
        return session

    @pytest.mark.asyncio
    async def test_fresh_input_rejected_under_reject(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings.multitask, "MULTITASK_PAUSED_THREAD_POLICY", "reject")
        with pytest.raises(HTTPException) as exc:
            await _apply_multitask_strategy(self._paused_session(), "thread-1", "enqueue", _USER)
        assert exc.value.status_code == 409

    @pytest.mark.parametrize("kwargs", [{"is_resume": True}, {"may_run_on_pause": True}])
    @pytest.mark.asyncio
    async def test_resume_and_state_ops_still_admitted_under_reject(
        self, kwargs: dict[str, bool], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings.multitask, "MULTITASK_PAUSED_THREAD_POLICY", "reject")
        should_run, _, _ = await _apply_multitask_strategy(
            self._paused_session(), "thread-1", "enqueue", _USER, **kwargs
        )
        assert should_run is True

    @pytest.mark.asyncio
    async def test_fresh_input_admitted_by_default(self) -> None:
        assert settings.multitask.MULTITASK_PAUSED_THREAD_POLICY == "admit"
        should_run, _, _ = await _apply_multitask_strategy(self._paused_session(), "thread-1", "enqueue", _USER)
        assert should_run is True


class TestGatePreemptionOccupancy:
    """A pre-empted run keeps the thread occupied exactly as long as something may execute it."""

    @pytest.mark.asyncio
    async def test_live_worker_run_keeps_row_and_lease(self) -> None:
        active = _fake_run(run_id="run-1", status="running", live=True)
        active.execution_params = {"execution": {"command": None}}
        session = _session_with_active([active])

        should_run, cancel_ids, _ = await _apply_multitask_strategy(session, "thread-1", "interrupt", _USER)

        assert should_run is False  # parked until run-1's task exits
        assert cancel_ids == ["run-1"]  # the stop request goes through the broker
        assert active.status == "running"
        assert active.claimed_by == "worker-1"

    @pytest.mark.asyncio
    async def test_expired_lease_run_is_interrupted_in_place(self) -> None:
        active = _fake_run(run_id="run-1", status="running", live=True)
        active.lease_expires_at = datetime.now(UTC) - timedelta(minutes=1)
        session = _session_with_active([active])

        should_run, cancel_ids, _ = await _apply_multitask_strategy(session, "thread-1", "interrupt", _USER)

        assert should_run is True  # no live owner: the same predicate interrupt_unowned_run uses
        assert cancel_ids == ["run-1"]
        assert active.status == "interrupted"
        assert active.claimed_by is None
        assert active.lease_expires_at is None

    @pytest.mark.asyncio
    async def test_local_task_counts_as_live(self, monkeypatch: pytest.MonkeyPatch) -> None:
        active = _fake_run(run_id="run-1", status="running")
        monkeypatch.setitem(active_runs, "run-1", MagicMock())
        session = _session_with_active([active])

        should_run, _cancel_ids, _ = await _apply_multitask_strategy(session, "thread-1", "interrupt", _USER)

        assert should_run is False
        assert active.status == "running"


class TestGateResumeInFlightProtection:
    """interrupt/rollback must not cancel an in-flight resume — the fresh input would
    land on the pending-interrupt checkpoint and silently consume the HITL pause."""

    @pytest.mark.parametrize("strategy", ["interrupt", "rollback"])
    @pytest.mark.asyncio
    async def test_preempting_an_active_resume_is_rejected_409(self, strategy: str) -> None:
        resume = _fake_run(run_id="resume-1", status="running")
        resume.execution_params = {"execution": {"command": {"resume": "answer"}}}
        session = _session_with_active([resume])

        with pytest.raises(HTTPException) as exc:
            await _apply_multitask_strategy(session, "thread-1", strategy, _USER)

        assert exc.value.status_code == 409
        assert resume.status == "running"  # untouched

    @pytest.mark.asyncio
    async def test_enqueue_behind_an_active_resume_still_parks(self) -> None:
        resume = _fake_run(run_id="resume-1", status="running")
        resume.execution_params = {"execution": {"command": {"resume": "answer"}}}
        session = _session_with_active([resume])

        should_run, cancel_ids, _ = await _apply_multitask_strategy(session, "thread-1", "enqueue", _USER)

        assert should_run is False
        assert cancel_ids == []

    @pytest.mark.asyncio
    async def test_preempting_a_plain_run_is_still_allowed(self) -> None:
        plain = _fake_run(run_id="run-1", status="running", live=True)
        plain.execution_params = {"execution": {"command": None}}
        session = _session_with_active([plain])

        should_run, cancel_ids, _ = await _apply_multitask_strategy(session, "thread-1", "interrupt", _USER)

        assert should_run is False  # admitted (no 409); parked until run-1 exits
        assert cancel_ids == ["run-1"]


class TestLockStatementPinning:
    """Pin the FOR UPDATE serialization and FIFO ordering the docs promise — mocks would
    otherwise pass green with the locks or the ORDER BY silently removed."""

    @pytest.mark.asyncio
    async def test_admission_gate_locks_the_thread_row(self) -> None:
        session = _session_with_active([])

        await _apply_multitask_strategy(session, "thread-1", "enqueue", _USER)

        lock_stmt = str(session.execute.await_args_list[0].args[0])
        assert "FOR UPDATE" in lock_stmt
        assert "FROM thread " in lock_stmt  # the thread row lock (table name is singular)

    @pytest.mark.asyncio
    async def test_finalize_locks_the_thread_row_first(self) -> None:
        # The comment in finalize_run says its absence deadlocks (40P01) with the gate.
        session, captured = _capturing_session(rowcount=1)
        with (
            patch("aegra_api.services.run_status._get_session_maker", return_value=_make_session_maker(session)),
            patch("aegra_api.services.executor.executor", MagicMock(dispatch_next_for_thread=AsyncMock())),
        ):
            await finalize_run("r", "t", user_id="u", status="success", thread_status="idle")

        assert "FOR UPDATE" in captured[0]
        assert captured[0].startswith("SELECT thread")  # thread row locked before the run CAS

    @pytest.mark.asyncio
    async def test_admission_gate_locks_the_active_run_rows(self) -> None:
        # Without the run-row locks a pre-empted run's start CAS could win between the gate's
        # read (pending) and its write, and the gate would wrongly start the new run at once.
        session = _session_with_active([])

        await _apply_multitask_strategy(session, "thread-1", "enqueue", _USER)

        runs_stmt = str(session.scalars.await_args.args[0])
        assert "FROM runs" in runs_stmt
        assert "FOR UPDATE" in runs_stmt

    @pytest.mark.asyncio
    async def test_dispatch_locks_thread_and_promotes_fifo(self) -> None:
        ex = _RecordingExecutor()
        queued = _fake_run(run_id="queued-1", status="queued")
        session = AsyncMock()
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        statements: list[str] = []

        async def _scalar(stmt: Any, *a: object, **k: object) -> object:
            statements.append(str(stmt.compile(dialect=postgresql.dialect())))
            if len(statements) == 1:
                return MagicMock(status="busy")  # thread row (locked read)
            if len(statements) == 2:
                return None  # no occupying run
            return queued

        session.scalar = _scalar
        fake_job = MagicMock()
        fake_job.identity.run_id = "queued-1"
        with (
            patch("aegra_api.services.base_executor._get_session_maker", return_value=_make_session_maker(session)),
            patch("aegra_api.services.base_executor.RunJob.from_run_orm", return_value=fake_job),
        ):
            await ex.dispatch_next_for_thread("thread-1")

        assert "FOR UPDATE" in statements[0]  # thread lock serializes concurrent dispatchers
        assert "ORDER BY runs.created_at ASC" in statements[2]  # FIFO: oldest queued first
        assert "FOR UPDATE SKIP LOCKED" in statements[2]

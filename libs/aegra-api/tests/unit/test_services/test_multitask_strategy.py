"""Unit tests for double-texting (multitask) strategy handling.

Covers the admission gate (`_apply_multitask_strategy`) and the
finalize-time dispatch (`BaseExecutor.dispatch_next_for_thread`).
"""

from collections.abc import AsyncIterator, Callable
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from sqlalchemy.dialects import postgresql

from aegra_api.models.auth import User
from aegra_api.services import run_preparation as run_preparation_mod
from aegra_api.services.base_executor import BaseExecutor
from aegra_api.services.run_executor import _resolve_rollback_base, _rollback_fork_base
from aegra_api.services.run_preparation import _apply_multitask_strategy, _validate_resume_command
from aegra_api.services.run_status import finalize_run

_USER = User(identity="test-user")


def _fake_run(run_id: str = "run-1", status: str = "running") -> MagicMock:
    run = MagicMock()
    run.run_id = run_id
    run.status = status
    return run


def _session_with_active(active: list[MagicMock], *, terminal_run: MagicMock | None = None) -> AsyncMock:
    """Mock session whose active-run query returns ``active``.

    ``terminal_run`` is the most-recent-terminal-run row the rollback no-active
    path looks up (it reads ``.run_id`` and ``.status``); None means none exists.
    """
    session = AsyncMock()
    result = MagicMock()
    result.all.return_value = active
    session.scalars = AsyncMock(return_value=result)
    session.scalar = AsyncMock(return_value=terminal_run)
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
    async def test_interrupt_marks_active_and_returns_cancel_id(self) -> None:
        active = _fake_run(status="running")
        session = _session_with_active([active])

        should_run, cancel_ids, target = await _apply_multitask_strategy(session, "thread-1", "interrupt", _USER)

        assert should_run is True
        assert cancel_ids == ["run-1"]  # caller cancels post-commit to avoid deadlock
        assert active.status == "interrupted"
        assert target is None  # interrupt does not revert checkpoints
        session.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_rollback_marks_active_and_returns_target_without_delete(self) -> None:
        active = _fake_run(status="running")
        session = _session_with_active([active])

        should_run, cancel_ids, target = await _apply_multitask_strategy(session, "thread-1", "rollback", _USER)

        assert should_run is True
        assert cancel_ids == ["run-1"]
        assert target == "run-1"  # worker forks the new run from before this run
        assert active.status == "interrupted"
        session.delete.assert_not_called()  # rollback reverts via fork, never deletes

    @pytest.mark.asyncio
    async def test_rollback_no_active_reverts_broken_terminal_run(self) -> None:
        # #191: the dirtying run was already interrupted to a terminal state — repair it.
        session = _session_with_active([], terminal_run=_fake_run("old-interrupted", "interrupted"))

        should_run, cancel_ids, target = await _apply_multitask_strategy(session, "thread-1", "rollback", _USER)

        assert should_run is True
        assert cancel_ids == []
        assert target == "old-interrupted"

    @pytest.mark.asyncio
    async def test_rollback_no_active_ignores_successful_last_run(self) -> None:
        # A cleanly completed last turn must NOT be silently reverted (no undo-last-turn footgun).
        session = _session_with_active([], terminal_run=_fake_run("done", "success"))

        should_run, cancel_ids, target = await _apply_multitask_strategy(session, "thread-1", "rollback", _USER)

        assert should_run is True
        assert target is None

    @pytest.mark.asyncio
    async def test_rollback_no_prior_run_has_no_target(self) -> None:
        session = _session_with_active([], terminal_run=None)

        should_run, cancel_ids, target = await _apply_multitask_strategy(session, "thread-1", "rollback", _USER)

        assert should_run is True
        assert target is None

    @pytest.mark.asyncio
    async def test_queued_run_is_dropped_on_interrupt(self) -> None:
        # interrupt/rollback abandons in-flight work: a parked 'queued' double-text is
        # dropped from the queue (marked interrupted) but has no task, so no cancel id.
        queued = _fake_run(run_id="queued-1", status="queued")
        session = _session_with_active([queued])

        should_run, cancel_ids, target = await _apply_multitask_strategy(session, "thread-1", "interrupt", _USER)

        assert should_run is True
        assert cancel_ids == []
        assert queued.status == "interrupted"

    @pytest.mark.asyncio
    async def test_interrupt_drops_queued_behind_active(self) -> None:
        running = _fake_run(run_id="r1", status="running")
        queued = _fake_run(run_id="q1", status="queued")
        session = _session_with_active([running, queued])

        should_run, cancel_ids, target = await _apply_multitask_strategy(session, "thread-1", "interrupt", _USER)

        assert should_run is True
        assert cancel_ids == ["r1"]  # only the running run holds a task to cancel
        assert running.status == "interrupted"
        assert queued.status == "interrupted"  # the stale double-text is dropped, not run later

    @pytest.mark.asyncio
    async def test_rollback_targets_running_not_queued(self) -> None:
        running = _fake_run(run_id="r1", status="running")
        queued = _fake_run(run_id="q1", status="queued")
        session = _session_with_active([running, queued])

        _should_run, cancel_ids, target = await _apply_multitask_strategy(session, "thread-1", "rollback", _USER)

        assert cancel_ids == ["r1"]
        assert target == "r1"  # revert the run that executed, never a queued one
        assert queued.status == "interrupted"

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


class TestDispatchNextForThread:
    @pytest.mark.asyncio
    async def test_noop_when_not_accepting(self) -> None:
        ex = _RecordingExecutor()
        ex._accepting = False

        await ex.dispatch_next_for_thread("thread-1")

        assert ex.submitted == []

    @pytest.mark.asyncio
    async def test_noop_when_thread_interrupted(self) -> None:
        # A HITL-paused thread must NOT have a queued fresh-input run promoted onto it.
        # side_effect is padded with the occupying + queued lookups so that REMOVING the
        # guard reaches the promotion and fails on `submitted == []` (a behavioral failure),
        # not on StopAsyncIteration.
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


class _FakeSnap:
    """Minimal LangGraph StateSnapshot stand-in for rollback-base resolution."""

    def __init__(self, run_id: str | None, checkpoint_id: str, parent_checkpoint_id: str | None = None) -> None:
        self.metadata = {"run_id": run_id} if run_id is not None else {}
        self.config = {"configurable": {"checkpoint_id": checkpoint_id}}
        self.parent_config = (
            {"configurable": {"checkpoint_id": parent_checkpoint_id}} if parent_checkpoint_id is not None else None
        )


def _history(*snaps: _FakeSnap) -> Callable[..., AsyncIterator[_FakeSnap]]:
    # Mirror aget_state_history's keyword-only ``limit`` and its SQL LIMIT semantics
    # so the scan-cap / fail-loud path can be exercised deterministically.
    async def _gen(config: object, *, limit: int | None = None) -> AsyncIterator[_FakeSnap]:
        for i, s in enumerate(snaps):
            if limit is not None and i >= limit:
                return
            yield s

    return _gen


class TestResolveRollbackBase:
    """Worker-side resolution of the pre-target checkpoint to fork from (by lineage)."""

    @staticmethod
    async def _resolve(graph: MagicMock, target: str) -> str | None:
        with patch("aegra_api.services.run_executor.create_thread_config", return_value={"configurable": {}}):
            return await _resolve_rollback_base(graph, "t", User(identity="u"), target)

    @pytest.mark.asyncio
    async def test_returns_parent_of_oldest_target_checkpoint(self) -> None:
        graph = MagicMock()
        # target wrote cp-3 (parent cp-2) and cp-2 (input, parent cp-1=base); cp-1 is a prior run.
        graph.aget_state_history = _history(
            _FakeSnap("target", "cp-3", parent_checkpoint_id="cp-2"),
            _FakeSnap("target", "cp-2", parent_checkpoint_id="cp-1"),
            _FakeSnap("prev", "cp-1", parent_checkpoint_id="cp-0"),
        )
        assert await self._resolve(graph, "target") == "cp-1"

    @pytest.mark.asyncio
    async def test_second_rollback_anchors_to_lineage_not_sibling(self) -> None:
        # After P -> A(rolled back) -> B(forked from cp-P): history DESC interleaves the
        # abandoned A branch. A second rollback targeting B must fork from cp-P (B's parent),
        # NOT cp-A — the flat "first different run_id" scan would wrongly pick cp-A.
        graph = MagicMock()
        graph.aget_state_history = _history(
            _FakeSnap("B", "cp-B", parent_checkpoint_id="cp-P"),
            _FakeSnap("A", "cp-A", parent_checkpoint_id="cp-P"),
            _FakeSnap("P", "cp-P", parent_checkpoint_id=None),
        )
        assert await self._resolve(graph, "B") == "cp-P"

    @pytest.mark.asyncio
    async def test_retry_skips_own_partial_checkpoints(self) -> None:
        # Retry of crashed run B (target still A): B's own newer partial must be skipped;
        # base is A's oldest checkpoint's parent (cp-P), never B's partial.
        graph = MagicMock()
        graph.aget_state_history = _history(
            _FakeSnap("B", "cp-Bpart", parent_checkpoint_id="cp-P"),  # newer, current run's own partial
            _FakeSnap("A", "cp-A2", parent_checkpoint_id="cp-A1"),
            _FakeSnap("A", "cp-A1", parent_checkpoint_id="cp-P"),
            _FakeSnap("P", "cp-P", parent_checkpoint_id=None),
        )
        assert await self._resolve(graph, "A") == "cp-P"

    @pytest.mark.asyncio
    async def test_first_run_target_has_no_parent(self) -> None:
        # Target was the thread's first run: its oldest checkpoint has no parent -> None.
        graph = MagicMock()
        graph.aget_state_history = _history(
            _FakeSnap("target", "cp-1", parent_checkpoint_id="cp-0"),
            _FakeSnap("target", "cp-0", parent_checkpoint_id=None),
        )
        assert await self._resolve(graph, "target") is None

    @pytest.mark.asyncio
    async def test_returns_none_when_target_has_no_checkpoints(self) -> None:
        graph = MagicMock()
        graph.aget_state_history = _history(_FakeSnap("other", "cp-1", parent_checkpoint_id=None))
        assert await self._resolve(graph, "target") is None

    @pytest.mark.asyncio
    async def test_returns_none_on_empty_history(self) -> None:
        graph = MagicMock()
        graph.aget_state_history = _history()
        assert await self._resolve(graph, "target") is None

    @pytest.mark.asyncio
    async def test_ignores_straggler_from_cancelled_target(self) -> None:
        # The cancelled target writes a late 'straggler' checkpoint after the new run forked
        # and wrote its own. The base must still be the target's ORIGINAL oldest parent (cp-P),
        # not the straggler's parent. The order-independent full scan handles this; the old
        # early-break version would mis-anchor on the straggler.
        graph = MagicMock()
        graph.aget_state_history = _history(
            _FakeSnap("A", "cp-straggler", parent_checkpoint_id="cp-Bnew"),  # newest: A's late write
            _FakeSnap("B", "cp-Bnew", parent_checkpoint_id="cp-P"),  # the new run's own checkpoint
            _FakeSnap("A", "cp-A1", parent_checkpoint_id="cp-P"),  # A's original (oldest) checkpoint
            _FakeSnap("P", "cp-P", parent_checkpoint_id=None),
        )
        assert await self._resolve(graph, "A") == "cp-P"

    @pytest.mark.asyncio
    async def test_raises_when_target_block_exceeds_scan_cap(self) -> None:
        # Pathological: the target run has more checkpoints than the scan cap and the
        # window never exits its block. Fail loud rather than fork a mid-turn checkpoint.
        graph = MagicMock()
        graph.aget_state_history = _history(
            _FakeSnap("target", "cp-3", parent_checkpoint_id="cp-2"),
            _FakeSnap("target", "cp-2", parent_checkpoint_id="cp-1"),
            _FakeSnap("target", "cp-1", parent_checkpoint_id="cp-0"),
        )
        with (
            patch("aegra_api.services.run_executor._ROLLBACK_HISTORY_LIMIT", 3),
            pytest.raises(RuntimeError, match="exceeds 3 checkpoints"),
        ):
            await self._resolve(graph, "target")

    @pytest.mark.asyncio
    async def test_raises_when_cap_ends_on_non_target_row(self) -> None:
        # The window fills with interleaved sibling rows ending on a NON-target row while
        # older target checkpoints lie beyond it — the in-window "oldest" target is really
        # mid-run, so resolution must fail loud instead of forking from it.
        graph = MagicMock()
        graph.aget_state_history = _history(
            _FakeSnap("target", "cp-9", parent_checkpoint_id="cp-8"),
            _FakeSnap("sibling", "cp-8", parent_checkpoint_id="cp-7"),
            _FakeSnap("sibling", "cp-7", parent_checkpoint_id="cp-6"),
            _FakeSnap("target", "cp-6", parent_checkpoint_id="cp-5"),  # older target beyond the cap
        )
        with (
            patch("aegra_api.services.run_executor._ROLLBACK_HISTORY_LIMIT", 3),
            pytest.raises(RuntimeError, match="exceeds 3 checkpoints"),
        ):
            await self._resolve(graph, "target")

    @pytest.mark.asyncio
    async def test_first_run_at_scan_cap_returns_none_not_raise(self) -> None:
        # Target is the thread's first run and fills the scan cap; its oldest checkpoint has no
        # parent, so it is provably the root — return None (fork fresh), not a fail-loud error.
        graph = MagicMock()
        graph.aget_state_history = _history(
            _FakeSnap("target", "cp-1", parent_checkpoint_id="cp-0"),
            _FakeSnap("target", "cp-0", parent_checkpoint_id=None),  # root, no parent
        )
        with patch("aegra_api.services.run_executor._ROLLBACK_HISTORY_LIMIT", 2):
            assert await self._resolve(graph, "target") is None


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
        graph = MagicMock()
        graph.aget_state_history = _history(_FakeSnap("target", "cp-1", parent_checkpoint_id="cp-0"))
        with patch("aegra_api.services.run_executor.get_run_status", new=AsyncMock(return_value="success")):
            assert await _rollback_fork_base(graph, self._job()) is None

    @pytest.mark.asyncio
    async def test_resolves_base_when_target_not_success(self) -> None:
        graph = MagicMock()
        graph.aget_state_history = _history(
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
    async def test_fresh_input_on_paused_thread_rejected_409(self) -> None:
        # A plain fresh-input run on a HITL-paused thread would silently consume the
        # pending interrupt — it must be rejected so the pause stays resumable.
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
    async def test_null_resume_command_on_paused_thread_rejected(self) -> None:
        # command={'resume': None} must NOT slip past as a state op and error out on a paused
        # thread (flipping it to 'error'); it is gated like fresh input -> 409, pause intact.
        session = _session_with_thread("interrupted")
        with pytest.raises(HTTPException) as exc:
            await _validate_resume_command(session, "thread-1", {"resume": None}, _USER)
        assert exc.value.status_code == 409

    @pytest.mark.parametrize("command", [{"update": {}}, {"goto": []}, {"resume": None, "update": {}}])
    @pytest.mark.asyncio
    async def test_empty_container_command_on_paused_thread_rejected(self, command: dict[str, object]) -> None:
        # Empty update/goto produce no LangGraph writes and would crash a paused thread to
        # 'error'; truthiness classification routes them to the 409 gate, keeping the pause.
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
        with patch("aegra_api.services.run_status._get_session_maker", return_value=_make_session_maker(session)):
            await finalize_run("run-1", "thread-1", status="success", thread_status="idle")

        run_update = next(s for s in captured if s.startswith("UPDATE runs"))
        assert "runs.status IN" in run_update

    @pytest.mark.asyncio
    async def test_skips_thread_update_and_dispatch_when_run_not_owned(self) -> None:
        # rowcount 0 => the run was already terminalized by a pre-emption gate; finalize must
        # not clobber the thread (e.g. stomp a HITL pause to 'idle') nor dispatch the queue.
        executed: list[str] = []
        session = AsyncMock()

        async def _exec(stmt: object, *a: object, **k: object) -> MagicMock:
            text = str(stmt)
            executed.append(text)
            result = MagicMock()
            result.rowcount = 0 if text.startswith("UPDATE runs") else 1
            return result

        session.execute = _exec
        session.commit = AsyncMock()
        dispatched = AsyncMock()
        fake_executor = MagicMock()
        fake_executor.dispatch_next_for_thread = dispatched
        with (
            patch("aegra_api.services.run_status._get_session_maker", return_value=_make_session_maker(session)),
            patch("aegra_api.services.executor.executor", fake_executor),
        ):
            await finalize_run("run-1", "thread-1", status="success", thread_status="idle")

        assert not any(s.startswith("UPDATE thread") for s in executed)  # thread not clobbered (table is singular)
        dispatched.assert_not_awaited()  # no dispatch on a non-owned finalize

    @pytest.mark.asyncio
    async def test_terminal_override_still_excludes_committed_success(self) -> None:
        # The timeout corrector (allow_terminal_override) may overwrite 'interrupted' but NOT a
        # committed 'success'/'error': the run UPDATE filter widens, it never drops.
        captured: list[str] = []
        session = AsyncMock()

        async def _exec(stmt: object, *a: object, **k: object) -> MagicMock:
            captured.append(str(stmt.compile(compile_kwargs={"literal_binds": True})))
            result = MagicMock()
            result.rowcount = 1
            return result

        session.execute = _exec
        session.commit = AsyncMock()
        with (
            patch("aegra_api.services.run_status._get_session_maker", return_value=_make_session_maker(session)),
            patch("aegra_api.services.executor.executor", MagicMock(dispatch_next_for_thread=AsyncMock())),
        ):
            await finalize_run("r", "t", status="error", thread_status="error", allow_terminal_override=True)

        run_update = next(s for s in captured if s.startswith("UPDATE runs"))
        assert "'interrupted'" in run_update  # overrides the cancel handler's interrupted
        assert "'success'" not in run_update  # but never a committed success


def _capturing_session(*, rowcount: int = 1) -> tuple[AsyncMock, list[str]]:
    """Mock session whose execute() records compiled statements and reports ``rowcount``."""
    captured: list[str] = []
    session = AsyncMock()

    async def _exec(stmt: object, *a: object, **k: object) -> MagicMock:
        captured.append(str(stmt.compile(compile_kwargs={"literal_binds": True})))
        result = MagicMock()
        result.rowcount = rowcount
        return result

    session.execute = _exec
    session.commit = AsyncMock()
    return session, captured


class TestTryMarkRunRunning:
    """The pending→running transition is a CAS so a gate-pre-empted run cannot resurrect itself."""

    @pytest.mark.asyncio
    async def test_returns_true_when_run_still_active(self) -> None:
        session, captured = _capturing_session(rowcount=1)
        with patch("aegra_api.services.run_status._get_session_maker", return_value=_make_session_maker(session)):
            from aegra_api.services.run_status import try_mark_run_running

            assert await try_mark_run_running("run-1") is True
        # The UPDATE is guarded on the run still being pending/running.
        assert "'pending'" in captured[0] and "'running'" in captured[0]

    @pytest.mark.asyncio
    async def test_returns_false_when_gate_terminalized_the_run(self) -> None:
        session, _ = _capturing_session(rowcount=0)
        with patch("aegra_api.services.run_status._get_session_maker", return_value=_make_session_maker(session)):
            from aegra_api.services.run_status import try_mark_run_running

            assert await try_mark_run_running("run-1") is False


class TestTerminalizeUserCancel:
    """User-initiated cancels converge: unpark a queued run, else CAS-finalize the active one."""

    @pytest.mark.asyncio
    async def test_queued_run_is_unparked_without_finalize(self) -> None:
        session, captured = _capturing_session(rowcount=1)
        mock_executor = MagicMock(dispatch_next_for_thread=AsyncMock())
        with (
            patch("aegra_api.services.run_status._get_session_maker", return_value=_make_session_maker(session)),
            patch("aegra_api.services.run_status.finalize_run", new_callable=AsyncMock) as mock_finalize,
            patch("aegra_api.services.executor.executor", mock_executor),
        ):
            from aegra_api.services.run_status import terminalize_user_cancel

            await terminalize_user_cancel("run-1", "thread-1")

        assert "'queued'" in captured[0]  # guarded flip, only a still-queued run matches
        mock_finalize.assert_not_awaited()
        # If the unparked run headed a stranded queue, the runs behind it must not
        # wait for the recovery sweep — dispatch follows the unpark (idempotent).
        mock_executor.dispatch_next_for_thread.assert_awaited_once_with("thread-1")

    @pytest.mark.asyncio
    async def test_active_run_falls_through_to_finalize_with_lease_clear(self) -> None:
        session, _ = _capturing_session(rowcount=0)
        with (
            patch("aegra_api.services.run_status._get_session_maker", return_value=_make_session_maker(session)),
            patch("aegra_api.services.run_status.finalize_run", new_callable=AsyncMock) as mock_finalize,
        ):
            from aegra_api.services.run_status import terminalize_user_cancel

            await terminalize_user_cancel("run-1", "thread-1")

        mock_finalize.assert_awaited_once()
        kwargs = mock_finalize.await_args.kwargs
        assert kwargs["status"] == "interrupted"
        assert kwargs["thread_status"] == "idle"
        # Lease cleared so a worker whose pub/sub cancel was lost self-cancels on heartbeat.
        assert kwargs["clear_lease"] is True


class TestFinalizeLeaseHandling:
    @pytest.mark.asyncio
    async def test_clear_lease_releases_claim_in_the_terminal_write(self) -> None:
        session, captured = _capturing_session(rowcount=1)
        with (
            patch("aegra_api.services.run_status._get_session_maker", return_value=_make_session_maker(session)),
            patch("aegra_api.services.executor.executor", MagicMock(dispatch_next_for_thread=AsyncMock())),
        ):
            await finalize_run("r", "t", status="interrupted", thread_status="idle", clear_lease=True)

        run_update = next(s for s in captured if s.startswith("UPDATE runs"))
        assert "claimed_by" in run_update and "lease_expires_at" in run_update

    @pytest.mark.asyncio
    async def test_claimed_by_narrows_the_override_to_the_callers_lease(self) -> None:
        # The timeout corrector must not stomp an 'interrupted' the multitask gate wrote:
        # gate cancels clear the lease, so scoping the WHERE to our claim excludes them.
        session, captured = _capturing_session(rowcount=0)
        with (
            patch("aegra_api.services.run_status._get_session_maker", return_value=_make_session_maker(session)),
            patch("aegra_api.services.executor.executor", MagicMock(dispatch_next_for_thread=AsyncMock())),
        ):
            await finalize_run(
                "r",
                "t",
                status="error",
                thread_status="error",
                allow_terminal_override=True,
                claimed_by="worker-0",
            )

        run_update = next(s for s in captured if s.startswith("UPDATE runs"))
        assert "claimed_by = 'worker-0'" in run_update


class TestGatePreemptionLeaseClearing:
    @pytest.mark.asyncio
    async def test_interrupt_clears_lease_of_preempted_run(self) -> None:
        active = _fake_run(run_id="run-1", status="running")
        active.execution_params = {"execution": {"command": None}}
        session = _session_with_active([active])

        should_run, cancel_ids, _ = await _apply_multitask_strategy(session, "thread-1", "interrupt", _USER)

        assert should_run is True
        assert cancel_ids == ["run-1"]
        assert active.status == "interrupted"
        # Lease released with the pre-emption: a worker whose pub/sub cancel is lost
        # detects lease loss on its next heartbeat and self-cancels the job.
        assert active.claimed_by is None
        assert active.lease_expires_at is None


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
        plain = _fake_run(run_id="run-1", status="running")
        plain.execution_params = {"execution": {"command": None}}
        session = _session_with_active([plain])

        should_run, cancel_ids, _ = await _apply_multitask_strategy(session, "thread-1", "interrupt", _USER)

        assert should_run is True
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
            await finalize_run("r", "t", status="success", thread_status="idle")

        assert "FOR UPDATE" in captured[0]
        assert captured[0].startswith("SELECT thread")  # thread row locked before the run CAS

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

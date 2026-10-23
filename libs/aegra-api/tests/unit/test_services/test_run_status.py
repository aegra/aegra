"""Unit tests for run_status service."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.dialects import postgresql

from aegra_api.services.run_status import (
    INTERRUPT_REQUESTED_KEY,
    _safe_serialize,
    drop_queued_runs,
    finalize_overdue_preemptions,
    finalize_run,
    interrupt_requested,
    interrupt_requested_expr,
    interrupt_unowned_run,
    mark_interrupt_requested,
    queued_threads_stmt,
    set_thread_status,
    set_thread_status_if_no_active_runs,
    start_run,
)
from aegra_api.settings import settings


@pytest.fixture(autouse=True)
def _no_queue_dispatch(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """finalize_run / interrupt_unowned_run / cancel_queued_run dispatch the thread's queued
    runs afterwards; keep that off the database here."""
    mock = AsyncMock()
    monkeypatch.setattr("aegra_api.services.run_status.dispatch_next_queued_run", mock)
    return mock


def _make_mock_session() -> AsyncMock:
    """Create a mock async session with execute and commit."""
    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    # interrupt_unowned_run wraps its lock+CAS in a SAVEPOINT; `await session.begin_nested()`
    # must hand back something whose commit()/rollback() are awaitable.
    session.begin_nested = AsyncMock(return_value=AsyncMock())
    return session


def _make_mock_session_maker(session: AsyncMock) -> MagicMock:
    """Wrap a mock session in a context-manager-returning maker."""
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    maker = MagicMock(return_value=ctx)
    return maker


class TestStartRun:
    @pytest.mark.asyncio
    async def test_starts_active_run(self) -> None:
        session = _make_mock_session()
        result = MagicMock()
        result.scalar_one_or_none.return_value = "run-1"
        session.execute = AsyncMock(return_value=result)

        with patch("aegra_api.services.run_status._get_session_maker", return_value=_make_mock_session_maker(session)):
            started = await start_run("run-1", user_id="user-1")

        assert started is True
        statement = session.execute.await_args.args[0]
        compiled = statement.compile()
        assert "runs.user_id" in str(compiled)
        assert "user-1" in compiled.params.values()
        session.commit.assert_awaited_once()
        session.rollback.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_does_not_revive_terminal_run(self) -> None:
        session = _make_mock_session()
        result = MagicMock()
        result.scalar_one_or_none.return_value = None
        session.execute = AsyncMock(return_value=result)

        with patch("aegra_api.services.run_status._get_session_maker", return_value=_make_mock_session_maker(session)):
            started = await start_run("run-1", user_id="user-1")

        assert started is False
        session.commit.assert_not_awaited()
        session.rollback.assert_awaited_once()


class TestFinalizeRun:
    @pytest.mark.asyncio
    async def test_finalizes_only_an_active_run(self) -> None:
        session = _make_mock_session()
        result = MagicMock()
        result.scalar_one_or_none.return_value = "run-1"
        session.execute = AsyncMock(return_value=result)

        with (
            patch("aegra_api.services.run_status._get_session_maker", return_value=_make_mock_session_maker(session)),
            patch(
                "aegra_api.services.run_status.set_thread_status_if_no_active_runs",
                new_callable=AsyncMock,
            ) as mock_set_thread,
        ):
            finalized = await finalize_run(
                "run-1",
                "thread-1",
                user_id="user-1",
                status="success",
                thread_status="idle",
            )

        assert finalized is True
        statement = session.execute.await_args.args[0]
        compiled = statement.compile()
        assert "runs.user_id" in str(compiled)
        assert "user-1" in compiled.params.values()
        mock_set_thread.assert_awaited_once_with(session, ["thread-1"], "idle", user_id="user-1")
        session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_serializes_output_and_records_error_when_provided(self) -> None:
        session = _make_mock_session()
        result = MagicMock()
        result.scalar_one_or_none.return_value = "run-1"
        session.execute = AsyncMock(return_value=result)

        with (
            patch("aegra_api.services.run_status._get_session_maker", return_value=_make_mock_session_maker(session)),
            patch("aegra_api.services.run_status._safe_serialize", return_value={"key": "val"}) as mock_ser,
            patch(
                "aegra_api.services.run_status.set_thread_status_if_no_active_runs",
                new_callable=AsyncMock,
            ),
        ):
            finalized = await finalize_run(
                "run-1",
                "thread-1",
                user_id="user-1",
                status="error",
                thread_status="error",
                output={"key": "val"},
                error="something broke",
            )

        assert finalized is True
        mock_ser.assert_called_once_with({"key": "val"}, "run-1")
        params = session.execute.await_args.args[0].compile().params
        assert params["error_message"] == "something broke"

    @pytest.mark.asyncio
    async def test_omits_output_serialization_when_not_provided(self) -> None:
        session = _make_mock_session()
        result = MagicMock()
        result.scalar_one_or_none.return_value = "run-1"
        session.execute = AsyncMock(return_value=result)

        with (
            patch("aegra_api.services.run_status._get_session_maker", return_value=_make_mock_session_maker(session)),
            patch("aegra_api.services.run_status._safe_serialize") as mock_ser,
            patch(
                "aegra_api.services.run_status.set_thread_status_if_no_active_runs",
                new_callable=AsyncMock,
            ),
        ):
            await finalize_run(
                "run-1",
                "thread-1",
                user_id="user-1",
                status="success",
                thread_status="idle",
            )

        mock_ser.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_run_that_is_already_terminal(self) -> None:
        session = _make_mock_session()
        result = MagicMock()
        result.scalar_one_or_none.return_value = None
        session.execute = AsyncMock(return_value=result)

        with (
            patch("aegra_api.services.run_status._get_session_maker", return_value=_make_mock_session_maker(session)),
            patch(
                "aegra_api.services.run_status.set_thread_status_if_no_active_runs",
                new_callable=AsyncMock,
            ) as mock_set_thread,
        ):
            finalized = await finalize_run(
                "run-1",
                "thread-1",
                user_id="user-1",
                status="success",
                thread_status="idle",
            )

        assert finalized is False
        mock_set_thread.assert_not_awaited()
        session.commit.assert_not_awaited()
        session.rollback.assert_awaited_once()


class TestSetThreadStatus:
    @pytest.mark.asyncio
    async def test_updates_thread_status(self) -> None:
        session = _make_mock_session()
        mock_result = MagicMock()
        mock_result.rowcount = 1
        session.execute = AsyncMock(return_value=mock_result)

        await set_thread_status(session, "thread-1", "idle")

        session.execute.assert_awaited_once()
        session.commit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_raises_when_thread_not_found(self) -> None:
        session = _make_mock_session()
        mock_result = MagicMock()
        mock_result.rowcount = 0
        session.execute = AsyncMock(return_value=mock_result)

        with pytest.raises(ValueError, match="Thread 'thread-missing' not found"):
            await set_thread_status(session, "thread-missing", "idle")


class TestSetThreadStatusIfNoActiveRuns:
    @pytest.mark.asyncio
    async def test_updates_owned_threads_without_committing(self) -> None:
        session = _make_mock_session()

        await set_thread_status_if_no_active_runs(
            session,
            ["thread-1"],
            "idle",
            user_id="user-1",
        )

        session.execute.assert_awaited_once()
        statement = session.execute.await_args.args[0]
        compiled = statement.compile()
        sql = str(compiled)
        assert "thread.user_id" in sql
        assert "runs.user_id" in sql
        assert list(compiled.params.values()).count("user-1") == 2
        session.commit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_database_when_thread_ids_are_empty(self) -> None:
        session = _make_mock_session()

        await set_thread_status_if_no_active_runs(session, [], "idle", user_id="user-1")

        session.execute.assert_not_awaited()


class TestInterruptUnownedRun:
    @pytest.mark.asyncio
    async def test_reconciles_run_and_thread_in_one_transaction(self, _no_queue_dispatch: AsyncMock) -> None:
        session = _make_mock_session()
        result = MagicMock()
        result.scalar_one_or_none.return_value = "run-1"
        session.execute = AsyncMock(return_value=result)

        with patch(
            "aegra_api.services.run_status.set_thread_status_if_no_active_runs",
            new_callable=AsyncMock,
        ) as mock_set_thread:
            interrupted = await interrupt_unowned_run(session, "run-1", "thread-1", user_id="user-1")

        assert interrupted is True
        statement = session.execute.await_args.args[0]
        compiled = statement.compile()
        sql = str(compiled)
        assert "runs.user_id" in sql
        assert "runs.claimed_by IS NULL" in sql
        assert "runs.lease_expires_at <" in sql
        assert "user-1" in compiled.params.values()
        mock_set_thread.assert_awaited_once_with(session, ["thread-1"], "idle", user_id="user-1")
        session.commit.assert_awaited_once()
        # Whether the queue may move is the caller's call (a local task may still be running).
        _no_queue_dispatch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_does_not_commit_when_live_owner_wins_race(self) -> None:
        session = _make_mock_session()
        result = MagicMock()
        result.scalar_one_or_none.return_value = None
        session.execute = AsyncMock(return_value=result)

        with patch(
            "aegra_api.services.run_status.set_thread_status_if_no_active_runs",
            new_callable=AsyncMock,
        ) as mock_set_thread:
            interrupted = await interrupt_unowned_run(session, "run-1", "thread-1", user_id="user-1")

        assert interrupted is False
        mock_set_thread.assert_not_awaited()
        session.commit.assert_not_awaited()
        # Thread row locked first (gate lock order) inside a savepoint; a live owner means nothing
        # was written, so the savepoint rolls back to free the lock without expiring the session.
        session.begin_nested.return_value.rollback.assert_awaited_once()
        session.rollback.assert_not_awaited()


class TestSafeSerialize:
    def test_returns_serialized_output(self) -> None:
        with patch("aegra_api.services.run_status._serializer") as mock_ser:
            mock_ser.serialize.return_value = {"a": 1}
            result = _safe_serialize({"a": 1}, "run-1")

        assert result == {"a": 1}

    def test_returns_fallback_on_failure(self) -> None:
        with patch("aegra_api.services.run_status._serializer") as mock_ser:
            mock_ser.serialize.side_effect = TypeError("boom")
            result = _safe_serialize(object(), "run-1")

        assert result["error"] == "Output serialization failed"
        assert "original_type" in result


class TestQueuedThreadsStmt:
    """The recovery sweeps' thread scan honours the paused-thread policy."""

    @staticmethod
    def _sql() -> str:
        return str(queued_threads_stmt().compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}))

    def test_reject_policy_skips_paused_threads(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Promotion onto a paused thread is refused under reject, so retrying it every sweep
        # would only log a warning for as long as the pause lasts.
        monkeypatch.setattr(settings.multitask, "MULTITASK_PAUSED_THREAD_POLICY", "reject")
        sql = self._sql()
        assert "JOIN thread" in sql
        assert "thread.status != 'interrupted'" in sql

    def test_admit_policy_scans_every_queued_thread(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings.multitask, "MULTITASK_PAUSED_THREAD_POLICY", "admit")
        sql = self._sql()
        assert "runs.status = 'queued'" in sql
        assert "JOIN" not in sql


class TestDropQueuedRuns:
    @pytest.mark.asyncio
    async def test_flips_parked_rows_and_returns_their_ids(self) -> None:
        session = AsyncMock()
        result = MagicMock()
        result.all.return_value = [("q1",), ("q2",)]
        session.execute = AsyncMock(return_value=result)

        dropped = await drop_queued_runs(session, "thread-1", user_id="user-1")

        assert dropped == ["q1", "q2"]
        sql = str(
            session.execute.await_args.args[0].compile(
                dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
            )
        )
        assert "runs.status = 'queued'" in sql and "'interrupted'" in sql and "runs.user_id = 'user-1'" in sql
        session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_nothing_parked_commits_nothing(self) -> None:
        session = AsyncMock()
        result = MagicMock()
        result.all.return_value = []
        session.execute = AsyncMock(return_value=result)

        assert await drop_queued_runs(session, "thread-1", user_id="user-1") == []
        session.commit.assert_not_awaited()


class TestPreemptionStamp:
    def test_mark_and_read(self) -> None:
        run = MagicMock()
        run.execution_params = {"behavior": {"multitask_strategy": "enqueue"}}
        now = datetime(2026, 10, 19, tzinfo=UTC)

        mark_interrupt_requested(run, now=now)

        assert run.execution_params["behavior"] == {"multitask_strategy": "enqueue"}  # rest preserved
        assert run.execution_params[INTERRUPT_REQUESTED_KEY] == now.isoformat()
        assert interrupt_requested(run.execution_params) is True
        assert interrupt_requested({}) is False and interrupt_requested(None) is False

    def test_first_stamp_is_kept(self) -> None:
        # A repeated interrupt double-text must not restart the pre-emption timeout clock.
        run = MagicMock()
        run.execution_params = {}
        first = datetime(2026, 10, 19, tzinfo=UTC)

        mark_interrupt_requested(run, now=first)
        mark_interrupt_requested(run, now=datetime(2026, 10, 19, 0, 0, 30, tzinfo=UTC))

        assert run.execution_params[INTERRUPT_REQUESTED_KEY] == first.isoformat()

    def test_sql_expression_reads_the_key_as_text(self) -> None:
        sql = str(
            interrupt_requested_expr()
            .is_(None)
            .compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True})
        )
        assert "runs.execution_params ->> '_interrupt_requested_at'" in sql

    @pytest.mark.asyncio
    async def test_overdue_preemptions_are_interrupted_and_threads_freed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings.multitask, "MULTITASK_PREEMPT_TIMEOUT_SECONDS", 60)
        session = AsyncMock()
        result = MagicMock()
        result.all.return_value = [("run-1", "thread-1", "user-1"), ("run-2", "thread-1", "user-1")]
        session.execute = AsyncMock(return_value=result)

        with patch(
            "aegra_api.services.run_status.set_thread_status_if_no_active_runs", new_callable=AsyncMock
        ) as mock_thread:
            done = await finalize_overdue_preemptions(session)

        assert done == [("run-1", "thread-1"), ("run-2", "thread-1")]
        sql = str(
            session.execute.await_args_list[0]
            .args[0]
            .compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True})
        )
        assert "runs.status = 'running'" in sql
        assert "CAST(" in sql and "TIMESTAMP WITH TIME ZONE" in sql and "<" in sql  # stamp older than the timeout
        assert "'interrupted'" in sql
        lock_sql = str(session.execute.await_args_list[1].args[0].compile(dialect=postgresql.dialect()))
        assert "FOR UPDATE" in lock_sql and "ORDER BY thread.thread_id" in lock_sql  # thread rows locked first
        mock_thread.assert_awaited_once_with(session, {"thread-1"}, "idle", user_id="user-1")
        assert session.commit.await_count == 2  # run rows first, thread rows in a second transaction

    @pytest.mark.asyncio
    async def test_nothing_overdue_commits_nothing(self) -> None:
        session = AsyncMock()
        result = MagicMock()
        result.all.return_value = []
        session.execute = AsyncMock(return_value=result)

        assert await finalize_overdue_preemptions(session) == []
        session.commit.assert_not_awaited()

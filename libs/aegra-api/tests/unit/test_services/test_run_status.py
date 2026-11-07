"""Unit tests for run_status service."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aegra_api.core.orm import Run as RunORM
from aegra_api.services.run_status import (
    _safe_serialize,
    finalize_run,
    interrupt_run_before_delete,
    interrupt_unowned_run,
    set_thread_status,
    set_thread_status_if_no_active_runs,
    start_run,
)


def _make_mock_session() -> AsyncMock:
    """Create a mock async session with execute and commit."""
    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
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
    async def test_reconciles_run_and_thread_in_one_transaction(self) -> None:
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


class TestInterruptRunBeforeDelete:
    @staticmethod
    def _run() -> RunORM:
        return RunORM(
            run_id="run-1",
            thread_id="thread-1",
            assistant_id="asst-1",
            user_id="user-1",
            status="running",
            execution_params={"behavior": {"webhook": "https://h.example/x"}},
            created_at=datetime(2026, 11, 1, tzinfo=UTC),
        )

    @pytest.mark.asyncio
    async def test_commits_the_cancel_and_reports_it_when_it_wins(self) -> None:
        run = self._run()

        with (
            patch("aegra_api.services.run_status.finalize_run", new_callable=AsyncMock, return_value=True) as finalize,
            patch("aegra_api.services.run_status.report_cancelled_run") as report,
        ):
            await interrupt_run_before_delete(run)

        finalize.assert_awaited_once_with(
            "run-1", "thread-1", user_id="user-1", status="interrupted", thread_status="idle", output={}
        )
        report.assert_called_once_with(
            run.execution_params,
            run_id="run-1",
            thread_id="thread-1",
            assistant_id="asst-1",
            created_at=run.created_at,
        )

    @pytest.mark.asyncio
    async def test_leaves_the_report_to_whoever_finalized_first(self) -> None:
        with (
            patch("aegra_api.services.run_status.finalize_run", new_callable=AsyncMock, return_value=False),
            patch("aegra_api.services.run_status.report_cancelled_run") as report,
        ):
            await interrupt_run_before_delete(self._run())

        report.assert_not_called()

"""Unit tests for orphan_thread_sweeper service."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from psycopg import Error as PsycopgError
from sqlalchemy.dialects import postgresql

from aegra_api.observability.metrics import ORPHAN_THREADS_SWEPT
from aegra_api.services.orphan_thread_sweeper import OrphanThreadSweeper, _claim_stmt


def _swept_count(outcome: str) -> float:
    """Read the current value of the sweeper counter for one outcome label."""
    return ORPHAN_THREADS_SWEPT.labels(outcome=outcome)._value.get()


def _make_session_maker(session: AsyncMock) -> MagicMock:
    """Wrap a mock session in a context-manager-returning maker."""
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return MagicMock(return_value=ctx)


class TestClaimStmt:
    """The claim statement locks the thread row and skips threads with active runs."""

    def test_claim_shape(self) -> None:
        stmt = _claim_stmt(cutoff=datetime.now(UTC), limit=100)
        sql = str(stmt.compile(dialect=postgresql.dialect()))

        assert "FOR UPDATE OF thread SKIP LOCKED" in sql
        assert "is_ephemeral" in sql
        assert "updated_at <" in sql
        assert "EXISTS" in sql
        assert "runs" in sql  # active-run guard subquery
        assert "LIMIT" in sql

    def test_orders_by_updated_at_ascending(self) -> None:
        stmt = _claim_stmt(cutoff=datetime.now(UTC), limit=100)
        sql = str(stmt.compile(dialect=postgresql.dialect()))

        assert "ORDER BY thread.updated_at ASC" in sql


class TestSweep:
    """_sweep claims a batch, deletes checkpoints then the thread row."""

    @pytest.mark.asyncio
    async def test_deletes_claimed_threads(self) -> None:
        session = AsyncMock()
        claim_result = MagicMock()
        claim_result.scalars.return_value.all.return_value = ["thread-1", "thread-2"]
        session.execute = AsyncMock(side_effect=[claim_result, MagicMock(), MagicMock()])
        maker = _make_session_maker(session)

        checkpointer = AsyncMock()
        db = MagicMock()
        db.get_checkpointer.return_value = checkpointer

        with (
            patch("aegra_api.services.orphan_thread_sweeper._get_session_maker", return_value=maker),
            patch("aegra_api.services.orphan_thread_sweeper.db_manager", db),
        ):
            await OrphanThreadSweeper()._sweep()

        assert checkpointer.adelete_thread.await_count == 2
        checkpointer.adelete_thread.assert_any_await("thread-1")
        checkpointer.adelete_thread.assert_any_await("thread-2")
        # One claim SELECT + one DELETE per surviving thread.
        assert session.execute.await_count == 3
        session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_noop_when_nothing_claimed(self) -> None:
        session = AsyncMock()
        claim_result = MagicMock()
        claim_result.scalars.return_value.all.return_value = []
        session.execute = AsyncMock(return_value=claim_result)
        maker = _make_session_maker(session)

        with patch("aegra_api.services.orphan_thread_sweeper._get_session_maker", return_value=maker):
            await OrphanThreadSweeper()._sweep()

        session.commit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_checkpoint_failure_skips_item_and_continues(self) -> None:
        """A thread whose checkpoint delete fails is left in place; others still delete."""
        session = AsyncMock()
        claim_result = MagicMock()
        claim_result.scalars.return_value.all.return_value = ["thread-bad", "thread-ok"]
        session.execute = AsyncMock(side_effect=[claim_result, MagicMock()])
        maker = _make_session_maker(session)

        checkpointer = AsyncMock()
        checkpointer.adelete_thread.side_effect = [PsycopgError("backend down"), None]
        db = MagicMock()
        db.get_checkpointer.return_value = checkpointer

        with (
            patch("aegra_api.services.orphan_thread_sweeper._get_session_maker", return_value=maker),
            patch("aegra_api.services.orphan_thread_sweeper.db_manager", db),
        ):
            await OrphanThreadSweeper()._sweep()

        # Claim SELECT + exactly one DELETE (for thread-ok only).
        assert session.execute.await_count == 2
        delete_sql = str(session.execute.await_args_list[1].args[0])
        assert "thread" in delete_sql.lower()
        session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_metrics_increment_per_outcome(self) -> None:
        session = AsyncMock()
        claim_result = MagicMock()
        claim_result.scalars.return_value.all.return_value = ["thread-bad", "thread-ok"]
        session.execute = AsyncMock(side_effect=[claim_result, MagicMock()])
        maker = _make_session_maker(session)

        checkpointer = AsyncMock()
        checkpointer.adelete_thread.side_effect = [OSError("disk full"), None]
        db = MagicMock()
        db.get_checkpointer.return_value = checkpointer

        deleted_before = _swept_count("deleted")
        error_before = _swept_count("error")

        with (
            patch("aegra_api.services.orphan_thread_sweeper._get_session_maker", return_value=maker),
            patch("aegra_api.services.orphan_thread_sweeper.db_manager", db),
        ):
            await OrphanThreadSweeper()._sweep()

        assert _swept_count("deleted") == deleted_before + 1
        assert _swept_count("error") == error_before + 1


class TestStartStop:
    @pytest.mark.asyncio
    async def test_start_creates_background_task(self) -> None:
        sweeper = OrphanThreadSweeper()

        with patch("aegra_api.services.orphan_thread_sweeper.settings") as mock_settings:
            mock_settings.orphan_thread.ORPHAN_THREAD_SWEEP_INTERVAL_SECONDS = 60
            mock_settings.orphan_thread.ORPHAN_THREAD_RETENTION_MINUTES = 60
            await sweeper.start()

        assert sweeper._task is not None
        assert not sweeper._task.done()

        await sweeper.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_background_task(self) -> None:
        sweeper = OrphanThreadSweeper()

        with patch("aegra_api.services.orphan_thread_sweeper.settings") as mock_settings:
            mock_settings.orphan_thread.ORPHAN_THREAD_SWEEP_INTERVAL_SECONDS = 60
            mock_settings.orphan_thread.ORPHAN_THREAD_RETENTION_MINUTES = 60
            await sweeper.start()
            task = sweeper._task
            await sweeper.stop()

        assert sweeper._task is None
        assert task is not None
        assert task.done()

    @pytest.mark.asyncio
    async def test_stop_noop_when_not_started(self) -> None:
        sweeper = OrphanThreadSweeper()
        # Should not raise
        await sweeper.stop()
        assert sweeper._task is None


class TestLoop:
    @pytest.mark.asyncio
    async def test_loop_ticks_first_then_sleeps(self) -> None:
        """The loop sweeps immediately on start, not only after the first sleep."""
        sweeper = OrphanThreadSweeper()
        sweeper._running = True
        call_count = 0

        async def counting_sweep() -> None:
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                sweeper._running = False

        with (
            patch.object(sweeper, "_sweep", side_effect=counting_sweep),
            patch("aegra_api.services.orphan_thread_sweeper.settings") as mock_settings,
            patch("aegra_api.services.orphan_thread_sweeper.asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_settings.orphan_thread.ORPHAN_THREAD_SWEEP_INTERVAL_SECONDS = 0
            await sweeper._loop()

        assert call_count == 2

    @pytest.mark.asyncio
    async def test_loop_survives_sweep_exception(self) -> None:
        """An exception in one tick is logged, not fatal to the loop."""
        sweeper = OrphanThreadSweeper()
        sweeper._running = True
        call_count = 0

        async def failing_sweep() -> None:
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                sweeper._running = False
            raise RuntimeError("boom")

        with (
            patch.object(sweeper, "_sweep", side_effect=failing_sweep),
            patch("aegra_api.services.orphan_thread_sweeper.settings") as mock_settings,
            patch("aegra_api.services.orphan_thread_sweeper.asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_settings.orphan_thread.ORPHAN_THREAD_SWEEP_INTERVAL_SECONDS = 0
            await sweeper._loop()

        assert call_count == 2

"""Unit tests for lease_reaper service."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis import RedisError

from aegra_api.services.lease_reaper import LeaseReaper


def _make_session_maker(session: AsyncMock) -> MagicMock:
    """Wrap a mock session in a context-manager-returning maker."""
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    maker = MagicMock(return_value=ctx)
    return maker


class TestFindRecoverable:
    @pytest.mark.asyncio
    async def test_returns_crashed_and_stuck_separately(self) -> None:
        session = AsyncMock()
        crashed_result = MagicMock()
        crashed_result.fetchall.return_value = [("run-1",)]
        stuck_result = MagicMock()
        stuck_result.fetchall.return_value = [("run-2",)]
        session.execute = AsyncMock(side_effect=[crashed_result, stuck_result])
        maker = _make_session_maker(session)

        with patch("aegra_api.services.lease_reaper._get_session_maker", return_value=maker):
            crashed, stuck = await LeaseReaper._find_recoverable()

        assert crashed == ["run-1"]
        assert stuck == ["run-2"]

    @pytest.mark.asyncio
    async def test_returns_empty_when_nothing_to_recover(self) -> None:
        session = AsyncMock()
        empty_result = MagicMock()
        empty_result.fetchall.return_value = []
        session.execute = AsyncMock(return_value=empty_result)
        maker = _make_session_maker(session)

        with patch("aegra_api.services.lease_reaper._get_session_maker", return_value=maker):
            crashed, stuck = await LeaseReaper._find_recoverable()

        assert crashed == []
        assert stuck == []

    @pytest.mark.asyncio
    async def test_stuck_pending_predicate_keys_on_updated_at_not_created_at(self) -> None:
        # A queued run promoted to pending keeps its old created_at, so the stuck-pending
        # check must filter on updated_at — else fresh promotions get falsely reaped.
        captured: list[str] = []
        session = AsyncMock()

        async def _exec(stmt: object) -> MagicMock:
            captured.append(str(stmt))
            result = MagicMock()
            result.fetchall.return_value = []
            return result

        session.execute = _exec
        with patch("aegra_api.services.lease_reaper._get_session_maker", return_value=_make_session_maker(session)):
            await LeaseReaper._find_recoverable()

        stuck_sql = captured[1]  # second query is the stuck-pending scan
        assert "updated_at" in stuck_sql
        assert "created_at" not in stuck_sql


class TestResetToPending:
    @pytest.mark.asyncio
    async def test_returns_actually_reset_ids(self) -> None:
        session = AsyncMock()
        mock_result = MagicMock()
        # Only run-1 was actually reset (run-2 may have been claimed by another worker)
        mock_result.fetchall.return_value = [("run-1",)]
        session.execute = AsyncMock(return_value=mock_result)
        session.commit = AsyncMock()
        maker = _make_session_maker(session)

        with patch("aegra_api.services.lease_reaper._get_session_maker", return_value=maker):
            result = await LeaseReaper._reset_to_pending(["run-1", "run-2"])

        assert result == ["run-1"]
        session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_returns_empty_when_none_reset(self) -> None:
        session = AsyncMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        session.execute = AsyncMock(return_value=mock_result)
        session.commit = AsyncMock()
        maker = _make_session_maker(session)

        with patch("aegra_api.services.lease_reaper._get_session_maker", return_value=maker):
            result = await LeaseReaper._reset_to_pending(["run-1"])

        assert result == []


class TestReenqueue:
    @pytest.mark.asyncio
    async def test_pushes_to_redis(self) -> None:
        mock_client = AsyncMock()

        with (
            patch("aegra_api.services.lease_reaper.redis_manager") as mock_rm,
            patch("aegra_api.services.lease_reaper.settings") as mock_settings,
        ):
            mock_settings.worker.WORKER_QUEUE_KEY = "aegra:jobs"
            mock_rm.get_client.return_value = mock_client

            await LeaseReaper._reenqueue(["run-1", "run-2"])

        assert mock_client.rpush.await_count == 2

    @pytest.mark.asyncio
    async def test_logs_warning_when_redis_unavailable(self) -> None:
        with (
            patch("aegra_api.services.lease_reaper.redis_manager") as mock_rm,
            patch("aegra_api.services.lease_reaper.settings") as mock_settings,
        ):
            mock_settings.worker.WORKER_QUEUE_KEY = "aegra:jobs"
            mock_rm.get_client.side_effect = RedisError("connection refused")

            # Should not raise
            await LeaseReaper._reenqueue(["run-1"])

    @pytest.mark.asyncio
    async def test_noop_when_empty_list(self) -> None:
        mock_client = AsyncMock()

        with (
            patch("aegra_api.services.lease_reaper.redis_manager") as mock_rm,
            patch("aegra_api.services.lease_reaper.settings") as mock_settings,
        ):
            mock_settings.worker.WORKER_QUEUE_KEY = "aegra:jobs"
            mock_rm.get_client.return_value = mock_client

            await LeaseReaper._reenqueue([])

        mock_client.rpush.assert_not_awaited()


class TestReap:
    @pytest.mark.asyncio
    async def test_crashed_runs_reset_before_retry_check(self) -> None:
        """Reset claims ownership atomically, then retry check runs on claimed set only."""
        reaper = LeaseReaper()

        with (
            patch.object(
                LeaseReaper, "_find_recoverable", new_callable=AsyncMock, return_value=(["run-1", "run-2"], [])
            ),
            patch.object(LeaseReaper, "_find_stranded_queued_threads", new_callable=AsyncMock, return_value=[]),
            patch.object(
                LeaseReaper, "_reset_to_pending", new_callable=AsyncMock, return_value=["run-1", "run-2"]
            ) as mock_reset,
            patch.object(
                LeaseReaper, "_check_retry_limits", new_callable=AsyncMock, return_value=(["run-1"], ["run-2"])
            ) as mock_retry,
            patch.object(LeaseReaper, "_reenqueue", new_callable=AsyncMock) as mock_reenqueue,
            patch.object(LeaseReaper, "_mark_permanently_failed", new_callable=AsyncMock) as mock_fail,
        ):
            await reaper._reap()

        # Reset called with ALL crashed (atomic ownership claim)
        mock_reset.assert_awaited_once_with(["run-1", "run-2"])
        # Retry check only runs on actually_reset set
        mock_retry.assert_awaited_once_with(["run-1", "run-2"])
        mock_reenqueue.assert_awaited_once_with(["run-1"])
        mock_fail.assert_awaited_once_with(["run-2"])

    @pytest.mark.asyncio
    async def test_stuck_pending_reenqueued_without_retry_charge(self) -> None:
        """Stuck pending runs are re-enqueued directly, no retry count increment."""
        reaper = LeaseReaper()

        with (
            patch.object(LeaseReaper, "_find_recoverable", new_callable=AsyncMock, return_value=([], ["run-3"])),
            patch.object(LeaseReaper, "_find_stranded_queued_threads", new_callable=AsyncMock, return_value=[]),
            patch.object(LeaseReaper, "_check_retry_limits", new_callable=AsyncMock) as mock_retry,
            patch.object(LeaseReaper, "_reenqueue", new_callable=AsyncMock) as mock_reenqueue,
        ):
            await reaper._reap()

        mock_retry.assert_not_awaited()
        mock_reenqueue.assert_awaited_once_with(["run-3"])

    @pytest.mark.asyncio
    async def test_skips_when_nothing_to_recover(self) -> None:
        reaper = LeaseReaper()

        with (
            patch.object(LeaseReaper, "_find_recoverable", new_callable=AsyncMock, return_value=([], [])),
            patch.object(LeaseReaper, "_find_stranded_queued_threads", new_callable=AsyncMock, return_value=[]),
            patch.object(LeaseReaper, "_reset_to_pending", new_callable=AsyncMock) as mock_reset,
            patch.object(LeaseReaper, "_reenqueue", new_callable=AsyncMock) as mock_reenqueue,
        ):
            await reaper._reap()

        mock_reset.assert_not_awaited()
        mock_reenqueue.assert_not_awaited()


class TestStartStop:
    @pytest.mark.asyncio
    async def test_start_creates_background_task(self) -> None:
        reaper = LeaseReaper()

        with patch("aegra_api.services.lease_reaper.settings") as mock_settings:
            mock_settings.worker.REAPER_INTERVAL_SECONDS = 60

            await reaper.start()

        assert reaper._task is not None
        assert not reaper._task.done()

        # Cleanup
        await reaper.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_background_task(self) -> None:
        reaper = LeaseReaper()

        with patch("aegra_api.services.lease_reaper.settings") as mock_settings:
            mock_settings.worker.REAPER_INTERVAL_SECONDS = 60

            await reaper.start()
            task = reaper._task
            await reaper.stop()

        assert reaper._task is None
        assert task is not None
        assert task.done()

    @pytest.mark.asyncio
    async def test_stop_noop_when_not_started(self) -> None:
        reaper = LeaseReaper()
        # Should not raise
        await reaper.stop()
        assert reaper._task is None


class TestStrandedQueued:
    @pytest.mark.asyncio
    async def test_find_returns_only_threads_without_active_run(self) -> None:
        session = AsyncMock()
        queued_res = MagicMock()
        queued_res.all.return_value = [("t1",), ("t2",)]
        active_res = MagicMock()
        active_res.all.return_value = [("t2",)]  # t2 still has a running/pending run
        session.execute = AsyncMock(side_effect=[queued_res, active_res])

        with patch("aegra_api.services.lease_reaper._get_session_maker", return_value=_make_session_maker(session)):
            result = await LeaseReaper._find_stranded_queued_threads()

        assert set(result) == {"t1"}

    @pytest.mark.asyncio
    async def test_find_short_circuits_when_no_queued(self) -> None:
        session = AsyncMock()
        queued_res = MagicMock()
        queued_res.all.return_value = []
        session.execute = AsyncMock(side_effect=[queued_res])

        with patch("aegra_api.services.lease_reaper._get_session_maker", return_value=_make_session_maker(session)):
            result = await LeaseReaper._find_stranded_queued_threads()

        assert result == []
        assert session.execute.await_count == 1  # no active-run query when nothing queued

    @pytest.mark.asyncio
    async def test_dispatch_calls_executor_per_thread(self) -> None:
        with patch("aegra_api.services.executor.executor") as ex:
            ex.dispatch_next_for_thread = AsyncMock()
            await LeaseReaper._dispatch_stranded_queued(["t1", "t2"])

        assert ex.dispatch_next_for_thread.await_count == 2

    @pytest.mark.asyncio
    async def test_one_thread_failure_does_not_abort_batch(self) -> None:
        with patch("aegra_api.services.executor.executor") as ex:
            ex.dispatch_next_for_thread = AsyncMock(side_effect=[RedisError("boom"), None])
            await LeaseReaper._dispatch_stranded_queued(["t1", "t2"])

        assert ex.dispatch_next_for_thread.await_count == 2  # second thread still attempted

    @pytest.mark.asyncio
    async def test_reap_dispatches_stranded_queued(self) -> None:
        reaper = LeaseReaper()
        with (
            patch.object(LeaseReaper, "_find_recoverable", new_callable=AsyncMock, return_value=([], [])),
            patch.object(LeaseReaper, "_find_stranded_queued_threads", new_callable=AsyncMock, return_value=["t1"]),
            patch.object(LeaseReaper, "_dispatch_stranded_queued", new_callable=AsyncMock) as mock_dispatch,
        ):
            await reaper._reap()

        mock_dispatch.assert_awaited_once_with(["t1"])

    @pytest.mark.asyncio
    async def test_stranded_scan_runs_after_crashed_handling(self) -> None:
        """Stranded scan must run after mark-failed so a freshly-failed head's successor is caught."""
        reaper = LeaseReaper()
        order: list[str] = []

        async def _mark(_ids: list[str]) -> None:
            order.append("mark_failed")

        async def _find_stranded() -> list[str]:
            order.append("find_stranded")
            return []

        with (
            patch.object(LeaseReaper, "_find_recoverable", new_callable=AsyncMock, return_value=(["r1"], [])),
            patch.object(LeaseReaper, "_reset_to_pending", new_callable=AsyncMock, return_value=["r1"]),
            patch.object(LeaseReaper, "_check_retry_limits", new_callable=AsyncMock, return_value=([], ["r1"])),
            patch.object(LeaseReaper, "_mark_permanently_failed", side_effect=_mark),
            patch.object(LeaseReaper, "_find_stranded_queued_threads", side_effect=_find_stranded),
        ):
            await reaper._reap()

        assert order == ["mark_failed", "find_stranded"]

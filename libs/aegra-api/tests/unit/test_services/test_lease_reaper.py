"""Unit tests for lease_reaper service."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis import RedisError

from aegra_api.observability.metrics import REAPER_RECOVERED_RUNS
from aegra_api.services.lease_reaper import LeaseReaper


def _recovered_count(outcome: str) -> float:
    """Read the current value of the reaper counter for one outcome label."""
    return REAPER_RECOVERED_RUNS.labels(outcome=outcome)._value.get()


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


class TestRecoverCrashedRuns:
    @pytest.mark.asyncio
    async def test_classifies_and_transitions_under_one_transaction(self) -> None:
        session = AsyncMock()
        locked = MagicMock()
        locked.fetchall.return_value = [
            ("run-1", "thread-1", "user-1", {"_retry_count": 0}),
            ("run-2", "thread-2", "user-2", {"_retry_count": 1}),
        ]
        updated_run_1 = MagicMock()
        updated_run_1.scalar_one_or_none.return_value = "run-1"
        updated_run_2 = MagicMock()
        updated_run_2.scalar_one_or_none.return_value = "run-2"
        session.execute = AsyncMock(side_effect=[locked, updated_run_1, updated_run_2])
        session.commit = AsyncMock()
        maker = _make_session_maker(session)

        with (
            patch("aegra_api.services.lease_reaper._get_session_maker", return_value=maker),
            patch("aegra_api.services.lease_reaper.settings") as mock_settings,
            patch(
                "aegra_api.services.lease_reaper.set_thread_status_if_no_active_runs",
                new_callable=AsyncMock,
            ) as mock_set_thread,
        ):
            mock_settings.worker.BG_JOB_MAX_RETRIES = 1
            retryable, exhausted = await LeaseReaper._recover_crashed_runs(["run-1", "run-2"])

        assert retryable == ["run-1"]
        assert exhausted == ["run-2"]
        for call in session.execute.await_args_list[1:]:
            compiled = call.args[0].compile()
            assert "runs.user_id" in str(compiled)
        mock_set_thread.assert_awaited_once_with(session, {"thread-2"}, "error", user_id="user-2")
        session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_returns_empty_when_rows_are_no_longer_expired(self) -> None:
        session = AsyncMock()
        locked = MagicMock()
        locked.fetchall.return_value = []
        session.execute = AsyncMock(return_value=locked)
        session.commit = AsyncMock()
        maker = _make_session_maker(session)

        with patch("aegra_api.services.lease_reaper._get_session_maker", return_value=maker):
            retryable, exhausted = await LeaseReaper._recover_crashed_runs(["run-1"])

        assert retryable == []
        assert exhausted == []
        session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_skips_run_when_guarded_transition_loses_race(self) -> None:
        session = AsyncMock()
        locked = MagicMock()
        locked.fetchall.return_value = [
            ("run-1", "thread-1", "user-1", {"_retry_count": 1}),
        ]
        unchanged = MagicMock()
        unchanged.scalar_one_or_none.return_value = None
        session.execute = AsyncMock(side_effect=[locked, unchanged])
        session.commit = AsyncMock()
        maker = _make_session_maker(session)

        with (
            patch("aegra_api.services.lease_reaper._get_session_maker", return_value=maker),
            patch("aegra_api.services.lease_reaper.settings") as mock_settings,
            patch(
                "aegra_api.services.lease_reaper.set_thread_status_if_no_active_runs",
                new_callable=AsyncMock,
            ) as mock_set_thread,
        ):
            mock_settings.worker.BG_JOB_MAX_RETRIES = 1
            retryable, exhausted = await LeaseReaper._recover_crashed_runs(["run-1"])

        assert retryable == []
        assert exhausted == []
        mock_set_thread.assert_not_awaited()
        session.commit.assert_awaited_once()


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

            pushed = await LeaseReaper._reenqueue(["run-1", "run-2"])

        assert mock_client.rpush.await_count == 2
        assert pushed == ["run-1", "run-2"]

    @pytest.mark.asyncio
    async def test_returns_empty_when_redis_unavailable(self) -> None:
        with (
            patch("aegra_api.services.lease_reaper.redis_manager") as mock_rm,
            patch("aegra_api.services.lease_reaper.settings") as mock_settings,
        ):
            mock_settings.worker.WORKER_QUEUE_KEY = "aegra:jobs"
            mock_rm.get_client.side_effect = RedisError("connection refused")

            # Should not raise
            pushed = await LeaseReaper._reenqueue(["run-1"])

        assert pushed == []

    @pytest.mark.asyncio
    async def test_returns_partial_batch_when_redis_fails_mid_push(self) -> None:
        """Only IDs pushed before the failure count as confirmed."""
        mock_client = AsyncMock()
        mock_client.rpush = AsyncMock(side_effect=[1, RedisError("connection reset")])

        with (
            patch("aegra_api.services.lease_reaper.redis_manager") as mock_rm,
            patch("aegra_api.services.lease_reaper.settings") as mock_settings,
        ):
            mock_settings.worker.WORKER_QUEUE_KEY = "aegra:jobs"
            mock_rm.get_client.return_value = mock_client

            pushed = await LeaseReaper._reenqueue(["run-1", "run-2", "run-3"])

        assert pushed == ["run-1"]

    @pytest.mark.asyncio
    async def test_noop_when_empty_list(self) -> None:
        mock_client = AsyncMock()

        with (
            patch("aegra_api.services.lease_reaper.redis_manager") as mock_rm,
            patch("aegra_api.services.lease_reaper.settings") as mock_settings,
        ):
            mock_settings.worker.WORKER_QUEUE_KEY = "aegra:jobs"
            mock_rm.get_client.return_value = mock_client

            pushed = await LeaseReaper._reenqueue([])

        mock_client.rpush.assert_not_awaited()
        assert pushed == []


class TestReap:
    @pytest.fixture(autouse=True)
    def _no_stranded_queues(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """_reap also scans for stranded queued threads; keep that off the database here."""
        monkeypatch.setattr(LeaseReaper, "_find_stranded_queued_threads", AsyncMock(return_value=[]))

    @pytest.mark.asyncio
    async def test_crashed_runs_are_classified_before_becoming_claimable(self) -> None:
        reaper = LeaseReaper()

        with (
            patch.object(
                LeaseReaper, "_find_recoverable", new_callable=AsyncMock, return_value=(["run-1", "run-2"], [])
            ),
            patch.object(LeaseReaper, "_find_stranded_queued_threads", new_callable=AsyncMock, return_value=[]),
            patch.object(
                LeaseReaper, "_recover_crashed_runs", new_callable=AsyncMock, return_value=(["run-1"], ["run-2"])
            ) as mock_recover,
            patch.object(LeaseReaper, "_reenqueue", new_callable=AsyncMock, return_value=["run-1"]) as mock_reenqueue,
        ):
            await reaper._reap()

        mock_recover.assert_awaited_once_with(["run-1", "run-2"])
        mock_reenqueue.assert_awaited_once_with(["run-1"])

    @pytest.mark.asyncio
    async def test_stuck_pending_reenqueued_without_retry_charge(self) -> None:
        """Stuck pending runs are re-enqueued directly, no retry count increment."""
        reaper = LeaseReaper()

        with (
            patch.object(LeaseReaper, "_find_recoverable", new_callable=AsyncMock, return_value=([], ["run-3"])),
            patch.object(LeaseReaper, "_recover_crashed_runs", new_callable=AsyncMock) as mock_recover,
            patch.object(LeaseReaper, "_reenqueue", new_callable=AsyncMock, return_value=["run-3"]) as mock_reenqueue,
        ):
            await reaper._reap()

        mock_recover.assert_not_awaited()
        mock_reenqueue.assert_awaited_once_with(["run-3"])

    @pytest.mark.asyncio
    async def test_skips_when_nothing_to_recover(self) -> None:
        reaper = LeaseReaper()

        with (
            patch.object(LeaseReaper, "_find_recoverable", new_callable=AsyncMock, return_value=([], [])),
            patch.object(LeaseReaper, "_recover_crashed_runs", new_callable=AsyncMock) as mock_recover,
            patch.object(LeaseReaper, "_reenqueue", new_callable=AsyncMock) as mock_reenqueue,
        ):
            await reaper._reap()

        mock_recover.assert_not_awaited()
        mock_reenqueue.assert_not_awaited()


class TestReapMetrics:
    @pytest.fixture(autouse=True)
    def _no_stranded_queues(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """_reap also scans for stranded queued threads; keep that off the database here."""
        monkeypatch.setattr(LeaseReaper, "_find_stranded_queued_threads", AsyncMock(return_value=[]))

    @pytest.mark.asyncio
    async def test_increments_counters_per_outcome_on_crashed_recovery(self) -> None:
        """Retried and exhausted crashed runs each increment their own outcome series."""
        reaper = LeaseReaper()
        retried_before = _recovered_count("crashed_retried")
        exhausted_before = _recovered_count("crashed_exhausted")

        with (
            patch.object(
                LeaseReaper, "_find_recoverable", new_callable=AsyncMock, return_value=(["run-1", "run-2"], [])
            ),
            patch.object(
                LeaseReaper, "_recover_crashed_runs", new_callable=AsyncMock, return_value=(["run-1"], ["run-2"])
            ),
            patch.object(LeaseReaper, "_reenqueue", new_callable=AsyncMock, return_value=["run-1"]),
        ):
            await reaper._reap()

        assert _recovered_count("crashed_retried") == retried_before + 1
        assert _recovered_count("crashed_exhausted") == exhausted_before + 1

    @pytest.mark.asyncio
    async def test_increments_stuck_pending_by_batch_size(self) -> None:
        reaper = LeaseReaper()
        before = _recovered_count("stuck_pending")

        with (
            patch.object(
                LeaseReaper, "_find_recoverable", new_callable=AsyncMock, return_value=([], ["run-3", "run-4"])
            ),
            patch.object(LeaseReaper, "_reenqueue", new_callable=AsyncMock, return_value=["run-3", "run-4"]),
        ):
            await reaper._reap()

        assert _recovered_count("stuck_pending") == before + 2

    @pytest.mark.asyncio
    async def test_no_increment_when_nothing_to_recover(self) -> None:
        reaper = LeaseReaper()
        before = {o: _recovered_count(o) for o in ("crashed_retried", "crashed_exhausted", "stuck_pending")}

        with patch.object(LeaseReaper, "_find_recoverable", new_callable=AsyncMock, return_value=([], [])):
            await reaper._reap()

        for outcome, value in before.items():
            assert _recovered_count(outcome) == value

    @pytest.mark.asyncio
    async def test_no_increment_when_all_crashed_claimed_elsewhere(self) -> None:
        """Runs found crashed but re-claimed before reset must not count as recovered."""
        reaper = LeaseReaper()
        before = _recovered_count("crashed_retried")

        with (
            patch.object(LeaseReaper, "_find_recoverable", new_callable=AsyncMock, return_value=(["run-1"], [])),
            patch.object(
                LeaseReaper, "_recover_crashed_runs", new_callable=AsyncMock, return_value=([], [])
            ) as mock_recover,
        ):
            await reaper._reap()

        mock_recover.assert_awaited_once_with(["run-1"])
        assert _recovered_count("crashed_retried") == before

    @pytest.mark.asyncio
    async def test_stuck_pending_not_counted_when_push_unconfirmed(self) -> None:
        """Redis down during re-enqueue: recovery falls back to PG poll, counter stays put."""
        reaper = LeaseReaper()
        before = _recovered_count("stuck_pending")

        with (
            patch.object(LeaseReaper, "_find_recoverable", new_callable=AsyncMock, return_value=([], ["run-3"])),
            patch.object(LeaseReaper, "_reenqueue", new_callable=AsyncMock, return_value=[]),
        ):
            await reaper._reap()

        assert _recovered_count("stuck_pending") == before

    @pytest.mark.asyncio
    async def test_crashed_retried_counts_only_confirmed_pushes(self) -> None:
        """Partial Redis push mid-batch: only confirmed IDs increment the counter."""
        reaper = LeaseReaper()
        before = _recovered_count("crashed_retried")

        with (
            patch.object(
                LeaseReaper, "_find_recoverable", new_callable=AsyncMock, return_value=(["run-1", "run-2"], [])
            ),
            patch.object(
                LeaseReaper, "_recover_crashed_runs", new_callable=AsyncMock, return_value=(["run-1", "run-2"], [])
            ),
            patch.object(LeaseReaper, "_reenqueue", new_callable=AsyncMock, return_value=["run-1"]),
        ):
            await reaper._reap()

        assert _recovered_count("crashed_retried") == before + 1

    @pytest.mark.asyncio
    async def test_crashed_exhausted_counts_only_rows_atomically_failed(self) -> None:
        reaper = LeaseReaper()
        before = _recovered_count("crashed_exhausted")

        with (
            patch.object(LeaseReaper, "_find_recoverable", new_callable=AsyncMock, return_value=(["run-1"], [])),
            patch.object(LeaseReaper, "_recover_crashed_runs", new_callable=AsyncMock, return_value=([], [])),
        ):
            await reaper._reap()

        assert _recovered_count("crashed_exhausted") == before


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
            # Infra errors and a corrupt-row KeyError alike: the remaining threads still get their turn.
            ex.dispatch_next_for_thread = AsyncMock(side_effect=[RedisError("boom"), KeyError("graph_id"), None])
            await LeaseReaper._dispatch_stranded_queued(["t1", "t2", "t3"])

        assert ex.dispatch_next_for_thread.await_count == 3  # later threads still attempted

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
        """Stranded scan must run after crashed recovery so a freshly-failed head's successor is caught."""
        reaper = LeaseReaper()
        order: list[str] = []

        async def _recover(_ids: list[str]) -> tuple[list[str], list[str]]:
            order.append("recover_crashed")
            return [], ["r1"]  # r1 exhausted its retries: its thread is free now

        async def _find_stranded() -> list[str]:
            order.append("stranded_scan")
            return []

        with (
            patch.object(LeaseReaper, "_find_recoverable", new_callable=AsyncMock, return_value=(["r1"], [])),
            patch.object(LeaseReaper, "_recover_crashed_runs", AsyncMock(side_effect=_recover)),
            patch.object(LeaseReaper, "_find_stranded_queued_threads", AsyncMock(side_effect=_find_stranded)),
            patch.object(LeaseReaper, "_reenqueue", new_callable=AsyncMock, return_value=[]),
        ):
            await reaper._reap()

        assert order == ["recover_crashed", "stranded_scan"]

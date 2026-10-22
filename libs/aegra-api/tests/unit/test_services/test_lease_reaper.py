"""Unit tests for lease_reaper service."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis import RedisError
from sqlalchemy.dialects import postgresql

from aegra_api.observability.metrics import REAPER_RECOVERED_RUNS
from aegra_api.services.lease_reaper import _SWEEP_LOCK_RETRIES, _SWEEP_MAX_BATCHES, LeaseReaper


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


class TestMissingFromQueue:
    @pytest.mark.asyncio
    async def test_returns_only_runs_absent_from_the_queue(self) -> None:
        mock_client = AsyncMock()
        mock_client.lrange = AsyncMock(return_value=["run-1", "run-9"])

        with (
            patch("aegra_api.services.lease_reaper.redis_manager") as mock_rm,
            patch("aegra_api.services.lease_reaper.settings") as mock_settings,
        ):
            mock_settings.worker.WORKER_QUEUE_KEY = "aegra:jobs"
            mock_rm.get_client.return_value = mock_client

            missing = await LeaseReaper._missing_from_queue(["run-1", "run-2"])

        mock_client.lrange.assert_awaited_once_with("aegra:jobs", 0, -1)
        assert missing == ["run-2"]

    @pytest.mark.asyncio
    async def test_returns_empty_when_redis_unavailable(self) -> None:
        mock_client = AsyncMock()
        mock_client.lrange = AsyncMock(side_effect=RedisError("connection reset"))

        with patch("aegra_api.services.lease_reaper.redis_manager") as mock_rm:
            mock_rm.get_client.return_value = mock_client

            missing = await LeaseReaper._missing_from_queue(["run-1"])

        assert missing == []


class TestReap:
    @pytest.mark.asyncio
    async def test_crashed_runs_are_classified_before_becoming_claimable(self) -> None:
        reaper = LeaseReaper()

        with (
            patch.object(
                LeaseReaper, "_find_recoverable", new_callable=AsyncMock, return_value=(["run-1", "run-2"], [])
            ),
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
            patch.object(LeaseReaper, "_missing_from_queue", new_callable=AsyncMock, return_value=["run-3"]),
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
            patch.object(LeaseReaper, "_missing_from_queue", new_callable=AsyncMock, return_value=["run-3", "run-4"]),
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
            patch.object(LeaseReaper, "_missing_from_queue", new_callable=AsyncMock, return_value=["run-3"]),
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


class _FakeQueue:
    """The job queue as a list: RPUSH appends, LRANGE reads, and nothing drains it."""

    def __init__(self, entries: list[str] | None = None) -> None:
        self.entries: list[str] = list(entries or [])

    async def rpush(self, _key: str, value: str) -> int:
        self.entries.append(value)
        return len(self.entries)

    async def lrange(self, _key: str, start: int, end: int) -> list[str]:
        assert (start, end) == (0, -1)
        return list(self.entries)


async def _sweep(queue: _FakeQueue, stuck: list[str], *, sweeps: int, replicas: int = 1) -> None:
    """Run the real stuck-pending path while every worker stays busy, so the same runs stay stuck."""
    reapers = [LeaseReaper() for _ in range(replicas)]
    with (
        patch.object(LeaseReaper, "_find_recoverable", new_callable=AsyncMock, return_value=([], stuck)),
        patch("aegra_api.services.lease_reaper.redis_manager") as mock_rm,
    ):
        mock_rm.get_client.return_value = queue
        for _ in range(sweeps):
            for reaper in reapers:
                await reaper._reap()


class TestStuckPendingBacklog:
    @pytest.mark.asyncio
    async def test_run_lost_from_queue_is_pushed_once_across_sweeps(self) -> None:
        queue = _FakeQueue()

        await _sweep(queue, ["run-1"], sweeps=100)

        assert queue.entries == ["run-1"]

    @pytest.mark.asyncio
    async def test_queued_backlog_is_not_pushed_again(self) -> None:
        backlog = [f"run-{i}" for i in range(60)]
        queue = _FakeQueue(backlog)

        # 15 minutes of 15-second sweeps on two replicas
        await _sweep(queue, backlog, sweeps=60, replicas=2)

        assert queue.entries == backlog


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


class TestSweepOneBatch:
    @pytest.mark.asyncio
    async def test_marks_orphan_error_and_reconciles_its_thread(self: "TestSweepOneBatch") -> None:
        session = AsyncMock()
        locked = MagicMock()
        locked.fetchall.return_value = [("run-1", "thread-1", "user-1")]
        updated = MagicMock()
        updated.scalar_one_or_none.return_value = "run-1"
        session.execute = AsyncMock(side_effect=[locked, updated, MagicMock(), MagicMock()])
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
            mock_settings.worker.ORPHAN_SWEEP_MIN_AGE_SECONDS = 300
            seen, swept = await LeaseReaper._sweep_one_batch()

        assert (seen, swept) == (1, ["run-1"])
        mock_set_thread.assert_awaited_once_with(session, {"thread-1"}, "error", user_id="user-1")
        session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_locks_swept_threads_in_order_under_a_lock_timeout_before_reconciling(
        self: "TestSweepOneBatch",
    ) -> None:
        session = AsyncMock()
        locked = MagicMock()
        locked.fetchall.return_value = [("run-1", "thread-b", "user-1"), ("run-2", "thread-a", "user-2")]
        updated = [MagicMock(), MagicMock()]
        updated[0].scalar_one_or_none.return_value = "run-1"
        updated[1].scalar_one_or_none.return_value = "run-2"
        session.execute = AsyncMock(side_effect=[locked, *updated, MagicMock(), MagicMock()])
        session.commit = AsyncMock()
        maker = _make_session_maker(session)

        with (
            patch("aegra_api.services.lease_reaper._get_session_maker", return_value=maker),
            patch("aegra_api.services.lease_reaper.settings") as mock_settings,
            patch(
                "aegra_api.services.lease_reaper.set_thread_status_if_no_active_runs",
                new_callable=AsyncMock,
            ),
        ):
            mock_settings.worker.ORPHAN_SWEEP_MIN_AGE_SECONDS = 300
            await LeaseReaper._sweep_one_batch()

        timeout_stmt, lock_stmt = (call.args[0] for call in session.execute.await_args_list[3:5])
        assert "set_config" in str(timeout_stmt.compile(dialect=postgresql.dialect()))
        assert "lock_timeout" in timeout_stmt.compile().params.values()
        compiled_lock = str(lock_stmt.compile(dialect=postgresql.dialect()))
        assert "ORDER BY thread.thread_id" in compiled_lock
        assert compiled_lock.endswith("FOR UPDATE")
        assert ["thread-a", "thread-b"] in lock_stmt.compile().params.values()

    @pytest.mark.asyncio
    async def test_selects_only_running_rows_that_never_held_a_lease(self: "TestSweepOneBatch") -> None:
        session = AsyncMock()
        locked = MagicMock()
        locked.fetchall.return_value = []
        session.execute = AsyncMock(return_value=locked)
        session.commit = AsyncMock()
        maker = _make_session_maker(session)

        with (
            patch("aegra_api.services.lease_reaper._get_session_maker", return_value=maker),
            patch("aegra_api.services.lease_reaper.settings") as mock_settings,
            patch(
                "aegra_api.services.lease_reaper.set_thread_status_if_no_active_runs",
                new_callable=AsyncMock,
            ),
        ):
            mock_settings.worker.ORPHAN_SWEEP_MIN_AGE_SECONDS = 300
            await LeaseReaper._sweep_one_batch()

        statement = session.execute.await_args_list[0].args[0]
        compiled = str(statement.compile(dialect=postgresql.dialect()))
        assert "runs.claimed_by IS NULL" in compiled
        assert "runs.lease_expires_at IS NULL" in compiled
        assert "runs.updated_at <" in compiled
        assert "FOR UPDATE SKIP LOCKED" in compiled

    @pytest.mark.asyncio
    async def test_counts_a_row_seen_but_not_swept_when_its_update_loses(self: "TestSweepOneBatch") -> None:
        session = AsyncMock()
        locked = MagicMock()
        locked.fetchall.return_value = [("run-1", "thread-1", "user-1")]
        updated = MagicMock()
        updated.scalar_one_or_none.return_value = None
        session.execute = AsyncMock(side_effect=[locked, updated])
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
            mock_settings.worker.ORPHAN_SWEEP_MIN_AGE_SECONDS = 300
            seen, swept = await LeaseReaper._sweep_one_batch()

        assert (seen, swept) == (1, [])
        mock_set_thread.assert_not_awaited()


class TestSweepLeaselessOrphans:
    @pytest.mark.asyncio
    async def test_keeps_batching_until_a_batch_sees_nothing(self: "TestSweepLeaselessOrphans") -> None:
        batches = [(2, ["run-1", "run-2"]), (1, ["run-3"]), (0, [])]
        before = _recovered_count("local_orphan")

        with (
            patch("aegra_api.services.lease_reaper.settings") as mock_settings,
            patch.object(LeaseReaper, "_sweep_one_batch", new_callable=AsyncMock, side_effect=batches) as mock_batch,
            patch.object(LeaseReaper, "_count_leaseless_orphans", new_callable=AsyncMock, return_value=0),
        ):
            mock_settings.worker.ORPHAN_SWEEP_ENABLED = True
            total = await LeaseReaper.sweep_leaseless_orphans()

        assert total == 3
        assert mock_batch.await_count == 3
        assert _recovered_count("local_orphan") == before + 3

    @pytest.mark.asyncio
    async def test_stops_at_the_batch_limit_instead_of_looping_forever(self: "TestSweepLeaselessOrphans") -> None:
        with (
            patch("aegra_api.services.lease_reaper.settings") as mock_settings,
            patch.object(LeaseReaper, "_sweep_one_batch", new_callable=AsyncMock, return_value=(1, [])) as mock_batch,
        ):
            mock_settings.worker.ORPHAN_SWEEP_ENABLED = True
            total = await LeaseReaper.sweep_leaseless_orphans()

        assert total == 0
        assert mock_batch.await_count == _SWEEP_MAX_BATCHES

    @pytest.mark.asyncio
    async def test_does_not_touch_the_database_when_disabled(self: "TestSweepLeaselessOrphans") -> None:
        with (
            patch("aegra_api.services.lease_reaper.settings") as mock_settings,
            patch.object(LeaseReaper, "_sweep_one_batch", new_callable=AsyncMock) as mock_batch,
        ):
            mock_settings.worker.ORPHAN_SWEEP_ENABLED = False
            total = await LeaseReaper.sweep_leaseless_orphans()

        assert total == 0
        mock_batch.assert_not_awaited()


class TestSweepWithLockedRows:
    """SKIP LOCKED returns nothing when every orphan is locked, which is not the same as none left."""

    @pytest.mark.asyncio
    async def test_retries_when_an_empty_batch_still_leaves_orphans(self: "TestSweepWithLockedRows") -> None:
        batches = [(0, []), (1, ["run-1"]), (0, [])]

        with (
            patch("aegra_api.services.lease_reaper.settings") as mock_settings,
            patch.object(LeaseReaper, "_sweep_one_batch", new_callable=AsyncMock, side_effect=batches) as mock_batch,
            patch.object(LeaseReaper, "_count_leaseless_orphans", new_callable=AsyncMock, side_effect=[1, 0]),
            patch("aegra_api.services.lease_reaper.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            mock_settings.worker.ORPHAN_SWEEP_ENABLED = True
            total = await LeaseReaper.sweep_leaseless_orphans()

        assert total == 1
        assert mock_batch.await_count == 3
        mock_sleep.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_gives_up_with_a_warning_when_orphans_stay_locked(self: "TestSweepWithLockedRows") -> None:
        with (
            patch("aegra_api.services.lease_reaper.settings") as mock_settings,
            patch.object(LeaseReaper, "_sweep_one_batch", new_callable=AsyncMock, return_value=(0, [])) as mock_batch,
            patch.object(LeaseReaper, "_count_leaseless_orphans", new_callable=AsyncMock, return_value=4),
            patch("aegra_api.services.lease_reaper.asyncio.sleep", new_callable=AsyncMock),
            patch("aegra_api.services.lease_reaper.logger.warning") as mock_warning,
        ):
            mock_settings.worker.ORPHAN_SWEEP_ENABLED = True
            total = await LeaseReaper.sweep_leaseless_orphans()

        assert total == 0
        assert mock_batch.await_count == _SWEEP_LOCK_RETRIES + 1
        mock_warning.assert_any_call(
            "Orphan sweep left runs that stayed locked; the next startup retries them",
            count=4,
        )

    @pytest.mark.asyncio
    async def test_stops_without_retrying_when_nothing_is_left(self: "TestSweepWithLockedRows") -> None:
        with (
            patch("aegra_api.services.lease_reaper.settings") as mock_settings,
            patch.object(LeaseReaper, "_sweep_one_batch", new_callable=AsyncMock, return_value=(0, [])) as mock_batch,
            patch.object(LeaseReaper, "_count_leaseless_orphans", new_callable=AsyncMock, return_value=0),
            patch("aegra_api.services.lease_reaper.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            mock_settings.worker.ORPHAN_SWEEP_ENABLED = True
            await LeaseReaper.sweep_leaseless_orphans()

        assert mock_batch.await_count == 1
        mock_sleep.assert_not_awaited()

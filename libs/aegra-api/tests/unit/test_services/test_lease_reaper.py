"""Unit tests for lease_reaper service."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import prometheus_client
import pytest
from redis import RedisError
from sqlalchemy.exc import SQLAlchemyError

from aegra_api.observability import metrics as metrics_module
from aegra_api.observability.metrics import get_worker_metrics, setup_worker_metrics
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


def _make_run_orm_default(run_id: str) -> MagicMock:
    """Build a RunORM-shaped MagicMock with valid default execution_params."""
    run_orm = MagicMock()
    run_orm.run_id = run_id
    run_orm.execution_params = {"graph_id": "g", "trace": {}}
    return run_orm


def _make_run_orm(run_id: str, *, execution_params: dict[str, Any] | None) -> MagicMock:
    """Build a RunORM-shaped MagicMock with the explicit execution_params.

    Pass ``execution_params=None`` to simulate data corruption (no params).
    """
    run_orm = MagicMock()
    run_orm.run_id = run_id
    run_orm.execution_params = execution_params
    return run_orm


class TestReenqueue:
    @pytest.mark.asyncio
    async def test_pushes_to_redis(self) -> None:
        mock_client = AsyncMock()
        session = AsyncMock()
        session.scalar = AsyncMock(side_effect=[_make_run_orm_default("run-1"), _make_run_orm_default("run-2")])
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        maker = _make_session_maker(session)

        with (
            patch("aegra_api.services.lease_reaper._get_session_maker", return_value=maker),
            patch("aegra_api.services.lease_reaper.redis_manager") as mock_rm,
            patch("aegra_api.services.lease_reaper.settings") as mock_settings,
        ):
            mock_settings.worker.WORKER_QUEUE_KEY = "aegra:jobs"
            mock_rm.get_client.return_value = mock_client

            await LeaseReaper._reenqueue(["run-1", "run-2"])

        assert mock_client.rpush.await_count == 2

    @pytest.mark.asyncio
    async def test_logs_warning_when_redis_unavailable(self) -> None:
        session = AsyncMock()
        session.scalar = AsyncMock(return_value=_make_run_orm_default("run-1"))
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        maker = _make_session_maker(session)

        with (
            patch("aegra_api.services.lease_reaper._get_session_maker", return_value=maker),
            patch("aegra_api.services.lease_reaper.redis_manager") as mock_rm,
            patch("aegra_api.services.lease_reaper.settings") as mock_settings,
        ):
            mock_settings.worker.WORKER_QUEUE_KEY = "aegra:jobs"
            mock_rm.get_client.side_effect = RedisError("connection refused")

            # Should not raise
            await LeaseReaper._reenqueue(["run-1"])

    @pytest.mark.asyncio
    async def test_skips_rpush_when_db_update_fails(self) -> None:
        """Per-run try/except: a SQLAlchemyError on one run must not block the others."""

        mock_client = AsyncMock()

        # First call: raise (first run fails the SELECT). Second call: succeed.
        session = AsyncMock()
        session.scalar = AsyncMock(side_effect=[SQLAlchemyError("deadlock"), _make_run_orm_default("run-2")])
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        maker = _make_session_maker(session)

        with (
            patch("aegra_api.services.lease_reaper._get_session_maker", return_value=maker),
            patch("aegra_api.services.lease_reaper.redis_manager") as mock_rm,
            patch("aegra_api.services.lease_reaper.settings") as mock_settings,
        ):
            mock_settings.worker.WORKER_QUEUE_KEY = "aegra:jobs"
            mock_rm.get_client.return_value = mock_client

            await LeaseReaper._reenqueue(["run-1", "run-2"])

        # Only run-2 should be rpush'ed — run-1 silently failed
        assert mock_client.rpush.await_count == 1
        rpush_args = mock_client.rpush.await_args_list[0].args
        assert rpush_args[1] == "run-2"

    @pytest.mark.asyncio
    async def test_skips_rpush_when_execution_params_missing(self) -> None:
        """Runs with no execution_params (data corruption) are skipped, others continue."""
        mock_client = AsyncMock()
        session = AsyncMock()
        session.scalar = AsyncMock(
            side_effect=[
                _make_run_orm("run-1", execution_params=None),  # missing
                _make_run_orm_default("run-2"),  # ok
            ]
        )
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        maker = _make_session_maker(session)

        with (
            patch("aegra_api.services.lease_reaper._get_session_maker", return_value=maker),
            patch("aegra_api.services.lease_reaper.redis_manager") as mock_rm,
            patch("aegra_api.services.lease_reaper.settings") as mock_settings,
        ):
            mock_settings.worker.WORKER_QUEUE_KEY = "aegra:jobs"
            mock_rm.get_client.return_value = mock_client

            await LeaseReaper._reenqueue(["run-1", "run-2"])

        assert mock_client.rpush.await_count == 1

    @pytest.mark.asyncio
    async def test_increments_runs_dispatched_per_rpush(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Identity invariant: every retry-driven rpush must bump ``runs_dispatched``
        with the run's graph_id, mirroring ``WorkerExecutor.submit``. Without this
        the invariant ``dispatched == completed + permanently_failed + in_flight + discarded``
        breaks for every retried run."""
        registry = prometheus_client.CollectorRegistry()
        monkeypatch.setattr(metrics_module, "_worker_metrics", None)
        monkeypatch.setattr(metrics_module, "_reaper_metrics", None)
        setup_worker_metrics(registry=registry)
        worker_metrics = get_worker_metrics()
        assert worker_metrics is not None

        mock_client = AsyncMock()
        session = AsyncMock()
        session.scalar = AsyncMock(
            side_effect=[
                _make_run_orm("run-1", execution_params={"graph_id": "graph-a"}),
                _make_run_orm("run-2", execution_params={"graph_id": "graph-b"}),
            ]
        )
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        maker = _make_session_maker(session)

        with (
            patch("aegra_api.services.lease_reaper._get_session_maker", return_value=maker),
            patch("aegra_api.services.lease_reaper.redis_manager") as mock_rm,
            patch("aegra_api.services.lease_reaper.settings") as mock_settings,
        ):
            mock_settings.worker.WORKER_QUEUE_KEY = "aegra:jobs"
            mock_rm.get_client.return_value = mock_client

            await LeaseReaper._reenqueue(["run-1", "run-2"], bump_dispatched=True)

        assert worker_metrics.runs_dispatched.labels(graph_id="graph-a")._value.get() == 1.0
        assert worker_metrics.runs_dispatched.labels(graph_id="graph-b")._value.get() == 1.0

    @pytest.mark.asyncio
    async def test_does_not_double_count_dispatched_when_bump_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Stuck-pending re-enqueue calls ``_reenqueue(bump_dispatched=False)``.
        The original submit already counted these runs; bumping again would
        break the identity invariant by double-counting."""
        registry = prometheus_client.CollectorRegistry()
        monkeypatch.setattr(metrics_module, "_worker_metrics", None)
        monkeypatch.setattr(metrics_module, "_reaper_metrics", None)
        setup_worker_metrics(registry=registry)
        worker_metrics = get_worker_metrics()
        assert worker_metrics is not None

        mock_client = AsyncMock()
        session = AsyncMock()
        session.scalar = AsyncMock(
            side_effect=[_make_run_orm("run-stuck", execution_params={"graph_id": "graph-stuck"})]
        )
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        maker = _make_session_maker(session)

        with (
            patch("aegra_api.services.lease_reaper._get_session_maker", return_value=maker),
            patch("aegra_api.services.lease_reaper.redis_manager") as mock_rm,
            patch("aegra_api.services.lease_reaper.settings") as mock_settings,
        ):
            mock_settings.worker.WORKER_QUEUE_KEY = "aegra:jobs"
            mock_rm.get_client.return_value = mock_client

            await LeaseReaper._reenqueue(["run-stuck"], bump_dispatched=False)

        # rpush still happened (the run reaches the queue).
        assert mock_client.rpush.await_count == 1
        # But dispatched counter must remain at 0 — the original submit counted.
        assert worker_metrics.runs_dispatched.labels(graph_id="graph-stuck")._value.get() == 0.0

    @pytest.mark.asyncio
    async def test_noop_when_empty_list(self) -> None:
        mock_client = AsyncMock()
        session = AsyncMock()
        session.scalar = AsyncMock(return_value=None)
        session.commit = AsyncMock()
        maker = _make_session_maker(session)

        with (
            patch("aegra_api.services.lease_reaper._get_session_maker", return_value=maker),
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
        # Crashed retries pass bump_dispatched=True (each retry is a fresh dispatch)
        mock_reenqueue.assert_awaited_once_with(["run-1"], bump_dispatched=True)
        mock_fail.assert_awaited_once_with(["run-2"])

    @pytest.mark.asyncio
    async def test_stuck_pending_reenqueued_without_retry_charge(self) -> None:
        """Stuck pending runs are re-enqueued directly, no retry count increment.
        The reaper passes ``bump_dispatched=False`` so the original submit isn't
        double-counted in ``runs_dispatched``."""
        reaper = LeaseReaper()

        with (
            patch.object(LeaseReaper, "_find_recoverable", new_callable=AsyncMock, return_value=([], ["run-3"])),
            patch.object(LeaseReaper, "_check_retry_limits", new_callable=AsyncMock) as mock_retry,
            patch.object(LeaseReaper, "_reenqueue", new_callable=AsyncMock) as mock_reenqueue,
        ):
            await reaper._reap()

        mock_retry.assert_not_awaited()
        mock_reenqueue.assert_awaited_once_with(["run-3"], bump_dispatched=False)

    @pytest.mark.asyncio
    async def test_crashed_retry_reenqueued_with_dispatched_bump(self) -> None:
        """Crashed-worker retries pass ``bump_dispatched=True`` so each retry is
        counted as a fresh dispatch event."""
        reaper = LeaseReaper()

        with (
            patch.object(LeaseReaper, "_find_recoverable", new_callable=AsyncMock, return_value=(["run-c"], [])),
            patch.object(LeaseReaper, "_reset_to_pending", new_callable=AsyncMock, return_value=["run-c"]),
            patch.object(LeaseReaper, "_check_retry_limits", new_callable=AsyncMock, return_value=(["run-c"], [])),
            patch.object(LeaseReaper, "_reenqueue", new_callable=AsyncMock) as mock_reenqueue,
        ):
            await reaper._reap()

        mock_reenqueue.assert_awaited_once_with(["run-c"], bump_dispatched=True)

    @pytest.mark.asyncio
    async def test_skips_when_nothing_to_recover(self) -> None:
        reaper = LeaseReaper()

        with (
            patch.object(LeaseReaper, "_find_recoverable", new_callable=AsyncMock, return_value=([], [])),
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


class TestBumpRetryCount:
    @pytest.mark.asyncio
    async def test_returns_none_when_run_missing(self) -> None:
        """Missing row → return None so the caller skips the run for this cycle."""
        session = AsyncMock()
        session.scalar = AsyncMock(return_value=None)
        maker = _make_session_maker(session)

        with patch("aegra_api.services.lease_reaper._get_session_maker", return_value=maker):
            outcome = await LeaseReaper._bump_retry_count("run-missing", max_retries=3)

        assert outcome is None
        session.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_exhausted_count_without_committing(self) -> None:
        """If new retry_count exceeds max, return the count + params but do
        NOT commit — the row keeps the previous _retry_count, and the caller
        marks the run as permanently failed via a separate UPDATE."""
        run_orm = MagicMock()
        run_orm.execution_params = {"_retry_count": 3, "graph_id": "g"}

        session = AsyncMock()
        session.scalar = AsyncMock(return_value=run_orm)
        maker = _make_session_maker(session)

        with patch("aegra_api.services.lease_reaper._get_session_maker", return_value=maker):
            outcome = await LeaseReaper._bump_retry_count("run-exhausted", max_retries=3)

        assert outcome is not None
        retry_count, params = outcome
        assert retry_count == 4
        assert params["_retry_count"] == 3  # not bumped on exhaustion
        session.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_commits_on_retryable_increment(self) -> None:
        """Within-budget retry: bump _retry_count, commit. Does NOT touch
        ``_enqueued_at`` — that refresh belongs to ``_update_enqueued_at``
        (called by ``_reenqueue`` right after) so a write here is a redundant
        field assignment immediately overwritten."""
        run_orm = MagicMock()
        run_orm.execution_params = {"_retry_count": 1, "graph_id": "g"}

        session = AsyncMock()
        session.scalar = AsyncMock(return_value=run_orm)
        session.execute = AsyncMock()
        maker = _make_session_maker(session)

        with patch("aegra_api.services.lease_reaper._get_session_maker", return_value=maker):
            outcome = await LeaseReaper._bump_retry_count("run-retry", max_retries=5)

        assert outcome is not None
        retry_count, params = outcome
        assert retry_count == 2
        assert params["_retry_count"] == 2
        assert "_enqueued_at" not in params
        session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_returns_none_on_sqlalchemy_error(self) -> None:
        """DB errors are swallowed (logged), the run is skipped this cycle."""
        session = AsyncMock()
        session.scalar = AsyncMock(side_effect=SQLAlchemyError("deadlock"))
        maker = _make_session_maker(session)

        with patch("aegra_api.services.lease_reaper._get_session_maker", return_value=maker):
            outcome = await LeaseReaper._bump_retry_count("run-error", max_retries=3)

        assert outcome is None

    @pytest.mark.asyncio
    async def test_returns_none_when_execution_params_missing(self) -> None:
        """Corrupted row (execution_params=None) must NOT be revived with a
        synthetic dict — a run with no graph/input metadata cannot be retried
        meaningfully. Mirror _update_enqueued_at: log + skip."""
        run_orm = MagicMock()
        run_orm.execution_params = None

        session = AsyncMock()
        session.scalar = AsyncMock(return_value=run_orm)
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        maker = _make_session_maker(session)

        with patch("aegra_api.services.lease_reaper._get_session_maker", return_value=maker):
            outcome = await LeaseReaper._bump_retry_count("run-corrupt", max_retries=3)

        assert outcome is None
        session.commit.assert_not_called()
        session.execute.assert_not_called()


class TestCheckRetryLimits:
    @pytest.mark.asyncio
    async def test_splits_runs_into_retryable_and_exhausted(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Each run is processed independently; retry_count > max → exhausted."""

        # Mock _bump_retry_count to return controlled outcomes per run.
        async def fake_bump(run_id: str, max_retries: int) -> tuple[int, dict[str, Any]] | None:
            if run_id == "run-retryable":
                return 2, {"graph_id": "graph-r"}
            if run_id == "run-exhausted":
                return 5, {"graph_id": "graph-r"}
            return None  # missing row

        monkeypatch.setattr(LeaseReaper, "_bump_retry_count", staticmethod(fake_bump))
        with patch("aegra_api.services.lease_reaper.settings") as mock_settings:
            mock_settings.worker.BG_JOB_MAX_RETRIES = 3
            retryable, exhausted = await LeaseReaper._check_retry_limits(
                ["run-retryable", "run-exhausted", "run-missing"]
            )

        assert retryable == ["run-retryable"]
        assert exhausted == ["run-exhausted"]

    @pytest.mark.asyncio
    async def test_increments_retry_metric_per_retryable_run(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Each successful retry bumps run_retries{graph_id, retry_number}."""
        registry = prometheus_client.CollectorRegistry()
        setup_worker_metrics(registry=registry)
        metrics = get_worker_metrics()
        assert metrics is not None

        async def fake_bump(run_id: str, max_retries: int) -> tuple[int, dict[str, Any]] | None:
            return 2, {"graph_id": "graph-x"}

        monkeypatch.setattr(LeaseReaper, "_bump_retry_count", staticmethod(fake_bump))
        with patch("aegra_api.services.lease_reaper.settings") as mock_settings:
            mock_settings.worker.BG_JOB_MAX_RETRIES = 3
            retryable, exhausted = await LeaseReaper._check_retry_limits(["run-1"])

        assert retryable == ["run-1"]
        assert exhausted == []
        assert metrics.run_retries.labels(graph_id="graph-x", retry_number="2")._value.get() == 1.0


class TestMarkPermanentlyFailed:
    @pytest.mark.asyncio
    async def test_updates_run_rows_to_error_with_message(self) -> None:
        """``_mark_permanently_failed`` sets status='error' with the explicit
        max-retries-exceeded error_message and clears the lease columns."""
        session = AsyncMock()
        session.execute = AsyncMock()
        maker = _make_session_maker(session)

        with patch("aegra_api.services.lease_reaper._get_session_maker", return_value=maker):
            await LeaseReaper._mark_permanently_failed(["run-a", "run-b"])

        session.execute.assert_awaited_once()
        session.commit.assert_awaited_once()

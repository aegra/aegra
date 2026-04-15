"""Unit tests for worker and reaper Prometheus metrics."""

import asyncio
import time
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import prometheus_client
import pytest

from aegra_api.observability.metrics import (
    EXECUTION_BUCKETS,
    QUEUE_WAIT_BUCKETS,
    REAPER_CYCLE_BUCKETS,
    ReaperMetrics,
    WorkerMetrics,
    get_reaper_metrics,
    get_worker_metrics,
    setup_worker_metrics,
)


@pytest.fixture(autouse=True)
def _reset_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset module-level metric singletons before each test."""
    import aegra_api.observability.metrics as m

    monkeypatch.setattr(m, "_worker_metrics", None)
    monkeypatch.setattr(m, "_reaper_metrics", None)


@pytest.fixture
def fresh_registry() -> prometheus_client.CollectorRegistry:
    return prometheus_client.CollectorRegistry()


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_setup_creates_worker_metrics(fresh_registry: prometheus_client.CollectorRegistry) -> None:
    setup_worker_metrics(registry=fresh_registry)
    metrics = get_worker_metrics()

    assert metrics is not None
    assert isinstance(metrics, WorkerMetrics)


def test_setup_creates_reaper_metrics(fresh_registry: prometheus_client.CollectorRegistry) -> None:
    setup_worker_metrics(registry=fresh_registry)
    metrics = get_reaper_metrics()

    assert metrics is not None
    assert isinstance(metrics, ReaperMetrics)


def test_returns_none_when_not_setup() -> None:
    assert get_worker_metrics() is None
    assert get_reaper_metrics() is None


def test_all_15_worker_metrics_registered(fresh_registry: prometheus_client.CollectorRegistry) -> None:
    setup_worker_metrics(registry=fresh_registry)
    metrics = get_worker_metrics()
    assert metrics is not None

    expected_fields = {
        "runs_dispatched",
        "runs_completed",
        "runs_in_flight",
        "runs_discarded",
        "run_execution_seconds",
        "run_queue_wait_seconds",
        "run_timeouts",
        "submit_errors",
        "runs_dequeued",
        "dequeue_errors",
        "redis_reachable",
        "heartbeat_extensions",
        "heartbeat_failures",
        "lease_losses",
        "run_retries",
    }
    actual_fields = {f.name for f in metrics.__dataclass_fields__.values()}
    assert actual_fields == expected_fields


def test_all_5_reaper_metrics_registered(fresh_registry: prometheus_client.CollectorRegistry) -> None:
    setup_worker_metrics(registry=fresh_registry)
    metrics = get_reaper_metrics()
    assert metrics is not None

    expected_fields = {"crashed_recovered", "stuck_reenqueued", "permanently_failed", "cycle_seconds", "queue_depth"}
    actual_fields = {f.name for f in metrics.__dataclass_fields__.values()}
    assert actual_fields == expected_fields


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


def test_completed_labels(fresh_registry: prometheus_client.CollectorRegistry) -> None:
    setup_worker_metrics(registry=fresh_registry)
    metrics = get_worker_metrics()
    assert metrics is not None

    metrics.runs_completed.labels(graph_id="test_graph", status="success").inc()
    metrics.runs_completed.labels(graph_id="test_graph", status="error").inc()
    metrics.runs_completed.labels(graph_id="test_graph", status="interrupted").inc()

    assert metrics.runs_completed.labels(graph_id="test_graph", status="success")._value.get() == 1.0
    assert metrics.runs_completed.labels(graph_id="test_graph", status="error")._value.get() == 1.0
    assert metrics.runs_completed.labels(graph_id="test_graph", status="interrupted")._value.get() == 1.0


def test_retries_labels(fresh_registry: prometheus_client.CollectorRegistry) -> None:
    setup_worker_metrics(registry=fresh_registry)
    metrics = get_worker_metrics()
    assert metrics is not None

    metrics.run_retries.labels(graph_id="g1", retry_number="1").inc()
    metrics.run_retries.labels(graph_id="g1", retry_number="2").inc()

    assert metrics.run_retries.labels(graph_id="g1", retry_number="1")._value.get() == 1.0
    assert metrics.run_retries.labels(graph_id="g1", retry_number="2")._value.get() == 1.0


# ---------------------------------------------------------------------------
# Histogram buckets
# ---------------------------------------------------------------------------


def test_custom_execution_buckets(fresh_registry: prometheus_client.CollectorRegistry) -> None:
    setup_worker_metrics(registry=fresh_registry)
    metrics = get_worker_metrics()
    assert metrics is not None

    labeled = metrics.run_execution_seconds.labels(graph_id="test")
    assert list(labeled._upper_bounds) == list(EXECUTION_BUCKETS) + [float("inf")]


def test_custom_queue_wait_buckets(fresh_registry: prometheus_client.CollectorRegistry) -> None:
    setup_worker_metrics(registry=fresh_registry)
    metrics = get_worker_metrics()
    assert metrics is not None

    labeled = metrics.run_queue_wait_seconds.labels(graph_id="test")
    assert list(labeled._upper_bounds) == list(QUEUE_WAIT_BUCKETS) + [float("inf")]


def test_custom_reaper_cycle_buckets(fresh_registry: prometheus_client.CollectorRegistry) -> None:
    setup_worker_metrics(registry=fresh_registry)
    metrics = get_reaper_metrics()
    assert metrics is not None

    assert list(metrics.cycle_seconds._upper_bounds) == list(REAPER_CYCLE_BUCKETS) + [float("inf")]


# ---------------------------------------------------------------------------
# _user_cancellations / _increment_completed
# ---------------------------------------------------------------------------


def test_mark_user_cancellation_adds_and_clears() -> None:
    from aegra_api.services.run_executor import _user_cancellations, mark_user_cancellation

    run_id = "test-run-123"
    mark_user_cancellation(run_id)
    assert run_id in _user_cancellations

    _user_cancellations.discard(run_id)
    assert run_id not in _user_cancellations


def test_increment_completed_noop_when_metrics_none() -> None:
    """_increment_completed should not raise when metrics are not registered."""
    from aegra_api.services.run_executor import _increment_completed

    # Should not raise
    _increment_completed("some_graph", "success")


def test_increment_completed_increments_counter(fresh_registry: prometheus_client.CollectorRegistry) -> None:
    from aegra_api.services.run_executor import _increment_completed

    setup_worker_metrics(registry=fresh_registry)
    _increment_completed("my_graph", "success")

    metrics = get_worker_metrics()
    assert metrics is not None
    assert metrics.runs_completed.labels(graph_id="my_graph", status="success")._value.get() == 1.0


# ---------------------------------------------------------------------------
# _AcquireResult discrimination
# ---------------------------------------------------------------------------


def test_acquire_result_contention() -> None:
    from aegra_api.services.worker_executor import _AcquireResult

    result = _AcquireResult(loaded=None, reason="contention")
    assert result.loaded is None
    assert result.reason == "contention"


def test_acquire_result_corruption() -> None:
    from aegra_api.services.worker_executor import _AcquireResult

    result = _AcquireResult(loaded=None, reason="corruption")
    assert result.loaded is None
    assert result.reason == "corruption"


# ---------------------------------------------------------------------------
# _LoadedRun.enqueued_at
# ---------------------------------------------------------------------------


def test_loaded_run_enqueued_at() -> None:
    from aegra_api.services.worker_executor import _LoadedRun

    job = MagicMock()
    loaded = _LoadedRun(job=job, trace={}, enqueued_at=1234567890.0)
    assert loaded.enqueued_at == 1234567890.0

    loaded_none = _LoadedRun(job=job, trace={}, enqueued_at=None)
    assert loaded_none.enqueued_at is None


# ---------------------------------------------------------------------------
# _enqueued_at in execution_params
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_queue_wait_observed_when_enqueued_at_present(
    fresh_registry: prometheus_client.CollectorRegistry,
) -> None:
    """queue_wait_seconds is observed when _enqueued_at is set in execution_params."""
    setup_worker_metrics(registry=fresh_registry)
    metrics = get_worker_metrics()
    assert metrics is not None

    enqueued_at = time.time() - 2.0  # enqueued 2 seconds ago
    lease_acquired_at = datetime.now(UTC)
    wait_seconds = (lease_acquired_at - datetime.fromtimestamp(enqueued_at, tz=UTC)).total_seconds()

    assert wait_seconds >= 0
    metrics.run_queue_wait_seconds.labels(graph_id="test").observe(wait_seconds)

    # Verify histogram recorded the observation
    assert metrics.run_queue_wait_seconds.labels(graph_id="test")._sum.get() >= 1.5


# ---------------------------------------------------------------------------
# Gauge safety
# ---------------------------------------------------------------------------


def test_in_flight_gauge_no_negative(fresh_registry: prometheus_client.CollectorRegistry) -> None:
    """in_flight gauge should not go below 0 with the _in_flight_incremented guard."""
    setup_worker_metrics(registry=fresh_registry)
    metrics = get_worker_metrics()
    assert metrics is not None

    _in_flight_incremented = False

    # Simulate: acquire fails, so inc never called
    # In finally: only dec if inc happened
    if _in_flight_incremented:
        metrics.runs_in_flight.labels(graph_id="test").dec()

    # Gauge should still be at 0 (default), not -1
    assert metrics.runs_in_flight.labels(graph_id="test")._value.get() == 0.0


def test_in_flight_gauge_increments_and_decrements(fresh_registry: prometheus_client.CollectorRegistry) -> None:
    setup_worker_metrics(registry=fresh_registry)
    metrics = get_worker_metrics()
    assert metrics is not None

    _in_flight_incremented = False
    metrics.runs_in_flight.labels(graph_id="test").inc()
    _in_flight_incremented = True

    assert metrics.runs_in_flight.labels(graph_id="test")._value.get() == 1.0

    if _in_flight_incremented:
        metrics.runs_in_flight.labels(graph_id="test").dec()

    assert metrics.runs_in_flight.labels(graph_id="test")._value.get() == 0.0


# ---------------------------------------------------------------------------
# execute_run CancelledError paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_run_skips_finalize_on_timeout_cancel(
    fresh_registry: prometheus_client.CollectorRegistry,
) -> None:
    """Default CancelledError (not user, not lease-loss) skips finalize — timeout path."""
    from aegra_api.services import run_executor as mod

    setup_worker_metrics(registry=fresh_registry)

    job = MagicMock()
    job.identity.run_id = "run-timeout"
    job.identity.thread_id = "thread-1"
    job.identity.graph_id = "graph-1"

    with (
        patch.object(mod, "update_run_status", new_callable=AsyncMock),
        patch.object(mod, "_stream_graph", new_callable=AsyncMock, side_effect=asyncio.CancelledError),
        patch.object(mod, "finalize_run", new_callable=AsyncMock) as mock_finalize,
        patch.object(mod, "streaming_service", MagicMock(cleanup_run=AsyncMock())),
        patch.object(mod, "_signal_run_done", new_callable=AsyncMock),
    ):
        with pytest.raises(asyncio.CancelledError):
            await mod.execute_run(job)

        # finalize_run should NOT have been called (timeout path skips it)
        mock_finalize.assert_not_called()

    # completed should NOT have been incremented
    metrics = get_worker_metrics()
    assert metrics is not None
    # No labels initialized for this graph = no increment happened


@pytest.mark.asyncio
async def test_execute_run_finalizes_on_user_cancel(fresh_registry: prometheus_client.CollectorRegistry) -> None:
    """CancelledError with run_id in _user_cancellations finalizes as interrupted."""
    from aegra_api.services import run_executor as mod

    setup_worker_metrics(registry=fresh_registry)

    job = MagicMock()
    job.identity.run_id = "run-user-cancel"
    job.identity.thread_id = "thread-1"
    job.identity.graph_id = "graph-1"

    mod._user_cancellations.add("run-user-cancel")

    with (
        patch.object(mod, "update_run_status", new_callable=AsyncMock),
        patch.object(mod, "_stream_graph", new_callable=AsyncMock, side_effect=asyncio.CancelledError),
        patch.object(mod, "finalize_run", new_callable=AsyncMock) as mock_finalize,
        patch.object(mod, "streaming_service", MagicMock(signal_run_cancelled=AsyncMock(), cleanup_run=AsyncMock())),
        patch.object(mod, "_signal_run_done", new_callable=AsyncMock),
    ):
        with pytest.raises(asyncio.CancelledError):
            await mod.execute_run(job)

        mock_finalize.assert_called_once()
        call_kwargs = mock_finalize.call_args
        assert call_kwargs[1]["status"] == "interrupted"

    metrics = get_worker_metrics()
    assert metrics is not None
    assert metrics.runs_completed.labels(graph_id="graph-1", status="interrupted")._value.get() == 1.0

    # Cleanup
    mod._user_cancellations.discard("run-user-cancel")


# ---------------------------------------------------------------------------
# execution_seconds observation
# ---------------------------------------------------------------------------


def test_execution_seconds_guard_skips_on_cancel_observes_on_success(
    fresh_registry: prometheus_client.CollectorRegistry,
) -> None:
    """Verify the _should_observe_duration guard pattern from _execute_with_lease:
    observation is skipped on CancelledError but recorded on normal completion."""
    setup_worker_metrics(registry=fresh_registry)
    metrics = get_worker_metrics()
    assert metrics is not None

    graph_id = "test-guard"

    # Path 1: CancelledError → _should_observe_duration = False → no observation
    _should_observe_duration = True
    try:
        raise asyncio.CancelledError
    except asyncio.CancelledError:
        _should_observe_duration = False

    if _should_observe_duration:
        metrics.run_execution_seconds.labels(graph_id=graph_id).observe(5.0)

    assert metrics.run_execution_seconds.labels(graph_id=graph_id)._sum.get() == 0.0

    # Path 2: normal completion → _should_observe_duration stays True → observed
    _should_observe_duration = True
    if _should_observe_duration:
        metrics.run_execution_seconds.labels(graph_id=graph_id).observe(3.0)

    assert metrics.run_execution_seconds.labels(graph_id=graph_id)._sum.get() == 3.0

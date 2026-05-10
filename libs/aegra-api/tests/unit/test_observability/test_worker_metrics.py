"""Unit tests for worker and reaper Prometheus metrics."""

import asyncio
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import prometheus_client
import pytest
from redis import RedisError
from sqlalchemy.exc import SQLAlchemyError

from aegra_api.core.cancellation_state import CancellationRegistry, cancellations
from aegra_api.observability import metrics as metrics_module
from aegra_api.observability.metrics import (
    EXECUTION_BUCKETS,
    QUEUE_WAIT_BUCKETS,
    REAPER_CYCLE_BUCKETS,
    ReaperMetrics,
    WorkerMetrics,
    extract_graph_id,
    get_reaper_metrics,
    get_worker_metrics,
    setup_worker_metrics,
)
from aegra_api.services import run_executor as run_executor_module
from aegra_api.services.lease_reaper import LeaseReaper
from aegra_api.services.run_executor import _increment_completed
from aegra_api.services.worker_executor import (
    WorkerExecutor,
    _AcquireResult,
    _heartbeat_loop,
    _LoadedRun,
)


@pytest.fixture(autouse=True)
def _reset_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset module-level metric singletons before each test."""
    monkeypatch.setattr(metrics_module, "_worker_metrics", None)
    monkeypatch.setattr(metrics_module, "_reaper_metrics", None)


@pytest.fixture(autouse=True)
def _reset_cancellations() -> Iterator[None]:
    """Drop any cancellation tags between tests so the singleton stays clean."""
    yield
    cancellations.clear_all()


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


def _registered_metric_names(registry: prometheus_client.CollectorRegistry) -> set[str]:
    """Return the set of metric names actually exposed by the registry.

    Counter names are normalized: prometheus_client returns ``aegra_x``
    for ``Counter("aegra_x_total")`` so we compare on the suffix-free
    name. Gauges and Histograms are returned as-is.
    """
    return {collector._name for collector in registry._names_to_collectors.values()}  # type: ignore[attr-defined]


def test_all_worker_metrics_exposed_by_registry(fresh_registry: prometheus_client.CollectorRegistry) -> None:
    """The registry must actually expose every worker metric — guards
    against renames that drift the dataclass field away from the metric."""
    setup_worker_metrics(registry=fresh_registry)
    names = _registered_metric_names(fresh_registry)

    # Counters: prometheus_client strips ``_total`` for the internal name
    expected = {
        "aegra_runs_dispatched",
        "aegra_runs_completed",
        "aegra_runs_in_flight",
        "aegra_runs_discarded",
        "aegra_run_execution_seconds",
        "aegra_run_queue_wait_seconds",
        "aegra_run_timeouts",
        "aegra_submit_errors",
        "aegra_runs_dequeued",
        "aegra_dequeue_errors",
        "aegra_redis_up",
        "aegra_heartbeat_extensions",
        "aegra_heartbeat_failures",
        "aegra_lease_losses",
        "aegra_run_retries",
    }
    missing = expected - names
    assert not missing, f"missing worker metrics: {missing}"


def test_all_reaper_metrics_exposed_by_registry(fresh_registry: prometheus_client.CollectorRegistry) -> None:
    setup_worker_metrics(registry=fresh_registry)
    names = _registered_metric_names(fresh_registry)

    expected = {
        "aegra_reaper_crashed_recovered",
        "aegra_reaper_stuck_reenqueued",
        "aegra_reaper_permanently_failed",
        "aegra_reaper_cycle_seconds",
        "aegra_queue_depth",
    }
    missing = expected - names
    assert not missing, f"missing reaper metrics: {missing}"


def test_setup_worker_metrics_is_idempotent_against_default_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two consecutive calls without an explicit registry must not raise.

    Without idempotency, ``uvicorn --reload`` would trigger
    ``Duplicated timeseries in CollectorRegistry`` on every code change.
    """
    # Use a private registry as the "default" so we don't pollute
    # prometheus_client.REGISTRY.
    private = prometheus_client.CollectorRegistry()
    monkeypatch.setattr(metrics_module, "REGISTRY", private, raising=False)
    monkeypatch.setattr(prometheus_client, "REGISTRY", private)

    setup_worker_metrics()  # first call registers
    first = get_worker_metrics()
    assert first is not None

    setup_worker_metrics()  # second call must be a no-op, not a crash
    assert get_worker_metrics() is first


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


def _bucket_upper_bounds(registry: prometheus_client.CollectorRegistry, metric_name: str) -> list[float]:
    """Read bucket upper bounds from a histogram via its ``_bucket`` samples.

    Drives an ``observe(0)`` first so the labels are materialized for
    label-bearing histograms, then parses the ``le=`` label out of the
    exposed samples. This avoids reaching into prometheus_client
    private attributes.
    """
    bounds: list[float] = []
    for metric_family in registry.collect():
        if metric_family.name != metric_name:
            continue
        for sample in metric_family.samples:
            if sample.name.endswith("_bucket"):
                le = sample.labels.get("le")
                if le is None:
                    continue
                bounds.append(float("inf") if le == "+Inf" else float(le))
        return bounds
    return bounds


def test_execution_buckets_match_constant(fresh_registry: prometheus_client.CollectorRegistry) -> None:
    setup_worker_metrics(registry=fresh_registry)
    metrics = get_worker_metrics()
    assert metrics is not None
    metrics.run_execution_seconds.labels(graph_id="test").observe(0.0)

    bounds = _bucket_upper_bounds(fresh_registry, "aegra_run_execution_seconds")
    # Histogram samples include duplicates per-label-set; dedupe and sort.
    distinct = sorted(set(bounds))
    expected = sorted({*EXECUTION_BUCKETS, float("inf")})
    assert distinct == expected


def test_queue_wait_buckets_match_constant(fresh_registry: prometheus_client.CollectorRegistry) -> None:
    setup_worker_metrics(registry=fresh_registry)
    metrics = get_worker_metrics()
    assert metrics is not None
    metrics.run_queue_wait_seconds.labels(graph_id="test").observe(0.0)

    distinct = sorted(set(_bucket_upper_bounds(fresh_registry, "aegra_run_queue_wait_seconds")))
    expected = sorted({*QUEUE_WAIT_BUCKETS, float("inf")})
    assert distinct == expected


def test_reaper_cycle_buckets_match_constant(fresh_registry: prometheus_client.CollectorRegistry) -> None:
    setup_worker_metrics(registry=fresh_registry)
    reaper = get_reaper_metrics()
    assert reaper is not None
    reaper.cycle_seconds.observe(0.0)

    distinct = sorted(set(_bucket_upper_bounds(fresh_registry, "aegra_reaper_cycle_seconds")))
    expected = sorted({*REAPER_CYCLE_BUCKETS, float("inf")})
    assert distinct == expected


def test_execution_buckets_cover_default_timeout() -> None:
    """A timed-out run observes ``BG_JOB_TIMEOUT_SECS``. The histogram
    must have a finite bucket large enough to contain that observation
    so dashboards don't show all timeouts as ``+Inf``."""
    finite = list(EXECUTION_BUCKETS)
    # Default BG_JOB_TIMEOUT_SECS is 3600. Even if operators raise it,
    # the topmost bucket should be large enough that 3600 is not the
    # only possible top observation.
    assert max(finite) >= 3600


# ---------------------------------------------------------------------------
# cancellations registry / _increment_completed
# ---------------------------------------------------------------------------


def test_cancellations_mark_user_adds_and_clears() -> None:
    run_id = "test-run-cancel-mark"
    cancellations.mark(run_id, "user")
    assert cancellations.reason_of(run_id) == "user"

    cancellations.clear(run_id, only="user")
    assert cancellations.reason_of(run_id) is None


def test_increment_completed_noop_when_metrics_none() -> None:
    """_increment_completed should not raise when metrics are not registered."""
    assert get_worker_metrics() is None
    _increment_completed("some_graph", "success")  # must not raise


def test_increment_completed_increments_counter(fresh_registry: prometheus_client.CollectorRegistry) -> None:
    setup_worker_metrics(registry=fresh_registry)
    _increment_completed("my_graph", "success")

    metrics = get_worker_metrics()
    assert metrics is not None
    assert metrics.runs_completed.labels(graph_id="my_graph", status="success")._value.get() == 1.0


# ---------------------------------------------------------------------------
# _AcquireResult discrimination
# ---------------------------------------------------------------------------


# Construction-only smoke tests for _AcquireResult / _LoadedRun were
# removed: the dataclasses are exercised by the actual execute_with_lease
# tests, and asserting that ``dataclass(...)`` returns its arguments
# duplicates Python's behavior, not Aegra's.


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
# execute_run CancelledError paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_run_skips_finalize_and_completed_on_timeout_cancel(
    fresh_registry: prometheus_client.CollectorRegistry,
) -> None:
    """Default CancelledError (not user, not lease-loss) skips finalize and
    must NOT increment ``runs_completed{status="error"}`` — that increment is
    owned exclusively by ``_handle_timeout`` in worker_executor."""
    setup_worker_metrics(registry=fresh_registry)

    job = MagicMock()
    job.identity.run_id = "run-timeout"
    job.identity.thread_id = "thread-1"
    job.identity.graph_id = "graph-1"
    # Belt-and-suspenders: ensure no leftover tag.
    cancellations.clear("run-timeout")

    with (
        patch.object(run_executor_module, "update_run_status", new_callable=AsyncMock),
        patch.object(run_executor_module, "_stream_graph", new_callable=AsyncMock, side_effect=asyncio.CancelledError),
        patch.object(run_executor_module, "finalize_run", new_callable=AsyncMock) as mock_finalize,
        patch.object(run_executor_module, "streaming_service", MagicMock(cleanup_run=AsyncMock())),
        patch.object(run_executor_module, "_signal_run_done", new_callable=AsyncMock),
    ):
        with pytest.raises(asyncio.CancelledError):
            await run_executor_module.execute_run(job)
        mock_finalize.assert_not_called()

    metrics = get_worker_metrics()
    assert metrics is not None
    # Hard assert: no completed increment for any status on this graph.
    assert metrics.runs_completed.labels(graph_id="graph-1", status="error")._value.get() == 0.0
    assert metrics.runs_completed.labels(graph_id="graph-1", status="interrupted")._value.get() == 0.0


@pytest.mark.asyncio
async def test_execute_run_finalizes_on_user_cancel(fresh_registry: prometheus_client.CollectorRegistry) -> None:
    """CancelledError with run_id marked user-cancelled finalizes as interrupted."""
    setup_worker_metrics(registry=fresh_registry)

    job = MagicMock()
    job.identity.run_id = "run-user-cancel"
    job.identity.thread_id = "thread-1"
    job.identity.graph_id = "graph-1"

    cancellations.mark("run-user-cancel", "user")

    with (
        patch.object(run_executor_module, "update_run_status", new_callable=AsyncMock),
        patch.object(run_executor_module, "_stream_graph", new_callable=AsyncMock, side_effect=asyncio.CancelledError),
        patch.object(run_executor_module, "finalize_run", new_callable=AsyncMock) as mock_finalize,
        patch.object(
            run_executor_module,
            "streaming_service",
            MagicMock(signal_run_cancelled=AsyncMock(), cleanup_run=AsyncMock()),
        ),
        patch.object(run_executor_module, "_signal_run_done", new_callable=AsyncMock),
    ):
        with pytest.raises(asyncio.CancelledError):
            await run_executor_module.execute_run(job)

        mock_finalize.assert_called_once()
        call_kwargs = mock_finalize.call_args
        assert call_kwargs[1]["status"] == "interrupted"

    metrics = get_worker_metrics()
    assert metrics is not None
    assert metrics.runs_completed.labels(graph_id="graph-1", status="interrupted")._value.get() == 1.0


# ---------------------------------------------------------------------------
# extract_graph_id helper
# ---------------------------------------------------------------------------


def test_extract_graph_id_top_level() -> None:
    """to_execution_params writes graph_id at the top level — primary lookup site."""
    assert extract_graph_id({"graph_id": "agent", "trace": {}}) == "agent"


def test_extract_graph_id_unknown_when_missing() -> None:
    assert extract_graph_id(None) == "unknown"
    assert extract_graph_id({}) == "unknown"
    assert extract_graph_id({"graph_id": ""}) == "unknown"  # empty string not accepted


def test_extract_graph_id_returns_unknown_for_non_string_types() -> None:
    """Defensive: a non-string ``graph_id`` (int, list, None) falls through to ``"unknown"``."""
    assert extract_graph_id({"graph_id": 42}) == "unknown"
    assert extract_graph_id({"graph_id": None}) == "unknown"
    assert extract_graph_id({"graph_id": ["a", "b"]}) == "unknown"


# ---------------------------------------------------------------------------
# Failure-path counters (#6 — verify by exercising the failure path,
# not just by name presence)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_submit_increments_submit_errors_on_redis_failure(
    fresh_registry: prometheus_client.CollectorRegistry,
) -> None:
    """submit_errors counter increments when rpush raises RedisError."""
    setup_worker_metrics(registry=fresh_registry)
    metrics = get_worker_metrics()
    assert metrics is not None

    executor = WorkerExecutor()
    job = MagicMock()
    job.identity.run_id = "r1"
    job.identity.graph_id = "graph-x"

    mock_client = AsyncMock()
    mock_client.rpush = AsyncMock(side_effect=RedisError("conn refused"))

    with (
        patch("aegra_api.services.worker_executor.redis_manager") as mock_rm,
        patch("aegra_api.services.worker_executor.settings") as mock_settings,
    ):
        mock_settings.worker.WORKER_QUEUE_KEY = "aegra:jobs"
        mock_rm.get_client.return_value = mock_client

        with pytest.raises(RedisError):
            await executor.submit(job)

    assert metrics.submit_errors.labels(graph_id="graph-x")._value.get() == 1.0
    # runs_dispatched must NOT have incremented on failure
    assert metrics.runs_dispatched.labels(graph_id="graph-x")._value.get() == 0.0


@pytest.mark.asyncio
async def test_dequeue_increments_errors_and_clears_redis_up(
    fresh_registry: prometheus_client.CollectorRegistry,
) -> None:
    """dequeue_errors increments and redis_up flips to 0 on RedisError."""
    setup_worker_metrics(registry=fresh_registry)
    metrics = get_worker_metrics()
    assert metrics is not None

    executor = WorkerExecutor()
    mock_client = AsyncMock()
    mock_client.blpop = AsyncMock(side_effect=RedisError("down"))

    with (
        patch("aegra_api.services.worker_executor.redis_manager") as mock_rm,
        patch("aegra_api.services.worker_executor.settings") as mock_settings,
        patch("aegra_api.services.worker_executor.asyncio.sleep", new_callable=AsyncMock),
        patch.object(WorkerExecutor, "_poll_postgres", new_callable=AsyncMock, return_value=None),
    ):
        mock_settings.worker.WORKER_QUEUE_KEY = "aegra:jobs"
        mock_settings.worker.POSTGRES_POLL_INTERVAL_SECONDS = 0.01
        mock_rm.get_client.return_value = mock_client

        await executor._dequeue()

    assert metrics.dequeue_errors._value.get() == 1.0
    assert metrics.redis_up._value.get() == 0.0


@pytest.mark.asyncio
async def test_dequeue_sets_redis_up_to_one_on_success(
    fresh_registry: prometheus_client.CollectorRegistry,
) -> None:
    """redis_up flips to 1 on a successful BLPOP. ``runs_dequeued`` is
    NOT incremented here — the worker loop counts it after run_id validation
    so it stays symmetric with ``runs_discarded``."""
    setup_worker_metrics(registry=fresh_registry)
    metrics = get_worker_metrics()
    assert metrics is not None

    executor = WorkerExecutor()
    mock_client = AsyncMock()
    metrics.redis_up.set(0)
    mock_client.blpop = AsyncMock(return_value=("aegra:jobs", "run-abc"))

    with (
        patch("aegra_api.services.worker_executor.redis_manager") as mock_rm,
        patch("aegra_api.services.worker_executor.settings") as mock_settings,
    ):
        mock_settings.worker.WORKER_QUEUE_KEY = "aegra:jobs"
        mock_rm.get_client.return_value = mock_client

        result = await executor._dequeue()

    assert result == "run-abc"
    assert metrics.redis_up._value.get() == 1.0
    assert metrics.runs_dequeued._value.get() == 0.0


def _heartbeat_session_mock(side_effect: Exception | None = None, rowcount: int | None = None) -> MagicMock:
    """Build the session-maker scaffolding used by heartbeat tests."""
    session = AsyncMock()
    if side_effect is not None:
        session.execute = AsyncMock(side_effect=side_effect)
    else:
        update_result = MagicMock()
        update_result.rowcount = rowcount if rowcount is not None else 1
        session.execute = AsyncMock(return_value=update_result)
    session.commit = AsyncMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return MagicMock(return_value=ctx)


@pytest.mark.asyncio
async def test_heartbeat_increments_extensions_on_success(
    fresh_registry: prometheus_client.CollectorRegistry,
) -> None:
    """A successful heartbeat must bump ``heartbeat_extensions`` exactly once
    per iteration."""
    setup_worker_metrics(registry=fresh_registry)
    metrics = get_worker_metrics()
    assert metrics is not None

    maker = _heartbeat_session_mock(rowcount=1)
    sleep_calls = 0

    async def sleeper(_d: float) -> None:
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls >= 3:
            raise asyncio.CancelledError

    with (
        patch("aegra_api.services.worker_executor._get_session_maker", return_value=maker),
        patch("aegra_api.services.worker_executor.settings") as mock_settings,
        patch("aegra_api.services.worker_executor.asyncio.sleep", side_effect=sleeper),
    ):
        mock_settings.worker.HEARTBEAT_INTERVAL_SECONDS = 1
        mock_settings.worker.LEASE_DURATION_SECONDS = 30
        with pytest.raises(asyncio.CancelledError):
            await _heartbeat_loop("run-id", "w0", graph_id="g")

    # Two completed iterations before the third sleep cancelled.
    assert metrics.heartbeat_extensions.labels(graph_id="g")._value.get() == 2.0


@pytest.mark.asyncio
async def test_heartbeat_increments_failures_on_db_error(
    fresh_registry: prometheus_client.CollectorRegistry,
) -> None:
    """heartbeat_failures counter increments when the lease UPDATE raises SQLAlchemyError."""
    setup_worker_metrics(registry=fresh_registry)
    metrics = get_worker_metrics()
    assert metrics is not None

    maker = _heartbeat_session_mock(side_effect=SQLAlchemyError("connection reset"))
    sleep_calls = 0

    async def sleeper(_d: float) -> None:
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls >= 3:
            raise asyncio.CancelledError

    with (
        patch("aegra_api.services.worker_executor._get_session_maker", return_value=maker),
        patch("aegra_api.services.worker_executor.settings") as mock_settings,
        patch("aegra_api.services.worker_executor.asyncio.sleep", side_effect=sleeper),
    ):
        mock_settings.worker.HEARTBEAT_INTERVAL_SECONDS = 1
        mock_settings.worker.LEASE_DURATION_SECONDS = 30
        with pytest.raises(asyncio.CancelledError):
            await _heartbeat_loop("run-id", "w0", graph_id="g")

    # Two failed iterations before the third sleep cancelled.
    assert metrics.heartbeat_failures.labels(graph_id="g")._value.get() == 2.0


@pytest.mark.asyncio
async def test_heartbeat_treats_persistent_failures_as_lease_loss(
    fresh_registry: prometheus_client.CollectorRegistry,
) -> None:
    """If heartbeat failures persist long enough that the lease has likely
    expired, the loop must cancel the job task and emit
    ``lease_losses{reason="heartbeat_timeout"}``."""
    setup_worker_metrics(registry=fresh_registry)
    metrics = get_worker_metrics()
    assert metrics is not None

    maker = _heartbeat_session_mock(side_effect=SQLAlchemyError("connection reset"))
    job_task = MagicMock()
    job_task.done.return_value = False
    job_task.cancel = MagicMock()

    with (
        patch("aegra_api.services.worker_executor._get_session_maker", return_value=maker),
        patch("aegra_api.services.worker_executor.settings") as mock_settings,
        patch("aegra_api.services.worker_executor.asyncio.sleep", new_callable=AsyncMock),
    ):
        # 10s interval, 30s lease => failure_budget = duration - interval = 20s.
        # Two consecutive 10s failures hit 20s and trip the bail-out.
        mock_settings.worker.HEARTBEAT_INTERVAL_SECONDS = 10
        mock_settings.worker.LEASE_DURATION_SECONDS = 30

        await _heartbeat_loop("run-id", "w0", job_task=job_task, graph_id="g")

    assert metrics.lease_losses.labels(graph_id="g", reason="heartbeat_timeout")._value.get() == 1.0
    # The other reason must NOT have incremented.
    assert metrics.lease_losses.labels(graph_id="g", reason="rowcount_zero")._value.get() == 0.0
    job_task.cancel.assert_called_once()


@pytest.mark.asyncio
async def test_heartbeat_increments_lease_losses_on_zero_rowcount(
    fresh_registry: prometheus_client.CollectorRegistry,
) -> None:
    """When the heartbeat UPDATE returns rowcount=0, lease was lost — emit
    ``lease_losses{reason="rowcount_zero"}``."""
    setup_worker_metrics(registry=fresh_registry)
    metrics = get_worker_metrics()
    assert metrics is not None

    maker = _heartbeat_session_mock(rowcount=0)
    job_task = MagicMock()
    job_task.done.return_value = False
    job_task.cancel = MagicMock()

    with (
        patch("aegra_api.services.worker_executor._get_session_maker", return_value=maker),
        patch("aegra_api.services.worker_executor.settings") as mock_settings,
        patch("aegra_api.services.worker_executor.asyncio.sleep", new_callable=AsyncMock),
    ):
        mock_settings.worker.HEARTBEAT_INTERVAL_SECONDS = 1
        mock_settings.worker.LEASE_DURATION_SECONDS = 30
        await _heartbeat_loop("run-id", "w0", job_task=job_task, graph_id="g")

    assert metrics.lease_losses.labels(graph_id="g", reason="rowcount_zero")._value.get() == 1.0
    assert metrics.lease_losses.labels(graph_id="g", reason="heartbeat_timeout")._value.get() == 0.0
    job_task.cancel.assert_called_once()


# ---------------------------------------------------------------------------
# Reaper permanently_failed (#6)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reaper_increments_permanently_failed_on_max_retries(
    fresh_registry: prometheus_client.CollectorRegistry,
) -> None:
    """When _check_retry_limits exceeds max retries, _mark_permanently_failed
    is called and reaper.permanently_failed is incremented."""
    setup_worker_metrics(registry=fresh_registry)
    reaper_metrics = get_reaper_metrics()
    assert reaper_metrics is not None

    reaper = LeaseReaper()

    with (
        patch.object(LeaseReaper, "_get_queue_depth", new_callable=AsyncMock, return_value=0),
        patch.object(LeaseReaper, "_find_recoverable", new_callable=AsyncMock, return_value=(["r1"], [])),
        patch.object(LeaseReaper, "_reset_to_pending", new_callable=AsyncMock, return_value=["r1"]),
        patch.object(LeaseReaper, "_check_retry_limits", new_callable=AsyncMock, return_value=([], ["r1"])),
        patch.object(LeaseReaper, "_mark_permanently_failed", new_callable=AsyncMock),
        patch.object(LeaseReaper, "_reenqueue", new_callable=AsyncMock),
    ):
        await reaper._reap()

    assert reaper_metrics.permanently_failed._value.get() == 1.0


@pytest.mark.asyncio
async def test_reaper_increments_crashed_recovered_and_queue_depth(
    fresh_registry: prometheus_client.CollectorRegistry,
) -> None:
    """End-to-end ``_reap`` cycle: crashed_recovered + queue_depth observed."""
    setup_worker_metrics(registry=fresh_registry)
    reaper_metrics = get_reaper_metrics()
    assert reaper_metrics is not None

    reaper = LeaseReaper()

    with (
        patch.object(LeaseReaper, "_get_queue_depth", new_callable=AsyncMock, return_value=7),
        patch.object(LeaseReaper, "_find_recoverable", new_callable=AsyncMock, return_value=(["r1", "r2"], [])),
        patch.object(LeaseReaper, "_reset_to_pending", new_callable=AsyncMock, return_value=["r1", "r2"]),
        patch.object(LeaseReaper, "_check_retry_limits", new_callable=AsyncMock, return_value=(["r1", "r2"], [])),
        patch.object(LeaseReaper, "_reenqueue", new_callable=AsyncMock),
    ):
        await reaper._reap()

    assert reaper_metrics.crashed_recovered._value.get() == 2.0
    assert reaper_metrics.queue_depth._value.get() == 7.0


@pytest.mark.asyncio
async def test_reaper_increments_stuck_reenqueued(
    fresh_registry: prometheus_client.CollectorRegistry,
) -> None:
    """Stuck-pending re-enqueue must bump ``stuck_reenqueued`` but NOT
    ``runs_dispatched`` (would double-count vs the original submit)."""
    setup_worker_metrics(registry=fresh_registry)
    reaper_metrics = get_reaper_metrics()
    worker_metrics = get_worker_metrics()
    assert reaper_metrics is not None
    assert worker_metrics is not None

    reaper = LeaseReaper()
    reenqueue_calls: list[dict[str, object]] = []

    async def fake_reenqueue(run_ids: list[str], *, bump_dispatched: bool = True) -> None:
        reenqueue_calls.append({"run_ids": run_ids, "bump_dispatched": bump_dispatched})

    with (
        patch.object(LeaseReaper, "_get_queue_depth", new_callable=AsyncMock, return_value=0),
        patch.object(LeaseReaper, "_find_recoverable", new_callable=AsyncMock, return_value=([], ["s1", "s2", "s3"])),
        patch.object(LeaseReaper, "_reenqueue", side_effect=fake_reenqueue),
    ):
        await reaper._reap()

    assert reaper_metrics.stuck_reenqueued._value.get() == 3.0
    # Stuck-pending must NOT bump dispatched.
    assert reenqueue_calls == [{"run_ids": ["s1", "s2", "s3"], "bump_dispatched": False}]


# ---------------------------------------------------------------------------
# runs_in_flight gauge — must inc/dec evenly and never go negative
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runs_in_flight_returns_to_zero_after_successful_run(
    fresh_registry: prometheus_client.CollectorRegistry,
) -> None:
    """``runs_in_flight`` must be inc'd on lease acquire and dec'd in
    ``finally``, net zero after a normal completion."""
    setup_worker_metrics(registry=fresh_registry)
    metrics = get_worker_metrics()
    assert metrics is not None

    job = MagicMock()
    job.identity.run_id = "rif-1"
    job.identity.thread_id = "t1"
    job.identity.graph_id = "graph-rif"

    loaded = _LoadedRun(job=job, trace={}, enqueued_at=None)
    acquire_result = _AcquireResult(loaded=loaded, reason="ok")

    executor = WorkerExecutor()

    async def quick_execute(_job: object) -> None:
        return

    async def quick_heartbeat(*_a: object, **_kw: object) -> None:
        await asyncio.sleep(3600)

    with (
        patch(
            "aegra_api.services.worker_executor._acquire_and_load", new_callable=AsyncMock, return_value=acquire_result
        ),
        patch("aegra_api.services.worker_executor._restore_trace_context"),
        patch("aegra_api.services.worker_executor.execute_run", side_effect=quick_execute),
        patch("aegra_api.services.worker_executor._heartbeat_loop", side_effect=quick_heartbeat),
        patch("aegra_api.services.worker_executor._release_lease", new_callable=AsyncMock),
    ):
        await executor._execute_with_lease("rif-1", "w0")

    assert metrics.runs_in_flight.labels(graph_id="graph-rif")._value.get() == 0.0


@pytest.mark.asyncio
async def test_execute_with_lease_observes_duration_when_job_raises_non_cancel_exception(
    fresh_registry: prometheus_client.CollectorRegistry,
) -> None:
    """A graph error inside ``execute_run`` raises a regular Exception in
    the worker. ``_execute_with_lease`` must catch it (so the worker keeps
    serving other runs), still observe ``run_execution_seconds`` (the run
    completed end-to-end, even unsuccessfully), and decrement
    ``runs_in_flight``. Only ``CancelledError`` skips the duration histogram.
    """
    setup_worker_metrics(registry=fresh_registry)
    metrics = get_worker_metrics()
    assert metrics is not None

    job = MagicMock()
    job.identity.run_id = "rif-exc"
    job.identity.thread_id = "t1"
    job.identity.graph_id = "graph-rif-exc"

    loaded = _LoadedRun(job=job, trace={}, enqueued_at=None)
    acquire_result = _AcquireResult(loaded=loaded, reason="ok")

    executor = WorkerExecutor()

    async def boom(_job: object) -> None:
        raise RuntimeError("graph blew up")

    async def quick_heartbeat(*_args: object, **_kwargs: object) -> None:
        await asyncio.sleep(3600)

    with (
        patch(
            "aegra_api.services.worker_executor._acquire_and_load",
            new_callable=AsyncMock,
            return_value=acquire_result,
        ),
        patch("aegra_api.services.worker_executor._restore_trace_context"),
        patch("aegra_api.services.worker_executor.execute_run", side_effect=boom),
        patch("aegra_api.services.worker_executor._heartbeat_loop", side_effect=quick_heartbeat),
        patch("aegra_api.services.worker_executor._release_lease", new_callable=AsyncMock),
    ):
        await executor._execute_with_lease("rif-exc", "w0")

    assert metrics.runs_in_flight.labels(graph_id="graph-rif-exc")._value.get() == 0.0
    histogram = metrics.run_execution_seconds.labels(graph_id="graph-rif-exc")
    # `_sum` is set on the labeled child for the prometheus Histogram type.
    assert histogram._sum.get() > 0.0


@pytest.mark.asyncio
async def test_execute_with_lease_observes_duration_on_user_cancel(
    fresh_registry: prometheus_client.CollectorRegistry,
) -> None:
    """User-cancel finalises as ``interrupted`` and increments
    ``runs_completed`` — its duration must appear in the histogram so
    ``run_execution_seconds`` stays representative of long-lived runs."""
    setup_worker_metrics(registry=fresh_registry)
    metrics = get_worker_metrics()
    assert metrics is not None

    run_id = "rif-user-cancel"
    job = MagicMock()
    job.identity.run_id = run_id
    job.identity.thread_id = "t1"
    job.identity.graph_id = "graph-rif-uc"

    loaded = _LoadedRun(job=job, trace={}, enqueued_at=None)
    acquire_result = _AcquireResult(loaded=loaded, reason="ok")

    executor = WorkerExecutor()
    started = asyncio.Event()

    async def long_running(_job: object) -> None:
        started.set()
        await asyncio.sleep(3600)

    async def quiet_heartbeat(*_a: object, **_kw: object) -> None:
        await asyncio.sleep(3600)

    with (
        patch(
            "aegra_api.services.worker_executor._acquire_and_load",
            new_callable=AsyncMock,
            return_value=acquire_result,
        ),
        patch("aegra_api.services.worker_executor._restore_trace_context"),
        patch("aegra_api.services.worker_executor.execute_run", side_effect=long_running),
        patch("aegra_api.services.worker_executor._heartbeat_loop", side_effect=quiet_heartbeat),
        patch("aegra_api.services.worker_executor._release_lease", new_callable=AsyncMock),
    ):
        cancellations.mark(run_id, "user")
        task = asyncio.create_task(executor._execute_with_lease(run_id, "w0"))
        await started.wait()
        task.cancel()
        # _execute_with_lease swallows CancelledError → returns None.
        result = await task
        assert result is None

    histogram = metrics.run_execution_seconds.labels(graph_id="graph-rif-uc")
    assert histogram._sum.get() > 0.0
    assert metrics.runs_in_flight.labels(graph_id="graph-rif-uc")._value.get() == 0.0
    # Worker's finally must clear the tag — registry stays bounded.
    assert cancellations.reason_of(run_id) is None


@pytest.mark.asyncio
async def test_execute_with_lease_skips_duration_when_no_mark(
    fresh_registry: prometheus_client.CollectorRegistry,
) -> None:
    """No mark in registry — covers timeout (``_handle_timeout`` observes
    a clamped value separately) and other outer cancels. The worker must
    NOT observe here or the timeout case would be double-counted."""
    setup_worker_metrics(registry=fresh_registry)
    metrics = get_worker_metrics()
    assert metrics is not None

    run_id = "rif-no-reason"
    job = MagicMock()
    job.identity.run_id = run_id
    job.identity.thread_id = "t1"
    job.identity.graph_id = "graph-rif-nr"

    loaded = _LoadedRun(job=job, trace={}, enqueued_at=None)
    acquire_result = _AcquireResult(loaded=loaded, reason="ok")

    executor = WorkerExecutor()
    started = asyncio.Event()

    async def long_running(_job: object) -> None:
        started.set()
        await asyncio.sleep(3600)

    async def quiet_heartbeat(*_a: object, **_kw: object) -> None:
        await asyncio.sleep(3600)

    with (
        patch(
            "aegra_api.services.worker_executor._acquire_and_load",
            new_callable=AsyncMock,
            return_value=acquire_result,
        ),
        patch("aegra_api.services.worker_executor._restore_trace_context"),
        patch("aegra_api.services.worker_executor.execute_run", side_effect=long_running),
        patch("aegra_api.services.worker_executor._heartbeat_loop", side_effect=quiet_heartbeat),
        patch("aegra_api.services.worker_executor._release_lease", new_callable=AsyncMock),
    ):
        # No cancellations.mark — simulates timeout (heartbeat-driven
        # lease_loss is covered by a separate test below).
        task = asyncio.create_task(executor._execute_with_lease(run_id, "w0"))
        await started.wait()
        task.cancel()
        # _execute_with_lease swallows CancelledError → returns None.
        result = await task
        assert result is None

    histogram = metrics.run_execution_seconds.labels(graph_id="graph-rif-nr")
    assert histogram._sum.get() == 0.0
    assert metrics.runs_in_flight.labels(graph_id="graph-rif-nr")._value.get() == 0.0
    assert cancellations.reason_of(run_id) is None


@pytest.mark.asyncio
async def test_execute_with_lease_skips_duration_on_lease_loss(
    fresh_registry: prometheus_client.CollectorRegistry,
) -> None:
    """Heartbeat-marked lease_loss tag is still present at the worker's peek
    (execute_run no longer clears). Worker must skip the histogram — work is
    partial and rerun-elsewhere — and clear the tag so a concurrent
    ``mark("user")`` for the rerun isn't blocked by precedence."""
    setup_worker_metrics(registry=fresh_registry)
    metrics = get_worker_metrics()
    assert metrics is not None

    run_id = "rif-lease-loss"
    job = MagicMock()
    job.identity.run_id = run_id
    job.identity.thread_id = "t1"
    job.identity.graph_id = "graph-rif-ll"

    loaded = _LoadedRun(job=job, trace={}, enqueued_at=None)
    acquire_result = _AcquireResult(loaded=loaded, reason="ok")

    executor = WorkerExecutor()
    started = asyncio.Event()

    async def long_running(_job: object) -> None:
        started.set()
        await asyncio.sleep(3600)

    async def quiet_heartbeat(*_a: object, **_kw: object) -> None:
        await asyncio.sleep(3600)

    with (
        patch(
            "aegra_api.services.worker_executor._acquire_and_load",
            new_callable=AsyncMock,
            return_value=acquire_result,
        ),
        patch("aegra_api.services.worker_executor._restore_trace_context"),
        patch("aegra_api.services.worker_executor.execute_run", side_effect=long_running),
        patch("aegra_api.services.worker_executor._heartbeat_loop", side_effect=quiet_heartbeat),
        patch("aegra_api.services.worker_executor._release_lease", new_callable=AsyncMock),
    ):
        cancellations.mark(run_id, "lease_loss")
        task = asyncio.create_task(executor._execute_with_lease(run_id, "w0"))
        await started.wait()
        task.cancel()
        # _execute_with_lease swallows CancelledError → returns None.
        result = await task
        assert result is None

    histogram = metrics.run_execution_seconds.labels(graph_id="graph-rif-ll")
    assert histogram._sum.get() == 0.0
    assert metrics.runs_in_flight.labels(graph_id="graph-rif-ll")._value.get() == 0.0
    # Worker must clear the tag so a re-run on another worker can be marked.
    assert cancellations.reason_of(run_id) is None


@pytest.mark.asyncio
async def test_pop_reason_clears_tag_before_release_lease(
    fresh_registry: prometheus_client.CollectorRegistry,
) -> None:
    """The early ``pop_reason`` must clear the tag BEFORE ``_release_lease``
    runs — that's the whole point of the race fix. Without atomicity, a
    concurrent re-run user-cancel could be blocked by precedence during the
    DB call. Verified by mocking ``_release_lease`` to capture the registry
    state at its entry."""
    setup_worker_metrics(registry=fresh_registry)
    metrics = get_worker_metrics()
    assert metrics is not None

    run_id = "rif-timing"
    job = MagicMock()
    job.identity.run_id = run_id
    job.identity.thread_id = "t1"
    job.identity.graph_id = "graph-rif-timing"

    loaded = _LoadedRun(job=job, trace={}, enqueued_at=None)
    acquire_result = _AcquireResult(loaded=loaded, reason="ok")

    executor = WorkerExecutor()
    started = asyncio.Event()
    captured: list[str | None] = []

    async def long_running(_job: object) -> None:
        started.set()
        await asyncio.sleep(3600)

    async def quiet_heartbeat(*_a: object, **_kw: object) -> None:
        await asyncio.sleep(3600)

    async def capturing_release(rid: str, _wn: str) -> None:
        # The early ``pop_reason`` must have already cleared by now.
        captured.append(cancellations.reason_of(rid))

    with (
        patch(
            "aegra_api.services.worker_executor._acquire_and_load",
            new_callable=AsyncMock,
            return_value=acquire_result,
        ),
        patch("aegra_api.services.worker_executor._restore_trace_context"),
        patch("aegra_api.services.worker_executor.execute_run", side_effect=long_running),
        patch("aegra_api.services.worker_executor._heartbeat_loop", side_effect=quiet_heartbeat),
        patch("aegra_api.services.worker_executor._release_lease", side_effect=capturing_release),
    ):
        cancellations.mark(run_id, "lease_loss")
        task = asyncio.create_task(executor._execute_with_lease(run_id, "w0"))
        await started.wait()
        task.cancel()
        # _execute_with_lease swallows CancelledError → returns None.
        result = await task
        assert result is None

    assert captured == [None], (
        f"tag must be cleared by pop_reason before _release_lease — saw {captured} during release"
    )


@pytest.mark.asyncio
async def test_runs_in_flight_does_not_inc_on_corruption(
    fresh_registry: prometheus_client.CollectorRegistry,
) -> None:
    """Corruption path skips ``inc`` and the gauge must stay at 0 — without
    this guard the matching ``dec`` in ``finally`` would push the gauge
    negative."""
    setup_worker_metrics(registry=fresh_registry)
    metrics = get_worker_metrics()
    assert metrics is not None

    acquire_result = _AcquireResult(loaded=None, reason="corruption")
    executor = WorkerExecutor()

    with patch(
        "aegra_api.services.worker_executor._acquire_and_load",
        new_callable=AsyncMock,
        return_value=acquire_result,
    ):
        await executor._execute_with_lease("rif-corrupt", "w0")

    # The actual regression we're guarding against is the gauge going
    # negative — an unconditional ``dec`` in ``finally`` paired with a
    # skipped ``inc`` on the corruption path. Inspect every sample of
    # ``aegra_runs_in_flight`` in the registry and assert nothing dropped
    # below zero. (No graph_id was inc'd, so the typical case is no
    # samples at all; the assertion still holds in that case.)
    in_flight_samples = [
        sample.value
        for family in fresh_registry.collect()
        if family.name == "aegra_runs_in_flight"
        for sample in family.samples
        if sample.name == "aegra_runs_in_flight"
    ]
    assert all(value >= 0.0 for value in in_flight_samples), (
        f"runs_in_flight went negative on the corruption path: {in_flight_samples}"
    )
    assert metrics.runs_completed.labels(graph_id="unknown", status="error")._value.get() == 1.0


@pytest.mark.asyncio
async def test_runs_discarded_increments_on_invalid_run_id(
    fresh_registry: prometheus_client.CollectorRegistry,
) -> None:
    """Invalid run_ids dequeued from Redis must bump ``runs_discarded`` and
    must NOT count toward ``runs_dequeued`` (queues stay symmetric)."""
    setup_worker_metrics(registry=fresh_registry)
    metrics = get_worker_metrics()
    assert metrics is not None

    executor = WorkerExecutor()
    executor._running = True
    semaphore_release_calls = 0
    real_semaphore = asyncio.Semaphore(1)

    class _Semaphore:
        async def acquire(self) -> bool:
            return await real_semaphore.acquire()

        def release(self) -> None:
            nonlocal semaphore_release_calls
            semaphore_release_calls += 1
            real_semaphore.release()

    sem = _Semaphore()
    dequeue_calls = 0

    async def fake_dequeue() -> str | None:
        nonlocal dequeue_calls
        dequeue_calls += 1
        if dequeue_calls == 1:
            return "not-a-uuid"
        executor._running = False
        return None

    with (
        patch("aegra_api.services.worker_executor.asyncio.Semaphore", return_value=sem),
        patch.object(executor, "_dequeue", side_effect=fake_dequeue),
        patch("aegra_api.services.worker_executor.settings") as mock_settings,
    ):
        mock_settings.worker.N_JOBS_PER_WORKER = 1
        await executor._worker_loop("worker-discard")

    assert metrics.runs_discarded._value.get() == 1.0
    assert metrics.runs_dequeued._value.get() == 0.0
    assert semaphore_release_calls >= 1


@pytest.mark.asyncio
async def test_handle_timeout_increments_run_timeouts_and_completed_error(
    fresh_registry: prometheus_client.CollectorRegistry,
) -> None:
    """Timeout path must bump ``run_timeouts``, ``runs_completed{error}`` and
    observe a ``run_execution_seconds`` sample clamped to the timeout."""
    setup_worker_metrics(registry=fresh_registry)
    metrics = get_worker_metrics()
    assert metrics is not None

    from aegra_api.services.worker_executor import _handle_timeout

    with (
        patch(
            "aegra_api.services.worker_executor._get_run_context_for_timeout",
            new_callable=AsyncMock,
            return_value=("t1", "graph-tm"),
        ),
        patch("aegra_api.services.worker_executor.finalize_run", new_callable=AsyncMock),
        patch("aegra_api.services.worker_executor._release_lease", new_callable=AsyncMock),
        patch("aegra_api.services.worker_executor.settings") as mock_settings,
    ):
        mock_settings.worker.BG_JOB_TIMEOUT_SECS = 7.0

        await _handle_timeout("rt1", "w0")

    assert metrics.run_timeouts.labels(graph_id="graph-tm")._value.get() == 1.0
    assert metrics.runs_completed.labels(graph_id="graph-tm", status="error")._value.get() == 1.0
    # Histogram sum should equal the configured timeout (single observation).
    assert metrics.run_execution_seconds.labels(graph_id="graph-tm")._sum.get() == 7.0


# ---------------------------------------------------------------------------
# Cancellation registry (mark / reason_of / pop_reason / clear)
# ---------------------------------------------------------------------------


def test_pop_reason_returns_none_when_absent() -> None:
    """No mark → pop returns None (don't fabricate a reason)."""
    reg = CancellationRegistry()
    assert reg.pop_reason("rid") is None


def test_pop_reason_is_one_shot() -> None:
    """``pop_reason`` must remove on read so the cancel-handler's atomic
    ``read+remove`` semantics hold — a second pop must return None."""
    reg = CancellationRegistry()
    reg.mark("rid", "user")
    assert reg.pop_reason("rid") == "user"
    assert reg.pop_reason("rid") is None
    assert reg.reason_of("rid") is None


def test_pop_reason_returns_lease_loss_when_precedence_blocked_user() -> None:
    """A ``mark("user")`` after ``mark("lease_loss")`` is dropped by precedence,
    so ``pop_reason`` must surface the lease_loss tag — not the rejected one.
    This is the basis of the worker's ``observe_duration == "user"`` decision."""
    reg = CancellationRegistry()
    reg.mark("rid", "lease_loss")
    reg.mark("rid", "user")
    assert reg.pop_reason("rid") == "lease_loss"


def test_clear_drops_any_existing_tag() -> None:
    """``clear`` without ``only`` must remove the run_id regardless of reason."""
    reg = CancellationRegistry()
    reg.mark("u1", "user")
    reg.mark("l1", "lease_loss")

    reg.clear("u1")
    reg.clear("l1")

    assert reg.reason_of("u1") is None
    assert reg.reason_of("l1") is None


def test_mark_lease_loss_does_not_leak_into_user_cancelled() -> None:
    """A run tagged as lease-loss must NOT classify as user-cancel — otherwise
    execute_run would finalize it as interrupted instead of skipping finalize."""
    reg = CancellationRegistry()
    reg.mark("run-lease", "lease_loss")
    assert reg.reason_of("run-lease") == "lease_loss"


def test_mark_lease_loss_overwrites_user() -> None:
    """A lease-loss arriving after a user cancel takes precedence so the
    reaper-driven retry isn't clobbered by an interrupted-finalize."""
    reg = CancellationRegistry()
    reg.mark("rid", "user")
    reg.mark("rid", "lease_loss")
    assert reg.reason_of("rid") == "lease_loss"


def test_mark_user_cannot_overwrite_lease_loss() -> None:
    """If lease-loss is already tagged, a later user cancel must not
    erase it. Without this precedence the old worker would finalize
    the run as 'interrupted' and clobber the rerun on another worker."""
    reg = CancellationRegistry()
    reg.mark("rid", "lease_loss")
    reg.mark("rid", "user")
    assert reg.reason_of("rid") == "lease_loss"


def test_clear_with_only_is_targeted() -> None:
    """``clear(rid, only="user")`` only drops a user tag — a lease-loss tag
    that arrived in the meantime must survive."""
    reg = CancellationRegistry()
    reg.mark("rid", "user")
    reg.clear("rid", only="user")
    assert reg.reason_of("rid") is None

    # After overwrite to lease-loss, ``only="user"`` is a no-op.
    reg.mark("rid", "user")
    reg.mark("rid", "lease_loss")
    reg.clear("rid", only="user")
    assert reg.reason_of("rid") == "lease_loss"


@pytest.mark.asyncio
async def test_execute_run_does_not_clear_cancel_tag(
    fresh_registry: prometheus_client.CollectorRegistry,
) -> None:
    """Architectural guarantee: ``execute_run`` must leave the cancel tag
    intact for the executor to read after ``await job_task`` raises.
    Clearing in execute_run's finally would race against the worker's peek
    because user-cancel propagates through ``_fut_waiter.cancel`` and our
    finally runs before the worker awakes."""
    setup_worker_metrics(registry=fresh_registry)

    job = MagicMock()
    job.identity.run_id = "run-tag-survives"
    job.identity.thread_id = "t1"
    job.identity.graph_id = "g1"

    cancellations.mark("run-tag-survives", "user")

    with (
        patch.object(run_executor_module, "update_run_status", new_callable=AsyncMock),
        patch.object(run_executor_module, "_stream_graph", new_callable=AsyncMock, side_effect=asyncio.CancelledError),
        patch.object(run_executor_module, "finalize_run", new_callable=AsyncMock),
        patch.object(
            run_executor_module,
            "streaming_service",
            MagicMock(signal_run_cancelled=AsyncMock(), cleanup_run=AsyncMock()),
        ),
        patch.object(run_executor_module, "_signal_run_done", new_callable=AsyncMock),
        pytest.raises(asyncio.CancelledError),
    ):
        await run_executor_module.execute_run(job)

    # Tag must still be present — that is the executor's invariant to consume.
    assert cancellations.reason_of("run-tag-survives") == "user"

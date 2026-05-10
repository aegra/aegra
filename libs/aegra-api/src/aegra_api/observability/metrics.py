"""Optional Prometheus metrics via prometheus-fastapi-instrumentator.

Controlled by ``ENABLE_PROMETHEUS_METRICS`` env var (default: false).
When enabled, exposes a ``/metrics`` endpoint with standard HTTP and
Python runtime metrics in Prometheus exposition format.

Custom worker and reaper metrics are registered separately via
``setup_worker_metrics()`` when both ``ENABLE_PROMETHEUS_METRICS`` and
``REDIS_BROKER_ENABLED`` are true.
"""

from dataclasses import dataclass
from typing import Any

import prometheus_client
import structlog
from fastapi import FastAPI
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from aegra_api.settings import settings

logger = structlog.getLogger(__name__)


def extract_graph_id(params: dict[str, Any] | None) -> str:
    """Read ``graph_id`` from a Run's ``execution_params`` JSON.

    ``RunJob.to_execution_params`` writes ``graph_id`` at the top level
    of the execution_params dict — that is the single source of truth.

    Returns ``"unknown"`` when ``params`` is None/empty, the value is
    missing, not a string, or an empty string — the metric label must
    be a non-empty identifier.
    """
    if not params:
        return "unknown"
    top = params.get("graph_id")
    if isinstance(top, str) and top:
        return top
    return "unknown"


# ---------------------------------------------------------------------------
# Custom histogram buckets sized for agent workloads (seconds to minutes)
# ---------------------------------------------------------------------------
# EXECUTION_BUCKETS goes up to 3600s — matches BG_JOB_TIMEOUT_SECS default,
# so timed-out runs (clamped to the timeout value when observed) land in a
# meaningful bucket rather than the +Inf overflow.
EXECUTION_BUCKETS = (0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600, 1200, 1800, 3600)
QUEUE_WAIT_BUCKETS = (0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300)
REAPER_CYCLE_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5)


# ---------------------------------------------------------------------------
# Worker metrics (15 metrics)
# ---------------------------------------------------------------------------
@dataclass
class WorkerMetrics:
    """Prometheus metrics for the worker executor."""

    # Run lifecycle
    runs_dispatched: Counter
    runs_completed: Counter
    runs_in_flight: Gauge
    runs_discarded: Counter
    run_execution_seconds: Histogram
    run_queue_wait_seconds: Histogram
    run_timeouts: Counter
    submit_errors: Counter

    # Dequeue
    runs_dequeued: Counter
    dequeue_errors: Counter
    redis_up: Gauge

    # Heartbeat & lease
    heartbeat_extensions: Counter
    heartbeat_failures: Counter
    lease_losses: Counter

    # Retries
    run_retries: Counter


# ---------------------------------------------------------------------------
# Reaper metrics (5 metrics)
# ---------------------------------------------------------------------------
@dataclass
class ReaperMetrics:
    """Prometheus metrics for the lease reaper."""

    crashed_recovered: Counter
    stuck_reenqueued: Counter
    permanently_failed: Counter
    cycle_seconds: Histogram
    queue_depth: Gauge


# ---------------------------------------------------------------------------
# Module-level singletons (None when metrics not enabled)
# ---------------------------------------------------------------------------
_worker_metrics: WorkerMetrics | None = None
_reaper_metrics: ReaperMetrics | None = None


def setup_worker_metrics(registry: CollectorRegistry | None = None) -> None:
    """Register worker + reaper metrics on the given (or default) registry.

    Called from the lifespan when ``ENABLE_PROMETHEUS_METRICS`` is true.
    Redis-specific metrics remain zero-valued when Redis is disabled.
    Accepts a custom ``registry`` for test isolation.

    Idempotent: a second call without an explicit ``registry`` is a no-op
    so that ``uvicorn --reload`` (or any double-lifespan scenario) does
    not raise ``Duplicated timeseries in CollectorRegistry``. Tests that
    pass an explicit ``registry`` always re-register on that registry.
    """
    global _worker_metrics, _reaper_metrics  # noqa: PLW0603
    if registry is None and _worker_metrics is not None:
        return
    reg = registry or prometheus_client.REGISTRY

    # Counter names use the explicit ``_total`` suffix so grep finds them
    # under their exposition name. prometheus_client recognizes the suffix
    # and does not double-append.
    _worker_metrics = WorkerMetrics(
        runs_dispatched=Counter(
            "aegra_runs_dispatched_total",
            "Total runs submitted to queue",
            ["graph_id"],
            registry=reg,
        ),
        runs_completed=Counter(
            "aegra_runs_completed_total",
            "Total runs that reached a terminal state",
            ["graph_id", "status"],
            registry=reg,
        ),
        runs_in_flight=Gauge(
            "aegra_runs_in_flight",
            "Currently executing runs",
            ["graph_id"],
            registry=reg,
        ),
        runs_discarded=Counter(
            "aegra_runs_discarded_total",
            "Invalid run_ids dequeued and dropped",
            registry=reg,
        ),
        run_execution_seconds=Histogram(
            "aegra_run_execution_seconds",
            "Run execution duration",
            ["graph_id"],
            buckets=EXECUTION_BUCKETS,
            registry=reg,
        ),
        run_queue_wait_seconds=Histogram(
            "aegra_run_queue_wait_seconds",
            "Time from submit to lease acquire",
            ["graph_id"],
            buckets=QUEUE_WAIT_BUCKETS,
            registry=reg,
        ),
        run_timeouts=Counter(
            "aegra_run_timeouts_total",
            "Runs killed by timeout",
            ["graph_id"],
            registry=reg,
        ),
        submit_errors=Counter(
            "aegra_submit_errors_total",
            "Failed rpush to Redis on submit",
            ["graph_id"],
            registry=reg,
        ),
        runs_dequeued=Counter(
            "aegra_runs_dequeued_total",
            "Successful BLPOP dequeues",
            registry=reg,
        ),
        dequeue_errors=Counter(
            "aegra_dequeue_errors_total",
            "Redis errors during dequeue",
            registry=reg,
        ),
        redis_up=Gauge(
            "aegra_redis_up",
            "1 if Redis responded to the last BLPOP, 0 otherwise",
            registry=reg,
        ),
        heartbeat_extensions=Counter(
            "aegra_heartbeat_extensions_total",
            "Successful lease heartbeats",
            ["graph_id"],
            registry=reg,
        ),
        heartbeat_failures=Counter(
            "aegra_heartbeat_failures_total",
            "Failed lease heartbeats",
            ["graph_id"],
            registry=reg,
        ),
        lease_losses=Counter(
            "aegra_lease_losses_total",
            "Lease lost during execution. ``reason`` is ``rowcount_zero`` "
            "when the heartbeat UPDATE found no row owned by this worker "
            "(claimed by another worker), or ``heartbeat_timeout`` when "
            "consecutive heartbeat failures exceeded the lease duration.",
            ["graph_id", "reason"],
            registry=reg,
        ),
        run_retries=Counter(
            "aegra_run_retries_total",
            "Runs retried after crash",
            ["graph_id", "retry_number"],
            registry=reg,
        ),
    )

    _reaper_metrics = ReaperMetrics(
        crashed_recovered=Counter(
            "aegra_reaper_crashed_recovered_total",
            "Crashed runs recovered by reaper",
            registry=reg,
        ),
        stuck_reenqueued=Counter(
            "aegra_reaper_stuck_reenqueued_total",
            "Stuck pending runs re-enqueued",
            registry=reg,
        ),
        permanently_failed=Counter(
            "aegra_reaper_permanently_failed_total",
            "Runs exceeding max retries",
            registry=reg,
        ),
        cycle_seconds=Histogram(
            "aegra_reaper_cycle_seconds",
            "Reaper cycle duration",
            buckets=REAPER_CYCLE_BUCKETS,
            registry=reg,
        ),
        queue_depth=Gauge(
            "aegra_queue_depth",
            "Number of run_ids in Redis job queue",
            registry=reg,
        ),
    )

    logger.info("Worker and reaper metrics registered", worker_count=15, reaper_count=5)


def get_worker_metrics() -> WorkerMetrics | None:
    """Return worker metrics or None if not registered."""
    return _worker_metrics


def get_reaper_metrics() -> ReaperMetrics | None:
    """Return reaper metrics or None if not registered."""
    return _reaper_metrics


def setup_prometheus_metrics(
    app: FastAPI,
    registry: prometheus_client.CollectorRegistry | None = None,
) -> None:
    """Conditionally attach Prometheus instrumentator to the app.

    No-op when ``ENABLE_PROMETHEUS_METRICS`` is false.

    Args:
        app: FastAPI application instance.
        registry: Optional Prometheus collector registry. When provided, metrics
            are collected into this registry instead of the global default.
            Primarily useful in tests to avoid cross-test pollution.

    Note:
        The ``/metrics`` endpoint is **not** protected by Aegra's authentication
        middleware. This is intentional — Prometheus scrapers typically do not
        support application-level auth. If the endpoint must be restricted, use
        network-level controls (firewall rules, internal load-balancer, etc.).
    """
    if not settings.observability.ENABLE_PROMETHEUS_METRICS:
        return

    instrumentator = Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
        excluded_handlers=["/health", "/ready", "/live", "/info", "/metrics", "/docs", "/redoc", "/openapi.json"],
        registry=registry,
    )
    instrumentator.instrument(app)
    instrumentator.expose(app, endpoint="/metrics", include_in_schema=False)
    logger.info("Prometheus metrics enabled at /metrics")

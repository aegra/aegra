"""Optional Prometheus metrics via prometheus-fastapi-instrumentator.

Controlled by ``ENABLE_PROMETHEUS_METRICS`` env var (default: false).
When enabled, exposes a ``/metrics`` endpoint with standard HTTP and
Python runtime metrics in Prometheus exposition format.
"""

import prometheus_client
import structlog
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from aegra_api.settings import settings

logger = structlog.getLogger(__name__)

REAPER_RECOVERED_RUNS = prometheus_client.Counter(
    "aegra_reaper_recovered_runs_total",
    "Runs recovered by the lease reaper, by outcome: crashed_retried "
    "(expired lease, re-enqueued), crashed_exhausted (max retries exceeded, "
    "marked failed), stuck_pending (never claimed, re-enqueued). Counts only "
    "confirmed Redis pushes and DB updates; recovery that falls back to the "
    "workers' Postgres poll during a Redis outage is not counted.",
    labelnames=["outcome"],
)

# Pre-create label children so every outcome series renders as 0 on /metrics
# before the first recovery event (absent series break rate() alerts).
for _outcome in ("crashed_retried", "crashed_exhausted", "stuck_pending"):
    REAPER_RECOVERED_RUNS.labels(outcome=_outcome)


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

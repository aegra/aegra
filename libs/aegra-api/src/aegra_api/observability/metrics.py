"""Optional Prometheus metrics via prometheus-fastapi-instrumentator.

Controlled by ``ENABLE_PROMETHEUS_METRICS`` env var (default: false).
When enabled, exposes a ``/metrics`` endpoint with standard HTTP and
Python runtime metrics in Prometheus text, or JSON when ``format=json``.
"""

import math
import os
from typing import Any, Literal

import prometheus_client
import structlog
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, Response
from prometheus_client.multiprocess import MultiProcessCollector
from prometheus_fastapi_instrumentator import Instrumentator

from aegra_api.settings import settings

logger = structlog.getLogger(__name__)

REAPER_RECOVERED_RUNS = prometheus_client.Counter(
    "aegra_reaper_recovered_runs_total",
    "Runs recovered by the lease reaper, by outcome: crashed_retried "
    "(expired lease, re-enqueued), crashed_exhausted (max retries exceeded, "
    "marked failed), stuck_pending (missing from the queue, re-enqueued). Counts only "
    "confirmed Redis pushes and DB updates; recovery that falls back to the "
    "workers' Postgres poll during a Redis outage is not counted.",
    labelnames=["outcome"],
)

# Pre-create label children so every outcome series renders as 0 on /metrics
# before the first recovery event (absent series break rate() alerts).
for _outcome in ("crashed_retried", "crashed_exhausted", "stuck_pending"):
    REAPER_RECOVERED_RUNS.labels(outcome=_outcome)

THREAD_TTL_SWEPT = prometheus_client.Counter(
    "aegra_thread_ttl_swept_threads_total",
    "Threads processed by the TTL sweep or POST /threads/prune, by outcome: "
    "deleted (strategy=delete, thread removed), pruned (strategy=keep_latest, "
    "history compacted), error (item failed and will be retried next tick).",
    labelnames=["outcome"],
)

for _outcome in ("deleted", "pruned", "error"):
    THREAD_TTL_SWEPT.labels(outcome=_outcome)


def _json_sample_value(value: float) -> float | None:
    # Starlette's JSON encoder rejects NaN and Infinity.
    if not math.isfinite(value):
        return None
    return value


def _prometheus_family_name(name: str, metric_type: str) -> str:
    # prometheus_client stores counters without the _total suffix it writes in text.
    if metric_type == "counter":
        return f"{name}_total"
    if metric_type == "info":
        return f"{name}_info"
    return name


def registry_to_json(registry: prometheus_client.CollectorRegistry) -> dict[str, Any]:
    """One object keyed by family name, each value ``{type, help, samples}``."""
    families: dict[str, Any] = {}
    for metric in registry.collect():
        families[_prometheus_family_name(metric.name, metric.type)] = {
            "type": metric.type,
            "help": metric.documentation,
            "samples": [
                {
                    "name": sample.name,
                    "labels": dict(sample.labels),
                    "value": _json_sample_value(sample.value),
                }
                for sample in metric.samples
            ],
        }
    return families


def _scrape_registry(registry: prometheus_client.CollectorRegistry) -> prometheus_client.CollectorRegistry:
    # Multiprocess values live in mmap files, so each scrape needs a fresh collector.
    if "PROMETHEUS_MULTIPROC_DIR" not in os.environ:
        return registry
    ephemeral = prometheus_client.CollectorRegistry()
    MultiProcessCollector(ephemeral)
    return ephemeral


def _add_metrics_route(app: FastAPI, registry: prometheus_client.CollectorRegistry) -> None:
    @app.get("/metrics", include_in_schema=False)
    def metrics(
        output_format: Literal["prometheus", "json"] = Query(default="prometheus", alias="format"),
    ) -> Response:
        scrape_registry = _scrape_registry(registry)
        if output_format == "json":
            return JSONResponse(registry_to_json(scrape_registry))
        return Response(
            prometheus_client.generate_latest(scrape_registry),
            media_type=prometheus_client.CONTENT_TYPE_LATEST,
        )


def setup_prometheus_metrics(
    app: FastAPI,
    registry: prometheus_client.CollectorRegistry | None = None,
) -> None:
    """Conditionally attach Prometheus instrumentator to the app.

    No-op when ``ENABLE_PROMETHEUS_METRICS`` is false. ``GET /metrics``
    returns Prometheus text, or the same series as JSON when ``format=json``.

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
    target = registry if registry is not None else prometheus_client.REGISTRY
    _add_metrics_route(app, target)
    logger.info("Prometheus metrics enabled at /metrics")

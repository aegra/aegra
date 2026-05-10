"""Integration tests for worker and reaper Prometheus metrics on /metrics endpoint."""

import prometheus_client
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from aegra_api.observability import metrics as metrics_module
from aegra_api.observability.metrics import (
    setup_prometheus_metrics,
    setup_worker_metrics,
)


@pytest.fixture(autouse=True)
def _reset_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset module-level metric singletons before each test."""
    monkeypatch.setattr(metrics_module, "_worker_metrics", None)
    monkeypatch.setattr(metrics_module, "_reaper_metrics", None)


@pytest.fixture
def fresh_registry() -> prometheus_client.CollectorRegistry:
    """Return an isolated Prometheus registry to avoid global state leaks."""
    return prometheus_client.CollectorRegistry()


def _make_app(
    monkeypatch: pytest.MonkeyPatch,
    registry: prometheus_client.CollectorRegistry,
    *,
    enable_prometheus: bool = True,
    register_worker_metrics: bool = True,
) -> FastAPI:
    """Create a minimal FastAPI app with Prometheus and optionally worker metrics."""
    app = FastAPI()

    @app.get("/hello")
    def hello() -> dict[str, str]:
        return {"msg": "world"}

    monkeypatch.setattr(metrics_module.settings.observability, "ENABLE_PROMETHEUS_METRICS", enable_prometheus)
    setup_prometheus_metrics(app, registry=registry)

    if register_worker_metrics:
        setup_worker_metrics(registry=registry)

    return app


def test_metrics_endpoint_includes_worker_metrics(
    monkeypatch: pytest.MonkeyPatch,
    fresh_registry: prometheus_client.CollectorRegistry,
) -> None:
    """Worker metric names should appear in the /metrics scrape output."""
    app = _make_app(monkeypatch, fresh_registry)
    client = TestClient(app)

    response = client.get("/metrics")
    assert response.status_code == 200

    body = response.text
    # Spot-check representative worker metrics — Counters are exposed with
    # ``_total`` suffix, Gauges/Histograms without.
    assert "aegra_runs_dispatched_total" in body
    assert "aegra_runs_completed_total" in body
    assert "aegra_runs_in_flight" in body
    assert "aegra_run_execution_seconds" in body
    assert "aegra_run_queue_wait_seconds" in body
    assert "aegra_redis_up" in body
    assert "aegra_run_retries_total" in body


def test_metrics_endpoint_includes_reaper_metrics(
    monkeypatch: pytest.MonkeyPatch,
    fresh_registry: prometheus_client.CollectorRegistry,
) -> None:
    """Reaper metric names should appear in the /metrics scrape output."""
    app = _make_app(monkeypatch, fresh_registry)
    client = TestClient(app)

    response = client.get("/metrics")
    assert response.status_code == 200

    body = response.text
    assert "aegra_reaper_crashed_recovered_total" in body
    assert "aegra_reaper_stuck_reenqueued_total" in body
    assert "aegra_reaper_permanently_failed_total" in body
    assert "aegra_reaper_cycle_seconds" in body
    assert "aegra_queue_depth" in body


def test_worker_metrics_not_in_output_when_not_registered(
    monkeypatch: pytest.MonkeyPatch,
    fresh_registry: prometheus_client.CollectorRegistry,
) -> None:
    """When worker metrics are not registered, they should not appear in /metrics."""
    app = _make_app(monkeypatch, fresh_registry, register_worker_metrics=False)
    client = TestClient(app)

    response = client.get("/metrics")
    assert response.status_code == 200

    body = response.text
    assert "aegra_runs_dispatched_total" not in body
    assert "aegra_reaper_crashed_recovered_total" not in body


def test_worker_metrics_not_exposed_when_prometheus_disabled(
    monkeypatch: pytest.MonkeyPatch,
    fresh_registry: prometheus_client.CollectorRegistry,
) -> None:
    """When Prometheus is disabled, /metrics endpoint should not exist."""
    app = _make_app(monkeypatch, fresh_registry, enable_prometheus=False, register_worker_metrics=False)
    client = TestClient(app)

    response = client.get("/metrics")
    assert response.status_code == 404


def test_metrics_value_changes_after_increment(
    monkeypatch: pytest.MonkeyPatch,
    fresh_registry: prometheus_client.CollectorRegistry,
) -> None:
    """Proves the registry is *wired* to the /metrics endpoint, not just
    that names are present. Drives an increment then asserts the scrape
    output reflects the new value."""
    app = _make_app(monkeypatch, fresh_registry)
    metrics = metrics_module.get_worker_metrics()
    assert metrics is not None

    metrics.runs_dispatched.labels(graph_id="test-pipe").inc()

    client = TestClient(app)
    response = client.get("/metrics")
    assert response.status_code == 200

    body = response.text
    # The exposition format is ``aegra_runs_dispatched_total{graph_id="test-pipe"} 1.0``.
    assert 'aegra_runs_dispatched_total{graph_id="test-pipe"} 1.0' in body


def test_lease_losses_carries_reason_label(
    monkeypatch: pytest.MonkeyPatch,
    fresh_registry: prometheus_client.CollectorRegistry,
) -> None:
    """The ``reason`` label must distinguish ``rowcount_zero`` from
    ``heartbeat_timeout`` so on-call can tell the two failure modes
    apart from a single dashboard."""
    app = _make_app(monkeypatch, fresh_registry)
    metrics = metrics_module.get_worker_metrics()
    assert metrics is not None

    metrics.lease_losses.labels(graph_id="g", reason="rowcount_zero").inc()
    metrics.lease_losses.labels(graph_id="g", reason="heartbeat_timeout").inc(2)

    client = TestClient(app)
    body = client.get("/metrics").text

    assert 'aegra_lease_losses_total{graph_id="g",reason="rowcount_zero"} 1.0' in body
    assert 'aegra_lease_losses_total{graph_id="g",reason="heartbeat_timeout"} 2.0' in body

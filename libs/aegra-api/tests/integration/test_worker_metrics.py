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
    # Spot-check representative worker metrics
    assert "aegra_runs_dispatched" in body
    assert "aegra_runs_completed" in body
    assert "aegra_runs_in_flight" in body
    assert "aegra_run_execution_seconds" in body
    assert "aegra_run_queue_wait_seconds" in body
    assert "aegra_redis_reachable" in body
    assert "aegra_run_retries" in body


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
    assert "aegra_reaper_crashed_recovered" in body
    assert "aegra_reaper_stuck_reenqueued" in body
    assert "aegra_reaper_permanently_failed" in body
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
    assert "aegra_runs_dispatched" not in body
    assert "aegra_reaper_crashed_recovered" not in body


def test_worker_metrics_not_exposed_when_prometheus_disabled(
    monkeypatch: pytest.MonkeyPatch,
    fresh_registry: prometheus_client.CollectorRegistry,
) -> None:
    """When Prometheus is disabled, /metrics endpoint should not exist."""
    app = _make_app(monkeypatch, fresh_registry, enable_prometheus=False, register_worker_metrics=False)
    client = TestClient(app)

    response = client.get("/metrics")
    assert response.status_code == 404

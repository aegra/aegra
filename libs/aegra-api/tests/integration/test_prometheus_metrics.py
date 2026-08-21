"""Integration tests for the Prometheus /metrics endpoint."""

import prometheus_client
import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from aegra_api.observability import metrics as metrics_module
from aegra_api.observability.metrics import setup_prometheus_metrics


@pytest.fixture
def fresh_registry() -> prometheus_client.CollectorRegistry:
    """Return an isolated Prometheus registry to avoid global state leaks."""
    return prometheus_client.CollectorRegistry()


def _make_app(
    monkeypatch: pytest.MonkeyPatch,
    registry: prometheus_client.CollectorRegistry,
) -> FastAPI:
    """Create a minimal FastAPI app with Prometheus metrics enabled."""
    app = FastAPI()

    @app.get("/hello")
    def hello() -> dict[str, str]:
        return {"msg": "world"}

    monkeypatch.setattr(metrics_module.settings.observability, "ENABLE_PROMETHEUS_METRICS", True)
    setup_prometheus_metrics(app, registry=registry)
    return app


def test_metrics_endpoint_returns_prometheus_format(
    monkeypatch: pytest.MonkeyPatch,
    fresh_registry: prometheus_client.CollectorRegistry,
) -> None:
    """Test that /metrics returns text in Prometheus exposition format."""
    app = _make_app(monkeypatch, fresh_registry)
    client = TestClient(app)

    # Make a request so there's something to report
    response = client.get("/hello")
    assert response.status_code == 200

    # Scrape metrics
    metrics_response = client.get("/metrics")
    assert metrics_response.status_code == 200
    assert "text/plain" in metrics_response.headers["content-type"]

    body = metrics_response.text
    # Should contain standard HTTP metrics from the instrumentator
    assert "http_request_duration" in body or "http_requests" in body


def test_instrumented_request_on_included_router_route_does_not_500(
    monkeypatch: pytest.MonkeyPatch,
    fresh_registry: prometheus_client.CollectorRegistry,
) -> None:
    """A templated route added via include_router must not 500 with metrics on.

    Regression for FastAPI >=0.137: include_router puts an _IncludedRouter (no
    .path) in app.routes. The instrumentator walks app.routes and reads .path on
    a match, which raised AttributeError on every request with the old pin. Aegra
    mounts all its real routers via include_router, so this is the production path.
    """
    app = FastAPI()
    router = APIRouter()

    @router.get("/items/{item_id}")
    def get_item(item_id: str) -> dict[str, str]:
        return {"item_id": item_id}

    app.include_router(router)
    monkeypatch.setattr(metrics_module.settings.observability, "ENABLE_PROMETHEUS_METRICS", True)
    setup_prometheus_metrics(app, registry=fresh_registry)

    client = TestClient(app)
    response = client.get("/items/42")
    assert response.status_code == 200, f"instrumented include_router route 500'd: {response.text}"
    assert response.json() == {"item_id": "42"}


def test_reaper_counter_exposed_in_default_registry() -> None:
    """The reaper counter must live in the default registry the app exposes.

    ``setup_prometheus_metrics`` is called without a registry in production
    (main.py), so /metrics serves ``prometheus_client.REGISTRY``. All three
    outcome series are pre-created and must render even before any recovery.
    """
    body = prometheus_client.generate_latest(prometheus_client.REGISTRY).decode()

    assert "aegra_reaper_recovered_runs_total" in body
    for outcome in ("crashed_retried", "crashed_exhausted", "stuck_pending"):
        assert f'outcome="{outcome}"' in body


def test_metrics_endpoint_not_exposed_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that /metrics is not available when metrics are disabled."""
    app = FastAPI()

    @app.get("/hello")
    def hello() -> dict[str, str]:
        return {"msg": "world"}

    monkeypatch.setattr(metrics_module.settings.observability, "ENABLE_PROMETHEUS_METRICS", False)
    setup_prometheus_metrics(app)

    client = TestClient(app)
    response = client.get("/metrics")
    assert response.status_code == 404

"""Unit tests for the Prometheus metrics setup module."""

from unittest.mock import MagicMock

import prometheus_client
import pytest
from fastapi import FastAPI

from aegra_api.observability import metrics as metrics_module
from aegra_api.observability.metrics import setup_prometheus_metrics


@pytest.fixture
def app() -> FastAPI:
    return FastAPI()


@pytest.fixture
def fresh_registry() -> prometheus_client.CollectorRegistry:
    return prometheus_client.CollectorRegistry()


def test_noop_when_disabled(app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that no instrumentation is attached when metrics are disabled."""
    monkeypatch.setattr(metrics_module.settings.observability, "ENABLE_PROMETHEUS_METRICS", False)
    setup_prometheus_metrics(app)

    paths = {route.path for route in app.routes if hasattr(route, "path")}
    assert "/metrics" not in paths


def test_exposes_metrics_endpoint_when_enabled(
    app: FastAPI,
    fresh_registry: prometheus_client.CollectorRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that /metrics endpoint is added when metrics are enabled."""
    monkeypatch.setattr(metrics_module.settings.observability, "ENABLE_PROMETHEUS_METRICS", True)
    setup_prometheus_metrics(app, registry=fresh_registry)

    paths = {route.path for route in app.routes if hasattr(route, "path")}
    assert "/metrics" in paths


def test_excludes_health_and_docs(
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that the instrumentator excludes health and docs endpoints."""
    monkeypatch.setattr(metrics_module.settings.observability, "ENABLE_PROMETHEUS_METRICS", True)
    mock_cls = MagicMock()
    mock_instance = MagicMock()
    mock_cls.return_value = mock_instance
    mock_instance.instrument.return_value = mock_instance
    monkeypatch.setattr(metrics_module, "Instrumentator", mock_cls)

    setup_prometheus_metrics(app)

    mock_cls.assert_called_once_with(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
        should_instrument_requests_inprogress=True,
        excluded_handlers=[
            "/health",
            "/ready",
            "/live",
            "/info",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json",
        ],
        registry=None,
        inprogress_name="http_server_active_requests",
        inprogress_labels=True,
    )
    mock_instance.instrument.assert_called_once_with(app)
    mock_instance.expose.assert_called_once_with(
        app,
        endpoint="/metrics",
        include_in_schema=False,
    )


def test_logs_when_enabled(
    app: FastAPI,
    fresh_registry: prometheus_client.CollectorRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that a log message is emitted when metrics are enabled."""
    monkeypatch.setattr(metrics_module.settings.observability, "ENABLE_PROMETHEUS_METRICS", True)
    mock_logger = MagicMock()
    monkeypatch.setattr(metrics_module, "logger", mock_logger)

    setup_prometheus_metrics(app, registry=fresh_registry)

    mock_logger.info.assert_called_once_with("Prometheus metrics enabled at /metrics")

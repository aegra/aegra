"""Unit tests for the Prometheus metrics setup module."""

from unittest.mock import MagicMock

import prometheus_client
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from aegra_api.observability import metrics as metrics_module
from aegra_api.observability.metrics import registry_to_json, setup_prometheus_metrics


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
    )
    mock_instance.instrument.assert_called_once_with(app)
    mock_instance.expose.assert_not_called()
    paths = {route.path for route in app.routes if hasattr(route, "path")}
    assert "/metrics" in paths


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


def test_json_format_serves_default_registry_counters(
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no registry argument, format=json reads prometheus_client.REGISTRY."""
    monkeypatch.setattr(metrics_module.settings.observability, "ENABLE_PROMETHEUS_METRICS", True)
    mock_cls = MagicMock()
    mock_instance = MagicMock()
    mock_cls.return_value = mock_instance
    mock_instance.instrument.return_value = mock_instance
    monkeypatch.setattr(metrics_module, "Instrumentator", mock_cls)

    setup_prometheus_metrics(app)

    response = TestClient(app).get("/metrics", params={"format": "json"})
    assert response.status_code == 200
    assert "application/json" in response.headers["content-type"]

    reaper = response.json()["aegra_reaper_recovered_runs_total"]
    assert reaper["type"] == "counter"
    outcomes = {sample["labels"]["outcome"] for sample in reaper["samples"] if "outcome" in sample["labels"]}
    assert outcomes == {"crashed_retried", "crashed_exhausted", "stuck_pending"}


def test_registry_to_json_coerces_non_finite_samples(
    fresh_registry: prometheus_client.CollectorRegistry,
) -> None:
    """NaN and Infinity become null so the JSON scrape cannot 500."""
    gauge = prometheus_client.Gauge(
        "aegra_test_nonfinite",
        "non-finite sample",
        ["kind"],
        registry=fresh_registry,
    )
    gauge.labels(kind="nan").set(float("nan"))
    gauge.labels(kind="inf").set(float("inf"))
    gauge.labels(kind="ok").set(1.5)

    samples = {
        sample["labels"]["kind"]: sample["value"]
        for sample in registry_to_json(fresh_registry)["aegra_test_nonfinite"]["samples"]
        if sample["name"] == "aegra_test_nonfinite"
    }
    assert samples["nan"] is None
    assert samples["inf"] is None
    assert samples["ok"] == 1.5


def test_registry_to_json_uses_info_exposition_name(
    fresh_registry: prometheus_client.CollectorRegistry,
) -> None:
    """Info families keep the _info suffix Prometheus writes in text."""
    info = prometheus_client.Info(
        "aegra_test_build",
        "build information",
        registry=fresh_registry,
    )
    info.info({"version": "1"})

    body = registry_to_json(fresh_registry)
    assert "aegra_test_build_info" in body
    assert body["aegra_test_build_info"]["type"] == "info"
    samples = body["aegra_test_build_info"]["samples"]
    assert samples[0]["name"] == "aegra_test_build_info"
    assert samples[0]["labels"] == {"version": "1"}
    assert samples[0]["value"] == 1.0

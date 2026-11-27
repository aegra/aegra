"""E2E tests for the Prometheus /metrics endpoint.

Requires a running server with ENABLE_PROMETHEUS_METRICS=true.
"""

import httpx
import pytest

from aegra_api.settings import settings


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_metrics_endpoint_returns_prometheus_format() -> None:
    """Test that /metrics returns Prometheus exposition text from a real server."""
    if not settings.observability.ENABLE_PROMETHEUS_METRICS:
        pytest.skip("ENABLE_PROMETHEUS_METRICS=false; /metrics endpoint is intentionally disabled")

    server_url = settings.app.SERVER_URL

    async with httpx.AsyncClient(base_url=server_url, timeout=5.0) as client:
        # Make a request so there's something to report
        await client.get("/health")

        response = await client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]

        body = response.text
        assert "# HELP" in body
        assert "# TYPE" in body


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_metrics_endpoint_returns_json_format() -> None:
    """Test that /metrics?format=json returns the same registry as JSON."""
    if not settings.observability.ENABLE_PROMETHEUS_METRICS:
        pytest.skip("ENABLE_PROMETHEUS_METRICS=false; /metrics endpoint is intentionally disabled")

    server_url = settings.app.SERVER_URL

    async with httpx.AsyncClient(base_url=server_url, timeout=5.0) as client:
        response = await client.get("/metrics", params={"format": "json"})
        assert response.status_code == 200
        assert "application/json" in response.headers["content-type"]

        body = response.json()
        assert isinstance(body, dict)
        assert body
        family = next(iter(body.values()))
        assert set(family) >= {"type", "help", "samples"}

        rejected = await client.get("/metrics", params={"format": "xml"})
        assert rejected.status_code == 422

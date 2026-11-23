"""Integration tests for health and info endpoints."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from aegra_api.core import health
from aegra_api.main import agent_protocol_exception_handler
from aegra_api.settings import settings
from tests.fixtures.clients import create_test_app, make_client


def test_info_reports_cron_flag_from_runtime_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """GET /info should expose the current CRON_ENABLED runtime flag."""
    monkeypatch.setattr(settings.cron, "CRON_ENABLED", False)

    client = make_client(create_test_app())
    response = client.get("/info")

    assert response.status_code == 200
    assert response.json()["flags"] == {"assistants": True, "crons": False}


@pytest.fixture
def ready_backend(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    backend = MagicMock()
    backend.get_checkpointer.return_value.aget_tuple = AsyncMock(return_value=None)
    backend.get_store.return_value.aget = AsyncMock(return_value=None)
    monkeypatch.setattr(health, "db_manager", backend)
    return backend


@pytest.fixture
def ready_client() -> TestClient:
    app = create_test_app(include_runs=False, include_threads=False)
    app.add_exception_handler(HTTPException, agent_protocol_exception_handler)
    return make_client(app)


@pytest.mark.parametrize("failure", ["connect", "query", "checkpointer", "store"])
def test_ready_redacts_backend_exception_details(
    ready_backend: MagicMock, ready_client: TestClient, failure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    warning = MagicMock()
    monkeypatch.setattr(health.logger, "warning", warning)
    marker = "connection failed to db-internal.example.invalid:5432; database=synthetic_test_db"
    error = RuntimeError(marker)
    context = ready_backend.engine.begin.return_value
    if failure == "connect":
        context.__aenter__.side_effect = error
    elif failure == "query":
        context.__aenter__.return_value.execute.side_effect = error
    elif failure == "checkpointer":
        ready_backend.get_checkpointer.side_effect = error
    else:
        ready_backend.get_store.side_effect = error
    category = "database error" if failure in {"connect", "query"} else "components unavailable"

    response = ready_client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {
        "error": "service_unavailable",
        "message": f"Service not ready - {category}",
        "details": None,
    }
    assert marker not in response.text
    assert "db-internal.example.invalid" not in response.text
    assert "synthetic_test_db" not in response.text
    log_category = "database" if failure in {"connect", "query"} else "components"
    warning.assert_called_once_with(f"readiness check failed: {log_category}", exc_info=True)


def test_ready_preserves_missing_engine_response(ready_backend: MagicMock, ready_client: TestClient) -> None:
    ready_backend.engine = None

    response = ready_client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {
        "error": "service_unavailable",
        "message": "Service not ready - database engine not initialized",
        "details": None,
    }


@pytest.mark.parametrize("probe_failure", [False, True])
def test_ready_preserves_success_response(
    ready_backend: MagicMock, ready_client: TestClient, probe_failure: bool
) -> None:
    if probe_failure:
        ready_backend.get_checkpointer.return_value.aget_tuple.side_effect = RuntimeError("synthetic probe failure")
        ready_backend.get_store.return_value.aget.side_effect = RuntimeError("synthetic probe failure")

    response = ready_client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}

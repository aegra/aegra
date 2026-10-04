"""Integration tests for the A2A agent card endpoints"""

from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from aegra_api.api import a2a as a2a_module
from aegra_api.services.assistant_service import get_assistant_service
from aegra_api.utils.assistants import resolve_assistant_id
from tests.fixtures.clients import create_test_app, make_client
from tests.fixtures.test_helpers import make_assistant

GRAPHS: dict[str, str] = {"agent": "agent.py"}


@pytest.fixture
def client(mock_assistant_service: AsyncMock) -> Iterator[TestClient]:
    """Test client with the A2A router and a mocked assistant service."""
    mock_assistant_service.langgraph_service = MagicMock()
    mock_assistant_service.langgraph_service.list_graphs.return_value = GRAPHS
    mock_assistant_service.get_assistant.return_value = make_assistant(name="Support Agent", graph_id="agent")
    mock_assistant_service.get_assistant_schemas.return_value = {
        "input_schema": {"properties": {"messages": {}}, "required": ["messages"]},
        "state_schema": None,
    }

    app = create_test_app(include_runs=False, include_threads=False)
    app.include_router(a2a_module.router)
    app.dependency_overrides[get_assistant_service] = lambda: mock_assistant_service

    a2a_module.get_default_a2a_assistant.cache_clear()
    yield make_client(app)
    a2a_module.get_default_a2a_assistant.cache_clear()


@pytest.mark.parametrize(
    "path",
    ["/a2a/asst-1", "/a2a/asst-1/.well-known/agent-card.json", "/a2a/asst-1/.well-known/agent.json"],
)
def test_agent_card_is_served_on_every_card_path(client: TestClient, path: str) -> None:
    """All three per-assistant paths return the same card."""
    resp = client.get(path)

    assert resp.status_code == 200
    card = resp.json()
    assert card["protocolVersion"] == "1.0"
    assert card["name"] == "Support Agent"
    assert card["url"] == "http://testserver/a2a/asst-1"
    assert card["skills"][0]["metadata"]["inputSchema"]["supportsA2A"] is True


def test_agent_card_returns_404_when_assistant_not_found(client: TestClient, mock_assistant_service: AsyncMock) -> None:
    """The service's 404 for a missing or unowned assistant reaches the client."""
    mock_assistant_service.get_assistant.side_effect = HTTPException(404, "Assistant 'nope' not found")

    resp = client.get("/a2a/nope")

    assert resp.status_code == 404


def test_agent_card_accepts_a_graph_id(client: TestClient, mock_assistant_service: AsyncMock) -> None:
    """A graph id is resolved to its assistant for the lookup but kept in the card's URL."""
    resp = client.get("/a2a/agent")

    assert resp.status_code == 200
    assert resp.json()["url"] == "http://testserver/a2a/agent"
    mock_assistant_service.get_assistant.assert_awaited_once_with(resolve_assistant_id("agent", GRAPHS))


def test_host_root_card_serves_the_configured_assistant(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """With http.a2a_default_assistant set, the host root serves that assistant's card."""
    monkeypatch.setattr(a2a_module, "load_http_config", lambda: {"a2a_default_assistant": "agent"})

    resp = client.get("/.well-known/agent-card.json")

    assert resp.status_code == 200
    assert resp.json()["url"] == "http://testserver/a2a/agent"


def test_host_root_card_returns_404_naming_the_config_key_when_unset(
    client: TestClient, mock_assistant_service: AsyncMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a configured default the route must not guess an assistant."""
    monkeypatch.setattr(a2a_module, "load_http_config", lambda: None)

    resp = client.get("/.well-known/agent-card.json")

    assert resp.status_code == 404
    assert "http.a2a_default_assistant" in resp.json()["detail"]
    mock_assistant_service.get_assistant.assert_not_awaited()

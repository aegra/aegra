"""Integration tests for the MCP adapter endpoint."""

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from aegra_api.adapters.mcp_adapter import mcp_lifespan, mount_mcp, register_mcp_tools


async def _make_mcp_app(registry: dict[str, Any] | None = None) -> FastAPI:
    """Build a minimal FastAPI app with MCP mounted and its lifespan wired."""

    @asynccontextmanager
    async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with mcp_lifespan():
            yield

    app = FastAPI(lifespan=_lifespan)

    svc = MagicMock()
    svc._graph_registry = registry or {}
    mock_graph = MagicMock()
    mock_graph.get_input_jsonschema.return_value = {"type": "object"}
    svc._get_base_graph = AsyncMock(return_value=mock_graph)

    await register_mcp_tools(svc)
    mount_mcp(app)
    return app


def _parse_sse_json(text: str) -> dict[str, Any]:
    """Extract the first JSON-RPC payload from an SSE response body."""
    for line in text.strip().split("\n"):
        if line.startswith("data: "):
            return json.loads(line[6:])
    raise ValueError(f"No SSE data line found in response: {text!r}")


@pytest.fixture(autouse=True)
def _reset_module_state() -> None:
    """Reset module-level state between tests."""
    from aegra_api.adapters import mcp_adapter

    mcp_adapter._final_response_only = None
    mcp_adapter._mcp_app_lifespan = None
    mcp_adapter.mcp_server._local_provider._components.clear()


def test_unmounted_mcp_returns_404() -> None:
    """When MCP is not mounted (e.g. disable_mcp=true), /mcp returns 404."""
    app = FastAPI()
    client = TestClient(app)
    resp = client.post("/mcp", json={})
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_mcp_endpoint_responds_to_initialize() -> None:
    """POST /mcp with an MCP initialize request returns a valid JSON-RPC response."""
    app = await _make_mcp_app()

    with TestClient(app, raise_server_exceptions=False) as client:
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "0.0.1"},
            },
        }
        resp = client.post(
            "/mcp",
            json=payload,
            headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
        )

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    data = _parse_sse_json(resp.text)
    assert data.get("jsonrpc") == "2.0"
    assert data.get("id") == 1
    assert "result" in data


@pytest.mark.asyncio
async def test_mcp_endpoint_tools_list() -> None:
    """POST /mcp tools/list returns one tool per registered graph."""
    registry: dict[str, Any] = {
        "my_agent": {"file_path": "agent.py", "export_name": "graph"},
    }
    app = await _make_mcp_app(registry)

    with TestClient(app, raise_server_exceptions=False) as client:
        init_payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "0.0.1"},
            },
        }
        init_resp = client.post(
            "/mcp",
            json=init_payload,
            headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
        )
        assert init_resp.status_code == 200

        tools_payload = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        }
        tools_resp = client.post(
            "/mcp",
            json=tools_payload,
            headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
        )

    assert tools_resp.status_code == 200
    tools_data = _parse_sse_json(tools_resp.text)
    assert "result" in tools_data
    tools = tools_data["result"].get("tools", [])
    assert len(tools) == 1
    assert tools[0]["name"] == "my_agent"


@pytest.mark.asyncio
async def test_mcp_auth_rejects_unauthenticated() -> None:
    """MCP endpoint returns 401 when auth backend rejects."""
    from starlette.authentication import AuthenticationError

    app = await _make_mcp_app({"agent": {}})

    mock_backend = MagicMock()
    mock_backend.authenticate = AsyncMock(side_effect=AuthenticationError("Invalid token"))

    with patch("aegra_api.adapters.mcp_adapter.get_auth_backend", return_value=mock_backend):
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            headers={"Accept": "application/json, text/event-stream"},
        )

    assert resp.status_code == 401


def test_mcp_final_response_only_loads_from_config(tmp_path: Any, monkeypatch: Any) -> None:
    """_is_final_response_only reads the mcp section from aegra.json."""
    from aegra_api.adapters import mcp_adapter
    from aegra_api.adapters.mcp_adapter import _is_final_response_only

    monkeypatch.chdir(tmp_path)
    mcp_adapter._final_response_only = None

    config_file = tmp_path / "aegra.json"
    config_file.write_text(
        json.dumps(
            {
                "graphs": {"test": "./test.py:graph"},
                "mcp": {"final_response_only": True},
            }
        )
    )

    assert _is_final_response_only() is True

    mcp_adapter._final_response_only = None
    config_file.write_text(
        json.dumps(
            {
                "graphs": {"test": "./test.py:graph"},
            }
        )
    )

    assert _is_final_response_only() is False


# ============================================================================
# Auth provider integration tests
# ============================================================================


def test_mount_mcp_with_auth_provider() -> None:
    """When mcp.auth.path is configured, mount_mcp loads the provider and sets mcp_server.auth."""
    from aegra_api.adapters import mcp_adapter

    mcp_config: dict[str, Any] = {"final_response_only": False, "auth": {"path": "./mcp_auth.py:provider"}}

    app = FastAPI()
    mock_provider = MagicMock()
    mock_provider.get_well_known_routes.return_value = []
    mock_starlette_app = MagicMock()
    mock_starlette_app.lifespan = None

    with (
        patch("aegra_api.adapters.mcp_adapter.load_mcp_config", return_value=mcp_config),
        patch("aegra_api.adapters.mcp_adapter._load_mcp_auth_provider", return_value=mock_provider) as mock_load,
        patch.object(mcp_adapter.mcp_server, "http_app", return_value=mock_starlette_app),
    ):
        mount_mcp(app)

    mock_load.assert_called_once_with("./mcp_auth.py:provider")
    assert mcp_adapter.mcp_server.auth is mock_provider


def test_mount_mcp_without_auth_config() -> None:
    """When no mcp.auth config, mount_mcp uses _AuthMiddleware only (auth stays None)."""
    from aegra_api.adapters import mcp_adapter

    mcp_adapter.mcp_server.auth = None

    app = FastAPI()

    with patch("aegra_api.adapters.mcp_adapter.load_mcp_config", return_value=None):
        mount_mcp(app)

    assert mcp_adapter.mcp_server.auth is None


def test_mcp_auth_config_loads_from_aegra_json(tmp_path: Any, monkeypatch: Any) -> None:
    """load_mcp_config correctly parses mcp.auth from aegra.json."""
    from aegra_api.config import load_mcp_config

    monkeypatch.chdir(tmp_path)

    config_file = tmp_path / "aegra.json"
    config_file.write_text(
        json.dumps(
            {
                "graphs": {"test": "./test.py:graph"},
                "mcp": {
                    "final_response_only": False,
                    "auth": {
                        "path": "./mcp_auth.py:provider",
                    },
                },
            }
        )
    )

    mcp_cfg = load_mcp_config()
    assert mcp_cfg is not None
    assert mcp_cfg.get("final_response_only") is False

    auth = mcp_cfg.get("auth")
    assert auth is not None
    assert auth["path"] == "./mcp_auth.py:provider"

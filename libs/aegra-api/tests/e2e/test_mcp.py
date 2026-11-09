"""E2E tests for the MCP adapter endpoint.

Requires a running server with MCP enabled (default).
"""

import httpx
import pytest

from aegra_api.settings import settings

MCP_HEADERS: dict[str, str] = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
}


def _server_url() -> str:
    return settings.app.SERVER_URL or f"http://localhost:{settings.app.PORT}"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_mcp_initialize_returns_server_info() -> None:
    """POST /mcp with initialize returns a valid JSON-RPC result with serverInfo."""
    async with httpx.AsyncClient(base_url=_server_url(), timeout=10.0) as client:
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "e2e-test", "version": "0.0.1"},
            },
        }
        resp = await client.post("/mcp", json=payload, headers=MCP_HEADERS)

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    data = resp.json()
    assert data.get("jsonrpc") == "2.0"
    assert data.get("id") == 1
    assert "result" in data, f"Missing 'result' key: {data}"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_mcp_tools_list_returns_registered_tools() -> None:
    """POST /mcp with tools/list returns at least one registered tool."""
    async with httpx.AsyncClient(base_url=_server_url(), timeout=10.0) as client:
        init_resp = await client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "e2e-test", "version": "0.0.1"},
                },
            },
            headers=MCP_HEADERS,
        )
        assert init_resp.status_code == 200

        tools_resp = await client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            headers=MCP_HEADERS,
        )

    assert tools_resp.status_code == 200, f"Expected 200, got {tools_resp.status_code}: {tools_resp.text}"
    data = tools_resp.json()
    assert "result" in data
    tools = data["result"].get("tools", [])
    assert len(tools) >= 1, f"Expected at least 1 tool, got {len(tools)}"
    assert all("name" in t for t in tools), "All tools must have a 'name' field"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_mcp_endpoint_rejects_invalid_jsonrpc() -> None:
    """POST /mcp with a malformed JSON-RPC request returns an error response."""
    async with httpx.AsyncClient(base_url=_server_url(), timeout=10.0) as client:
        resp = await client.post("/mcp", json={"invalid": "payload"}, headers=MCP_HEADERS)

    assert resp.status_code in (200, 400), f"Unexpected status: {resp.status_code}"
    data = resp.json()
    assert "error" in data, "Expected JSON-RPC error for invalid request"

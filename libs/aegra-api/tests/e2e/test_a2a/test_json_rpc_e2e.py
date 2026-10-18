"""E2E: the A2A JSON-RPC envelope against a running server."""

import pytest
from httpx import AsyncClient

from aegra_api.settings import settings
from tests.e2e._utils import elog


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_malformed_json_returns_parse_error_with_http_200() -> None:
    async with AsyncClient(base_url=settings.app.SERVER_URL, timeout=30.0) as http_client:
        resp = await http_client.post("/a2a/agent", content=b'{"jsonrpc": ')
    elog("malformed json", {"status": resp.status_code, "body": resp.text[:300]})

    assert resp.status_code == 200
    assert resp.json() == {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Invalid JSON payload"}}


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_unknown_method_returns_method_not_found_echoing_the_id() -> None:
    async with AsyncClient(base_url=settings.app.SERVER_URL, timeout=30.0) as http_client:
        resp = await http_client.post("/a2a/agent", json={"jsonrpc": "2.0", "id": 99, "method": "unknown/method"})
    elog("unknown method", {"status": resp.status_code, "body": resp.text[:300]})

    assert resp.status_code == 200
    assert resp.json() == {
        "jsonrpc": "2.0",
        "id": 99,
        "error": {"code": -32601, "message": "Method not found: unknown/method"},
    }

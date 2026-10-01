"""E2E: oversized client thread_id is 422, not a Postgres 500."""

import secrets

import pytest
from httpx import AsyncClient

from aegra_api.settings import settings
from tests.e2e._utils import elog


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_create_thread_rejects_oversized_id_with_422() -> None:
    thread_id = secrets.token_hex(2500)
    async with AsyncClient(base_url=settings.app.SERVER_URL, timeout=30.0) as http_client:
        resp = await http_client.post("/threads", json={"thread_id": thread_id})
    elog("oversized thread_id", {"status": resp.status_code, "body": resp.text[:500]})
    assert resp.status_code == 422
    assert "thread_id" in resp.text

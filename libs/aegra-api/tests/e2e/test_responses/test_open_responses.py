"""E2E coverage for the optional Open Responses surface.

Requires a running server whose config sets ``http.responses`` to true
(``aegra.responses.json`` in CI and ``make e2e-*``) and deploys the ``agent`` graph.
"""

import pytest
from httpx import AsyncClient

from aegra_api.settings import settings
from tests.e2e._utils import check_and_skip_if_geo_blocked, elog


def _output_text(body: dict) -> str:
    parts: list[str] = []
    for item in body.get("output") or []:
        if item.get("type") != "message":
            continue
        for block in item.get("content") or []:
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_responses_conversation_round_trip() -> None:
    """A conversation accepts a Responses turn and lists the messages back with roles."""
    async with AsyncClient(base_url=settings.app.SERVER_URL, timeout=120.0) as http:
        created = await http.post("/v1/conversations", json={"metadata": {"source": "e2e"}})
        elog("POST /v1/conversations", {"status": created.status_code, "body": created.text})
        assert created.status_code == 200, created.text
        conversation_id = created.json()["id"]

        response = await http.post(
            "/v1/responses",
            json={
                "model": "agent",
                "input": "Say hello in one word.",
                "conversation": conversation_id,
                "store": True,
            },
        )
        elog("POST /v1/responses", {"status": response.status_code, "body": response.text[:2000]})
        assert response.status_code == 200, response.text
        body = response.json()
        if body.get("status") == "failed":
            check_and_skip_if_geo_blocked(
                {"status": "error", "error_message": (body.get("error") or {}).get("message", "")}
            )
        assert body["status"] == "completed", body
        assert body["conversation"] == {"id": conversation_id}
        assert _output_text(body)

        items = await http.get(f"/v1/conversations/{conversation_id}/items")
        assert items.status_code == 200, items.text
        roles = [item.get("role") for item in items.json()["data"]]
        assert "user" in roles
        assert "assistant" in roles

        await http.delete(f"/v1/conversations/{conversation_id}")

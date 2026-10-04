"""E2E: the A2A agent card, fetched over HTTP and through the a2a-sdk client."""

import pytest
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.compat.v0_3.versions import is_legacy_version
from httpx import AsyncClient

from aegra_api.settings import settings
from tests.e2e._utils import elog


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_agent_card_is_served_for_a_graph_id() -> None:
    async with AsyncClient(base_url=settings.app.SERVER_URL, timeout=30.0) as http_client:
        resp = await http_client.get("/a2a/agent/.well-known/agent-card.json")
    elog("agent card", {"status": resp.status_code, "body": resp.text[:800]})

    assert resp.status_code == 200
    card = resp.json()
    assert card["protocolVersion"] == "1.0"
    assert card["url"].endswith("/a2a/agent")
    assert len(card["skills"]) == 1


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_agent_card_returns_404_for_unknown_assistant() -> None:
    async with AsyncClient(base_url=settings.app.SERVER_URL, timeout=30.0) as http_client:
        resp = await http_client.get("/a2a/no-such-assistant")
    elog("unknown assistant card", {"status": resp.status_code, "body": resp.text[:300]})

    assert resp.status_code == 404


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_a2a_sdk_reads_the_card_as_protocol_1_0() -> None:
    """The real client must pick the modern dialect from our card, not the legacy 0.3 one."""
    async with AsyncClient(timeout=30.0) as http_client:
        resolver = A2ACardResolver(http_client, f"{settings.app.SERVER_URL}/a2a/agent")
        card = await resolver.get_agent_card()
        # Raises "no compatible transports found" when the card has no usable interface.
        ClientFactory(ClientConfig(httpx_client=http_client)).create(card)

    interface = card.supported_interfaces[0]
    elog("card via a2a-sdk", {"name": card.name, "version": interface.protocol_version, "url": interface.url})
    assert interface.protocol_binding == "JSONRPC"
    assert interface.protocol_version == "1.0"
    assert is_legacy_version(interface.protocol_version) is False
    assert interface.url.endswith("/a2a/agent")

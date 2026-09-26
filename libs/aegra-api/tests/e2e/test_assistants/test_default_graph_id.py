"""The default graph_id rule, against a real server and its real aegra.json.

This repo's own `aegra.json` registers several graphs and nominates no
`default_graph_id`, so the deployment these tests run against is the "no default
resolves" case: `graph_id` stays required, and omitting it is a 422 that names
the graphs the caller could have meant.

The positive halves of the rule — a configured `default_graph_id`, and the
single-graph deployment that needs no key — cannot be exercised here without
changing the example deployment's own configuration, which would change what
every other E2E test runs against. They are covered at the unit level, where the
resolver is pointed at a temporary config instead.
"""

import uuid
from collections.abc import AsyncIterator

import httpx
import pytest

from aegra_api.settings import settings
from tests.e2e._utils import elog


@pytest.fixture
async def http_client() -> AsyncIterator[httpx.AsyncClient]:
    """Raw HTTP, because the SDK's create() cannot omit graph_id."""
    async with httpx.AsyncClient(base_url=settings.app.SERVER_URL, timeout=30.0) as client:
        yield client


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_create_without_graph_id_is_rejected_naming_the_graphs(http_client: httpx.AsyncClient) -> None:
    """With several graphs and no default, an omitted graph_id is a 422 that helps."""
    response = await http_client.post("/assistants", json={"name": "No graph id"})

    assert response.status_code == 422, response.text

    message = response.json()["message"]
    elog("Create without graph_id rejected", {"status": response.status_code, "message": message})

    assert "graph_id is required" in message
    assert "default_graph_id" in message, "the error should say how to configure a default"
    # Naming the real registered graphs is the part that makes this actionable,
    # and the part a mocked service cannot prove.
    assert "agent" in message


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_explicit_graph_id_still_creates(http_client: httpx.AsyncClient) -> None:
    """The deployment still creates normally when the caller names a graph.

    The config is unique per run because a create dedupes on
    ``(graph_id, config)`` for the calling user: with the default empty config
    this would match an assistant another test left behind, hand back its id,
    and have the cleanup below delete something this test never created.
    """
    config = {"tags": [f"default-graph-id-e2e-{uuid.uuid4()}"]}

    response = await http_client.post(
        "/assistants",
        json={"name": "Explicit graph id", "graph_id": "agent", "config": config},
    )

    assert response.status_code == 200, response.text
    created = response.json()

    # Everything that can fail goes inside, so a failed assertion still cleans
    # up the assistant it created.
    try:
        assert created["graph_id"] == "agent"
        assert created["config"] == config, "a pre-existing assistant was returned instead of a new one"
        elog("Create with explicit graph_id succeeded", {"assistant_id": created["assistant_id"]})
    finally:
        await http_client.delete(f"/assistants/{created['assistant_id']}")

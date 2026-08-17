"""Concurrent assistant creation must not surface unique violations as 500s.

``if_exists`` used to be implemented as a SELECT followed by an unconditional
INSERT. Two requests that miss the SELECT both INSERT, and the loser's
``UniqueViolationError`` escapes as a 500 — on ``assistant_pkey``,
``idx_assistant_user_assistant`` or ``idx_assistant_user_graph_config``. These
tests drive the real server against a real database so the arbitration is
exercised where it actually happens — in Postgres — rather than against a mock.
"""

import asyncio
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from aegra_api.settings import settings
from tests.e2e._utils import elog

# Enough concurrency to interleave several requests inside one another's
# create window, over enough rounds that a lost race is not merely unlucky.
CONCURRENCY = 30
ROUNDS = 6


async def _burst(client: httpx.AsyncClient, payloads: list[dict[str, Any]]) -> list[httpx.Response]:
    return await asyncio.gather(*(client.post("/assistants", json=p) for p in payloads))


def _unique_config() -> dict[str, Any]:
    """A config no other round shares, so only this round's requests can collide."""
    return {"tags": [f"concurrent-create-{uuid.uuid4()}"]}


async def _delete_all(client: httpx.AsyncClient, responses: list[httpx.Response]) -> None:
    created = {r.json()["assistant_id"] for r in responses if r.status_code == 200}
    await asyncio.gather(*(client.delete(f"/assistants/{a}") for a in created))


@pytest.fixture
async def warm_client() -> AsyncIterator[httpx.AsyncClient]:
    """A client whose connections — and the server's DB pool — are already up.

    Without this the burst serialises behind TCP and connection setup and never
    overlaps server-side, which would let the old racy implementation pass.
    """
    async with httpx.AsyncClient(
        base_url=settings.app.SERVER_URL,
        timeout=60.0,
        limits=httpx.Limits(
            max_connections=CONCURRENCY + 5,
            max_keepalive_connections=CONCURRENCY + 5,
        ),
    ) as client:
        await asyncio.gather(*(client.get("/assistants") for _ in range(CONCURRENCY)))
        yield client


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_concurrent_assistant_create_by_id_never_500s(warm_client: httpx.AsyncClient) -> None:
    """Concurrent do_nothing creates of one assistant_id all resolve to one row."""
    for round_number in range(ROUNDS):
        assistant_id = f"concurrent-create-{uuid.uuid4()}"
        # Distinct markers per request. With identical payloads, a response
        # assembled from the caller's own input would pass for the stored row.
        markers = [f"round-{round_number}-request-{i}" for i in range(CONCURRENCY)]
        config = _unique_config()
        payloads = [
            {
                "assistant_id": assistant_id,
                "graph_id": "agent",
                "config": config,
                "if_exists": "do_nothing",
                "metadata": {"request_marker": marker},
            }
            for marker in markers
        ]

        responses = await _burst(warm_client, payloads)

        statuses = sorted({r.status_code for r in responses})
        failed = [r for r in responses if r.status_code != 200]
        if failed:
            elog(
                "Concurrent assistant create failed",
                {
                    "round": round_number + 1,
                    "assistant_id": assistant_id,
                    "statuses": statuses,
                    "sample_body": failed[0].text[:500],
                },
            )
        assert statuses == [200], f"round {round_number + 1} returned {statuses}"
        assert {r.json()["assistant_id"] for r in responses} == {assistant_id}

        # do_nothing owes every caller the one stored assistant, so all of them must
        # carry the winner's marker rather than the one they each submitted.
        returned = {r.json()["metadata"]["request_marker"] for r in responses}
        assert len(returned) == 1, f"round {round_number + 1} returned {len(returned)} rows: {returned}"
        assert returned <= set(markers)

        await _delete_all(warm_client, responses)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_concurrent_assistant_create_by_graph_and_config_never_500s(
    warm_client: httpx.AsyncClient,
) -> None:
    """Server-generated IDs collide on idx_assistant_user_graph_config instead.

    Nothing here shares an assistant_id, so this exercises the unique index the
    primary key cannot cover.
    """
    for round_number in range(ROUNDS):
        config = _unique_config()
        markers = [f"round-{round_number}-request-{i}" for i in range(CONCURRENCY)]
        payloads = [
            {
                "graph_id": "agent",
                "config": config,
                "if_exists": "do_nothing",
                "metadata": {"request_marker": marker},
            }
            for marker in markers
        ]

        responses = await _burst(warm_client, payloads)

        statuses = sorted({r.status_code for r in responses})
        failed = [r for r in responses if r.status_code != 200]
        if failed:
            elog(
                "Concurrent assistant create on graph/config failed",
                {"round": round_number + 1, "statuses": statuses, "sample_body": failed[0].text[:500]},
            )
        assert statuses == [200], f"round {round_number + 1} returned {statuses}"

        # One graph/config pair per user is one assistant, whichever request won.
        assistant_ids = {r.json()["assistant_id"] for r in responses}
        assert len(assistant_ids) == 1, f"round {round_number + 1} created {len(assistant_ids)} assistants"

        returned = {r.json()["metadata"]["request_marker"] for r in responses}
        assert returned <= set(markers)

        await _delete_all(warm_client, responses)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_concurrent_assistant_create_raise_yields_conflict_not_error(
    warm_client: httpx.AsyncClient,
) -> None:
    """With the default if_exists, losers get 409 — never 500."""
    assistant_id = f"concurrent-conflict-{uuid.uuid4()}"
    payload = {"assistant_id": assistant_id, "graph_id": "agent", "config": _unique_config()}

    responses = await _burst(warm_client, [payload] * CONCURRENCY)

    statuses = [r.status_code for r in responses]
    elog("Concurrent assistant create with default if_exists=error", {"statuses": sorted(set(statuses))})
    assert set(statuses) <= {200, 409}, f"unexpected statuses: {sorted(set(statuses))}"
    assert statuses.count(200) == 1, f"expected exactly one winner, got {statuses.count(200)}"

    await _delete_all(warm_client, responses)

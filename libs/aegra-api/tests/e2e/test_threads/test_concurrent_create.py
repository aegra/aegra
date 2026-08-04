"""Concurrent creation must not surface unique violations as 500s.

``if_exists`` used to be implemented as a SELECT followed by an unconditional
INSERT. Two requests for the same ID both miss the SELECT, both INSERT, and the
loser's ``UniqueViolationError`` escapes as a 500. These tests drive the real
server against a real database so the arbitration is exercised where it actually
happens — in Postgres — rather than against a mocked session.
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


async def _burst(client: httpx.AsyncClient, path: str, payloads: list[dict[str, Any]]) -> list[httpx.Response]:
    return await asyncio.gather(*(client.post(path, json=p) for p in payloads))


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
        await asyncio.gather(*(client.get("/threads") for _ in range(CONCURRENCY)))
        yield client


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_concurrent_thread_create_never_500s(warm_client: httpx.AsyncClient) -> None:
    """Concurrent do_nothing creates all resolve to one canonical thread."""
    for round_number in range(ROUNDS):
        thread_id = f"concurrent-create-{uuid.uuid4()}"
        # Distinct markers per request. With identical payloads, a response
        # assembled from the caller's own input would pass for the stored row.
        markers = [f"round-{round_number}-request-{i}" for i in range(CONCURRENCY)]
        payloads = [
            {
                "thread_id": thread_id,
                "if_exists": "do_nothing",
                "metadata": {"request_marker": marker},
            }
            for marker in markers
        ]

        responses = await _burst(warm_client, "/threads", payloads)

        statuses = sorted({r.status_code for r in responses})
        failed = [r for r in responses if r.status_code != 200]
        if failed:
            elog(
                "Concurrent create failed",
                {
                    "round": round_number + 1,
                    "thread_id": thread_id,
                    "statuses": statuses,
                    "sample_body": failed[0].text[:500],
                },
            )
        assert statuses == [200], f"round {round_number + 1} returned {statuses}"
        assert {r.json()["thread_id"] for r in responses} == {thread_id}

        # do_nothing owes every caller the one stored thread, so all of them must
        # carry the winner's marker rather than the one they each submitted.
        returned = {r.json()["metadata"]["request_marker"] for r in responses}
        assert len(returned) == 1, f"round {round_number + 1} returned {len(returned)} rows: {returned}"
        assert returned <= set(markers)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_concurrent_thread_create_raise_yields_conflict_not_error(
    warm_client: httpx.AsyncClient,
) -> None:
    """With the default if_exists, losers get 409 — never 500."""
    thread_id = f"concurrent-conflict-{uuid.uuid4()}"

    responses = await _burst(warm_client, "/threads", [{"thread_id": thread_id}] * CONCURRENCY)

    statuses = [r.status_code for r in responses]
    elog("Concurrent create with if_exists=raise", {"statuses": sorted(set(statuses))})
    assert set(statuses) <= {200, 409}, f"unexpected statuses: {sorted(set(statuses))}"
    assert statuses.count(200) == 1, f"expected exactly one winner, got {statuses.count(200)}"

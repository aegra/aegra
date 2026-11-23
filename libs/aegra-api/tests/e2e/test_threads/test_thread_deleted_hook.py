"""E2E: the thread_deleted hook runs once for each thread a delete path removes.

The server's hook is ``tests/e2e/thread_hook_recorder.py``, which appends every event to the
file ``AEGRA_E2E_THREAD_HOOK_LOG`` names. The E2E workflow configures both; anywhere else the
variable is unset and these tests skip.
"""

import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Any

import httpx
import pytest

from aegra_api.settings import settings
from tests.e2e._utils import elog, get_e2e_client
from tests.e2e.thread_hook_recorder import LOG_ENV


@pytest.fixture
def hook_log() -> Path:
    path = os.environ.get(LOG_ENV)
    if not path:
        pytest.skip(f"{LOG_ENV} is unset; the E2E workflow configures a recording thread_deleted hook")
    return Path(path)


def _server_url() -> str:
    return settings.app.SERVER_URL


def _events(log: Path, thread_id: str) -> list[dict[str, Any]]:
    if not log.exists():
        return []
    lines = log.read_text(encoding="utf-8").splitlines()
    return [event for event in (json.loads(line) for line in lines if line.strip()) if event["thread_id"] == thread_id]


async def _only_event(log: Path, thread_id: str, *, timeout: float = 30.0) -> dict[str, Any]:
    """The one event recorded for *thread_id*; a second call is given time to show up."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not _events(log, thread_id):
        if loop.time() >= deadline:
            raise AssertionError(f"No thread_deleted event for {thread_id} within {timeout}s")
        await asyncio.sleep(0.1)
    await asyncio.sleep(1.0)
    events = _events(log, thread_id)
    assert len(events) == 1, events
    elog("thread_deleted event", events[0])
    return events[0]


def _stress_input() -> dict[str, Any]:
    return {"messages": [{"role": "user", "content": json.dumps({"delay": 0})}]}


def _assert_owner(event: dict[str, Any]) -> None:
    assert isinstance(event["user_id"], str)
    assert event["user_id"]


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_delete_route_reports_the_thread_with_owner_and_metadata(hook_log: Path) -> None:
    client = get_e2e_client()
    marker = uuid.uuid4().hex
    thread = await client.threads.create(metadata={"hook_case": marker})

    await client.threads.delete(thread["thread_id"])

    event = await _only_event(hook_log, thread["thread_id"])
    _assert_owner(event)
    assert event["metadata"]["hook_case"] == marker


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_delete_that_answers_404_reports_nothing(hook_log: Path) -> None:
    client = get_e2e_client()
    missing = str(uuid.uuid4())
    with pytest.raises(httpx.HTTPStatusError) as raised:
        await client.threads.delete(missing)
    assert raised.value.response.status_code == 404

    # A delete made afterwards reaching the hook shows the hook ran in the meantime.
    control = await client.threads.create()
    await client.threads.delete(control["thread_id"])
    await _only_event(hook_log, control["thread_id"])

    assert _events(hook_log, missing) == []


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_prune_by_id_reports_each_deleted_thread(hook_log: Path) -> None:
    client = get_e2e_client()
    marker = uuid.uuid4().hex
    threads = [await client.threads.create(metadata={"hook_case": marker}) for _ in range(2)]

    async with httpx.AsyncClient(base_url=_server_url(), timeout=30) as http:
        response = await http.post(
            "/threads/prune",
            json={"thread_ids": [thread["thread_id"] for thread in threads], "strategy": "delete"},
        )
    response.raise_for_status()
    assert response.json()["deleted"] == 2

    for thread in threads:
        event = await _only_event(hook_log, thread["thread_id"])
        _assert_owner(event)
        assert event["metadata"]["hook_case"] == marker


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_expired_thread_with_delete_strategy_is_reported(hook_log: Path) -> None:
    marker = uuid.uuid4().hex
    async with httpx.AsyncClient(base_url=_server_url(), timeout=30) as http:
        created = await http.post(
            "/threads",
            json={"metadata": {"hook_case": marker}, "ttl": {"strategy": "delete", "ttl": 0.01}},
        )
        created.raise_for_status()
        thread_id = created.json()["thread_id"]
        await asyncio.sleep(1.5)
        # Without a body, prune removes the caller's expired threads; the TTL sweep may have got there first.
        (await http.post("/threads/prune")).raise_for_status()

    event = await _only_event(hook_log, thread_id)
    _assert_owner(event)
    assert event["metadata"]["hook_case"] == marker


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_stateless_wait_run_reports_its_thread(hook_log: Path) -> None:
    async with httpx.AsyncClient(base_url=_server_url(), timeout=60) as http:
        response = await http.post("/runs/wait", json={"assistant_id": "stress_test", "input": _stress_input()})
    response.raise_for_status()
    # Content-Location: /threads/{thread_id}/runs/{run_id}
    location = response.headers["content-location"]
    thread_id = location.split("/threads/", 1)[1].split("/", 1)[0]

    _assert_owner(await _only_event(hook_log, thread_id))


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_stateless_background_run_reports_its_thread(hook_log: Path) -> None:
    client = get_e2e_client()
    run = await client.runs.create(None, "stress_test", input=_stress_input())

    _assert_owner(await _only_event(hook_log, run["thread_id"]))

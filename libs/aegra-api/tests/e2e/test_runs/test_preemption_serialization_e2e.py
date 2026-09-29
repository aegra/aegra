"""Pre-emption must retain thread occupancy until cancelled execution stops.

Uses the stress_test graph from the repository's aegra.json. No LLMs.
The reaper case requires prod mode with REAPER_INTERVAL_SECONDS <= 15.
"""

import asyncio
import json
from collections.abc import AsyncIterator
from functools import partial
from typing import Any

import httpx
import pytest

from aegra_api.settings import settings
from tests.e2e._utils import elog


async def _marker(client: httpx.AsyncClient, namespace: list[str], key: str) -> dict[str, Any] | None:
    response = await client.get("/store/items", params={"namespace": ".".join(namespace), "key": key})
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()["value"]


async def _wait_marker(
    client: httpx.AsyncClient, namespace: list[str], key: str, *, timeout: float = 10
) -> dict[str, Any]:
    async with asyncio.timeout(timeout):
        while (value := await _marker(client, namespace, key)) is None:
            await asyncio.sleep(0.05)
    return value


async def _release(client: httpx.AsyncClient, namespace: list[str]) -> None:
    response = await client.put(
        "/store/items", json={"namespace": namespace, "key": "release", "value": {"open": True}}
    )
    response.raise_for_status()


@pytest.fixture
async def cancellation_probe_thread() -> AsyncIterator[tuple[httpx.AsyncClient, str, str, list[str]]]:
    server_url = settings.app.SERVER_URL
    assert server_url is not None
    async with httpx.AsyncClient(base_url=server_url, timeout=30) as client:
        response = await client.post("/assistants", json={"graph_id": "stress_test", "if_exists": "do_nothing"})
        response.raise_for_status()
        assistant_id = response.json()["assistant_id"]
        response = await client.post("/threads", json={})
        response.raise_for_status()
        thread_id = response.json()["thread_id"]
        namespace = ["preemption-test", thread_id]
        try:
            yield client, assistant_id, thread_id, namespace
        finally:
            await _release(client, namespace)
            # Database run status is already terminal during the bug: wait for actual node exit.
            if await _marker(client, namespace, "A.started") is not None:
                await _wait_marker(client, namespace, "A.exited")
            response = await client.delete(f"/threads/{thread_id}")
            response.raise_for_status()
            for key in (
                "control",
                "release",
                *(f"{label}.{event}" for label in "ABC" for event in ("started", "cancelling", "exited")),
            ):
                response = await client.request("DELETE", "/store/items", json={"namespace": namespace, "key": key})
                response.raise_for_status()


async def _create_run(
    client: httpx.AsyncClient,
    *,
    assistant_id: str,
    thread_id: str,
    namespace: list[str],
    label: str,
    strategy: str,
) -> None:
    # Ask the store API for the resolved tenant namespace rather than hardcoding an identity.
    response = await client.put("/store/items", json={"namespace": namespace, "key": "control", "value": {}})
    response.raise_for_status()
    response = await client.get("/store/items", params={"namespace": ".".join(namespace), "key": "control"})
    response.raise_for_status()
    scoped_namespace = response.json()["namespace"]
    response = await client.post(
        f"/threads/{thread_id}/runs",
        json={
            "assistant_id": assistant_id,
            "multitask_strategy": strategy,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "delay": 60 if label == "A" else 0,
                                "steps": 1,
                                "_cancellation_probe": {
                                    "namespace": scoped_namespace,
                                    "label": label,
                                    "hold_cancellation": label == "A",
                                },
                            }
                        ),
                    }
                ]
            },
        },
    )
    response.raise_for_status()


@pytest.mark.e2e
@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ["interrupt", "rollback"])
@pytest.mark.parametrize("trigger", ["third_request", pytest.param("reaper", marks=pytest.mark.prod_only)])
async def test_preemption_does_not_overlap_execution_e2e(
    cancellation_probe_thread: tuple[httpx.AsyncClient, str, str, list[str]], strategy: str, trigger: str
) -> None:
    client, assistant_id, thread_id, namespace = cancellation_probe_thread
    create_run = partial(_create_run, client, assistant_id=assistant_id, thread_id=thread_id, namespace=namespace)
    await create_run(label="A", strategy="enqueue")
    await _wait_marker(client, namespace, "A.started")
    await create_run(label="B", strategy=strategy)
    await _wait_marker(client, namespace, "A.cancelling")
    assert await _marker(client, namespace, "A.exited") is None, "Probe did not hold A in cancellation cleanup"

    successor = "B"
    if trigger == "third_request":
        successor = "C"
        await create_run(label=successor, strategy=strategy)

    # Give admission (or one default 15-second reaper sweep) a chance to dispatch illegally.
    observation_seconds = 3 if trigger == "third_request" else 20
    deadline = asyncio.get_running_loop().time() + observation_seconds
    while asyncio.get_running_loop().time() < deadline:
        if await _marker(client, namespace, f"{successor}.started") is not None:
            break
        await asyncio.sleep(0.05)
    assert await _marker(client, namespace, "A.exited") is None, "Probe released A before the test opened the gate"
    await _release(client, namespace)
    started = await _wait_marker(client, namespace, f"{successor}.started")
    await _wait_marker(client, namespace, f"{successor}.exited")

    elog("Pre-emption serialization", {"strategy": strategy, "trigger": trigger, "successor": successor, **started})
    assert started["predecessor_exited"] is True, (
        f"Run {successor} entered its graph before cancelled run A exited ({strategy}, {trigger})"
    )

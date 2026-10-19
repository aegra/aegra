"""E2E: runs and crons with ``webhook`` POST the finished run to a receiver.

The receiver is the custom route in examples/custom_routes_example.py. The
repo's aegra.json allows loopback webhooks, so the server can call itself at
SERVER_URL in both the docker-compose and the CI setups.
"""

import asyncio
import json
from datetime import datetime
from typing import Any

import httpx
import pytest

from aegra_api.settings import settings
from tests.e2e._utils import elog, get_e2e_client

_RECEIVER = f"{settings.app.SERVER_URL}/custom/webhooks/runs"


def _stress_input(**options: Any) -> dict[str, Any]:
    return {"messages": [{"role": "user", "content": json.dumps({"delay": 0, **options})}]}


async def _received_webhook(run_id: str, *, timeout: float = 30.0) -> dict[str, Any]:
    """Poll the receiver until the run's webhook has arrived."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    async with httpx.AsyncClient(timeout=5.0) as http:
        while True:
            response = await http.get(f"{_RECEIVER}/{run_id}")
            if response.status_code == 200:
                return response.json()
            if loop.time() >= deadline:
                raise AssertionError(f"No webhook received for run {run_id} within {timeout}s")
            await asyncio.sleep(0.25)


async def _first_run_on(client: Any, thread_id: str, *, timeout: float = 30.0) -> dict[str, Any]:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        runs = await client.runs.list(thread_id)
        if runs:
            return runs[0]
        if loop.time() >= deadline:
            raise AssertionError(f"No run appeared on thread {thread_id} within {timeout}s")
        await asyncio.sleep(0.25)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_run_webhook_posts_the_finished_run() -> None:
    client = get_e2e_client()
    assistant = await client.assistants.create(graph_id="stress_test", if_exists="do_nothing")
    thread = await client.threads.create()
    webhook = f"{_RECEIVER}?token=e2e-secret"

    run = await client.runs.create(
        thread["thread_id"], assistant["assistant_id"], input=_stress_input(steps=2), webhook=webhook
    )
    payload = await _received_webhook(run["run_id"])
    elog("Run webhook payload", payload)

    assert payload["status"] == "success"
    assert payload["run_id"] == run["run_id"]
    assert payload["thread_id"] == thread["thread_id"]
    assert payload["assistant_id"] == assistant["assistant_id"]
    assert payload["values"]["step_count"] == 2
    assert payload["kwargs"]["webhook"] == webhook
    assert payload["kwargs"]["input"] == _stress_input(steps=2)
    timeline = [
        datetime.fromisoformat(payload[key])
        for key in ("created_at", "run_started_at", "run_ended_at", "webhook_sent_at")
    ]
    assert timeline == sorted(timeline)
    assert "error" not in payload


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_failed_run_webhook_reports_the_error() -> None:
    client = get_e2e_client()
    assistant = await client.assistants.create(graph_id="stress_test", if_exists="do_nothing")
    thread = await client.threads.create()

    run = await client.runs.create(
        thread["thread_id"], assistant["assistant_id"], input=_stress_input(steps=1, fail=True), webhook=_RECEIVER
    )
    payload = await _received_webhook(run["run_id"])
    elog("Failed run webhook payload", payload)

    assert payload["status"] == "error"
    assert payload["error"] == {"error": "RuntimeError", "message": "Intentional failure at step 1"}


@pytest.mark.e2e
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("webhook", "reason"),
    [
        ("http://169.254.169.254/latest/meta-data", "cloud metadata endpoint"),
        ("ftp://hooks.example.com/runs", "Webhook must use http or https"),
        ("/custom/webhooks/runs", "Relative webhook URLs are not supported"),
    ],
)
async def test_webhook_the_server_will_not_call_is_rejected(webhook: str, reason: str) -> None:
    client = get_e2e_client()
    assistant = await client.assistants.create(graph_id="stress_test", if_exists="do_nothing")
    thread = await client.threads.create()

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await client.runs.create(thread["thread_id"], assistant["assistant_id"], input=_stress_input(), webhook=webhook)

    assert exc_info.value.response.status_code == 422
    assert reason in exc_info.value.response.text
    assert await client.runs.list(thread["thread_id"]) == []


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_cron_fired_run_posts_its_webhook() -> None:
    client = get_e2e_client()
    assistant = await client.assistants.create(graph_id="stress_test", if_exists="do_nothing")
    thread = await client.threads.create()
    webhook = f"{_RECEIVER}?cron=e2e"

    # Creating an enabled cron fires its first run right away.
    await client.crons.create_for_thread(
        thread["thread_id"],
        assistant["assistant_id"],
        schedule="0 4 * * *",
        input=_stress_input(steps=1),
        webhook=webhook,
    )
    try:
        run = await _first_run_on(client, thread["thread_id"])
        payload = await _received_webhook(run["run_id"])
        elog("Cron run webhook payload", payload)

        assert payload["status"] == "success"
        assert payload["thread_id"] == thread["thread_id"]
        assert payload["kwargs"]["webhook"] == webhook
        assert payload["values"]["step_count"] == 1
    finally:
        for cron in await client.crons.search(thread_id=thread["thread_id"]):
            await client.crons.delete(cron["cron_id"])


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_cron_with_a_blocked_webhook_is_rejected() -> None:
    client = get_e2e_client()
    assistant = await client.assistants.create(graph_id="stress_test", if_exists="do_nothing")

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await client.crons.create(
            assistant["assistant_id"],
            schedule="0 4 * * *",
            input=_stress_input(),
            webhook="http://169.254.169.254/latest/meta-data",
        )

    assert exc_info.value.response.status_code == 422
    assert "cloud metadata endpoint" in exc_info.value.response.text

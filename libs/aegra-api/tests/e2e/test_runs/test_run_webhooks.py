"""E2E: runs and crons with ``webhook`` POST the finished run to a receiver.

The receiver is a small HTTP server inside the test process on 127.0.0.1. Loopback
targets are blocked by default, so delivery tests need a server configured like the
E2E workflow (``webhooks.url.disable_loopback: false``) and skip anywhere else.
"""

import asyncio
import json
import threading
from collections.abc import Iterator
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx
import pytest

from tests.e2e._utils import elog, get_e2e_client


class _Receiver:
    """Collects webhook POSTs per run_id on a background thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._deliveries: dict[str, list[tuple[dict[str, Any], dict[str, str]]]] = {}
        receiver = self

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                headers = {name.lower(): value for name, value in self.headers.items()}
                with receiver._lock:
                    receiver._deliveries.setdefault(str(payload.get("run_id")), []).append((payload, headers))
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, format: str, *args: Any) -> None:
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}/runs"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def deliveries(self, run_id: str) -> list[tuple[dict[str, Any], dict[str, str]]]:
        with self._lock:
            return list(self._deliveries.get(run_id, []))

    async def wait_for(self, run_id: str, *, timeout: float = 30.0) -> tuple[dict[str, Any], dict[str, str]]:
        """The first delivery for *run_id*."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while not (received := self.deliveries(run_id)):
            if loop.time() >= deadline:
                raise AssertionError(f"No webhook received for run {run_id} within {timeout}s")
            await asyncio.sleep(0.1)
        return received[0]


@pytest.fixture(scope="module")
def receiver() -> Iterator[_Receiver]:
    started = _Receiver()
    started.start()
    yield started
    started.stop()


def _stress_input(**options: Any) -> dict[str, Any]:
    return {"messages": [{"role": "user", "content": json.dumps({"delay": 0, **options})}]}


def _skip_if_loopback_blocked(exc: httpx.HTTPStatusError) -> None:
    if exc.response.status_code == 422 and "loopback" in exc.response.text:
        pytest.skip("The server blocks loopback webhooks; the E2E workflow config allows them")


async def _create_run(client: Any, thread_id: str, assistant_id: str, **kwargs: Any) -> dict[str, Any]:
    try:
        return await client.runs.create(thread_id, assistant_id, **kwargs)
    except httpx.HTTPStatusError as exc:
        _skip_if_loopback_blocked(exc)
        raise


async def _first_run_on(client: Any, thread_id: str, *, timeout: float = 30.0) -> dict[str, Any]:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not (runs := await client.runs.list(thread_id)):
        if loop.time() >= deadline:
            raise AssertionError(f"No run appeared on thread {thread_id} within {timeout}s")
        await asyncio.sleep(0.25)
    return runs[0]


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_run_webhook_posts_the_finished_run(receiver: _Receiver) -> None:
    client = get_e2e_client()
    assistant = await client.assistants.create(graph_id="stress_test", if_exists="do_nothing")
    thread = await client.threads.create()
    webhook = f"{receiver.url}?token=e2e-secret"

    run = await _create_run(
        client, thread["thread_id"], assistant["assistant_id"], input=_stress_input(steps=2), webhook=webhook
    )
    payload, headers = await receiver.wait_for(run["run_id"])
    elog("Run webhook payload", payload)

    assert headers["content-type"] == "application/json"
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
async def test_updates_only_run_still_reports_its_final_state(receiver: _Receiver) -> None:
    client = get_e2e_client()
    assistant = await client.assistants.create(graph_id="stress_test", if_exists="do_nothing")
    thread = await client.threads.create()

    run = await _create_run(
        client,
        thread["thread_id"],
        assistant["assistant_id"],
        input=_stress_input(steps=3),
        stream_mode="updates",
        webhook=receiver.url,
    )
    payload, _headers = await receiver.wait_for(run["run_id"])

    assert payload["status"] == "success"
    assert payload["values"]["step_count"] == 3


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_failed_run_webhook_reports_the_error(receiver: _Receiver) -> None:
    client = get_e2e_client()
    assistant = await client.assistants.create(graph_id="stress_test", if_exists="do_nothing")
    thread = await client.threads.create()

    run = await _create_run(
        client,
        thread["thread_id"],
        assistant["assistant_id"],
        input=_stress_input(steps=1, fail=True),
        webhook=receiver.url,
    )
    payload, _headers = await receiver.wait_for(run["run_id"])
    elog("Failed run webhook payload", payload)

    assert payload["status"] == "error"
    assert payload["error"] == {"error": "RuntimeError", "message": "Intentional failure at step 1"}


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_cancelled_run_reports_interrupted_once(receiver: _Receiver) -> None:
    client = get_e2e_client()
    assistant = await client.assistants.create(graph_id="stress_test", if_exists="do_nothing")
    thread = await client.threads.create()

    run = await _create_run(
        client,
        thread["thread_id"],
        assistant["assistant_id"],
        input=_stress_input(delay=2, steps=5),
        webhook=receiver.url,
    )
    await client.runs.cancel(thread["thread_id"], run["run_id"])
    payload, _headers = await receiver.wait_for(run["run_id"])
    # Executor and cancel endpoint can both see the run end; only the one that commits it reports.
    await asyncio.sleep(2)

    assert payload["status"] == "interrupted"
    assert len(receiver.deliveries(run["run_id"])) == 1


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
async def test_cron_fired_run_posts_its_webhook(receiver: _Receiver) -> None:
    client = get_e2e_client()
    assistant = await client.assistants.create(graph_id="stress_test", if_exists="do_nothing")
    thread = await client.threads.create()
    webhook = f"{receiver.url}?cron=e2e"

    # Creating an enabled cron fires its first run right away.
    try:
        await client.crons.create_for_thread(
            thread["thread_id"],
            assistant["assistant_id"],
            schedule="0 4 * * *",
            input=_stress_input(steps=1),
            webhook=webhook,
        )
    except httpx.HTTPStatusError as exc:
        _skip_if_loopback_blocked(exc)
        raise
    try:
        run = await _first_run_on(client, thread["thread_id"])
        payload, _headers = await receiver.wait_for(run["run_id"])
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

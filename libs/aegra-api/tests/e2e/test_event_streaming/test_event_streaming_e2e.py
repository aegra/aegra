"""E2E tests for Agent Protocol v2 event streaming.

Drives the v2 endpoints against a running server with
``FF_V2_EVENT_STREAMING=true``. Uses the ``stress_test`` graph (no LLM, so
the test is hermetic) to exercise run.start + an SSE event stream.

These are skipped unless the flag is on; the server returns 503 otherwise.
"""

import json
from typing import Any

import httpx
import pytest

from aegra_api.settings import settings
from tests.e2e._utils import elog, get_e2e_client


def _base_url() -> str:
    url = settings.app.SERVER_URL
    assert url is not None
    return url


async def _v2_enabled(http: httpx.AsyncClient, thread_id: str) -> bool:
    """Probe whether the server has v2 streaming enabled (else 503)."""
    resp = await http.post(f"/threads/{thread_id}/commands", json={"id": 0, "method": "run.start", "params": {}})
    return resp.status_code != 503


async def _setup_thread_and_assistant() -> tuple[str, str]:
    client = get_e2e_client()
    assistant = await client.assistants.create(graph_id="stress_test", if_exists="do_nothing")
    thread = await client.threads.create()
    return assistant["assistant_id"], thread["thread_id"]


def _parse_sse(body: str) -> list[dict[str, Any]]:
    """Parse SSE text into a list of {event, data, id} frames."""
    frames: list[dict[str, Any]] = []
    for block in body.split("\n\n"):
        frame: dict[str, Any] = {}
        for line in block.splitlines():
            if line.startswith("event:"):
                frame["event"] = line[len("event:") :].strip()
            elif line.startswith("data:"):
                frame["data"] = json.loads(line[len("data:") :].strip())
            elif line.startswith("id:"):
                frame["id"] = line[len("id:") :].strip()
        if frame.get("event"):
            frames.append(frame)
    return frames


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_run_start_and_stream_v2_events() -> None:
    """run.start enqueues a run; the SSE stream emits v2 envelopes + lifecycle."""
    assistant_id, thread_id = await _setup_thread_and_assistant()

    async with httpx.AsyncClient(base_url=_base_url(), timeout=30.0) as http:
        if not await _v2_enabled(http, thread_id):
            pytest.skip("FF_V2_EVENT_STREAMING is disabled on the server under test")

        command = {
            "id": 1,
            "method": "run.start",
            "params": {
                "assistant_id": assistant_id,
                "input": {"messages": [{"role": "user", "content": json.dumps({"delay": 0.1, "steps": 1})}]},
            },
        }
        started = await http.post(f"/threads/{thread_id}/commands", json=command)
        elog("v2 run.start", {"status": started.status_code, "body": started.text})
        assert started.status_code == 200
        run_id = started.json()["result"]["run_id"]

        async with http.stream(
            "POST",
            f"/threads/{thread_id}/stream/events",
            json={"run_id": run_id, "channels": ["values", "updates", "lifecycle"]},
        ) as resp:
            assert resp.status_code == 200
            body = ""
            async for chunk in resp.aiter_text():
                body += chunk

    frames = _parse_sse(body)
    elog("v2 stream frames", [f["event"] for f in frames])
    methods = {f["event"] for f in frames}
    assert "lifecycle" in methods
    lifecycle = [f for f in frames if f["event"] == "lifecycle"]
    assert lifecycle[-1]["data"]["params"]["event"] in ("completed", "interrupted", "failed")
    # Every frame carries the v2 envelope shape.
    for frame in frames:
        assert frame["data"]["type"] == "event"
        assert frame["data"]["method"] == frame["event"]
        assert isinstance(frame["data"]["seq"], int)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_unknown_command_returns_not_supported() -> None:
    """An unsupported command method returns a 400 error envelope."""
    _assistant_id, thread_id = await _setup_thread_and_assistant()

    async with httpx.AsyncClient(base_url=_base_url(), timeout=15.0) as http:
        if not await _v2_enabled(http, thread_id):
            pytest.skip("FF_V2_EVENT_STREAMING is disabled on the server under test")
        resp = await http.post(
            f"/threads/{thread_id}/commands", json={"id": 9, "method": "agent.getTree", "params": {}}
        )
        assert resp.status_code == 400
        assert resp.json() == {"type": "error", "id": 9, "error": "not_supported", "message": resp.json()["message"]}


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_stream_unknown_channel_rejected() -> None:
    """An unsupported channel on the SSE filter is rejected with 400."""
    _assistant_id, thread_id = await _setup_thread_and_assistant()

    async with httpx.AsyncClient(base_url=_base_url(), timeout=15.0) as http:
        if not await _v2_enabled(http, thread_id):
            pytest.skip("FF_V2_EVENT_STREAMING is disabled on the server under test")
        resp = await http.post(f"/threads/{thread_id}/stream/events", json={"run_id": "x", "channels": ["bogus"]})
        assert resp.status_code == 400

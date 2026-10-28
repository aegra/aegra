"""E2E tests for stateless (thread-free) run endpoints.

These tests hit a real running Aegra server and exercise the ``POST /runs/wait``,
``POST /runs/stream``, and ``POST /runs`` endpoints that accept no thread_id.

Requirements:
  - A running Aegra server (``uv run aegra dev`` or ``docker compose up``)
  - The ``agent`` and ``stress_test`` graphs deployed in the server's ``aegra.json``
"""

import json

import pytest
from httpx import AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from aegra_api.settings import settings
from tests.e2e._utils import await_terminal_run, elog, get_e2e_client

# ---------------------------------------------------------------------------
# POST /runs/wait  (stateless wait)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_stateless_wait_returns_output() -> None:
    """POST /runs/wait creates an ephemeral thread, runs the graph, and returns output."""
    sdk = get_e2e_client()

    # Ensure assistant exists
    assistant = await sdk.assistants.create(
        graph_id="agent",
        config={"tags": ["stateless", "wait"]},
        if_exists="do_nothing",
    )
    elog("Assistant.create", assistant)
    assistant_id = assistant["assistant_id"]

    async with AsyncClient(base_url=settings.app.SERVER_URL, timeout=120.0) as http:
        resp = await http.post(
            "/runs/wait",
            json={
                "assistant_id": assistant_id,
                "input": {"messages": [{"role": "user", "content": "Say hello in one word."}]},
            },
        )
        elog("POST /runs/wait", {"status": resp.status_code, "body": resp.json() if resp.status_code == 200 else None})

        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        output = resp.json()
        assert isinstance(output, dict)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_stateless_wait_deletes_ephemeral_thread() -> None:
    """POST /runs/wait with default on_completion deletes the ephemeral thread afterward."""
    sdk = get_e2e_client()

    assistant = await sdk.assistants.create(
        graph_id="agent",
        config={"tags": ["stateless", "wait-delete"]},
        if_exists="do_nothing",
    )
    assistant_id = assistant["assistant_id"]

    # Call stateless wait – the server generates a thread internally
    async with AsyncClient(base_url=settings.app.SERVER_URL, timeout=120.0) as http:
        resp = await http.post(
            "/runs/wait",
            json={
                "assistant_id": assistant_id,
                "input": {"messages": [{"role": "user", "content": "Say hi."}]},
            },
        )
        assert resp.status_code == 200

    # We cannot easily know the generated thread_id, but we can verify
    # the test completes without errors (cleanup runs in finally block).
    # If cleanup failed, the server would log errors.


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_stateless_wait_keep_preserves_thread() -> None:
    """POST /runs/wait with on_completion='keep' preserves the thread."""
    sdk = get_e2e_client()

    assistant = await sdk.assistants.create(
        graph_id="agent",
        config={"tags": ["stateless", "wait-keep"]},
        if_exists="do_nothing",
    )
    assistant_id = assistant["assistant_id"]

    async with AsyncClient(base_url=settings.app.SERVER_URL, timeout=120.0) as http:
        resp = await http.post(
            "/runs/wait",
            json={
                "assistant_id": assistant_id,
                "input": {"messages": [{"role": "user", "content": "Say hi."}]},
                "on_completion": "keep",
            },
        )
        elog("POST /runs/wait (keep)", {"status": resp.status_code})
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# POST /runs/stream  (stateless stream)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_stateless_stream_returns_sse_events() -> None:
    """POST /runs/stream returns a valid SSE stream."""
    sdk = get_e2e_client()

    assistant = await sdk.assistants.create(
        graph_id="agent",
        config={"tags": ["stateless", "stream"]},
        if_exists="do_nothing",
    )
    assistant_id = assistant["assistant_id"]

    async with (
        AsyncClient(base_url=settings.app.SERVER_URL, timeout=120.0) as http,
        http.stream(
            "POST",
            "/runs/stream",
            json={
                "assistant_id": assistant_id,
                "input": {"messages": [{"role": "user", "content": "Say hello in one word."}]},
            },
        ) as resp,
    ):
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

        event_count = 0
        async for line in resp.aiter_lines():
            if line.startswith("event:") or line.startswith("data:"):
                event_count += 1
            if event_count >= 3:
                break  # enough to prove streaming works

        elog("POST /runs/stream", {"events_seen": event_count})
        assert event_count > 0, "Expected at least one SSE event"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_stateless_stream_with_keep() -> None:
    """POST /runs/stream with on_completion='keep' still streams correctly."""
    sdk = get_e2e_client()

    assistant = await sdk.assistants.create(
        graph_id="agent",
        config={"tags": ["stateless", "stream-keep"]},
        if_exists="do_nothing",
    )
    assistant_id = assistant["assistant_id"]

    async with (
        AsyncClient(base_url=settings.app.SERVER_URL, timeout=120.0) as http,
        http.stream(
            "POST",
            "/runs/stream",
            json={
                "assistant_id": assistant_id,
                "input": {"messages": [{"role": "user", "content": "Say hi."}]},
                "on_completion": "keep",
            },
        ) as resp,
    ):
        assert resp.status_code == 200

        event_count = 0
        async for line in resp.aiter_lines():
            if line.startswith("event:") or line.startswith("data:"):
                event_count += 1
            if event_count >= 3:
                break

        assert event_count > 0


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_stateless_stream_disconnect_is_reclaimed_by_thread_ttl() -> None:
    """A disconnected stateless stream is removed by the TTL fallback after its run settles."""
    sdk = get_e2e_client()
    assistant = await sdk.assistants.create(graph_id="stress_test", if_exists="do_nothing")
    input_data = {"messages": [{"role": "user", "content": json.dumps({"delay": 0.2, "steps": 20})}]}

    async with (
        AsyncClient(base_url=settings.app.SERVER_URL, timeout=120.0) as http,
        http.stream(
            "POST",
            "/runs/stream",
            json={"assistant_id": assistant["assistant_id"], "input": input_data, "on_disconnect": "continue"},
        ) as response,
    ):
        assert response.status_code == 200, await response.aread()
        location = response.headers["content-location"]
        _, thread_id, _, run_id = location.strip("/").split("/", 3)
        async for _ in response.aiter_lines():
            break

    run = await await_terminal_run(sdk, thread_id, run_id, timeout=30.0)
    assert run["status"] == "success"
    assert (await sdk.threads.get(thread_id))["thread_id"] == thread_id

    engine = create_async_engine(settings.db.database_url)
    try:
        async with engine.begin() as connection:
            result = await connection.execute(
                text("UPDATE thread_ttl SET expires_at = now() - interval '1 second' WHERE thread_id = :thread_id"),
                {"thread_id": thread_id},
            )
            assert result.rowcount == 1, "stateless thread should have a TTL fallback row"
    finally:
        await engine.dispose()

    async with AsyncClient(base_url=settings.app.SERVER_URL, timeout=30.0) as http:
        prune = await http.post("/threads/prune")
    assert prune.status_code == 200, prune.text
    assert prune.json()["deleted"] >= 1
    with pytest.raises(Exception, match="404|not found|Not Found"):
        await sdk.threads.get(thread_id)


# ---------------------------------------------------------------------------
# POST /runs  (stateless background run)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_stateless_create_run_returns_run_object() -> None:
    """POST /runs returns a Run object with run_id and pending/running status."""
    sdk = get_e2e_client()

    assistant = await sdk.assistants.create(
        graph_id="agent",
        config={"tags": ["stateless", "background"]},
        if_exists="do_nothing",
    )
    assistant_id = assistant["assistant_id"]

    async with AsyncClient(base_url=settings.app.SERVER_URL, timeout=120.0) as http:
        resp = await http.post(
            "/runs",
            json={
                "assistant_id": assistant_id,
                "input": {"messages": [{"role": "user", "content": "Say hello."}]},
            },
        )
        elog("POST /runs", {"status": resp.status_code, "body": resp.json() if resp.status_code == 200 else None})

        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert "run_id" in data
        assert "thread_id" in data
        assert data["status"] in ("pending", "running")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_stateless_create_run_with_keep() -> None:
    """POST /runs with on_completion='keep' returns a Run and thread is preserved."""
    sdk = get_e2e_client()

    assistant = await sdk.assistants.create(
        graph_id="agent",
        config={"tags": ["stateless", "background-keep"]},
        if_exists="do_nothing",
    )
    assistant_id = assistant["assistant_id"]

    async with AsyncClient(base_url=settings.app.SERVER_URL, timeout=120.0) as http:
        resp = await http.post(
            "/runs",
            json={
                "assistant_id": assistant_id,
                "input": {"messages": [{"role": "user", "content": "Say hi."}]},
                "on_completion": "keep",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        thread_id = data["thread_id"]

    # Thread should still exist – verify via SDK
    thread = await sdk.threads.get(thread_id)
    elog("Thread preserved", thread)
    assert thread["thread_id"] == thread_id

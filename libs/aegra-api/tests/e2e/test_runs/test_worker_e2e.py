"""E2E tests for the worker executor architecture.

These tests verify that runs execute correctly through the Redis job
queue → worker coroutine → broker pipeline, not just via in-process
asyncio tasks.  They exercise:

  - Basic run execution via worker (create → join → verify output)
  - Concurrent runs processed by multiple workers
  - Wait endpoint returning results from worker execution
  - Cancel propagation via Redis pub/sub to worker
  - Stateless run execution via worker
  - Stream reconnection after worker produces events
  - SSE client disconnect → cross-instance cancel via Redis pub/sub
"""

import asyncio

import httpx
import pytest

from aegra_api.settings import settings
from tests.e2e._utils import await_terminal_run, check_and_skip_if_geo_blocked, elog, get_e2e_client


async def _run_and_check_geo(client, thread_id: str, run_id: str) -> dict:
    """Join a run and skip if geo-blocked."""
    output = await client.runs.join(thread_id=thread_id, run_id=run_id)
    run = await client.runs.get(thread_id=thread_id, run_id=run_id)
    check_and_skip_if_geo_blocked(run)
    return output


@pytest.mark.e2e
@pytest.mark.prod_only
@pytest.mark.asyncio
async def test_worker_executes_run() -> None:
    """A run submitted to the queue is picked up by a worker and produces output."""
    client = get_e2e_client()
    # geo-block check happens after run completes

    thread = await client.threads.create()
    elog("Thread", thread)

    run = await client.runs.create(
        thread_id=thread["thread_id"],
        assistant_id="agent",
        input={"messages": [{"role": "user", "content": "Say 'worker-ok' and nothing else."}]},
    )
    elog("Run created", run)
    assert run["status"] in ("pending", "running")

    output = await _run_and_check_geo(client, thread["thread_id"], run["run_id"])
    elog("Join output", output)
    assert output, "Expected non-empty output from worker execution"


@pytest.mark.e2e
@pytest.mark.prod_only
@pytest.mark.asyncio
async def test_worker_concurrent_runs() -> None:
    """Multiple runs execute concurrently across workers."""
    client = get_e2e_client()
    # geo-block check happens after run completes

    threads_and_runs: list[tuple[str, str]] = []

    for i in range(3):
        thread = await client.threads.create()
        run = await client.runs.create(
            thread_id=thread["thread_id"],
            assistant_id="agent",
            input={"messages": [{"role": "user", "content": f"Say 'concurrent-{i}' and nothing else."}]},
        )
        threads_and_runs.append((thread["thread_id"], run["run_id"]))
        elog(f"Run {i} created", run)

    results = await asyncio.gather(*[client.runs.join(thread_id=tid, run_id=rid) for tid, rid in threads_and_runs])

    for i, result in enumerate(results):
        elog(f"Run {i} result", result)
        assert result, f"Run {i} returned empty output"

    elog("All concurrent runs completed", {"count": len(results)})


@pytest.mark.e2e
@pytest.mark.prod_only
@pytest.mark.asyncio
async def test_worker_wait_endpoint() -> None:
    """The /runs/wait endpoint returns output after worker execution."""
    client = get_e2e_client()
    # geo-block check happens after run completes

    thread = await client.threads.create()
    elog("Thread", thread)

    output = await client.runs.wait(
        thread_id=thread["thread_id"],
        assistant_id="agent",
        input={"messages": [{"role": "user", "content": "Say 'wait-ok' and nothing else."}]},
    )
    elog("Wait output", output)
    assert output, "Expected non-empty output from wait endpoint"


@pytest.mark.e2e
@pytest.mark.prod_only
@pytest.mark.asyncio
async def test_worker_cancel_via_redis() -> None:
    """Cancel propagates through Redis pub/sub to the executing worker."""
    client = get_e2e_client()
    # geo-block check happens after run completes

    thread = await client.threads.create()

    run = await client.runs.create(
        thread_id=thread["thread_id"],
        assistant_id="agent",
        input={"messages": [{"role": "user", "content": "Write a very long essay about the history of computing."}]},
    )
    elog("Run created", run)

    await asyncio.sleep(1)

    cancelled = await client.runs.cancel(
        thread_id=thread["thread_id"],
        run_id=run["run_id"],
    )
    elog("Cancel response", cancelled)

    # The worker owns this run, so it finalizes asynchronously after the cancel.
    final_run = await await_terminal_run(client, thread["thread_id"], run["run_id"])
    elog("Final run state", final_run)
    assert final_run["status"] == "interrupted", f"Expected interrupted, got {final_run['status']}"


@pytest.mark.e2e
@pytest.mark.prod_only
@pytest.mark.asyncio
async def test_worker_stateless_run() -> None:
    """Stateless /runs/wait works through the worker pipeline."""
    client = get_e2e_client()

    # Stateless runs still need a thread via SDK — use an ephemeral one
    thread = await client.threads.create()
    output = await client.runs.wait(
        thread_id=thread["thread_id"],
        assistant_id="agent",
        input={"messages": [{"role": "user", "content": "Say 'stateless-ok' and nothing else."}]},
    )
    elog("Stateless wait output", output)
    assert output, "Expected non-empty output from stateless wait"


@pytest.mark.e2e
@pytest.mark.prod_only
@pytest.mark.asyncio
async def test_worker_stream_produces_events() -> None:
    """Streaming a run executed by a worker produces SSE events."""
    client = get_e2e_client()
    # geo-block check happens after run completes

    thread = await client.threads.create()
    elog("Thread", thread)

    events: list[dict] = []
    async for event in client.runs.stream(
        thread_id=thread["thread_id"],
        assistant_id="agent",
        input={"messages": [{"role": "user", "content": "Say 'stream-ok' and nothing else."}]},
    ):
        events.append(
            {
                "event": event.event,
                "data_keys": list(event.data.keys()) if isinstance(event.data, dict) else str(type(event.data)),
            }
        )

    elog("Stream events", events)
    event_types = [e["event"] for e in events]
    assert len(events) > 0, "Expected at least one stream event"
    assert "metadata" in event_types, "Expected metadata event in stream"
    assert any(e in event_types for e in ("values", "updates")), "Expected values or updates event"


@pytest.mark.e2e
@pytest.mark.prod_only
@pytest.mark.asyncio
async def test_sse_client_disconnect_cancels_via_redis() -> None:
    """SSE client disconnect propagates cancel through Redis pub/sub.

    Default ``on_disconnect="cancel"`` wires
    ``client_close_handler_callable`` to ``broker_manager.request_cancel``
    which, in prod mode, publishes on a Redis cancel channel any worker
    can receive. Closing the HTTP stream mid-flight must drive the run
    to ``interrupted`` even when the worker that picked it up is on a
    different instance.

    Uses raw httpx so we control connection close timing precisely; the
    SDK's stream helper drains to completion and won't reproduce the
    abort.
    """
    sdk_client = get_e2e_client()
    thread = await sdk_client.threads.create()
    thread_id = thread["thread_id"]
    elog("Thread", thread)

    server_url = settings.app.SERVER_URL.rstrip("/")
    payload = {
        "assistant_id": "agent",
        "input": {"messages": [{"role": "user", "content": "Write a very long essay about computing history."}]},
        "stream_mode": ["values"],
    }

    run_id: str | None = None
    async with (
        httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=30.0)) as http,
        http.stream(
            "POST",
            f"{server_url}/threads/{thread_id}/runs/stream",
            json=payload,
            headers={"Accept": "text/event-stream"},
        ) as response,
    ):
        assert response.status_code == 200, f"stream failed: {response.status_code}"
        location = response.headers.get("content-location", "")
        parts = location.strip("/").split("/")
        if len(parts) >= 4 and parts[2] == "runs":
            run_id = parts[3]

        # Disconnect after the first complete SSE frame. ``aiter_bytes``
        # chunk boundaries are HTTP-transport-defined (chunked encoding
        # may coalesce frames or split one across many), so keying off
        # ``chunks_seen >= N`` is timing-flaky. Buffer until we see the
        # frame terminator (``\n\n`` per our format_sse_message wire
        # format) — that's deterministic across transports.
        buffered = b""
        async for chunk in response.aiter_bytes():
            buffered += chunk
            if b"\n\n" in buffered:
                break
        elog("Disconnecting after first SSE frame", {"bytes_seen": len(buffered), "run_id": run_id})
        # Drop out of the `async with` to close the connection — that's
        # the http.disconnect that fires the close handler.

    assert run_id is not None, "Could not extract run_id from stream response headers"

    # Cancel propagates via Redis pub/sub; allow worker some time to react
    # and persist the status update.
    final_status: str = "pending"
    for _ in range(30):
        await asyncio.sleep(0.5)
        run = await sdk_client.runs.get(thread_id=thread_id, run_id=run_id)
        check_and_skip_if_geo_blocked(run)
        final_status = run["status"]
        if final_status in ("interrupted", "success", "error"):
            break

    elog("Final run state", {"run_id": run_id, "status": final_status})
    assert final_status == "interrupted", (
        f"Expected interrupted after SSE disconnect, got {final_status}. "
        "Cross-instance cancel via Redis pub/sub may have regressed."
    )

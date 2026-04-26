"""Integration tests for SSE keepalive and client-disconnect handling.

Regression cover for a production bug where agents that opened upstream
WebSockets (e.g. Sberdevices voice gateway) and held them silently for
~60s were softly interrupted by Aegra. Root cause: the SSE response had
no keepalive; an idle proxy dropped the HTTP connection, Starlette turned
that into ``CancelledError`` inside the streaming generator, and the
generator's ``cancel_on_disconnect`` branch fired ``request_cancel``.

The fix migrated SSE to ``sse_starlette.EventSourceResponse`` (periodic
``: heartbeat`` comment) and moved cancel-on-disconnect into
``client_close_handler_callable`` so only a real ``http.disconnect`` ASGI
event triggers cancellation.

These tests exercise the exact wiring used by the real endpoints:
``streaming_service.stream_run_execution`` → ``sse_to_bytes`` →
``EventSourceResponse``.
"""

import asyncio
import contextlib
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from sse_starlette import EventSourceResponse

from aegra_api.core.sse import get_sse_headers, heartbeat_factory, sse_to_bytes
from aegra_api.models import Run
from aegra_api.services import streaming_service as streaming_service_module
from aegra_api.services.broker import BrokerManager, RunBroker

Scope = dict[str, Any]
Message = dict[str, Any]


def _make_run(run_id: str) -> Run:
    now = datetime.now(UTC)
    return Run(
        run_id=run_id,
        thread_id=str(uuid4()),
        assistant_id=str(uuid4()),
        status="running",
        user_id="test-user",
        input={},
        created_at=now,
        updated_at=now,
    )


@pytest.fixture
def run_id() -> str:
    return str(uuid4())


@pytest.fixture
def local_broker_manager(monkeypatch: pytest.MonkeyPatch) -> BrokerManager:
    """Fresh BrokerManager patched into the streaming service module."""
    manager = BrokerManager()
    monkeypatch.setattr(streaming_service_module, "broker_manager", manager)
    return manager


@pytest.mark.asyncio
async def test_keepalive_pings_during_silent_broker(run_id: str, local_broker_manager: BrokerManager) -> None:
    """Silent brokers must still produce ping bytes so idle proxies don't drop us.

    Without sse-starlette's ``_ping`` task, a proxy with a 60s idle timeout
    closes the HTTP connection when the graph node blocks on an upstream
    WebSocket without emitting events — the exact prod scenario this fix
    addresses.
    """
    broker = RunBroker(run_id)
    local_broker_manager._brokers[run_id] = broker
    run = _make_run(run_id)

    app = FastAPI()

    @app.get("/stream")
    async def _stream() -> EventSourceResponse:
        return EventSourceResponse(
            sse_to_bytes(streaming_service_module.streaming_service.stream_run_execution(run)),
            ping=1,  # fast ping for test speed
            ping_message_factory=heartbeat_factory,
            headers=get_sse_headers(),
        )

    collected = bytearray()

    async def _collect() -> None:
        transport = httpx.ASGITransport(app=app)
        async with (
            httpx.AsyncClient(transport=transport, base_url="http://test") as client,
            client.stream("GET", "/stream", timeout=12.0) as resp,
        ):
            assert resp.status_code == 200
            async for chunk in resp.aiter_bytes():
                collected.extend(chunk)

    async def _drive_broker() -> None:
        # Stay silent long enough that at least two ping intervals fire even
        # when ASGI startup eats a noticeable slice at the beginning (CI can
        # be slow). ping=1, sleep=4.0 → ~3 expected heartbeats, asserting >=2
        # leaves ~2s of safety margin.
        await asyncio.sleep(4.0)
        await broker.put("evt-1", ("end", {"status": "success"}))

    await asyncio.gather(_collect(), _drive_broker())

    body = bytes(collected)
    heartbeat_hits = body.count(b": heartbeat")
    assert heartbeat_hits >= 2, f"Expected >= 2 heartbeat comments during silence, got {heartbeat_hits}. Body={body!r}"


@pytest.mark.asyncio
async def test_client_disconnect_triggers_close_handler(run_id: str, local_broker_manager: BrokerManager) -> None:
    """A real ``http.disconnect`` must fire ``client_close_handler_callable``.

    This is where ``cancel_on_disconnect`` now lives — it no longer depends
    on ``CancelledError`` inside the generator, so a proxy idle-drop can't
    be misread as a real client disconnect.

    We drive the ASGI protocol directly because ``httpx.ASGITransport`` does
    not synthesize an ``http.disconnect`` when the client context closes —
    it just cancels the app task, which produces ``CancelledError`` (the
    exact false-positive the fix aims to prevent).
    """
    broker = RunBroker(run_id)
    local_broker_manager._brokers[run_id] = broker
    run = _make_run(run_id)

    handler_call_count = 0

    async def _on_close(_msg: Message) -> None:
        nonlocal handler_call_count
        handler_call_count += 1

    response = EventSourceResponse(
        sse_to_bytes(streaming_service_module.streaming_service.stream_run_execution(run)),
        ping=60,  # don't let ping interfere
        client_close_handler_callable=_on_close,
        headers=get_sse_headers(),
    )

    # Push an event so the stream produces at least one body chunk
    await broker.put("evt-1", ("values", {"x": 1}))

    scope: Scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/stream",
        "raw_path": b"/stream",
        "query_string": b"",
        "root_path": "",
        "headers": [],
        "server": ("test", 80),
        "client": ("testclient", 50000),
        "state": {},
    }

    client_messages: asyncio.Queue[Message] = asyncio.Queue()
    await client_messages.put({"type": "http.request", "body": b"", "more_body": False})

    async def _receive() -> Message:
        return await client_messages.get()

    sent: list[Message] = []
    first_body_seen = asyncio.Event()

    async def _send(msg: Message) -> None:
        sent.append(msg)
        if msg["type"] == "http.response.body" and msg.get("body"):
            first_body_seen.set()

    async def _disconnect_after_first_body() -> None:
        await asyncio.wait_for(first_body_seen.wait(), timeout=3.0)
        await client_messages.put({"type": "http.disconnect"})

    disconnect_task = asyncio.create_task(_disconnect_after_first_body())
    try:
        await asyncio.wait_for(response(scope, _receive, _send), timeout=5.0)
    finally:
        disconnect_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await disconnect_task

    assert handler_call_count == 1, (
        f"http.disconnect should fire client_close_handler_callable exactly once; got {handler_call_count}"
    )
    # Response started + at least one body chunk flushed before disconnect
    assert any(m["type"] == "http.response.start" for m in sent)
    assert any(m["type"] == "http.response.body" and m.get("body") for m in sent)


@pytest.mark.asyncio
async def test_cancelled_error_no_longer_requests_cancel(
    run_id: str,
    local_broker_manager: BrokerManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``stream_run_execution`` must not call ``request_cancel`` on cancellation.

    Pre-fix, a CancelledError inside the generator triggered
    ``broker_manager.request_cancel`` when ``cancel_on_disconnect=True``.
    With the new transport-layer disconnect handling, that side-effect was
    removed; cancellation of the generator is now side-effect-free.
    """
    broker = RunBroker(run_id)
    local_broker_manager._brokers[run_id] = broker
    run = _make_run(run_id)

    cancel_requests: list[str] = []

    async def _record_cancel(rid: str, _action: str) -> None:
        cancel_requests.append(rid)

    monkeypatch.setattr(local_broker_manager, "request_cancel", _record_cancel)

    gen = streaming_service_module.streaming_service.stream_run_execution(run)

    # Drive the generator to the exact state we want to cancel in: inside
    # its live-events loop, blocked on the next broker event. Feeding one
    # event and waiting for ``_drain`` to consume it guarantees the loop
    # is running (not a fixed sleep, which could fire before the generator
    # has even started on a loaded CI machine).
    entered_loop = asyncio.Event()

    async def _drain() -> None:
        async for _ in gen:
            entered_loop.set()

    task = asyncio.create_task(_drain())
    await broker.put("evt-1", ("values", {"x": 1}))
    await asyncio.wait_for(entered_loop.wait(), timeout=2.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert cancel_requests == [], (
        f"Cancellation of the generator must not trigger request_cancel; got {cancel_requests}"
    )

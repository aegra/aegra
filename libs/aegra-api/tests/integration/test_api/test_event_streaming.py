"""Integration tests for the v2 event streaming routes."""

from __future__ import annotations

import asyncio
import contextlib
import json
import uuid
from collections.abc import Iterator
from functools import partial
from typing import Any

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient

from aegra_api.api import event_streaming as es_module
from aegra_api.core.auth_deps import get_current_user, require_auth
from aegra_api.models.auth import User
from aegra_api.models.event_streaming import EventStreamRequest
from aegra_api.services.broker import broker_manager
from aegra_api.services.event_streaming import capabilities as caps
from aegra_api.services.event_streaming import commands as cmd_module
from aegra_api.services.event_streaming.session import ThreadEventSession

_USER = "test-user"


class _Session:
    """Test session: scalar() returns the thread's owner id, execute() lists runs.

    ``owner`` is the existing thread's user_id, or None when the thread does
    not exist yet (the run.start-creates-it path the SDK relies on). The run
    lister selects (run_id, status) rows; status stays "running" so tests
    exercise the live-tail path.
    """

    def __init__(self, *, owner: str | None, run_ids: list[str] | None = None) -> None:
        self._owner = owner
        self._run_ids = run_ids or []

    async def __aenter__(self) -> _Session:
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        return None

    async def scalar(self, _stmt: Any) -> Any:
        return self._owner

    async def execute(self, _stmt: Any) -> Any:
        rows = [(run_id, "running", "test_graph") for run_id in self._run_ids]

        class _Result:
            def all(self) -> list[tuple[str, str, str]]:
                return rows

        return _Result()


def _make_app(
    monkeypatch: pytest.MonkeyPatch, *, owner: str | None = _USER, run_ids: list[str] | None = None
) -> FastAPI:
    app = FastAPI()
    user = User(identity=_USER)
    app.dependency_overrides[require_auth] = lambda: user
    app.dependency_overrides[get_current_user] = lambda: user
    # Routes open short-lived sessions via _get_session_maker(); patch it (per
    # test, auto-restored) to return a maker yielding the in-memory test session.
    monkeypatch.setattr(es_module, "_get_session_maker", lambda: lambda: _Session(owner=owner, run_ids=run_ids))
    app.include_router(es_module.router)
    return app


@pytest.fixture(autouse=True)
def _v2_enabled(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Turn the flag on and clear the capability cache for each test."""
    monkeypatch.setattr(caps.settings.event_streaming, "FF_V2_EVENT_STREAMING", True)
    caps._probe_runtime_symbols.cache_clear()
    yield
    caps._probe_runtime_symbols.cache_clear()


class TestCommandRoute:
    def test_run_start_returns_success_envelope(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_prepare(*_args: Any, **_kwargs: Any) -> tuple[str, object, object]:
            return "run-1", object(), object()

        monkeypatch.setattr(cmd_module, "_prepare_run", fake_prepare)
        client = TestClient(_make_app(monkeypatch))

        resp = client.post(
            "/threads/t1/commands",
            json={"id": 1, "method": "run.start", "params": {"assistant_id": "agent", "input": {"messages": []}}},
        )
        assert resp.status_code == 200
        assert resp.json() == {
            "type": "success",
            "id": 1,
            "result": {"run_id": "run-1"},
            "meta": {"applied_through_seq": 0},
        }

    def test_unknown_command_returns_error_envelope_on_200(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Protocol errors ride HTTP 200 so envelope-parsing clients see the code."""
        client = TestClient(_make_app(monkeypatch))
        resp = client.post("/threads/t1/commands", json={"id": 1, "method": "agent.getTree", "params": {}})
        assert resp.status_code == 200
        assert resp.json()["error"] == "unknown_command"

    def test_cross_tenant_thread_is_404(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = TestClient(_make_app(monkeypatch, owner="other-user"))
        resp = client.post("/threads/t1/commands", json={"id": 1, "method": "run.start", "params": {}})
        assert resp.status_code == 404

    def test_new_thread_is_allowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """run.start against a not-yet-created thread is allowed (the SDK path)."""

        async def fake_prepare(*_args: Any, **_kwargs: Any) -> tuple[str, object, object]:
            return "run-1", object(), object()

        monkeypatch.setattr(cmd_module, "_prepare_run", fake_prepare)
        client = TestClient(_make_app(monkeypatch, owner=None))  # thread does not exist yet
        resp = client.post(
            "/threads/new-thread/commands",
            json={"id": 1, "method": "run.start", "params": {"assistant_id": "agent", "input": {"messages": []}}},
        )
        assert resp.status_code == 200

    def test_disabled_flag_returns_503(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(caps.settings.event_streaming, "FF_V2_EVENT_STREAMING", False)
        client = TestClient(_make_app(monkeypatch))
        resp = client.post("/threads/t1/commands", json={"id": 1, "method": "run.start", "params": {}})
        assert resp.status_code == 503
        assert "FF_V2_EVENT_STREAMING" in resp.json()["detail"]


class TestStreamRoute:
    def test_missing_channels_is_422(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = TestClient(_make_app(monkeypatch))
        resp = client.post("/threads/t1/stream/events", json={})
        assert resp.status_code == 422

    def test_unknown_channel_is_400(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = TestClient(_make_app(monkeypatch))
        resp = client.post("/threads/t1/stream/events", json={"channels": ["bogus"]})
        assert resp.status_code == 400

    def test_cross_tenant_thread_is_404(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = TestClient(_make_app(monkeypatch, owner="other-user"))
        resp = client.post("/threads/t1/stream/events", json={"channels": ["messages"]})
        assert resp.status_code == 404

    def test_stream_emits_v2_frames(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A run on the thread streams content-block frames over SSE."""
        run_id = f"run-{uuid.uuid4().hex[:8]}"

        def _msg(event_kind: str, **extra: Any) -> tuple[str, dict[str, Any]]:
            data = {"event": event_kind, **extra}
            return ("messages", {"type": "event", "method": "messages", "params": {"namespace": [], "data": data}})

        async def seed() -> None:
            broker = broker_manager.get_or_create_broker(run_id)
            await broker.put(f"{run_id}_event_1", _msg("message-start", role="ai", id="m1"))
            await broker.put(
                f"{run_id}_event_2", _msg("content-block-delta", index=0, delta={"type": "text-delta", "text": "hi"})
            )
            await broker.put(f"{run_id}_event_3", _msg("message-finish"))
            await broker.put(f"{run_id}_event_4", ("end", {"status": "success"}))

        asyncio.run(seed())
        monkeypatch.setattr(
            es_module,
            "ThreadEventSession",
            partial(ThreadEventSession, idle_grace_seconds=0.01),
        )
        client = TestClient(_make_app(monkeypatch, run_ids=[run_id]))

        with client.stream("POST", "/threads/t1/stream/events", json={"channels": ["messages", "lifecycle"]}) as resp:
            assert resp.status_code == 200
            body = "".join(resp.iter_text())

        assert "event: messages" in body
        assert "message-start" in body
        assert "content-block-delta" in body
        assert "event: lifecycle" in body
        assert "completed" in body

    async def test_stream_emits_dual_sdk_interrupt_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        run_id = f"run-{uuid.uuid4().hex[:8]}"
        interrupt_value = {"question": "Approve?"}

        async def seed() -> None:
            broker = broker_manager.get_or_create_broker(run_id)
            event = {
                "type": "event",
                "method": "updates",
                "params": {
                    "namespace": [],
                    "data": {"__interrupt__": [{"id": "int-1", "value": interrupt_value}]},
                },
            }
            await broker.put(f"{run_id}_event_1", ("updates", event))
            await broker.put(f"{run_id}_event_2", ("end", {"status": "interrupted"}))

        await seed()
        monkeypatch.setattr(
            es_module,
            "ThreadEventSession",
            partial(ThreadEventSession, idle_grace_seconds=0.01),
        )
        monkeypatch.setattr(
            es_module,
            "_get_session_maker",
            lambda: lambda: _Session(owner=_USER, run_ids=[run_id]),
        )
        response = await es_module.stream_thread_events(
            "t1",
            EventStreamRequest(channels=["input"], namespaces=None, depth=None, since=None),
            User(identity=_USER),
        )
        chunks: list[str] = []
        async for chunk in response.body_iterator:
            assert isinstance(chunk, (bytes, str))
            chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
        body = "".join(chunks)

        input_frame = next(frame for frame in body.split("\n\n") if frame.startswith("event: input.requested"))
        data_line = next(line for line in input_frame.splitlines() if line.startswith("data: "))
        envelope = json.loads(data_line.removeprefix("data: "))
        assert envelope["params"]["data"] == {
            "interrupt_id": "int-1",
            "value": interrupt_value,
            "payload": interrupt_value,
        }

    async def test_lister_session_close_survives_sse_cancel(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Disconnect during a lister poll must still run session ``__aexit__``.

        sse_starlette cancels the generator via an anyio scope (not
        ``asyncio.Task.cancel``), which is the production leak in #530.
        """
        closed = asyncio.Event()
        aexit_cancelled = False
        aexits = 0

        class _SlowSession(_Session):
            async def execute(self, _stmt: Any) -> Any:
                await asyncio.sleep(0.4)
                return await super().execute(_stmt)

            async def __aexit__(self, *_exc: Any) -> None:
                nonlocal aexit_cancelled, aexits
                try:
                    await asyncio.sleep(0)
                except asyncio.CancelledError:
                    aexit_cancelled = True
                    raise
                aexits += 1
                # First close is the pre-SSE ownership check; the second is the
                # in-stream lister poll that cancel_on_finish used to interrupt.
                if aexits >= 2:
                    closed.set()

        monkeypatch.setattr(
            es_module,
            "ThreadEventSession",
            partial(ThreadEventSession, idle_grace_seconds=5),
        )
        app = _make_app(monkeypatch)
        monkeypatch.setattr(es_module, "_get_session_maker", lambda: lambda: _SlowSession(owner=_USER))
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error"))
        serve = asyncio.create_task(server.serve())
        try:
            for _ in range(50):
                if server.started:
                    break
                await asyncio.sleep(0.02)
            port = server.servers[0].sockets[0].getsockname()[1]
            async with httpx.AsyncClient(timeout=None) as client, client.stream(
                "POST",
                f"http://127.0.0.1:{port}/threads/t1/stream/events",
                json={"channels": ["messages"]},
            ) as resp:
                assert resp.status_code == 200
                await asyncio.sleep(0.1)
                await resp.aclose()
            await asyncio.wait_for(closed.wait(), timeout=1.5)
        finally:
            server.should_exit = True
            server.force_exit = True
            try:
                await asyncio.wait_for(serve, timeout=3)
            except TimeoutError:
                serve.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await serve

        assert not aexit_cancelled

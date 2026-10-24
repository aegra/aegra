"""HTTP-level coverage for rejoining a finished run via Last-Event-ID (#472)."""

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import FastAPI

from aegra_api.core.orm import Run as RunORM
from aegra_api.services.broker import BrokerManager
from tests.fixtures.clients import create_test_app
from tests.fixtures.database import DummySessionBase


def _make_session_maker(session_instance: DummySessionBase) -> MagicMock:
    """Return a callable mimicking ``async_sessionmaker``."""
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session_instance)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return MagicMock(return_value=ctx)


def _finished_run_orm(*, status: str = "success") -> RunORM:
    now = datetime.now(UTC)
    return RunORM(
        run_id="test-run-123",
        thread_id="test-thread-123",
        assistant_id="test-assistant-123",
        user_id="test-user",
        status=status,
        input={},
        created_at=now,
        updated_at=now,
    )


def _app_with_finished_run(run_orm: RunORM) -> tuple[FastAPI, DummySessionBase]:
    app = create_test_app(include_runs=True, include_threads=False)

    class Session(DummySessionBase):
        async def scalar(self, _stmt: object) -> RunORM:
            return run_orm

    return app, Session()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["success", "error", "interrupted"])
@pytest.mark.parametrize("broker_state", ["missing", "buffered_end", "acknowledged_end", "swept"])
async def test_stream_emits_end_when_rejoining_finished_run_with_last_event_id(status: str, broker_state: str) -> None:
    """GET /stream on a terminal run with Last-Event-ID must emit end and close.

    After restart the in-memory broker is gone. The header-free path already
    emits end; the JS SDK always sends Last-Event-ID (defaulting to -1) and
    used to hang on heartbeats instead.
    """
    run = _finished_run_orm(status=status)
    app, session = _app_with_finished_run(run)
    manager = BrokerManager()
    last_event_id = "-1"
    if broker_state != "missing":
        broker = manager.get_or_create_broker(run.run_id)
        await broker.put(f"{run.run_id}_event_1", ("values", {"a": 1}))
        await broker.put(f"{run.run_id}_event_2", ("end", {"status": status}))
        if broker_state == "swept":
            manager.remove_broker(run.run_id)
        elif broker_state == "acknowledged_end":
            last_event_id = f"{run.run_id}_event_2"

    with (
        patch("aegra_api.api.runs._get_session_maker", return_value=_make_session_maker(session)),
        patch("aegra_api.services.streaming_service.broker_manager", manager),
    ):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            for _ in range(2):
                resp = await asyncio.wait_for(
                    client.get(
                        "/threads/test-thread-123/runs/test-run-123/stream",
                        headers={"Accept": "text/event-stream", "Last-Event-ID": last_event_id},
                    ),
                    timeout=2.0,
                )
                assert resp.status_code == 200
                assert resp.text.count("event: end") == 1
                assert f'{{"status":"{status}"}}' in resp.text
                if broker_state in {"missing", "swept"}:
                    assert manager.get_broker(run.run_id) is None
                if broker_state == "buffered_end":
                    assert "event: values" in resp.text
                else:
                    assert "event: values" not in resp.text


@pytest.mark.asyncio
async def test_stream_header_free_terminal_run_still_emits_end() -> None:
    """Without Last-Event-ID, a finished run still takes the short-circuit end."""
    app, session = _app_with_finished_run(_finished_run_orm(status="success"))

    with patch("aegra_api.api.runs._get_session_maker", return_value=_make_session_maker(session)):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await asyncio.wait_for(
                client.get(
                    "/threads/test-thread-123/runs/test-run-123/stream",
                    headers={"Accept": "text/event-stream"},
                ),
                timeout=2.0,
            )
            assert resp.status_code == 200
            text = resp.text

    assert text.startswith("event: end")
    assert text.count("event: end") == 1
    assert '{"status":"success"}' in text

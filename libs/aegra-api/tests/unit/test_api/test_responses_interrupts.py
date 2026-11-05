"""Interrupt lookup for an approval that arrives before the run row commits."""

from types import SimpleNamespace
from typing import Any

import pytest

from aegra_api.models.responses import ResponsesError
from aegra_api.services.broker import broker_manager
from aegra_api.services.responses_service import _await_pending_interrupts, pending_interrupt_ids


class _Session:
    """First read is the interrupted-run query, which misses while the run is still active."""

    def __init__(self, row: Any) -> None:
        self.row = row
        self.calls = 0

    async def scalar(self, _stmt: Any) -> Any:
        self.calls += 1
        if self.calls == 1:
            return None
        return self.row


@pytest.mark.asyncio
async def test_running_run_exposes_the_interrupt_from_the_broker() -> None:
    """The SSE interrupt is in the broker before finalize_run writes the run row."""
    run_id = "run-live-interrupt"
    broker = broker_manager.get_or_create_broker(run_id)
    try:
        await broker.put(
            f"{run_id}_event_1",
            ("values", {"__interrupt__": [{"id": "int-1", "value": "wait"}]}),
        )
        row = SimpleNamespace(run_id=run_id, status="running", output=None)
        ids = await pending_interrupt_ids(_Session(row), "thread-1", "user-1")
    finally:
        broker_manager.remove_broker(run_id)

    assert ids == ["int-1"]


@pytest.mark.asyncio
async def test_interrupt_lookup_retries_until_the_row_commits(monkeypatch: pytest.MonkeyPatch) -> None:
    """A client that answers the moment it sees the interrupt must not get 409."""
    calls = 0

    async def fake(_thread_id: str, _user_id: str) -> list[str]:
        nonlocal calls
        calls += 1
        if calls < 3:
            return []
        return ["int-1"]

    monkeypatch.setattr("aegra_api.services.responses_service._pending_on_fresh_session", fake)
    monkeypatch.setattr("aegra_api.services.responses_service._INTERRUPT_SETTLE_INTERVAL_SECONDS", 0)

    pending = await _await_pending_interrupts("thread-1", "user-1", "int-1")

    assert pending == ["int-1"]
    assert calls == 3


@pytest.mark.asyncio
async def test_interrupt_lookup_rejects_when_the_id_never_appears(monkeypatch: pytest.MonkeyPatch) -> None:
    """An approval for an interrupt that is not pending stays a conflict."""

    async def fake(_thread_id: str, _user_id: str) -> list[str]:
        return []

    monkeypatch.setattr("aegra_api.services.responses_service._pending_on_fresh_session", fake)
    monkeypatch.setattr("aegra_api.services.responses_service._INTERRUPT_SETTLE_ATTEMPTS", 1)
    monkeypatch.setattr("aegra_api.services.responses_service._INTERRUPT_SETTLE_INTERVAL_SECONDS", 0)

    with pytest.raises(ResponsesError, match="not pending") as exc_info:
        await _await_pending_interrupts("thread-1", "user-1", "missing")

    assert exc_info.value.status_code == 409

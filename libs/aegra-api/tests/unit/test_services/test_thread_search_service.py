"""Unit tests for ThreadSearchService projection and truncation."""

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aegra_api.models.auth import User
from aegra_api.models.threads import Thread, ThreadSearchRequest, ThreadState
from aegra_api.services.thread_search_service import (
    ThreadSearchService,
    _maybe_truncate_values,
)


def _thread(thread_id: str = "t1") -> Thread:
    now = datetime.now(UTC)
    return Thread(
        thread_id=thread_id,
        status="idle",
        metadata={"graph_id": "agent"},
        user_id="user-1",
        created_at=now,
        updated_at=now,
    )


def _orm(thread_id: str = "t1", *, graph_id: str | None = "agent") -> MagicMock:
    row = MagicMock()
    row.thread_id = thread_id
    row.metadata_json = {"graph_id": graph_id} if graph_id else {}
    return row


@pytest.mark.asyncio
async def test_build_response_default_thin_shape() -> None:
    service = ThreadSearchService()
    row = _orm()
    base = _thread()

    out = await service.build_response(
        [row],
        ThreadSearchRequest(),
        user=User(identity="user-1"),
        serialize_thread=lambda _r: base,
    )
    assert len(out) == 1
    assert "values" not in out[0]
    assert out[0]["thread_id"] == "t1"


@pytest.mark.asyncio
async def test_build_response_select_values_joins_state(monkeypatch: pytest.MonkeyPatch) -> None:
    service = ThreadSearchService()
    row = _orm()
    base = _thread()

    async def fake_fetch(rows: Any, *, user: User) -> dict[str, dict[str, Any]]:
        return {
            "t1": {
                "values": {"messages": [{"content": "hi"}]},
                "interrupts": [],
                "config": {"configurable": {"thread_id": "t1"}},
            }
        }

    monkeypatch.setattr(service, "_fetch_states_for_threads", fake_fetch)

    out = await service.build_response(
        [row],
        ThreadSearchRequest(select=["thread_id", "values"]),
        user=User(identity="user-1"),
        serialize_thread=lambda _r: base,
    )
    assert out[0] == {
        "thread_id": "t1",
        "values": {"messages": [{"content": "hi"}]},
    }


@pytest.mark.asyncio
async def test_build_response_extract_without_full_values(monkeypatch: pytest.MonkeyPatch) -> None:
    service = ThreadSearchService()
    row = _orm()
    base = _thread()

    async def fake_fetch(rows: Any, *, user: User) -> dict[str, dict[str, Any]]:
        return {
            "t1": {
                "values": {
                    "messages": [
                        {"content": "what is rust?"},
                        {"content": "borrow checking"},
                    ]
                },
                "interrupts": [],
                "config": {},
            }
        }

    monkeypatch.setattr(service, "_fetch_states_for_threads", fake_fetch)

    out = await service.build_response(
        [row],
        ThreadSearchRequest(
            select=["thread_id"],
            extract={
                "title": "values.messages[0].content",
                "last_msg": "values.messages[-1].content",
            },
        ),
        user=User(identity="user-1"),
        serialize_thread=lambda _r: base,
    )
    assert out[0]["thread_id"] == "t1"
    assert "values" not in out[0]
    assert out[0]["extracted"] == {
        "title": "what is rust?",
        "last_msg": "borrow checking",
    }


@pytest.mark.asyncio
async def test_load_thread_state_empty_without_graph_id() -> None:
    service = ThreadSearchService()
    row = _orm(graph_id=None)
    result = await service._load_thread_state_fields(row, user=User(identity="u"))
    assert result == {"values": {}, "interrupts": [], "config": {}}


def test_maybe_truncate_values_marks_large_payloads() -> None:
    huge = {"messages": ["x" * (300 * 1024)]}
    truncated = _maybe_truncate_values(huge)
    assert truncated == {"__truncated__": True}
    assert _maybe_truncate_values({"ok": True}) == {"ok": True}


@pytest.mark.asyncio
async def test_load_thread_state_fields_converts_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    service = ThreadSearchService()
    row = _orm()

    snapshot = SimpleNamespace(
        values={"messages": [{"content": "hi"}]},
        config={"configurable": {"thread_id": "t1"}},
    )
    thread_state = ThreadState(
        values={"messages": [{"content": "hi"}]},
        next=[],
        tasks=[],
        interrupts=[],
        metadata={},
        created_at=None,
        checkpoint={"checkpoint_id": "cp1", "thread_id": "t1", "checkpoint_ns": ""},
        parent_checkpoint=None,
    )

    mock_agent = MagicMock()
    mock_agent.with_config.return_value = mock_agent
    mock_agent.aget_state = AsyncMock(return_value=snapshot)

    class _GraphCtx:
        async def __aenter__(self) -> MagicMock:
            return mock_agent

        async def __aexit__(self, *args: object) -> None:
            return None

    mock_lg = MagicMock()
    mock_lg.get_graph.return_value = _GraphCtx()

    monkeypatch.setattr(
        "aegra_api.services.langgraph_service.get_langgraph_service",
        lambda: mock_lg,
    )
    monkeypatch.setattr(
        "aegra_api.services.langgraph_service.create_thread_config",
        lambda thread_id, user: {"configurable": {"thread_id": thread_id}},
    )
    monkeypatch.setattr(
        service._state_service,
        "convert_snapshot_to_thread_state",
        lambda snap, tid, subgraphs=False: thread_state,
    )

    result = await service._load_thread_state_fields(row, user=User(identity="u"))
    assert result["values"]["messages"][0]["content"] == "hi"
    assert result["config"]["configurable"]["thread_id"] == "t1"

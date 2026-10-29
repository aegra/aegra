"""Branches in the Responses facade that the happy-path tests do not reach."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from aegra_api.api.responses import _cancel_on_disconnect, _http_message, _validation_message
from aegra_api.core.orm import Thread as ThreadORM
from aegra_api.models import Run, User
from aegra_api.models.responses import (
    ConversationCreate,
    ResponseCreate,
    ResponsesError,
    code_for_status,
    openai_error_body,
    type_for_status,
)
from aegra_api.models.responses import _validate_metadata as validate_metadata
from aegra_api.services.broker import broker_manager
from aegra_api.services.redis_broker import RedisRunBroker
from aegra_api.services.responses_mapping import (
    _as_mapping,
    _function_call_item,
    _input_text,
    _items_to_messages,
    _text_from_content,
    _tool_output_item,
    build_responses_meta,
    conversation_id_of,
    decode_resume,
    interrupt_items,
    meta_from_params,
    prepare_turn,
    project_conversation_items,
    project_response,
    project_response_output,
    public_thread_metadata,
    response_status,
)
from aegra_api.services.responses_service import (
    ResolvedThread,
    StartedRun,
    _authorize_run,
    _authorize_thread,
    _await_pending_interrupts,
    _conversation_metadata,
    _interrupt_ids_from_broker,
    _interrupt_ids_from_output,
    _message_offset,
    _persist_rejection,
    _retains_run,
    _run_create,
    _timeout_payload,
    execute_response,
    iter_response_sse,
    list_conversation_items,
    load_assistant,
    pending_interrupt_ids,
    resolve_thread,
)


def _create(**overrides: object) -> ResponseCreate:
    payload: dict[str, object] = {"model": "agent", "input": "Hello"}
    payload.update(overrides)
    return ResponseCreate.model_validate(payload)


def _now() -> datetime:
    return datetime(2026, 1, 2, tzinfo=UTC)


class _Session:
    def __init__(self, row: Any) -> None:
        self.row = row
        self.added: list[Any] = []

    async def __aenter__(self) -> _Session:
        return self

    async def __aexit__(self, *_exc: object) -> bool:
        return False

    async def scalar(self, _stmt: Any) -> Any:
        return self.row

    def add(self, obj: Any) -> None:
        self.added.append(obj)

    async def commit(self) -> None:
        return None

    async def refresh(self, _obj: Any) -> None:
        return None


class _DumpFails:
    def model_dump(self) -> dict[str, Any]:
        raise TypeError("no dump")


class _DumpOk:
    def model_dump(self) -> dict[str, Any]:
        return {"id": "m1", "type": "ai", "content": "Hi"}


class _Message:
    id = "m2"
    type = "ai"
    role = None
    content = "Hello"
    tool_calls = None
    tool_call_id = None
    name = None


def test_error_helpers_cover_auth_and_server_statuses() -> None:
    """OpenAI error codes differ for auth failures, conflicts, and server errors."""
    assert code_for_status(401) == "unauthorized"
    assert code_for_status(403) == "permission_denied"
    assert code_for_status(500) == "server_error"
    assert code_for_status(422) == "invalid_request"
    assert type_for_status(401) == "permission_error"
    assert type_for_status(500) == "server_error"
    body = ResponsesError(409, "busy", code="conflict").body()
    assert body["error"]["code"] == "conflict"
    assert openai_error_body(404, "missing")["error"]["code"] == "not_found"
    assert _http_message(["nope"]) == "Request failed"


def test_validation_message_joins_fields_and_has_a_fallback() -> None:
    """A validation error names the field, and an empty error list still returns text."""
    detailed = RequestValidationError(
        [
            {"loc": ("body", "input"), "msg": "required", "type": "missing"},
            {"loc": (), "msg": "bad", "type": "missing"},
        ]
    )
    assert _validation_message(detailed) == "input: required; bad"
    assert _validation_message(RequestValidationError([])) == "Invalid request"


def test_metadata_and_unsupported_fields_are_rejected() -> None:
    """Metadata and sampling fields that this facade cannot store are 422."""
    with pytest.raises(ValueError, match="object of strings"):
        validate_metadata("nope")
    with pytest.raises(ValueError, match="exceeds"):
        validate_metadata({f"k{i}": "v" for i in range(17)})
    with pytest.raises(ValidationError, match="strings"):
        _create(metadata={"n": 1})
    with pytest.raises(ValidationError, match="too long"):
        _create(metadata={"k" * 65: "v"})
    with pytest.raises(ValidationError, match="tool_choice"):
        _create(tool_choice="required")
    with pytest.raises(ValidationError, match="input is required"):
        ResponseCreate.model_validate({"model": "agent", "input": None})
    with pytest.raises(ValidationError, match="service_tier"):
        _create(service_tier="default")
    with pytest.raises(ValidationError, match="seeding"):
        ConversationCreate.model_validate({"items": ["seed"]})
    with pytest.raises(ValidationError, match="limit"):
        ConversationCreate.model_validate({"limit": 1})


def test_prepare_turn_rejects_malformed_input_items() -> None:
    """Each unsupported input shape names the field the client has to change."""
    with pytest.raises(ResponsesError, match="empty"):
        conversation_id_of("")
    with pytest.raises(ResponsesError, match="non-empty"):
        prepare_turn(_create(input=[]))
    with pytest.raises(ResponsesError, match="object"):
        prepare_turn(ResponseCreate.model_construct(model="agent", input=["hi"]))
    with pytest.raises(ResponsesError, match="not supported"):
        prepare_turn(_create(input=[{"type": "image"}]))
    with pytest.raises(ResponsesError, match="not both"):
        prepare_turn(
            _create(
                conversation="thread-1",
                input=[
                    {"type": "mcp_approval_response", "approve": True, "approval_request_id": "int-1"},
                    {"type": "function_call_output", "call_id": "int-1", "output": "{}"},
                ],
            )
        )
    with pytest.raises(ResponsesError, match="only one"):
        prepare_turn(
            _create(
                conversation="thread-1",
                input=[
                    {"type": "mcp_approval_response", "approve": True, "approval_request_id": "a"},
                    {"type": "mcp_approval_response", "approve": True, "approval_request_id": "b"},
                ],
            )
        )
    with pytest.raises(ResponsesError, match="cannot be combined"):
        prepare_turn(
            _create(
                conversation="thread-1",
                input=[
                    {"type": "mcp_approval_response", "approve": True, "approval_request_id": "int-1"},
                    {"role": "user", "content": "Hi"},
                ],
            )
        )
    with pytest.raises(ResponsesError, match="boolean"):
        prepare_turn(
            _create(
                conversation="thread-1",
                input=[{"type": "mcp_approval_response", "approve": "yes", "approval_request_id": "int-1"}],
            )
        )
    with pytest.raises(ResponsesError, match="reason"):
        prepare_turn(
            _create(
                conversation="thread-1",
                input=[
                    {
                        "type": "mcp_approval_response",
                        "approve": False,
                        "approval_request_id": "int-1",
                        "reason": 1,
                    }
                ],
            )
        )
    with pytest.raises(ResponsesError, match="JSON"):
        decode_resume(object())
    assert decode_resume(None) is None
    assert response_status("weird", None) == "failed"


def test_message_content_and_tool_shapes_are_projected() -> None:
    """Partial messages, tool calls, and interrupts still produce response items."""
    with pytest.raises(ResponsesError, match="include a message"):
        _items_to_messages("Rules", [])
    with pytest.raises(ResponsesError, match="role"):
        _items_to_messages(None, [{"role": "tool", "content": "x"}])
    assert _input_text(None) == ""
    with pytest.raises(ResponsesError, match="string or a list"):
        _input_text(1)
    assert _input_text(["Hi", {"type": "input_text", "text": " there"}]) == "Hi there"
    with pytest.raises(ResponsesError, match="objects"):
        _input_text([1])
    with pytest.raises(ResponsesError, match="string text"):
        _input_text([{"type": "input_text", "text": 1}])
    assert _text_from_content(None) == ""
    assert _text_from_content([{"type": "text", "text": "A"}, "B"]) == "AB"
    assert _text_from_content(1) == ""

    output = project_response_output(
        {
            "messages": [
                "skip",
                {"role": "assistant", "content": "Hi", "tool_calls": [{"name": "search", "args": {"q": 1}}]},
                {"role": "tool", "content": "done"},
            ]
        }
    )
    assert output[0]["role"] == "assistant"
    assert output[1]["type"] == "function_call"
    assert output[2]["type"] == "function_call_output"
    assert project_response_output(None) == []
    assert _function_call_item("nope", "0") is None
    assert _function_call_item({"id": 1}, "0") is None
    assert _tool_output_item({"content": "x"}, 3)["call_id"] == "call_3"

    items = project_conversation_items(
        [
            "skip",
            {"role": "assistant", "content": "", "tool_calls": [{"name": "search", "arguments": "{}"}]},
            {"type": "tool", "tool_call_id": "call-9", "content": "ok"},
        ]
    )
    assert any(item["type"] == "function_call" for item in items)
    assert any(item["type"] == "function_call_output" for item in items)

    named = interrupt_items({"__interrupt__": [{"id": "int-1", "value": {"name": "search"}}]})
    skipped = interrupt_items({"__interrupt__": [{"id": 1}, "nope"]})
    assert named[0]["name"] == "search"
    assert skipped == []
    assert interrupt_items(None) == []
    assert public_thread_metadata(None) == {}
    assert public_thread_metadata({"responses_metadata": "nope"}) == {}
    assert meta_from_params(None, None) == {"model": ""}
    assert build_responses_meta(_create(), conversation_id=None, error_code="interrupt_rejected")["error_code"] == (
        "interrupt_rejected"
    )

    projected = project_response(
        response_id="run-1",
        run_status="success",
        output={"messages": []},
        error_message=None,
        created_at=_now(),
        assistant_id=None,
        meta={"user": "alice", "model": "", "metadata": {"ok": "yes", 1: "no"}},
    )
    assert projected["user"] == "alice"
    assert projected["metadata"] == {"ok": "yes"}
    assert project_conversation_items([{"role": 1, "content": "x"}]) == []
    assert _as_mapping(_DumpOk())["content"] == "Hi"
    assert _as_mapping(_DumpFails()) is None
    assert _as_mapping(_Message())["content"] == "Hello"
    assert _as_mapping(1) is None


def test_conversation_metadata_keeps_handler_keys_that_are_not_public() -> None:
    """Owner is always the caller, and client keys are not copied to the metadata root."""
    metadata = _conversation_metadata(
        {"topic": "weather", 1: "skip"},  # type: ignore[dict-item]
        {"metadata": {"team_id": "team-1", "topic": "other", "owner": "someone"}},
        {},
        owner="user-1",
    )
    assert metadata["team_id"] == "team-1"
    assert metadata["owner"] == "user-1"
    assert metadata["responses_metadata"] == {"topic": "weather"}
    from_value = _conversation_metadata(None, None, {"metadata": {"team_id": "team-2"}}, owner="user-1")
    assert from_value["team_id"] == "team-2"


@pytest.mark.asyncio
async def test_disconnect_cancel_logs_and_continues(monkeypatch: pytest.MonkeyPatch) -> None:
    """A broker failure while cancelling a disconnected stream does not raise."""

    async def boom(_run_id: str, _action: str) -> None:
        raise OSError("broker down")

    monkeypatch.setattr("aegra_api.api.responses.broker_manager.request_cancel", boom)
    await _cancel_on_disconnect("run-1")({})


def _user() -> User:
    return User(identity="user-1")


@pytest.mark.asyncio
async def test_resolve_thread_reports_missing_response_and_conversation() -> None:
    """A previous response or conversation the caller cannot see is a 404."""
    with pytest.raises(HTTPException) as missing_response:
        await resolve_thread(_Session(None), _user(), _create(previous_response_id="run-missing"))
    assert missing_response.value.status_code == 404

    with pytest.raises(HTTPException) as missing_conversation:
        await resolve_thread(_Session(None), _user(), _create(conversation="thread-missing"))
    assert missing_conversation.value.status_code == 404

    previous = SimpleNamespace(thread_id="thread-1")
    seen = {"n": 0}

    class _ThenMissing(_Session):
        async def scalar(self, _stmt: Any) -> Any:
            seen["n"] += 1
            if seen["n"] == 1:
                return previous
            return None

    with pytest.raises(HTTPException) as missing_thread:
        await resolve_thread(_ThenMissing(None), _user(), _create(previous_response_id="run-1"))
    assert missing_thread.value.status_code == 404
    assert "Thread for response" in str(missing_thread.value.detail)


@pytest.mark.asyncio
async def test_execute_response_rejects_a_missing_row_and_a_quiet_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    """A run that was not stored is 500, and a rejection needs an interrupted thread."""

    async def fake_prepare(*_args: Any, **_kwargs: Any) -> tuple[str, Run, object]:
        run = Run(
            run_id="run-1",
            thread_id="thread-1",
            assistant_id="assistant-1",
            status="pending",
            user_id="user-1",
            created_at=_now(),
            updated_at=_now(),
        )
        return "run-1", run, object()

    monkeypatch.setattr("aegra_api.services.responses_service._prepare_run", fake_prepare)
    monkeypatch.setattr("aegra_api.services.responses_service._get_session_maker", lambda: lambda: _Session(None))
    with pytest.raises(HTTPException) as missing:
        await execute_response(_user(), _create(store=True))
    assert missing.value.status_code == 500

    idle = SimpleNamespace(
        thread_id="thread-1",
        status="idle",
        run_id="run-prev",
        output={"__interrupt__": [{"id": "int-1", "value": "wait"}]},
        assistant_id="asst",
        graph_id="agent",
    )
    monkeypatch.setattr("aegra_api.services.responses_service._get_session_maker", lambda: lambda: _Session(idle))
    body = _create(
        conversation="thread-1",
        input=[{"type": "mcp_approval_response", "approve": False, "approval_request_id": "int-1"}],
    )
    with pytest.raises(HTTPException) as quiet:
        await execute_response(_user(), body)
    assert quiet.value.status_code == 409


@pytest.mark.asyncio
async def test_rejected_approval_is_stored_without_resuming(monkeypatch: pytest.MonkeyPatch) -> None:
    """approve=false writes a failed run and does not call the executor."""
    row = SimpleNamespace(
        thread_id="thread-1",
        status="interrupted",
        run_id="run-prev",
        output={"__interrupt__": [{"id": "int-1", "value": "wait"}]},
        assistant_id="asst",
        graph_id="agent",
        user_id="user-1",
        created_at=_now(),
        updated_at=_now(),
        execution_params={},
        error_message=None,
        input=None,
        config={},
        context={},
    )

    class _Graphs:
        def list_graphs(self) -> dict[str, str]:
            return {}

    monkeypatch.setattr("aegra_api.services.responses_service.get_langgraph_service", lambda: _Graphs())
    monkeypatch.setattr("aegra_api.services.responses_service._get_session_maker", lambda: lambda: _Session(row))
    body = _create(
        conversation="thread-1",
        input=[{"type": "mcp_approval_response", "approve": False, "approval_request_id": "int-1", "reason": "no"}],
    )
    result = await execute_response(_user(), body)
    assert isinstance(result, dict)
    assert result["status"] == "failed"
    assert result["error"]["message"] == "no"


@pytest.mark.asyncio
async def test_rejection_sees_the_thread_status_committed_during_the_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """approve=false waits out the interrupt commit, then reads the refreshed status."""
    row = SimpleNamespace(
        thread_id="thread-1",
        status="busy",
        run_id="run-prev",
        output={"__interrupt__": [{"id": "int-1", "value": "wait"}]},
        assistant_id="asst",
        graph_id="agent",
        user_id="user-1",
        created_at=_now(),
        updated_at=_now(),
        execution_params={},
        error_message=None,
        input=None,
        config={},
        context={},
    )

    class _Refreshing(_Session):
        async def refresh(self, obj: Any) -> None:
            obj.status = "interrupted"

    class _Graphs:
        def list_graphs(self) -> dict[str, str]:
            return {}

    async def pending(*_args: Any) -> list[str]:
        return ["int-1"]

    monkeypatch.setattr("aegra_api.services.responses_service.get_langgraph_service", lambda: _Graphs())
    monkeypatch.setattr("aegra_api.services.responses_service._await_pending_interrupts", pending)
    monkeypatch.setattr("aegra_api.services.responses_service._get_session_maker", lambda: lambda: _Refreshing(row))
    body = _create(
        conversation="thread-1",
        input=[{"type": "mcp_approval_response", "approve": False, "approval_request_id": "int-1"}],
    )
    result = await execute_response(_user(), body)
    assert isinstance(result, dict)
    assert result["status"] == "failed"


@pytest.mark.asyncio
async def test_wait_timeout_and_missing_reload(monkeypatch: pytest.MonkeyPatch) -> None:
    """A wait that expires becomes a failed response, and a vanished run is 404."""
    finished = SimpleNamespace(
        run_id="run-1",
        thread_id="thread-1",
        assistant_id="assistant-1",
        status="running",
        output=None,
        error_message=None,
        created_at=_now(),
        updated_at=_now(),
        execution_params={"responses_meta": {"model": "agent", "metadata": {}}},
        user_id="user-1",
    )
    calls = {"n": 0}

    class _Counting(_Session):
        async def scalar(self, _stmt: Any) -> Any:
            calls["n"] += 1
            if calls["n"] == 1:
                return finished
            return None

    async def fake_prepare(*_args: Any, **_kwargs: Any) -> tuple[str, Run, object]:
        run = Run(
            run_id="run-1",
            thread_id="thread-1",
            assistant_id="assistant-1",
            status="pending",
            user_id="user-1",
            created_at=_now(),
            updated_at=_now(),
        )
        return "run-1", run, object()

    async def timed_out(*_args: Any, **_kwargs: Any) -> None:
        raise TimeoutError

    monkeypatch.setattr("aegra_api.services.responses_service._prepare_run", fake_prepare)

    async def cancelled(_run_id: str) -> None:
        return None

    monkeypatch.setattr("aegra_api.services.responses_service.executor.wait_for_completion", timed_out)
    monkeypatch.setattr("aegra_api.services.responses_service.streaming_service.cancel_run", cancelled)
    monkeypatch.setattr(
        "aegra_api.services.responses_service._get_session_maker",
        lambda: lambda: _Session(finished),
    )
    payload = await execute_response(_user(), _create(store=True))
    assert isinstance(payload, dict)
    assert payload["status"] == "failed"
    assert payload["error"]["code"] == "timeout"

    monkeypatch.setattr(
        "aegra_api.services.responses_service.executor.wait_for_completion",
        lambda *_args, **_kwargs: _done(),
    )
    monkeypatch.setattr(
        "aegra_api.services.responses_service._get_session_maker",
        lambda: lambda: _Counting(None),
    )
    with pytest.raises(HTTPException) as missing:
        await execute_response(_user(), _create(store=True))
    assert missing.value.status_code == 404

    monkeypatch.setattr(
        "aegra_api.services.responses_service._get_session_maker",
        lambda: lambda: _Session(None),
    )
    with pytest.raises(HTTPException) as timeout_missing:
        await _timeout_payload("run-1", "user-1")
    assert timeout_missing.value.status_code == 404


async def _done() -> None:
    return None


@pytest.mark.asyncio
async def test_ephemeral_stream_deletes_unless_the_run_is_interrupted(monkeypatch: pytest.MonkeyPatch) -> None:
    """A store=false stream deletes the thread unless the run stopped on an interrupt."""
    deleted: list[str] = []

    async def frames(*_args: Any, **_kwargs: Any) -> Any:
        yield "event: response.completed\ndata: {}\n\n"

    async def fake_delete(thread_id: str, _user_id: str) -> None:
        deleted.append(thread_id)

    kept = SimpleNamespace(status="interrupted", output={"__interrupt__": [{"id": "int-1", "value": "wait"}]})
    state = {"run": None}

    async def reload(*_args: Any) -> Any:
        return state["run"]

    monkeypatch.setattr("aegra_api.services.responses_service.iter_responses_sse", frames)
    monkeypatch.setattr("aegra_api.services.responses_service.delete_thread_by_id", fake_delete)
    monkeypatch.setattr("aegra_api.services.responses_service._reload_run", reload)
    started = StartedRun(
        run=Run(
            run_id="run-1",
            thread_id="thread-1",
            assistant_id="assistant-1",
            status="pending",
            user_id="user-1",
            created_at=_now(),
            updated_at=_now(),
        ),
        payload={},
        ephemeral=True,
        thread_id="thread-1",
        user_id="user-1",
        cancel_on_disconnect=True,
        message_offset=0,
    )
    assert [frame async for frame in iter_response_sse(started, starting_after=None)]
    assert deleted == ["thread-1"]
    assert _retains_run(None) is False
    assert _retains_run(kept) is True

    deleted.clear()
    state["run"] = kept
    assert [frame async for frame in iter_response_sse(started, starting_after=None)]
    assert deleted == []


@pytest.mark.asyncio
async def test_conversation_items_and_message_offset_follow_thread_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Checkpoint misses are empty, and other state errors propagate."""
    thread = SimpleNamespace(thread_id="thread-1")

    async def visible(*_args: Any) -> Any:
        return thread

    monkeypatch.setattr("aegra_api.services.responses_service._thread_visible_to", visible)

    async def missing(*_args: Any) -> Any:
        raise HTTPException(404, "no checkpoint")

    monkeypatch.setattr("aegra_api.services.responses_service.get_thread_state", missing)
    assert await list_conversation_items(_Session(None), _user(), "thread-1") == []
    assert await _message_offset("thread-1", _user()) == 0

    async def broken(*_args: Any) -> Any:
        raise HTTPException(500, "down")

    monkeypatch.setattr("aegra_api.services.responses_service.get_thread_state", broken)
    with pytest.raises(HTTPException) as exc:
        await list_conversation_items(_Session(None), _user(), "thread-1")
    assert exc.value.status_code == 500
    with pytest.raises(HTTPException) as offset_exc:
        await _message_offset("thread-1", _user())
    assert offset_exc.value.status_code == 500

    async def odd_state(*_args: Any) -> Any:
        return SimpleNamespace(values="nope")

    monkeypatch.setattr("aegra_api.services.responses_service.get_thread_state", odd_state)
    assert await list_conversation_items(_Session(None), _user(), "thread-1") == []
    assert await _message_offset("thread-1", _user()) == 0

    async def messages(*_args: Any) -> Any:
        return SimpleNamespace(values={"messages": [{"role": "user", "content": "Hi"}]})

    monkeypatch.setattr("aegra_api.services.responses_service.get_thread_state", messages)
    assert await _message_offset("thread-1", _user()) == 1
    assert (await list_conversation_items(_Session(None), _user(), "thread-1"))[0]["role"] == "user"

    async def hidden(*_args: Any) -> None:
        return None

    monkeypatch.setattr("aegra_api.services.responses_service._thread_visible_to", hidden)
    with pytest.raises(HTTPException) as not_found:
        await list_conversation_items(_Session(None), _user(), "thread-1")
    assert not_found.value.status_code == 404


@pytest.mark.asyncio
async def test_load_assistant_and_authorize_existing_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Model lookup 404s when the assistant is absent, and an existing thread skips create auth."""

    class _Graphs:
        def list_graphs(self) -> dict[str, str]:
            return {}

    monkeypatch.setattr("aegra_api.services.responses_service.get_langgraph_service", lambda: _Graphs())
    with pytest.raises(HTTPException) as missing:
        await load_assistant(_Session(None), _user(), "agent")
    assert missing.value.status_code == 404

    assistant = SimpleNamespace(assistant_id="asst", graph_id="agent")
    found = await load_assistant(_Session(assistant), _user(), "agent")
    assert found.assistant_id == "asst"

    calls: list[str] = []

    async def spy(ctx: Any, _value: dict[str, Any]) -> None:
        calls.append(ctx.action)

    monkeypatch.setattr("aegra_api.services.responses_service.handle_event", spy)
    existing = ResolvedThread(thread_id="thread-1", ephemeral=False, thread=None, minted=False)
    await _authorize_thread(_user(), existing)
    assert calls == []
    await _authorize_run(_user(), "thread-1", "agent")
    assert calls == ["create_run", "read"]


def test_run_create_uses_a_resume_command_for_an_interrupt() -> None:
    """An approval resumes with Command(resume=...) instead of a new user message."""
    turn = prepare_turn(
        _create(
            conversation="thread-1",
            input=[{"type": "mcp_approval_response", "approve": True, "approval_request_id": "int-1"}],
        )
    )
    created = _run_create(_create(conversation="thread-1"), turn, resume={"decisions": [{"type": "approve"}]})
    assert created.command == {"resume": {"decisions": [{"type": "approve"}]}}
    assert created.input is None


@pytest.mark.asyncio
async def test_pending_interrupts_read_the_row_and_ignore_unrelated_broker_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Committed interrupt ids win, and non-values broker events are ignored."""
    row = SimpleNamespace(
        run_id="run-live",
        status="interrupted",
        output={"__interrupt__": [{"id": "int-1", "value": "wait"}, {"id": 2}, "skip"]},
    )
    monkeypatch.setattr("aegra_api.services.responses_service._get_session_maker", lambda: lambda: _Session(row))
    assert await _await_pending_interrupts("thread-1", "user-1", "int-1") == ["int-1"]
    assert _interrupt_ids_from_output(None) == []
    assert _interrupt_ids_from_output({"__interrupt__": "nope"}) == []
    assert await pending_interrupt_ids(_Session(None), "thread-1", "user-1") == []
    try:
        assert await _interrupt_ids_from_broker("run-absent") == []
    finally:
        broker_manager.remove_broker("run-absent")

    run_id = "run-broker-ids"
    broker = broker_manager.get_or_create_broker(run_id)
    try:
        await broker.put(f"{run_id}_event_1", ("messages", [{"id": "m"}]))
        await broker.put(f"{run_id}_event_2", (1, 2))
        await broker.put(
            f"{run_id}_event_3",
            ("values|inner", {"__interrupt__": [{"id": "int-9", "value": "wait"}]}),
        )
        await broker.put(
            f"{run_id}_event_4",
            ("values", {"__interrupt__": [{"id": "int-9", "value": "wait"}]}),
        )
        empty = SimpleNamespace(run_id=run_id, status="interrupted", output={})
        assert await pending_interrupt_ids(_Session(empty), "thread-1", "user-1") == ["int-9"]
    finally:
        broker_manager.remove_broker(run_id)


@pytest.mark.asyncio
async def test_interrupt_ids_from_broker_reads_redis_when_the_run_is_not_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An approval check on this instance reads interrupt events stored by another."""
    run_id = "run-remote-interrupt"
    remote = RedisRunBroker(run_id, f"chan:{run_id}", f"cache:{run_id}", f"counter:{run_id}")
    seen: list[str] = []

    def get_or_create(requested: str) -> RedisRunBroker:
        seen.append(requested)
        return remote

    monkeypatch.setattr(
        "aegra_api.services.responses_service.broker_manager.get_or_create_broker",
        get_or_create,
    )
    client = AsyncMock()
    client.lrange.return_value = [
        json.dumps({"event_id": f"{run_id}_event_1", "payload": ["messages", [{"id": "m"}]]}),
        json.dumps(
            {
                "event_id": f"{run_id}_event_2",
                "payload": ["values", {"__interrupt__": [{"id": "int-remote"}]}],
            }
        ),
    ]
    with patch("aegra_api.services.redis_broker.redis_manager") as redis:
        redis.get_client.return_value = client
        assert await _interrupt_ids_from_broker(run_id) == ["int-remote"]

    assert seen == [run_id]
    assert client.lrange.await_args.args[0] == f"cache:{run_id}"


@pytest.mark.asyncio
async def test_persist_rejection_records_the_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    """The stored rejection uses the assistant id and the caller's reason."""

    class _Graphs:
        def list_graphs(self) -> dict[str, str]:
            return {}

    monkeypatch.setattr("aegra_api.services.responses_service.get_langgraph_service", lambda: _Graphs())
    assistant = SimpleNamespace(assistant_id="asst", graph_id="agent")
    thread = ThreadORM(thread_id="thread-1", status="interrupted", user_id="user-1", metadata_json={})
    body = _create(
        conversation="thread-1",
        input=[{"type": "mcp_approval_response", "approve": False, "approval_request_id": "int-1"}],
    )
    result = await _persist_rejection(_Session(assistant), _user(), body, thread, reason="stop")
    assert result["error"]["message"] == "stop"
    assert result["error"]["code"] == "interrupt_rejected"

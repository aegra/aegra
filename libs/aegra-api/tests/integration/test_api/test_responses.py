"""HTTP tests for the Open Responses facade with the database and executor mocked."""

from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.dialects import postgresql

from aegra_api.core.orm import get_session
from aegra_api.models import Run, User
from tests.fixtures.clients import create_test_app, make_client
from tests.fixtures.database import DummyScalarResult, DummySessionBase, override_get_session_dep


def _now() -> datetime:
    return datetime(2026, 1, 2, tzinfo=UTC)


def _run(
    *,
    status: str = "success",
    output: dict[str, Any] | None = None,
    error_message: str | None = None,
    thread_id: str = "thread-1",
    run_id: str = "run-1",
) -> SimpleNamespace:
    if output is None:
        output = {"messages": [{"id": "a1", "type": "ai", "content": "Hello"}]}
    return SimpleNamespace(
        run_id=run_id,
        thread_id=thread_id,
        assistant_id="assistant-1",
        status=status,
        input={"messages": [{"role": "user", "content": "Hi"}]},
        output=output,
        error_message=error_message,
        config={},
        context={},
        user_id="test-user",
        created_at=_now(),
        updated_at=_now(),
        execution_params={
            "responses_meta": {
                "model": "agent",
                "metadata": {},
                "background": False,
                "conversation_id": None,
                "previous_response_id": None,
                "instructions": None,
            }
        },
    )


class ScalarSession(DummySessionBase):
    """Session whose scalar() returns one scripted row and records the statement."""

    def __init__(self, row: Any) -> None:
        self.row = row
        self.statements: list[Any] = []

    async def scalar(self, stmt: Any) -> Any:
        self.statements.append(stmt)
        return self.row

    async def scalars(self, stmt: Any = None) -> DummyScalarResult:
        return DummyScalarResult([])

    async def delete(self, _obj: Any) -> None:
        return None


def _install_auth_spy(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    calls: list[tuple[str, str]] = []

    async def spy(ctx: Any, value: dict[str, Any]) -> None:
        calls.append((ctx.resource, ctx.action))
        return None

    for target in (
        "aegra_api.services.responses_service.handle_event",
        "aegra_api.api.runs.handle_event",
        "aegra_api.api.threads.handle_event",
    ):
        monkeypatch.setattr(target, spy)
    return calls


def _use_maker(monkeypatch: pytest.MonkeyPatch, row: Any) -> None:
    """Point response preparation at the scripted session instead of a live engine."""
    monkeypatch.setattr(
        "aegra_api.services.responses_service._get_session_maker",
        lambda: lambda: ScalarSession(row),
    )


def _client(monkeypatch: pytest.MonkeyPatch, row: Any) -> tuple[TestClient, ScalarSession, list[tuple[str, str]]]:
    app = create_test_app(include_runs=False, include_threads=False, include_responses=True)
    session = ScalarSession(row)
    app.dependency_overrides[get_session] = override_get_session_dep(lambda: session)
    calls = _install_auth_spy(monkeypatch)
    return make_client(app), session, calls


@pytest.fixture
def prepared_run(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Skip graph execution and return the scripted run after the wait."""

    async def fake_prepare(
        _session: Any,
        thread_id: str,
        _request: Any,
        user: User,
        *,
        initial_status: str,
        event_streaming_v2: bool = False,
        responses_meta: dict[str, Any] | None = None,
    ) -> tuple[str, Run, MagicMock]:
        del initial_status, event_streaming_v2, responses_meta
        run = Run(
            run_id="run-1",
            thread_id=thread_id,
            assistant_id="assistant-1",
            status="pending",
            user_id=user.identity,
            created_at=_now(),
            updated_at=_now(),
        )
        return "run-1", run, MagicMock()

    monkeypatch.setattr("aegra_api.services.responses_service._prepare_run", fake_prepare)
    monkeypatch.setattr("aegra_api.services.responses_service.executor.wait_for_completion", AsyncMock())
    yield


def test_sync_response_returns_assistant_text_and_authorizes_the_run(
    monkeypatch: pytest.MonkeyPatch,
    prepared_run: None,
) -> None:
    """POST /v1/responses waits and projects the run, dispatching thread and assistant auth."""
    del prepared_run
    finished = _run()
    monkeypatch.setattr(
        "aegra_api.services.responses_service._get_session_maker",
        lambda: lambda: ScalarSession(finished),
    )
    deleted: list[str] = []

    async def fake_delete(_thread_id: str, user_id: str) -> None:
        deleted.append(user_id)

    monkeypatch.setattr("aegra_api.services.responses_service.delete_thread_by_id", fake_delete)
    client, _session, calls = _client(monkeypatch, finished)

    response = client.post("/v1/responses", json={"model": "agent", "input": "Hi", "store": False})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["id"] == "run-1"
    assert body["status"] == "completed"
    assert body["output"][0]["content"][0]["text"] == "Hello"
    assert body["object"] == "response"
    assert deleted == ["test-user"]
    assert ("threads", "create") in calls
    assert ("threads", "create_run") in calls
    assert ("assistants", "read") in calls


def test_omitted_store_keeps_the_response_for_a_later_turn(
    monkeypatch: pytest.MonkeyPatch,
    prepared_run: None,
) -> None:
    """A create that leaves store unset is kept, so previous_response_id can chain."""
    del prepared_run
    finished = _run()
    monkeypatch.setattr(
        "aegra_api.services.responses_service._get_session_maker",
        lambda: lambda: ScalarSession(finished),
    )
    deleted: list[str] = []

    async def fake_delete(_thread_id: str, user_id: str) -> None:
        deleted.append(user_id)

    async def empty_state(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(values={"messages": []})

    monkeypatch.setattr("aegra_api.services.responses_service.delete_thread_by_id", fake_delete)
    monkeypatch.setattr("aegra_api.services.responses_service.get_thread_state", empty_state)
    client, _session, _calls = _client(monkeypatch, finished)

    first = client.post("/v1/responses", json={"model": "agent", "input": "Hi"})
    assert first.status_code == 200, first.text
    assert deleted == []

    follow_up = client.post(
        "/v1/responses",
        json={"model": "agent", "input": "Continue", "previous_response_id": first.json()["id"]},
    )
    assert follow_up.status_code == 200, follow_up.text


def test_background_response_returns_without_waiting(
    monkeypatch: pytest.MonkeyPatch,
    prepared_run: None,
) -> None:
    """background=true returns the in-progress response and does not block."""
    del prepared_run
    pending = _run(status="pending", output=None)
    pending.execution_params["responses_meta"]["background"] = True
    wait = AsyncMock()
    monkeypatch.setattr("aegra_api.services.responses_service.executor.wait_for_completion", wait)
    _use_maker(monkeypatch, pending)
    client, _session, _calls = _client(monkeypatch, pending)

    response = client.post(
        "/v1/responses",
        json={"model": "agent", "input": "Hi", "background": True, "store": True},
    )

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "in_progress"
    assert response.json()["background"] is True
    wait.assert_not_awaited()


def test_stream_response_emits_responses_sse(
    monkeypatch: pytest.MonkeyPatch,
    prepared_run: None,
) -> None:
    """stream=true speaks Responses SSE rather than Agent Protocol events."""
    del prepared_run

    async def fake_iter(_run: Run, _last_event_id: str | None = None) -> Any:
        yield ("run-1_event_1", "messages/partial", [{"id": "m1", "type": "ai", "content": "Hi"}])
        yield ("run-1_event_2", "messages/complete", [{"id": "m1", "type": "ai", "content": "Hi"}])
        yield ("run-1_event_3", "end", {"status": "success"})

    monkeypatch.setattr(
        "aegra_api.services.responses_stream.streaming_service.iter_broker_events",
        fake_iter,
    )
    pending = _run(status="pending", output=None)
    _use_maker(monkeypatch, pending)
    client, _session, _calls = _client(monkeypatch, pending)

    response = client.post("/v1/responses", json={"model": "agent", "input": "Hi", "stream": True, "store": True})

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: response.created" in response.text
    assert "event: response.output_text.delta" in response.text
    assert "event: response.completed" in response.text


def test_get_response_is_scoped_to_the_caller(monkeypatch: pytest.MonkeyPatch) -> None:
    """GET loads the run with the caller's user id and returns 404 when it is missing."""
    client, session, calls = _client(monkeypatch, None)

    missing = client.get("/v1/responses/run-missing")

    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "not_found"
    compiled = str(session.statements[0].compile(dialect=postgresql.dialect()))
    assert "user_id" in compiled
    assert ("threads", "read") not in calls

    found_client, _found_session, found_calls = _client(monkeypatch, _run())
    found = found_client.get("/v1/responses/run-1")
    assert found.status_code == 200
    assert found.json()["output"][0]["content"][0]["text"] == "Hello"
    assert ("threads", "read") in found_calls


def test_get_and_cancel_authorize_the_thread_not_the_run_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """Thread handlers see thread_id, and their metadata filter is part of the thread lookup."""
    seen: list[dict[str, Any]] = []

    async def spy(ctx: Any, value: dict[str, Any]) -> dict[str, str] | None:
        seen.append(dict(value))
        if ctx.resource == "threads" and ctx.action in ("read", "update"):
            return {"metadata": {"team_id": "team-1"}}
        return None

    client, session, _calls = _client(monkeypatch, _run())
    monkeypatch.setattr("aegra_api.services.responses_service.handle_event", spy)
    monkeypatch.setattr("aegra_api.api.responses._request_run_interruption", AsyncMock())

    fetched = client.get("/v1/responses/run-1")
    assert fetched.status_code == 200, fetched.text
    cancelled = client.post("/v1/responses/run-1/cancel")
    assert cancelled.status_code == 200, cancelled.text

    assert seen == [{"thread_id": "thread-1"}, {"thread_id": "thread-1"}]
    assert all("run_id" not in value for value in seen)
    thread_sql = [str(stmt.compile(dialect=postgresql.dialect())) for stmt in session.statements[1::2]]
    assert thread_sql
    assert all("thread_id" in sql and "metadata_json @>" in sql for sql in thread_sql)


def test_cancel_projects_a_cancelled_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancel stops the run and returns a cancelled response when no interrupt is pending."""
    interrupted = AsyncMock()
    monkeypatch.setattr("aegra_api.api.responses._request_run_interruption", interrupted)
    client, _session, calls = _client(monkeypatch, _run(status="interrupted", output={}))

    response = client.post("/v1/responses/run-1/cancel")

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "cancelled"
    interrupted.assert_awaited()
    assert ("threads", "update") in calls


def test_mismatched_conversation_and_previous_response_is_422(monkeypatch: pytest.MonkeyPatch) -> None:
    """The two thread selectors have to name the same thread."""
    previous = SimpleNamespace(thread_id="thread-a")
    _use_maker(monkeypatch, previous)
    client, _session, _calls = _client(monkeypatch, previous)

    response = client.post(
        "/v1/responses",
        json={
            "model": "agent",
            "input": "Hi",
            "conversation": "thread-b",
            "previous_response_id": "run-0",
        },
    )

    assert response.status_code == 422
    assert "different threads" in response.json()["error"]["message"]


def test_unknown_fields_and_tools_use_the_openai_error_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unsupported values are 422 with an OpenAI error document, not Agent Protocol detail."""
    client, _session, _calls = _client(monkeypatch, None)

    response = client.post(
        "/v1/responses",
        json={"model": "agent", "input": "Hi", "tools": [{"type": "function", "name": "search"}]},
    )

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["code"] == "invalid_request"
    assert "tools" in error["message"]


def test_conversation_crud_and_items(monkeypatch: pytest.MonkeyPatch) -> None:
    """Conversations are threads, and items are checkpoint messages with OpenAI roles."""
    thread = SimpleNamespace(
        thread_id="conv-1",
        user_id="test-user",
        status="idle",
        created_at=_now(),
        metadata_json={"owner": "test-user", "responses_metadata": {"topic": "weather"}},
    )

    async def fake_state(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(
            values={"messages": [{"id": "u1", "type": "human", "content": "Hi"}]},
        )

    monkeypatch.setattr("aegra_api.services.responses_service.get_thread_state", fake_state)
    monkeypatch.setattr(
        "aegra_api.api.threads.db_manager.get_checkpointer",
        lambda: SimpleNamespace(adelete_thread=AsyncMock()),
    )
    client, _session, calls = _client(monkeypatch, thread)

    created = client.post("/v1/conversations", json={"metadata": {"topic": "weather"}})
    assert created.status_code == 200, created.text
    assert created.json()["object"] == "conversation"
    assert created.json()["metadata"] == {"topic": "weather"}
    assert ("threads", "create") in calls

    fetched = client.get("/v1/conversations/conv-1")
    assert fetched.status_code == 200
    assert fetched.json()["metadata"] == {"topic": "weather"}

    items = client.get("/v1/conversations/conv-1/items")
    assert items.status_code == 200, items.text
    assert items.json()["data"][0]["role"] == "user"
    assert items.json()["data"][0]["content"][0]["text"] == "Hi"

    deleted = client.delete("/v1/conversations/conv-1")
    assert deleted.status_code == 200, deleted.text
    assert deleted.json()["deleted"] is True
    assert ("threads", "delete") in calls


def test_create_conversation_keeps_handler_metadata_and_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    """threads.create may inject metadata, and the server TTL row is written with the thread."""
    added: list[Any] = []

    class RecordingSession(ScalarSession):
        def add(self, obj: Any) -> None:
            added.append(obj)

    async def spy(ctx: Any, value: dict[str, Any]) -> dict[str, Any] | None:
        if ctx.resource == "threads" and ctx.action == "create":
            return {"metadata": {"team_id": "team-1"}}
        return None

    monkeypatch.setattr("aegra_api.services.responses_service.handle_event", spy)
    monkeypatch.setattr("aegra_api.api.threads.handle_event", spy)
    monkeypatch.setattr(
        "aegra_api.services.responses_service._resolve_ttl_row",
        lambda thread_id, _requested: SimpleNamespace(thread_id=thread_id, kind="ttl"),
    )
    app = create_test_app(include_runs=False, include_threads=False, include_responses=True)
    app.dependency_overrides[get_session] = override_get_session_dep(lambda: RecordingSession(None))
    client = make_client(app)

    response = client.post("/v1/conversations", json={"metadata": {"topic": "weather"}})

    assert response.status_code == 200, response.text
    assert response.json()["metadata"] == {"topic": "weather"}
    thread = added[0]
    assert thread.metadata_json["team_id"] == "team-1"
    assert thread.metadata_json["owner"] == "test-user"
    assert thread.metadata_json["responses_metadata"] == {"topic": "weather"}
    assert added[1].kind == "ttl"


def test_create_conversation_keeps_metadata_the_handler_writes_in_place(monkeypatch: pytest.MonkeyPatch) -> None:
    """threads.create may inject team metadata by mutating the value dict and returning nothing."""
    added: list[Any] = []

    class RecordingSession(ScalarSession):
        def add(self, obj: Any) -> None:
            added.append(obj)

    async def spy(ctx: Any, value: dict[str, Any]) -> None:
        if ctx.resource == "threads" and ctx.action == "create":
            metadata = value.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
                value["metadata"] = metadata
            metadata["team_id"] = "team-1"
        return None

    monkeypatch.setattr("aegra_api.services.responses_service.handle_event", spy)
    app = create_test_app(include_runs=False, include_threads=False, include_responses=True)
    app.dependency_overrides[get_session] = override_get_session_dep(lambda: RecordingSession(None))
    client = make_client(app)

    response = client.post("/v1/conversations", json={"metadata": {"topic": "weather"}})

    assert response.status_code == 200, response.text
    assert response.json()["metadata"] == {"topic": "weather"}
    thread = added[0]
    assert thread.metadata_json["team_id"] == "team-1"
    assert thread.metadata_json["responses_metadata"] == {"topic": "weather"}


def test_read_conversation_applies_the_handler_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    """A threads.read metadata filter is part of the conversation lookup, not discarded."""

    async def spy(_ctx: Any, _value: dict[str, Any]) -> dict[str, str]:
        return {"metadata": {"team_id": "team-1"}}

    client, session, _calls = _client(monkeypatch, None)
    monkeypatch.setattr("aegra_api.services.responses_service.handle_event", spy)

    response = client.get("/v1/conversations/conv-1")

    assert response.status_code == 404
    compiled = str(session.statements[-1].compile(dialect=postgresql.dialect()))
    assert "metadata_json @>" in compiled


def test_starting_after_without_stream_is_an_openai_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """starting_after on a JSON read is a Responses error, not an Agent Protocol body."""
    client, _session, _calls = _client(monkeypatch, _run())

    response = client.get("/v1/responses/run-1", params={"starting_after": "run-1_event_1"})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


def test_cancel_missing_response_is_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancel of an unknown response is an OpenAI 404."""
    client, _session, _calls = _client(monkeypatch, None)

    response = client.post("/v1/responses/missing/cancel")

    assert response.status_code == 404
    assert response.json()["error"]["message"] == "Response 'missing' not found"


def test_get_stream_reattaches_with_the_stored_offset(monkeypatch: pytest.MonkeyPatch) -> None:
    """GET stream=true reattaches SSE and only keeps a positive integer message offset."""

    async def fake_iter(
        _run: Run,
        _response: dict[str, Any],
        *,
        starting_after: str | None,
        message_offset: int = 0,
    ) -> AsyncIterator[str]:
        del starting_after
        yield f"event: response.completed\ndata: {message_offset}\n\n"

    monkeypatch.setattr("aegra_api.api.responses.iter_responses_sse", fake_iter)
    stored = _run()
    stored.execution_params["responses_meta"]["message_offset"] = 3
    client, _session, _calls = _client(monkeypatch, stored)

    response = client.get("/v1/responses/run-1", params={"stream": "true"})

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert "data: 3" in response.text

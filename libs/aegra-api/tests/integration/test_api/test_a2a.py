"""Integration tests for the A2A agent card endpoints"""

from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from aegra_api.api import a2a as a2a_module
from aegra_api.services.assistant_service import get_assistant_service
from aegra_api.utils.assistants import resolve_assistant_id
from tests.fixtures.clients import create_test_app, make_client
from tests.fixtures.test_helpers import make_assistant

GRAPHS: dict[str, str] = {"agent": "agent.py"}


@pytest.fixture
def client(mock_assistant_service: AsyncMock) -> Iterator[TestClient]:
    """Test client with the A2A router and a mocked assistant service."""
    mock_assistant_service.langgraph_service = MagicMock()
    mock_assistant_service.langgraph_service.list_graphs.return_value = GRAPHS
    mock_assistant_service.get_assistant.return_value = make_assistant(name="Support Agent", graph_id="agent")
    mock_assistant_service.get_assistant_schemas.return_value = {
        "input_schema": {"properties": {"messages": {}}, "required": ["messages"]},
        "state_schema": None,
    }

    app = create_test_app(include_runs=False, include_threads=False)
    app.include_router(a2a_module.router)
    app.dependency_overrides[get_assistant_service] = lambda: mock_assistant_service

    a2a_module.get_default_a2a_assistant.cache_clear()
    yield make_client(app)
    a2a_module.get_default_a2a_assistant.cache_clear()


@pytest.mark.parametrize(
    "path",
    ["/a2a/asst-1", "/a2a/asst-1/.well-known/agent-card.json", "/a2a/asst-1/.well-known/agent.json"],
)
def test_agent_card_is_served_on_every_card_path(client: TestClient, path: str) -> None:
    """All three per-assistant paths return the same card."""
    resp = client.get(path)

    assert resp.status_code == 200
    card = resp.json()
    assert card["protocolVersion"] == "1.0"
    assert card["name"] == "Support Agent"
    assert card["url"] == "http://testserver/a2a/asst-1"
    assert card["skills"][0]["metadata"]["inputSchema"]["supportsA2A"] is True


def test_agent_card_returns_404_when_assistant_not_found(client: TestClient, mock_assistant_service: AsyncMock) -> None:
    """The service's 404 for a missing or unowned assistant reaches the client."""
    mock_assistant_service.get_assistant.side_effect = HTTPException(404, "Assistant 'nope' not found")

    resp = client.get("/a2a/nope")

    assert resp.status_code == 404


def test_agent_card_accepts_a_graph_id(client: TestClient, mock_assistant_service: AsyncMock) -> None:
    """A graph id is resolved to its assistant for the lookup but kept in the card's URL."""
    resp = client.get("/a2a/agent")

    assert resp.status_code == 200
    assert resp.json()["url"] == "http://testserver/a2a/agent"
    mock_assistant_service.get_assistant.assert_awaited_once_with(resolve_assistant_id("agent", GRAPHS))


def test_host_root_card_serves_the_configured_assistant(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """With http.a2a_default_assistant set, the host root serves that assistant's card."""
    monkeypatch.setattr(a2a_module, "load_http_config", lambda: {"a2a_default_assistant": "agent"})

    resp = client.get("/.well-known/agent-card.json")

    assert resp.status_code == 200
    assert resp.json()["url"] == "http://testserver/a2a/agent"


def test_host_root_card_returns_404_naming_the_config_key_when_unset(
    client: TestClient, mock_assistant_service: AsyncMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a configured default the route must not guess an assistant."""
    monkeypatch.setattr(a2a_module, "load_http_config", lambda: None)

    resp = client.get("/.well-known/agent-card.json")

    assert resp.status_code == 404
    assert "http.a2a_default_assistant" in resp.json()["detail"]
    mock_assistant_service.get_assistant.assert_not_awaited()


@pytest.mark.parametrize(
    ("raw_body", "code"),
    [
        pytest.param(b'{"jsonrpc": ', -32700, id="malformed-json"),
        pytest.param(b'{"jsonrpc": "1.0", "id": 1, "method": "SendMessage"}', -32600, id="wrong-version"),
        pytest.param(b'{"jsonrpc": "2.0", "id": {"bad": "type"}, "method": "SendMessage"}', -32600, id="bad-id"),
        pytest.param(b'{"jsonrpc": "2.0", "id": 1, "method": "Foo"}', -32601, id="unknown-method"),
        pytest.param(b'{"jsonrpc": "2.0", "id": NaN, "method": "SendMessage"}', -32600, id="nan-id"),
        pytest.param(b'{"jsonrpc": "2.0", "id": Infinity, "method": "SendMessage"}', -32600, id="infinite-id"),
        pytest.param(b"[" * 200_000, -32700, id="nested-too-deeply"),
    ],
)
def test_json_rpc_errors_are_returned_with_http_200(client: TestClient, raw_body: bytes, code: int) -> None:
    """Clients read the body only on 2xx, so a non-200 would hide the error code from them."""
    resp = client.post("/a2a/agent", content=raw_body)

    assert resp.status_code == 200
    assert resp.json()["error"]["code"] == code


def test_json_rpc_echoed_response_returns_202_with_empty_body(client: TestClient) -> None:
    resp = client.post("/a2a/agent", content=b'{"jsonrpc": "2.0", "id": 3, "result": {}}')

    assert resp.status_code == 202
    assert resp.content == b""


@pytest.mark.parametrize("request_id", [1, "1"])
def test_json_rpc_echoes_id_with_its_json_type(client: TestClient, request_id: int | str) -> None:
    resp = client.post("/a2a/agent", json={"jsonrpc": "2.0", "id": request_id, "method": "Foo"})

    echoed = resp.json()["id"]
    assert echoed == request_id
    assert type(echoed) is type(request_id)


def test_json_rpc_answers_a_repeated_id_both_times(client: TestClient) -> None:
    """The conformance suite sends the same id twice and expects both to answer."""
    body = {"jsonrpc": "2.0", "id": 9, "method": "Foo"}

    first = client.post("/a2a/agent", json=body)
    second = client.post("/a2a/agent", json=body)

    assert first.json() == second.json()
    assert first.json()["id"] == 9


@pytest.mark.parametrize(
    ("method", "params"),
    [
        pytest.param("SendMessage", {"message": {"parts": "invalid"}}, id="modern-semantically-wrong-message"),
        pytest.param("SendMessage", {"": "not_a_dict"}, id="modern-no-message"),
        pytest.param("message/send", {"message": {"parts": "invalid"}}, id="legacy-semantically-wrong-message"),
    ],
)
def test_send_with_invalid_params_returns_invalid_params(client: TestClient, method: str, params: dict) -> None:
    """The conformance suite's well-formed-envelope, wrong-payload cases: -32602, not -32600 or -32603."""
    resp = client.post("/a2a/agent", json={"jsonrpc": "2.0", "id": 5, "method": method, "params": params})

    assert resp.status_code == 200
    assert resp.json()["id"] == 5
    assert resp.json()["error"]["code"] == -32602


@pytest.mark.parametrize("method", ["SendMessage", "message/send"])
def test_send_with_an_invalid_part_returns_content_type_not_supported(client: TestClient, method: str) -> None:
    message = {"messageId": "msg-1", "role": "ROLE_USER", "parts": [{"foo": 1}]}

    resp = client.post("/a2a/agent", json={"jsonrpc": "2.0", "id": 6, "method": method, "params": {"message": message}})

    assert resp.status_code == 200
    assert resp.json()["id"] == 6
    assert resp.json()["error"]["code"] == -32005


SEND_MESSAGE: dict[str, Any] = {"messageId": "msg-1", "role": "ROLE_USER", "parts": [{"text": "hello"}]}


def _send(client: TestClient, method: str = "SendMessage", **message_overrides: Any) -> dict[str, Any]:
    params = {"message": {**SEND_MESSAGE, **message_overrides}}
    resp = client.post("/a2a/agent", json={"jsonrpc": "2.0", "id": 7, "method": method, "params": params})
    assert resp.status_code == 200
    return resp.json()


@pytest.fixture
def send_stubs(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace the run machinery behind a send: no thread exists, the run is ``run-1`` and it completes."""
    session = MagicMock()
    session.scalar = AsyncMock(return_value=None)
    maker = MagicMock()
    maker.return_value.__aenter__.return_value = session
    stubs: dict[str, Any] = {
        "session": session,
        "authorize": AsyncMock(),
        "prepare_run": AsyncMock(return_value=("run-1", None, None)),
        "wait": AsyncMock(),
        "run": MagicMock(status="success", output={"messages": [{"type": "ai", "content": "Hi there!"}]}),
        "schemas": AsyncMock(return_value={"graph_id": "agent", "input_schema": {"properties": {"messages": {}}}}),
    }
    langgraph_service = MagicMock()
    langgraph_service.list_graphs.return_value = GRAPHS
    monkeypatch.setattr(a2a_module, "get_langgraph_service", lambda: langgraph_service)
    monkeypatch.setattr(
        a2a_module, "AssistantService", MagicMock(return_value=MagicMock(get_assistant_schemas=stubs["schemas"]))
    )
    monkeypatch.setattr(a2a_module, "_get_session_maker", lambda: maker)
    monkeypatch.setattr(a2a_module, "_apply_create_run_auth", stubs["authorize"])
    monkeypatch.setattr(a2a_module, "_prepare_run", stubs["prepare_run"])
    monkeypatch.setattr(a2a_module.executor, "wait_for_completion", stubs["wait"])
    monkeypatch.setattr(a2a_module, "_read_run", AsyncMock(return_value=stubs["run"]))
    return stubs


def test_send_message_runs_the_graph_and_returns_a_wrapped_task(client: TestClient, send_stubs: dict[str, Any]) -> None:
    body = _send(client, contextId="thread-1")

    task = body["result"]["task"]
    assert body["id"] == 7
    assert (task["id"], task["contextId"]) == ("thread-1:run-1", "thread-1")
    assert task["status"]["state"] == "TASK_STATE_COMPLETED"
    assert task["artifacts"][0]["parts"] == [{"text": "Hi there!"}]

    _session, thread_id, run_create, _user = send_stubs["prepare_run"].await_args.args
    assert thread_id == "thread-1"
    assert run_create.assistant_id == "agent"
    assert run_create.input == {"messages": [{"role": "user", "content": "hello", "id": "msg-1"}]}
    send_stubs["wait"].assert_awaited_once()


def test_legacy_send_returns_the_task_unwrapped_and_tagged(client: TestClient, send_stubs: dict[str, Any]) -> None:
    message = {"kind": "message", "messageId": "msg-1", "role": "user", "parts": [{"kind": "text", "text": "hello"}]}

    resp = client.post(
        "/a2a/agent",
        json={"jsonrpc": "2.0", "id": "abc", "method": "message/send", "params": {"message": message}},
    )

    task = resp.json()["result"]
    assert resp.json()["id"] == "abc"
    assert task["kind"] == "task"
    assert task["status"]["state"] == "completed"
    assert task["artifacts"][0]["parts"] == [{"kind": "text", "text": "Hi there!"}]


def test_send_without_a_context_id_starts_a_new_thread(client: TestClient, send_stubs: dict[str, Any]) -> None:
    task = _send(client)["result"]["task"]

    assert task["contextId"]
    assert task["id"] == f"{task['contextId']}:run-1"
    assert send_stubs["prepare_run"].await_args.args[1] == task["contextId"]


def test_send_echoes_a_client_supplied_task_id_verbatim(client: TestClient, send_stubs: dict[str, Any]) -> None:
    task = _send(client, contextId="thread-1", taskId="not-composite")["result"]["task"]

    assert task["id"] == "not-composite"


def test_send_authorizes_before_creating_the_run(client: TestClient, send_stubs: dict[str, Any]) -> None:
    """The route is self-dispatching, so a denial from an ``@auth.on`` handler must stop the run."""
    send_stubs["authorize"].side_effect = HTTPException(403, "Forbidden")

    body = _send(client, contextId="thread-1")

    assert body["error"] == {"code": -32602, "message": "Forbidden"}
    send_stubs["prepare_run"].assert_not_awaited()


def test_send_to_another_users_thread_is_not_found(client: TestClient, send_stubs: dict[str, Any]) -> None:
    send_stubs["session"].scalar.return_value = MagicMock(user_id="someone-else")

    body = _send(client, contextId="thread-1")

    assert body["error"] == {"code": -32602, "message": "Thread 'thread-1' not found"}
    send_stubs["authorize"].assert_not_awaited()
    send_stubs["prepare_run"].assert_not_awaited()


def test_send_to_a_thread_awaiting_input_is_rejected_without_creating_a_run(
    client: TestClient, send_stubs: dict[str, Any]
) -> None:
    """Resume is not available yet, and a plain turn would restart the graph and drop the pending interrupt."""
    send_stubs["session"].scalar.return_value = MagicMock(user_id="test-user", status="interrupted")

    body = _send(client, contextId="thread-1", taskId="thread-1:run-0")

    assert body["error"]["code"] == -32602
    assert body["error"]["message"].startswith("Task is awaiting input")
    send_stubs["authorize"].assert_not_awaited()
    send_stubs["prepare_run"].assert_not_awaited()


def test_send_with_a_command_is_rejected_without_creating_a_run(client: TestClient, send_stubs: dict[str, Any]) -> None:
    body = _send(client, contextId="thread-1", command={"resume": "yes"})

    assert body["error"]["code"] == -32602
    assert "'command' is not supported" in body["error"]["message"]
    send_stubs["prepare_run"].assert_not_awaited()


@pytest.mark.parametrize(
    ("method", "state", "text_part"),
    [
        pytest.param("SendMessage", "TASK_STATE_CANCELED", {"text": "Task was canceled"}, id="modern"),
        pytest.param("message/send", "canceled", {"kind": "text", "text": "Task was canceled"}, id="legacy"),
    ],
)
def test_send_whose_run_was_cancelled_reports_canceled_not_completed(
    *, client: TestClient, send_stubs: dict[str, Any], method: str, state: str, text_part: dict[str, str]
) -> None:
    """CancelTask leaves the run interrupted with empty output; that must not read as a finished answer."""
    send_stubs["run"].status = "interrupted"
    send_stubs["run"].output = {}

    result = _send(client, method, contextId="thread-1")["result"]

    task = result if method == "message/send" else result["task"]
    assert task["id"] == "thread-1:run-1"
    assert task["status"]["state"] == state
    assert task["status"]["message"]["parts"] == [text_part]
    assert "artifacts" not in task


def test_send_whose_run_paused_on_an_interrupt_is_input_required(
    client: TestClient, send_stubs: dict[str, Any]
) -> None:
    send_stubs["run"].status = "interrupted"
    send_stubs["run"].output = {"__interrupt__": [{"id": "int-1", "value": "Approve?"}]}

    task = _send(client, contextId="thread-1")["result"]["task"]

    assert task["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"
    assert task["artifacts"][0]["parts"] == [{"data": {"id": "int-1", "value": "Approve?"}}]


@pytest.mark.parametrize(
    "message_overrides",
    [
        pytest.param({"contextId": 123}, id="context-id-not-a-string"),
        pytest.param({"contextId": "x" * 5000}, id="context-id-too-long"),
        pytest.param({"taskId": 5}, id="task-id-not-a-string"),
    ],
)
def test_send_with_a_malformed_id_is_invalid_params_not_an_internal_error(
    client: TestClient, send_stubs: dict[str, Any], message_overrides: dict[str, Any]
) -> None:
    body = _send(client, **message_overrides)

    assert body["error"]["code"] == -32602
    send_stubs["prepare_run"].assert_not_awaited()


def test_send_with_a_non_object_context_is_invalid_params(client: TestClient, send_stubs: dict[str, Any]) -> None:
    params = {"message": SEND_MESSAGE, "context": "str"}

    resp = client.post("/a2a/agent", json={"jsonrpc": "2.0", "id": 7, "method": "SendMessage", "params": params})

    assert resp.json()["error"] == {"code": -32602, "message": "Invalid params: 'context' must be an object"}
    send_stubs["prepare_run"].assert_not_awaited()


def test_send_to_an_unknown_assistant_returns_invalid_params(client: TestClient, send_stubs: dict[str, Any]) -> None:
    send_stubs["prepare_run"].side_effect = HTTPException(404, "Assistant 'agent' not found")

    body = _send(client)

    assert body["error"] == {"code": -32602, "message": "Assistant 'agent' not found"}


def test_unexpected_failure_returns_internal_error_without_the_exception_text(
    client: TestClient, send_stubs: dict[str, Any]
) -> None:
    send_stubs["prepare_run"].side_effect = RuntimeError("secret connection string")

    resp = client.post(
        "/a2a/agent", json={"jsonrpc": "2.0", "id": 7, "method": "SendMessage", "params": {"message": SEND_MESSAGE}}
    )

    assert resp.status_code == 200
    assert resp.json()["error"] == {"code": -32603, "message": "Internal server error"}
    assert "secret" not in resp.text


def test_send_looks_up_the_schema_of_the_assistant_behind_a_graph_id(
    client: TestClient, send_stubs: dict[str, Any]
) -> None:
    _send(client)

    send_stubs["schemas"].assert_awaited_once_with(resolve_assistant_id("agent", GRAPHS))


def test_text_to_a_graph_without_a_messages_field_is_rejected_before_running(
    client: TestClient, send_stubs: dict[str, Any]
) -> None:
    send_stubs["schemas"].return_value = {
        "graph_id": "search",
        "input_schema": {"properties": {"query": {}, "city": {}}},
    }

    body = _send(client)

    assert body["error"] == {
        "code": -32602,
        "message": (
            "Assistant 'agent' (graph 'search') does not support A2A conversational messages. Graph input schema "
            "must include a 'messages' field to accept text or file parts. Available input fields: city, query"
        ),
    }
    send_stubs["prepare_run"].assert_not_awaited()


def test_text_to_an_assistant_without_any_schema_is_rejected(client: TestClient, send_stubs: dict[str, Any]) -> None:
    send_stubs["schemas"].return_value = {"graph_id": "agent", "input_schema": None, "state_schema": None}

    body = _send(client)

    assert body["error"] == {
        "code": -32602,
        "message": (
            "Assistant 'agent' has no input schema defined. A2A conversational agents using text or file parts "
            "must have an input schema with a 'messages' field."
        ),
    }
    send_stubs["prepare_run"].assert_not_awaited()


def test_state_schema_stands_in_when_there_is_no_input_schema(client: TestClient, send_stubs: dict[str, Any]) -> None:
    send_stubs["schemas"].return_value = {
        "graph_id": "agent",
        "input_schema": None,
        "state_schema": {"properties": {"messages": {}}},
    }

    assert _send(client)["result"]["task"]["status"]["state"] == "TASK_STATE_COMPLETED"


def test_data_only_message_skips_the_messages_field_check(client: TestClient, send_stubs: dict[str, Any]) -> None:
    """A graph that takes structured input needs no ``messages`` field to receive data parts."""
    send_stubs["schemas"].return_value = {"graph_id": "search", "input_schema": {"properties": {"query": {}}}}

    body = _send(client, parts=[{"data": {"query": "weather"}}])

    assert body["result"]["task"]["status"]["state"] == "TASK_STATE_COMPLETED"
    send_stubs["schemas"].assert_not_awaited()
    assert send_stubs["prepare_run"].await_args.args[2].input == {"query": "weather"}


def _rpc(client: TestClient, method: str, params: Any) -> dict[str, Any]:
    resp = client.post("/a2a/agent", json={"jsonrpc": "2.0", "id": 7, "method": method, "params": params})
    assert resp.status_code == 200
    return resp.json()


@pytest.fixture
def task_stubs(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace the run lookup behind a task read: by default a ``success`` run on an idle thread."""
    session = MagicMock()
    session.scalar = AsyncMock(side_effect=[MagicMock(status="success", output={}), "idle"])
    maker = MagicMock()
    maker.return_value.__aenter__.return_value = session
    stubs: dict[str, Any] = {"session": session, "authorize": AsyncMock(return_value=None)}
    monkeypatch.setattr(a2a_module, "_get_session_maker", lambda: maker)
    monkeypatch.setattr(a2a_module, "handle_event", stubs["authorize"])
    return stubs


@pytest.mark.parametrize(
    ("run_status", "thread_status", "state", "message_text"),
    [
        pytest.param("pending", "busy", "TASK_STATE_SUBMITTED", None, id="pending"),
        pytest.param("running", "busy", "TASK_STATE_WORKING", None, id="running"),
        pytest.param("success", "idle", "TASK_STATE_COMPLETED", "Task completed successfully", id="success"),
        pytest.param("interrupted", "interrupted", "TASK_STATE_INPUT_REQUIRED", None, id="paused-for-input"),
        pytest.param("interrupted", "idle", "TASK_STATE_CANCELED", None, id="canceled"),
        pytest.param("error", "idle", "TASK_STATE_FAILED", "Task failed with status: error", id="error"),
        pytest.param("timeout", "idle", "TASK_STATE_FAILED", "Task failed with status: timeout", id="timeout"),
        pytest.param("success", "interrupted", "TASK_STATE_INPUT_REQUIRED", None, id="success-on-interrupted-thread"),
        pytest.param("mystery", "idle", "TASK_STATE_SUBMITTED", None, id="unknown-status"),
    ],
)
def test_get_task_reports_the_run_status_as_a_task_state(
    *,
    client: TestClient,
    task_stubs: dict[str, Any],
    run_status: str,
    thread_status: str,
    state: str,
    message_text: str | None,
) -> None:
    """A cancel and a pause both leave the run interrupted; only the pause keeps an interrupt payload."""
    output = {"__interrupt__": [{"id": "int-1", "value": "Approve?"}]} if thread_status == "interrupted" else {}
    task_stubs["session"].scalar.side_effect = [MagicMock(status=run_status, output=output), thread_status]

    task = _rpc(client, "GetTask", {"id": "thread-1:run-1"})["result"]

    assert (task["id"], task["contextId"]) == ("thread-1:run-1", "thread-1")
    assert task["status"]["state"] == state
    assert task["status"].get("message", {}).get("parts") == ([{"text": message_text}] if message_text else None)
    assert "artifacts" not in task
    assert "timestamp" not in task["status"]


def test_get_task_for_a_paused_run_without_an_interrupt_payload_is_still_input_required(
    client: TestClient, task_stubs: dict[str, Any]
) -> None:
    """Runs started through the v2 commands endpoint save no ``__interrupt__``; the thread status marks the pause."""
    task_stubs["session"].scalar.side_effect = [MagicMock(status="interrupted", output={}), "interrupted"]

    task = _rpc(client, "GetTask", {"id": "thread-1:run-1"})["result"]

    assert task["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"


def test_legacy_get_task_is_tagged_and_lowercase(client: TestClient, task_stubs: dict[str, Any]) -> None:
    task = _rpc(client, "tasks/get", {"id": "thread-1:run-1"})["result"]

    assert (task["kind"], task["status"]["state"]) == ("task", "completed")


def test_get_task_echoes_the_raw_id_beside_the_resolved_context(client: TestClient, task_stubs: dict[str, Any]) -> None:
    task = _rpc(client, "GetTask", {"id": "run-1", "contextId": "thread-1"})["result"]

    assert (task["id"], task["contextId"]) == ("run-1", "thread-1")


def test_get_task_authorizes_a_thread_read_before_touching_the_database(
    client: TestClient, task_stubs: dict[str, Any]
) -> None:
    task_stubs["authorize"].side_effect = HTTPException(403, "Forbidden")

    body = _rpc(client, "GetTask", {"id": "thread-1:run-1"})

    assert body["error"] == {"code": -32602, "message": "Forbidden"}
    context, value = task_stubs["authorize"].await_args.args
    assert (context.resource, context.action) == ("threads", "read")
    assert value == {"run_id": "run-1", "thread_id": "thread-1"}
    task_stubs["session"].scalar.assert_not_awaited()


def test_get_task_for_a_missing_or_foreign_run_is_task_not_found(
    client: TestClient, task_stubs: dict[str, Any]
) -> None:
    """The lookup filters on the caller's identity, so another user's run looks exactly like no run."""
    task_stubs["session"].scalar.side_effect = [None]

    body = _rpc(client, "GetTask", {"id": "thread-1:run-1"})

    assert body["error"] == {"code": -32001, "message": "Task 'run-1' not found in thread 'thread-1'"}


@pytest.mark.parametrize("params", [{}, {"id": ""}, {"id": 5}, "not-an-object"])
def test_get_task_without_an_id_is_invalid_params(client: TestClient, task_stubs: dict[str, Any], params: Any) -> None:
    body = _rpc(client, "GetTask", params)

    assert body["error"] == {"code": -32602, "message": "Missing required parameter: id (task_id)"}


@pytest.mark.parametrize(
    "extra",
    [
        pytest.param({"historyLength": 5}, id="history-requested"),
        pytest.param({"historyLength": "abc"}, id="history-not-an-integer"),
        pytest.param({"historyLength": 99}, id="history-too-large"),
        pytest.param({"contextId": 5}, id="context-id-not-a-string"),
    ],
)
def test_get_task_with_a_param_it_cannot_honour_is_invalid_params(
    client: TestClient, task_stubs: dict[str, Any], extra: dict[str, Any]
) -> None:
    """A field is either honoured or rejected; responses carry no history yet."""
    body = _rpc(client, "GetTask", {"id": "thread-1:run-1", **extra})

    assert body["error"]["code"] == -32602
    task_stubs["session"].scalar.assert_not_awaited()


def test_get_task_with_history_length_zero_is_served(client: TestClient, task_stubs: dict[str, Any]) -> None:
    body = _rpc(client, "GetTask", {"id": "thread-1:run-1", "historyLength": 0})

    assert body["result"]["status"]["state"] == "TASK_STATE_COMPLETED"
    assert "history" not in body["result"]


def test_get_task_applies_the_auth_handlers_filter_to_the_thread(
    client: TestClient, task_stubs: dict[str, Any]
) -> None:
    """A filter returned by ``@auth.on.threads.read`` must narrow the lookup, not only allow or deny it."""
    task_stubs["authorize"].return_value = {"team": "sales"}
    task_stubs["session"].scalar.side_effect = [None]

    body = _rpc(client, "GetTask", {"id": "thread-1:run-1"})

    assert body["error"]["code"] == -32001
    query = str(task_stubs["session"].scalar.await_args_list[0].args[0])
    assert "JOIN thread" in query
    assert "metadata_json" in query


def test_get_task_without_a_handler_filter_looks_up_the_run_alone(
    client: TestClient, task_stubs: dict[str, Any]
) -> None:
    _rpc(client, "GetTask", {"id": "thread-1:run-1"})

    assert "JOIN" not in str(task_stubs["session"].scalar.await_args_list[0].args[0])


def test_get_task_with_no_resolvable_context_is_task_not_found(client: TestClient, task_stubs: dict[str, Any]) -> None:
    body = _rpc(client, "GetTask", {"id": "run-1"})

    assert body["error"] == {"code": -32001, "message": "Task not found: run-1"}
    task_stubs["authorize"].assert_not_awaited()


@pytest.fixture
def cancel_stubs(task_stubs: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Task stubs plus a stubbed interruption and an instant sleep, so the settle loop costs no time."""
    task_stubs["interrupt"] = AsyncMock()
    task_stubs["sleep"] = AsyncMock()
    monkeypatch.setattr(a2a_module, "_request_run_interruption", task_stubs["interrupt"])
    monkeypatch.setattr(a2a_module.asyncio, "sleep", task_stubs["sleep"])
    return task_stubs


@pytest.mark.parametrize("run_status", ["success", "error", "interrupted", "timeout"])
def test_cancel_of_a_finished_task_is_acknowledged_without_touching_the_run(
    client: TestClient, cancel_stubs: dict[str, Any], run_status: str
) -> None:
    """Cancel is idempotent by design: a terminal task is a success result, never a not-cancelable error."""
    cancel_stubs["session"].scalar.side_effect = [MagicMock(status=run_status)]

    task = _rpc(client, "CancelTask", {"id": "thread-1:run-1"})["result"]

    assert (task["id"], task["contextId"]) == ("thread-1:run-1", "thread-1")
    assert task["status"]["state"] == "TASK_STATE_CANCELED"
    assert task["status"]["message"]["parts"] == [{"text": f"Task cancel acknowledged (was: {run_status})"}]
    assert "artifacts" not in task
    cancel_stubs["interrupt"].assert_not_awaited()


@pytest.mark.parametrize("run_status", ["pending", "running"])
def test_cancel_of_a_live_task_interrupts_the_run_and_waits_for_it_to_settle(
    client: TestClient, cancel_stubs: dict[str, Any], run_status: str
) -> None:
    run = MagicMock(status=run_status)
    cancel_stubs["session"].scalar.side_effect = [run, "running", "running", "interrupted"]

    task = _rpc(client, "CancelTask", {"id": "thread-1:run-1"})["result"]

    assert task["status"]["state"] == "TASK_STATE_CANCELED"
    assert task["status"]["message"]["parts"] == [{"text": "Task was canceled"}]
    _session, interrupted_run, action = cancel_stubs["interrupt"].await_args.args
    assert (interrupted_run, action) == (run, "interrupt")
    assert cancel_stubs["sleep"].await_count == 3


def test_cancel_stops_waiting_after_twenty_polls(client: TestClient, cancel_stubs: dict[str, Any]) -> None:
    cancel_stubs["session"].scalar.side_effect = [MagicMock(status="running"), *["running"] * 20]

    task = _rpc(client, "CancelTask", {"id": "thread-1:run-1"})["result"]

    assert task["status"]["message"]["parts"] == [{"text": "Task was canceled"}]
    assert cancel_stubs["sleep"].await_count == 20


def test_cancel_applies_the_auth_handlers_filter_and_scopes_every_query_to_the_caller(
    client: TestClient, cancel_stubs: dict[str, Any]
) -> None:
    cancel_stubs["authorize"].return_value = {"team": "sales"}
    cancel_stubs["session"].scalar.side_effect = [MagicMock(status="running"), "interrupted"]

    _rpc(client, "CancelTask", {"id": "thread-1:run-1"})

    lookup, poll = (str(call.args[0]) for call in cancel_stubs["session"].scalar.await_args_list)
    assert "JOIN thread" in lookup
    assert "metadata_json" in lookup
    assert "user_id" in poll


def test_legacy_cancel_is_tagged_and_lowercase(client: TestClient, cancel_stubs: dict[str, Any]) -> None:
    task = _rpc(client, "tasks/cancel", {"id": "thread-1:run-1"})["result"]

    assert (task["kind"], task["status"]["state"]) == ("task", "canceled")


def test_cancel_authorizes_a_thread_update_before_touching_the_database(
    client: TestClient, cancel_stubs: dict[str, Any]
) -> None:
    cancel_stubs["authorize"].side_effect = HTTPException(403, "Forbidden")

    body = _rpc(client, "CancelTask", {"id": "thread-1:run-1"})

    assert body["error"] == {"code": -32602, "message": "Forbidden"}
    context, value = cancel_stubs["authorize"].await_args.args
    assert (context.resource, context.action) == ("threads", "update")
    assert value == {"run_id": "run-1", "thread_id": "thread-1"}
    cancel_stubs["session"].scalar.assert_not_awaited()
    cancel_stubs["interrupt"].assert_not_awaited()


def test_cancel_of_a_missing_or_foreign_run_is_task_not_found(client: TestClient, cancel_stubs: dict[str, Any]) -> None:
    cancel_stubs["session"].scalar.side_effect = [None]

    body = _rpc(client, "CancelTask", {"id": "thread-1:run-1"})

    assert body["error"] == {"code": -32001, "message": "Task not found: thread-1:run-1"}
    cancel_stubs["interrupt"].assert_not_awaited()


def test_cancel_without_an_id_is_invalid_params(client: TestClient, cancel_stubs: dict[str, Any]) -> None:
    body = _rpc(client, "CancelTask", {})

    assert body["error"] == {"code": -32602, "message": "Missing required parameter: id (task_id)"}

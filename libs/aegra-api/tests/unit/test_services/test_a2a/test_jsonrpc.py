"""Tests for A2A JSON-RPC envelope validation."""

import json
from typing import Any

import pytest

from aegra_api.services.a2a.jsonrpc import (
    Dialect,
    JsonRpcErrorCode,
    JsonRpcRequest,
    Operation,
    build_error_response,
    parse_json_rpc_request,
)

VERSION_ERROR = "Invalid JSON-RPC message: missing or invalid jsonrpc version"
ID_ERROR = "Invalid JSON-RPC request: 'id' must be a string, number, or null"
NOT_A_REQUEST_ERROR = "Invalid message format: must be a JSON-RPC request or notification"
MISSING_METHOD_ERROR = "Invalid JSON-RPC request: missing 'method' field"
METHOD_TYPE_ERROR = "Method not found: method must be a string"


def _request(**fields: Any) -> bytes:
    return json.dumps({"jsonrpc": "2.0", **fields}).encode()


def test_error_response_has_exact_envelope_shape() -> None:
    response = build_error_response(JsonRpcErrorCode.TASK_NOT_FOUND, "Task not found: abc", "req-1")

    assert response == {"jsonrpc": "2.0", "id": "req-1", "error": {"code": -32001, "message": "Task not found: abc"}}


@pytest.mark.parametrize("request_id", [1, "1", 1.5, None])
def test_error_response_echoes_id_with_its_json_type(request_id: str | int | float | None) -> None:
    """The JS client compares ids with ``!==``, so ``1`` must not come back as ``"1"`` or ``1.0``."""
    response = build_error_response(JsonRpcErrorCode.INVALID_REQUEST, "x", request_id)

    assert "id" in response
    assert response["id"] == request_id
    assert type(response["id"]) is type(request_id)


@pytest.mark.parametrize(
    ("raw_body", "code", "message", "request_id"),
    [
        pytest.param(b'{"jsonrpc": ', -32700, "Invalid JSON payload", None, id="truncated-json"),
        pytest.param(b"\xff\xfe", -32700, "Invalid JSON payload", None, id="invalid-utf8"),
        pytest.param(b"[1, 2]", -32600, "Invalid message format: expected object", None, id="json-array"),
        pytest.param(b'{"id": 7, "method": "SendMessage"}', -32600, VERSION_ERROR, 7, id="version-missing"),
        pytest.param(b'{"jsonrpc": "aaa", "id": "x"}', -32600, VERSION_ERROR, "x", id="version-wrong-value"),
        pytest.param(b'{"jsonrpc": 2.0, "id": 1}', -32600, VERSION_ERROR, 1, id="version-is-a-number"),
        pytest.param(b'{"jsonrpc": "1.0", "id": {"a": 1}}', -32600, VERSION_ERROR, None, id="version-wrong-id-object"),
        pytest.param(b'{"jsonrpc": "1.0", "id": true}', -32600, VERSION_ERROR, None, id="version-wrong-id-bool"),
        pytest.param(_request(id={"bad": "type"}, method="SendMessage"), -32600, ID_ERROR, None, id="id-object"),
        pytest.param(_request(id=True, method="SendMessage"), -32600, ID_ERROR, None, id="id-bool"),
        pytest.param(_request(params={}), -32600, NOT_A_REQUEST_ERROR, None, id="no-id-no-method"),
        pytest.param(_request(id=None), -32600, NOT_A_REQUEST_ERROR, None, id="null-id-no-method"),
        pytest.param(_request(id=3), -32600, MISSING_METHOD_ERROR, 3, id="id-without-method"),
        pytest.param(_request(id=0, method=None), -32600, MISSING_METHOD_ERROR, 0, id="zero-id-null-method"),
        pytest.param(_request(id=1, method=5), -32601, METHOD_TYPE_ERROR, 1, id="method-number"),
    ],
)
def test_invalid_envelope_returns_error(
    raw_body: bytes, code: int, message: str, request_id: str | int | float | None
) -> None:
    """First failing check wins; the id is echoed only once it is known to be a valid type."""
    result = parse_json_rpc_request(raw_body)

    assert result == {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


@pytest.mark.parametrize("key", ["result", "error"])
def test_echoed_response_gets_no_answer(key: str) -> None:
    """An id with no method plus a result or error is a client echoing a response at us."""
    assert parse_json_rpc_request(_request(id=3, **{key: {}})) is None


@pytest.mark.parametrize(
    ("method", "operation", "dialect"),
    [
        ("SendMessage", "send", "modern"),
        ("message/send", "send", "legacy"),
        ("SendStreamingMessage", "stream", "modern"),
        ("message/stream", "stream", "legacy"),
        ("GetTask", "get_task", "modern"),
        ("tasks/get", "get_task", "legacy"),
        ("CancelTask", "cancel_task", "modern"),
        ("tasks/cancel", "cancel_task", "legacy"),
        ("ListTasks", "list_tasks", "modern"),
        ("GetExtendedAgentCard", "get_extended_agent_card", "modern"),
        ("agent/getAuthenticatedExtendedCard", "get_extended_agent_card", "legacy"),
    ],
)
def test_known_method_resolves_operation_and_dialect(method: str, operation: Operation, dialect: Dialect) -> None:
    result = parse_json_rpc_request(_request(id=1, method=method))

    assert isinstance(result, JsonRpcRequest)
    assert (result.method, result.operation, result.dialect) == (method, operation, dialect)


@pytest.mark.parametrize("method", ["Foo", "sendmessage", "tasks/list", "SubscribeToTask", ""])
def test_unknown_method_returns_method_not_found(method: str) -> None:
    """Matching is exact and case-sensitive; out-of-scope methods are simply absent from the table."""
    result = parse_json_rpc_request(_request(id="req-1", method=method))

    assert result == {
        "jsonrpc": "2.0",
        "id": "req-1",
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


@pytest.mark.parametrize("request_id", [1, "1", 0])
def test_valid_request_keeps_id_and_params(request_id: str | int) -> None:
    result = parse_json_rpc_request(_request(id=request_id, method="SendMessage", params={"message": {"a": 1}}))

    assert isinstance(result, JsonRpcRequest)
    assert result.id == request_id
    assert type(result.id) is type(request_id)
    assert result.params == {"message": {"a": 1}}


@pytest.mark.parametrize("literal", [b"NaN", b"Infinity", b"-Infinity"])
def test_non_finite_id_is_rejected_because_it_cannot_be_echoed(literal: bytes) -> None:
    """``json.loads`` accepts these, but a JSON response cannot carry them, so echoing one would be a 500."""
    result = parse_json_rpc_request(b'{"jsonrpc": "2.0", "id": ' + literal + b', "method": "SendMessage"}')

    assert result == build_error_response(JsonRpcErrorCode.INVALID_REQUEST, ID_ERROR, None)


def test_body_nested_too_deeply_to_parse_is_a_parse_error() -> None:
    result = parse_json_rpc_request(b"[" * 200_000)

    assert result == build_error_response(JsonRpcErrorCode.PARSE_ERROR, "Invalid JSON payload", None)

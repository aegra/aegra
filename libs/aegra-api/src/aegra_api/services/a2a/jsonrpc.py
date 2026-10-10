"""JSON-RPC envelope validation and errors for the A2A endpoint.

The envelope parser returns errors as JSON-RPC error bodies; handlers past it raise ``JsonRpcError``.
The wire contract is issue #625.
"""

import json
import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Final, Literal

Dialect = Literal["modern", "legacy"]
Operation = Literal["send", "stream", "get_task", "cancel_task", "list_tasks", "get_extended_agent_card"]


class JsonRpcErrorCode(IntEnum):
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    TASK_NOT_FOUND = -32001
    CONTENT_TYPE_NOT_SUPPORTED = -32005


@dataclass(frozen=True)
class JsonRpcRequest:
    id: str | int | float | None
    method: str
    dialect: Dialect
    operation: Operation
    params: Any


METHOD_TO_DIALECT_OPERATION: Final[dict[str, tuple[Dialect, Operation]]] = {
    "SendMessage": ("modern", "send"),
    "message/send": ("legacy", "send"),
    "SendStreamingMessage": ("modern", "stream"),
    "message/stream": ("legacy", "stream"),
    "GetTask": ("modern", "get_task"),
    "tasks/get": ("legacy", "get_task"),
    "CancelTask": ("modern", "cancel_task"),
    "tasks/cancel": ("legacy", "cancel_task"),
    "ListTasks": ("modern", "list_tasks"),
    "GetExtendedAgentCard": ("modern", "get_extended_agent_card"),
    "agent/getAuthenticatedExtendedCard": ("legacy", "get_extended_agent_card"),
}


class JsonRpcError(Exception):
    def __init__(self, code: JsonRpcErrorCode, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


def build_error_response(code: JsonRpcErrorCode, message: str, request_id: str | int | float | None) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code.value, "message": message}}


def build_success_response(result: dict[str, Any], request_id: str | int | float | None) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _is_valid_request_id(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    # ``json.loads`` accepts NaN and Infinity, which cannot be written back in a JSON response.
    if isinstance(value, float) and not math.isfinite(value):
        return False
    return value is None or isinstance(value, (str, int, float))


def parse_json_rpc_request(raw_body: bytes) -> JsonRpcRequest | dict[str, Any] | None:
    try:
        body = json.loads(raw_body)
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError):
        return build_error_response(JsonRpcErrorCode.PARSE_ERROR, "Invalid JSON payload", None)

    if not isinstance(body, dict):
        return build_error_response(JsonRpcErrorCode.INVALID_REQUEST, "Invalid message format: expected object", None)

    if body.get("jsonrpc") != "2.0":
        return build_error_response(
            JsonRpcErrorCode.INVALID_REQUEST,
            "Invalid JSON-RPC message: missing or invalid jsonrpc version",
            body.get("id") if _is_valid_request_id(body.get("id")) else None,
        )

    if not _is_valid_request_id(body.get("id")):
        return build_error_response(
            JsonRpcErrorCode.INVALID_REQUEST, "Invalid JSON-RPC request: 'id' must be a string, number, or null", None
        )

    request_id = body.get("id")
    method = body.get("method")

    if method is None:
        if request_id is None:
            return build_error_response(
                JsonRpcErrorCode.INVALID_REQUEST,
                "Invalid message format: must be a JSON-RPC request or notification",
                None,
            )
        if "result" in body or "error" in body:
            return None
        return build_error_response(
            JsonRpcErrorCode.INVALID_REQUEST, "Invalid JSON-RPC request: missing 'method' field", request_id
        )

    if not isinstance(method, str):
        return build_error_response(
            JsonRpcErrorCode.METHOD_NOT_FOUND, "Method not found: method must be a string", request_id
        )

    entry = METHOD_TO_DIALECT_OPERATION.get(method)
    if entry is None:
        return build_error_response(JsonRpcErrorCode.METHOD_NOT_FOUND, f"Method not found: {method}", request_id)
    dialect, operation = entry

    return JsonRpcRequest(id=request_id, method=method, dialect=dialect, operation=operation, params=body.get("params"))

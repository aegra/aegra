"""Tests for A2A send params validation."""

from typing import Any

import pytest

from aegra_api.services.a2a.jsonrpc import JsonRpcError, JsonRpcErrorCode
from aegra_api.services.a2a.send import SendParams, parse_send_params

MESSAGE_ERROR = "Missing or invalid 'message' in params"
MESSAGE_ID_ERROR = "Missing required 'messageId' in message"
ROLE_ERROR = "Missing required 'role' in message"
PARTS_TYPE_ERROR = "Invalid params: 'parts' must be an array"
CONFIGURATION_ERROR = "Invalid params: 'configuration' must be an object"
HISTORY_TYPE_ERROR = "historyLength must be a non-negative integer"
HISTORY_UNAVAILABLE_ERROR = "Task history is not available yet: omit 'historyLength' or set it to 0"
CONTEXT_ID_ERROR = "'contextId' must be a non-blank string of at most 255 characters"
TASK_ID_ERROR = "'taskId' must be a non-blank string of at most 255 characters"


def _message(**overrides: Any) -> dict[str, Any]:
    return {"messageId": "msg-1", "role": "ROLE_USER", "parts": [{"text": "hi"}], **overrides}


def _without(key: str) -> dict[str, Any]:
    return {k: v for k, v in _message().items() if k != key}


@pytest.mark.parametrize(
    ("params", "message"),
    [
        pytest.param("x", "Invalid params: must be an object", id="params-not-an-object"),
        pytest.param({}, MESSAGE_ERROR, id="message-missing"),
        pytest.param({"message": "x"}, MESSAGE_ERROR, id="message-not-an-object"),
        pytest.param({"message": {}}, MESSAGE_ERROR, id="message-empty"),
        pytest.param({"message": _without("messageId")}, MESSAGE_ID_ERROR, id="message-id-missing"),
        pytest.param({"message": _message(messageId="")}, MESSAGE_ID_ERROR, id="message-id-empty"),
        pytest.param({"message": _message(messageId=5)}, MESSAGE_ID_ERROR, id="message-id-not-a-string"),
        pytest.param({"message": _without("role")}, ROLE_ERROR, id="role-missing"),
        pytest.param({"message": _message(role="")}, ROLE_ERROR, id="role-empty"),
        pytest.param({"message": _without("parts")}, PARTS_TYPE_ERROR, id="parts-missing"),
        pytest.param({"message": _message(parts="invalid")}, PARTS_TYPE_ERROR, id="parts-not-a-list"),
        pytest.param({"message": _message(parts=[])}, "Message must contain at least one part", id="parts-empty"),
        pytest.param({"message": _message(), "configuration": "x"}, CONFIGURATION_ERROR, id="configuration-string"),
        pytest.param({"message": _message(), "configuration": []}, CONFIGURATION_ERROR, id="configuration-list"),
        pytest.param(
            {"message": _message(), "configuration": {"historyLength": -1}}, HISTORY_TYPE_ERROR, id="history-negative"
        ),
        pytest.param(
            {"message": _message(), "configuration": {"historyLength": True}}, HISTORY_TYPE_ERROR, id="history-bool"
        ),
        pytest.param(
            {"message": _message(), "configuration": {"historyLength": "5"}}, HISTORY_TYPE_ERROR, id="history-string"
        ),
        pytest.param(
            {"message": _message(), "configuration": {"historyLength": 11}},
            "historyLength cannot exceed 10",
            id="history-too-large",
        ),
        pytest.param(
            {"message": _message(), "configuration": {"historyLength": 3}},
            HISTORY_UNAVAILABLE_ERROR,
            id="history-requested",
        ),
        pytest.param({"message": _message(contextId=123)}, CONTEXT_ID_ERROR, id="context-id-not-a-string"),
        pytest.param({"message": _message(contextId="   ")}, CONTEXT_ID_ERROR, id="context-id-blank"),
        pytest.param({"message": _message(contextId="x" * 256)}, CONTEXT_ID_ERROR, id="context-id-too-long"),
        pytest.param({"message": _message(taskId=5)}, TASK_ID_ERROR, id="task-id-not-a-string"),
        pytest.param({"message": _message(taskId="x" * 256)}, TASK_ID_ERROR, id="task-id-too-long"),
        pytest.param(
            {"message": _message(), "context": "str"},
            "Invalid params: 'context' must be an object",
            id="context-string",
        ),
        pytest.param(
            {"message": _message(command={"resume": "yes"})},
            "'command' is not supported: resuming an interrupted task over A2A is not available yet",
            id="command",
        ),
    ],
)
def test_invalid_params_raise_invalid_params(params: Any, message: str) -> None:
    """First failing check wins, and every one of them is -32602 with the spec's exact wording."""
    with pytest.raises(JsonRpcError) as exc_info:
        parse_send_params(params)

    assert exc_info.value.code == JsonRpcErrorCode.INVALID_PARAMS
    assert exc_info.value.message == message


def test_valid_params_return_every_parsed_field() -> None:
    params = {
        "message": _message(contextId="thread-1", taskId="thread-1:run-1"),
        "context": {"tenant": "acme"},
    }

    assert parse_send_params(params) == SendParams(
        message_id="msg-1",
        role="ROLE_USER",
        parts=[{"text": "hi"}],
        context_id="thread-1",
        task_id="thread-1:run-1",
        context={"tenant": "acme"},
    )


@pytest.mark.parametrize(
    "extra",
    [
        pytest.param({}, id="no-configuration"),
        pytest.param({"configuration": {}}, id="modern-sdk-default"),
        pytest.param({"configuration": {"blocking": True}}, id="legacy-sdk-default"),
    ],
)
def test_optional_fields_default_to_none(extra: dict[str, Any]) -> None:
    """Both SDKs omit empty fields, so their absence must never be an error."""
    parsed = parse_send_params({"message": _message(), **extra})

    assert (parsed.context_id, parsed.task_id, parsed.context) == (None, None, None)


def test_empty_ids_count_as_absent() -> None:
    """Both SDKs drop empty strings from the wire, so one that does arrive means "not given"."""
    parsed = parse_send_params({"message": _message(contextId="", taskId="")})

    assert (parsed.context_id, parsed.task_id) == (None, None)


def test_a_context_id_does_not_have_to_be_a_uuid() -> None:
    assert parse_send_params({"message": _message(contextId="x" * 255)}).context_id == "x" * 255


def test_history_length_zero_is_accepted_because_no_history_is_what_it_asks_for() -> None:
    assert parse_send_params({"message": _message(), "configuration": {"historyLength": 0}}).message_id == "msg-1"


def test_unknown_role_passes_through_unvalidated() -> None:
    """The spec treats any non-user role as an agent message; rejecting it would diverge from the platform."""
    assert parse_send_params({"message": _message(role="wizard")}).role == "wizard"


@pytest.mark.parametrize(
    "configuration",
    [
        pytest.param({"blocking": False}, id="legacy-blocking-false"),
        pytest.param({"returnImmediately": True}, id="modern-return-immediately"),
    ],
)
def test_non_blocking_send_is_rejected_in_either_polarity(configuration: dict[str, Any]) -> None:
    """The send path always blocks; blocking silently when asked not to is the bug this guards against."""
    with pytest.raises(JsonRpcError) as exc_info:
        parse_send_params({"message": _message(), "configuration": configuration})

    assert exc_info.value.code == JsonRpcErrorCode.INVALID_PARAMS
    assert "Non-blocking send is not supported" in exc_info.value.message


@pytest.mark.parametrize(
    ("configuration", "fragment"),
    [
        pytest.param({"pushNotificationConfig": {"url": "https://example.com/hook"}}, "Push notifications", id="push"),
        pytest.param(
            {"taskPushNotificationConfig": {"url": "https://example.com/hook"}}, "Push notifications", id="push-modern"
        ),
        pytest.param({"acceptedOutputModes": ["image/png"]}, "acceptedOutputModes", id="mode-not-offered"),
        pytest.param({"acceptedOutputModes": ["text/plain", "image/png"]}, "acceptedOutputModes", id="one-bad-mode"),
        pytest.param({"acceptedOutputModes": "text/plain"}, "acceptedOutputModes", id="modes-not-a-list"),
    ],
)
def test_configuration_the_card_does_not_advertise_is_rejected(configuration: dict[str, Any], fragment: str) -> None:
    with pytest.raises(JsonRpcError) as exc_info:
        parse_send_params({"message": _message(), "configuration": configuration})

    assert exc_info.value.code == JsonRpcErrorCode.INVALID_PARAMS
    assert fragment in exc_info.value.message


@pytest.mark.parametrize(
    "configuration",
    [
        pytest.param({"blocking": True}, id="blocking-default"),
        pytest.param({"returnImmediately": False}, id="return-immediately-default"),
        pytest.param({"pushNotificationConfig": None}, id="push-null"),
        pytest.param({"taskPushNotificationConfig": None}, id="push-modern-null"),
        pytest.param({"acceptedOutputModes": ["text/plain"]}, id="one-offered-mode"),
        pytest.param({"acceptedOutputModes": ["application/json", "text/plain"]}, id="both-offered-modes"),
        pytest.param({"acceptedOutputModes": []}, id="no-mode-preference"),
    ],
)
def test_configuration_at_its_defaults_is_accepted(configuration: dict[str, Any]) -> None:
    assert parse_send_params({"message": _message(), "configuration": configuration}).message_id == "msg-1"

"""Params validation for A2A ``SendMessage`` and ``message/send``. The wire contract is issue #625."""

from dataclasses import dataclass
from typing import Any, Final

from aegra_api.services.a2a.jsonrpc import JsonRpcError, JsonRpcErrorCode
from aegra_api.services.a2a.task import parse_optional_id, validate_history_length

# What the agent card advertises in defaultOutputModes.
SUPPORTED_OUTPUT_MODES: Final[tuple[str, ...]] = ("text/plain", "application/json")


@dataclass(frozen=True)
class SendParams:
    message_id: str
    role: str
    parts: list[Any]
    context_id: str | None
    task_id: str | None
    context: dict[str, Any] | None


def parse_send_params(params: Any) -> SendParams:
    if not isinstance(params, dict):
        raise JsonRpcError(JsonRpcErrorCode.INVALID_PARAMS, "Invalid params: must be an object")

    message = params.get("message")
    if not isinstance(message, dict) or not message:
        raise JsonRpcError(JsonRpcErrorCode.INVALID_PARAMS, "Missing or invalid 'message' in params")

    message_id = message.get("messageId")
    if not isinstance(message_id, str) or not message_id:
        raise JsonRpcError(JsonRpcErrorCode.INVALID_PARAMS, "Missing required 'messageId' in message")

    role = message.get("role")
    if not isinstance(role, str) or not role:
        raise JsonRpcError(JsonRpcErrorCode.INVALID_PARAMS, "Missing required 'role' in message")

    parts = message.get("parts")
    if not isinstance(parts, list):
        raise JsonRpcError(JsonRpcErrorCode.INVALID_PARAMS, "Invalid params: 'parts' must be an array")
    if not parts:
        raise JsonRpcError(JsonRpcErrorCode.INVALID_PARAMS, "Message must contain at least one part")

    configuration = params.get("configuration")
    if configuration is not None and not isinstance(configuration, dict):
        raise JsonRpcError(JsonRpcErrorCode.INVALID_PARAMS, "Invalid params: 'configuration' must be an object")

    if configuration:
        validate_history_length(configuration.get("historyLength"))

        # Legacy clients say ``blocking: false``; modern ones say ``returnImmediately: true``.
        if configuration.get("blocking") is False or configuration.get("returnImmediately") is True:
            raise JsonRpcError(
                JsonRpcErrorCode.INVALID_PARAMS,
                "Non-blocking send is not supported: leave 'blocking' and 'returnImmediately' at their defaults",
            )

        # Legacy clients name it ``pushNotificationConfig``; modern ones ``taskPushNotificationConfig``.
        if (
            configuration.get("pushNotificationConfig") is not None
            or configuration.get("taskPushNotificationConfig") is not None
        ):
            raise JsonRpcError(JsonRpcErrorCode.INVALID_PARAMS, "Push notifications are not supported")

        output_modes = configuration.get("acceptedOutputModes")
        if output_modes is not None and (
            not isinstance(output_modes, list) or any(mode not in SUPPORTED_OUTPUT_MODES for mode in output_modes)
        ):
            raise JsonRpcError(
                JsonRpcErrorCode.INVALID_PARAMS,
                f"Unsupported 'acceptedOutputModes': only {', '.join(SUPPORTED_OUTPUT_MODES)} are available",
            )

    # Resume is not implemented, and running a plain turn in its place would drop the client's answer.
    if message.get("command") is not None:
        raise JsonRpcError(
            JsonRpcErrorCode.INVALID_PARAMS,
            "'command' is not supported: resuming an interrupted task over A2A is not available yet",
        )

    context = params.get("context")
    if context is not None and not isinstance(context, dict):
        raise JsonRpcError(JsonRpcErrorCode.INVALID_PARAMS, "Invalid params: 'context' must be an object")

    return SendParams(
        message_id=message_id,
        role=role,
        parts=parts,
        context_id=parse_optional_id(message.get("contextId"), "contextId"),
        task_id=parse_optional_id(message.get("taskId"), "taskId"),
        context=context,
    )

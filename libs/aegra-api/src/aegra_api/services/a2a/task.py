"""A2A task responses in both dialects. The wire contract is issue #625."""

from datetime import UTC, datetime
from typing import Any, Final, Literal
from uuid import uuid4

from aegra_api.models.entity_ids import MAX_ENTITY_ID_LENGTH
from aegra_api.services.a2a.jsonrpc import Dialect, JsonRpcError, JsonRpcErrorCode

TaskState = Literal[
    "SUBMITTED", "WORKING", "COMPLETED", "FAILED", "CANCELED", "INPUT_REQUIRED", "REJECTED", "AUTH_REQUIRED"
]

RUN_STATUS_TO_TASK_STATE: Final[dict[str, TaskState]] = {
    "pending": "SUBMITTED",
    "running": "WORKING",
    "success": "COMPLETED",
    "interrupted": "INPUT_REQUIRED",
    "error": "FAILED",
    "timeout": "FAILED",
}


def parse_optional_id(value: Any, field: str) -> str | None:
    """A client-supplied id, or None when absent. Both SDKs omit empty ids, so ``""`` counts as absent."""
    if value is None or value == "":
        return None
    if not isinstance(value, str) or not value.strip() or len(value) > MAX_ENTITY_ID_LENGTH:
        raise JsonRpcError(
            JsonRpcErrorCode.INVALID_PARAMS,
            f"'{field}' must be a non-blank string of at most {MAX_ENTITY_ID_LENGTH} characters",
        )
    return value


def validate_history_length(value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise JsonRpcError(JsonRpcErrorCode.INVALID_PARAMS, "historyLength must be a non-negative integer")
    if value > 10:
        raise JsonRpcError(JsonRpcErrorCode.INVALID_PARAMS, "historyLength cannot exceed 10")
    # Responses carry no history yet, which is only what the client asked for when it sent 0.
    if value != 0:
        raise JsonRpcError(
            JsonRpcErrorCode.INVALID_PARAMS,
            "Task history is not available yet: omit 'historyLength' or set it to 0",
        )


def format_task_id(context_id: str, run_id: str) -> str:
    return f"{context_id}:{run_id}"


def parse_task_id(task_id: str, context_id: str | None) -> tuple[str, str]:
    embedded_context_id, separator, run_id = task_id.partition(":")
    if not separator:
        embedded_context_id, run_id = "", task_id

    resolved_context_id = context_id or embedded_context_id
    if not resolved_context_id:
        raise JsonRpcError(JsonRpcErrorCode.TASK_NOT_FOUND, f"Task not found: {task_id}")
    # No thread or run id is longer than this, so an oversized one cannot name a task.
    if len(resolved_context_id) > MAX_ENTITY_ID_LENGTH or len(run_id) > MAX_ENTITY_ID_LENGTH:
        raise JsonRpcError(JsonRpcErrorCode.TASK_NOT_FOUND, "Task not found: task id is too long")
    return resolved_context_id, run_id


def task_state(state: TaskState, dialect: Dialect) -> str:
    if dialect == "legacy":
        return state.lower().replace("_", "-")
    return f"TASK_STATE_{state}"


def agent_role(dialect: Dialect) -> str:
    return "agent" if dialect == "legacy" else "ROLE_AGENT"


def _tagged(obj: dict[str, Any], kind: str, dialect: Dialect) -> dict[str, Any]:
    # Legacy clients require the ``kind`` discriminator; modern ones reject it as an unknown field.
    return {"kind": kind, **obj} if dialect == "legacy" else obj


def text_part(text: str, dialect: Dialect) -> dict[str, Any]:
    return _tagged({"text": text}, "text", dialect)


def data_part(data: dict[str, Any], dialect: Dialect) -> dict[str, Any]:
    return _tagged({"data": data}, "data", dialect)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: list[str] = []
        for item in content:
            if isinstance(item, str):
                pieces.append(item)
            elif isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                pieces.append(item["text"])
        return "".join(pieces)
    if content is None:
        return ""
    return str(content)


def reply_text(output: dict[str, Any]) -> str:
    """The agent's latest non-empty message, or the whole output when the graph has no ``messages`` list."""
    messages = output.get("messages")
    if not isinstance(messages, list):
        return str(output)

    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        is_agent = message.get("type") == "ai" or message.get("role") in ("assistant", "agent")
        if not is_agent:
            continue
        # A tool-calling turn has no text, so keep looking for the last one that does.
        text = _content_text(message.get("content"))
        if text:
            return text

    return ""


def response_artifact(output: dict[str, Any], assistant_id: str, dialect: Dialect) -> dict[str, Any]:
    return {
        "artifactId": str(uuid4()),
        "name": "Assistant Response",
        "description": f"Response from assistant {assistant_id}",
        "parts": [text_part(reply_text(output), dialect)],
    }


def interrupt_artifact(output: dict[str, Any], dialect: Dialect) -> dict[str, Any]:
    interrupts = output.get("__interrupt__") or []
    parts = [
        data_part({"id": interrupt.get("id"), "value": interrupt.get("value")}, dialect)
        for interrupt in interrupts
        if isinstance(interrupt, dict)
    ]
    return {
        "artifactId": str(uuid4()),
        "name": "Interrupt",
        "description": "Agent requires input to continue",
        "parts": parts,
    }


def build_task(
    output: dict[str, Any], *, context_id: str, task_id: str, assistant_id: str, dialect: Dialect
) -> dict[str, Any]:
    task: dict[str, Any] = {"id": task_id, "contextId": context_id}
    timestamp = datetime.now(UTC).isoformat()

    error = output.get("__error__")
    if error:
        reason = error.get("error") if isinstance(error, dict) else error
        message = {
            "role": agent_role(dialect),
            "parts": [text_part(f"Error executing assistant: {reason}", dialect)],
            "messageId": str(uuid4()),
            "taskId": task_id,
            "contextId": context_id,
        }
        task["status"] = {"state": task_state("FAILED", dialect), "message": _tagged(message, "message", dialect)}
    elif output.get("__interrupt__"):
        task["status"] = {"state": task_state("INPUT_REQUIRED", dialect), "timestamp": timestamp}
        task["artifacts"] = [interrupt_artifact(output, dialect)]
    else:
        task["status"] = {"state": task_state("COMPLETED", dialect), "timestamp": timestamp}
        task["artifacts"] = [response_artifact(output, assistant_id, dialect)]

    return _tagged(task, "task", dialect)


def build_status_task(
    *, task_id: str, context_id: str, state: TaskState, message_text: str | None, dialect: Dialect
) -> dict[str, Any]:
    status: dict[str, Any] = {"state": task_state(state, dialect)}
    if message_text is not None:
        message = {
            "role": agent_role(dialect),
            "parts": [text_part(message_text, dialect)],
            "messageId": str(uuid4()),
            "taskId": task_id,
        }
        status["message"] = _tagged(message, "message", dialect)

    return _tagged({"id": task_id, "contextId": context_id, "status": status}, "task", dialect)

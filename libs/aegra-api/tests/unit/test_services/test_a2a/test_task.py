"""Tests for A2A task responses."""

from datetime import datetime
from typing import Any
from uuid import UUID

import pytest

from aegra_api.services.a2a.jsonrpc import Dialect, JsonRpcError, JsonRpcErrorCode
from aegra_api.services.a2a.task import (
    RUN_STATUS_TO_TASK_STATE,
    TaskState,
    build_status_task,
    build_task,
    data_part,
    format_task_id,
    interrupt_artifact,
    parse_optional_id,
    parse_task_id,
    reply_text,
    response_artifact,
    task_state,
    text_part,
    validate_history_length,
)

COMPLETED_OUTPUT: dict[str, Any] = {
    "messages": [{"type": "human", "content": "hello"}, {"type": "ai", "content": "Hi there!"}]
}


def _ai(content: Any) -> dict[str, Any]:
    return {"type": "ai", "content": content}


def _task(output: dict[str, Any], dialect: Dialect) -> dict[str, Any]:
    return build_task(output, context_id="thread-1", task_id="thread-1:run-1", assistant_id="agent", dialect=dialect)


def test_task_id_joins_context_and_run_with_a_colon() -> None:
    assert format_task_id("thread-1", "run-1") == "thread-1:run-1"


@pytest.mark.parametrize(
    ("state", "dialect", "expected"),
    [
        ("COMPLETED", "modern", "TASK_STATE_COMPLETED"),
        ("COMPLETED", "legacy", "completed"),
        ("FAILED", "legacy", "failed"),
        ("INPUT_REQUIRED", "modern", "TASK_STATE_INPUT_REQUIRED"),
        ("INPUT_REQUIRED", "legacy", "input-required"),
    ],
)
def test_task_state_is_spelled_per_dialect(state: TaskState, dialect: Dialect, expected: str) -> None:
    """Legacy hyphenates where modern underscores; each client rejects the other's spelling."""
    assert task_state(state, dialect) == expected


def test_parts_carry_kind_in_legacy_only() -> None:
    assert text_part("hi", "modern") == {"text": "hi"}
    assert text_part("hi", "legacy") == {"kind": "text", "text": "hi"}
    assert data_part({"id": 1}, "modern") == {"data": {"id": 1}}
    assert data_part({"id": 1}, "legacy") == {"kind": "data", "data": {"id": 1}}


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        pytest.param(COMPLETED_OUTPUT, "Hi there!", id="last-ai-message"),
        pytest.param(
            {"messages": [_ai("first"), {"type": "human", "content": "more"}, _ai("second")]},
            "second",
            id="latest-wins",
        ),
        pytest.param({"messages": [_ai("answer"), _ai("")]}, "answer", id="skips-empty-tool-call-turn"),
        pytest.param(
            {"messages": [_ai("answer"), {"type": "tool", "content": "tool output"}]}, "answer", id="skips-tool-message"
        ),
        pytest.param(
            {"messages": [_ai([{"type": "text", "text": "Here is "}, {"type": "tool_use", "id": "x"}, "the answer."])]},
            "Here is the answer.",
            id="joins-text-blocks-without-separator",
        ),
        pytest.param({"messages": [{"role": "assistant", "content": "by role"}]}, "by role", id="agent-by-role"),
        pytest.param({"messages": ["junk", None, _ai("fine")]}, "fine", id="ignores-non-dict-entries"),
        pytest.param({"messages": [{"type": "human", "content": "hi"}]}, "", id="no-agent-message"),
        pytest.param({"city": "Paris"}, "{'city': 'Paris'}", id="no-messages-key"),
        pytest.param({"messages": "oops"}, "{'messages': 'oops'}", id="messages-not-a-list"),
    ],
)
def test_reply_text_is_the_latest_agent_message_with_text(output: dict[str, Any], expected: str) -> None:
    assert reply_text(output) == expected


def test_response_artifact_has_the_exact_literals_clients_match_on() -> None:
    artifact = response_artifact(COMPLETED_OUTPUT, "agent", "modern")

    assert UUID(artifact.pop("artifactId")).version == 4
    assert artifact == {
        "name": "Assistant Response",
        "description": "Response from assistant agent",
        "parts": [{"text": "Hi there!"}],
    }


def test_interrupt_artifact_forwards_only_id_and_value_per_interrupt() -> None:
    output = {"__interrupt__": [{"id": "a", "value": "Approve?", "resumable": True}, {"value": {"q": 2}}]}

    artifact = interrupt_artifact(output, "legacy")

    assert UUID(artifact.pop("artifactId")).version == 4
    assert artifact == {
        "name": "Interrupt",
        "description": "Agent requires input to continue",
        "parts": [
            {"kind": "data", "data": {"id": "a", "value": "Approve?"}},
            {"kind": "data", "data": {"id": None, "value": {"q": 2}}},
        ],
    }


def test_completed_task_in_modern_has_no_kind_and_only_spec_keys() -> None:
    """Modern clients parse strictly, so an extra key here would break every one of them."""
    task = _task(COMPLETED_OUTPUT, "modern")

    assert list(task) == ["id", "contextId", "status", "artifacts"]
    assert (task["id"], task["contextId"]) == ("thread-1:run-1", "thread-1")
    assert list(task["status"]) == ["state", "timestamp"]
    assert task["status"]["state"] == "TASK_STATE_COMPLETED"
    assert datetime.fromisoformat(task["status"]["timestamp"]).tzinfo is not None
    assert [artifact["name"] for artifact in task["artifacts"]] == ["Assistant Response"]


def test_completed_task_in_legacy_is_tagged_and_lowercase() -> None:
    task = _task(COMPLETED_OUTPUT, "legacy")

    assert list(task) == ["kind", "id", "contextId", "status", "artifacts"]
    assert task["kind"] == "task"
    assert task["status"]["state"] == "completed"
    assert task["artifacts"][0]["parts"] == [{"kind": "text", "text": "Hi there!"}]


@pytest.mark.parametrize(("dialect", "state"), [("modern", "TASK_STATE_INPUT_REQUIRED"), ("legacy", "input-required")])
def test_interrupted_run_becomes_input_required_with_the_interrupt_artifact(dialect: Dialect, state: str) -> None:
    task = _task({"messages": [_ai("checking")], "__interrupt__": [{"id": "a", "value": "Approve?"}]}, dialect)

    assert task["status"]["state"] == state
    assert "timestamp" in task["status"]
    assert [artifact["name"] for artifact in task["artifacts"]] == ["Interrupt"]


def test_failed_run_has_a_status_message_and_neither_timestamp_nor_artifacts() -> None:
    """Only the error's type name is forwarded, so upstream detail never reaches the client."""
    task = _task({"__error__": {"error": "TimeoutError", "message": "secret upstream detail"}}, "modern")

    assert list(task) == ["id", "contextId", "status"]
    assert list(task["status"]) == ["state", "message"]
    assert task["status"]["state"] == "TASK_STATE_FAILED"
    message = task["status"]["message"]
    assert UUID(message.pop("messageId")).version == 4
    assert message == {
        "role": "ROLE_AGENT",
        "parts": [{"text": "Error executing assistant: TimeoutError"}],
        "taskId": "thread-1:run-1",
        "contextId": "thread-1",
    }


def test_failed_run_message_is_tagged_in_legacy() -> None:
    message = _task({"__error__": {"error": "Error"}}, "legacy")["status"]["message"]

    assert (message["kind"], message["role"]) == ("message", "agent")
    assert message["parts"] == [{"kind": "text", "text": "Error executing assistant: Error"}]


def test_error_takes_precedence_over_an_interrupt() -> None:
    task = _task({"__error__": {"error": "Error"}, "__interrupt__": [{"id": "a", "value": 1}]}, "modern")

    assert task["status"]["state"] == "TASK_STATE_FAILED"
    assert "artifacts" not in task


@pytest.mark.parametrize(
    ("task_id", "context_id", "expected"),
    [
        pytest.param("thread-1:run-1", None, ("thread-1", "run-1"), id="composite"),
        pytest.param("a:b:c", None, ("a", "b:c"), id="splits-on-first-colon-only"),
        pytest.param("run-1", "thread-1", ("thread-1", "run-1"), id="bare-run-id-with-context"),
        pytest.param("thread-1:run-1", "other", ("other", "run-1"), id="explicit-context-wins"),
        pytest.param("thread-1:run-1", "", ("thread-1", "run-1"), id="empty-context-is-absent"),
    ],
)
def test_task_id_is_parsed_into_context_and_run(
    task_id: str, context_id: str | None, expected: tuple[str, str]
) -> None:
    assert parse_task_id(task_id, context_id) == expected


def test_parsing_reverses_formatting() -> None:
    assert parse_task_id(format_task_id("thread-1", "run-1"), None) == ("thread-1", "run-1")


@pytest.mark.parametrize("task_id", ["run-1", ":run-1", ""])
def test_task_id_without_any_context_is_not_found(task_id: str) -> None:
    with pytest.raises(JsonRpcError) as exc_info:
        parse_task_id(task_id, None)

    assert exc_info.value.code == JsonRpcErrorCode.TASK_NOT_FOUND
    assert exc_info.value.message == f"Task not found: {task_id}"


def test_run_statuses_map_onto_the_spec_states() -> None:
    assert RUN_STATUS_TO_TASK_STATE == {
        "pending": "SUBMITTED",
        "running": "WORKING",
        "success": "COMPLETED",
        "interrupted": "INPUT_REQUIRED",
        "error": "FAILED",
        "timeout": "FAILED",
    }


def test_status_task_without_a_message_holds_only_the_state() -> None:
    task = build_status_task(
        task_id="thread-1:run-1", context_id="thread-1", state="WORKING", message_text=None, dialect="modern"
    )

    assert task == {"id": "thread-1:run-1", "contextId": "thread-1", "status": {"state": "TASK_STATE_WORKING"}}


def test_status_task_message_has_no_context_id_and_the_task_no_artifacts_or_timestamp() -> None:
    """These omissions are what set a polled or canceled task apart from a send response."""
    task = build_status_task(
        task_id="thread-1:run-1",
        context_id="thread-1",
        state="COMPLETED",
        message_text="Task completed successfully",
        dialect="modern",
    )

    assert list(task) == ["id", "contextId", "status"]
    assert list(task["status"]) == ["state", "message"]
    message = task["status"]["message"]
    assert UUID(message.pop("messageId")).version == 4
    assert message == {
        "role": "ROLE_AGENT",
        "parts": [{"text": "Task completed successfully"}],
        "taskId": "thread-1:run-1",
    }


def test_status_task_is_tagged_at_every_level_in_legacy() -> None:
    task = build_status_task(
        task_id="thread-1:run-1",
        context_id="thread-1",
        state="CANCELED",
        message_text="Task was canceled",
        dialect="legacy",
    )

    assert (task["kind"], task["status"]["state"]) == ("task", "canceled")
    assert (task["status"]["message"]["kind"], task["status"]["message"]["role"]) == ("message", "agent")
    assert task["status"]["message"]["parts"] == [{"kind": "text", "text": "Task was canceled"}]


@pytest.mark.parametrize("task_id", ["x" * 256 + ":run-1", "thread-1:" + "x" * 256])
def test_task_id_with_an_oversized_half_is_not_found(task_id: str) -> None:
    """No thread or run id is that long, so the lookup is skipped and the id is not echoed back."""
    with pytest.raises(JsonRpcError) as exc_info:
        parse_task_id(task_id, None)

    assert exc_info.value.code == JsonRpcErrorCode.TASK_NOT_FOUND
    assert exc_info.value.message == "Task not found: task id is too long"


@pytest.mark.parametrize("value", [None, ""])
def test_optional_id_is_none_when_absent_or_empty(value: str | None) -> None:
    assert parse_optional_id(value, "contextId") is None


@pytest.mark.parametrize("value", [5, True, {"a": 1}, "   ", "x" * 256])
def test_optional_id_rejects_anything_but_a_bounded_non_blank_string(value: Any) -> None:
    with pytest.raises(JsonRpcError) as exc_info:
        parse_optional_id(value, "contextId")

    assert exc_info.value.code == JsonRpcErrorCode.INVALID_PARAMS
    assert exc_info.value.message == "'contextId' must be a non-blank string of at most 255 characters"


@pytest.mark.parametrize("value", [None, 0])
def test_history_length_absent_or_zero_is_accepted(value: int | None) -> None:
    validate_history_length(value)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        pytest.param("abc", "historyLength must be a non-negative integer", id="string"),
        pytest.param(True, "historyLength must be a non-negative integer", id="bool"),
        pytest.param(-1, "historyLength must be a non-negative integer", id="negative"),
        pytest.param(99, "historyLength cannot exceed 10", id="too-large"),
        pytest.param(5, "Task history is not available yet: omit 'historyLength' or set it to 0", id="requested"),
    ],
)
def test_history_length_that_cannot_be_honoured_is_invalid_params(value: Any, message: str) -> None:
    with pytest.raises(JsonRpcError) as exc_info:
        validate_history_length(value)

    assert exc_info.value.code == JsonRpcErrorCode.INVALID_PARAMS
    assert exc_info.value.message == message

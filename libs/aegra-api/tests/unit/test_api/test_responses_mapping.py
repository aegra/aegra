"""Unit tests for Open Responses request and output mapping."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from aegra_api.models.responses import ConversationCreate, ResponseCreate, ResponsesError
from aegra_api.services.responses_mapping import (
    bind_resume,
    conversation_id_of,
    decode_resume,
    is_ephemeral,
    prepare_turn,
    project_conversation_items,
    project_response,
    project_response_output,
    response_status,
    retains_stateless_thread,
    threads_conflict,
)


def _create(**overrides: object) -> ResponseCreate:
    payload: dict[str, object] = {"model": "agent", "input": "Hello"}
    payload.update(overrides)
    return ResponseCreate.model_validate(payload)


def test_string_input_becomes_a_user_message_with_instructions() -> None:
    """A string input is one user message, and instructions lead as a system message."""
    turn = prepare_turn(_create(instructions="Be brief"))

    assert turn.messages == {
        "messages": [
            {"role": "system", "content": "Be brief"},
            {"role": "user", "content": "Hello"},
        ]
    }
    assert turn.resume is None
    assert turn.rejected is False


def test_message_items_keep_user_assistant_and_system_roles() -> None:
    """Role-bearing input items become graph messages."""
    turn = prepare_turn(
        _create(
            input=[
                {"type": "message", "role": "system", "content": "Rules"},
                {"role": "user", "content": [{"type": "input_text", "text": "Hi"}]},
            ]
        )
    )

    assert turn.messages == {
        "messages": [
            {"role": "system", "content": "Rules"},
            {"role": "user", "content": "Hi"},
        ]
    }


def test_image_input_is_rejected() -> None:
    """Image parts would be dropped, so they are a 422 instead."""
    body = _create(
        input=[{"role": "user", "content": [{"type": "input_image", "image_url": "https://example.com/a.png"}]}]
    )

    with pytest.raises(ResponsesError, match="input_image"):
        prepare_turn(body)


def test_approval_true_resumes_with_a_hitl_decision() -> None:
    """approve=true maps to the LangChain human-in-the-loop decision."""
    turn = prepare_turn(
        _create(
            conversation="thread-1",
            input=[{"type": "mcp_approval_response", "approve": True, "approval_request_id": "int-1"}],
        )
    )

    assert turn.resume == {"decisions": [{"type": "approve"}]}
    assert turn.interrupt_id == "int-1"
    assert turn.rejected is False


def test_approval_false_does_not_resume() -> None:
    """approve=false records a rejection and leaves the checkpoint alone."""
    turn = prepare_turn(
        _create(
            previous_response_id="run-1",
            input=[
                {
                    "type": "mcp_approval_response",
                    "approve": False,
                    "approval_request_id": "int-1",
                    "reason": "no",
                }
            ],
        )
    )

    assert turn.rejected is True
    assert turn.reject_reason == "no"
    assert turn.resume is None


def test_function_call_output_is_the_resume_payload() -> None:
    """A paired function_call_output carries an arbitrary Command(resume=...) value."""
    turn = prepare_turn(
        _create(
            conversation={"id": "thread-1"},
            input=[{"type": "function_call_output", "call_id": "int-1", "output": '[{"type": "accept"}]'}],
        )
    )

    assert turn.resume == [{"type": "accept"}]


def test_resume_items_require_an_id() -> None:
    """The approval id and call id select the interrupt, so they cannot be omitted."""
    missing_approval = _create(
        conversation="thread-1",
        input=[{"type": "mcp_approval_response", "approve": True}],
    )
    missing_call = _create(
        conversation="thread-1",
        input=[{"type": "function_call_output", "output": "{}"}],
    )

    with pytest.raises(ResponsesError, match="approval_request_id"):
        prepare_turn(missing_approval)
    with pytest.raises(ResponsesError, match="call_id"):
        prepare_turn(missing_call)


def test_function_call_output_requires_json() -> None:
    """A plain string is not a resume value."""
    body = _create(
        conversation="thread-1",
        input=[{"type": "function_call_output", "call_id": "int-1", "output": "not-json"}],
    )

    with pytest.raises(ResponsesError, match="JSON"):
        prepare_turn(body)


def test_approval_without_a_thread_is_rejected() -> None:
    """A resume has to name the conversation or the previous response."""
    body = _create(input=[{"type": "mcp_approval_response", "approve": True}])

    with pytest.raises(ResponsesError, match="conversation or previous_response_id"):
        prepare_turn(body)


def test_unsupported_request_fields_are_rejected() -> None:
    """Non-default sampling and tool fields are 422, while null SDK defaults are kept."""
    with pytest.raises(ValidationError, match="temperature"):
        _create(temperature=0.2)
    with pytest.raises(ValidationError, match="tools"):
        _create(tools=[{"type": "function", "name": "search"}])
    with pytest.raises(ValidationError, match="background=true requires store=true"):
        _create(background=True, store=False)

    accepted = _create(tools=[], tool_choice="auto", temperature=None, service_tier=None)
    assert accepted.model == "agent"


def test_conversation_items_cannot_be_seeded() -> None:
    """Creating a conversation with items would need a graph write this surface does not do."""
    with pytest.raises(ValidationError, match="seeding"):
        ConversationCreate.model_validate({"items": [{"role": "user", "content": "Hi"}]})


def test_conversation_id_accepts_a_string_or_object() -> None:
    """Both shapes the Responses spec allows are read."""
    assert conversation_id_of("thread-1") == "thread-1"
    assert conversation_id_of({"id": "thread-2"}) == "thread-2"
    with pytest.raises(ResponsesError):
        conversation_id_of({"name": "missing"})


def test_omitted_store_keeps_the_response() -> None:
    """The OpenAI SDK omits store, and that call must remain available for a later turn."""
    body = _create()
    assert body.store is True
    assert is_ephemeral(store=body.store, conversation_id=None, previous_response_id=None) is False


def test_ephemeral_and_thread_conflict_helpers() -> None:
    """Stateless unstored calls are ephemeral, and mismatched ids conflict."""
    assert is_ephemeral(store=False, conversation_id=None, previous_response_id=None) is True
    assert is_ephemeral(store=True, conversation_id=None, previous_response_id=None) is False
    assert is_ephemeral(store=False, conversation_id="thread-1", previous_response_id=None) is False
    assert threads_conflict("thread-1", "thread-2") is True
    assert threads_conflict("thread-1", "thread-1") is False
    assert threads_conflict(None, "thread-1") is False


@pytest.mark.parametrize(
    ("run_status", "output", "expected"),
    [
        ("pending", None, "in_progress"),
        ("running", {}, "in_progress"),
        ("success", {"messages": []}, "completed"),
        ("error", {}, "failed"),
        ("timeout", {}, "failed"),
        ("interrupted", {"__interrupt__": [{"id": "int-1", "value": {"name": "search"}}]}, "incomplete"),
        ("interrupted", {}, "cancelled"),
    ],
)
def test_response_status_mapping(run_status: str, output: dict | None, expected: str) -> None:
    """Run statuses land on the Responses status the client polls."""
    assert response_status(run_status, output) == expected


def test_project_response_output_skips_user_turns_and_keeps_tool_calls() -> None:
    """Response output is the assistant side of the checkpoint, plus approval requests."""
    items = project_response_output(
        {
            "messages": [
                {"id": "u1", "type": "human", "content": "Hi"},
                {
                    "id": "a1",
                    "type": "ai",
                    "content": "Done",
                    "tool_calls": [{"id": "call-1", "name": "search", "args": {"q": "aegra"}}],
                },
                {"id": "t1", "type": "tool", "tool_call_id": "call-1", "content": "result"},
            ],
            "__interrupt__": [{"id": "int-1", "value": {"action_request": {"action": "search"}}}],
        }
    )

    kinds = [item["type"] for item in items]
    assert kinds == ["message", "function_call", "function_call_output", "mcp_approval_request"]
    assert items[0]["content"][0]["text"] == "Done"
    assert items[1]["name"] == "search"
    assert items[1]["arguments"] == '{"q": "aegra"}'
    assert items[3]["id"] == "int-1"
    assert items[3]["name"] == "search"


def test_project_conversation_items_uses_openai_roles() -> None:
    """Thread messages come back with user/assistant roles, not LangChain types."""
    items = project_conversation_items(
        [
            {"id": "u1", "type": "human", "content": "Hi"},
            {"id": "a1", "type": "ai", "content": "Hello"},
        ]
    )

    assert [item["role"] for item in items] == ["user", "assistant"]
    assert items[0]["content"][0]["type"] == "input_text"
    assert items[1]["content"][0]["type"] == "output_text"


def test_project_response_failed_and_incomplete() -> None:
    """A stored error code and an interrupt both survive the projection."""
    created = datetime(2026, 1, 2, tzinfo=UTC)
    failed = project_response(
        response_id="run-1",
        run_status="error",
        output={},
        error_message="no",
        created_at=created,
        assistant_id="assistant-1",
        meta={"model": "agent", "error_code": "interrupt_rejected", "conversation_id": "thread-1"},
    )
    assert failed["status"] == "failed"
    assert failed["error"] == {"code": "interrupt_rejected", "message": "no"}
    assert failed["conversation"] == {"id": "thread-1"}
    assert failed["model"] == "agent"

    paused = project_response(
        response_id="run-2",
        run_status="interrupted",
        output={"__interrupt__": [{"id": "int-1", "value": "wait"}]},
        error_message=None,
        created_at=created,
        assistant_id=None,
        meta={"model": "agent"},
    )
    assert paused["status"] == "incomplete"
    assert paused["incomplete_details"] == {"reason": "approval_required"}
    assert paused["output"][0]["type"] == "mcp_approval_request"


def test_output_offset_drops_messages_from_earlier_turns() -> None:
    """A continued thread's checkpoint includes history; the response shows only this run."""
    output = {
        "messages": [
            {"id": "old", "type": "ai", "content": "Earlier"},
            {"id": "user", "type": "human", "content": "Next"},
            {"id": "new", "type": "ai", "content": "Now"},
        ]
    }

    items = project_response_output(output, offset=2)

    assert [item["id"] for item in items] == ["new"]
    assert items[0]["content"][0]["text"] == "Now"


def test_stateless_thread_is_kept_only_for_a_pending_interrupt() -> None:
    """A store=false interrupt has to remain so the client can send an approval."""
    interrupt = {"__interrupt__": [{"id": "int-1", "value": "wait"}]}

    assert retains_stateless_thread("interrupted", interrupt) is True
    assert retains_stateless_thread("interrupted", {}) is False
    assert retains_stateless_thread("success", interrupt) is False


def test_bind_resume_keeps_a_single_interrupt_value_and_maps_several() -> None:
    """A lone interrupt gets the value directly; several are keyed so the id cannot be dropped."""
    decision = {"decisions": [{"type": "approve"}]}

    assert bind_resume("int-1", decision, ["int-1"]) == decision
    assert bind_resume("int-2", decision, ["int-1", "int-2"]) == {"int-2": decision}
    with pytest.raises(ResponsesError, match="not pending"):
        bind_resume("int-9", decision, ["int-1"])


def test_decode_resume_passes_structured_values_through() -> None:
    """Dict and list outputs are already the resume value."""
    assert decode_resume({"decisions": [{"type": "approve"}]}) == {"decisions": [{"type": "approve"}]}
    assert decode_resume([{"type": "accept"}]) == [{"type": "accept"}]

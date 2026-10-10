"""Tests for the A2A agent card."""

from typing import Any

from aegra_api.services.a2a.card import build_agent_card


def _card(input_schema: dict[str, Any] | None = None) -> dict[str, Any]:
    return build_agent_card(
        assistant_id="asst-1",
        assistant_name="Support Agent",
        graph_id="support",
        json_rpc_url="https://example.com/a2a/asst-1",
        version="1.2.3",
        input_schema=input_schema,
    )


def test_card_has_exactly_the_spec_keys() -> None:
    """The key set decides the client dialect, so an added or dropped key must fail here."""
    assert set(_card()) == {
        "protocolVersion",
        "name",
        "description",
        "version",
        "url",
        "supportedInterfaces",
        "capabilities",
        "defaultInputModes",
        "defaultOutputModes",
        "skills",
    }


def test_card_matches_the_spec_shape() -> None:
    """The whole card for a chat graph, value for value."""
    schema = {"properties": {"messages": {"type": "array"}, "city": {"type": "string"}}, "required": ["messages"]}

    assert _card(schema) == {
        "protocolVersion": "1.0",
        "name": "Support Agent",
        "description": "LangGraph agent: support",
        "version": "1.2.3",
        "url": "https://example.com/a2a/asst-1",
        "supportedInterfaces": [
            {"protocolBinding": "JSONRPC", "protocolVersion": "1.0", "url": "https://example.com/a2a/asst-1"}
        ],
        "capabilities": {"streaming": False, "pushNotifications": False, "stateTransitionHistory": False},
        "defaultInputModes": ["text/plain", "application/json"],
        "defaultOutputModes": ["text/plain", "application/json"],
        "skills": [
            {
                "id": "asst-1-main",
                "name": "Support Agent",
                "description": "Execute the support LangGraph agent",
                "tags": ["langgraph", "agent"],
                "examples": [],
                "inputModes": ["text/plain", "application/json"],
                "outputModes": ["text/plain", "application/json"],
                "metadata": {
                    "inputSchema": {"required": ["messages"], "properties": ["city", "messages"], "supportsA2A": True}
                },
            }
        ],
    }


def test_skill_metadata_does_not_support_a2a_without_a_messages_field() -> None:
    """Text parts become ``messages``, so a graph without that field cannot take A2A chat."""
    schema = {"properties": {"query": {"type": "string"}}, "required": ["query"]}

    assert _card(schema)["skills"][0]["metadata"] == {
        "inputSchema": {"required": ["query"], "properties": ["query"], "supportsA2A": False}
    }


def test_skill_metadata_is_empty_when_there_is_no_schema() -> None:
    """A missing schema or a null properties value still yields a well-formed summary."""
    empty = {"inputSchema": {"required": [], "properties": [], "supportsA2A": False}}

    assert _card(None)["skills"][0]["metadata"] == empty
    assert _card({"properties": None})["skills"][0]["metadata"] == empty

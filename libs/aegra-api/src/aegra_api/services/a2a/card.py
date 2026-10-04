"""
The A2A agent card for one assistant.
"""

from typing import Any


def _build_input_schema(input_schema: dict[str, Any] | None) -> dict[str, Any]:
    if not input_schema:
        return {"required": [], "properties": [], "supportsA2A": False}

    properties = input_schema.get("properties") or {}
    supports_a2a: bool = "messages" in properties

    return {"required": input_schema.get("required", []), "properties": sorted(properties), "supportsA2A": supports_a2a}


def build_agent_card(
    *,
    assistant_id: str,
    assistant_name: str,
    graph_id: str,
    json_rpc_url: str,
    version: str,
    input_schema: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "protocolVersion": "1.0",
        "name": assistant_name,
        "description": f"LangGraph agent: {graph_id}",
        "version": version,
        "url": json_rpc_url,
        "supportedInterfaces": [{"protocolBinding": "JSONRPC", "protocolVersion": "1.0", "url": json_rpc_url}],
        "capabilities": {"streaming": False, "pushNotifications": False, "stateTransitionHistory": False},
        "defaultInputModes": ["text/plain", "application/json"],
        "defaultOutputModes": ["text/plain", "application/json"],
        "skills": [
            {
                "id": f"{assistant_id}-main",
                "name": assistant_name,
                "description": f"Execute the {graph_id} LangGraph agent",
                "tags": ["langgraph", "agent"],
                "examples": [],
                "inputModes": ["text/plain", "application/json"],
                "outputModes": ["text/plain", "application/json"],
                "metadata": {"inputSchema": _build_input_schema(input_schema)},
            }
        ],
    }

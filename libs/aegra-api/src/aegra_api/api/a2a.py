"""A2A (Agent2Agent) protocol endpoints.

Serves the agent card for an assistant. The wire contract is issue #625.
"""

from functools import cache
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from aegra_api import __version__
from aegra_api.config import HttpConfig, load_http_config
from aegra_api.core.auth_deps import auth_dependency
from aegra_api.services.a2a.card import build_agent_card
from aegra_api.services.assistant_service import AssistantService, get_assistant_service
from aegra_api.utils.assistants import resolve_assistant_id

router = APIRouter(tags=["A2A"], dependencies=auth_dependency)


@cache
def get_default_a2a_assistant() -> str | None:
    """The assistant served at the host-root card, from http.a2a_default_assistant; None when unset."""
    http_config: HttpConfig | None = load_http_config()
    return (http_config or {}).get("a2a_default_assistant") or None


@router.get("/.well-known/agent-card.json")
async def get_default_assistant_agent_card(
    request: Request, service: AssistantService = Depends(get_assistant_service)
) -> dict[str, Any]:
    """Get the A2A agent card for the server's default assistant (http.a2a_default_assistant)."""
    assistant_id = get_default_a2a_assistant()
    if assistant_id is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "No default A2A agent is configured for this server. Fetch an agent's card at "
                "/a2a/{assistant_id}/.well-known/agent-card.json, or set http.a2a_default_assistant "
                "in aegra.json to serve one here."
            ),
        )
    resolved_assistant_id = resolve_assistant_id(assistant_id, service.langgraph_service.list_graphs())

    assistant = await service.get_assistant(resolved_assistant_id)
    assistant_schema = await service.get_assistant_schemas(resolved_assistant_id)

    base = str(request.base_url).rstrip("/")
    json_rpc_url: str = f"{base}/a2a/{assistant_id}"

    return build_agent_card(
        assistant_id=assistant_id,
        assistant_name=assistant.name or assistant_id,
        graph_id=assistant.graph_id,
        json_rpc_url=json_rpc_url,
        version=__version__,
        input_schema=assistant_schema.get("input_schema") or assistant_schema.get("state_schema"),
    )


@router.get("/a2a/{assistant_id}")
@router.get("/a2a/{assistant_id}/.well-known/agent.json")
@router.get("/a2a/{assistant_id}/.well-known/agent-card.json")
async def get_assistant_agent_card(
    assistant_id: str, request: Request, service: AssistantService = Depends(get_assistant_service)
) -> dict[str, Any]:
    """Get the A2A agent card for an assistant, by assistant id or graph id."""
    resolved_assistant_id = resolve_assistant_id(assistant_id, service.langgraph_service.list_graphs())

    assistant = await service.get_assistant(resolved_assistant_id)
    assistant_schema = await service.get_assistant_schemas(resolved_assistant_id)

    base = str(request.base_url).rstrip("/")
    json_rpc_url: str = f"{base}/a2a/{assistant_id}"

    return build_agent_card(
        assistant_id=assistant_id,
        assistant_name=assistant.name or assistant_id,
        graph_id=assistant.graph_id,
        json_rpc_url=json_rpc_url,
        version=__version__,
        input_schema=assistant_schema.get("input_schema") or assistant_schema.get("state_schema"),
    )

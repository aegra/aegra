"""Assistant endpoints for Agent Protocol

NOTE: This API follows a layered architecture pattern with business logic
separated into a service layer (assistant_service.py). This was the first
API to be refactored, and the plan is to gradually refactor all other APIs
(runs, threads, etc.) to follow this same pattern for better code
organization, testability, and maintainability.

Architecture:
- API Layer (this file): Thin FastAPI route handlers, request/response handling
- Service Layer (assistant_service.py): Business logic, validation, orchestration
"""

from fastapi import APIRouter, Body, Depends, Query

from aegra_api.core.auth_deps import auth_dependency
from aegra_api.core.orm import Assistant as AssistantORM
from aegra_api.models import (
    AgentSchemas,
    Assistant,
    AssistantCreate,
    AssistantList,
    AssistantSearchRequest,
    AssistantShareCreate,
    AssistantShareResponse,
    AssistantUpdate,
)
from aegra_api.models.errors import NOT_FOUND
from aegra_api.services.assistant_service import AssistantService, get_assistant_service

router = APIRouter(tags=["Assistants"], dependencies=auth_dependency)


def _resolve_sort(request: AssistantSearchRequest) -> tuple[object, bool]:
    """Resolve (ORM column, is_ascending) for /assistants/search.

    sort_by is Pydantic-validated against a Literal — invalid values 422 at
    the request boundary. Default is created_at DESC.
    """
    if request.sort_by:
        return getattr(AssistantORM, request.sort_by), (request.sort_order or "desc").lower() == "asc"
    return AssistantORM.created_at, False


@router.post("/assistants", response_model=Assistant, response_model_by_alias=False)
async def create_assistant(
    request: AssistantCreate,
    service: AssistantService = Depends(get_assistant_service),
):
    """Create a new assistant.

    An assistant is a configured instance of a graph. Provide a `graph_id`
    referencing a graph defined in your `aegra.json`. If `assistant_id` is
    omitted, one is auto-generated. Set `if_exists` to `"do_nothing"` for
    idempotent creation.
    """
    return await service.create_assistant(request)


@router.get("/assistants", response_model=AssistantList, response_model_by_alias=False)
async def list_assistants(
    service: AssistantService = Depends(get_assistant_service),
):
    """List all assistants owned by the authenticated user.

    Returns every assistant without filtering. Use the search endpoint for
    filtered queries.
    """
    assistants = await service.list_assistants()
    return AssistantList(assistants=assistants, total=len(assistants))


@router.post("/assistants/search", response_model=list[Assistant], response_model_by_alias=False)
async def search_assistants(
    request: AssistantSearchRequest,
    service: AssistantService = Depends(get_assistant_service),
):
    """Search assistants with filters.

    Filter by name, description, graph ID, or metadata. Results are paginated
    via `limit` and `offset`.
    """
    column, asc = _resolve_sort(request)
    return await service.search_assistants(request, sort_column=column, sort_asc=asc)


@router.post("/assistants/count", response_model=int)
async def count_assistants(
    request: AssistantSearchRequest,
    service: AssistantService = Depends(get_assistant_service),
):
    """Count assistants matching the given filters.

    Accepts the same filter parameters as the search endpoint but returns only
    the total count.
    """
    return await service.count_assistants(request)


@router.get(
    "/assistants/{assistant_id}",
    response_model=Assistant,
    response_model_by_alias=False,
    responses={**NOT_FOUND},
)
async def get_assistant(
    assistant_id: str,
    service: AssistantService = Depends(get_assistant_service),
):
    """Get an assistant by its ID.

    Returns the latest version of the assistant. Returns 404 if the assistant
    does not exist or does not belong to the authenticated user.
    """
    return await service.get_assistant(assistant_id)


@router.patch(
    "/assistants/{assistant_id}",
    response_model=Assistant,
    response_model_by_alias=False,
    responses={**NOT_FOUND},
)
async def update_assistant(
    assistant_id: str,
    request: AssistantUpdate,
    service: AssistantService = Depends(get_assistant_service),
):
    """Update an assistant by its ID.

    Partial update: only fields included in the request body are changed.
    Creates a new version of the assistant.
    """
    return await service.update_assistant(assistant_id, request)


@router.delete("/assistants/{assistant_id}", responses={**NOT_FOUND})
async def delete_assistant(
    assistant_id: str,
    service: AssistantService = Depends(get_assistant_service),
):
    """Delete an assistant by its ID.

    Permanently removes the assistant and all of its versions. This action
    cannot be undone.
    """
    return await service.delete_assistant(assistant_id)


@router.post(
    "/assistants/{assistant_id}/latest",
    response_model=Assistant,
    response_model_by_alias=False,
    responses={**NOT_FOUND},
)
async def set_assistant_latest(
    assistant_id: str,
    version: int = Body(..., embed=True, description="The version number to set as latest"),
    service: AssistantService = Depends(get_assistant_service),
):
    """Pin a specific version as the latest version of an assistant.

    After calling this endpoint, the assistant will use the specified version's
    configuration when executing runs.
    """
    return await service.set_assistant_latest(assistant_id, version)


@router.post(
    "/assistants/{assistant_id}/versions",
    response_model=list[Assistant],
    response_model_by_alias=False,
    responses={**NOT_FOUND},
)
async def list_assistant_versions(
    assistant_id: str,
    service: AssistantService = Depends(get_assistant_service),
):
    """List all versions of an assistant.

    Returns versions ordered from newest to oldest. Each version captures the
    assistant's configuration at the time of creation or update.
    """
    return await service.list_assistant_versions(assistant_id)


@router.get(
    "/assistants/{assistant_id}/schemas",
    response_model=AgentSchemas,
    responses={**NOT_FOUND},
)
async def get_assistant_schemas(
    assistant_id: str,
    service: AssistantService = Depends(get_assistant_service),
):
    """Get the JSON schemas for an assistant's graph.

    Returns the input, output, state, and config schemas derived from the
    underlying graph's type annotations.
    """
    return await service.get_assistant_schemas(assistant_id)


@router.get("/assistants/{assistant_id}/graph", responses={**NOT_FOUND})
async def get_assistant_graph(
    assistant_id: str,
    xray: bool | int | None = Query(
        None, description="Expand subgraph nodes. Pass true or a depth integer to control nesting."
    ),
    service: AssistantService = Depends(get_assistant_service),
):
    """Get the graph structure for visualization.

    Returns a JSON representation of the graph's nodes and edges suitable for
    rendering in graph visualizers. Use `xray` to expand subgraph nodes into
    their internal structure.
    """
    xray_value = xray if xray is not None else False
    return await service.get_assistant_graph(assistant_id, xray_value)


@router.get("/assistants/{assistant_id}/subgraphs", responses={**NOT_FOUND})
async def get_assistant_subgraphs(
    assistant_id: str,
    recurse: bool = Query(False, description="Recursively include nested subgraphs."),
    namespace: str | None = Query(None, description="Filter to a specific subgraph namespace."),
    service: AssistantService = Depends(get_assistant_service),
):
    """Get subgraphs of an assistant.

    Returns the subgraph definitions used by this assistant's graph. Set
    `recurse=true` to include deeply nested subgraphs, or filter to a single
    namespace.
    """
    return await service.get_assistant_subgraphs(assistant_id, namespace, recurse)


@router.post(
    "/assistants/{assistant_id}/share",
    response_model=AssistantShareResponse,
    responses={**NOT_FOUND},
)
async def create_assistant_share(
    assistant_id: str,
    request: AssistantShareCreate,
    service: AssistantService = Depends(get_assistant_service),
):
    """Share an assistant with a specific user, a specific tenant, or fully public.

    Only the owner may share. Grantees can read (config/context secrets are redacted) and execute
    the assistant (execution reuses the owner-stored config, including real secrets).
    """
    return await service.create_share(assistant_id, request)


@router.get(
    "/assistants/{assistant_id}/shares",
    response_model=list[AssistantShareResponse],
    responses={**NOT_FOUND},
)
async def list_assistant_shares(
    assistant_id: str,
    service: AssistantService = Depends(get_assistant_service),
):
    """List all share records for this assistant. Owner only."""
    return await service.list_shares(assistant_id)


@router.delete("/assistants/{assistant_id}/shares/{share_id}", responses={**NOT_FOUND})
async def delete_assistant_share(
    assistant_id: str,
    share_id: str,
    service: AssistantService = Depends(get_assistant_service),
):
    """Delete a share record. Owner only."""
    return await service.delete_share(assistant_id, share_id)

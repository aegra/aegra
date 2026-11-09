"""Open Responses and Conversations routes.

Mounted only when ``http.responses`` is true. Errors use the OpenAI error
document; Agent Server routes keep their own error shape.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Coroutine, MutableMapping
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from redis import RedisError
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette import EventSourceResponse
from starlette.responses import Response

from aegra_api.api.runs import _request_run_interruption
from aegra_api.api.threads import delete_thread
from aegra_api.core.auth_deps import auth_dependency, get_current_user
from aegra_api.core.orm import get_session
from aegra_api.core.sse import get_sse_headers, make_sse_response, sse_to_bytes
from aegra_api.models import Run, User
from aegra_api.models.responses import (
    ConversationCreate,
    ConversationDeleted,
    ConversationItemList,
    ConversationResource,
    OpenAIErrorResponse,
    ResponseCreate,
    ResponseResource,
    ResponsesError,
    openai_error_body,
)
from aegra_api.services.broker import broker_manager
from aegra_api.services.responses_mapping import meta_from_params
from aegra_api.services.responses_service import (
    StartedRun,
    create_conversation,
    execute_response,
    iter_response_sse,
    list_conversation_items,
    load_visible_response,
    project_run,
    read_conversation,
)
from aegra_api.services.responses_stream import iter_responses_sse

logger = structlog.getLogger(__name__)


class ResponsesRoute(APIRoute):
    """Render HTTP and validation errors as OpenAI error documents."""

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        original = super().get_route_handler()

        async def handler(request: Request) -> Response:
            try:
                return await original(request)
            except ResponsesError as exc:
                return JSONResponse(status_code=exc.status_code, content=exc.body())
            except HTTPException as exc:
                return JSONResponse(
                    status_code=exc.status_code,
                    content=openai_error_body(exc.status_code, _http_message(exc.detail)),
                )
            except RequestValidationError as exc:
                return JSONResponse(status_code=422, content=openai_error_body(422, _validation_message(exc)))

        return handler


router = APIRouter(
    prefix="/v1",
    route_class=ResponsesRoute,
    dependencies=auth_dependency,
    responses={
        422: {"model": OpenAIErrorResponse, "description": "The request was rejected."},
    },
)

_NOT_FOUND = {
    404: {
        "model": OpenAIErrorResponse,
        "description": "The response, conversation, or assistant was not found.",
    }
}
_STREAM_OR_JSON = {
    200: {
        "description": "The response object, or a Server-Sent Events stream when stream is true.",
        "content": {"text/event-stream": {"schema": {"type": "string"}}},
    }
}


def _http_message(detail: Any) -> str:
    if isinstance(detail, str):
        return detail
    return "Request failed"


def _validation_message(exc: RequestValidationError) -> str:
    parts: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(item) for item in err.get("loc", ()) if item != "body")
        message = str(err.get("msg", "invalid request"))
        parts.append(f"{loc}: {message}" if loc else message)
    return "; ".join(parts) or "Invalid request"


def _cancel_on_disconnect(run_id: str) -> Callable[[MutableMapping[str, Any]], Awaitable[None]]:
    async def _cancel(_message: MutableMapping[str, Any]) -> None:
        try:
            await broker_manager.request_cancel(run_id, "cancel")
        except (RedisError, OSError):
            logger.exception("Failed to cancel response on client disconnect", run_id=run_id)

    return _cancel


@router.post(
    "/responses", response_model=ResponseResource, tags=["Responses"], responses={**_STREAM_OR_JSON, **_NOT_FOUND}
)
async def create_response(
    body: ResponseCreate,
    user: User = Depends(get_current_user),
) -> ResponseResource | EventSourceResponse:
    """Create a response. ``stream: true`` returns SSE. Mounted only when http.responses is true."""
    result = await execute_response(user, body)
    if isinstance(result, StartedRun):
        close_handler = _cancel_on_disconnect(result.run.run_id) if result.cancel_on_disconnect else None
        return make_sse_response(
            sse_to_bytes(iter_response_sse(result, starting_after=None)),
            close_handler=close_handler,
            headers=get_sse_headers(),
        )
    return ResponseResource.model_validate(result)


@router.get(
    "/responses/{response_id}",
    response_model=ResponseResource,
    tags=["Responses"],
    responses={**_STREAM_OR_JSON, **_NOT_FOUND},
)
async def get_response(
    response_id: str,
    *,
    stream: bool = Query(False, description="Reattach to the response as SSE."),
    starting_after: str | None = Query(None, description="Replay broker events after this id."),
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> ResponseResource | EventSourceResponse:
    """Read a response, or reattach to its stream. Mounted only when http.responses is true."""
    if starting_after is not None and not stream:
        raise ResponsesError(422, "starting_after requires stream=true")
    run = await load_visible_response(session, user, response_id, action="read")
    if run is None:
        raise HTTPException(404, f"Response '{response_id}' not found")
    payload = project_run(run)
    if not stream:
        return ResponseResource.model_validate(payload)
    raw_offset = meta_from_params(run.execution_params, run.assistant_id).get("message_offset")
    message_offset = raw_offset if isinstance(raw_offset, int) and raw_offset > 0 else 0
    return make_sse_response(
        sse_to_bytes(
            iter_responses_sse(
                Run.model_validate(run),
                payload,
                starting_after=starting_after,
                message_offset=message_offset,
            )
        ),
        headers=get_sse_headers(),
    )


@router.post(
    "/responses/{response_id}/cancel",
    response_model=ResponseResource,
    tags=["Responses"],
    responses={**_NOT_FOUND},
)
async def cancel_response(
    response_id: str,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> ResponseResource:
    """Cancel an in-progress response. Mounted only when http.responses is true."""
    run = await load_visible_response(session, user, response_id, action="update")
    if run is None:
        raise HTTPException(404, f"Response '{response_id}' not found")
    await _request_run_interruption(session, run, "cancel")
    await session.refresh(run)
    return ResponseResource.model_validate(project_run(run))


@router.post("/conversations", response_model=ConversationResource, tags=["Conversations"])
async def post_conversation(
    body: ConversationCreate,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> ConversationResource:
    """Create a conversation backed by a thread. Mounted only when http.responses is true."""
    return ConversationResource.model_validate(await create_conversation(session, user, body))


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationResource,
    tags=["Conversations"],
    responses={**_NOT_FOUND},
)
async def get_conversation(
    conversation_id: str,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> ConversationResource:
    """Read a conversation. Mounted only when http.responses is true."""
    return ConversationResource.model_validate(await read_conversation(session, user, conversation_id))


@router.delete(
    "/conversations/{conversation_id}",
    response_model=ConversationDeleted,
    tags=["Conversations"],
    responses={**_NOT_FOUND},
)
async def remove_conversation(
    conversation_id: str,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> ConversationDeleted:
    """Delete a conversation and its thread. Mounted only when http.responses is true."""
    await delete_thread(conversation_id, user, session)
    return ConversationDeleted(id=conversation_id)


@router.get(
    "/conversations/{conversation_id}/items",
    response_model=ConversationItemList,
    tags=["Conversations"],
    responses={**_NOT_FOUND},
)
async def get_conversation_items(
    conversation_id: str,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> ConversationItemList:
    """List checkpoint messages as conversation items. Mounted only when http.responses is true."""
    items = await list_conversation_items(session, user, conversation_id)
    return ConversationItemList(data=items)

"""A2A (Agent2Agent) protocol endpoints.

Serves the agent card for an assistant and the JSON-RPC endpoint at ``/a2a/{assistant_id}``.
The wire contract is issue #625.
"""

import asyncio
from functools import cache
from typing import Any
from uuid import uuid4

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from aegra_api import __version__
from aegra_api.api.runs import _apply_create_run_auth, _request_run_interruption
from aegra_api.config import HttpConfig, load_http_config
from aegra_api.core.auth_deps import auth_dependency, get_current_user
from aegra_api.core.auth_filters import build_metadata_filter
from aegra_api.core.auth_handlers import build_auth_context, handle_event
from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import Thread as ThreadORM
from aegra_api.core.orm import _get_session_maker
from aegra_api.models import RunCreate, User
from aegra_api.services.a2a.card import build_agent_card
from aegra_api.services.a2a.jsonrpc import (
    JsonRpcError,
    JsonRpcErrorCode,
    JsonRpcRequest,
    build_error_response,
    build_success_response,
    parse_json_rpc_request,
)
from aegra_api.services.a2a.parts import convert_parts_to_graph_input
from aegra_api.services.a2a.send import parse_send_params
from aegra_api.services.a2a.task import (
    RUN_STATUS_TO_TASK_STATE,
    TaskState,
    build_status_task,
    build_task,
    format_task_id,
    is_canceled_run,
    parse_optional_id,
    parse_task_id,
    validate_history_length,
)
from aegra_api.services.assistant_service import AssistantService, get_assistant_service
from aegra_api.services.executor import executor
from aegra_api.services.langgraph_service import get_langgraph_service
from aegra_api.services.run_preparation import _prepare_run
from aegra_api.services.run_waiters import TERMINAL_STATES, run_result_body
from aegra_api.settings import settings
from aegra_api.utils.assistants import resolve_assistant_id

logger = structlog.getLogger(__name__)

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


async def _read_run(run_id: str, thread_id: str, user_id: str) -> RunORM | None:
    maker = _get_session_maker()
    async with maker() as session:
        return await session.scalar(
            select(RunORM).where(RunORM.run_id == run_id, RunORM.thread_id == thread_id, RunORM.user_id == user_id)
        )


async def _handle_send(rpc_request: JsonRpcRequest, assistant_id: str, user: User) -> dict[str, Any]:
    send_params = parse_send_params(rpc_request.params)
    converted = convert_parts_to_graph_input(send_params.parts, send_params.message_id, send_params.role)

    thread_id = send_params.context_id or str(uuid4())
    run_create = RunCreate(assistant_id=assistant_id, input=converted.graph_input, context=send_params.context)

    maker = _get_session_maker()
    async with maker() as session:
        thread = await session.scalar(select(ThreadORM).where(ThreadORM.thread_id == thread_id))
        if thread and thread.user_id != user.identity:
            raise HTTPException(404, f"Thread '{thread_id}' not found")
        # A plain turn on a paused thread restarts the graph and discards the pending interrupt.
        if thread and thread.status == "interrupted":
            raise JsonRpcError(
                JsonRpcErrorCode.INVALID_PARAMS,
                "Task is awaiting input: resuming an interrupted task over A2A is not available yet",
            )

        # Text becomes ``messages``, so a graph without that field cannot take it; data-only input skips this.
        if converted.has_conversational_content:
            langgraph_service = get_langgraph_service()
            assistants = AssistantService(session, user, langgraph_service)
            schemas = await assistants.get_assistant_schemas(
                resolve_assistant_id(assistant_id, langgraph_service.list_graphs())
            )
            input_schema = schemas.get("input_schema") or schemas.get("state_schema")
            if not input_schema:
                raise JsonRpcError(
                    JsonRpcErrorCode.INVALID_PARAMS,
                    f"Assistant '{assistant_id}' has no input schema defined. A2A conversational agents using "
                    "text or file parts must have an input schema with a 'messages' field.",
                )
            fields = sorted(input_schema.get("properties") or {})
            if "messages" not in fields:
                raise JsonRpcError(
                    JsonRpcErrorCode.INVALID_PARAMS,
                    f"Assistant '{assistant_id}' (graph '{schemas.get('graph_id')}') does not support A2A "
                    "conversational messages. Graph input schema must include a 'messages' field to accept "
                    f"text or file parts. Available input fields: {', '.join(fields)}",
                )

        await _apply_create_run_auth(user, thread_id, run_create)
        run_id, _run, _job = await _prepare_run(session, thread_id, run_create, user, initial_status="pending")

    # The session is closed before the wait so a long run holds no pool connection.
    timed_out = False
    try:
        await executor.wait_for_completion(run_id, timeout=settings.worker.BG_JOB_TIMEOUT_SECS)
    except TimeoutError:
        timed_out = True
        logger.warning("A2A send timed out waiting for run", run_id=run_id)

    run = await _read_run(run_id, thread_id, user.identity)
    output = run_result_body(run, run_id, timed_out=timed_out)

    task_id = send_params.task_id or format_task_id(thread_id, run_id)
    # A run canceled while this send was waiting has no answer to report as completed.
    if run is not None and is_canceled_run(run.status, output):
        task = build_status_task(
            task_id=task_id,
            context_id=thread_id,
            state="CANCELED",
            message_text="Task was canceled",
            dialect=rpc_request.dialect,
        )
    else:
        task = build_task(
            output, context_id=thread_id, task_id=task_id, assistant_id=assistant_id, dialect=rpc_request.dialect
        )
    return task if rpc_request.dialect == "legacy" else {"task": task}


async def _load_task_run(
    session: AsyncSession, *, run_id: str, context_id: str, user: User, filters: dict[str, Any] | None
) -> RunORM | None:
    """The caller's run, or None when it is missing, foreign, or on a thread the auth handler's filter excludes."""
    stmt = select(RunORM).where(
        RunORM.run_id == run_id, RunORM.thread_id == context_id, RunORM.user_id == user.identity
    )
    auth_filter = build_metadata_filter(ThreadORM.metadata_json, filters)
    if auth_filter is not None:
        stmt = stmt.join(ThreadORM, ThreadORM.thread_id == RunORM.thread_id).where(auth_filter)
    return await session.scalar(stmt)


async def _handle_get_task(rpc_request: JsonRpcRequest, user: User) -> dict[str, Any]:
    params = rpc_request.params if isinstance(rpc_request.params, dict) else {}
    task_id = params.get("id")
    if not isinstance(task_id, str) or not task_id:
        raise JsonRpcError(JsonRpcErrorCode.INVALID_PARAMS, "Missing required parameter: id (task_id)")
    context_id, run_id = parse_task_id(task_id, parse_optional_id(params.get("contextId"), "contextId"))
    validate_history_length(params.get("historyLength"))

    filters = await handle_event(
        build_auth_context(user, "threads", "read"), {"run_id": run_id, "thread_id": context_id}
    )

    maker = _get_session_maker()
    async with maker() as session:
        run = await _load_task_run(session, run_id=run_id, context_id=context_id, user=user, filters=filters)
        if run is None:
            raise JsonRpcError(JsonRpcErrorCode.TASK_NOT_FOUND, f"Task '{run_id}' not found in thread '{context_id}'")
        thread_status = await session.scalar(
            select(ThreadORM.status).where(ThreadORM.thread_id == context_id, ThreadORM.user_id == user.identity)
        )

    state: TaskState = RUN_STATUS_TO_TASK_STATE.get(run.status, "SUBMITTED")
    if is_canceled_run(run.status, run.output):
        state = "CANCELED"
    # A finished run whose thread is now waiting on input reports as input-required, as the platform does.
    elif run.status == "success" and thread_status == "interrupted":
        state = "INPUT_REQUIRED"

    message_text: str | None = None
    if state == "COMPLETED":
        message_text = "Task completed successfully"
    elif state == "FAILED":
        message_text = f"Task failed with status: {run.status}"

    return build_status_task(
        task_id=task_id, context_id=context_id, state=state, message_text=message_text, dialect=rpc_request.dialect
    )


async def _handle_cancel_task(rpc_request: JsonRpcRequest, user: User) -> dict[str, Any]:
    params = rpc_request.params if isinstance(rpc_request.params, dict) else {}
    task_id = params.get("id")
    if not isinstance(task_id, str) or not task_id:
        raise JsonRpcError(JsonRpcErrorCode.INVALID_PARAMS, "Missing required parameter: id (task_id)")
    context_id, run_id = parse_task_id(task_id, parse_optional_id(params.get("contextId"), "contextId"))

    filters = await handle_event(
        build_auth_context(user, "threads", "update"), {"run_id": run_id, "thread_id": context_id}
    )

    maker = _get_session_maker()
    async with maker() as session:
        run = await _load_task_run(session, run_id=run_id, context_id=context_id, user=user, filters=filters)
        if run is None:
            raise JsonRpcError(JsonRpcErrorCode.TASK_NOT_FOUND, f"Task not found: {task_id}")

        if run.status in ("pending", "running"):
            await _request_run_interruption(session, run, "interrupt")
            # Give the run up to 10 seconds to settle, as the runs cancel endpoint does with wait=1.
            for _ in range(20):
                await asyncio.sleep(0.5)
                session.expire_all()
                status = await session.scalar(
                    select(RunORM.status).where(RunORM.run_id == run_id, RunORM.user_id == user.identity)
                )
                if status in TERMINAL_STATES:
                    break
            message_text = "Task was canceled"
        else:
            message_text = f"Task cancel acknowledged (was: {run.status})"

    return build_status_task(
        task_id=task_id, context_id=context_id, state="CANCELED", message_text=message_text, dialect=rpc_request.dialect
    )


@router.post("/a2a/{assistant_id}")
async def a2a_json_rpc(assistant_id: str, request: Request, user: User = Depends(get_current_user)) -> Response:
    """A2A JSON-RPC 2.0 endpoint for an assistant.

    Accepts `SendMessage`, `GetTask` and `CancelTask`, and their protocol 0.3
    names `message/send`, `tasks/get` and `tasks/cancel`. The request body is a
    JSON-RPC envelope; every JSON-RPC error is returned with HTTP 200 and an
    `error` object, because A2A clients read the error code from the body.
    """
    raw_body = await request.body()

    result = parse_json_rpc_request(raw_body)
    if result is None:
        return Response(status_code=202)
    if not isinstance(result, JsonRpcRequest):
        return JSONResponse(status_code=200, content=result)

    try:
        if result.operation == "send":
            payload = await _handle_send(result, assistant_id, user)
            return JSONResponse(status_code=200, content=build_success_response(payload, result.id))
        if result.operation == "get_task":
            payload = await _handle_get_task(result, user)
            return JSONResponse(status_code=200, content=build_success_response(payload, result.id))
        if result.operation == "cancel_task":
            payload = await _handle_cancel_task(result, user)
            return JSONResponse(status_code=200, content=build_success_response(payload, result.id))

    except JsonRpcError as exc:
        return JSONResponse(status_code=200, content=build_error_response(exc.code, exc.message, result.id))

    except HTTPException as exc:
        if exc.status_code >= 500:
            logger.exception("A2A request failed", method=result.method)
            code, message = JsonRpcErrorCode.INTERNAL_ERROR, "Internal server error"
        else:
            code, message = JsonRpcErrorCode.INVALID_PARAMS, str(exc.detail)
        return JSONResponse(status_code=200, content=build_error_response(code, message, result.id))

    except Exception:
        # A2A clients only read the error from an HTTP 200 body, so nothing may escape as a 500.
        logger.exception("A2A request failed", method=result.method)
        return JSONResponse(
            status_code=200,
            content=build_error_response(JsonRpcErrorCode.INTERNAL_ERROR, "Internal server error", result.id),
        )

    return JSONResponse(
        status_code=200,
        content=build_error_response(
            JsonRpcErrorCode.METHOD_NOT_FOUND, f"Method not found: {result.method}", result.id
        ),
    )

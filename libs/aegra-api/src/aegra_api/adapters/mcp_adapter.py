"""MCP adapter — exposes configured graphs as MCP tools at /mcp."""

import contextvars
import importlib
import importlib.util
import json
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

import mcp.types as mcp_types
import structlog
from fastapi import FastAPI, HTTPException
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.tools.base import Tool, ToolResult
from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
from pydantic import SkipValidation
from sqlalchemy import select
from starlette.middleware import Middleware
from starlette.requests import Request as StarletteRequest
from starlette.responses import JSONResponse as StarletteJSONResponse
from starlette.routing import Route
from starlette.types import ASGIApp, Receive, Scope, Send

from aegra_api.api.stateless_runs import _delete_thread_by_id
from aegra_api.config import get_config_dir, load_mcp_config
from aegra_api.core.auth_deps import _to_user_model
from aegra_api.core.auth_middleware import get_auth_backend
from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import _get_session_maker
from aegra_api.models.auth import User
from aegra_api.models.runs import RunCreate
from aegra_api.services.executor import executor
from aegra_api.services.langgraph_service import LangGraphService
from aegra_api.services.run_preparation import _prepare_run
from aegra_api.settings import settings
from aegra_api.utils.assistants import resolve_assistant_id

logger = structlog.get_logger(__name__)

_current_user: contextvars.ContextVar[User | None] = contextvars.ContextVar("_mcp_current_user", default=None)
_final_response_only: bool | None = None
mcp_server = FastMCP("aegra")


def _is_final_response_only() -> bool:
    global _final_response_only
    if _final_response_only is None:
        mcp_cfg = load_mcp_config()
        _final_response_only = bool(mcp_cfg.get("final_response_only", False)) if mcp_cfg else False
    return _final_response_only


def _extract_final_response(output: dict[str, Any]) -> str:
    messages = output.get("messages", [])
    if not messages:
        return json.dumps(output)
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("type") == "ai":
            return json.dumps(msg)
    return json.dumps(output)


async def _read_run_result(run_id: str, thread_id: str, user_id: str) -> tuple[str, dict[str, Any], str | None]:
    maker = _get_session_maker()
    async with maker() as session:
        run_orm = await session.scalar(
            select(RunORM).where(
                RunORM.run_id == run_id,
                RunORM.thread_id == thread_id,
                RunORM.user_id == user_id,
            )
        )
    if not run_orm:
        return "error", {}, "Run not found"
    return run_orm.status, run_orm.output or {}, run_orm.error_message


_OAUTH_FLOW_SUFFIXES = ("/register", "/authorize", "/token", "/consent", "/auth/callback")
_mcp_oauth_enabled: bool = False


class _AuthMiddleware:
    """Bridges MCP requests to Aegra's auth backend, swapping upstream OAuth tokens when present."""

    def __init__(self, app: ASGIApp, **_kwargs: Any) -> None:
        self._app = app

    @staticmethod
    def _is_oauth_path(path: str) -> bool:
        stripped = path.rstrip("/")
        return stripped.endswith(_OAUTH_FLOW_SUFFIXES)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            await self._app(scope, receive, send)
            return
        if scope["type"] != "http":
            response = StarletteJSONResponse(status_code=400, content={"error": "Unsupported scope type"})
            await response(scope, receive, send)
            return

        path = scope.get("path", "")
        if self._is_oauth_path(path):
            await self._app(scope, receive, send)
            return

        auth_user = scope.get("user")
        if isinstance(auth_user, AuthenticatedUser):
            upstream = f"Bearer {auth_user.access_token.token}".encode()
            scope["headers"] = [(k, upstream if k == b"authorization" else v) for k, v in scope["headers"]]

        request = StarletteRequest(scope, receive)
        backend = get_auth_backend()

        try:
            result = await backend.authenticate(request)
        except Exception as exc:
            if _mcp_oauth_enabled:
                await self._app(scope, receive, send)
                return
            logger.warning("mcp_auth_failed", error=str(exc))
            response = StarletteJSONResponse(status_code=401, content={"error": "Authentication failed"})
            await response(scope, receive, send)
            return

        if result is None:
            anonymous = User(identity="anonymous", is_authenticated=True, permissions=[])
            token = _current_user.set(anonymous)
            try:
                await self._app(scope, receive, send)
            finally:
                _current_user.reset(token)
            return

        _credentials, user_obj = result
        token = _current_user.set(_to_user_model(user_obj))
        try:
            await self._app(scope, receive, send)
        finally:
            _current_user.reset(token)


class _GraphTool(Tool):
    graph_id: str
    langgraph_service: SkipValidation[LangGraphService]

    model_config = {"arbitrary_types_allowed": True}

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        user = _current_user.get()
        if user is None:
            raise RuntimeError("No authenticated user in MCP context")

        registry = self.langgraph_service._graph_registry
        assistant_id = resolve_assistant_id(self.graph_id, registry)
        thread_id = str(uuid4())
        request = RunCreate(assistant_id=assistant_id, input=arguments)

        maker = _get_session_maker()
        try:
            try:
                async with maker() as session:
                    run_id, _run, _job = await _prepare_run(session, thread_id, request, user, initial_status="pending")
            except HTTPException as exc:
                raise ToolError(exc.detail) from exc

            try:
                await executor.wait_for_completion(run_id, timeout=settings.worker.BG_JOB_TIMEOUT_SECS)
            except TimeoutError as exc:
                raise ToolError("Graph execution timed out") from exc

            status, output, error_msg = await _read_run_result(run_id, thread_id, user.identity)
        finally:
            try:
                await _delete_thread_by_id(thread_id, user.identity)
            except Exception:
                logger.exception("mcp_thread_cleanup_failed", thread_id=thread_id)

        if status == "error":
            raise ToolError(error_msg or "Graph execution failed")

        text = _extract_final_response(output) if _is_final_response_only() else json.dumps(output)
        return ToolResult(content=[mcp_types.TextContent(type="text", text=text)])


async def register_mcp_tools(service: LangGraphService) -> None:
    for graph_id, graph_meta in service._graph_registry.items():
        try:
            graph = await service._get_base_graph(graph_id)
            input_schema = graph.get_input_jsonschema()
        except Exception:
            logger.exception("mcp_register_tool_failed", graph_id=graph_id)
            continue

        description = (
            graph_meta.get("description", f"Run the {graph_id} agent")
            if isinstance(graph_meta, dict)
            else f"Run the {graph_id} agent"
        )

        tool = _GraphTool(
            name=graph_id,
            description=description,
            parameters=input_schema,
            graph_id=graph_id,
            langgraph_service=service,
        )
        mcp_server.add_tool(tool)
        logger.info("mcp_tool_registered", graph_id=graph_id)


_mcp_app_lifespan: Callable[..., AbstractAsyncContextManager[Any]] | None = None


def _load_mcp_auth_provider(path: str) -> Any:
    """Load auth provider from ``./file.py:var`` or ``module:var`` path."""
    if ":" not in path:
        raise ValueError(f"Invalid MCP auth path format (missing ':'): {path}")

    module_path, var_name = path.rsplit(":", 1)

    is_file_path = module_path.endswith(".py") or module_path.startswith("./") or module_path.startswith("../")
    if is_file_path:
        file_path = Path(module_path)
        if not file_path.is_absolute():
            config_dir = get_config_dir()
            file_path = ((config_dir / file_path) if config_dir else (Path.cwd() / file_path)).resolve()

        if not file_path.exists():
            raise FileNotFoundError(f"MCP auth file not found: {file_path}")
        if not file_path.is_file():
            raise ValueError(f"MCP auth path is not a file: {file_path}")

        module_name = f"mcp_auth_module_{file_path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, str(file_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create module spec from {file_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_path)

    provider = getattr(module, var_name, None)
    if provider is None:
        raise AttributeError(f"Variable '{var_name}' not found in {module_path}")

    logger.info("mcp_auth_provider_loaded", path=path)
    return provider


def mount_mcp(app: FastAPI) -> None:
    """Mount the MCP Streamable HTTP endpoint at ``/mcp``."""
    global _mcp_app_lifespan, _mcp_oauth_enabled

    mcp_cfg = load_mcp_config()
    auth_cfg = (mcp_cfg or {}).get("auth")

    auth_provider: Any = None
    if auth_cfg and "path" in auth_cfg:
        auth_provider = _load_mcp_auth_provider(auth_cfg["path"])
        mcp_server.auth = auth_provider
        _mcp_oauth_enabled = True

    mcp_app = mcp_server.http_app(
        path="/",
        stateless_http=True,
        middleware=[Middleware(_AuthMiddleware)],
    )

    if auth_provider is not None and hasattr(auth_provider, "get_well_known_routes"):
        # Provider's base_url already includes mount path; passing mcp_path would double it.
        well_known_routes: list[Route] = auth_provider.get_well_known_routes()
        for route in well_known_routes:
            app.routes.insert(0, route)
        logger.info("mcp_mounted", auth_provider=True, well_known_routes=[r.path for r in well_known_routes])
    else:
        logger.info("mcp_mounted")

    _mcp_app_lifespan = mcp_app.lifespan
    app.mount("/mcp", mcp_app)


@asynccontextmanager
async def mcp_lifespan() -> AsyncIterator[None]:
    """FastAPI doesn't propagate lifespan to mounted apps."""
    if _mcp_app_lifespan is None:
        yield
        return
    async with _mcp_app_lifespan(None):
        yield

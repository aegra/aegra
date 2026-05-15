"""Unit tests for the MCP adapter."""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp.exceptions import ToolError
from fastmcp.server.auth.auth import AccessToken
from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser

from aegra_api.adapters.mcp_adapter import (
    _AuthMiddleware,
    _current_user,
    _extract_final_response,
    _GraphTool,
    _load_mcp_auth_provider,
    mcp_server,
    register_mcp_tools,
)
from aegra_api.models.auth import User

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_module_state() -> None:
    """Reset module-level state between tests."""
    from aegra_api.adapters import mcp_adapter

    mcp_adapter._final_response_only = None
    mcp_adapter._mcp_oauth_enabled = False
    mcp_adapter.mcp_server._local_provider._components.clear()


def _mock_langgraph_service(registry: dict[str, Any] | None = None) -> MagicMock:
    """Build a mocked LangGraphService with a fake graph registry."""
    svc = MagicMock()
    svc._graph_registry = registry or {}
    mock_graph = MagicMock()
    mock_graph.get_input_jsonschema.return_value = {
        "type": "object",
        "properties": {"messages": {"type": "array"}},
    }
    svc._get_base_graph = AsyncMock(return_value=mock_graph)
    return svc


def _make_graph_tool(*, graph_id: str = "agent", service: MagicMock | None = None) -> _GraphTool:
    """Create a _GraphTool with a mocked LangGraphService."""
    if service is None:
        service = _mock_langgraph_service({"agent": {"file_path": "agent.py", "export_name": "graph"}})
    return _GraphTool(
        name=graph_id,
        description=f"Run the {graph_id} agent",
        parameters={"type": "object", "properties": {"messages": {"type": "array"}}},
        graph_id=graph_id,
        langgraph_service=service,
    )


def _make_access_token(*, client_id: str = "test-client", token: str = "upstream-jwt") -> AccessToken:
    """Create a minimal AccessToken for testing."""
    return AccessToken(
        token=token,
        client_id=client_id,
        scopes=["openid"],
        claims={"sub": "user-123", "email": "alice@example.com"},
    )


def _make_authenticated_scope(access_token: AccessToken) -> dict[str, Any]:
    """Build an ASGI scope with an AuthenticatedUser (simulating FastMCP auth)."""
    return {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"authorization", b"Bearer fastmcp-internal-jwt")],
        "user": AuthenticatedUser(access_token),
    }


# ---------------------------------------------------------------------------
# register_mcp_tools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_register_mcp_tools_registers_one_tool_per_graph() -> None:
    """One _GraphTool per graph in the registry."""
    registry: dict[str, Any] = {
        "agent": {"file_path": "agent.py", "export_name": "graph"},
        "researcher": {"file_path": "researcher.py", "export_name": "graph"},
    }
    svc = _mock_langgraph_service(registry)

    await register_mcp_tools(svc)

    tools = await mcp_server._local_provider.list_tools()
    assert len(tools) == 2
    names = {t.name for t in tools}
    assert names == {"agent", "researcher"}


@pytest.mark.asyncio
async def test_register_mcp_tools_skips_graphs_that_fail_to_load() -> None:
    """Graphs whose _get_base_graph raises are skipped."""
    registry: dict[str, Any] = {"broken": {"file_path": "broken.py", "export_name": "graph"}}
    svc = _mock_langgraph_service(registry)
    svc._get_base_graph = AsyncMock(side_effect=RuntimeError("load failed"))

    await register_mcp_tools(svc)

    tools = await mcp_server._local_provider.list_tools()
    assert len(tools) == 0


@pytest.mark.asyncio
async def test_register_mcp_tools_includes_input_schema() -> None:
    """Registered tools include the graph's input JSON schema."""
    registry: dict[str, Any] = {"agent": {"file_path": "agent.py", "export_name": "graph"}}
    svc = _mock_langgraph_service(registry)

    await register_mcp_tools(svc)

    tool = await mcp_server._local_provider.get_tool("agent")
    assert tool.parameters == {
        "type": "object",
        "properties": {"messages": {"type": "array"}},
    }


@pytest.mark.asyncio
async def test_register_mcp_tools_uses_description_from_config() -> None:
    """Uses description from graph config when available."""
    registry: dict[str, Any] = {
        "agent": {"file_path": "agent.py", "export_name": "graph", "description": "My custom agent"},
    }
    svc = _mock_langgraph_service(registry)

    await register_mcp_tools(svc)

    tool = await mcp_server._local_provider.get_tool("agent")
    assert tool.description == "My custom agent"


@pytest.mark.asyncio
async def test_register_mcp_tools_uses_default_description() -> None:
    """Falls back to a default description."""
    registry: dict[str, Any] = {"agent": {"file_path": "agent.py", "export_name": "graph"}}
    svc = _mock_langgraph_service(registry)

    await register_mcp_tools(svc)

    tool = await mcp_server._local_provider.get_tool("agent")
    assert tool.description == "Run the agent agent"


# ---------------------------------------------------------------------------
# _GraphTool.run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graph_tool_run_raises_when_no_user() -> None:
    """Raises RuntimeError when no user is in the ContextVar."""
    tool = _make_graph_tool()
    with pytest.raises(RuntimeError, match="No authenticated user"):
        await tool.run({"messages": []})


@pytest.mark.asyncio
async def test_graph_tool_run_executes_graph_and_returns_output() -> None:
    """Prepares a run, waits, reads output, and returns ToolResult."""
    svc = _mock_langgraph_service({"agent": {"file_path": "agent.py", "export_name": "graph"}})
    tool = _make_graph_tool(service=svc)

    user = User(identity="alice", is_authenticated=True, permissions=[])
    token = _current_user.set(user)

    try:
        fake_output: dict[str, Any] = {"messages": [{"role": "assistant", "content": "Hello!"}]}

        with (
            patch("aegra_api.adapters.mcp_adapter._get_session_maker") as mock_maker,
            patch("aegra_api.adapters.mcp_adapter._prepare_run") as mock_prepare,
            patch("aegra_api.adapters.mcp_adapter.executor") as mock_executor,
            patch("aegra_api.adapters.mcp_adapter._read_run_result", new_callable=AsyncMock) as mock_read,
            patch("aegra_api.adapters.mcp_adapter._delete_thread_by_id", new_callable=AsyncMock),
        ):
            mock_session = AsyncMock()
            mock_maker.return_value = MagicMock(
                __aenter__=AsyncMock(return_value=mock_session), __aexit__=AsyncMock(return_value=False)
            )
            mock_prepare.return_value = ("run-123", MagicMock(), MagicMock())
            mock_executor.wait_for_completion = AsyncMock()
            mock_read.return_value = ("success", fake_output, None)

            result = await tool.run({"messages": [{"role": "user", "content": "hi"}]})

        assert len(result.content) == 1
        assert result.content[0].type == "text"
        assert '"Hello!"' in result.content[0].text
    finally:
        _current_user.reset(token)


@pytest.mark.asyncio
async def test_graph_tool_run_returns_filtered_output_when_final_response_only() -> None:
    """Returns only the last AI message when final_response_only is enabled."""
    from aegra_api.adapters import mcp_adapter

    svc = _mock_langgraph_service({"agent": {"file_path": "agent.py", "export_name": "graph"}})
    tool = _make_graph_tool(service=svc)

    user = User(identity="alice", is_authenticated=True, permissions=[])
    token = _current_user.set(user)

    try:
        fake_output: dict[str, Any] = {
            "messages": [
                {"type": "human", "content": "Hi"},
                {"type": "ai", "content": "Thinking..."},
                {"type": "tool", "content": "Tool result"},
                {"type": "ai", "content": "Final answer"},
            ]
        }

        mcp_adapter._final_response_only = True

        with (
            patch("aegra_api.adapters.mcp_adapter._get_session_maker") as mock_maker,
            patch("aegra_api.adapters.mcp_adapter._prepare_run") as mock_prepare,
            patch("aegra_api.adapters.mcp_adapter.executor") as mock_executor,
            patch("aegra_api.adapters.mcp_adapter._read_run_result", new_callable=AsyncMock) as mock_read,
            patch("aegra_api.adapters.mcp_adapter._delete_thread_by_id", new_callable=AsyncMock),
        ):
            mock_session = AsyncMock()
            mock_maker.return_value = MagicMock(
                __aenter__=AsyncMock(return_value=mock_session), __aexit__=AsyncMock(return_value=False)
            )
            mock_prepare.return_value = ("run-123", MagicMock(), MagicMock())
            mock_executor.wait_for_completion = AsyncMock()
            mock_read.return_value = ("success", fake_output, None)

            result = await tool.run({"messages": [{"role": "user", "content": "hi"}]})

        assert len(result.content) == 1
        parsed = json.loads(result.content[0].text)
        assert parsed["type"] == "ai"
        assert parsed["content"] == "Final answer"
    finally:
        _current_user.reset(token)


@pytest.mark.asyncio
async def test_graph_tool_run_returns_full_output_when_final_response_only_disabled() -> None:
    """Returns full output when final_response_only is disabled (default)."""
    from aegra_api.adapters import mcp_adapter

    svc = _mock_langgraph_service({"agent": {"file_path": "agent.py", "export_name": "graph"}})
    tool = _make_graph_tool(service=svc)

    user = User(identity="alice", is_authenticated=True, permissions=[])
    token = _current_user.set(user)

    try:
        fake_output: dict[str, Any] = {
            "messages": [
                {"type": "human", "content": "Hi"},
                {"type": "ai", "content": "Final answer"},
            ]
        }

        mcp_adapter._final_response_only = False

        with (
            patch("aegra_api.adapters.mcp_adapter._get_session_maker") as mock_maker,
            patch("aegra_api.adapters.mcp_adapter._prepare_run") as mock_prepare,
            patch("aegra_api.adapters.mcp_adapter.executor") as mock_executor,
            patch("aegra_api.adapters.mcp_adapter._read_run_result", new_callable=AsyncMock) as mock_read,
            patch("aegra_api.adapters.mcp_adapter._delete_thread_by_id", new_callable=AsyncMock),
        ):
            mock_session = AsyncMock()
            mock_maker.return_value = MagicMock(
                __aenter__=AsyncMock(return_value=mock_session), __aexit__=AsyncMock(return_value=False)
            )
            mock_prepare.return_value = ("run-123", MagicMock(), MagicMock())
            mock_executor.wait_for_completion = AsyncMock()
            mock_read.return_value = ("success", fake_output, None)

            result = await tool.run({"messages": [{"role": "user", "content": "hi"}]})

        assert len(result.content) == 1
        parsed = json.loads(result.content[0].text)
        assert parsed == fake_output
    finally:
        _current_user.reset(token)


@pytest.mark.asyncio
async def test_graph_tool_run_returns_error_when_run_fails() -> None:
    """Raises ToolError when the graph run fails."""
    svc = _mock_langgraph_service({"agent": {"file_path": "agent.py", "export_name": "graph"}})
    tool = _make_graph_tool(service=svc)

    user = User(identity="alice", is_authenticated=True, permissions=[])
    token = _current_user.set(user)

    try:
        with (
            patch("aegra_api.adapters.mcp_adapter._get_session_maker") as mock_maker,
            patch("aegra_api.adapters.mcp_adapter._prepare_run") as mock_prepare,
            patch("aegra_api.adapters.mcp_adapter.executor") as mock_executor,
            patch("aegra_api.adapters.mcp_adapter._read_run_result", new_callable=AsyncMock) as mock_read,
            patch("aegra_api.adapters.mcp_adapter._delete_thread_by_id", new_callable=AsyncMock),
        ):
            mock_session = AsyncMock()
            mock_maker.return_value = MagicMock(
                __aenter__=AsyncMock(return_value=mock_session), __aexit__=AsyncMock(return_value=False)
            )
            mock_prepare.return_value = ("run-123", MagicMock(), MagicMock())
            mock_executor.wait_for_completion = AsyncMock()
            mock_read.return_value = ("error", {}, "ValueError: bad input")

            with pytest.raises(ToolError, match="ValueError: bad input"):
                await tool.run({"messages": [{"role": "user", "content": "hi"}]})
    finally:
        _current_user.reset(token)


@pytest.mark.asyncio
async def test_graph_tool_run_raises_tool_error_on_timeout() -> None:
    """Converts TimeoutError into ToolError."""
    svc = _mock_langgraph_service({"agent": {"file_path": "agent.py", "export_name": "graph"}})
    tool = _make_graph_tool(service=svc)

    user = User(identity="alice", is_authenticated=True, permissions=[])
    token = _current_user.set(user)

    try:
        with (
            patch("aegra_api.adapters.mcp_adapter._get_session_maker") as mock_maker,
            patch("aegra_api.adapters.mcp_adapter._prepare_run") as mock_prepare,
            patch("aegra_api.adapters.mcp_adapter.executor") as mock_executor,
            patch("aegra_api.adapters.mcp_adapter._delete_thread_by_id", new_callable=AsyncMock) as mock_delete,
        ):
            mock_session = AsyncMock()
            mock_maker.return_value = MagicMock(
                __aenter__=AsyncMock(return_value=mock_session), __aexit__=AsyncMock(return_value=False)
            )
            mock_prepare.return_value = ("run-123", MagicMock(), MagicMock())
            mock_executor.wait_for_completion = AsyncMock(side_effect=TimeoutError("deadline exceeded"))

            with pytest.raises(ToolError, match="timed out"):
                await tool.run({"messages": [{"role": "user", "content": "hi"}]})

            mock_delete.assert_awaited_once()
    finally:
        _current_user.reset(token)


# ---------------------------------------------------------------------------
# _AuthMiddleware
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auth_middleware_sets_anonymous_when_backend_returns_none() -> None:
    """When auth backend returns None, middleware sets anonymous user."""
    captured_users: list[User | None] = []

    async def _capture_app(scope: Any, receive: Any, send: Any) -> None:
        captured_users.append(_current_user.get())
        from starlette.responses import JSONResponse

        response = JSONResponse({"ok": True})
        await response(scope, receive, send)

    middleware = _AuthMiddleware(_capture_app)

    mock_backend = MagicMock()
    mock_backend.authenticate = AsyncMock(return_value=None)

    with patch("aegra_api.adapters.mcp_adapter.get_auth_backend", return_value=mock_backend):
        import httpx

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=middleware),
            base_url="http://test",
        ) as client:
            resp = await client.get("/anything")

    assert resp.status_code == 200
    assert len(captured_users) == 1
    assert captured_users[0] is not None
    assert captured_users[0].identity == "anonymous"


@pytest.mark.asyncio
async def test_auth_middleware_rejects_non_http_non_lifespan_scopes() -> None:
    """Non-http, non-lifespan scopes get 400."""

    async def _noop_app(scope: Any, receive: Any, send: Any) -> None:
        pass

    middleware = _AuthMiddleware(_noop_app)

    scope: dict[str, Any] = {"type": "websocket", "asgi": {"version": "3.0"}}
    responses: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {}

    async def send(msg: dict[str, Any]) -> None:
        responses.append(msg)

    await middleware(scope, receive, send)
    assert any(r.get("status") == 400 for r in responses)


@pytest.mark.asyncio
async def test_auth_middleware_passes_through_when_oauth_enabled() -> None:
    """When MCP OAuth is enabled, auth failures pass through to FastMCP."""
    from starlette.authentication import AuthenticationError

    from aegra_api.adapters import mcp_adapter

    mcp_adapter._mcp_oauth_enabled = True

    inner_called = False

    async def _inner_app(scope: Any, receive: Any, send: Any) -> None:
        nonlocal inner_called
        inner_called = True
        from starlette.responses import JSONResponse

        response = JSONResponse({"ok": True})
        await response(scope, receive, send)

    middleware = _AuthMiddleware(_inner_app)

    mock_backend = MagicMock()
    mock_backend.authenticate = AsyncMock(side_effect=AuthenticationError("Missing token"))

    with patch("aegra_api.adapters.mcp_adapter.get_auth_backend", return_value=mock_backend):
        import httpx

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=middleware),
            base_url="http://test",
        ) as client:
            resp = await client.post("/mcp")

    assert inner_called, "Inner app should be called so FastMCP can return proper 401"
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_auth_middleware_rejects_when_oauth_disabled() -> None:
    """Without MCP OAuth, auth failures return 401 directly."""
    from starlette.authentication import AuthenticationError

    from aegra_api.adapters import mcp_adapter

    mcp_adapter._mcp_oauth_enabled = False

    async def _inner_app(scope: Any, receive: Any, send: Any) -> None:
        pytest.fail("Inner app should NOT be called when OAuth is disabled")

    middleware = _AuthMiddleware(_inner_app)

    mock_backend = MagicMock()
    mock_backend.authenticate = AsyncMock(side_effect=AuthenticationError("Missing token"))

    with patch("aegra_api.adapters.mcp_adapter.get_auth_backend", return_value=mock_backend):
        import httpx

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=middleware),
            base_url="http://test",
        ) as client:
            resp = await client.post("/mcp")

    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_auth_middleware_swaps_upstream_token() -> None:
    """When scope has AuthenticatedUser, swaps the upstream token into headers."""
    access_token = _make_access_token()
    captured_headers: list[dict[str, str]] = []

    mock_backend = MagicMock()

    async def _capture_authenticate(conn: Any) -> tuple[Any, Any]:
        captured_headers.append(dict(conn.headers))
        user_data = MagicMock()
        user_data.identity = "alice"
        user_data.is_authenticated = True
        user_data.display_name = "Alice"
        user_data.permissions = []
        user_data._user_data = {"identity": "alice", "is_authenticated": True}
        user_data.to_dict.return_value = {"identity": "alice", "is_authenticated": True}
        return MagicMock(), user_data

    mock_backend.authenticate = AsyncMock(side_effect=_capture_authenticate)

    responses: list[dict[str, Any]] = []

    async def _inner_app(scope: Any, receive: Any, send: Any) -> None:
        from starlette.responses import JSONResponse

        response = JSONResponse({"ok": True})
        await response(scope, receive, send)

    middleware = _AuthMiddleware(_inner_app)

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b""}

    async def send(msg: dict[str, Any]) -> None:
        responses.append(msg)

    scope = _make_authenticated_scope(access_token)
    with patch("aegra_api.adapters.mcp_adapter.get_auth_backend", return_value=mock_backend):
        await middleware(scope, receive, send)

    assert len(captured_headers) == 1
    assert captured_headers[0]["authorization"] == "Bearer upstream-jwt"


@pytest.mark.asyncio
async def test_auth_middleware_no_swap_without_authenticated_user() -> None:
    """Without AuthenticatedUser in scope, passes headers as-is."""
    captured_headers: list[dict[str, str]] = []

    mock_backend = MagicMock()

    async def _capture_authenticate(conn: Any) -> None:
        captured_headers.append(dict(conn.headers))
        return None

    mock_backend.authenticate = AsyncMock(side_effect=_capture_authenticate)

    async def _inner_app(scope: Any, receive: Any, send: Any) -> None:
        from starlette.responses import JSONResponse

        response = JSONResponse({"ok": True})
        await response(scope, receive, send)

    middleware = _AuthMiddleware(_inner_app)
    scope: dict[str, Any] = {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"authorization", b"Bearer direct-token")],
    }

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b""}

    async def send(msg: dict[str, Any]) -> None:
        pass

    with patch("aegra_api.adapters.mcp_adapter.get_auth_backend", return_value=mock_backend):
        await middleware(scope, receive, send)

    assert len(captured_headers) == 1
    assert captured_headers[0]["authorization"] == "Bearer direct-token"


# ---------------------------------------------------------------------------
# _extract_final_response
# ---------------------------------------------------------------------------


def test_extract_final_response_returns_last_ai_message() -> None:
    """Extracts the last AI message from a multi-type messages list."""
    output: dict[str, Any] = {
        "messages": [
            {"type": "human", "content": "Hello"},
            {"type": "ai", "content": "Let me search for that.", "tool_calls": [{"name": "search"}]},
            {"type": "tool", "content": "Search result: ..."},
            {"type": "ai", "content": "Here is your answer."},
        ]
    }
    result = _extract_final_response(output)
    parsed = json.loads(result)
    assert parsed["type"] == "ai"
    assert parsed["content"] == "Here is your answer."


def test_extract_final_response_returns_full_ai_message_dict() -> None:
    """Returns the entire AI message dict, not just content."""
    ai_msg: dict[str, Any] = {
        "type": "ai",
        "content": "Done!",
        "id": "msg-123",
        "response_metadata": {"model": "gpt-4o"},
    }
    output: dict[str, Any] = {"messages": [{"type": "human", "content": "Hi"}, ai_msg]}
    result = _extract_final_response(output)
    parsed = json.loads(result)
    assert parsed == ai_msg


def test_extract_final_response_with_list_content() -> None:
    """AI messages with list content blocks are returned as-is."""
    output: dict[str, Any] = {
        "messages": [
            {"type": "human", "content": "Hi"},
            {"type": "ai", "content": [{"type": "text", "text": "Hello!"}, {"type": "image_url", "url": "..."}]},
        ]
    }
    result = _extract_final_response(output)
    parsed = json.loads(result)
    assert parsed["type"] == "ai"
    assert isinstance(parsed["content"], list)
    assert len(parsed["content"]) == 2


def test_extract_final_response_no_messages_key() -> None:
    """Falls back to full output when there is no messages key."""
    output: dict[str, Any] = {"result": "42"}
    result = _extract_final_response(output)
    assert json.loads(result) == output


def test_extract_final_response_empty_messages() -> None:
    """Falls back to full output when messages list is empty."""
    output: dict[str, Any] = {"messages": []}
    result = _extract_final_response(output)
    assert json.loads(result) == output


def test_extract_final_response_no_ai_message() -> None:
    """Falls back to full output when no AI message exists."""
    output: dict[str, Any] = {
        "messages": [
            {"type": "human", "content": "Hello"},
            {"type": "tool", "content": "Result"},
        ]
    }
    result = _extract_final_response(output)
    assert json.loads(result) == output


# ---------------------------------------------------------------------------
# _load_mcp_auth_provider
# ---------------------------------------------------------------------------


def test_load_mcp_auth_provider_raises_on_missing_colon() -> None:
    """Raises ValueError when path has no ':' separator."""
    with pytest.raises(ValueError, match="missing ':'"):
        _load_mcp_auth_provider("./mcp_auth.py")


def test_load_mcp_auth_provider_raises_on_missing_file() -> None:
    """Raises FileNotFoundError for non-existent file."""
    with pytest.raises(FileNotFoundError, match="not found"):
        _load_mcp_auth_provider("./nonexistent_mcp_auth.py:provider")


def test_load_mcp_auth_provider_loads_from_file(tmp_path: Any) -> None:
    """Loads a variable from a Python file."""
    auth_file = tmp_path / "mcp_auth.py"
    auth_file.write_text("my_provider = {'type': 'test_provider'}\n")

    with patch("aegra_api.adapters.mcp_adapter.get_config_dir", return_value=tmp_path):
        provider = _load_mcp_auth_provider("./mcp_auth.py:my_provider")

    assert provider == {"type": "test_provider"}


def test_load_mcp_auth_provider_loads_from_module() -> None:
    """Loads a variable from an installed module."""
    provider = _load_mcp_auth_provider("json:dumps")
    assert provider is json.dumps


def test_load_mcp_auth_provider_raises_on_missing_variable(tmp_path: Any) -> None:
    """Raises AttributeError when variable doesn't exist."""
    auth_file = tmp_path / "mcp_auth.py"
    auth_file.write_text("other_var = 42\n")

    with (
        patch("aegra_api.adapters.mcp_adapter.get_config_dir", return_value=tmp_path),
        pytest.raises(AttributeError, match="not found"),
    ):
        _load_mcp_auth_provider("./mcp_auth.py:missing_var")


def test_load_mcp_auth_provider_resolves_relative_to_config_dir(tmp_path: Any) -> None:
    """Resolves relative paths from config directory."""
    sub_dir = tmp_path / "src" / "auth"
    sub_dir.mkdir(parents=True)
    auth_file = sub_dir / "mcp.py"
    auth_file.write_text("provider = 'loaded_from_subdir'\n")

    with patch("aegra_api.adapters.mcp_adapter.get_config_dir", return_value=tmp_path):
        provider = _load_mcp_auth_provider("./src/auth/mcp.py:provider")

    assert provider == "loaded_from_subdir"


def test_load_mcp_auth_provider_raises_on_directory(tmp_path: Any) -> None:
    """Raises ValueError when path points to a directory."""
    dir_path = tmp_path / "mcp_auth.py"
    dir_path.mkdir()

    with (
        patch("aegra_api.adapters.mcp_adapter.get_config_dir", return_value=tmp_path),
        pytest.raises(ValueError, match="not a file"),
    ):
        _load_mcp_auth_provider("./mcp_auth.py:provider")

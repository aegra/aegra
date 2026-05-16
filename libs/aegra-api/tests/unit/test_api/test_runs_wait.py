"""Unit tests for wait_for_run endpoint exception paths and edge cases.

Since wait_for_run now returns a StreamingResponse with heartbeat keep-alive,
tests consume the body and parse the JSON result from the concatenated chunks.
"""

import json
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import HTTPException

from aegra_api.api.runs import wait_for_run
from aegra_api.core.orm import Assistant as AssistantORM
from aegra_api.core.orm import Run as RunORM
from aegra_api.models import User


def _make_session_maker(session: AsyncMock) -> MagicMock:
    """Build a mock async_sessionmaker that always yields the given session."""
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return MagicMock(return_value=ctx)


async def _single_event_stream() -> AsyncGenerator:
    """Minimal async generator standing in for the wait-response body."""
    yield "data"


def _make_multi_session_maker(*sessions: AsyncMock) -> MagicMock:
    """Build a mock async_sessionmaker that yields a different session each call."""
    sessions_iter = iter(sessions)

    def _factory() -> MagicMock:
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(side_effect=lambda: next(sessions_iter))
        ctx.__aexit__ = AsyncMock(return_value=False)
        return ctx

    return MagicMock(side_effect=_factory)


def _make_request() -> MagicMock:
    """Build a standard mock RunCreate request."""
    request = MagicMock()
    request.assistant_id = "test-assistant"
    request.input = {"message": "test"}
    request.command = None
    request.config = {}
    request.context = None
    request.checkpoint = None
    request.stream_mode = None
    request.interrupt_before = None
    request.interrupt_after = None
    request.multitask_strategy = None
    request.stream_subgraphs = False
    request.metadata = None
    return request


def _make_assistant() -> AssistantORM:
    return AssistantORM(
        assistant_id="test-assistant",
        graph_id="test-graph",
        config={},
        context={},
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


def _make_run_orm(
    run_id: str,
    thread_id: str,
    *,
    status: str = "success",
    output: dict | None = None,
    error_message: str | None = None,
) -> RunORM:
    return RunORM(
        run_id=run_id,
        thread_id=thread_id,
        assistant_id="test-assistant",
        status=status,
        input={"message": "test"},
        config={},
        context={},
        user_id="test-user",
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        output=output or {},
        error_message=error_message,
    )


async def _consume_streaming_response(response: object) -> dict:
    """Read all chunks from a StreamingResponse and parse the JSON."""
    body = b""
    async for chunk in response.body_iterator:
        body += chunk if isinstance(chunk, bytes) else chunk.encode()
    return json.loads(body) if body else {}


# Standard patches for _prepare_run dependencies
_PREPARE_RUN_PATCHES = {
    "aegra_api.services.run_preparation._validate_resume_command": AsyncMock,
    "aegra_api.services.run_preparation.set_thread_status": AsyncMock,
    "aegra_api.services.run_preparation.update_thread_metadata": AsyncMock,
    "aegra_api.services.run_preparation.resolve_assistant_id": None,
}


class TestWaitForRunExceptionPaths:
    """Test exception handling and edge cases in wait_for_run endpoint."""

    @pytest.mark.asyncio
    async def test_wait_for_run_timeout(self) -> None:
        """Test that TimeoutError is handled gracefully and returns current state."""
        thread_id = "test-thread-123"
        run_id = str(uuid4())
        user = User(identity="test-user", scopes=[])
        request = _make_request()

        # Pre-execution session (for thread ownership check + _prepare_run)
        session_1 = AsyncMock()
        session_1.add = MagicMock()
        session_1.scalar.side_effect = [None, _make_assistant()]

        # Post-wait session (for _fetch_run_output)
        session_2 = AsyncMock()
        session_2.scalar.return_value = _make_run_orm(run_id, thread_id, status="running", output={"partial": "output"})

        mock_maker = _make_multi_session_maker(session_1, session_2)

        async def timeout_wait(rid: str, *, timeout: float) -> None:
            raise TimeoutError()

        with (
            patch("aegra_api.api.runs._get_session_maker", return_value=mock_maker),
            patch("aegra_api.services.run_waiters._get_session_maker", return_value=mock_maker),
            patch("aegra_api.services.run_preparation._validate_resume_command", new_callable=AsyncMock),
            patch("aegra_api.services.run_preparation.get_langgraph_service") as mock_lg_service,
            patch("aegra_api.services.run_preparation.resolve_assistant_id", return_value="test-assistant"),
            patch("aegra_api.services.run_preparation.set_thread_status", new_callable=AsyncMock),
            patch("aegra_api.services.run_preparation.update_thread_metadata", new_callable=AsyncMock),
            patch("aegra_api.services.run_preparation.uuid4", return_value=run_id),
            patch("aegra_api.services.run_waiters.executor") as mock_executor,
            patch("aegra_api.services.run_waiters.settings") as mock_settings,
            patch("aegra_api.api.runs.active_runs", {}),
        ):
            mock_lg_service.return_value.list_graphs.return_value = ["test-graph"]
            mock_executor.wait_for_completion = AsyncMock(side_effect=timeout_wait)
            mock_executor.submit = AsyncMock()
            mock_settings.app.KEEPALIVE_INTERVAL_SECS = 5
            mock_settings.worker.BG_JOB_TIMEOUT_SECS = 3600

            response = await wait_for_run(thread_id, request, user)
            result = await _consume_streaming_response(response)

        assert result == {"partial": "output"}

    @pytest.mark.asyncio
    async def test_wait_for_run_success(self) -> None:
        """Test that completed run output is returned via streaming response."""
        thread_id = "test-thread-123"
        run_id = str(uuid4())
        user = User(identity="test-user", scopes=[])
        request = _make_request()

        session_1 = AsyncMock()
        session_1.add = MagicMock()
        session_1.scalar.side_effect = [None, _make_assistant()]

        session_2 = AsyncMock()
        session_2.scalar.return_value = _make_run_orm(run_id, thread_id, output={"result": "success"})

        mock_maker = _make_multi_session_maker(session_1, session_2)

        with (
            patch("aegra_api.api.runs._get_session_maker", return_value=mock_maker),
            patch("aegra_api.services.run_waiters._get_session_maker", return_value=mock_maker),
            patch("aegra_api.services.run_preparation._validate_resume_command", new_callable=AsyncMock),
            patch("aegra_api.services.run_preparation.get_langgraph_service") as mock_lg_service,
            patch("aegra_api.services.run_preparation.resolve_assistant_id", return_value="test-assistant"),
            patch("aegra_api.services.run_preparation.set_thread_status", new_callable=AsyncMock),
            patch("aegra_api.services.run_preparation.update_thread_metadata", new_callable=AsyncMock),
            patch("aegra_api.services.run_preparation.uuid4", return_value=run_id),
            patch("aegra_api.services.run_waiters.executor") as mock_executor,
            patch("aegra_api.services.run_waiters.settings") as mock_settings,
            patch("aegra_api.api.runs.active_runs", {}),
        ):
            mock_lg_service.return_value.list_graphs.return_value = ["test-graph"]
            mock_executor.wait_for_completion = AsyncMock()
            mock_executor.submit = AsyncMock()
            mock_settings.app.KEEPALIVE_INTERVAL_SECS = 5
            mock_settings.worker.BG_JOB_TIMEOUT_SECS = 3600

            response = await wait_for_run(thread_id, request, user)
            result = await _consume_streaming_response(response)

        assert result == {"result": "success"}

    @pytest.mark.asyncio
    async def test_wait_for_run_failed_status(self) -> None:
        """Test that failed runs return their output as-is."""
        thread_id = "test-thread-123"
        run_id = str(uuid4())
        user = User(identity="test-user", scopes=[])
        request = _make_request()

        session_1 = AsyncMock()
        session_1.add = MagicMock()
        session_1.scalar.side_effect = [None, _make_assistant()]

        session_2 = AsyncMock()
        session_2.scalar.return_value = _make_run_orm(
            run_id,
            thread_id,
            status="error",
            output={"error": "execution failed"},
            error_message="Graph execution error",
        )

        mock_maker = _make_multi_session_maker(session_1, session_2)

        with (
            patch("aegra_api.api.runs._get_session_maker", return_value=mock_maker),
            patch("aegra_api.services.run_waiters._get_session_maker", return_value=mock_maker),
            patch("aegra_api.services.run_preparation._validate_resume_command", new_callable=AsyncMock),
            patch("aegra_api.services.run_preparation.get_langgraph_service") as mock_lg_service,
            patch("aegra_api.services.run_preparation.resolve_assistant_id", return_value="test-assistant"),
            patch("aegra_api.services.run_preparation.set_thread_status", new_callable=AsyncMock),
            patch("aegra_api.services.run_preparation.update_thread_metadata", new_callable=AsyncMock),
            patch("aegra_api.services.run_preparation.uuid4", return_value=run_id),
            patch("aegra_api.services.run_waiters.executor") as mock_executor,
            patch("aegra_api.services.run_waiters.settings") as mock_settings,
            patch("aegra_api.api.runs.active_runs", {}),
        ):
            mock_lg_service.return_value.list_graphs.return_value = ["test-graph"]
            mock_executor.wait_for_completion = AsyncMock()
            mock_executor.submit = AsyncMock()
            mock_settings.app.KEEPALIVE_INTERVAL_SECS = 5
            mock_settings.worker.BG_JOB_TIMEOUT_SECS = 3600

            response = await wait_for_run(thread_id, request, user)
            result = await _consume_streaming_response(response)

        assert result == {"error": "execution failed"}

    @pytest.mark.asyncio
    async def test_wait_for_run_interrupted_status(self) -> None:
        """Test that interrupted runs return partial output."""
        thread_id = "test-thread-123"
        run_id = str(uuid4())
        user = User(identity="test-user", scopes=[])
        request = _make_request()
        request.interrupt_before = ["agent"]

        session_1 = AsyncMock()
        session_1.add = MagicMock()
        session_1.scalar.side_effect = [None, _make_assistant()]

        session_2 = AsyncMock()
        session_2.scalar.return_value = _make_run_orm(
            run_id,
            thread_id,
            status="interrupted",
            output={"partial": "result", "__interrupt__": [{"value": "test"}]},
        )

        mock_maker = _make_multi_session_maker(session_1, session_2)

        with (
            patch("aegra_api.api.runs._get_session_maker", return_value=mock_maker),
            patch("aegra_api.services.run_waiters._get_session_maker", return_value=mock_maker),
            patch("aegra_api.services.run_preparation._validate_resume_command", new_callable=AsyncMock),
            patch("aegra_api.services.run_preparation.get_langgraph_service") as mock_lg_service,
            patch("aegra_api.services.run_preparation.resolve_assistant_id", return_value="test-assistant"),
            patch("aegra_api.services.run_preparation.set_thread_status", new_callable=AsyncMock),
            patch("aegra_api.services.run_preparation.update_thread_metadata", new_callable=AsyncMock),
            patch("aegra_api.services.run_preparation.uuid4", return_value=run_id),
            patch("aegra_api.services.run_waiters.executor") as mock_executor,
            patch("aegra_api.services.run_waiters.settings") as mock_settings,
            patch("aegra_api.api.runs.active_runs", {}),
        ):
            mock_lg_service.return_value.list_graphs.return_value = ["test-graph"]
            mock_executor.wait_for_completion = AsyncMock()
            mock_executor.submit = AsyncMock()
            mock_settings.app.KEEPALIVE_INTERVAL_SECS = 5
            mock_settings.worker.BG_JOB_TIMEOUT_SECS = 3600

            response = await wait_for_run(thread_id, request, user)
            result = await _consume_streaming_response(response)

        assert result == {"partial": "result", "__interrupt__": [{"value": "test"}]}

    @pytest.mark.asyncio
    async def test_wait_for_run_graph_not_found(self) -> None:
        """Test that HTTPException 404 is raised if assistant's graph doesn't exist."""
        thread_id = "test-thread-123"
        user = User(identity="test-user", scopes=[])
        request = _make_request()

        assistant = _make_assistant()
        assistant.graph_id = "nonexistent-graph"
        session = AsyncMock()
        session.add = MagicMock()
        session.scalar.side_effect = [None, assistant]

        mock_maker = _make_session_maker(session)

        with (
            patch("aegra_api.api.runs._get_session_maker", return_value=mock_maker),
            patch("aegra_api.services.run_waiters._get_session_maker", return_value=mock_maker),
            patch("aegra_api.services.run_preparation._validate_resume_command", new_callable=AsyncMock),
            patch("aegra_api.services.run_preparation.get_langgraph_service") as mock_lg_service,
            patch("aegra_api.services.run_preparation.resolve_assistant_id", return_value="test-assistant"),
            patch("aegra_api.services.run_preparation.set_thread_status", new_callable=AsyncMock),
            patch("aegra_api.services.run_preparation.update_thread_metadata", new_callable=AsyncMock),
        ):
            mock_lg_service.return_value.list_graphs.return_value = ["other-graph"]

            with pytest.raises(HTTPException) as exc_info:
                await wait_for_run(thread_id, request, user)

            assert exc_info.value.status_code == 404
            assert "Graph" in exc_info.value.detail
            assert "not found" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_wait_for_run_returns_streaming_response(self) -> None:
        """Verify wait_for_run returns a StreamingResponse with correct headers."""
        thread_id = "test-thread-123"
        run_id = str(uuid4())
        user = User(identity="test-user", scopes=[])
        request = _make_request()

        session_1 = AsyncMock()
        session_1.add = MagicMock()
        session_1.scalar.side_effect = [None, _make_assistant()]

        session_2 = AsyncMock()
        session_2.scalar.return_value = _make_run_orm(run_id, thread_id, output={"ok": True})

        mock_maker = _make_multi_session_maker(session_1, session_2)

        with (
            patch("aegra_api.api.runs._get_session_maker", return_value=mock_maker),
            patch("aegra_api.services.run_waiters._get_session_maker", return_value=mock_maker),
            patch("aegra_api.services.run_preparation._validate_resume_command", new_callable=AsyncMock),
            patch("aegra_api.services.run_preparation.get_langgraph_service") as mock_lg_service,
            patch("aegra_api.services.run_preparation.resolve_assistant_id", return_value="test-assistant"),
            patch("aegra_api.services.run_preparation.set_thread_status", new_callable=AsyncMock),
            patch("aegra_api.services.run_preparation.update_thread_metadata", new_callable=AsyncMock),
            patch("aegra_api.services.run_preparation.uuid4", return_value=run_id),
            patch("aegra_api.services.run_waiters.executor") as mock_executor,
            patch("aegra_api.services.run_waiters.settings") as mock_settings,
            patch("aegra_api.api.runs.active_runs", {}),
        ):
            mock_lg_service.return_value.list_graphs.return_value = ["test-graph"]
            mock_executor.wait_for_completion = AsyncMock()
            mock_executor.submit = AsyncMock()
            mock_settings.app.KEEPALIVE_INTERVAL_SECS = 5
            mock_settings.worker.BG_JOB_TIMEOUT_SECS = 3600

            response = await wait_for_run(thread_id, request, user)

        from fastapi.responses import StreamingResponse

        assert isinstance(response, StreamingResponse)
        assert response.media_type == "application/json"
        assert "Location" in response.headers
        assert f"/runs/{run_id}/join" in response.headers["Location"]


class TestWaitForRunAuthHandlers:
    """Regression coverage for the threads.create_run auth dispatch in wait_for_run.

    ``@auth.on.threads.create_run`` was previously silently skipped on the wait
    endpoint. These tests pin the dispatch (deny, merge, invocation) so the
    auth bypass cannot silently return.
    """

    @pytest.mark.asyncio
    async def test_wait_for_run_invokes_create_run_auth_handler(self) -> None:
        """The threads.create_run handler must fire for the wait endpoint."""
        thread_id = "t"
        run_id = str(uuid4())
        user = User(identity="test-user", scopes=[])
        request = _make_request()

        session = AsyncMock()
        session.scalar.return_value = None  # new thread passes ownership check

        with (
            patch("aegra_api.api.runs.handle_event", new_callable=AsyncMock) as mock_handle,
            patch("aegra_api.api.runs._prepare_run", new_callable=AsyncMock) as mock_prepare,
            patch("aegra_api.api.runs.heartbeat_wait_body", return_value=_single_event_stream()),
            patch("aegra_api.api.runs._get_session_maker", return_value=_make_session_maker(session)),
        ):
            mock_handle.return_value = None  # default-allow
            mock_prepare.return_value = (run_id, MagicMock(), MagicMock())

            response = await wait_for_run(thread_id, request, user)

        from fastapi.responses import StreamingResponse

        assert isinstance(response, StreamingResponse)
        mock_handle.assert_awaited_once()
        ctx, value = mock_handle.call_args[0]
        assert ctx.resource == "threads"
        assert ctx.action == "create_run"
        assert value["thread_id"] == thread_id

    @pytest.mark.asyncio
    async def test_wait_for_run_auth_handler_denies_with_403(self) -> None:
        """A create_run handler that denies must abort the wait before run prep."""
        thread_id = "t"
        user = User(identity="test-user", scopes=[])
        request = _make_request()

        session = AsyncMock()
        session.scalar.return_value = None

        with (
            patch("aegra_api.api.runs.handle_event", new_callable=AsyncMock) as mock_handle,
            patch("aegra_api.api.runs._prepare_run", new_callable=AsyncMock) as mock_prepare,
            patch("aegra_api.api.runs._get_session_maker", return_value=_make_session_maker(session)),
            pytest.raises(HTTPException) as exc,
        ):
            mock_handle.side_effect = HTTPException(status_code=403, detail="forbidden")

            await wait_for_run(thread_id, request, user)

        assert exc.value.status_code == 403
        assert exc.value.detail == "forbidden"
        mock_prepare.assert_not_called()  # never reached run creation

    @pytest.mark.asyncio
    async def test_wait_for_run_auth_handler_merges_config_context(self) -> None:
        """Handler-returned config/context overrides merge into the run request."""
        thread_id = "t"
        run_id = str(uuid4())
        user = User(identity="test-user", scopes=[])
        request = _make_request()
        request.config = {"original": "kept"}
        request.context = {"orig_ctx": "kept"}

        session = AsyncMock()
        session.scalar.return_value = None

        with (
            patch("aegra_api.api.runs.handle_event", new_callable=AsyncMock) as mock_handle,
            patch("aegra_api.api.runs._prepare_run", new_callable=AsyncMock) as mock_prepare,
            patch("aegra_api.api.runs.heartbeat_wait_body", return_value=_single_event_stream()),
            patch("aegra_api.api.runs._get_session_maker", return_value=_make_session_maker(session)),
        ):
            mock_handle.return_value = {"config": {"injected": True}, "context": {"injected_ctx": 2}}
            mock_prepare.return_value = (run_id, MagicMock(), MagicMock())

            await wait_for_run(thread_id, request, user)

        # Originals preserved, handler overrides merged on top.
        assert request.config == {"original": "kept", "injected": True}
        assert request.context == {"orig_ctx": "kept", "injected_ctx": 2}
        # Merged request reached run preparation.
        assert mock_prepare.call_args[0][2] is request

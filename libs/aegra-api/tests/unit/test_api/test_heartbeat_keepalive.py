"""Unit tests for heartbeat keep-alive on join/wait endpoints.

Tests the heartbeat_wait_body generator, read_run_result helper,
and the join_run endpoint's streaming behavior.
"""

import asyncio
import json
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aegra_api.api.runs import join_run
from aegra_api.core.orm import Run as RunORM
from aegra_api.models import User
from aegra_api.services.run_waiters import heartbeat_wait_body, read_run_result

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session_maker(session: AsyncMock) -> MagicMock:
    """Build a mock async_sessionmaker that always yields the given session."""
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return MagicMock(return_value=ctx)


def _make_run_orm(
    *,
    run_id: str = "run-1",
    thread_id: str = "thread-1",
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


async def _collect_body(gen: AsyncGenerator[bytes, None]) -> tuple[list[bytes], bytes]:
    """Consume the async generator and return (all_chunks, full_body)."""
    chunks: list[bytes] = []
    async for chunk in gen:
        chunks.append(chunk)
    return chunks, b"".join(chunks)


# ---------------------------------------------------------------------------
# read_run_result
# ---------------------------------------------------------------------------


class TestReadRunResult:
    @pytest.mark.asyncio
    async def test_returns_output_for_success(self) -> None:
        session = AsyncMock()
        session.scalar.return_value = _make_run_orm(output={"result": "ok"})
        maker = _make_session_maker(session)

        with patch("aegra_api.services.run_waiters._get_session_maker", return_value=maker):
            result = await read_run_result("run-1", "thread-1", "test-user")

        assert result == {"result": "ok"}

    @pytest.mark.asyncio
    async def test_reports_not_found_when_the_run_row_is_gone(self) -> None:
        """A row deleted mid-wait must not read as a run that produced nothing."""
        session = AsyncMock()
        session.scalar.return_value = None
        maker = _make_session_maker(session)

        with patch("aegra_api.services.run_waiters._get_session_maker", return_value=maker):
            result = await read_run_result("run-1", "thread-1", "test-user")

        assert result == {"__error__": {"error": "RunNotFound", "message": "Run 'run-1' no longer exists"}}

    @pytest.mark.asyncio
    async def test_returns_empty_dict_when_output_is_none(self) -> None:
        session = AsyncMock()
        session.scalar.return_value = _make_run_orm(output=None)
        maker = _make_session_maker(session)

        with patch("aegra_api.services.run_waiters._get_session_maker", return_value=maker):
            result = await read_run_result("run-1", "thread-1", "test-user")

        assert result == {}

    @pytest.mark.asyncio
    async def test_error_run_reports_its_error_message(self) -> None:
        """A failed run finalizes with an empty output; the reason lives on the row."""
        session = AsyncMock()
        session.scalar.return_value = _make_run_orm(
            status="error",
            output={},
            error_message="KeyError: 'topic'",
        )
        maker = _make_session_maker(session)

        with patch("aegra_api.services.run_waiters._get_session_maker", return_value=maker):
            result = await read_run_result("run-1", "thread-1", "test-user")

        assert result == {"__error__": {"error": "Error", "message": "KeyError: 'topic'"}}

    @pytest.mark.asyncio
    async def test_error_run_without_a_message_falls_back(self) -> None:
        session = AsyncMock()
        session.scalar.return_value = _make_run_orm(status="error", output={}, error_message=None)
        maker = _make_session_maker(session)

        with patch("aegra_api.services.run_waiters._get_session_maker", return_value=maker):
            result = await read_run_result("run-1", "thread-1", "test-user")

        assert result == {"__error__": {"error": "Error", "message": "Run failed"}}

    @pytest.mark.asyncio
    async def test_interrupted_run_keeps_its_partial_output(self) -> None:
        """An interrupt is a human-review pause, not a failure."""
        session = AsyncMock()
        session.scalar.return_value = _make_run_orm(
            status="interrupted",
            output={"__interrupt__": [{"value": "approve?"}]},
        )
        maker = _make_session_maker(session)

        with patch("aegra_api.services.run_waiters._get_session_maker", return_value=maker):
            result = await read_run_result("run-1", "thread-1", "test-user")

        assert result == {"__interrupt__": [{"value": "approve?"}]}

    @pytest.mark.asyncio
    async def test_non_terminal_run_reports_incomplete(self) -> None:
        session = AsyncMock()
        session.scalar.return_value = _make_run_orm(status="running", output={"partial": "data"})
        maker = _make_session_maker(session)

        with patch("aegra_api.services.run_waiters._get_session_maker", return_value=maker):
            result = await read_run_result("run-1", "thread-1", "test-user")

        assert result["__error__"]["error"] == "IncompleteRun"
        assert "running" in result["__error__"]["message"]

    @pytest.mark.asyncio
    async def test_non_terminal_run_after_a_timeout_reports_the_timeout(self) -> None:
        session = AsyncMock()
        session.scalar.return_value = _make_run_orm(status="running", output={"partial": "data"})
        maker = _make_session_maker(session)

        with patch("aegra_api.services.run_waiters._get_session_maker", return_value=maker):
            result = await read_run_result("run-1", "thread-1", "test-user", timed_out=True)

        assert result["__error__"]["error"] == "TimeoutError"


# ---------------------------------------------------------------------------
# heartbeat_wait_body
# ---------------------------------------------------------------------------


class TestHeartbeatWaitBody:
    @pytest.mark.asyncio
    async def test_immediate_completion_yields_json_only(self) -> None:
        """When the run completes instantly, no heartbeat bytes are emitted."""
        session = AsyncMock()
        session.scalar.return_value = _make_run_orm(output={"done": True})
        maker = _make_session_maker(session)

        with (
            patch("aegra_api.services.run_waiters._get_session_maker", return_value=maker),
            patch("aegra_api.services.run_waiters.executor") as mock_executor,
            patch("aegra_api.services.run_waiters.settings") as mock_settings,
        ):
            mock_executor.wait_for_completion = AsyncMock()
            mock_settings.app.KEEPALIVE_INTERVAL_SECS = 5
            mock_settings.worker.BG_JOB_TIMEOUT_SECS = 3600

            body = heartbeat_wait_body("run-1", "thread-1", "test-user", timeout=3600)
            chunks, full = await _collect_body(body)

        assert len(chunks) == 1
        assert json.loads(full) == {"done": True}

    @pytest.mark.asyncio
    async def test_heartbeat_bytes_emitted_during_wait(self) -> None:
        """When the run takes longer than KEEPALIVE_INTERVAL, heartbeat newlines appear."""
        session = AsyncMock()
        session.scalar.return_value = _make_run_orm(output={"result": "final"})
        maker = _make_session_maker(session)

        completion_event = asyncio.Event()

        async def slow_wait(run_id: str, *, timeout: float) -> None:
            await completion_event.wait()

        with (
            patch("aegra_api.services.run_waiters._get_session_maker", return_value=maker),
            patch("aegra_api.services.run_waiters.executor") as mock_executor,
            patch("aegra_api.services.run_waiters.settings") as mock_settings,
        ):
            mock_executor.wait_for_completion = AsyncMock(side_effect=slow_wait)
            mock_settings.app.KEEPALIVE_INTERVAL_SECS = 0.05  # 50ms for fast test
            mock_settings.worker.BG_JOB_TIMEOUT_SECS = 3600

            body = heartbeat_wait_body("run-1", "thread-1", "test-user", timeout=3600)
            chunks: list[bytes] = []

            async def consume() -> None:
                async for chunk in body:
                    chunks.append(chunk)

            consume_task = asyncio.create_task(consume())
            await asyncio.sleep(0.2)
            completion_event.set()
            await consume_task

        heartbeats = [c for c in chunks if c == b"\n"]
        assert len(heartbeats) >= 2, f"Expected at least 2 heartbeats, got {len(heartbeats)}"

        full = b"".join(chunks)
        assert json.loads(full) == {"result": "final"}

    @pytest.mark.asyncio
    async def test_timeout_yields_an_error_envelope(self) -> None:
        """A wait that gave up must not look like a run that returned nothing."""
        session = AsyncMock()
        session.scalar.return_value = _make_run_orm(status="running", output={"partial": "data"})
        maker = _make_session_maker(session)

        async def timeout_wait(run_id: str, *, timeout: float) -> None:
            raise TimeoutError(f"Run {run_id} timed out")

        with (
            patch("aegra_api.services.run_waiters._get_session_maker", return_value=maker),
            patch("aegra_api.services.run_waiters.executor") as mock_executor,
            patch("aegra_api.services.run_waiters.settings") as mock_settings,
        ):
            mock_executor.wait_for_completion = AsyncMock(side_effect=timeout_wait)
            mock_settings.app.KEEPALIVE_INTERVAL_SECS = 5
            mock_settings.worker.BG_JOB_TIMEOUT_SECS = 3600

            body = heartbeat_wait_body("run-1", "thread-1", "test-user", timeout=3600)
            chunks, full = await _collect_body(body)

        assert json.loads(full)["__error__"]["error"] == "TimeoutError"

    @pytest.mark.asyncio
    async def test_leading_whitespace_ignored_by_json_parser(self) -> None:
        """Verify the wire format: heartbeats + JSON can be parsed by json.loads."""
        session = AsyncMock()
        session.scalar.return_value = _make_run_orm(output={"value": 42})
        maker = _make_session_maker(session)

        completion_event = asyncio.Event()

        async def slow_wait(run_id: str, *, timeout: float) -> None:
            await completion_event.wait()

        with (
            patch("aegra_api.services.run_waiters._get_session_maker", return_value=maker),
            patch("aegra_api.services.run_waiters.executor") as mock_executor,
            patch("aegra_api.services.run_waiters.settings") as mock_settings,
        ):
            mock_executor.wait_for_completion = AsyncMock(side_effect=slow_wait)
            mock_settings.app.KEEPALIVE_INTERVAL_SECS = 0.02
            mock_settings.worker.BG_JOB_TIMEOUT_SECS = 3600

            body = heartbeat_wait_body("run-1", "thread-1", "test-user", timeout=3600)
            chunks: list[bytes] = []

            async def consume() -> None:
                async for chunk in body:
                    chunks.append(chunk)

            task = asyncio.create_task(consume())
            await asyncio.sleep(0.1)
            completion_event.set()
            await task

        full = b"".join(chunks)
        assert full.startswith(b"\n")  # Has heartbeat prefix
        assert json.loads(full) == {"value": 42}  # Still parseable


# ---------------------------------------------------------------------------
# join_run endpoint
# ---------------------------------------------------------------------------


class TestJoinRunEndpoint:
    """Tests for the join_run endpoint.

    join_run uses _get_session_maker from runs.py (for the initial DB check)
    and heartbeat_wait_body uses _get_session_maker from run_waiters.py
    (for reading the final output). We patch both paths via a shared maker.
    """

    @pytest.mark.asyncio
    async def test_terminal_run_returns_immediately(self) -> None:
        """Already-completed run returns JSON without streaming overhead."""
        run_orm = _make_run_orm(status="success", output={"answer": "42"})
        session = AsyncMock()
        session.scalar.return_value = run_orm
        maker = _make_session_maker(session)
        user = User(identity="test-user", scopes=[])

        with patch("aegra_api.api.runs._get_session_maker", return_value=maker):
            response = await join_run("thread-1", "run-1", user)

        assert response.media_type == "application/json"
        body = b""
        async for chunk in response.body_iterator:
            body += chunk if isinstance(chunk, bytes) else chunk.encode()
        assert json.loads(body) == {"answer": "42"}

    @pytest.mark.asyncio
    async def test_not_found_raises_404(self) -> None:
        """Non-existent run returns 404."""
        session = AsyncMock()
        session.scalar.return_value = None
        maker = _make_session_maker(session)
        user = User(identity="test-user", scopes=[])

        with patch("aegra_api.api.runs._get_session_maker", return_value=maker):
            with pytest.raises(Exception) as exc_info:
                await join_run("thread-1", "run-1", user)
            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_pending_run_returns_streaming_response(self) -> None:
        """Non-terminal run returns a StreamingResponse with heartbeat."""
        from fastapi.responses import StreamingResponse

        run_orm = _make_run_orm(status="pending", output={})
        session = AsyncMock()
        session.scalar.return_value = run_orm
        user = User(identity="test-user", scopes=[])

        # For read_run_result called inside the generator
        fetch_session = AsyncMock()
        fetch_session.scalar.return_value = _make_run_orm(status="success", output={"done": True})

        call_count = 0

        def session_factory() -> MagicMock:
            nonlocal call_count
            call_count += 1
            ctx = MagicMock()
            if call_count == 1:
                ctx.__aenter__ = AsyncMock(return_value=session)
            else:
                ctx.__aenter__ = AsyncMock(return_value=fetch_session)
            ctx.__aexit__ = AsyncMock(return_value=False)
            return ctx

        multi_maker = MagicMock(side_effect=session_factory)

        with (
            patch("aegra_api.api.runs._get_session_maker", return_value=multi_maker),
            patch("aegra_api.services.run_waiters._get_session_maker", return_value=multi_maker),
            patch("aegra_api.services.run_waiters.executor") as mock_executor,
            patch("aegra_api.services.run_waiters.settings") as mock_settings,
        ):
            mock_executor.wait_for_completion = AsyncMock()
            mock_settings.app.KEEPALIVE_INTERVAL_SECS = 5
            mock_settings.worker.BG_JOB_TIMEOUT_SECS = 3600

            response = await join_run("thread-1", "run-1", user)

        assert isinstance(response, StreamingResponse)
        assert response.media_type == "application/json"
        assert "Location" in response.headers

    @pytest.mark.asyncio
    async def test_error_run_returns_an_envelope_immediately(self) -> None:
        """The fast path reports a failure the same way the waiting path does."""
        run_orm = _make_run_orm(status="error", output={}, error_message="Graph crashed")
        session = AsyncMock()
        session.scalar.return_value = run_orm
        maker = _make_session_maker(session)
        user = User(identity="test-user", scopes=[])

        with patch("aegra_api.api.runs._get_session_maker", return_value=maker):
            response = await join_run("thread-1", "run-1", user)

        body = b""
        async for chunk in response.body_iterator:
            body += chunk if isinstance(chunk, bytes) else chunk.encode()
        assert json.loads(body) == {"__error__": {"error": "Error", "message": "Graph crashed"}}

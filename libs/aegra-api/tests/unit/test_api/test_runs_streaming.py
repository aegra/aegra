"""Unit tests for streaming run endpoints."""

from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import HTTPException

from aegra_api.api.runs import create_and_stream_run, stream_run
from aegra_api.core.orm import Assistant as AssistantORM
from aegra_api.core.orm import Run as RunORM
from aegra_api.models import RunCreate, User


class TestRunsStreamingEndpoints:
    """Test streaming run endpoints."""

    @pytest.fixture
    def mock_user(self) -> User:
        return User(identity="test-user", scopes=[])

    @pytest.fixture
    def mock_session(self) -> AsyncMock:
        session = AsyncMock()
        session.add = MagicMock()  # session.add is synchronous
        return session

    @pytest.fixture
    def sample_assistant(self) -> AssistantORM:
        return AssistantORM(
            assistant_id="test-assistant",
            graph_id="test-graph",
            config={"configurable": {"default_key": "val"}},
            context={"default_ctx": "val"},
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

    @pytest.mark.asyncio
    async def test_create_and_stream_run_success(
        self,
        mock_user: User,
        mock_session: AsyncMock,
        sample_assistant: AssistantORM,
    ) -> None:
        """Test creating and streaming a run."""
        thread_id = "test-thread-123"
        run_id = str(uuid4())

        request = RunCreate(
            assistant_id="test-assistant",
            input={"message": "stream me"},
            stream_mode=["events"],
        )

        with (
            patch("aegra_api.services.run_preparation._validate_resume_command", new_callable=AsyncMock),
            patch("aegra_api.services.run_preparation.get_langgraph_service") as mock_lg_service,
            patch(
                "aegra_api.services.run_preparation.resolve_assistant_id",
                return_value="test-assistant",
            ),
            patch("aegra_api.services.run_preparation.update_thread_metadata", new_callable=AsyncMock),
            patch("aegra_api.services.run_preparation.set_thread_status", new_callable=AsyncMock),
            patch("aegra_api.services.run_preparation.uuid4", return_value=run_id),
            patch("aegra_api.api.runs.asyncio.create_task") as mock_create_task,
            patch("aegra_api.api.runs.active_runs", {}),
            patch("aegra_api.api.runs.streaming_service.stream_run_execution") as mock_stream_exec,
        ):
            mock_lg_service.return_value.list_graphs.return_value = ["test-graph"]

            # DB setup
            mock_session.scalar.return_value = sample_assistant

            # Mock generator for streaming response
            async def mock_generator() -> AsyncGenerator:
                yield "data"

            mock_stream_exec.return_value = mock_generator()

            response = await create_and_stream_run(thread_id, request, mock_user, mock_session)

            # Verify Response
            assert response.status_code == 200
            assert response.headers["Content-Type"] == "text/event-stream"
            assert f"/runs/{run_id}" in response.headers["Location"]

            # Verify streaming service called
            mock_stream_exec.assert_called_once()

            # Verify DB interactions
            mock_session.add.assert_called_once()
            mock_session.commit.assert_called_once()

            # Verify background task creation
            mock_create_task.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "on_disconnect,expect_handler,expect_request_cancel",
        [
            (None, True, True),  # default → cancel
            ("cancel", True, True),
            ("continue", False, False),
        ],
    )
    async def test_create_and_stream_run_disconnect_handler_wiring(
        self,
        mock_user: User,
        mock_session: AsyncMock,
        sample_assistant: AssistantORM,
        on_disconnect: str | None,
        expect_handler: bool,
        expect_request_cancel: bool,
    ) -> None:
        """``client_close_handler_callable`` is wired only when on_disconnect cancels.

        Invokes the handler manually with a fake ``http.disconnect`` message
        and verifies it routes to ``broker_manager.request_cancel``. Regression
        for the transport-layer cancel wiring introduced alongside the
        EventSourceResponse migration.
        """
        thread_id = "t"
        run_id = str(uuid4())
        if on_disconnect is None:
            request = RunCreate(assistant_id="test-assistant", input={})
        else:
            request = RunCreate(
                assistant_id="test-assistant",
                input={},
                on_disconnect=on_disconnect,
            )

        async def _fake_stream() -> AsyncGenerator:
            yield "data"

        with (
            patch("aegra_api.services.run_preparation._validate_resume_command", new_callable=AsyncMock),
            patch("aegra_api.services.run_preparation.get_langgraph_service") as mock_lg_service,
            patch("aegra_api.services.run_preparation.resolve_assistant_id", return_value="test-assistant"),
            patch("aegra_api.services.run_preparation.update_thread_metadata", new_callable=AsyncMock),
            patch("aegra_api.services.run_preparation.set_thread_status", new_callable=AsyncMock),
            patch("aegra_api.services.run_preparation.uuid4", return_value=run_id),
            patch("aegra_api.api.runs.asyncio.create_task"),
            patch("aegra_api.api.runs.active_runs", {}),
            patch("aegra_api.api.runs.streaming_service.stream_run_execution", return_value=_fake_stream()),
            patch("aegra_api.api.runs.broker_manager.request_cancel", new_callable=AsyncMock) as mock_cancel,
        ):
            mock_lg_service.return_value.list_graphs.return_value = ["test-graph"]
            mock_session.scalar.return_value = sample_assistant

            response = await create_and_stream_run(thread_id, request, mock_user, mock_session)

            handler = response.client_close_handler_callable
            if not expect_handler:
                assert handler is None
                return

            assert handler is not None
            await handler({"type": "http.disconnect"})
            if expect_request_cancel:
                mock_cancel.assert_awaited_once_with(run_id, "cancel")
            else:
                mock_cancel.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_create_and_stream_run_handler_swallows_broker_errors(
        self,
        mock_user: User,
        mock_session: AsyncMock,
        sample_assistant: AssistantORM,
    ) -> None:
        """Broker failures in disconnect handler are logged, not re-raised.

        If the broker is unreachable when a client disconnects, the handler
        must not propagate the exception into sse-starlette's task group —
        otherwise the response teardown path breaks and the run leaks.
        """
        thread_id = "t"
        run_id = str(uuid4())
        request = RunCreate(assistant_id="test-assistant", input={})

        async def _fake_stream() -> AsyncGenerator:
            yield "data"

        with (
            patch("aegra_api.services.run_preparation._validate_resume_command", new_callable=AsyncMock),
            patch("aegra_api.services.run_preparation.get_langgraph_service") as mock_lg_service,
            patch("aegra_api.services.run_preparation.resolve_assistant_id", return_value="test-assistant"),
            patch("aegra_api.services.run_preparation.update_thread_metadata", new_callable=AsyncMock),
            patch("aegra_api.services.run_preparation.set_thread_status", new_callable=AsyncMock),
            patch("aegra_api.services.run_preparation.uuid4", return_value=run_id),
            patch("aegra_api.api.runs.asyncio.create_task"),
            patch("aegra_api.api.runs.active_runs", {}),
            patch("aegra_api.api.runs.streaming_service.stream_run_execution", return_value=_fake_stream()),
            patch(
                "aegra_api.api.runs.broker_manager.request_cancel",
                new_callable=AsyncMock,
                side_effect=RuntimeError("broker down"),
            ),
        ):
            mock_lg_service.return_value.list_graphs.return_value = ["test-graph"]
            mock_session.scalar.return_value = sample_assistant

            response = await create_and_stream_run(thread_id, request, mock_user, mock_session)
            handler = response.client_close_handler_callable
            assert handler is not None
            # Must not raise even though the broker side-effect blows up
            await handler({"type": "http.disconnect"})

    @pytest.mark.asyncio
    async def test_stream_run_success(self, mock_user: User, mock_session: AsyncMock) -> None:
        """Test reconnecting to existing run stream."""
        thread_id = "test-thread"
        run_id = "run-123"

        run_orm = RunORM(
            run_id=run_id,
            thread_id=thread_id,
            assistant_id="agent",
            user_id=mock_user.identity,
            status="running",
            input={},
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        mock_session.scalar.return_value = run_orm

        with patch("aegra_api.api.runs.streaming_service.stream_run_execution") as mock_stream_exec:
            # Mock generator
            async def mock_generator() -> AsyncGenerator:
                yield "data"

            mock_stream_exec.return_value = mock_generator()

            response = await stream_run(
                thread_id,
                run_id,
                last_event_id="evt-1",
                user=mock_user,
                session=mock_session,
            )

            assert response.status_code == 200
            mock_stream_exec.assert_called_once()
            # Verify passed params
            call_args = mock_stream_exec.call_args
            # First arg is run object, second is last_event_id
            assert call_args[0][0].run_id == run_id
            assert call_args[0][1] == "evt-1"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "run_status,last_event_id",
        [
            ("running", None),  # active branch
            ("success", None),  # terminal branch
        ],
    )
    async def test_stream_run_never_wires_close_handler(
        self,
        mock_user: User,
        mock_session: AsyncMock,
        run_status: str,
        last_event_id: str | None,
    ) -> None:
        """``stream_run`` (reconnect) must never wire ``client_close_handler_callable``.

        The endpoint is a reconnect-style join: multiple clients can attach
        to the same run. A single client disconnecting must NOT cancel the
        shared run — hence the endpoint deliberately omits the close handler.
        Covers both the terminal branch (early return with ``end`` event) and
        the active branch (live streaming via broker).
        """
        thread_id = "test-thread"
        run_id = "run-42"

        run_orm = RunORM(
            run_id=run_id,
            thread_id=thread_id,
            assistant_id="agent",
            user_id=mock_user.identity,
            status=run_status,
            input={},
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        mock_session.scalar.return_value = run_orm

        async def _fake_stream() -> AsyncGenerator:
            yield "data"

        with patch(
            "aegra_api.api.runs.streaming_service.stream_run_execution",
            return_value=_fake_stream(),
        ):
            response = await stream_run(
                thread_id,
                run_id,
                last_event_id=last_event_id,
                user=mock_user,
                session=mock_session,
            )

        assert response.client_close_handler_callable is None

    @pytest.mark.asyncio
    async def test_stream_run_not_found(self, mock_user: User, mock_session: AsyncMock) -> None:
        """Test streaming non-existent run."""
        mock_session.scalar.return_value = None

        with pytest.raises(HTTPException) as exc:
            await stream_run("t", "r", user=mock_user, session=mock_session)

        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_execute_run_async_error_handling(self, mock_user: User, mock_session: AsyncMock) -> None:
        """Test that errors during streaming are properly caught and sent to broker"""
        from aegra_api.models.run_job import RunExecution, RunIdentity, RunJob
        from aegra_api.services.broker import RunBroker, broker_manager
        from aegra_api.services.run_executor import execute_run as execute_run_async

        run_id = str(uuid4())
        thread_id = str(uuid4())
        graph_id = "test-graph"

        # Create broker to capture events
        broker = RunBroker(run_id)
        broker_manager._brokers[run_id] = broker

        async def failing_stream():
            """Stream that raises error immediately"""
            yield ("values", {"test": "data"})
            raise ValueError("Test error during streaming")

        with (
            patch("aegra_api.services.run_executor.get_langgraph_service") as mock_lg_service,
            patch(
                "aegra_api.services.run_executor.stream_graph_events",
                return_value=failing_stream(),
            ),
            patch("aegra_api.services.run_executor.update_run_status", new_callable=AsyncMock),
            patch("aegra_api.services.run_executor.finalize_run", new_callable=AsyncMock),
        ):
            mock_graph = MagicMock()
            mock_lg_service.return_value.get_graph.return_value.__aenter__ = AsyncMock(return_value=mock_graph)
            mock_lg_service.return_value.get_graph.return_value.__aexit__ = AsyncMock(return_value=None)

            job = RunJob(
                identity=RunIdentity(run_id=run_id, thread_id=thread_id, graph_id=graph_id),
                user=mock_user,
                execution=RunExecution(
                    input_data={},
                    config={},
                    context={},
                    stream_mode=["values"],
                ),
            )

            # Error is handled internally (no re-raise from background tasks)
            await execute_run_async(job)

            # Verify error event was sent to broker
            events = []
            try:
                async for event_id, raw_event in broker.aiter():
                    events.append((event_id, raw_event))
                    if len(events) >= 5:
                        break
            except Exception:
                pass

            # Should have error event
            error_events = [evt for _, evt in events if isinstance(evt, tuple) and evt[0] == "error"]
            assert len(error_events) > 0, "Error event should be sent to broker"

            error_event = error_events[0]
            assert error_event[0] == "error"
            assert error_event[1]["error"] == "ValueError"
            assert error_event[1]["message"] == "ValueError: execution failed"

"""Unit tests for standard run endpoints (create, get, list, update, join)."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import HTTPException

from aegra_api.api.runs import create_run, get_run, join_run, list_runs, update_run
from aegra_api.core.orm import Assistant as AssistantORM
from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import Thread as ThreadORM
from aegra_api.models import Run, RunCreate, RunStatus, User


class TestRunsEndpoints:
    """Test standard run endpoints."""

    @pytest.fixture
    def mock_user(self) -> User:
        return User(identity="test-user", scopes=[])

    @pytest.fixture
    def mock_session(self) -> AsyncMock:
        session = AsyncMock()
        session.refresh = AsyncMock()
        session.add = MagicMock()  # session.add is synchronous
        return session

    @pytest.fixture
    def sample_thread(self) -> ThreadORM:
        return ThreadORM(
            thread_id="test-thread-123",
            user_id="test-user",
            status="idle",
            metadata_json={},
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

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
    async def test_create_run_success(
        self, mock_user: User, mock_session: AsyncMock, sample_assistant: AssistantORM
    ) -> None:
        """Test successful run creation."""
        thread_id = "test-thread-123"
        run_id = str(uuid4())

        request = RunCreate(
            assistant_id="test-assistant",
            input={"message": "hello"},
            config={"configurable": {"key": "value"}},
        )

        # Mock dependencies
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
        ):
            mock_lg_service.return_value.list_graphs.return_value = ["test-graph"]

            # DB setup: first scalar = thread ownership check (None = new thread), second = assistant
            mock_session.scalar.side_effect = [None, sample_assistant]
            # No in-flight run on the thread → multitask gate lets this run start now.
            no_active = MagicMock()
            no_active.all.return_value = []
            mock_session.scalars.return_value = no_active

            result = await create_run(thread_id, request, mock_user, mock_session)

            # Assertions
            assert isinstance(result, Run)
            assert result.run_id == run_id
            assert result.thread_id == thread_id
            assert result.status == "pending"
            assert result.input == {"message": "hello"}

            # Verify DB interactions
            mock_session.add.assert_called_once()
            mock_session.commit.assert_called_once()

            # Verify background task creation
            mock_create_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_run_assistant_not_found(
        self, mock_user: User, mock_session: AsyncMock, sample_thread: ThreadORM
    ) -> None:
        """Test creation with non-existent assistant."""
        thread_id = "test-thread-123"
        request = RunCreate(assistant_id="nonexistent", input={})

        with (
            patch("aegra_api.services.run_preparation._validate_resume_command", new_callable=AsyncMock),
            patch("aegra_api.services.run_preparation.get_langgraph_service") as mock_lg_service,
            patch("aegra_api.services.run_preparation.resolve_assistant_id", return_value="nonexistent"),
        ):
            mock_lg_service.return_value.list_graphs.return_value = ["test-graph"]

            # First scalar call: thread ownership check (pass). Second: assistant lookup (None).
            mock_session.scalar.side_effect = [sample_thread, None]

            with pytest.raises(HTTPException) as exc:
                await create_run(thread_id, request, mock_user, mock_session)

            assert exc.value.status_code == 404
            assert "Assistant" in str(exc.value.detail) and "not found" in str(exc.value.detail)

    @pytest.mark.asyncio
    async def test_create_run_graph_not_found(
        self, mock_user: User, mock_session: AsyncMock, sample_assistant: AssistantORM
    ) -> None:
        """Test creation where assistant's graph is missing."""
        thread_id = "test-thread-123"
        request = RunCreate(assistant_id="test-assistant", input={})

        with (
            patch("aegra_api.services.run_preparation._validate_resume_command", new_callable=AsyncMock),
            patch("aegra_api.services.run_preparation.get_langgraph_service") as mock_lg_service,
            patch(
                "aegra_api.services.run_preparation.resolve_assistant_id",
                return_value="test-assistant",
            ),
        ):
            # Graph not in available graphs
            mock_lg_service.return_value.list_graphs.return_value = ["other-graph"]

            mock_session.scalar.side_effect = [None, sample_assistant]

            with pytest.raises(HTTPException) as exc:
                await create_run(thread_id, request, mock_user, mock_session)

            assert exc.value.status_code == 404
            assert "Graph" in str(exc.value.detail)

    @pytest.mark.asyncio
    async def test_create_run_config_context_allowed(
        self, mock_user: User, mock_session: AsyncMock, sample_thread: ThreadORM
    ) -> None:
        """Test both configurable and context are accepted."""
        thread_id = "test-thread-123"
        request = RunCreate(
            assistant_id="test-assistant",
            input={},
            config={"configurable": {"a": 1}},
            context={"b": 1},
        )

        with (
            patch("aegra_api.services.run_preparation._validate_resume_command", new_callable=AsyncMock),
            patch("aegra_api.services.run_preparation.get_langgraph_service") as mock_lg_service,
            patch(
                "aegra_api.services.run_preparation.resolve_assistant_id",
                return_value="test-assistant",
            ),
        ):
            mock_lg_service.return_value.list_graphs.return_value = ["test-graph"]
            # First scalar call: thread ownership check (pass). Second: assistant lookup (None).
            mock_session.scalar.side_effect = [sample_thread, None]

            with pytest.raises(HTTPException) as exc:
                await create_run(thread_id, request, mock_user, mock_session)

            # Validation conflict is removed; request proceeds to assistant lookup
            assert exc.value.status_code == 404
            assert "Assistant" in str(exc.value.detail) and "not found" in str(exc.value.detail)

    @pytest.mark.asyncio
    async def test_get_run_success(self, mock_user: User, mock_session: AsyncMock) -> None:
        """Test retrieving an existing run."""
        thread_id = "test-thread"
        run_id = "run-123"

        run_orm = RunORM(
            run_id=run_id,
            thread_id=thread_id,
            assistant_id="agent",
            user_id=mock_user.identity,
            status="pending",
            input={},
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        mock_session.scalar.return_value = run_orm

        result = await get_run(thread_id, run_id, mock_user, mock_session)

        assert result.run_id == run_id
        assert result.status == "pending"
        mock_session.refresh.assert_called_once_with(run_orm)

    @pytest.mark.asyncio
    async def test_get_run_not_found(self, mock_user: User, mock_session: AsyncMock) -> None:
        """Test retrieving non-existent run."""
        mock_session.scalar.return_value = None

        with pytest.raises(HTTPException) as exc:
            await get_run("thread", "missing", mock_user, mock_session)

        assert exc.value.status_code == 404
        assert "Run" in str(exc.value.detail)

    @pytest.mark.asyncio
    async def test_list_runs_success(self, mock_user: User, mock_session: AsyncMock) -> None:
        """Test listing runs."""
        thread_id = "test-thread"

        runs = [
            RunORM(
                run_id=f"run-{i}",
                thread_id=thread_id,
                assistant_id="agent",
                user_id=mock_user.identity,
                status="success",
                input={},
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
            for i in range(3)
        ]

        mock_result = MagicMock()
        mock_result.all.return_value = runs
        mock_session.scalars.return_value = mock_result

        result = await list_runs(
            thread_id,
            limit=10,
            offset=0,
            status=None,
            user=mock_user,
            session=mock_session,
        )

        assert len(result) == 3
        assert result[0].run_id == "run-0"

    @pytest.mark.asyncio
    async def test_update_run_cancel(self, mock_user: User, mock_session: AsyncMock) -> None:
        """Test cancelling a run."""
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

        # scalar called twice: first to find for update, second to return
        mock_session.scalar.side_effect = [run_orm, run_orm]

        with (
            patch(
                "aegra_api.api.runs.streaming_service.interrupt_run",
                new_callable=AsyncMock,
            ) as mock_interrupt,
            patch(
                "aegra_api.api.runs.terminalize_user_cancel",
                new_callable=AsyncMock,
            ) as mock_terminalize,
        ):
            result = await update_run(
                thread_id,
                run_id,
                RunStatus(run_id=run_id, status="interrupted"),
                mock_user,
                mock_session,
            )

            mock_interrupt.assert_called_once_with(run_id)
            # The endpoint converges through terminalize_user_cancel (thread reset +
            # queued dispatch happen there), not through a raw status UPDATE.
            mock_terminalize.assert_awaited_once_with(run_id, thread_id)
            assert result.run_id == run_id

    @pytest.mark.asyncio
    async def test_update_run_not_found(self, mock_user: User, mock_session: AsyncMock) -> None:
        """Test updating non-existent run."""
        mock_session.scalar.return_value = None

        with pytest.raises(HTTPException) as exc:
            await update_run(
                "t",
                "r",
                RunStatus(run_id="r", status="interrupted"),
                mock_user,
                mock_session,
            )

        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_join_run_terminal_state(self, mock_user: User, mock_session: AsyncMock) -> None:
        """Test joining a completed run returns output immediately via StreamingResponse."""
        import json

        run_orm = RunORM(
            run_id="run-1",
            thread_id="thread-1",
            user_id=mock_user.identity,
            status="success",
            input={},
            output={"result": "done"},
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        mock_session.scalar.return_value = run_orm

        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_session)
        ctx.__aexit__ = AsyncMock(return_value=False)
        mock_maker = MagicMock(return_value=ctx)

        with patch("aegra_api.api.runs._get_session_maker", return_value=mock_maker):
            response = await join_run("thread-1", "run-1", mock_user)

        # join_run now returns StreamingResponse; consume body to get JSON
        assert response.media_type == "application/json"
        body = b""
        async for chunk in response.body_iterator:
            body += chunk if isinstance(chunk, bytes) else chunk.encode()
        assert json.loads(body) == {"result": "done"}

    @pytest.mark.asyncio
    async def test_join_run_active_state(self, mock_user: User, mock_session: AsyncMock) -> None:
        """Test joining an active run returns a StreamingResponse with heartbeat."""
        from fastapi.responses import StreamingResponse

        # Setup run initially in running state
        run_orm_running = RunORM(
            run_id="run-1",
            thread_id="thread-1",
            user_id=mock_user.identity,
            status="running",
            input={},
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        mock_session.scalar.return_value = run_orm_running

        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_session)
        ctx.__aexit__ = AsyncMock(return_value=False)
        mock_maker = MagicMock(return_value=ctx)

        # Mock executor and settings for the heartbeat body
        with (
            patch("aegra_api.api.runs._get_session_maker", return_value=mock_maker),
            patch("aegra_api.services.run_waiters._get_session_maker", return_value=mock_maker),
            patch("aegra_api.services.run_waiters.executor") as mock_executor,
            patch("aegra_api.services.run_waiters.settings") as mock_settings,
        ):
            mock_executor.wait_for_completion = AsyncMock()
            mock_settings.app.KEEPALIVE_INTERVAL_SECS = 5
            mock_settings.worker.BG_JOB_TIMEOUT_SECS = 3600

            response = await join_run("thread-1", "run-1", mock_user)

        assert isinstance(response, StreamingResponse)
        assert response.media_type == "application/json"
        assert "Location" in response.headers


# Note: _resolve_context was removed from runs.py during the worker architecture
# refactor — context resolution is now handled in services/run_preparation.py.
# The equivalent tests live in tests/unit/test_services/.

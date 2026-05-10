"""Integration tests for runs CRUD operations"""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from aegra_api.core.cancellation_state import cancellations
from tests.fixtures.clients import create_test_app, make_client
from tests.fixtures.database import DummySessionBase
from tests.fixtures.session_fixtures import BasicSession, override_session_dependency
from tests.fixtures.test_helpers import DummyRun, DummyThread


def _thread_row(thread_id="test-thread-123", status="idle", metadata=None, user_id="test-user"):
    """Create a mock thread ORM object"""
    thread = DummyThread(thread_id, status, metadata, user_id)

    # Add ORM-specific attributes
    thread.metadata_json = metadata or {}

    class _Col:
        def __init__(self, name):
            self.name = name

    class _T:
        columns = [
            _Col("thread_id"),
            _Col("status"),
            _Col("metadata"),
            _Col("user_id"),
            _Col("created_at"),
            _Col("updated_at"),
        ]

    thread.__table__ = _T()
    return thread


def _assistant_row(assistant_id="test-assistant-123", graph_id="test-graph", user_id="test-user"):
    """Create a mock assistant ORM object"""
    from tests.fixtures.test_helpers import make_assistant

    assistant = make_assistant(assistant_id=assistant_id, graph_id=graph_id, user_id=user_id)

    # Add ORM-specific attributes
    assistant.graph_id = graph_id

    return assistant


def _run_row(
    run_id="test-run-123",
    thread_id="test-thread-123",
    assistant_id="test-assistant-123",
    status="running",
    user_id="test-user",
    metadata=None,
    input_data=None,
    output_data=None,
):
    """Create a mock run ORM object"""
    run = DummyRun(
        run_id,
        thread_id,
        assistant_id,
        status,
        user_id,
        metadata,
        input_data,
        output_data,
    )

    # Add ORM-specific attributes
    run.metadata_json = metadata or {}
    run.error_message = None
    run.config = {}
    run.context = {}

    class _Col:
        def __init__(self, name):
            self.name = name

    class _T:
        columns = [
            _Col("run_id"),
            _Col("thread_id"),
            _Col("assistant_id"),
            _Col("status"),
            _Col("user_id"),
            _Col("metadata"),
            _Col("input"),
            _Col("output"),
            _Col("error_message"),
            _Col("config"),
            _Col("context"),
            _Col("created_at"),
            _Col("updated_at"),
        ]

    run.__table__ = _T()
    return run


def _make_session_maker(session_instance: DummySessionBase) -> MagicMock:
    """Return a callable mimicking ``async_sessionmaker``.

    Calling the returned object produces an async context manager that yields
    *session_instance*, matching the ``async with maker() as session:`` pattern.
    """
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session_instance)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return MagicMock(return_value=ctx)


class TestCreateRun:
    """Test POST /threads/{thread_id}/runs"""

    def test_create_run_validation_error_no_input_or_command(self):
        """Test that run creation requires either input or command"""
        app = create_test_app(include_runs=True, include_threads=False)

        # Use BasicSession from shared fixtures

        override_session_dependency(app, BasicSession)
        client = make_client(app)

        resp = client.post(
            "/threads/test-thread-123/runs",
            json={"assistant_id": "asst-123"},
        )

        # Should get validation error (422) for missing input/command
        assert resp.status_code == 422


class TestGetRun:
    """Test GET /threads/{thread_id}/runs/{run_id}"""

    def test_get_run_success(self):
        """Test getting an existing run"""
        app = create_test_app(include_runs=True, include_threads=False)

        run = _run_row(status="success")

        class Session(DummySessionBase):
            async def scalar(self, _stmt):
                return run

        override_session_dependency(app, Session)
        client = make_client(app)

        resp = client.get("/threads/test-thread-123/runs/test-run-123")

        assert resp.status_code == 200
        data = resp.json()
        assert data["run_id"] == "test-run-123"
        assert data["thread_id"] == "test-thread-123"
        assert data["status"] == "success"

    def test_get_run_not_found(self):
        """Test getting a non-existent run"""
        app = create_test_app(include_runs=True, include_threads=False)

        class Session(DummySessionBase):
            async def scalar(self, _stmt):
                return None

        override_session_dependency(app, Session)
        client = make_client(app)

        resp = client.get("/threads/test-thread-123/runs/nonexistent")

        assert resp.status_code == 404


class TestListRuns:
    """Test GET /threads/{thread_id}/runs"""

    def test_list_runs_success(self):
        """Test listing runs for a thread"""
        app = create_test_app(include_runs=True, include_threads=False)

        runs = [
            _run_row("run-1", status="success"),
            _run_row("run-2", status="running"),
            _run_row("run-3", status="pending"),
        ]

        class Session(DummySessionBase):
            async def scalars(self, _stmt):
                class Result:
                    def all(self):
                        return runs

                return Result()

        override_session_dependency(app, Session)
        client = make_client(app)

        resp = client.get("/threads/test-thread-123/runs")

        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 3
        assert data[0]["run_id"] == "run-1"

    def test_list_runs_empty(self):
        """Test listing runs when thread has none"""
        app = create_test_app(include_runs=True, include_threads=False)

        class Session(DummySessionBase):
            async def scalars(self, _stmt):
                class Result:
                    def all(self):
                        return []

                return Result()

        override_session_dependency(app, BasicSession)
        client = make_client(app)

        resp = client.get("/threads/test-thread-123/runs")

        assert resp.status_code == 200
        data = resp.json()
        assert data == []

    def test_list_runs_with_limit(self):
        """Test listing runs with limit parameter"""
        app = create_test_app(include_runs=True, include_threads=False)

        runs = [_run_row(f"run-{i}") for i in range(5)]

        class Session(DummySessionBase):
            async def scalars(self, _stmt):
                class Result:
                    def all(self):
                        return runs[:2]  # Simulate limit

                return Result()

        override_session_dependency(app, BasicSession)
        client = make_client(app)

        resp = client.get("/threads/test-thread-123/runs?limit=2")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) <= 5

    def test_list_runs_with_offset(self):
        """Test listing runs with offset parameter"""
        app = create_test_app(include_runs=True, include_threads=False)

        runs = [_run_row(f"run-{i}") for i in range(10)]

        class Session(DummySessionBase):
            async def scalars(self, _stmt):
                class Result:
                    def all(self):
                        return runs[5:]  # Simulate offset

                return Result()

        override_session_dependency(app, BasicSession)
        client = make_client(app)

        resp = client.get("/threads/test-thread-123/runs?offset=5")

        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)


class TestUpdateRun:
    """Test PATCH /threads/{thread_id}/runs/{run_id}"""

    def test_update_run_validation(self):
        """Test updating run requires valid payload"""
        app = create_test_app(include_runs=True, include_threads=False)

        # Use BasicSession from shared fixtures

        override_session_dependency(app, BasicSession)
        client = make_client(app)

        # Empty update should fail validation
        resp = client.patch(
            "/threads/test-thread-123/runs/test-run-123",
            json={},
        )

        assert resp.status_code == 422


class TestCancelRun:
    """Test POST /threads/{thread_id}/runs/{run_id}/cancel"""

    def test_cancel_run_not_found(self):
        """Test canceling a non-existent run"""
        app = create_test_app(include_runs=True, include_threads=False)

        class Session(DummySessionBase):
            async def scalar(self, _stmt):
                return None

        override_session_dependency(app, BasicSession)
        client = make_client(app)

        resp = client.post("/threads/test-thread-123/runs/nonexistent/cancel")

        assert resp.status_code == 404

    def test_cancel_run_success(self):
        """Test successfully canceling a run"""
        app = create_test_app(include_runs=True, include_threads=False)

        run = _run_row(status="running")

        class Session(DummySessionBase):
            async def scalar(self, _stmt):
                return run

            async def execute(self, _stmt):
                pass

            async def commit(self):
                pass

        override_session_dependency(app, Session)
        client = make_client(app)

        with patch("aegra_api.api.runs.streaming_service") as mock_streaming:
            mock_streaming.cancel_run = AsyncMock()

            resp = client.post("/threads/test-thread-123/runs/test-run-123/cancel")

            assert resp.status_code == 200

    def test_cancel_pending_falls_through_to_running_when_worker_won_cas_race(self) -> None:
        """If a worker picks up a pending run between our SELECT and the CAS
        UPDATE, ``try_cancel_pending`` returns ``False`` and the endpoint
        must (a) re-fetch, (b) confirm the run is still active, (c) hand off
        to ``cancel_running_run`` to broadcast + write the safety-net status.

        The CAS-success metric must NOT be incremented on this path —
        execute_run owns the completed counter once the worker has it."""
        app = create_test_app(include_runs=True, include_threads=False)

        pending_run = _run_row(status="pending")
        pending_run.execution_params = {"graph_id": "graph-cas"}
        running_run = _run_row(status="running")
        running_run.execution_params = {"graph_id": "graph-cas"}

        scalar_call_count = 0

        class Session(DummySessionBase):
            async def scalar(self, _stmt: object) -> object:
                nonlocal scalar_call_count
                scalar_call_count += 1
                # 1st call: initial SELECT (pending)
                # 2nd call: re-fetch after CAS rowcount=0 (now running)
                # 3rd+ call: final reload at end of endpoint (still running)
                if scalar_call_count == 1:
                    return pending_run
                return running_run

            async def execute(self, _stmt: object) -> object:
                # CAS UPDATE returns rowcount=0 — worker already claimed.
                # mark_interrupted_unless_terminal also calls execute() but
                # rowcount isn't read on that path.
                class Result:
                    rowcount = 0

                return Result()

            async def rollback(self) -> None:
                pass

            async def commit(self) -> None:
                pass

        override_session_dependency(app, Session)
        client = make_client(app)

        with (
            patch("aegra_api.services.run_cancellation.streaming_service") as mock_streaming,
            patch(
                "aegra_api.services.run_cancellation.get_worker_metrics",
                return_value=None,
            ) as mock_metrics,
        ):
            mock_streaming.cancel_run = AsyncMock()

            resp = client.post("/threads/test-thread-123/runs/test-run-123/cancel")

            assert resp.status_code == 200
            # CAS lost → fall-through path broadcasts via streaming_service.
            mock_streaming.cancel_run.assert_awaited_once_with("test-run-123")
            # CAS-success metric path must not run; get_worker_metrics is only
            # called inside try_cancel_pending on a CAS hit.
            mock_metrics.assert_not_called()

    def test_cancel_terminal_run_short_circuits_without_marking(self) -> None:
        """Cancel for a run already in a terminal state must NOT seed the
        user-cancellation set — leaving an entry behind would leak across
        future runs that reuse the same run_id (or just grow unbounded)."""
        app = create_test_app(include_runs=True, include_threads=False)
        # Use a fresh UUID so we don't share state with other tests in the worker.
        run_id = str(uuid4())
        run = _run_row(run_id=run_id, status="success")

        class Session(DummySessionBase):
            async def scalar(self, _stmt: object) -> object:
                return run

            async def execute(self, _stmt: object) -> object:
                pytest.fail("execute() must not be called for a terminal-state cancel")

            async def commit(self) -> None:
                pytest.fail("commit() must not be called for a terminal-state cancel")

        override_session_dependency(app, Session)
        client = make_client(app)

        with patch("aegra_api.api.runs.streaming_service") as mock_streaming:
            mock_streaming.cancel_run = AsyncMock()

            resp = client.post(f"/threads/test-thread-123/runs/{run_id}/cancel")

        assert resp.status_code == 200
        # Critical: terminal guard means we never marked the run
        assert cancellations.reason_of(run_id) is None
        mock_streaming.cancel_run.assert_not_awaited()


class TestDeleteRun:
    """Test DELETE /threads/{thread_id}/runs/{run_id}"""

    def test_delete_run_not_found(self):
        """Test deleting a non-existent run"""
        app = create_test_app(include_runs=True, include_threads=False)

        class Session(DummySessionBase):
            async def scalar(self, _stmt):
                return None

        override_session_dependency(app, BasicSession)
        client = make_client(app)

        resp = client.delete("/threads/test-thread-123/runs/nonexistent")

        assert resp.status_code == 404

    def test_delete_run_active_not_allowed(self):
        """Test deleting an active run is not allowed"""
        app = create_test_app(include_runs=True, include_threads=False)

        run = _run_row(status="running")

        class Session(DummySessionBase):
            async def scalar(self, _stmt):
                return run

        override_session_dependency(app, BasicSession)
        client = make_client(app)

        resp = client.delete("/threads/test-thread-123/runs/test-run-123")

        # Should return error (400, 409, etc.) for active run
        assert resp.status_code >= 400
        assert resp.status_code < 500

    def test_delete_run_success(self):
        """Test successfully deleting a completed run"""
        app = create_test_app(include_runs=True, include_threads=False)

        run = _run_row(status="success")

        class Session(DummySessionBase):
            async def scalar(self, _stmt):
                return run

            async def execute(self, _stmt):
                pass

            async def commit(self):
                pass

        override_session_dependency(app, Session)
        client = make_client(app)

        resp = client.delete("/threads/test-thread-123/runs/test-run-123")

        assert resp.status_code == 204


class TestJoinRun:
    """Test GET /threads/{thread_id}/runs/{run_id}/join

    join_run manages sessions manually via _get_session_maker (not Depends),
    so we patch the maker to return a mock async context manager.
    """

    def test_join_run_not_found(self):
        """Test joining a non-existent run"""
        app = create_test_app(include_runs=True, include_threads=False)

        class Session(DummySessionBase):
            async def scalar(self, _stmt):
                return None

        override_session_dependency(app, BasicSession)
        client = make_client(app)

        with patch("aegra_api.api.runs._get_session_maker", return_value=_make_session_maker(Session())):
            resp = client.get("/threads/test-thread-123/runs/nonexistent/join")

        assert resp.status_code == 404

    def test_join_run_already_completed(self):
        """Test joining an already completed run"""
        app = create_test_app(include_runs=True, include_threads=False)

        run = _run_row(status="success")

        class Session(DummySessionBase):
            async def scalar(self, _stmt):
                return run

        override_session_dependency(app, Session)
        client = make_client(app)

        with patch("aegra_api.api.runs._get_session_maker", return_value=_make_session_maker(Session())):
            resp = client.get("/threads/test-thread-123/runs/test-run-123/join")

        assert resp.status_code == 200
        # Response may vary depending on run state
        assert isinstance(resp.json(), (dict, list))


class TestStreamRun:
    """Test GET /threads/{thread_id}/runs/{run_id}/stream"""

    def test_stream_run_not_found(self):
        """Test streaming a non-existent run"""
        app = create_test_app(include_runs=True, include_threads=False)

        class Session(DummySessionBase):
            async def scalar(self, _stmt):
                return None

        override_session_dependency(app, BasicSession)
        client = make_client(app)

        resp = client.get("/threads/test-thread-123/runs/nonexistent/stream")

        assert resp.status_code == 404


class TestRunWithInput:
    """Test creating runs with input vs command"""

    def test_create_run_with_input_validation(self):
        """Test creating run with input passes validation"""
        app = create_test_app(include_runs=True, include_threads=False)

        # Use BasicSession from shared fixtures

        override_session_dependency(app, BasicSession)
        client = make_client(app)

        resp = client.post(
            "/threads/test-thread-123/runs",
            json={
                "assistant_id": "test-assistant-123",
                "input": {"message": "Hello"},
            },
        )

        # Should not be a validation error
        assert resp.status_code != 422


class TestRunStatuses:
    """Test filtering runs by status"""

    def test_list_runs_filter_by_status(self):
        """Test filtering runs by status"""
        app = create_test_app(include_runs=True, include_threads=False)

        runs = [
            _run_row("run-1", status="success"),
            _run_row("run-2", status="success"),
        ]

        class Session(DummySessionBase):
            async def scalars(self, _stmt):
                class Result:
                    def all(self):
                        return runs

                return Result()

        override_session_dependency(app, BasicSession)
        client = make_client(app)

        resp = client.get("/threads/test-thread-123/runs?status=completed")

        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)


class TestWaitForRun:
    """Test POST /threads/{thread_id}/runs/wait

    wait_for_run manages sessions via _get_session_maker (not Depends), so tests
    that reach the handler must patch it. Pure-validation tests (422) don't need it.
    """

    def test_wait_for_run_validation_no_input(self):
        """Test that wait endpoint requires input or command"""
        app = create_test_app(include_runs=True, include_threads=False)

        override_session_dependency(app, BasicSession)
        client = make_client(app)

        resp = client.post(
            "/threads/test-thread-123/runs/wait",
            json={"assistant_id": "asst-123"},
        )

        # Should get validation error (422) for missing input/command
        assert resp.status_code == 422

    def test_wait_for_run_thread_not_found(self):
        """Test wait endpoint with non-existent thread"""
        app = create_test_app(include_runs=True, include_threads=False)

        class Session(DummySessionBase):
            async def scalar(self, _stmt):
                # Return None for thread lookup
                return None

        override_session_dependency(app, Session)
        client = make_client(app)

        with patch("aegra_api.api.runs._get_session_maker", return_value=_make_session_maker(Session())):
            resp = client.post(
                "/threads/nonexistent/runs/wait",
                json={
                    "assistant_id": "asst-123",
                    "input": {"message": "test"},
                },
            )

        # Should get 404 when thread doesn't exist
        assert resp.status_code == 404

    def test_wait_for_run_assistant_not_found(self):
        """Test wait endpoint with non-existent assistant"""
        app = create_test_app(include_runs=True, include_threads=False)

        class Session(DummySessionBase):
            async def scalar(self, _stmt):
                # Return None for assistant lookup (thread validation is skipped in wait endpoint)
                return None

            async def execute(self, _stmt):
                pass

            async def commit(self):
                pass

        override_session_dependency(app, Session)
        client = make_client(app)

        with patch("aegra_api.api.runs._get_session_maker", return_value=_make_session_maker(Session())):
            resp = client.post(
                "/threads/test-thread-123/runs/wait",
                json={
                    "assistant_id": "nonexistent-asst",
                    "input": {"message": "test"},
                },
            )

        # Should get 404 when assistant doesn't exist
        assert resp.status_code == 404

    def test_wait_for_run_with_interrupt_before(self):
        """Test wait endpoint accepts interrupt_before parameter"""
        app = create_test_app(include_runs=True, include_threads=False)

        override_session_dependency(app, BasicSession)
        client = make_client(app)

        with patch("aegra_api.api.runs._get_session_maker", return_value=_make_session_maker(BasicSession())):
            resp = client.post(
                "/threads/test-thread-123/runs/wait",
                json={
                    "assistant_id": "asst-123",
                    "input": {"message": "test"},
                    "interrupt_before": ["node1", "node2"],
                },
            )

        # Should accept the parameter (may fail later in execution, but not validation)
        assert resp.status_code != 422

    def test_wait_for_run_with_stream_subgraphs(self):
        """Test wait endpoint accepts stream_subgraphs parameter"""
        app = create_test_app(include_runs=True, include_threads=False)

        override_session_dependency(app, BasicSession)
        client = make_client(app)

        with patch("aegra_api.api.runs._get_session_maker", return_value=_make_session_maker(BasicSession())):
            resp = client.post(
                "/threads/test-thread-123/runs/wait",
                json={
                    "assistant_id": "asst-123",
                    "input": {"message": "test"},
                    "stream_subgraphs": True,
                },
            )

        # Should accept the parameter
        assert resp.status_code != 422

    def test_wait_for_run_with_command(self):
        """Test wait endpoint with command instead of input"""
        app = create_test_app(include_runs=True, include_threads=False)

        override_session_dependency(app, BasicSession)
        client = make_client(app)

        with patch("aegra_api.api.runs._get_session_maker", return_value=_make_session_maker(BasicSession())):
            resp = client.post(
                "/threads/test-thread-123/runs/wait",
                json={
                    "assistant_id": "asst-123",
                    "command": {"resume": "value"},
                },
            )

        # Should accept command parameter
        assert resp.status_code != 422

    def test_wait_for_run_cannot_have_both_input_and_command(self):
        """Test wait endpoint rejects both input and command"""
        app = create_test_app(include_runs=True, include_threads=False)

        override_session_dependency(app, BasicSession)
        client = make_client(app)

        resp = client.post(
            "/threads/test-thread-123/runs/wait",
            json={
                "assistant_id": "asst-123",
                "input": {"message": "test"},
                "command": {"resume": "value"},
            },
        )

        # Should reject having both
        assert resp.status_code == 422

    def test_wait_for_run_resume_requires_interrupted_thread(self):
        """Test wait endpoint with resume command requires interrupted thread"""
        app = create_test_app(include_runs=True, include_threads=False)

        # Thread is idle, not interrupted
        thread = _thread_row(status="idle")

        class Session(DummySessionBase):
            async def scalar(self, _stmt):
                return thread

        override_session_dependency(app, Session)
        client = make_client(app)

        with patch("aegra_api.api.runs._get_session_maker", return_value=_make_session_maker(Session())):
            resp = client.post(
                "/threads/test-thread-123/runs/wait",
                json={
                    "assistant_id": "asst-123",
                    "command": {"resume": "value"},
                },
            )

        # Should fail because thread is not interrupted
        assert resp.status_code == 400

    def test_wait_for_run_with_config_and_context_allowed(self) -> None:
        """Test wait endpoint allows both configurable and context."""
        app = create_test_app(include_runs=True, include_threads=False)

        override_session_dependency(app, BasicSession)
        client = make_client(app)

        with patch("aegra_api.api.runs._get_session_maker", return_value=_make_session_maker(BasicSession())):
            resp = client.post(
                "/threads/test-thread-123/runs/wait",
                json={
                    "assistant_id": "asst-123",
                    "input": {"message": "test"},
                    "config": {"configurable": {"key": "value"}},
                    "context": {"key": "value"},
                },
            )

        # Validation conflict is removed; request proceeds to assistant lookup
        assert resp.status_code == 404


class TestCreateRunValidation:
    """Additional validation tests for create_run."""

    def test_create_run_config_context_allowed(self) -> None:
        """Test create_run allows both configurable and context."""
        app = create_test_app(include_runs=True, include_threads=False)
        override_session_dependency(app, BasicSession)
        client = make_client(app)

        resp = client.post(
            "/threads/test-thread-123/runs",
            json={
                "assistant_id": "asst-123",
                "input": {"message": "test"},
                "config": {"configurable": {"key": "value"}},
                "context": {"key": "value"},
            },
        )
        # Validation conflict is removed; request proceeds to assistant lookup
        assert resp.status_code == 404

    def test_create_run_assistant_not_found(self):
        """Test create_run with non-existent assistant."""
        app = create_test_app(include_runs=True, include_threads=False)

        class Session(DummySessionBase):
            async def scalar(self, _stmt):
                return None  # Assistant not found

        override_session_dependency(app, Session)
        client = make_client(app)

        resp = client.post(
            "/threads/test-thread-123/runs",
            json={
                "assistant_id": "nonexistent",
                "input": {"message": "test"},
            },
        )
        assert resp.status_code == 404


class TestWaitForRunTimeouts:
    """Test wait_for_run timeout behavior.

    wait_for_run now returns a StreamingResponse wrapping heartbeat_wait_body.
    On timeout, the heartbeat generator reads the run's current output from DB
    and yields it as the final JSON chunk.
    """

    def test_wait_for_run_timeout(self):
        """Test that wait_for_run returns current state on timeout."""
        app = create_test_app(include_runs=True, include_threads=False)

        # Mock assistant and run
        assistant = _assistant_row()
        run = _run_row(status="running")
        run.output = {"partial": "data"}

        thread = _thread_row()

        class Session(DummySessionBase):
            async def scalar(self, stmt):
                stmt_str = str(stmt).lower()
                if "from thread" in stmt_str:
                    return thread
                if "from assistant" in stmt_str:
                    return assistant
                if "from run" in stmt_str:
                    return run
                return None

            async def refresh(self, obj):
                pass

            def add(self, obj):
                pass

            async def commit(self):
                pass

            async def execute(self, stmt):
                class Result:
                    rowcount = 1

                return Result()

        mock_maker = _make_session_maker(Session())

        # Mock executor so wait_for_completion raises TimeoutError immediately
        mock_executor = MagicMock()
        mock_executor.wait_for_completion = AsyncMock(side_effect=TimeoutError)
        mock_executor.submit = AsyncMock(return_value=None)

        override_session_dependency(app, Session)
        client = make_client(app)

        with (
            patch("aegra_api.api.runs._get_session_maker", return_value=mock_maker),
            patch("aegra_api.services.run_waiters._get_session_maker", return_value=mock_maker),
            patch("aegra_api.services.run_waiters.executor", mock_executor),
            patch("aegra_api.services.run_preparation.get_langgraph_service") as mock_service,
        ):
            mock_service.return_value.list_graphs.return_value = ["test-graph"]

            resp = client.post(
                "/threads/test-thread-123/runs/wait",
                json={
                    "assistant_id": "test-assistant-123",
                    "input": {"message": "test"},
                },
            )

            assert resp.status_code == 200
            # StreamingResponse: body is heartbeat newlines + final JSON
            assert resp.json() == {"partial": "data"}

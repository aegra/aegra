"""Integration tests for stateless (thread-free) run endpoints.

Tests hit the FastAPI routes via TestClient with mocked database sessions,
verifying HTTP status codes, request validation, and delegation behaviour.
"""

from unittest.mock import AsyncMock, MagicMock, patch

from tests.fixtures.clients import create_test_app, make_client
from tests.fixtures.database import DummySessionBase
from tests.fixtures.session_fixtures import BasicSession, override_session_dependency
from tests.fixtures.test_helpers import DummyRun, make_assistant

# ---------------------------------------------------------------------------
# ORM mock helpers (matching test_runs_crud.py patterns)
# ---------------------------------------------------------------------------


def _assistant_row(
    assistant_id: str = "test-assistant-123",
    graph_id: str = "test-graph",
    user_id: str = "test-user",
) -> object:
    assistant = make_assistant(assistant_id=assistant_id, graph_id=graph_id, user_id=user_id)
    assistant.graph_id = graph_id
    return assistant


def _run_row(
    run_id: str = "test-run-123",
    thread_id: str = "test-thread-123",
    assistant_id: str = "test-assistant-123",
    status: str = "running",
    user_id: str = "test-user",
    metadata: dict | None = None,
    input_data: dict | None = None,
    output_data: dict | None = None,
) -> DummyRun:
    run = DummyRun(run_id, thread_id, assistant_id, status, user_id, metadata, input_data, output_data)
    run.metadata_json = metadata or {}
    run.error_message = None
    run.config = {}
    run.context = {}

    class _Col:
        def __init__(self, name: str) -> None:
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


# ---------------------------------------------------------------------------
# POST /runs/wait
# ---------------------------------------------------------------------------


class TestStatelessWaitForRun:
    """Test POST /runs/wait (stateless)."""

    def test_validation_error_no_input_or_command(self) -> None:
        """Missing input AND command → 422."""
        app = create_test_app(include_runs=True, include_threads=False)
        override_session_dependency(app, BasicSession)
        client = make_client(app)

        resp = client.post("/runs/wait", json={"assistant_id": "asst-123"})
        assert resp.status_code == 422

    def test_validation_error_both_input_and_command(self) -> None:
        """Providing both input and command → 422."""
        app = create_test_app(include_runs=True, include_threads=False)
        override_session_dependency(app, BasicSession)
        client = make_client(app)

        resp = client.post(
            "/runs/wait",
            json={
                "assistant_id": "asst-123",
                "input": {"msg": "hi"},
                "command": {"resume": "value"},
            },
        )
        assert resp.status_code == 422

    def test_assistant_not_found(self) -> None:
        """Non-existent assistant → 404."""

        app = create_test_app(include_runs=True, include_threads=False)

        class Session(DummySessionBase):
            async def scalar(self, _stmt: object) -> None:
                return None

        session_instance = Session()
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=session_instance)
        ctx.__aexit__ = AsyncMock(return_value=False)
        mock_maker = MagicMock(return_value=ctx)

        override_session_dependency(app, Session)
        client = make_client(app)

        with (
            patch("aegra_api.api.runs._get_session_maker", return_value=mock_maker),
            patch("aegra_api.services.run_preparation.get_langgraph_service") as mock_service,
            patch("aegra_api.api.stateless_runs.delete_thread_by_id", new_callable=AsyncMock),
        ):
            mock_service.return_value.list_graphs.return_value = ["test-graph"]

            resp = client.post(
                "/runs/wait",
                json={"assistant_id": "nonexistent", "input": {"msg": "hi"}},
            )
        assert resp.status_code == 404

    def test_on_completion_keep_accepted(self) -> None:
        """on_completion='keep' is accepted without validation error."""

        app = create_test_app(include_runs=True, include_threads=False)

        session_instance = BasicSession()
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=session_instance)
        ctx.__aexit__ = AsyncMock(return_value=False)
        mock_maker = MagicMock(return_value=ctx)

        override_session_dependency(app, BasicSession)
        client = make_client(app)

        with patch("aegra_api.api.runs._get_session_maker", return_value=mock_maker):
            resp = client.post(
                "/runs/wait",
                json={
                    "assistant_id": "asst-123",
                    "input": {"msg": "hi"},
                    "on_completion": "keep",
                },
            )
        # Should NOT be a validation error; may fail later (404 for assistant, etc.)
        assert resp.status_code != 422

    def test_on_completion_invalid_value_rejected(self) -> None:
        """on_completion='invalid' → 422."""
        app = create_test_app(include_runs=True, include_threads=False)
        override_session_dependency(app, BasicSession)
        client = make_client(app)

        resp = client.post(
            "/runs/wait",
            json={
                "assistant_id": "asst-123",
                "input": {"msg": "hi"},
                "on_completion": "invalid",
            },
        )
        assert resp.status_code == 422

    def test_delegates_to_threaded_wait(self) -> None:
        """Successful call delegates to wait_for_run and returns output."""
        app = create_test_app(include_runs=True, include_threads=False)

        assistant = _assistant_row()
        run = _run_row(status="success")
        run.output = {"result": "done"}

        class Session(DummySessionBase):
            async def scalar(self, stmt: object) -> object:
                stmt_str = str(stmt).lower()
                if "from assistant" in stmt_str:
                    return assistant
                if "from run" in stmt_str:
                    return run
                return None

            async def refresh(self, obj: object) -> None:
                pass

            def add(self, obj: object) -> None:
                pass

            async def commit(self) -> None:
                pass

            async def execute(self, stmt: object) -> object:
                class Result:
                    rowcount = 1

                return Result()

        session_instance = Session()
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=session_instance)
        ctx.__aexit__ = AsyncMock(return_value=False)
        mock_maker = MagicMock(return_value=ctx)

        override_session_dependency(app, Session)
        client = make_client(app)

        # Mock executor so wait_for_completion completes immediately
        mock_executor = MagicMock()
        mock_executor.wait_for_completion = AsyncMock(return_value=None)
        mock_executor.submit = AsyncMock(return_value=None)

        with (
            patch("aegra_api.api.runs._get_session_maker", return_value=mock_maker),
            patch("aegra_api.services.run_waiters._get_session_maker", return_value=mock_maker),
            patch("aegra_api.services.run_waiters.executor", mock_executor),
            patch("aegra_api.services.run_preparation.get_langgraph_service") as mock_service,
            patch("aegra_api.api.stateless_runs.delete_thread_by_id", new_callable=AsyncMock),
        ):
            mock_service.return_value.list_graphs.return_value = ["test-graph"]

            resp = client.post(
                "/runs/wait",
                json={
                    "assistant_id": "test-assistant-123",
                    "input": {"msg": "hi"},
                },
            )

        assert resp.status_code == 200
        assert resp.json() == {"result": "done"}


# ---------------------------------------------------------------------------
# POST /runs/stream
# ---------------------------------------------------------------------------


class TestStatelessStreamRun:
    """Test POST /runs/stream (stateless)."""

    def test_validation_error_no_input_or_command(self) -> None:
        """Missing input AND command → 422."""
        app = create_test_app(include_runs=True, include_threads=False)
        override_session_dependency(app, BasicSession)
        client = make_client(app)

        resp = client.post("/runs/stream", json={"assistant_id": "asst-123"})
        assert resp.status_code == 422

    def test_assistant_not_found(self) -> None:
        """Non-existent assistant → 404."""
        app = create_test_app(include_runs=True, include_threads=False)

        class Session(DummySessionBase):
            async def scalar(self, _stmt: object) -> None:
                return None

        override_session_dependency(app, Session)
        client = make_client(app)

        with (
            patch("aegra_api.services.run_preparation.get_langgraph_service") as mock_service,
            patch("aegra_api.api.stateless_runs.delete_thread_by_id", new_callable=AsyncMock),
        ):
            mock_service.return_value.list_graphs.return_value = ["test-graph"]

            resp = client.post(
                "/runs/stream",
                json={"assistant_id": "nonexistent", "input": {"msg": "hi"}},
            )
        assert resp.status_code == 404

    def test_on_completion_keep_accepted(self) -> None:
        """on_completion='keep' passes validation."""
        app = create_test_app(include_runs=True, include_threads=False)
        override_session_dependency(app, BasicSession)
        client = make_client(app)

        resp = client.post(
            "/runs/stream",
            json={
                "assistant_id": "asst-123",
                "input": {"msg": "hi"},
                "on_completion": "keep",
            },
        )
        assert resp.status_code != 422


# ---------------------------------------------------------------------------
# POST /runs
# ---------------------------------------------------------------------------


class TestStatelessCreateRun:
    """Test POST /runs (stateless)."""

    def test_validation_error_no_input_or_command(self) -> None:
        """Missing input AND command → 422."""
        app = create_test_app(include_runs=True, include_threads=False)
        override_session_dependency(app, BasicSession)
        client = make_client(app)

        resp = client.post("/runs", json={"assistant_id": "asst-123"})
        assert resp.status_code == 422

    def test_assistant_not_found(self) -> None:
        """Non-existent assistant → 404."""
        app = create_test_app(include_runs=True, include_threads=False)

        class Session(DummySessionBase):
            async def scalar(self, _stmt: object) -> None:
                return None

        override_session_dependency(app, Session)
        client = make_client(app)

        with (
            patch("aegra_api.services.run_preparation.get_langgraph_service") as mock_service,
            patch("aegra_api.api.stateless_runs.delete_thread_by_id", new_callable=AsyncMock),
        ):
            mock_service.return_value.list_graphs.return_value = ["test-graph"]

            resp = client.post(
                "/runs",
                json={"assistant_id": "nonexistent", "input": {"msg": "hi"}},
            )
        assert resp.status_code == 404

    def test_on_completion_keep_accepted(self) -> None:
        """on_completion='keep' passes validation."""
        app = create_test_app(include_runs=True, include_threads=False)
        override_session_dependency(app, BasicSession)
        client = make_client(app)

        resp = client.post(
            "/runs",
            json={
                "assistant_id": "asst-123",
                "input": {"msg": "hi"},
                "on_completion": "keep",
            },
        )
        assert resp.status_code != 422

    def test_on_completion_invalid_value_rejected(self) -> None:
        """on_completion='bad' → 422."""
        app = create_test_app(include_runs=True, include_threads=False)
        override_session_dependency(app, BasicSession)
        client = make_client(app)

        resp = client.post(
            "/runs",
            json={
                "assistant_id": "asst-123",
                "input": {"msg": "hi"},
                "on_completion": "bad",
            },
        )
        assert resp.status_code == 422

    def test_config_context_allowed(self) -> None:
        """Both configurable and context are allowed."""
        app = create_test_app(include_runs=True, include_threads=False)

        class Session(DummySessionBase):
            async def scalar(self, _stmt: object) -> None:
                return None

        override_session_dependency(app, Session)
        client = make_client(app)

        # Mock _delete_thread_by_id: assistant lookup raises 404, the cleanup
        # path opens its own DB session via _get_session_maker which is not
        # initialized in this test harness.
        with patch("aegra_api.api.stateless_runs.delete_thread_by_id", new_callable=AsyncMock):
            resp = client.post(
                "/runs",
                json={
                    "assistant_id": "asst-123",
                    "input": {"msg": "hi"},
                    "config": {"configurable": {"key": "val"}},
                    "context": {"key": "val"},
                },
            )
        # Validation conflict is removed; request proceeds to assistant lookup
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Route existence sanity checks
# ---------------------------------------------------------------------------


class TestStatelessRouteRegistration:
    """Verify routes are registered and reachable."""

    def test_runs_wait_route_exists(self) -> None:
        app = create_test_app(include_runs=True)
        override_session_dependency(app, BasicSession)
        client = make_client(app)

        # GET should fail (method not allowed), proving the route exists
        resp = client.get("/runs/wait")
        assert resp.status_code == 405

    def test_runs_stream_route_exists(self) -> None:
        app = create_test_app(include_runs=True)
        override_session_dependency(app, BasicSession)
        client = make_client(app)

        resp = client.get("/runs/stream")
        assert resp.status_code == 405

    def test_runs_route_exists(self) -> None:
        app = create_test_app(include_runs=True)
        override_session_dependency(app, BasicSession)
        client = make_client(app)

        resp = client.get("/runs")
        assert resp.status_code == 405

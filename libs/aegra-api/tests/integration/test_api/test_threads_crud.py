"""Integration tests for threads CRUD operations"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from fastapi.testclient import TestClient
from langgraph.types import StateSnapshot

from aegra_api.core.orm import get_session as core_get_session
from tests.fixtures.clients import create_test_app, make_client
from tests.fixtures.database import DummySessionBase, override_get_session_dep
from tests.fixtures.session_fixtures import BasicSession, override_session_dependency
from tests.fixtures.test_helpers import DummyRun, DummyThread


def create_get_graph_mock(return_value=None, side_effect=None):
    """Create a mock for langgraph_service.get_graph that works as an async context manager."""
    mock = MagicMock()

    @asynccontextmanager
    async def async_cm(*args, **kwargs):
        if side_effect:
            raise side_effect
        yield return_value

    mock.side_effect = lambda *args, **kwargs: async_cm(*args, **kwargs)
    return mock


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


def _run_row(
    run_id="test-run-123",
    thread_id="test-thread-123",
    status="running",
    user_id="test-user",
):
    """Create a mock run ORM object"""
    return DummyRun(run_id, thread_id, status, user_id)


class TestCreateThread:
    """Test POST /threads endpoint"""

    @pytest.fixture
    def client(self) -> TestClient:
        app = create_test_app(include_runs=False, include_threads=True)
        override_session_dependency(app, BasicSession)
        return make_client(app)

    def test_create_thread_basic(self, client):
        """Test creating a thread with basic metadata"""
        resp = client.post(
            "/threads",
            json={"metadata": {"purpose": "testing"}, "initial_state": None},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "thread_id" in data
        assert data["status"] == "idle"
        assert data["metadata"]["purpose"] == "testing"
        assert data["metadata"]["owner"] == "test-user"
        assert data["metadata"]["assistant_id"] is None
        assert data["metadata"]["graph_id"] is None
        assert data["metadata"]["thread_name"] == ""

    def test_create_thread_with_name(self, client):
        """Test creating a thread with thread name in metadata"""
        resp = client.post(
            "/threads",
            json={"metadata": {"thread_name": "Test Thread"}, "initial_state": None},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "thread_id" in data
        assert data["status"] == "idle"
        assert data["metadata"]["thread_name"] == "Test Thread"

    def test_create_thread_preserves_graph_id_from_metadata(self, client):
        """Test that client-provided graph_id in metadata is preserved (fixes #254)."""
        resp = client.post(
            "/threads",
            json={"metadata": {"graph_id": "agent"}},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["metadata"]["graph_id"] == "agent"

    def test_create_thread_does_not_normalize_camel_case_graph_id(self, client):
        """Thread metadata should stay server-canonical; JS SDK already sends snake_case on the wire."""
        resp = client.post(
            "/threads",
            json={"metadata": {"graphId": "agent"}},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["metadata"]["graph_id"] is None
        assert data["metadata"]["graphId"] == "agent"

    def test_create_thread_snake_case_graph_id_takes_precedence(self, client):
        """Canonical graph_id is preserved even if an unrelated camelCase key is also present."""
        resp = client.post(
            "/threads",
            json={"metadata": {"graph_id": "snake", "graphId": "camel"}},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["metadata"]["graph_id"] == "snake"
        assert data["metadata"]["graphId"] == "camel"

    def test_create_thread_empty_request(self, client):
        """Test creating a thread with empty request body"""
        resp = client.post("/threads", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "idle"
        assert "owner" in data["metadata"]

    def test_create_thread_with_complex_metadata(self, client):
        """Test creating a thread with complex nested metadata"""
        resp = client.post(
            "/threads",
            json={
                "metadata": {
                    "tags": ["urgent", "production"],
                    "context": {"user_type": "premium", "tier": 3},
                }
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["metadata"]["tags"] == ["urgent", "production"]
        assert data["metadata"]["context"]["tier"] == 3

    def test_create_thread_with_custom_id(self, client):
        """Test creating a thread with a client-provided threadId"""
        custom_id = "my-custom-thread-id"
        resp = client.post("/threads", json={"threadId": custom_id})
        assert resp.status_code == 200
        data = resp.json()
        assert data["thread_id"] == custom_id

    def test_create_thread_with_custom_id_snake_case(self, client):
        """Test creating a thread with thread_id (snake_case) also works"""
        custom_id = "my-snake-case-thread-id"
        resp = client.post("/threads", json={"thread_id": custom_id})
        assert resp.status_code == 200
        data = resp.json()
        assert data["thread_id"] == custom_id

    def test_create_thread_if_exists_do_nothing(self):
        """Test ifExists='do_nothing' returns existing thread"""
        app = create_test_app(include_runs=False, include_threads=True)

        existing_thread = _thread_row("existing-thread-id", metadata={"original": True})

        class Session(DummySessionBase):
            call_count = 0

            async def scalar(self, _stmt):
                # First call is the existence check
                Session.call_count += 1
                if Session.call_count == 1:
                    return existing_thread
                return None

        app.dependency_overrides[core_get_session] = override_get_session_dep(Session)
        client = make_client(app)

        # Try to create with same ID and ifExists='do_nothing'
        resp = client.post(
            "/threads",
            json={"threadId": "existing-thread-id", "ifExists": "do_nothing"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["thread_id"] == "existing-thread-id"
        # Should return the existing thread's metadata
        assert data["metadata"].get("original") is True

    def test_create_thread_if_exists_raise(self):
        """Test ifExists='raise' (default) returns 409 on duplicate"""
        app = create_test_app(include_runs=False, include_threads=True)

        existing_thread = _thread_row("conflict-thread-id")

        class Session(DummySessionBase):
            async def scalar(self, _stmt):
                return existing_thread

        app.dependency_overrides[core_get_session] = override_get_session_dep(Session)
        client = make_client(app)

        # Try to create with same ID (default ifExists='raise')
        resp = client.post("/threads", json={"threadId": "conflict-thread-id"})
        assert resp.status_code == 409
        assert "already exists" in resp.json()["detail"]

    def test_create_thread_if_exists_raise_explicit(self):
        """Test explicit ifExists='raise' returns 409 on duplicate"""
        app = create_test_app(include_runs=False, include_threads=True)

        existing_thread = _thread_row("conflict-thread-id")

        class Session(DummySessionBase):
            async def scalar(self, _stmt):
                return existing_thread

        app.dependency_overrides[core_get_session] = override_get_session_dep(Session)
        client = make_client(app)

        # Explicitly set ifExists='raise'
        resp = client.post(
            "/threads",
            json={"threadId": "conflict-thread-id", "ifExists": "raise"},
        )
        assert resp.status_code == 409
        assert "already exists" in resp.json()["detail"]


class TestListThreads:
    """Test GET /threads endpoint"""

    def test_list_threads_with_results(self):
        """Test listing threads when user has threads"""
        app = create_test_app(include_runs=False, include_threads=True)

        threads = [
            _thread_row("thread-1", metadata={"name": "First"}),
            _thread_row("thread-2", metadata={"name": "Second"}),
            _thread_row("thread-3", metadata={"name": "Third"}),
        ]

        class Session(DummySessionBase):
            async def scalars(self, _stmt):
                class Result:
                    def all(self):
                        return threads

                return Result()

        app.dependency_overrides[core_get_session] = override_get_session_dep(Session)
        client = make_client(app)

        resp = client.get("/threads")
        assert resp.status_code == 200
        data = resp.json()
        assert "threads" in data
        assert "total" in data
        assert data["total"] == 3
        assert len(data["threads"]) == 3

    def test_list_threads_empty(self):
        """Test listing threads when user has no threads"""
        app = create_test_app(include_runs=False, include_threads=True)

        class Session(DummySessionBase):
            async def scalars(self, _stmt):
                class Result:
                    def all(self):
                        return []

                return Result()

        app.dependency_overrides[core_get_session] = override_get_session_dep(Session)
        client = make_client(app)

        resp = client.get("/threads")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["threads"] == []


class TestGetThread:
    """Test GET /threads/{thread_id} endpoint"""

    def test_get_thread_success(self):
        """Test getting an existing thread"""
        app = create_test_app(include_runs=False, include_threads=True)

        thread = _thread_row("test-123", metadata={"purpose": "testing"})

        class Session(DummySessionBase):
            async def scalar(self, _stmt):
                return thread

        app.dependency_overrides[core_get_session] = override_get_session_dep(Session)
        client = make_client(app)

        resp = client.get("/threads/test-123")
        assert resp.status_code == 200
        data = resp.json()
        assert data["thread_id"] == "test-123"
        assert data["metadata"]["purpose"] == "testing"

    def test_get_thread_not_found(self):
        """Test getting a non-existent thread"""
        app = create_test_app(include_runs=False, include_threads=True)

        class Session(DummySessionBase):
            async def scalar(self, _stmt):
                return None

        app.dependency_overrides[core_get_session] = override_get_session_dep(Session)
        client = make_client(app)

        resp = client.get("/threads/nonexistent")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"]


class TestDeleteThread:
    """Test DELETE /threads/{thread_id} endpoint"""

    def test_delete_thread_not_found(self):
        """Test deleting a non-existent thread"""
        app = create_test_app(include_runs=False, include_threads=True)

        class Session(DummySessionBase):
            async def scalar(self, _stmt):
                return None

        app.dependency_overrides[core_get_session] = override_get_session_dep(Session)
        client = make_client(app)

        resp = client.delete("/threads/nonexistent")
        assert resp.status_code == 404

    def test_delete_thread_no_active_runs(self):
        """Test deleting a thread with no active runs"""
        app = create_test_app(include_runs=False, include_threads=True)

        thread = _thread_row("test-123")

        class Session(DummySessionBase):
            async def scalar(self, _stmt):
                return thread

            async def scalars(self, _stmt):
                class Result:
                    def all(self):
                        return []

                return Result()

            async def delete(self, obj):
                pass

            async def commit(self):
                pass

        app.dependency_overrides[core_get_session] = override_get_session_dep(Session)
        client = make_client(app)

        resp = client.delete("/threads/test-123")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"


class TestSearchThreads:
    """Test POST /threads/search endpoint"""

    @pytest.fixture
    def client(self) -> TestClient:
        app = create_test_app(include_runs=False, include_threads=True)

        threads = [
            _thread_row("thread-1", status="idle", metadata={"env": "prod", "team": "alpha"}),
            _thread_row("thread-2", status="busy", metadata={"env": "dev", "team": "beta"}),
            _thread_row("thread-3", status="idle", metadata={"env": "prod", "team": "beta"}),
        ]

        from tests.fixtures.session_fixtures import ThreadSession

        override_session_dependency(app, ThreadSession, threads=threads)
        return make_client(app)

    def test_search_threads_no_filters(self, client):
        """Test searching without any filters"""
        resp = client.post("/threads/search", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 3

    def test_search_threads_with_status(self, client):
        """Test searching with status filter"""
        resp = client.post("/threads/search", json={"status": "idle"})
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_search_threads_with_metadata(self, client):
        """Test searching with metadata filter"""
        resp = client.post(
            "/threads/search",
            json={"metadata": {"env": "prod"}},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_search_threads_with_pagination(self, client):
        """Test searching with offset and limit"""
        resp = client.post(
            "/threads/search",
            json={"offset": 0, "limit": 2},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_search_threads_combined_filters(self, client):
        """Test searching with multiple filters combined"""
        resp = client.post(
            "/threads/search",
            json={
                "status": "idle",
                "metadata": {"env": "prod"},
                "offset": 0,
                "limit": 10,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_search_accepts_order_by_asc(self, client):
        """order_by='created_at ASC' is accepted without error."""
        resp = client.post("/threads/search", json={"order_by": "created_at ASC"})
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_search_malformed_order_by_does_not_500(self, client):
        """Malformed order_by falls back to default and returns 200."""
        for bad in ["password; DROP TABLE", "nonexistent_col", ""]:
            resp = client.post("/threads/search", json={"order_by": bad})
            assert resp.status_code == 200, f"order_by={bad!r} raised {resp.status_code}"

    def test_search_accepts_sdk_sort_shape(self, client):
        """SDK-style sort_by/sort_order is accepted."""
        resp = client.post(
            "/threads/search",
            json={"sort_by": "updated_at", "sort_order": "asc"},
        )
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_search_sdk_state_updated_at_falls_back(self, client):
        """sort_by='state_updated_at' (valid SDK literal, no column here) falls back without 500."""
        resp = client.post(
            "/threads/search",
            json={"sort_by": "state_updated_at", "sort_order": "desc"},
        )
        assert resp.status_code == 200

    def test_search_rejects_invalid_sort_order(self, client):
        """sort_order is a Literal['asc','desc']; other values are rejected by Pydantic."""
        resp = client.post(
            "/threads/search",
            json={"sort_by": "created_at", "sort_order": "sideways"},
        )
        assert resp.status_code == 422

    def test_search_accepts_bool_metadata_filter(self, client):
        """metadata={'active': True} is accepted end-to-end (real matching verified in E2E)."""
        resp = client.post("/threads/search", json={"metadata": {"active": True}})
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


class TestThreadGetState:
    """Test GET /threads/{thread_id}/state endpoint"""

    def test_get_latest_state_thread_not_found(self):
        """Thread lookup should 404 when record is missing."""
        app = create_test_app(include_runs=False, include_threads=True)

        class Session(DummySessionBase):
            async def scalar(self, _stmt):
                return None

        app.dependency_overrides[core_get_session] = override_get_session_dep(Session)
        client = make_client(app)

        resp = client.get("/threads/missing/state")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_get_latest_state_no_graph_id(self):
        """Threads without graph metadata should return empty state."""
        app = create_test_app(include_runs=False, include_threads=True)

        thread = _thread_row("test-123", metadata={})

        class Session(DummySessionBase):
            async def scalar(self, _stmt):
                return thread

        app.dependency_overrides[core_get_session] = override_get_session_dep(Session)
        client = make_client(app)

        resp = client.get("/threads/test-123/state")
        assert resp.status_code == 200
        state = resp.json()
        assert "values" in state
        assert "checkpoint" in state
        assert state["checkpoint"]["checkpoint_id"] is None

    def test_get_latest_state_success(self):
        """Test getting latest state successfully."""
        app = create_test_app(include_runs=False, include_threads=True)
        thread = _thread_row("test-123", metadata={"graph_id": "test-graph"})

        class Session(DummySessionBase):
            async def scalar(self, _stmt):
                return thread

        app.dependency_overrides[core_get_session] = override_get_session_dep(Session)
        client = make_client(app)

        # Mock langgraph service and agent
        mock_agent = AsyncMock()

        mock_snapshot = Mock(spec=StateSnapshot)
        mock_snapshot.values = {"messages": ["hello"]}
        mock_snapshot.next = []
        mock_snapshot.tasks = []
        mock_snapshot.interrupts = []
        mock_snapshot.metadata = {}
        mock_snapshot.config = {"configurable": {"checkpoint_id": "cp-1"}}
        mock_snapshot.created_at = "2024-01-01T00:00:00Z"
        mock_snapshot.parent_config = None
        mock_agent.aget_state.return_value = mock_snapshot
        mock_agent.with_config = Mock(return_value=mock_agent)

        with patch("aegra_api.services.langgraph_service.get_langgraph_service") as mock_get_service:
            mock_service = mock_get_service.return_value
            mock_service.get_graph = create_get_graph_mock(return_value=mock_agent)

            resp = client.get("/threads/test-123/state")
            assert resp.status_code == 200
            state = resp.json()
            assert state["values"]["messages"] == ["hello"]
            assert state["checkpoint"]["checkpoint_id"] == "cp-1"


class TestThreadUpdateState:
    """Test POST /threads/{thread_id}/state endpoint"""

    def test_update_state_as_get(self):
        """Test POST without values behaves like GET."""
        app = create_test_app(include_runs=False, include_threads=True)
        thread = _thread_row("test-123", metadata={"graph_id": "test-graph"})

        class Session(DummySessionBase):
            async def scalar(self, _stmt):
                return thread

        app.dependency_overrides[core_get_session] = override_get_session_dep(Session)
        client = make_client(app)

        mock_agent = AsyncMock()

        mock_snapshot = Mock(spec=StateSnapshot)
        mock_snapshot.values = {"key": "val"}
        mock_snapshot.next = []
        mock_snapshot.tasks = []
        mock_snapshot.interrupts = []
        mock_snapshot.metadata = {}
        mock_snapshot.config = {"configurable": {"checkpoint_id": "cp-1"}}
        mock_snapshot.created_at = "2024-01-01T00:00:00Z"
        mock_snapshot.parent_config = None
        mock_agent.aget_state.return_value = mock_snapshot
        mock_agent.with_config = Mock(return_value=mock_agent)

        with patch("aegra_api.services.langgraph_service.get_langgraph_service") as mock_get_service:
            mock_service = mock_get_service.return_value
            mock_service.get_graph = create_get_graph_mock(return_value=mock_agent)

            # POST with empty body or no values
            resp = client.post("/threads/test-123/state", json={})
            assert resp.status_code == 200
            state = resp.json()
            assert state["values"]["key"] == "val"

    def test_update_state_success(self):
        """Test updating state successfully."""
        app = create_test_app(include_runs=False, include_threads=True)
        thread = _thread_row("test-123", metadata={"graph_id": "test-graph"})

        class Session(DummySessionBase):
            async def scalar(self, _stmt):
                return thread

        app.dependency_overrides[core_get_session] = override_get_session_dep(Session)
        client = make_client(app)

        mock_agent = AsyncMock()
        # aupdate_state returns the new config
        mock_agent.aupdate_state.return_value = {"configurable": {"checkpoint_id": "new-cp", "checkpoint_ns": ""}}

        mock_agent.with_config = Mock(return_value=mock_agent)
        mock_agent.with_config.return_value = mock_agent

        with patch("aegra_api.services.langgraph_service.get_langgraph_service") as mock_get_service:
            mock_service = mock_get_service.return_value
            mock_service.get_graph = create_get_graph_mock(return_value=mock_agent)

            resp = client.post(
                "/threads/test-123/state",
                json={"values": {"foo": "bar"}, "checkpoint_id": "old-cp"},
            )
            assert resp.status_code == 200
            result = resp.json()
            assert result["checkpoint"]["checkpoint_id"] == "new-cp"

            # Verify aupdate_state called correctly
            mock_agent.aupdate_state.assert_called_once()
            call_args = mock_agent.aupdate_state.call_args
            assert call_args[0][1] == {"foo": "bar"}  # values

    def test_update_state_no_graph(self):
        """Test updating state when thread has no graph."""
        app = create_test_app(include_runs=False, include_threads=True)
        thread = _thread_row("test-123", metadata={})  # No graph_id

        class Session(DummySessionBase):
            async def scalar(self, _stmt):
                return thread

        app.dependency_overrides[core_get_session] = override_get_session_dep(Session)
        client = make_client(app)

        resp = client.post(
            "/threads/test-123/state",
            json={"values": {"foo": "bar"}},
        )
        assert resp.status_code == 400
        assert "no associated graph" in resp.json()["detail"]


class TestThreadStateCheckpoint:
    """Test GET /threads/{thread_id}/state/{checkpoint_id} endpoint"""

    def test_get_state_thread_not_found(self):
        """Test getting state when thread doesn't exist"""
        app = create_test_app(include_runs=False, include_threads=True)

        class Session(DummySessionBase):
            async def scalar(self, _stmt):
                return None

        app.dependency_overrides[core_get_session] = override_get_session_dep(Session)
        client = make_client(app)

        resp = client.get("/threads/nonexistent/state/checkpoint-1")
        assert resp.status_code == 404

    def test_get_state_no_graph_id(self):
        """Test getting state when thread has no associated graph"""
        app = create_test_app(include_runs=False, include_threads=True)

        thread = _thread_row("test-123", metadata={})

        class Session(DummySessionBase):
            async def scalar(self, _stmt):
                return thread

        app.dependency_overrides[core_get_session] = override_get_session_dep(Session)
        client = make_client(app)

        resp = client.get("/threads/test-123/state/checkpoint-1")
        assert resp.status_code == 404
        assert "no associated graph" in resp.json()["detail"]

    def test_get_state_with_subgraphs_param(self):
        """Test getting state with subgraphs query parameter"""
        app = create_test_app(include_runs=False, include_threads=True)

        thread = _thread_row("test-123", metadata={})

        class Session(DummySessionBase):
            async def scalar(self, _stmt):
                return thread

        app.dependency_overrides[core_get_session] = override_get_session_dep(Session)
        client = make_client(app)

        # Should fail because no graph_id, but tests that param is accepted
        resp = client.get("/threads/test-123/state/checkpoint-1?subgraphs=true")
        assert resp.status_code == 404

    def test_get_state_at_checkpoint_success(self):
        """Test getting state at specific checkpoint."""
        app = create_test_app(include_runs=False, include_threads=True)
        thread = _thread_row("test-123", metadata={"graph_id": "test-graph"})

        class Session(DummySessionBase):
            async def scalar(self, _stmt):
                return thread

        app.dependency_overrides[core_get_session] = override_get_session_dep(Session)
        client = make_client(app)

        mock_agent = AsyncMock()

        mock_snapshot = Mock(spec=StateSnapshot)
        mock_snapshot.values = {"foo": "bar"}
        mock_snapshot.next = []
        mock_snapshot.tasks = []
        mock_snapshot.interrupts = []
        mock_snapshot.metadata = {}
        mock_snapshot.config = {"configurable": {"checkpoint_id": "cp-target"}}
        mock_snapshot.created_at = "2024-01-01T00:00:00Z"
        mock_snapshot.parent_config = None
        mock_agent.aget_state.return_value = mock_snapshot
        mock_agent.with_config = Mock(return_value=mock_agent)

        with patch("aegra_api.services.langgraph_service.get_langgraph_service") as mock_get_service:
            mock_service = mock_get_service.return_value
            mock_service.get_graph = create_get_graph_mock(return_value=mock_agent)

            resp = client.get("/threads/test-123/state/cp-target")
            assert resp.status_code == 200
            state = resp.json()
            assert state["checkpoint"]["checkpoint_id"] == "cp-target"
            assert state["values"]["foo"] == "bar"


class TestThreadStateCheckpointPost:
    """Test POST /threads/{thread_id}/state/checkpoint endpoint"""

    def test_post_checkpoint_thread_not_found(self):
        """Test POST checkpoint when thread doesn't exist"""
        app = create_test_app(include_runs=False, include_threads=True)

        class Session(DummySessionBase):
            async def scalar(self, _stmt):
                return None

        app.dependency_overrides[core_get_session] = override_get_session_dep(Session)
        client = make_client(app)

        resp = client.post(
            "/threads/nonexistent/state/checkpoint",
            json={"checkpoint": {"checkpoint_id": "cp-1"}, "subgraphs": False},
        )
        assert resp.status_code == 404

    def test_post_checkpoint_no_graph_id(self):
        """Test POST checkpoint when thread has no graph"""
        app = create_test_app(include_runs=False, include_threads=True)

        thread = _thread_row("test-123", metadata={})

        class Session(DummySessionBase):
            async def scalar(self, _stmt):
                return thread

        app.dependency_overrides[core_get_session] = override_get_session_dep(Session)
        client = make_client(app)

        resp = client.post(
            "/threads/test-123/state/checkpoint",
            json={"checkpoint": {"checkpoint_id": "cp-1"}, "subgraphs": True},
        )
        assert resp.status_code == 404

    def test_post_checkpoint_success(self):
        """Test POST checkpoint success."""
        app = create_test_app(include_runs=False, include_threads=True)
        thread = _thread_row("test-123", metadata={"graph_id": "test-graph"})

        class Session(DummySessionBase):
            async def scalar(self, _stmt):
                return thread

        app.dependency_overrides[core_get_session] = override_get_session_dep(Session)
        client = make_client(app)

        mock_agent = AsyncMock()

        mock_snapshot = Mock(spec=StateSnapshot)
        mock_snapshot.values = {"foo": "bar"}
        mock_snapshot.next = []
        mock_snapshot.tasks = []
        mock_snapshot.interrupts = []
        mock_snapshot.metadata = {}
        mock_snapshot.config = {"configurable": {"checkpoint_id": "cp-post"}}
        mock_snapshot.created_at = "2024-01-01T00:00:00Z"
        mock_snapshot.parent_config = None
        mock_agent.aget_state.return_value = mock_snapshot
        mock_agent.with_config = Mock(return_value=mock_agent)

        with patch("aegra_api.services.langgraph_service.get_langgraph_service") as mock_get_service:
            mock_service = mock_get_service.return_value
            mock_service.get_graph = create_get_graph_mock(return_value=mock_agent)

            resp = client.post(
                "/threads/test-123/state/checkpoint",
                json={"checkpoint": {"checkpoint_id": "cp-post"}},
            )
            assert resp.status_code == 200
            state = resp.json()
            assert state["checkpoint"]["checkpoint_id"] == "cp-post"


class TestUpdateThread:
    """Test PATCH /threads/{thread_id} endpoint"""

    def test_update_thread_metadata_merge(self):
        """Test that new metadata is merged with existing metadata (not replaced)"""
        app = create_test_app(include_runs=False, include_threads=True)

        # 1. Setup: the thread already has some data (e.g., system data)
        initial_metadata = {
            "graph_id": "test-graph-v1",
            "assistant_id": "asst-123",
            "existing_user_field": "do-not-touch",
        }
        thread = _thread_row("thread-update-1", metadata=initial_metadata)

        class Session(DummySessionBase):
            async def scalar(self, _stmt):
                return thread

            async def commit(self):
                pass

            async def refresh(self, obj):
                # In a real DB, refresh updates the object; here we just simulate it
                pass

        app.dependency_overrides[core_get_session] = override_get_session_dep(Session)
        client = make_client(app)

        # 2. Action: update the thread name and add a new field
        patch_payload = {"metadata": {"thread_name": "My New Thread Name", "custom_tag": "important"}}

        resp = client.patch("/threads/thread-update-1", json=patch_payload)

        # 3. Verification
        assert resp.status_code == 200
        data = resp.json()

        # Verify that new fields were added
        assert data["metadata"]["thread_name"] == "My New Thread Name"
        assert data["metadata"]["custom_tag"] == "important"

        # IMPORTANT: Verify that old fields did NOT disappear
        assert data["metadata"]["graph_id"] == "test-graph-v1"
        assert data["metadata"]["existing_user_field"] == "do-not-touch"

    def test_update_thread_overwrite_field(self):
        """Test that existing metadata keys can be updated"""
        app = create_test_app(include_runs=False, include_threads=True)

        thread = _thread_row("thread-update-2", metadata={"status": "draft", "count": 1})

        class Session(DummySessionBase):
            async def scalar(self, _stmt):
                return thread

            async def commit(self):
                pass

            async def refresh(self, obj):
                pass

        app.dependency_overrides[core_get_session] = override_get_session_dep(Session)
        client = make_client(app)

        # Update the existing field 'count'
        resp = client.patch("/threads/thread-update-2", json={"metadata": {"count": 2}})

        assert resp.status_code == 200
        data = resp.json()
        assert data["metadata"]["status"] == "draft"  # did not change
        assert data["metadata"]["count"] == 2  # updated

    def test_update_thread_not_found(self):
        """Test updating a non-existent thread"""
        app = create_test_app(include_runs=False, include_threads=True)

        class Session(DummySessionBase):
            async def scalar(self, _stmt):
                return None

        app.dependency_overrides[core_get_session] = override_get_session_dep(Session)
        client = make_client(app)

        resp = client.patch("/threads/missing-thread", json={"metadata": {"a": 1}})
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"]

    def test_update_thread_empty_body(self):
        """Test patch with empty body (should just update timestamp, not crash)"""
        app = create_test_app(include_runs=False, include_threads=True)
        thread = _thread_row("thread-update-3", metadata={"initial": True})

        class Session(DummySessionBase):
            async def scalar(self, _stmt):
                return thread

            async def commit(self):
                pass

            async def refresh(self, obj):
                pass

        app.dependency_overrides[core_get_session] = override_get_session_dep(Session)
        client = make_client(app)

        # Empty JSON or JSON without metadata
        resp = client.patch("/threads/thread-update-3", json={})

        assert resp.status_code == 200
        data = resp.json()
        # Data should not have changed
        assert data["metadata"]["initial"] is True

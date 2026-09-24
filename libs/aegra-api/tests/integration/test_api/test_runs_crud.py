"""Integration tests for runs CRUD operations"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from tests.fixtures.clients import create_test_app, make_client
from tests.fixtures.database import DummyScalarResult, DummySessionBase
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
    run.claimed_by = None
    run.lease_expires_at = None
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


@pytest.fixture(autouse=True)
def mock_dispatch() -> Any:
    """Cancel/delete paths may promote the thread's queued runs; keep that off the database."""
    with patch("aegra_api.api.runs.dispatch_next_queued_run", new_callable=AsyncMock) as mock:
        yield mock


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
            def expire_all(self) -> None:
                pass

            async def scalar(self, _stmt):
                return run

            async def execute(self, _stmt):
                pass

            async def commit(self):
                pass

        override_session_dependency(app, Session)
        client = make_client(app)

        with (
            patch("aegra_api.api.runs.streaming_service") as mock_streaming,
            patch(
                "aegra_api.api.runs.interrupt_unowned_run",
                new_callable=AsyncMock,
                return_value=True,
            ),
        ):
            mock_streaming.cancel_run = AsyncMock()
            mock_streaming.signal_run_cancelled = AsyncMock()

            resp = client.post("/threads/test-thread-123/runs/test-run-123/cancel")

            assert resp.status_code == 200
            mock_streaming.cancel_run.assert_awaited_once_with("test-run-123", emit_end_event=False)
            mock_streaming.signal_run_cancelled.assert_awaited_once_with("test-run-123")


class TestCancelDispatchDecision:
    """After reconciling an unowned run, the queue moves only when nothing can still execute it."""

    @staticmethod
    def _client_with(run: Any) -> TestClient:
        app = create_test_app(include_runs=True, include_threads=False)

        class Session(DummySessionBase):
            async def scalar(self, _stmt: Any) -> Any:
                return run

            async def commit(self) -> None:
                pass

        override_session_dependency(app, Session)
        return make_client(app)

    def test_unclaimed_run_with_no_local_task_dispatches_immediately(self, mock_dispatch: AsyncMock) -> None:
        # Worker mode, never claimed: no task anywhere can be executing it.
        client = self._client_with(_run_row(status="pending"))
        with (
            patch("aegra_api.api.runs.streaming_service") as mock_streaming,
            patch("aegra_api.api.runs.interrupt_unowned_run", new_callable=AsyncMock, return_value=True),
            patch("aegra_api.api.runs.active_runs", {}),
        ):
            mock_streaming.cancel_run = AsyncMock()
            mock_streaming.signal_run_cancelled = AsyncMock()
            assert client.post("/threads/test-thread-123/runs/test-run-123/cancel").status_code == 200
        mock_dispatch.assert_awaited_once_with("test-thread-123")

    def test_local_task_defers_dispatch_to_its_exit(self, mock_dispatch: AsyncMock) -> None:
        # Dev mode: claimed_by is always NULL, but the task is right here and still running —
        # its finalize (losing the ownership CAS) is what promotes the queue, not this request.
        client = self._client_with(_run_row(status="running"))
        with (
            patch("aegra_api.api.runs.streaming_service") as mock_streaming,
            patch("aegra_api.api.runs.interrupt_unowned_run", new_callable=AsyncMock, return_value=True),
            patch("aegra_api.api.runs.active_runs", {"test-run-123": MagicMock()}),
        ):
            mock_streaming.cancel_run = AsyncMock()
            mock_streaming.signal_run_cancelled = AsyncMock()
            assert client.post("/threads/test-thread-123/runs/test-run-123/cancel").status_code == 200
        mock_streaming.cancel_run.assert_awaited_once_with("test-run-123", emit_end_event=False)
        mock_dispatch.assert_not_awaited()

    def test_claimed_run_defers_dispatch_to_the_worker(self, mock_dispatch: AsyncMock) -> None:
        # Lease lapsed but the row was claimed: the worker may be slow rather than dead. Its
        # finalize (or the stranded-queue sweep) promotes the queue once it has really stopped.
        run = _run_row(status="running")
        run.claimed_by = "worker-0"
        client = self._client_with(run)
        with (
            patch("aegra_api.api.runs.streaming_service") as mock_streaming,
            patch("aegra_api.api.runs.interrupt_unowned_run", new_callable=AsyncMock, return_value=True),
            patch("aegra_api.api.runs.active_runs", {}),
        ):
            mock_streaming.cancel_run = AsyncMock()
            mock_streaming.signal_run_cancelled = AsyncMock()
            assert client.post("/threads/test-thread-123/runs/test-run-123/cancel").status_code == 200
        mock_dispatch.assert_not_awaited()


class TestCancelRuns:
    """Test POST /runs/cancel (bulk cancel used by the SDK's cancel_many)."""

    @staticmethod
    def _app_with_runs(runs: list[Any], thread_exists: bool = True) -> TestClient:
        app = create_test_app(include_runs=True, include_threads=False)

        class Session(DummySessionBase):
            async def scalar(self, _stmt: Any) -> str | None:
                return "test-thread-123" if thread_exists else None

            async def scalars(self, _stmt: Any = None) -> DummyScalarResult:
                return DummyScalarResult(runs)

            async def commit(self) -> None:
                pass

        override_session_dependency(app, Session)
        return make_client(app)

    def test_cancel_runs_by_status_cancels_each_active_run(self) -> None:
        runs = [_run_row(run_id="run-a", status="running"), _run_row(run_id="run-b", status="pending")]
        client = self._app_with_runs(runs)

        with (
            patch("aegra_api.api.runs.streaming_service") as mock_streaming,
            patch("aegra_api.api.runs.interrupt_unowned_run", new_callable=AsyncMock, return_value=True),
        ):
            mock_streaming.interrupt_run = AsyncMock()
            mock_streaming.signal_run_cancelled = AsyncMock()

            resp = client.post("/runs/cancel", json={"status": "all"}, params={"action": "interrupt"})

            assert resp.status_code == 204
            assert mock_streaming.interrupt_run.await_count == 2
            assert mock_streaming.signal_run_cancelled.await_count == 2

    def test_cancel_runs_by_ids_skips_finished_runs(self) -> None:
        runs = [_run_row(run_id="run-a", status="running"), _run_row(run_id="run-b", status="success")]
        client = self._app_with_runs(runs)

        with (
            patch("aegra_api.api.runs.streaming_service") as mock_streaming,
            patch("aegra_api.api.runs.interrupt_unowned_run", new_callable=AsyncMock, return_value=True),
        ):
            mock_streaming.cancel_run = AsyncMock()
            mock_streaming.signal_run_cancelled = AsyncMock()

            resp = client.post(
                "/runs/cancel",
                json={"thread_id": "test-thread-123", "run_ids": ["run-a", "run-b"]},
                params={"action": "cancel"},
            )

            assert resp.status_code == 204
            mock_streaming.cancel_run.assert_awaited_once_with("run-a", emit_end_event=False)

    def test_cancel_runs_by_ids_unknown_thread_is_404(self) -> None:
        client = self._app_with_runs([], thread_exists=False)

        resp = client.post("/runs/cancel", json={"thread_id": "missing", "run_ids": ["run-a"]})

        assert resp.status_code == 404

    def test_cancel_runs_without_selector_is_422(self) -> None:
        client = self._app_with_runs([])

        resp = client.post("/runs/cancel", json={})

        assert resp.status_code == 422

    def test_cancel_runs_unsupported_action_is_422(self) -> None:
        client = self._app_with_runs([])

        resp = client.post("/runs/cancel", json={"status": "all"}, params={"action": "rollback"})

        assert resp.status_code == 422

    def test_cancel_runs_pending_selector_includes_parked_runs(self) -> None:
        """A parked (internal queued) run is reported as pending, so cancelling pending runs
        must include it — the SDK's cancel_many(status="pending") expects it gone."""
        app = create_test_app(include_runs=True, include_threads=False)
        seen: list[str] = []

        class Session(DummySessionBase):
            async def scalars(self, _stmt: Any = None) -> DummyScalarResult:
                seen.append(str(_stmt.compile(compile_kwargs={"literal_binds": True})))
                return DummyScalarResult([])

        override_session_dependency(app, Session)
        client = make_client(app)

        assert client.post("/runs/cancel", json={"status": "pending"}).status_code == 204
        assert client.post("/runs/cancel", json={"status": "all"}).status_code == 204
        assert client.post("/runs/cancel", json={"status": "running"}).status_code == 204

        pending_stmt, all_stmt, running_stmt = seen
        assert "'queued'" in pending_stmt and "'pending'" in pending_stmt
        assert "'queued'" in all_stmt and "'running'" in all_stmt
        assert "'queued'" not in running_stmt

    def test_cancel_runs_no_matches_is_204(self) -> None:
        client = self._app_with_runs([])

        with patch("aegra_api.api.runs.streaming_service") as mock_streaming:
            mock_streaming.interrupt_run = AsyncMock()

            resp = client.post("/runs/cancel", json={"status": "pending"})

            assert resp.status_code == 204
            mock_streaming.interrupt_run.assert_not_awaited()


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

    def test_delete_run_queued_not_allowed(self):
        """A queued run is active; deleting without force returns 409."""
        app = create_test_app(include_runs=True, include_threads=False)

        run = _run_row(status="queued")

        class Session(DummySessionBase):
            async def scalar(self, _stmt):
                return run

        override_session_dependency(app, Session)
        client = make_client(app)

        resp = client.delete("/threads/test-thread-123/runs/test-run-123")

        assert resp.status_code == 409
        assert "active" in resp.json()["detail"].lower()

    def test_delete_run_force_queued_drops_it_without_broker_cancel(self):
        """Force-deleting a queued run drops it in place (no task to cancel) and promotes the queue."""
        app = create_test_app(include_runs=True, include_threads=False)

        run = _run_row(status="queued")
        executed: list[str] = []

        class Session(DummySessionBase):
            async def scalar(self, _stmt):
                return run

            async def execute(self, _stmt):
                executed.append(str(_stmt))

            async def commit(self):
                pass

        with (
            patch("aegra_api.api.runs.streaming_service") as mock_streaming,
            patch("aegra_api.api.runs.cancel_queued_run", new_callable=AsyncMock, return_value=True) as mock_drop,
            patch("aegra_api.api.runs.interrupt_unowned_run", new_callable=AsyncMock) as mock_reconcile,
            patch("aegra_api.api.runs.dispatch_next_queued_run", new_callable=AsyncMock) as mock_dispatch,
        ):
            mock_streaming.cancel_run = AsyncMock()
            mock_streaming.signal_run_cancelled = AsyncMock()
            override_session_dependency(app, Session)
            client = make_client(app)
            resp = client.delete("/threads/test-thread-123/runs/test-run-123?force=1")

        assert resp.status_code == 204
        mock_drop.assert_awaited_once()
        assert mock_drop.await_args.args[1:] == ("test-run-123", "test-thread-123")
        mock_streaming.cancel_run.assert_not_awaited()  # queued run has no task to cancel
        mock_reconcile.assert_not_awaited()
        mock_streaming.signal_run_cancelled.assert_awaited_once_with("test-run-123")  # attached streams close
        assert any("DELETE FROM runs" in stmt for stmt in executed)  # row actually removed
        # Whatever was parked behind the deleted run may start now.
        mock_dispatch.assert_awaited_with("test-thread-123")

    def test_delete_run_force_queued_run_promoted_meanwhile_is_cancelled_not_deleted(self):
        """Regression for the promotion race: the run read as queued was promoted (and has
        started) before the drop — it is cancelled like an active run and, while it is still
        executing, neither deleted underneath the live task nor overtaken by the queue."""
        app = create_test_app(include_runs=True, include_threads=False)

        run = _run_row(status="queued")
        executed: list[str] = []

        class Session(DummySessionBase):
            async def scalar(self, _stmt):
                return run

            def expire_all(self) -> None:
                pass

            async def refresh(self, obj):
                # The concurrent dispatcher promoted it; a live task is now attached.
                obj.status = "running"

            async def execute(self, _stmt):
                executed.append(str(_stmt))

            async def commit(self):
                pass

        with (
            patch("aegra_api.api.runs.streaming_service") as mock_streaming,
            patch("aegra_api.api.runs.cancel_queued_run", new_callable=AsyncMock, return_value=False),
            patch("aegra_api.api.runs.interrupt_unowned_run", new_callable=AsyncMock, return_value=False),
            patch("aegra_api.api.runs.dispatch_next_queued_run", new_callable=AsyncMock) as mock_dispatch,
            # the run never settles in this scenario; do not sit out the 10s wait window
            patch("aegra_api.api.runs._SETTLE_ATTEMPTS", 1),
            patch("aegra_api.api.runs._SETTLE_INTERVAL_SECONDS", 0),
        ):
            mock_streaming.cancel_run = AsyncMock()
            override_session_dependency(app, Session)
            client = make_client(app)
            resp = client.delete("/threads/test-thread-123/runs/test-run-123?force=1")

        # Fell through to the live-run path: the executing task is told to stop ...
        mock_streaming.cancel_run.assert_awaited_once_with("test-run-123")
        # ... but it has not stopped within the window: the row stays, the queue stays parked.
        assert resp.status_code == 409
        assert "still executing" in resp.json()["detail"]
        assert not any("DELETE FROM runs" in stmt for stmt in executed)
        mock_dispatch.assert_not_awaited()

    def test_delete_run_force_waits_for_a_worker_owned_run_to_stop(self):
        """A run owned by a worker elsewhere stops asynchronously: the row is removed (and the
        queue behind it promoted) only after its terminal write lands, not while it may still
        be writing checkpoints."""
        app = create_test_app(include_runs=True, include_threads=False)

        run = _run_row(status="running")
        run.claimed_by = "worker-0"
        executed: list[str] = []
        reads = {"n": 0}

        class Session(DummySessionBase):
            async def scalar(self, _stmt):
                reads["n"] += 1
                if reads["n"] >= 3:
                    run.status = "interrupted"  # the worker's finalize landed on the 2nd poll
                return run

            def expire_all(self) -> None:
                pass

            async def execute(self, _stmt):
                executed.append(str(_stmt))

            async def commit(self):
                pass

        with (
            patch("aegra_api.api.runs.streaming_service") as mock_streaming,
            patch("aegra_api.api.runs.interrupt_unowned_run", new_callable=AsyncMock, return_value=False),
            patch("aegra_api.api.runs.dispatch_next_queued_run", new_callable=AsyncMock) as mock_dispatch,
            patch("aegra_api.api.runs.active_runs", {}),
            patch("aegra_api.api.runs._SETTLE_ATTEMPTS", 5),
            patch("aegra_api.api.runs._SETTLE_INTERVAL_SECONDS", 0),
        ):
            mock_streaming.cancel_run = AsyncMock()
            override_session_dependency(app, Session)
            client = make_client(app)
            resp = client.delete("/threads/test-thread-123/runs/test-run-123?force=1")

        assert resp.status_code == 204
        mock_streaming.cancel_run.assert_awaited_once_with("test-run-123")  # asked the worker to stop
        assert reads["n"] >= 3  # polled until the terminal write showed up
        assert any("DELETE FROM runs" in stmt for stmt in executed)  # only then is the row removed
        mock_dispatch.assert_awaited_once_with("test-thread-123")


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

        client = make_client(app)

        # stream_run manages its session via _get_session_maker (not Depends),
        # so the DI override doesn't apply — patch the maker instead.
        with patch("aegra_api.api.runs._get_session_maker", return_value=_make_session_maker(Session())):
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

        # Resume validation polls fresh sessions via run_preparation._get_session_maker
        # when the first read is not interrupted; keep those returning idle too, and
        # collapse the settle backoff so the reject path does not wait.
        maker = _make_session_maker(Session())
        with (
            patch("aegra_api.api.runs._get_session_maker", return_value=maker),
            patch("aegra_api.services.run_preparation._get_session_maker", return_value=maker),
            patch("aegra_api.services.run_preparation._RESUME_SETTLE_ATTEMPTS", 1),
            patch("aegra_api.services.run_preparation._RESUME_SETTLE_INTERVAL_SECONDS", 0),
        ):
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

    def test_create_run_with_only_checkpoint_id_passes_validation(self) -> None:
        app = create_test_app(include_runs=True, include_threads=False)

        class Session(DummySessionBase):
            async def scalar(self, _stmt: Any) -> None:
                return None

        override_session_dependency(app, Session)
        client = make_client(app)

        resp = client.post(
            "/threads/test-thread-123/runs",
            json={"assistant_id": "nonexistent", "checkpoint_id": "1ef4f797-8335-6428-8001-8a1503f9b875"},
        )
        # Past validation: the 404 comes from the assistant lookup.
        assert resp.status_code == 404

    def test_create_run_rejects_malformed_checkpoint_id(self) -> None:
        app = create_test_app(include_runs=True, include_threads=False)
        override_session_dependency(app, BasicSession)
        client = make_client(app)

        resp = client.post(
            "/threads/test-thread-123/runs",
            json={"assistant_id": "asst-123", "input": {"x": 1}, "checkpoint_id": "not-a-uuid"},
        )
        assert resp.status_code == 422


class TestWaitForRunTimeouts:
    """Test wait_for_run timeout behavior.

    wait_for_run returns a StreamingResponse wrapping heartbeat_wait_body.
    On timeout the run never reached a terminal state, so the generator yields
    an ``__error__`` envelope rather than whatever partial output is on the row.
    """

    def test_wait_for_run_timeout(self):
        """Test that wait_for_run reports the timeout instead of partial state."""
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
            assert resp.json()["__error__"]["error"] == "TimeoutError"

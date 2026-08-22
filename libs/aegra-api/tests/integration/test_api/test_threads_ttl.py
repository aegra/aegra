"""Integration tests for per-thread TTL on POST /threads."""

from datetime import timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import Insert

from aegra_api.api import threads as threads_module
from aegra_api.core.orm import ThreadTTL as ThreadTTLORM
from aegra_api.core.orm import get_session as core_get_session
from aegra_api.services.thread_ttl import ThreadTTLConfig
from tests.fixtures.clients import create_test_app, make_client
from tests.fixtures.database import DummyScalarResult, DummySessionBase, override_get_session_dep
from tests.fixtures.test_helpers import DummyThread


class RecordingSession(DummySessionBase):
    """Session that records add() calls and commits."""

    def __init__(self) -> None:
        self.added: list[object] = []
        self.commits: int = 0

    def add(self, obj: object) -> None:
        self.added.append(obj)

    async def commit(self) -> None:
        self.commits += 1


def _make_client(session: DummySessionBase) -> TestClient:
    app: FastAPI = create_test_app(include_runs=False, include_threads=True)
    app.dependency_overrides[core_get_session] = override_get_session_dep(lambda: session)
    return make_client(app)


@pytest.fixture
def ttl_config_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(threads_module, "get_thread_ttl_config", lambda: None)


class TestCreateThreadTTL:
    """POST /threads with a ttl payload writes a thread_ttl row atomically."""

    def test_create_with_ttl_adds_row_before_single_commit(self, ttl_config_off: None) -> None:
        session = RecordingSession()
        client = _make_client(session)

        resp = client.post("/threads", json={"ttl": {"default_ttl": 10, "strategy": "keep_latest"}})

        assert resp.status_code == 200
        assert len(session.added) == 1
        row = session.added[0]
        assert isinstance(row, ThreadTTLORM)
        assert row.thread_id == resp.json()["thread_id"]
        assert row.strategy == "keep_latest"
        assert row.ttl_minutes == 10
        assert row.expires_at - row.created_at == timedelta(minutes=10)
        assert session.commits == 1

    def test_create_accepts_sdk_ttl_key(self, ttl_config_off: None) -> None:
        """langgraph-sdk sends {'ttl': {'ttl': N}} — minutes under 'ttl'."""
        session = RecordingSession()
        client = _make_client(session)

        resp = client.post("/threads", json={"ttl": {"ttl": 5}})

        assert resp.status_code == 200
        row = session.added[0]
        assert isinstance(row, ThreadTTLORM)
        assert row.ttl_minutes == 5
        assert row.strategy == "delete"

    def test_create_without_ttl_and_no_config_adds_no_row(self, ttl_config_off: None) -> None:
        session = RecordingSession()
        client = _make_client(session)

        resp = client.post("/threads", json={"metadata": {}})

        assert resp.status_code == 200
        assert session.added == []

    def test_create_gets_default_row_when_server_config_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            threads_module,
            "get_thread_ttl_config",
            lambda: ThreadTTLConfig(default_ttl=100, strategy="delete"),
        )
        session = RecordingSession()
        client = _make_client(session)

        resp = client.post("/threads", json={"metadata": {}})

        assert resp.status_code == 200
        row = session.added[0]
        assert isinstance(row, ThreadTTLORM)
        assert row.ttl_minutes == 100
        assert row.strategy == "delete"

    def test_request_ttl_overrides_server_default_per_field(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            threads_module,
            "get_thread_ttl_config",
            lambda: ThreadTTLConfig(default_ttl=100, strategy="delete"),
        )
        session = RecordingSession()
        client = _make_client(session)

        resp = client.post("/threads", json={"ttl": {"strategy": "keep_latest"}})

        assert resp.status_code == 200
        row = session.added[0]
        assert isinstance(row, ThreadTTLORM)
        assert row.ttl_minutes == 100  # falls back to server default
        assert row.strategy == "keep_latest"  # request override

    def test_returns_422_when_ttl_lacks_default_and_no_server_config(self, ttl_config_off: None) -> None:
        session = RecordingSession()
        client = _make_client(session)

        resp = client.post("/threads", json={"ttl": {"strategy": "delete"}})

        assert resp.status_code == 422
        assert session.added == []
        assert session.commits == 0

    def test_conflict_path_does_not_touch_incumbent_ttl(self, ttl_config_off: None) -> None:
        """An idempotent re-create must not add a ttl row for the existing thread."""

        class ConflictSession(RecordingSession):
            async def scalars(self, stmt: object = None) -> DummyScalarResult:
                if isinstance(stmt, Insert):
                    return DummyScalarResult()  # ON CONFLICT DO NOTHING: no row returned
                return DummyScalarResult()

            async def scalar(self, _stmt: object) -> object:
                return DummyThread("taken-id", "idle", {}, "test-user")

        session = ConflictSession()
        client = _make_client(session)

        resp = client.post(
            "/threads",
            json={"thread_id": "taken-id", "if_exists": "do_nothing", "ttl": {"default_ttl": 10}},
        )

        assert resp.status_code == 200
        assert session.added == []


class TestPruneEndpoint:
    """POST /threads/prune maps the service result and scopes to the caller."""

    def test_prune_returns_counts_for_caller(self, monkeypatch: pytest.MonkeyPatch) -> None:
        called: dict[str, object] = {}

        async def fake_prune(session: object, *, user_id: str, auth_filter: object = None) -> tuple[int, int]:
            called["user_id"] = user_id
            called["auth_filter"] = auth_filter
            return 2, 1

        monkeypatch.setattr(threads_module, "prune_expired_threads_for_user", fake_prune)
        session = RecordingSession()
        client = _make_client(session)

        resp = client.post("/threads/prune")

        assert resp.status_code == 200
        assert resp.json() == {"deleted": 2, "pruned": 1}
        assert called["user_id"] == "test-user"

    def test_prune_with_nothing_expired_returns_zeros(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_prune(session: object, *, user_id: str, auth_filter: object = None) -> tuple[int, int]:
            return 0, 0

        monkeypatch.setattr(threads_module, "prune_expired_threads_for_user", fake_prune)
        session = RecordingSession()
        client = _make_client(session)

        resp = client.post("/threads/prune")

        assert resp.status_code == 200
        assert resp.json() == {"deleted": 0, "pruned": 0}

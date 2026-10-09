"""Integration tests for the multitask_strategy field at the HTTP boundary.

Stateful behaviours live in the e2e suite; this asserts the request-validation contract.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from aegra_api.services import run_preparation as run_preparation_mod
from aegra_api.settings import settings
from tests.fixtures.clients import create_test_app, make_client
from tests.fixtures.session_fixtures import BasicSession, override_session_dependency


def _client() -> TestClient:
    app = create_test_app(include_runs=True, include_threads=False)
    override_session_dependency(app, BasicSession)
    return make_client(app)


class TestMultitaskValidation:
    def test_invalid_strategy_returns_422(self) -> None:
        client = _client()

        resp = client.post(
            "/threads/test-thread-123/runs",
            json={"assistant_id": "asst-123", "input": {"messages": []}, "multitask_strategy": "banana"},
        )

        assert resp.status_code == 422

    @pytest.mark.parametrize("strategy", ["reject", "interrupt", "rollback", "enqueue"])
    def test_valid_strategy_passes_validation(self, strategy: str) -> None:
        # A valid enum value must reach the same outcome as omitting the field (never 422);
        # fresh clients keep the two posts independent.
        body = {"assistant_id": "asst-123", "input": {"messages": []}}
        baseline = _client().post("/threads/test-thread-123/runs", json=body)
        resp = _client().post("/threads/test-thread-123/runs", json={**body, "multitask_strategy": strategy})

        assert resp.status_code != 422
        assert resp.status_code == baseline.status_code


class _InterruptedThreadSession(BasicSession):
    """Session whose thread lookup returns a HITL-paused thread owned by the test user."""

    async def scalar(self, _stmt: object) -> object:
        thread = MagicMock()
        thread.status = "interrupted"
        thread.user_id = "test-user"
        return thread


class _IdleThreadSession(BasicSession):
    async def scalar(self, _stmt: object) -> object:
        thread = MagicMock()
        thread.status = "idle"
        thread.user_id = "test-user"
        return thread


class TestResumeValidationAtHttpBoundary:
    """The admission-time input-mode gate, exercised through the full route stack."""

    def test_malformed_command_returns_422(self) -> None:
        # {'goto': [0]} cannot map to a LangGraph Command; it must be rejected at
        # admission instead of crashing mid-run (which would corrupt a HITL pause).
        app = create_test_app(include_runs=True, include_threads=False)
        override_session_dependency(app, _InterruptedThreadSession)
        client = make_client(app)

        resp = client.post(
            "/threads/test-thread-123/runs",
            json={"assistant_id": "asst-123", "command": {"goto": [0]}},
        )

        assert resp.status_code == 422
        assert "Invalid command" in resp.json()["detail"]

    def test_fresh_input_on_paused_thread_returns_409_under_reject(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings.multitask, "MULTITASK_PAUSED_THREAD_POLICY", "reject")
        app = create_test_app(include_runs=True, include_threads=False)
        override_session_dependency(app, _InterruptedThreadSession)
        client = make_client(app)

        resp = client.post(
            "/threads/test-thread-123/runs",
            json={"assistant_id": "asst-123", "input": {"messages": []}},
        )

        assert resp.status_code == 409
        assert "resume" in resp.json()["detail"]

    def test_resume_on_idle_thread_returns_400(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Collapse the settle poll (no DB in integration tests) — idle stays idle.
        monkeypatch.setattr(run_preparation_mod, "_RESUME_SETTLE_INTERVAL_SECONDS", 0)
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=_IdleThreadSession())
        ctx.__aexit__ = AsyncMock(return_value=False)
        monkeypatch.setattr(run_preparation_mod, "_get_session_maker", lambda: MagicMock(return_value=ctx))
        app = create_test_app(include_runs=True, include_threads=False)
        override_session_dependency(app, _IdleThreadSession)
        client = make_client(app)

        resp = client.post(
            "/threads/test-thread-123/runs",
            json={"assistant_id": "asst-123", "command": {"resume": "go"}},
        )

        assert resp.status_code == 400
        assert "not in interrupted state" in resp.json()["detail"]

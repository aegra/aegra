import json

import pytest
from fastapi.testclient import TestClient

from aegra_api.core.orm import (
    get_session as core_get_session,  # for dependency override
)

# Reuse shared test helpers
from tests.fixtures.clients import create_test_app, make_client
from tests.fixtures.database import DummySessionBase, override_get_session_dep
from tests.fixtures.langgraph import FakeAgent, make_snapshot, patch_langgraph_service
from tests.fixtures.test_helpers import DummyThread


def _thread_row():
    thread = DummyThread(
        thread_id="11111111-1111-1111-1111-111111111111",
        status="idle",
        metadata={"graph_id": "dummy_graph"},
        user_id="test-user",
    )

    # Add ORM-specific attributes
    thread.metadata_json = {"graph_id": "dummy_graph"}
    thread.created_at = None
    thread.updated_at = None

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


@pytest.fixture()
def client() -> TestClient:
    # Build app with threads router only
    app = create_test_app(include_runs=False, include_threads=True)

    # Provide a DummySession that returns a thread row for scalar()
    class Session(DummySessionBase):
        async def scalar(self, _stmt):
            return _thread_row()

    # Override the ORM get_session dependency
    app.dependency_overrides[core_get_session] = override_get_session_dep(Session)

    return make_client(app)


# DummyUser and user injection provided by helpers via create_test_app


# Snapshot creation moved to helpers (make_snapshot)


# No longer needed: app and dependencies are provided by helpers in client() fixture


@pytest.fixture()
def mock_langgraph():
    """
    Patch get_langgraph_service().get_graph(...) so that aget_state_history yields
    deterministic snapshots for both GET and POST tests.
    """

    def _agent_for_config(config):
        c1 = {
            "configurable": {
                "thread_id": config.get("configurable", {}).get("thread_id"),
                "checkpoint_id": "cp_1",
                "checkpoint_ns": config.get("configurable", {}).get("checkpoint_ns", ""),
            }
        }
        c2 = {
            "configurable": {
                "thread_id": config.get("configurable", {}).get("thread_id"),
                "checkpoint_id": "cp_2",
                "checkpoint_ns": config.get("configurable", {}).get("checkpoint_ns", ""),
            }
        }
        return FakeAgent(
            [
                make_snapshot({"messages": ["hello"]}, c1, next_nodes=["step_b"]),
                make_snapshot({"messages": ["world"]}, c2, next_nodes=[]),
            ]
        )

    # Use a wrapper agent that builds snapshots using the provided config
    class DynamicFakeAgent(FakeAgent):
        def __init__(self):
            super().__init__(snapshots=[])

        async def aget_state_history(self, config, **kwargs):
            agent = _agent_for_config(config)
            async for s in agent.aget_state_history(config, **kwargs):
                yield s

    with patch_langgraph_service(agent=DynamicFakeAgent()):
        yield


def _ensure_thread(client: TestClient) -> str:
    resp = client.post(
        "/threads",
        json={"metadata": {"purpose": "test-history"}, "initial_state": None},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["thread_id"]


def test_post_history_basic(client: TestClient, mock_langgraph):
    thread_id = _ensure_thread(client)

    # POST history without checkpoint should return two snapshots from fake agent
    payload = {
        "limit": 10,
        "metadata": {"k": "v"},
        "checkpoint": None,
        "subgraphs": False,
        "checkpoint_ns": "",
    }
    resp = client.post(f"/threads/{thread_id}/history", json=payload)
    assert resp.status_code == 200, resp.text
    states = resp.json()
    assert isinstance(states, list)
    assert len(states) == 2

    # Validate shape of first state
    s0 = states[0]
    assert "values" in s0 and s0["values"]["messages"] == ["hello"]
    assert "checkpoint" in s0 and s0["checkpoint"]["checkpoint_id"] == "cp_1"
    assert s0.get("checkpoint_id") == "cp_1"
    assert s0.get("next") == ["step_b"]


def test_post_history_with_checkpoint_ns(client: TestClient, mock_langgraph):
    thread_id = _ensure_thread(client)

    payload = {
        "limit": 10,
        "checkpoint": {"checkpoint_ns": "nsA"},
        "subgraphs": True,
    }
    resp = client.post(f"/threads/{thread_id}/history", json=payload)
    assert resp.status_code == 200, resp.text
    states = resp.json()
    assert len(states) == 2
    for s in states:
        assert s["checkpoint"]["checkpoint_ns"] == "nsA"


def test_get_history_basic(client: TestClient, mock_langgraph):
    thread_id = _ensure_thread(client)

    # GET history with default params
    resp = client.get(f"/threads/{thread_id}/history")
    assert resp.status_code == 200, resp.text
    states = resp.json()
    assert isinstance(states, list)
    assert len(states) == 2

    # GET history with metadata as JSON string and checkpoint_ns
    resp = client.get(
        f"/threads/{thread_id}/history",
        params={
            "limit": 10,
            "metadata": json.dumps({"foo": "bar"}),
            "checkpoint_ns": "nsB",
            "subgraphs": "true",
        },
    )
    assert resp.status_code == 200, resp.text
    states = resp.json()
    assert len(states) == 2
    for s in states:
        assert s["checkpoint"]["checkpoint_ns"] == "nsB"


def test_history_invalid_limit(client: TestClient, mock_langgraph):
    thread_id = _ensure_thread(client)

    # POST invalid limit type
    resp = client.post(f"/threads/{thread_id}/history", json={"limit": "ten"})
    assert resp.status_code == 422

    # GET invalid metadata JSON
    resp = client.get(f"/threads/{thread_id}/history", params={"metadata": "{not-json"})
    assert resp.status_code == 422


# ------------------------------------------------------------------
# Regression: `before` parameter must not double-wrap RunnableConfig
# See: https://github.com/aegra/aegra/issues/274
# ------------------------------------------------------------------


class TestBeforeParameterFormats:
    """Verify that different `before` formats are normalized correctly."""

    @pytest.fixture()
    def capturing_client(self, client: TestClient) -> tuple[TestClient, list[dict]]:
        """Client with a FakeAgent that captures the `before` kwarg."""
        captured: list[dict] = []

        class CapturingAgent(FakeAgent):
            def __init__(self) -> None:
                super().__init__(snapshots=[])

            async def aget_state_history(self, config: dict, **kwargs: dict):  # type: ignore[override]
                captured.append(kwargs.get("before"))
                return
                yield  # intentional — presence of `yield` makes this an async generator

        with patch_langgraph_service(agent=CapturingAgent()):
            yield client, captured

    def test_before_as_string(self, capturing_client: tuple) -> None:
        """String checkpoint ID should be wrapped in configurable."""
        client, captured = capturing_client
        thread_id = _ensure_thread(client)

        resp = client.post(
            f"/threads/{thread_id}/history",
            json={"before": "cp_abc123"},
        )
        assert resp.status_code == 200

        assert len(captured) == 1
        assert captured[0] == {"configurable": {"checkpoint_id": "cp_abc123"}}

    def test_before_as_runnable_config(self, capturing_client: tuple) -> None:
        """RunnableConfig dict (with 'configurable' key) must pass through as-is."""
        client, captured = capturing_client
        thread_id = _ensure_thread(client)

        before = {"configurable": {"checkpoint_id": "cp_xyz789"}}
        resp = client.post(
            f"/threads/{thread_id}/history",
            json={"before": before},
        )
        assert resp.status_code == 200

        assert len(captured) == 1
        # Must NOT be double-wrapped as {"configurable": {"configurable": {...}}}
        assert captured[0] == {"configurable": {"checkpoint_id": "cp_xyz789"}}

    def test_before_as_raw_checkpoint_dict(self, capturing_client: tuple) -> None:
        """Raw checkpoint dict (no 'configurable' key) is wrapped, passed as-is.

        No thread_id scrub: aget_state_history reads only checkpoint_id from
        `before`; the thread comes from the main config.
        """
        client, captured = capturing_client
        thread_id = _ensure_thread(client)

        before = {"checkpoint_id": "cp_raw", "thread_id": "t1"}
        resp = client.post(
            f"/threads/{thread_id}/history",
            json={"before": before},
        )
        assert resp.status_code == 200

        assert len(captured) == 1
        assert captured[0] == {"configurable": {"checkpoint_id": "cp_raw", "thread_id": "t1"}}

    def test_before_none(self, capturing_client: tuple) -> None:
        """None before should pass through as None."""
        client, captured = capturing_client
        thread_id = _ensure_thread(client)

        resp = client.post(
            f"/threads/{thread_id}/history",
            json={"before": None},
        )
        assert resp.status_code == 200

        assert len(captured) == 1
        assert captured[0] is None

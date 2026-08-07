"""A registered ``@auth.on`` handler must fire on every protocol route.

Regression coverage for routes that reached the database without dispatching:
thread state/history, several run routes, cron creation and the v2
event-streaming pair. Verified against a real server — with a handler denying
``threads.read``, ``/threads/{id}/state`` and ``/threads/{id}/history`` returned
200 and served the conversation before this change, while ``/threads/{id}``
correctly returned 403.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langgraph_sdk import Auth

from aegra_api.core import auth_handlers
from aegra_api.core.auth_deps import get_current_user, require_auth
from aegra_api.core.auth_enforcement import (
    apply_auth_enforcement,
    build_auth_enforcer,
    reset_dispatch_state,
)
from aegra_api.models.auth import User
from tests.fixtures.session_fixtures import ThreadSession, override_session_dependency


@pytest.fixture(autouse=True)
def _clear_dispatch_state() -> Iterator[None]:
    """Each test starts with a clean per-request dispatch flag."""
    reset_dispatch_state()
    yield
    reset_dispatch_state()


def _build_app(auth: Auth, monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    """Mount every protocol router with enforcement and a stubbed auth instance."""
    # Patch the imported module object directly: a sibling suite monkeypatches
    # `aegra_api.core`, which breaks string-target resolution in a full run.
    monkeypatch.setattr(auth_handlers, "get_auth_instance", lambda: auth)

    from aegra_api.api import assistants, event_streaming, runs, stateless_runs, threads

    app = FastAPI()
    mock_user = User(identity="test-user", display_name="Test User")
    app.dependency_overrides[require_auth] = lambda: mock_user
    app.dependency_overrides[get_current_user] = lambda: mock_user

    for module in (assistants, threads, runs, stateless_runs, event_streaming):
        app.include_router(module.router)

    # Allowed requests fall through to the DB; stub it so these tests stay
    # about authorization dispatch rather than persistence.
    override_session_dependency(app, ThreadSession, threads=[])
    apply_auth_enforcement(app)
    return app


def _deny_all_auth() -> Auth:
    """An Auth instance whose global handler denies everything."""
    auth = Auth()

    @auth.on
    async def deny(ctx: Any, value: Any) -> bool:
        return False

    return auth


# (method, path, json body) for routes that reached the database without ever
# calling a handler. Assistants are excluded: they dispatch via the Authenticated
# service base, and stateless runs delegate to the threaded run functions.
PREVIOUSLY_UNPROTECTED = [
    ("GET", "/threads/t-1/state", None),
    ("POST", "/threads/t-1/state", {"values": {}}),
    ("GET", "/threads/t-1/state/ckpt-1", None),
    ("POST", "/threads/t-1/state/checkpoint", {}),
    ("GET", "/threads/t-1/history", None),
    ("POST", "/threads/t-1/history", {}),
    ("GET", "/threads/t-1/runs", None),
    ("PATCH", "/threads/t-1/runs/r-1", {}),
    ("POST", "/threads/t-1/runs/r-1/cancel", None),
    ("POST", "/threads/t-1/stream/events", {}),
    ("POST", "/threads/t-1/commands", {}),
]


@pytest.mark.parametrize(("method", "path", "body"), PREVIOUSLY_UNPROTECTED)
def test_deny_handler_blocks_previously_unprotected_route(
    method: str, path: str, body: dict[str, Any] | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A global deny handler must reach every route, not just the wired ones."""
    app = _build_app(_deny_all_auth(), monkeypatch)

    with TestClient(app) as client:
        response = client.request(method, path, json=body)

    assert response.status_code == 403, (
        f"{method} {path} returned {response.status_code}; a global @auth.on deny handler was not applied to this route"
    )


def test_thread_read_handler_reaches_state_and_history(monkeypatch: pytest.MonkeyPatch) -> None:
    """State and history authorize as a thread read, so one rule covers all three."""
    auth = Auth()
    seen: list[tuple[str, str]] = []

    @auth.on.threads
    async def record(ctx: Any, value: Any) -> bool:
        seen.append((ctx.resource, ctx.action))
        return False

    app = _build_app(auth, monkeypatch)

    with TestClient(app) as client:
        for path in ("/threads/t-1", "/threads/t-1/state", "/threads/t-1/history"):
            assert client.get(path).status_code == 403, f"{path} skipped the handler"

    assert seen == [("threads", "read")] * 3


def test_handler_runs_exactly_once_per_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """Routes that already call handle_event must not dispatch a second time."""
    auth = Auth()
    calls: list[str] = []

    @auth.on
    async def count_calls(ctx: Any, value: Any) -> bool:
        calls.append(ctx.action)
        return True

    app = _build_app(auth, monkeypatch)

    # /threads/search already calls handle_event in its body, so it is the case
    # where double dispatch would show up.
    with TestClient(app) as client:
        client.post("/threads/search", json={})

    assert len(calls) == 1, f"Expected one handler invocation, got {len(calls)}: {calls}"


def test_stateless_run_cannot_bypass_thread_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dropping the thread_id must not dodge an @auth.on.threads rule."""
    auth = Auth()

    @auth.on.threads
    async def deny_threads(ctx: Any, value: Any) -> bool:
        return False

    app = _build_app(auth, monkeypatch)

    with TestClient(app) as client:
        response = client.post("/runs", json={"assistant_id": "agent"})

    assert response.status_code == 403


def test_enforcer_authenticates_rather_than_reading_an_unset_scope() -> None:
    """The enforcer must not depend on a user that a later dependency installs.

    It is prepended, so it runs before the route's own `get_current_user`. If it
    depended on `get_current_user` it would read an empty request scope and every
    request would 500 — caught only against a real server, not with overrides.
    """
    from aegra_api.core.auth_deps import require_auth

    enforce = build_auth_enforcer("threads", "read")
    signature = inspect.signature(enforce)
    user_default = signature.parameters["user"].default

    assert user_default.dependency is require_auth, (
        "build_auth_enforcer must depend on require_auth; get_current_user only "
        "reads the scope that require_auth populates"
    )


def test_no_auth_configured_still_allows_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Aegra without an auth.py must keep working — dispatch is a no-op there."""
    monkeypatch.setattr(auth_handlers, "get_auth_instance", lambda: None)

    from aegra_api.api import assistants

    app = FastAPI()
    mock_user = User(identity="test-user", display_name="Test User")
    app.dependency_overrides[require_auth] = lambda: mock_user
    app.dependency_overrides[get_current_user] = lambda: mock_user
    app.include_router(assistants.router)
    override_session_dependency(app, ThreadSession, threads=[])
    apply_auth_enforcement(app)

    with TestClient(app) as client:
        response = client.get("/assistants")

    assert response.status_code != 403

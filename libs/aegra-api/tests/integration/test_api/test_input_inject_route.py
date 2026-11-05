"""``input.inject`` over ``POST /threads/{id}/commands``: the route's ``@auth.on`` result is honored."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langgraph_sdk import Auth

from aegra_api.api import event_streaming as es_module
from aegra_api.core import auth_handlers
from aegra_api.core.auth_deps import get_current_user, require_auth
from aegra_api.core.auth_enforcement import apply_auth_enforcement
from aegra_api.models.auth import User
from aegra_api.services.event_streaming import capabilities as caps
from aegra_api.services.event_streaming import inject

_USER = "inject-user"
_BODY: dict[str, Any] = {
    "id": 1,
    "method": "input.inject",
    "params": {"namespace": [], "message": {"role": "user", "content": "also check the tests", "id": "m1"}},
}


class _Session:
    """The route only checks the thread owner before dispatching; storage is stubbed."""

    async def __aenter__(self) -> _Session:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    async def scalar(self, _stmt: object) -> str:
        return _USER


@pytest.fixture(autouse=True)
def _v2_enabled(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(caps.settings.event_streaming, "FF_V2_EVENT_STREAMING", True)
    caps._probe_runtime_symbols.cache_clear()
    yield
    caps._probe_runtime_symbols.cache_clear()


@pytest.fixture
def stored(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    monkeypatch.setattr(inject, "thread_admission", AsyncMock(return_value="accepting"))
    mock = AsyncMock(return_value=True)
    monkeypatch.setattr(inject, "store_message", mock)
    return mock


def _client(monkeypatch: pytest.MonkeyPatch, auth: Auth | None) -> TestClient:
    monkeypatch.setattr(auth_handlers, "get_auth_instance", lambda: auth)
    monkeypatch.setattr(es_module, "_get_session_maker", lambda: _Session)
    app = FastAPI()
    user = User(identity=_USER)
    app.dependency_overrides[require_auth] = lambda: user
    app.dependency_overrides[get_current_user] = lambda: user
    app.include_router(es_module.router)
    apply_auth_enforcement(app)
    return TestClient(app)


def test_inject_without_handlers_stores_the_client_message(monkeypatch: pytest.MonkeyPatch, stored: AsyncMock) -> None:
    resp = _client(monkeypatch, None).post("/threads/t1/commands", json=_BODY)

    assert resp.status_code == 200
    assert resp.json() == {"type": "success", "id": 1, "result": {}, "meta": {"applied_through_seq": 0}}
    assert stored.call_args.kwargs["message"]["content"] == "also check the tests"
    assert stored.call_args.kwargs["thread_id"] == "t1"


def test_create_run_handler_mutation_reaches_the_stored_message(
    monkeypatch: pytest.MonkeyPatch, stored: AsyncMock
) -> None:
    auth = Auth()

    @auth.on.threads.create_run
    async def redact(ctx: Any, value: Any) -> None:
        value["params"]["message"]["content"] = "[redacted]"
        value["params"]["message"]["team"] = "support"

    resp = _client(monkeypatch, auth).post("/threads/t1/commands", json=_BODY)

    assert resp.json()["type"] == "success"
    message = stored.call_args.kwargs["message"]
    assert message["content"] == "[redacted]"
    assert message["additional_kwargs"] == {"team": "support", "injected_by": _USER}


def test_create_run_handler_returned_dict_merges_into_the_message(
    monkeypatch: pytest.MonkeyPatch, stored: AsyncMock
) -> None:
    auth = Auth()

    @auth.on.threads.create_run
    async def tag(ctx: Any, value: Any) -> dict[str, Any]:
        return {"params": {"message": {"tenant": "acme"}}}

    resp = _client(monkeypatch, auth).post("/threads/t1/commands", json=_BODY)

    assert resp.json()["type"] == "success"
    assert stored.call_args.kwargs["message"]["additional_kwargs"] == {"tenant": "acme", "injected_by": _USER}


def test_handler_sees_the_inject_with_the_url_thread(monkeypatch: pytest.MonkeyPatch, stored: AsyncMock) -> None:
    auth = Auth()
    seen: list[tuple[str, str, Any, Any]] = []

    @auth.on
    async def record(ctx: Any, value: Any) -> None:
        seen.append((ctx.resource, ctx.action, value.get("method"), value.get("thread_id")))

    _client(monkeypatch, auth).post("/threads/t1/commands", json={**_BODY, "thread_id": "elsewhere"})

    assert seen == [("threads", "create_run", "input.inject", "t1")]


def test_denying_handler_refuses_the_inject(monkeypatch: pytest.MonkeyPatch, stored: AsyncMock) -> None:
    auth = Auth()

    @auth.on.threads.create_run
    async def deny(ctx: Any, value: Any) -> bool:
        return value.get("method") != "input.inject"

    resp = _client(monkeypatch, auth).post("/threads/t1/commands", json=_BODY)

    assert resp.status_code == 403
    stored.assert_not_awaited()


def test_inject_errors_ride_http_200(monkeypatch: pytest.MonkeyPatch, stored: AsyncMock) -> None:
    monkeypatch.setattr(inject, "thread_admission", AsyncMock(return_value="no_active_run"))

    resp = _client(monkeypatch, None).post("/threads/t1/commands", json=_BODY)

    assert resp.status_code == 200
    assert resp.json()["error"] == "no_such_run"
    stored.assert_not_awaited()

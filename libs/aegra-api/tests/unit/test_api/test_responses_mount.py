"""The Open Responses router is mounted only when http.responses is true."""

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute

from aegra_api.config import responses_enabled
from aegra_api.main import _include_core_routers, create_app


def _responses_flag(name: str) -> bool:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / name
        if not candidate.is_file():
            continue
        loaded: Any = json.loads(candidate.read_text())
        http = loaded.get("http") if isinstance(loaded, dict) else None
        flag = http.get("responses") if isinstance(http, dict) else None
        if isinstance(flag, bool):
            return flag
        raise AssertionError(f"{name} http.responses must be a boolean")
    raise AssertionError(f"{name} not found")


def _paths(app: FastAPI) -> set[str]:
    found: set[str] = set()

    def walk(routes: list[object]) -> None:
        for route in routes:
            path = getattr(route, "path", None)
            if isinstance(route, APIRoute) and isinstance(path, str):
                found.add(path)
            nested = getattr(route, "original_router", None)
            if nested is not None:
                walk(list(nested.routes))
            elif hasattr(route, "routes"):
                walk(list(route.routes))

    walk(list(app.routes))
    return found


def test_repository_default_leaves_responses_opt_in() -> None:
    """The README server uses aegra.json. E2E opts in through a separate file."""
    assert responses_enabled({"responses": _responses_flag("aegra.json")}) is False
    assert responses_enabled({"responses": _responses_flag("aegra.responses.json")}) is True


def test_responses_enabled_requires_boolean_true() -> None:
    """Absent, false, and non-boolean values leave the surface off."""
    assert responses_enabled(None) is False
    assert responses_enabled({}) is False
    assert responses_enabled({"responses": False}) is False
    assert responses_enabled({"responses": "true"}) is False
    assert responses_enabled({"responses": True}) is True


def test_core_routers_omit_responses_unless_asked() -> None:
    """The flag is what mounts /v1, not merely importing the router."""
    disabled = FastAPI()
    _include_core_routers(disabled)
    enabled = FastAPI()
    _include_core_routers(enabled, include_responses=True)

    assert "/v1/responses" not in _paths(disabled)
    assert "/v1/responses" in _paths(enabled)
    assert "/v1/conversations" in _paths(enabled)


def test_create_app_honors_the_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """create_app reads http.responses and does not mount the routes when it is off."""
    monkeypatch.setattr("aegra_api.main.load_http_config", lambda: {"responses": False})
    disabled = create_app()
    monkeypatch.setattr("aegra_api.main.load_http_config", lambda: {"responses": True})
    enabled = create_app()

    assert "/v1/responses" not in _paths(disabled)
    assert "/v1/responses" in _paths(enabled)

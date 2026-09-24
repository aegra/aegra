"""create_app with enable_custom_route_auth: custom routes need auth, Aegra's public routes do not."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi import APIRouter, FastAPI, WebSocket
from fastapi.testclient import TestClient
from starlette.authentication import AuthCredentials
from starlette.requests import Request

from aegra_api import main
from aegra_api.core import auth_deps

AUTH_HEADER = {"Authorization": "Bearer alice"}
CUSTOM_PATHS = ["/custom/direct", "/custom/included"]


class _HeaderBackend:
    """Authenticates only requests that carry an Authorization header."""

    async def authenticate(self, request: Request) -> tuple[AuthCredentials, dict[str, Any]] | None:
        if "authorization" not in request.headers:
            return None
        return AuthCredentials(["authenticated"]), {"identity": "alice"}


def _custom_app() -> FastAPI:
    app = FastAPI()

    @app.get("/custom/direct")
    def direct() -> dict[str, str]:
        return {"route": "direct"}

    router = APIRouter()

    @router.get("/custom/included")
    def included() -> dict[str, str]:
        return {"route": "included"}

    @router.websocket("/custom/ws")
    async def socket(websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.send_text("connected")
        await websocket.close()

    app.include_router(router)
    return app


@pytest.fixture
def build_client(monkeypatch: pytest.MonkeyPatch) -> Callable[[bool], TestClient]:
    def build(enabled: bool) -> TestClient:
        http_config = {"app": "./custom.py:app", "enable_custom_route_auth": enabled}
        monkeypatch.setattr(main, "load_http_config", lambda: http_config)
        monkeypatch.setattr(main, "get_config_dir", lambda: Path("."))
        monkeypatch.setattr(main, "load_custom_app", lambda *_args, **_kwargs: _custom_app())
        monkeypatch.setattr(auth_deps, "get_auth_backend", _HeaderBackend)
        return TestClient(main.create_app())

    return build


@pytest.mark.parametrize("path", CUSTOM_PATHS)
def test_custom_route_returns_401_without_credentials_when_enabled(
    build_client: Callable[[bool], TestClient], path: str
) -> None:
    client = build_client(True)

    response = client.get(path)

    assert response.status_code == 401


@pytest.mark.parametrize("path", CUSTOM_PATHS)
def test_custom_route_serves_authenticated_request_when_enabled(
    build_client: Callable[[bool], TestClient], path: str
) -> None:
    client = build_client(True)

    response = client.get(path, headers=AUTH_HEADER)

    assert response.status_code == 200


@pytest.mark.parametrize("path", ["/live", "/info", "/"])
def test_aegra_public_routes_stay_public_when_enabled(build_client: Callable[[bool], TestClient], path: str) -> None:
    client = build_client(True)

    response = client.get(path)

    assert response.status_code == 200


def test_websocket_under_an_included_router_still_connects_when_enabled(
    build_client: Callable[[bool], TestClient],
) -> None:
    client = build_client(True)

    with client.websocket_connect("/custom/ws") as socket:
        message = socket.receive_text()

    assert message == "connected"


@pytest.mark.parametrize("path", CUSTOM_PATHS)
def test_custom_route_stays_public_when_disabled(build_client: Callable[[bool], TestClient], path: str) -> None:
    client = build_client(False)

    response = client.get(path)

    assert response.status_code == 200

"""enable_custom_route_auth must reach every custom FastAPI route, by real request.

Asserting on ``route.dependencies`` is not enough: FastAPI builds each route's
dependency graph at construction, and included routers cache their effective
routes, so a dependency can be listed and still never run.
"""

from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
from fastapi import APIRouter, Depends, FastAPI, HTTPException, WebSocket
from fastapi.testclient import TestClient
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.types import Receive, Scope, Send

from aegra_api import main
from aegra_api.main import _apply_auth_to_custom_routes


def _deny() -> None:
    raise HTTPException(status_code=401, detail="denied")


DENY = [Depends(_deny)]


def _app_with_every_route_shape() -> FastAPI:
    app = FastAPI()

    @app.get("/direct")
    def direct() -> dict[str, str]:
        return {"route": "direct"}

    nested = APIRouter()

    @nested.get("/nested")
    def nested_route() -> dict[str, str]:
        return {"route": "nested"}

    included = APIRouter()

    @included.get("/included")
    def included_route() -> dict[str, str]:
        return {"route": "included"}

    included.include_router(nested)
    app.include_router(included, prefix="/pre")

    sub = FastAPI()

    @sub.get("/x")
    def sub_route() -> dict[str, str]:
        return {"route": "sub"}

    app.mount("/sub", sub)
    return app


PROTECTED_PATHS = ["/direct", "/pre/included", "/pre/nested", "/sub/x"]


@pytest.mark.parametrize("path", PROTECTED_PATHS)
def test_rejects_unauthenticated_request_on_every_fastapi_route_shape(path: str) -> None:
    app = _app_with_every_route_shape()

    _apply_auth_to_custom_routes(app, DENY)

    assert TestClient(app).get(path).status_code == 401


@pytest.mark.parametrize("path", PROTECTED_PATHS)
def test_rejects_when_routes_were_resolved_before_auth_was_applied(path: str) -> None:
    app = _app_with_every_route_shape()
    app.openapi()
    TestClient(app).get("/pre/included")

    _apply_auth_to_custom_routes(app, DENY)

    assert TestClient(app).get(path).status_code == 401


def test_allows_request_that_passes_the_dependency() -> None:
    app = _app_with_every_route_shape()

    _apply_auth_to_custom_routes(app, [Depends(lambda: None)])

    response = TestClient(app).get("/pre/included")
    assert response.status_code == 200
    assert response.json() == {"route": "included"}


@pytest.mark.parametrize("path", ["/docs", "/openapi.json", "/redoc"])
def test_leaves_fastapi_docs_pages_public(path: str) -> None:
    app = _app_with_every_route_shape()

    _apply_auth_to_custom_routes(app, DENY)

    assert TestClient(app).get(path).status_code == 200


def test_returns_the_number_of_routes_protected() -> None:
    app = _app_with_every_route_shape()

    assert _apply_auth_to_custom_routes(app, DENY) == len(PROTECTED_PATHS)


def test_applying_twice_runs_the_dependency_once() -> None:
    app = FastAPI()
    calls: list[str] = []

    @app.get("/direct")
    def direct() -> dict[str, str]:
        return {}

    deps = [Depends(lambda: calls.append("auth"))]

    assert _apply_auth_to_custom_routes(app, deps) == 1
    assert _apply_auth_to_custom_routes(app, deps) == 0
    TestClient(app).get("/direct")

    assert calls == ["auth"]


def test_auth_runs_before_the_route_own_dependencies() -> None:
    app = FastAPI()
    order: list[str] = []

    def route_dependency() -> None:
        order.append("route")

    @app.get("/direct", dependencies=[Depends(route_dependency)])
    def direct() -> dict[str, str]:
        return {}

    _apply_auth_to_custom_routes(app, [Depends(lambda: order.append("auth"))])
    TestClient(app).get("/direct")

    assert order == ["auth", "route"]


@pytest.mark.parametrize("path", ["/direct", "/pre/included", "/pre/nested"])
def test_no_app_router_or_route_dependency_runs_before_auth(path: str) -> None:
    order: list[str] = []

    def record(name: str) -> Callable[[], None]:
        return lambda: order.append(name)

    def deny() -> None:
        order.append("auth")
        raise HTTPException(status_code=401)

    app = FastAPI(dependencies=[Depends(record("app"))])

    @app.get("/direct", dependencies=[Depends(record("route"))])
    def direct() -> dict[str, str]:
        return {}

    nested = APIRouter(dependencies=[Depends(record("nested-router"))])

    @nested.get("/nested", dependencies=[Depends(record("route"))])
    def nested_route() -> dict[str, str]:
        return {}

    router = APIRouter(dependencies=[Depends(record("router"))])

    @router.get("/included", dependencies=[Depends(record("route"))])
    def included() -> dict[str, str]:
        return {}

    router.include_router(nested, dependencies=[Depends(record("nested-include"))])
    app.include_router(router, prefix="/pre", dependencies=[Depends(record("include"))])

    _apply_auth_to_custom_routes(app, [Depends(deny)])
    status = TestClient(app).get(path).status_code

    assert status == 401
    assert order == ["auth"]


def test_warns_about_custom_routes_it_cannot_protect(monkeypatch: pytest.MonkeyPatch) -> None:
    app = FastAPI()

    def plain(_request: Request) -> PlainTextResponse:
        return PlainTextResponse("ok")

    async def raw_asgi(scope: Scope, receive: Receive, send: Send) -> None:
        await PlainTextResponse("ok")(scope, receive, send)

    @app.websocket("/ws")
    async def socket(websocket: WebSocket) -> None:
        await websocket.close()

    router = APIRouter()

    @router.websocket("/router-ws")
    async def router_socket(websocket: WebSocket) -> None:
        await websocket.close()

    app.include_router(router)
    app.add_route("/plain", plain)
    app.mount("/raw", raw_asgi)
    logger = MagicMock()
    monkeypatch.setattr(main, "logger", logger)

    _apply_auth_to_custom_routes(app, DENY)

    logger.warning.assert_called_once()
    assert sorted(logger.warning.call_args.kwargs["paths"]) == ["/plain", "/raw", "/router-ws", "/ws"]


def test_does_not_warn_when_every_custom_route_is_covered(monkeypatch: pytest.MonkeyPatch) -> None:
    logger = MagicMock()
    monkeypatch.setattr(main, "logger", logger)

    _apply_auth_to_custom_routes(_app_with_every_route_shape(), DENY)

    logger.warning.assert_not_called()

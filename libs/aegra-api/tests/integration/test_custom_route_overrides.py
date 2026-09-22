"""A custom app's route must win the published schema as well as the request.

Core routers are appended to the custom app, so Starlette already dispatches the
custom route for a shared path — first match wins. OpenAPI generation folds
duplicate paths by assignment, so before this the *core* operation was published
for a path the custom handler served.
"""

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from pydantic import BaseModel

from aegra_api.main import _include_core_routers

# Authorization for this one lives in the core handler's body, which is the
# common case: 38 of the 52 mapped operations are self-dispatching.
_SELF_DISPATCHING = "/assistants"
# Authorization for this one is attached as a dependency by path and method.
_ENFORCED = "/threads/{thread_id}/state"


class _CustomCreate(BaseModel):
    """Distinguishable from AssistantCreate in the generated schema."""

    shibboleth: str


def _api_routes(app: FastAPI) -> list[APIRoute]:
    """Every APIRoute, including those inside included routers.

    FastAPI wraps an included router rather than flattening it, so reading
    ``app.routes`` alone sees the custom app's own routes and none of the core ones.
    """

    def walk(routes: list[Any]) -> list[APIRoute]:
        found: list[APIRoute] = []
        for route in routes:
            if isinstance(route, APIRoute):
                found.append(route)
                continue
            nested = getattr(route, "original_router", None)
            if nested is not None:
                found.extend(walk(list(nested.routes)))
            elif hasattr(route, "routes"):
                found.extend(walk(list(route.routes)))
        return found

    return walk(list(app.routes))


def _serving(app: FastAPI, path: str, method: str) -> list[APIRoute]:
    return [route for route in _api_routes(app) if route.path == path and method in route.methods]


@pytest.fixture
def shadowing_app() -> FastAPI:
    app = FastAPI()

    @app.post(_SELF_DISPATCHING, tags=["Assistants"])
    async def create(request: _CustomCreate) -> dict[str, Any]:
        return {"shibboleth": request.shibboleth}

    _include_core_routers(app)
    return app


@pytest.fixture
def non_shadowing_app() -> FastAPI:
    app = FastAPI()

    @app.get("/custom/hello")
    async def hello() -> dict[str, str]:
        return {"message": "hello"}

    _include_core_routers(app)
    return app


def test_the_custom_operation_is_the_one_published(shadowing_app: FastAPI) -> None:
    body = shadowing_app.openapi()["paths"][_SELF_DISPATCHING]["post"]["requestBody"]
    ref = body["content"]["application/json"]["schema"]["$ref"]

    assert ref.endswith("/_CustomCreate")


def test_the_core_duplicate_is_not_registered(shadowing_app: FastAPI) -> None:
    serving = _serving(shadowing_app, _SELF_DISPATCHING, "POST")

    assert [route.endpoint.__name__ for route in serving] == ["create"]


def test_another_method_on_the_same_path_is_untouched(shadowing_app: FastAPI) -> None:
    """Only POST is claimed, so GET must still come from the core router."""
    assert "get" in shadowing_app.openapi()["paths"][_SELF_DISPATCHING]
    assert _serving(shadowing_app, _SELF_DISPATCHING, "GET")


def test_an_unrelated_custom_route_leaves_every_core_operation(non_shadowing_app: FastAPI) -> None:
    paths = non_shadowing_app.openapi()["paths"]

    assert "/custom/hello" in paths
    for expected in ("/assistants", "/assistants/search", "/threads", _ENFORCED):
        assert expected in paths, expected


def test_the_override_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    """An accidental collision has to be visible at boot, not a silent takeover."""
    app = FastAPI()

    @app.post(_SELF_DISPATCHING)
    async def create() -> dict[str, str]:
        return {}

    with caplog.at_level("INFO"):
        _include_core_routers(app)

    assert any("overrides core route" in record.getMessage() for record in caplog.records)


def test_a_shadow_of_an_enforced_route_keeps_its_authorization() -> None:
    """The enforcer is attached by path and method, so it finds the replacement."""
    app = FastAPI()

    @app.get(_ENFORCED)
    async def read_state(thread_id: str) -> dict[str, Any]:
        return {}

    _include_core_routers(app)

    shadow = _serving(app, _ENFORCED, "GET")
    assert [route.endpoint.__name__ for route in shadow] == ["read_state"]
    assert shadow[0].dependencies, "authorization dependency was not attached to the overriding route"


def test_a_shadow_of_a_self_dispatching_route_authorizes_nothing_by_itself() -> None:
    """Documents the sharp edge rather than asserting safety that is not there.

    Most operations authorize inside the core handler, so a shadow that does not
    delegate to it dispatches no ``@auth.on`` event. That is true whether or not
    the core duplicate is registered — the custom route wins dispatch either way —
    but it is the reason an override wants to be a deliberate act.
    """
    app = FastAPI()

    @app.post(_SELF_DISPATCHING)
    async def create() -> dict[str, str]:
        return {}

    _include_core_routers(app)

    shadow = _serving(app, _SELF_DISPATCHING, "POST")[0]
    assert not shadow.dependencies


def test_filtering_a_router_preserves_every_other_operation(shadowing_app: FastAPI) -> None:
    """Leaving one operation out must not disturb the rest of its router.

    The core routers carry ``tags`` and ``dependencies`` at router level, so a
    rebuilt router that dropped them would silently unauthenticate the lot.
    """
    plain = FastAPI()
    _include_core_routers(plain)

    def fingerprint(app: FastAPI) -> dict[tuple[str, str], tuple[tuple[str, ...], int]]:
        return {
            (route.path, method): (tuple(sorted(route.tags)), len(route.dependencies))
            for route in _api_routes(app)
            for method in route.methods
        }

    before, after = fingerprint(plain), fingerprint(shadowing_app)
    shadowed = (_SELF_DISPATCHING, "POST")

    assert set(before) == set(after), "an operation went missing"
    assert {k: v for k, v in before.items() if k != shadowed and after[k] != v} == {}

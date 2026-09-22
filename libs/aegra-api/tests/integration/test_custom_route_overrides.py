"""A custom app's route must win the published schema as well as the request.

The custom route already wins dispatch; before this the core operation won the
schema, so the published contract was not the one served.
"""

from typing import Any

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.routing import APIRoute
from pydantic import BaseModel

from aegra_api.core.auth_deps import get_current_user, require_auth
from aegra_api.main import _api_routes, _include_core_routers
from aegra_api.models.auth import User
from tests.fixtures.clients import make_client

# Authorization for this one lives in the core handler's body, which is the
# common case: 38 of the 52 mapped operations are self-dispatching.
_SELF_DISPATCHING = "/assistants"
# Authorization for this one is attached as a dependency by path and method.
_ENFORCED = "/threads/{thread_id}/state"


class _CustomCreate(BaseModel):
    """Distinguishable from AssistantCreate in the generated schema."""

    shibboleth: str


def _serving(app: FastAPI, path: str, method: str) -> list[APIRoute]:
    return [route for served, route in _api_routes(list(app.routes)) if served == path and method in route.methods]


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

    Most operations authorize in the core handler, so a shadow that does not
    delegate to it dispatches nothing — true before this change as well.
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
            (served, method): (tuple(sorted(route.tags)), len(route.dependencies))
            for served, route in _api_routes(list(app.routes))
            for method in route.methods
        }

    before, after = fingerprint(plain), fingerprint(shadowing_app)
    shadowed = (_SELF_DISPATCHING, "POST")

    assert set(before) == set(after), "an operation went missing"
    assert {k: v for k, v in before.items() if k != shadowed and after[k] != v} == {}


def test_a_router_mounted_with_include_router_also_claims_its_paths() -> None:
    """A custom app that composes with ``include_router`` has no top-level routes.

    FastAPI wraps the router instead of flattening it, so claim detection that
    reads ``app.routes`` directly sees nothing and the core duplicate survives.
    """
    router = APIRouter(tags=["Assistants"])

    @router.post(_SELF_DISPATCHING)
    async def create(request: _CustomCreate) -> dict[str, Any]:
        return {"shibboleth": request.shibboleth}

    app = FastAPI()
    app.include_router(router)
    _include_core_routers(app)

    body = app.openapi()["paths"][_SELF_DISPATCHING]["post"]["requestBody"]
    assert body["content"]["application/json"]["schema"]["$ref"].endswith("/_CustomCreate")
    assert [route.endpoint.__name__ for route in _serving(app, _SELF_DISPATCHING, "POST")] == ["create"]


def test_a_prefixed_router_claims_the_path_it_actually_serves() -> None:
    """A router included under a prefix declares one path and serves another.

    Reading the declared path would have a router mounted at ``/api`` claim the
    core ``/assistants``, deleting an Agent Protocol operation nothing replaced.
    """
    router = APIRouter()

    @router.post(_SELF_DISPATCHING)
    async def create() -> dict[str, Any]:
        return {}

    app = FastAPI()
    app.include_router(router, prefix="/api")
    _include_core_routers(app)

    published = app.openapi()["paths"]
    assert "post" in published[_SELF_DISPATCHING], "the core operation was removed by a prefixed route"
    assert "post" in published["/api" + _SELF_DISPATCHING]
    assert [route.endpoint.__name__ for route in _serving(app, _SELF_DISPATCHING, "POST")] == ["create_assistant"]


def test_a_nested_prefix_is_accumulated() -> None:
    """Prefixes combine down the tree, so only the innermost path is served."""
    inner = APIRouter()

    @inner.post(_SELF_DISPATCHING)
    async def create() -> dict[str, Any]:
        return {}

    outer = APIRouter()
    outer.include_router(inner, prefix="/v1")

    app = FastAPI()
    app.include_router(outer, prefix="/api")
    _include_core_routers(app)

    published = app.openapi()["paths"]
    assert "/api/v1" + _SELF_DISPATCHING in published
    assert "post" in published[_SELF_DISPATCHING], "the core operation was removed by a nested prefixed route"


def test_the_overriding_handler_serves_the_request() -> None:
    """The other half of the contract: the published operation is the one that runs.

    ``create_test_app`` mounts the core routers itself, so it cannot express a
    custom app that declares a route *before* they are appended, which is the
    whole scenario here. The app is built the way the server builds one and
    driven through ``make_client``.
    """
    user = User(identity="test-user", display_name="Test User", org_id="org-1")

    app = FastAPI()

    @app.post(_SELF_DISPATCHING, tags=["Assistants"])
    async def create(request: _CustomCreate) -> dict[str, Any]:
        return {"served_by": "custom", "shibboleth": request.shibboleth}

    _include_core_routers(app)
    app.dependency_overrides[require_auth] = lambda: user
    app.dependency_overrides[get_current_user] = lambda: user

    response = make_client(app).post(_SELF_DISPATCHING, json={"shibboleth": "xibalba"})

    assert response.status_code == 200
    assert response.json() == {"served_by": "custom", "shibboleth": "xibalba"}

    published = app.openapi()["paths"][_SELF_DISPATCHING]["post"]["requestBody"]
    assert published["content"]["application/json"]["schema"]["$ref"].endswith("/_CustomCreate")


def test_an_unclaimed_core_operation_still_answers_over_http() -> None:
    """Skipping one operation must not take its neighbours off the wire."""
    user = User(identity="test-user", display_name="Test User", org_id="org-1")

    app = FastAPI()

    @app.post(_SELF_DISPATCHING, tags=["Assistants"])
    async def create() -> dict[str, Any]:
        return {}

    _include_core_routers(app)
    app.dependency_overrides[require_auth] = lambda: user
    app.dependency_overrides[get_current_user] = lambda: user

    # Reaching the core handler is the assertion: it gets as far as wanting a
    # database, where a route that had been skipped would answer 405.
    with pytest.raises(RuntimeError, match="Database not initialized"):
        make_client(app).get("/assistants/does-not-exist")

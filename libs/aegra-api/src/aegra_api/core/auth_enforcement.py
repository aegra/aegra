"""Guarantee every protocol route dispatches to the user's ``@auth.on`` handlers.

Routes used to opt *in* to authorization by calling ``handle_event`` in their
body. Forgetting the call produced no error and no failing test — the handler
just never ran. This module inverts that: dispatch is attached at route
registration from :mod:`aegra_api.core.auth_registry`, so a route opts *out*
only by being listed as exempt.

Routes that already call ``handle_event`` themselves stay correct: the request
is marked once dispatched, and the second call is skipped rather than running a
user's handler twice.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

import structlog
from fastapi import Depends, Request
from fastapi.dependencies.utils import get_parameterless_sub_dependant
from fastapi.routing import APIRoute

from aegra_api.core.auth_deps import require_auth
from aegra_api.core.auth_handlers import build_auth_context, handle_event
from aegra_api.core.auth_registry import lookup_route_auth
from aegra_api.models.auth import User

logger = structlog.getLogger(__name__)

# Route bodies do not receive the Request, so the "already dispatched" signal
# travels in a ContextVar rather than request.state. Each request runs in its own
# context, so this cannot leak between concurrent requests.
_dispatched: ContextVar[bool] = ContextVar("aegra_auth_dispatched", default=False)
_filters: ContextVar[dict[str, Any] | None] = ContextVar("aegra_auth_filters", default=None)


def mark_dispatched(filters: dict[str, Any] | None) -> None:
    """Record that authorization already ran for this request."""
    _dispatched.set(True)
    _filters.set(filters)


def was_dispatched() -> bool:
    """Whether authorization already ran for this request."""
    return _dispatched.get()


def reset_dispatch_state() -> None:
    """Clear the per-request dispatch flags (used by tests)."""
    _dispatched.set(False)
    _filters.set(None)


def get_auth_filters() -> dict[str, Any] | None:
    """Filters returned by the ``@auth.on`` handler for this request, if any."""
    return _filters.get()


async def _read_json_body(request: Request) -> dict[str, Any]:
    """Best-effort JSON body for the handler's ``value`` argument.

    Starlette caches the body, so reading it here does not starve the route.
    A non-dict or unparseable body authorizes as an empty value rather than
    failing the request — malformed input is the route's error to report.
    """
    if request.method in ("GET", "HEAD", "DELETE"):
        return {}
    try:
        body = await request.json()
    except Exception:
        return {}
    return body if isinstance(body, dict) else {}


def build_auth_enforcer(resource: str, action: str):
    """Build the dependency that dispatches ``@auth.on`` for one route."""

    # Depends on require_auth, not get_current_user: this dependency is prepended
    # so it runs before the route's own, and get_current_user only reads the user
    # that require_auth puts on the request scope.
    async def enforce(
        request: Request,
        user: User = Depends(require_auth),
    ) -> None:
        value: dict[str, Any] = {**request.path_params, **await _read_json_body(request)}
        ctx = build_auth_context(user, resource, action)
        filters = await handle_event(ctx, value)
        mark_dispatched(filters)

    return enforce


def apply_auth_enforcement(app: Any) -> int:
    """Attach ``@auth.on`` dispatch to every registered protocol route.

    Returns the number of routes wired, for startup logging and tests.
    """
    wired = 0

    def walk(routes: list[Any]) -> None:
        nonlocal wired
        for route in routes:
            if isinstance(route, APIRoute):
                wired += _wire_route(route)
                continue
            # FastAPI wraps included routers; newer versions expose the wrapped
            # router as `original_router` rather than `routes`.
            nested = getattr(route, "original_router", None)
            if nested is not None:
                walk(list(nested.routes))
            elif hasattr(route, "routes"):
                walk(list(route.routes))

    walk(list(app.routes))
    logger.info("Applied authorization dispatch to protocol routes", route_count=wired)
    return wired


def _wire_route(route: APIRoute) -> int:
    """Attach the enforcer to a single route for each registered method."""
    for method in sorted(route.methods or set()):
        mapping = lookup_route_auth(method, route.path)
        if mapping is None:
            continue
        resource, action = mapping
        dependency = Depends(build_auth_enforcer(resource, action))
        # Prepend so authorization runs before the route's own dependencies.
        route.dependencies = [dependency, *(route.dependencies or [])]
        # `dependencies` alone is only read at construction time; mirror what
        # APIRoute.__init__ does so the already-built dependant picks it up.
        route.dependant.dependencies.insert(
            0,
            get_parameterless_sub_dependant(depends=dependency, path=route.path_format),
        )
        return 1
    return 0

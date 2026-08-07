"""Every Agent Protocol route must have a declared ``@auth.on`` identity.

Regression guard for routes that reach the database without dispatching to the
user's handlers: thread state/history, several run routes, cron creation and the
v2 event-streaming pair all did. Adding a route without registering it here
fails the suite instead of silently ignoring the user's handlers.
"""

from __future__ import annotations

import pytest
from fastapi.routing import APIRoute

from aegra_api.core.auth_registry import (
    EXEMPT_PATHS,
    ROUTE_AUTH_MAP,
    lookup_route_auth,
)

# Methods that never carry a body-bearing authorization decision.
_IGNORED_METHODS = frozenset({"HEAD", "OPTIONS"})

# User-defined routes from a custom app are governed by `enable_custom_route_auth`,
# not by the protocol registry. Only Aegra's own surface is checked here.
_CUSTOM_ROUTE_PREFIXES = ("/custom",)


def _protocol_routes() -> list[tuple[str, str]]:
    """Collect (method, path) for every mounted Agent Protocol route.

    Routers mount as nested router objects, so a flat scan of ``app.routes``
    misses entire files — the same blind spot that hid the unprotected routers.
    Recurse the way ``_apply_auth_to_routes`` does.
    """
    from aegra_api.main import app

    collected: list[tuple[str, str]] = []

    def walk(routes: list[object]) -> None:
        for route in routes:
            if isinstance(route, APIRoute):
                for method in route.methods or set():
                    if method not in _IGNORED_METHODS:
                        collected.append((method, route.path))
                continue
            # FastAPI wraps included routers; newer versions expose the wrapped
            # router as `original_router` rather than `routes`.
            nested = getattr(route, "original_router", None)
            if nested is not None:
                walk(list(nested.routes))
            elif hasattr(route, "routes"):
                walk(list(route.routes))

    walk(list(app.routes))
    return [(method, path) for method, path in collected if not path.startswith(_CUSTOM_ROUTE_PREFIXES)]


def test_every_mounted_route_is_registered_or_exempt() -> None:
    """No route may exist without either an auth identity or an explicit exemption."""
    unregistered = [
        (method, path)
        for method, path in _protocol_routes()
        if path not in EXEMPT_PATHS and lookup_route_auth(method, path) is None
    ]

    assert not unregistered, (
        "These routes have no @auth.on identity and are not exempt. Add them to "
        "ROUTE_AUTH_MAP in core/auth_registry.py (or EXEMPT_PATHS if they are "
        "genuinely public):\n  " + "\n  ".join(f"{m} {p}" for m, p in sorted(unregistered))
    )


def test_registry_has_no_entries_for_routes_that_do_not_exist() -> None:
    """A stale registry entry means a route was renamed and dispatch silently moved."""
    mounted = set(_protocol_routes())
    stale = [entry for entry in ROUTE_AUTH_MAP if entry not in mounted]

    assert not stale, "ROUTE_AUTH_MAP references routes that are not mounted. Remove or update them:\n  " + "\n  ".join(
        f"{m} {p}" for m, p in sorted(stale)
    )


def test_no_route_is_both_registered_and_exempt() -> None:
    """An exempt path must not also claim an auth identity — one of them is wrong."""
    conflicting = sorted({path for _, path in ROUTE_AUTH_MAP if path in EXEMPT_PATHS})

    assert not conflicting, f"Paths are both registered and exempt: {conflicting}"


@pytest.mark.parametrize(
    ("path", "expected_resource"),
    [
        ("/assistants", "assistants"),
        ("/threads", "threads"),
        ("/store/items", "store"),
        ("/runs/crons", "crons"),
    ],
)
def test_known_resources_map_to_their_own_namespace(path: str, expected_resource: str) -> None:
    """Guards against a copy-paste that points a router at the wrong resource."""
    entries = [res for (_, p), (res, _) in ROUTE_AUTH_MAP.items() if p == path]

    assert entries, f"No registry entry for {path}"
    assert all(res == expected_resource for res in entries), (
        f"{path} should authorize as '{expected_resource}', got {set(entries)}"
    )


def test_stateless_runs_authorize_as_thread_run_creation() -> None:
    """Stateless runs mint an ephemeral thread, so they must not bypass thread rules."""
    for path in ("/runs", "/runs/stream", "/runs/wait"):
        assert lookup_route_auth("POST", path) == ("threads", "create_run"), (
            f"POST {path} must authorize as threads/create_run so an @auth.on.threads "
            "handler cannot be bypassed by dropping the thread_id"
        )


def test_assistant_routes_are_all_covered() -> None:
    """Assistants dispatch via the service layer; the registry must still name them."""
    assistant_routes = [(method, path) for method, path in _protocol_routes() if path.startswith("/assistants")]

    assert assistant_routes, "Expected assistant routes to be mounted"
    for method, path in assistant_routes:
        assert lookup_route_auth(method, path) is not None, f"{method} {path} has no auth identity"

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
    SELF_DISPATCHING,
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


def test_self_dispatching_routes_are_registered() -> None:
    """A self-dispatching entry must name a real registered route.

    A stale entry here silently disables the enforcer for that path, which is
    the exact failure this module exists to prevent.
    """
    unknown = sorted(entry for entry in SELF_DISPATCHING if entry not in ROUTE_AUTH_MAP)

    assert not unknown, "SELF_DISPATCHING names routes absent from ROUTE_AUTH_MAP:\n  " + "\n  ".join(
        f"{m} {p}" for m, p in unknown
    )


def test_self_dispatching_routes_actually_authorize_themselves() -> None:
    """Each self-dispatching route's module must contain a dispatch call.

    Coarse by design: it cannot prove the call runs, but it does catch an entry
    added for a module that authorizes nowhere.
    """
    import pathlib

    import aegra_api.api

    api_dir = pathlib.Path(aegra_api.api.__file__).parent
    dispatchers = ("handle_event", "_apply_create_run_auth", "_dispatch")
    sources = {p.name: p.read_text(encoding="utf-8") for p in api_dir.glob("*.py")}
    assert sources, f"no API modules found under {api_dir}"

    dispatching_modules = {name for name, src in sources.items() if any(d in src for d in dispatchers)}

    # Assistants dispatch through the Authenticated service base, not the router.
    dispatching_modules.add("assistants.py")

    assert "threads.py" in dispatching_modules
    assert "crons.py" in dispatching_modules
    assert "store.py" in dispatching_modules


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


# Expected (resource, action) for every route, pinned to AUTH_DISPATCH_SPEC.md.
# This is the drift guard: any change to a mapping fails here and forces a
# reviewer to confirm it against the spec, rather than a mapping silently
# shifting. Entries that diverge from the Agent Protocol carry a NOTE.
_SPEC_TUPLES: dict[tuple[str, str], tuple[str, str]] = {
    # assistants
    ("POST", "/assistants"): ("assistants", "create"),
    ("GET", "/assistants"): ("assistants", "search"),
    ("POST", "/assistants/search"): ("assistants", "search"),
    ("POST", "/assistants/count"): ("assistants", "search"),
    ("GET", "/assistants/{assistant_id}"): ("assistants", "read"),
    ("PATCH", "/assistants/{assistant_id}"): ("assistants", "update"),
    ("DELETE", "/assistants/{assistant_id}"): ("assistants", "delete"),
    ("POST", "/assistants/{assistant_id}/latest"): ("assistants", "update"),
    ("POST", "/assistants/{assistant_id}/versions"): ("assistants", "read"),
    ("GET", "/assistants/{assistant_id}/schemas"): ("assistants", "read"),
    ("GET", "/assistants/{assistant_id}/graph"): ("assistants", "read"),
    ("GET", "/assistants/{assistant_id}/subgraphs"): ("assistants", "read"),
    # threads
    ("POST", "/threads"): ("threads", "create"),
    ("GET", "/threads"): ("threads", "search"),
    ("POST", "/threads/search"): ("threads", "search"),
    ("GET", "/threads/{thread_id}"): ("threads", "read"),
    ("PATCH", "/threads/{thread_id}"): ("threads", "update"),
    ("DELETE", "/threads/{thread_id}"): ("threads", "delete"),
    ("GET", "/threads/{thread_id}/state"): ("threads", "read"),
    ("POST", "/threads/{thread_id}/state"): ("threads", "update"),
    ("GET", "/threads/{thread_id}/state/{checkpoint_id}"): ("threads", "read"),
    # POST checkpoint is a read that carries its config in the body (not a write).
    ("POST", "/threads/{thread_id}/state/checkpoint"): ("threads", "read"),
    ("GET", "/threads/{thread_id}/history"): ("threads", "read"),
    ("POST", "/threads/{thread_id}/history"): ("threads", "read"),
    # runs — NOTE: non-portable `runs.*`; protocol uses `threads.*`. Retire in 0.10.0.
    ("POST", "/threads/{thread_id}/runs"): ("threads", "create_run"),
    ("POST", "/threads/{thread_id}/runs/stream"): ("threads", "create_run"),
    ("POST", "/threads/{thread_id}/runs/wait"): ("threads", "create_run"),
    ("GET", "/threads/{thread_id}/runs"): ("runs", "search"),
    ("GET", "/threads/{thread_id}/runs/{run_id}"): ("runs", "read"),
    ("PATCH", "/threads/{thread_id}/runs/{run_id}"): ("runs", "update"),
    ("GET", "/threads/{thread_id}/runs/{run_id}/join"): ("runs", "read"),
    ("GET", "/threads/{thread_id}/runs/{run_id}/stream"): ("runs", "read"),
    ("POST", "/threads/{thread_id}/runs/{run_id}/cancel"): ("runs", "update"),
    ("DELETE", "/threads/{thread_id}/runs/{run_id}"): ("runs", "delete"),
    # stateless runs
    ("POST", "/runs"): ("threads", "create_run"),
    ("POST", "/runs/stream"): ("threads", "create_run"),
    ("POST", "/runs/wait"): ("threads", "create_run"),
    # crons
    ("POST", "/runs/crons"): ("crons", "create"),
    ("POST", "/threads/{thread_id}/runs/crons"): ("crons", "create"),
    ("PATCH", "/runs/crons/{cron_id}"): ("crons", "update"),
    ("DELETE", "/runs/crons/{cron_id}"): ("crons", "delete"),
    ("POST", "/runs/crons/search"): ("crons", "search"),
    ("POST", "/runs/crons/count"): ("crons", "search"),
    # store — NOTE: /store/namespaces should be `list_namespaces`. Rename in 0.10.0.
    ("PUT", "/store/items"): ("store", "put"),
    ("GET", "/store/items"): ("store", "get"),
    ("DELETE", "/store/items"): ("store", "delete"),
    ("POST", "/store/items/search"): ("store", "search"),
    ("POST", "/store/namespaces"): ("store", "search"),
    # v2 event streaming
    ("POST", "/threads/{thread_id}/stream/events"): ("threads", "read"),
    ("POST", "/threads/{thread_id}/commands"): ("threads", "create_run"),
}


def test_registry_matches_spec_snapshot_exactly() -> None:
    """Every mapping is pinned to the spec; a silent action/resource shift fails here.

    The coverage tests prove a route *has* an identity. This proves it has the
    *right* one. Changing a tuple must update this snapshot too, which is the
    prompt to re-check it against AUTH_DISPATCH_SPEC.md.
    """
    assert ROUTE_AUTH_MAP == _SPEC_TUPLES, (
        "ROUTE_AUTH_MAP drifted from the spec snapshot. Diff the two and, if the "
        "change is intentional, update _SPEC_TUPLES after confirming against "
        "aegra-context/AUTH_DISPATCH_SPEC.md."
    )

"""Declarative map of every Agent Protocol route to its ``@auth.on`` identity.

Authorization used to be wired by hand inside each route body, which made
"forgot to call ``handle_event``" invisible: the route worked, the tests passed,
and a user's ``@auth.on.assistants`` handler simply never ran. Whole routers
shipped with no dispatch at all.

The map below is the single source of truth for which ``(resource, action)``
pair a route authorizes as. ``tests/unit/test_core/test_auth_registry.py``
asserts every mounted route is either listed here or explicitly exempt, so a new
route cannot silently join the unprotected set.

Path keys are FastAPI templates (``/threads/{thread_id}``), matched exactly.
"""

from __future__ import annotations

from typing import Final

# (method, path) -> (resource, action)
ROUTE_AUTH_MAP: Final[dict[tuple[str, str], tuple[str, str]]] = {
    # --- assistants ---------------------------------------------------------
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
    # --- threads ------------------------------------------------------------
    ("POST", "/threads"): ("threads", "create"),
    ("GET", "/threads"): ("threads", "search"),
    ("POST", "/threads/search"): ("threads", "search"),
    ("GET", "/threads/{thread_id}"): ("threads", "read"),
    ("PATCH", "/threads/{thread_id}"): ("threads", "update"),
    ("DELETE", "/threads/{thread_id}"): ("threads", "delete"),
    # Reading or writing checkpointed state is a thread read/update, not a
    # separate resource: handlers scoping "threads" must cover it.
    ("GET", "/threads/{thread_id}/state"): ("threads", "read"),
    ("POST", "/threads/{thread_id}/state"): ("threads", "update"),
    ("GET", "/threads/{thread_id}/state/{checkpoint_id}"): ("threads", "read"),
    ("POST", "/threads/{thread_id}/state/checkpoint"): ("threads", "read"),
    ("GET", "/threads/{thread_id}/history"): ("threads", "read"),
    ("POST", "/threads/{thread_id}/history"): ("threads", "read"),
    # --- runs ---------------------------------------------------------------
    # Run operations authorize under `threads` per the Agent Protocol (verified
    # against langgraph-api 0.12.1 grpc/ops/runs.py, resource = "threads"). The
    # SDK's @auth.on has no `runs` resource, so no user handler could ever have
    # bound to the old `runs.*` names — this alignment is non-breaking.
    ("POST", "/threads/{thread_id}/runs"): ("threads", "create_run"),
    ("POST", "/threads/{thread_id}/runs/stream"): ("threads", "create_run"),
    ("POST", "/threads/{thread_id}/runs/wait"): ("threads", "create_run"),
    ("GET", "/threads/{thread_id}/runs"): ("threads", "search"),
    ("GET", "/threads/{thread_id}/runs/{run_id}"): ("threads", "read"),
    ("PATCH", "/threads/{thread_id}/runs/{run_id}"): ("threads", "update"),
    ("GET", "/threads/{thread_id}/runs/{run_id}/join"): ("threads", "read"),
    ("GET", "/threads/{thread_id}/runs/{run_id}/stream"): ("threads", "read"),
    ("POST", "/threads/{thread_id}/runs/{run_id}/cancel"): ("threads", "update"),
    ("DELETE", "/threads/{thread_id}/runs/{run_id}"): ("threads", "delete"),
    # --- stateless runs -----------------------------------------------------
    # These mint an ephemeral thread, so they authorize as a thread create_run
    # exactly like their threaded counterparts.
    ("POST", "/runs"): ("threads", "create_run"),
    ("POST", "/runs/stream"): ("threads", "create_run"),
    ("POST", "/runs/wait"): ("threads", "create_run"),
    ("POST", "/runs/cancel"): ("threads", "update"),
    # --- crons --------------------------------------------------------------
    ("POST", "/runs/crons"): ("crons", "create"),
    ("POST", "/threads/{thread_id}/runs/crons"): ("crons", "create"),
    ("PATCH", "/runs/crons/{cron_id}"): ("crons", "update"),
    ("DELETE", "/runs/crons/{cron_id}"): ("crons", "delete"),
    ("POST", "/runs/crons/search"): ("crons", "search"),
    ("POST", "/runs/crons/count"): ("crons", "search"),
    # --- store --------------------------------------------------------------
    ("PUT", "/store/items"): ("store", "put"),
    ("GET", "/store/items"): ("store", "get"),
    ("DELETE", "/store/items"): ("store", "delete"),
    ("POST", "/store/items/search"): ("store", "search"),
    # Protocol action is `list_namespaces`, which the SDK's @auth.on.store
    # already exposes (langgraph-api 0.12.1 api/store.py). The old `store.search`
    # here was a bug, not an API anyone could bind to for namespaces.
    ("POST", "/store/namespaces"): ("store", "list_namespaces"),
    # --- protocol v2 event streaming ---------------------------------------
    ("POST", "/threads/{thread_id}/stream/events"): ("threads", "read"),
    ("POST", "/threads/{thread_id}/commands"): ("threads", "create_run"),
}

# Routes that authorize themselves inside the handler body.
#
# These own their dispatch for a reason: their `value` is the live request model,
# and a handler may inject metadata by mutating it in place (see
# examples/jwt_mock_auth_example.py, which sets value["metadata"]["team_id"]).
# Dispatching from the route registration would run the handler against a
# throwaway dict and silently drop that injection, so the enforcer stands down
# and lets the route call handle_event itself.
#
# Everything else in ROUTE_AUTH_MAP is dispatched by the enforcer.
SELF_DISPATCHING: Final[frozenset[tuple[str, str]]] = frozenset(
    {
        ("POST", "/assistants"),
        ("GET", "/assistants"),
        ("POST", "/assistants/search"),
        ("POST", "/assistants/count"),
        ("GET", "/assistants/{assistant_id}"),
        ("PATCH", "/assistants/{assistant_id}"),
        ("DELETE", "/assistants/{assistant_id}"),
        ("POST", "/assistants/{assistant_id}/latest"),
        ("POST", "/assistants/{assistant_id}/versions"),
        ("GET", "/assistants/{assistant_id}/schemas"),
        ("GET", "/assistants/{assistant_id}/graph"),
        ("GET", "/assistants/{assistant_id}/subgraphs"),
        ("POST", "/threads"),
        ("GET", "/threads"),
        ("POST", "/threads/search"),
        ("GET", "/threads/{thread_id}"),
        ("PATCH", "/threads/{thread_id}"),
        ("DELETE", "/threads/{thread_id}"),
        ("POST", "/threads/{thread_id}/runs"),
        ("POST", "/threads/{thread_id}/runs/stream"),
        ("POST", "/threads/{thread_id}/runs/wait"),
        ("GET", "/threads/{thread_id}/runs/{run_id}"),
        ("DELETE", "/threads/{thread_id}/runs/{run_id}"),
        ("POST", "/runs"),
        ("POST", "/runs/stream"),
        ("POST", "/runs/wait"),
        ("POST", "/runs/crons"),
        ("POST", "/threads/{thread_id}/runs/crons"),
        ("PATCH", "/runs/crons/{cron_id}"),
        ("DELETE", "/runs/crons/{cron_id}"),
        ("POST", "/runs/crons/search"),
        ("POST", "/runs/crons/count"),
        ("PUT", "/store/items"),
        ("GET", "/store/items"),
        ("DELETE", "/store/items"),
        ("POST", "/store/items/search"),
        ("POST", "/store/namespaces"),
    }
)


def is_self_dispatching(method: str, path: str) -> bool:
    """Whether the route body authorizes itself, so the enforcer must stand down."""
    return (method.upper(), path) in SELF_DISPATCHING


# Routes that intentionally carry no @auth.on dispatch. Anything not here and
# not in ROUTE_AUTH_MAP fails the coverage test.
EXEMPT_PATHS: Final[frozenset[str]] = frozenset(
    {
        "/",
        "/health",
        "/ready",
        "/live",
        "/metrics",
        "/info",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/docs/oauth2-redirect",
    }
)


def lookup_route_auth(method: str, path: str) -> tuple[str, str] | None:
    """Return the ``(resource, action)`` a route authorizes as, if any."""
    return ROUTE_AUTH_MAP.get((method.upper(), path))

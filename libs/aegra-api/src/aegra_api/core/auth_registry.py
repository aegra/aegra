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
    # --- stateless runs -----------------------------------------------------
    # These mint an ephemeral thread, so they authorize as a thread create_run
    # exactly like their threaded counterparts.
    ("POST", "/runs"): ("threads", "create_run"),
    ("POST", "/runs/stream"): ("threads", "create_run"),
    ("POST", "/runs/wait"): ("threads", "create_run"),
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
    ("POST", "/store/namespaces"): ("store", "search"),
    # --- protocol v2 event streaming ---------------------------------------
    ("POST", "/threads/{thread_id}/stream/events"): ("threads", "read"),
    ("POST", "/threads/{thread_id}/commands"): ("threads", "create_run"),
}

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
